"""Independently verify saved test predictions and summaries; no GPU or dataset needed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from industrial_surface_defects.constants import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    metadata = json.loads((FINAL / "ARTIFACTS.json").read_text())
    inventory = json.loads((FINAL / "INVENTORY.json").read_text())
    for relative, expected in inventory.items():
        content = (FINAL / relative).read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected, f"Artifact changed: {relative}")
    manifest_path = ROOT / "data/manifests/neu_cls_v2.csv"
    require(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() == metadata["manifest_sha256"],
        "Manifest changed",
    )
    manifest = pd.read_csv(manifest_path)
    freeze = json.loads((FINAL / "protocol_freeze.json").read_text())
    for relative, expected in freeze["source_files"].items():
        require(
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected,
            f"Frozen experimental source changed: {relative}",
        )
    notebook = ROOT / "notebooks/08_kaggle_results_checkpoint.ipynb"
    require(
        hashlib.sha256(notebook.read_bytes()).hexdigest() == metadata["notebook_sha256"],
        "Recorded notebook changed",
    )
    test = manifest.loc[manifest.split == "test"].set_index("path").sort_index()
    require(len(test) == 360 and (test.class_index.value_counts() == 60).all(), "Test support")
    conditions = pd.read_csv(FINAL / "test_scores_all_conditions.csv")
    require(len(conditions) == 336, "Expected 336 condition results")
    computed = []
    max_delta = 0.0
    errors = 0
    for record in conditions.to_dict("records"):
        name, seed = record["configuration"], int(record["seed"])
        condition = (
            "test"
            if record["corruption"] == "clean"
            else (f"corruptions/{record['corruption']}_s{int(record['severity'])}")
        )
        base = FINAL / f"outputs/{name}/seed_{seed}/evaluation" / condition
        predictions = pd.read_csv(base / "predictions.csv").set_index("path").sort_index()
        require(
            predictions.index.is_unique and predictions.index.equals(test.index),
            f"Mismatched test paths: {base}",
        )
        y = predictions.true_index.to_numpy()
        p = predictions[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy()
        require(np.array_equal(y, test.class_index.to_numpy()), f"Labels: {base}")
        require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), "Probabilities")
        require(np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0), "Probability sum")
        pred = p.argmax(axis=1)
        require(np.array_equal(pred, predictions.predicted_index.to_numpy()), "Argmax")
        require(
            (predictions.true_class.to_numpy() == np.array(CLASS_NAMES)[y]).all(),
            "True class names",
        )
        require(
            (predictions.predicted_class.to_numpy() == np.array(CLASS_NAMES)[pred]).all(),
            "Predicted class names",
        )
        # The frozen metric contract normalizes FP32 softmax rows in float64.
        # Match log_loss's float64 epsilon, including extremely small true-class scores.
        p = p / p.sum(axis=1, keepdims=True)
        confidence = p.max(axis=1)
        ece = 0.0
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
            mask = (confidence > lo) & (confidence <= hi)
            if mask.any():
                ece += mask.mean() * abs(confidence[mask].mean() - (pred[mask] == y[mask]).mean())
        values = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=range(6), average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "negative_log_likelihood": -np.log(
                p[np.arange(360), y].clip(np.finfo(np.float64).eps, 1)
            ).mean(),
            "expected_calibration_error_10_bins": ece,
            "brier_score": ((p - np.eye(6)[y]) ** 2).sum(axis=1).mean(),
        }
        saved = json.loads((base / "metrics.json").read_text())
        for metric, value in values.items():
            for expected in (saved[metric], record[metric]):
                delta = abs(value - float(expected))
                max_delta = max(max_delta, delta)
                require(delta <= 2e-6, f"{name}/{seed}/{condition}: {metric}: {delta}")
        computed.append({**record, **values})
        if condition == "test":
            errors += int((pred != y).sum())
    frame = pd.DataFrame(computed)
    clean = pd.read_csv(FINAL / "clean_scores_by_configuration.csv").set_index("configuration")
    robust = pd.read_csv(
        FINAL / "comparison_robustness/robustness_scores_by_configuration.csv"
    ).set_index("configuration")
    for name, subset in frame.groupby("configuration"):
        require(len(subset) == 84 and subset.seed.nunique() == 3, "Registered run coverage")
        c = subset.loc[subset.corruption == "clean", "macro_f1"]
        r = subset.loc[subset.corruption != "clean"].groupby("seed").macro_f1.mean()
        for value, expected in (
            (c.mean(), clean.loc[name, "macro_f1_mean"]),
            (c.std(ddof=1), clean.loc[name, "macro_f1_std"]),
            (r.mean(), robust.loc[name, "mean_corrupted_macro_f1_mean"]),
            (r.std(ddof=1), robust.loc[name, "mean_corrupted_macro_f1_sd"]),
        ):
            require(abs(value - expected) < 1e-10, f"Summary differs: {name}")
    require(errors == 13, "Clean error count")
    return {
        "status": "pass",
        "conditions": len(frame),
        "prediction_rows": len(frame) * 360,
        "clean_error_rows": errors,
        "max_metric_delta": max_delta,
        "frozen_artifact_hashes_checked": len(inventory),
        "training_performed": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
