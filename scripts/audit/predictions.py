"""Prediction-table validation and independent metric recomputation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import Audit, json_load, manual_metrics


CleanPredictions = dict[
    tuple[str, int],
    tuple[np.ndarray, np.ndarray, list[str]],
]


def condition_key(corruption: str, severity: int) -> str:
    return "clean" if corruption == "clean" else f"{corruption}_s{severity}"


def prediction_path(
    results_root: Path,
    configuration: str,
    seed: int,
    key: str,
) -> Path:
    evaluation = results_root / "outputs" / configuration / f"seed_{seed}" / "evaluation"
    if key == "clean":
        return evaluation / "test" / "predictions.csv"
    return evaluation / "corruptions" / key / "predictions.csv"


def metrics_path(
    results_root: Path,
    configuration: str,
    seed: int,
    key: str,
) -> Path:
    return prediction_path(results_root, configuration, seed, key).with_name("metrics.json")


def validate_prediction_table(
    table: pd.DataFrame,
    expected_rows: pd.DataFrame,
    class_names: list[str],
    artifact: str,
    expected_sample_count: int,
    probability_sum_tolerance: float,
    audit: Audit,
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "path",
        "true_index",
        "true_class",
        "predicted_index",
        "predicted_class",
        "confidence",
        "top2_index",
        "top2_class",
        "top2_probability",
        "margin",
        "correct",
        *(f"prob_{name}" for name in class_names),
    }
    audit.equal(
        "predictions",
        artifact,
        "required columns present",
        True,
        required.issubset(table.columns),
    )
    audit.equal(
        "predictions",
        artifact,
        "row count",
        expected_sample_count,
        len(table),
    )
    audit.equal(
        "predictions",
        artifact,
        "unique paths",
        expected_sample_count,
        table["path"].nunique(),
    )
    expected_labels = expected_rows.set_index("path")[["class_index", "class_name"]]
    observed = table.set_index("path")[["true_index", "true_class"]].sort_index()
    aligned_expected = expected_labels.loc[observed.index].sort_index()
    labels_match = np.array_equal(
        observed["true_index"].to_numpy(dtype=np.int64),
        aligned_expected["class_index"].to_numpy(dtype=np.int64),
    ) and observed["true_class"].astype(str).equals(
        aligned_expected["class_name"].astype(str)
    )
    audit.equal(
        "predictions",
        artifact,
        "paths and ground truth match test split",
        True,
        labels_match,
    )
    probabilities = table[
        [f"prob_{name}" for name in class_names]
    ].to_numpy(dtype=np.float64)
    targets = table["true_index"].to_numpy(dtype=np.int64)
    row_sums = probabilities.sum(axis=1)
    audit.close(
        "predictions",
        artifact,
        "maximum probability-sum error",
        0.0,
        float(np.max(np.abs(row_sums - 1.0))),
        tolerance=probability_sum_tolerance,
    )
    ranking = np.argsort(probabilities, axis=1)[:, ::-1]
    top1 = ranking[:, 0]
    top2 = ranking[:, 1]
    audit.equal(
        "predictions",
        artifact,
        "predicted_index equals probability argmax",
        True,
        bool(
            np.array_equal(
                top1,
                table["predicted_index"].to_numpy(dtype=np.int64),
            )
        ),
    )
    audit.equal(
        "predictions",
        artifact,
        "top2_index is consistent",
        True,
        bool(
            np.array_equal(
                top2,
                table["top2_index"].to_numpy(dtype=np.int64),
            )
        ),
    )
    confidence_error = np.max(
        np.abs(
            table["confidence"].to_numpy(dtype=float)
            - probabilities[np.arange(len(table)), top1]
        )
    )
    audit.close(
        "predictions",
        artifact,
        "maximum confidence error",
        0.0,
        float(confidence_error),
    )
    audit.equal(
        "predictions",
        artifact,
        "correct flag is consistent",
        True,
        bool(
            np.array_equal(
                table["correct"].astype(bool).to_numpy(),
                top1 == targets,
            )
        ),
    )
    return targets, probabilities


def audit_predictions_and_metrics(
    results_root: Path,
    manifest: pd.DataFrame,
    class_names: list[str],
    experiment: dict[str, Any],
    expectations: dict[str, Any],
    metric_config: dict[str, Any],
    audit: Audit,
) -> tuple[pd.DataFrame, CleanPredictions]:
    protocol = json_load(results_root / "experiment_protocol.json")
    conditions = [
        (row["corruption"], int(row["severity"]))
        for row in protocol["conditions"]
    ]
    if len(conditions) != int(expectations["conditions_per_checkpoint"]):
        raise ValueError(
            "Protocol condition count does not match YAML expectation: "
            f"{len(conditions)} != {expectations['conditions_per_checkpoint']}"
        )
    target_split = str(expectations["target_split"])
    expected_rows = manifest.loc[manifest["split"] == target_split].copy()
    saved_scores = pd.read_csv(results_root / "test_scores_all_conditions.csv")
    saved_index = saved_scores.set_index(
        ["configuration", "seed", "corruption", "severity"]
    )
    metric_names = list(metric_config["names"])
    rows: list[dict[str, Any]] = []
    clean_predictions: CleanPredictions = {}
    expected_paths: list[str] | None = None
    prediction_file_count = 0
    prediction_row_count = 0
    for configuration in experiment["configurations"]:
        for seed in experiment["seeds"]:
            for corruption, severity in conditions:
                key = condition_key(corruption, severity)
                path = prediction_path(results_root, configuration, seed, key)
                artifact = path.relative_to(results_root).as_posix()
                if not path.is_file():
                    audit.equal("predictions", artifact, "file exists", True, False)
                    continue
                prediction_file_count += 1
                table = pd.read_csv(path)
                prediction_row_count += len(table)
                targets, probabilities = validate_prediction_table(
                    table,
                    expected_rows,
                    class_names,
                    artifact,
                    int(expectations["samples_per_condition"]),
                    float(metric_config["probability_sum_tolerance"]),
                    audit,
                )
                current_paths = table["path"].astype(str).tolist()
                if expected_paths is None:
                    expected_paths = current_paths
                audit.equal(
                    "predictions",
                    artifact,
                    "path order matches reference",
                    True,
                    current_paths == expected_paths,
                )
                recomputed = manual_metrics(targets, probabilities)
                saved_json = json_load(metrics_path(results_root, configuration, seed, key))
                saved_row = saved_index.loc[
                    (configuration, seed, corruption, severity)
                ]
                for metric in metric_names:
                    audit.close(
                        "metrics_json",
                        artifact,
                        metric,
                        float(saved_json[metric]),
                        float(recomputed[metric]),
                    )
                    audit.close(
                        "metrics_summary",
                        "test_scores_all_conditions.csv",
                        f"{configuration}/{seed}/{key}/{metric}",
                        float(saved_row[metric]),
                        float(recomputed[metric]),
                    )
                rows.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "corruption": corruption,
                        "severity": severity,
                        **recomputed,
                    }
                )
                if corruption == "clean":
                    clean_predictions[(configuration, seed)] = (
                        targets,
                        probabilities.argmax(axis=1),
                        current_paths,
                    )
    audit.equal(
        "predictions",
        "outputs/*/evaluation",
        "test prediction files",
        int(expectations["prediction_files"]),
        prediction_file_count,
    )
    audit.equal(
        "predictions",
        "outputs/*/evaluation",
        "test prediction rows",
        int(expectations["prediction_rows"]),
        prediction_row_count,
    )
    return pd.DataFrame(rows), clean_predictions

