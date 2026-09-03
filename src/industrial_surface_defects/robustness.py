"""Measure synthetic camera-shift robustness on the locked test images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .compare import (
    REGISTERED_CONFIGURATIONS,
    REGISTERED_CONTRASTS,
    REGISTERED_SEEDS,
)
from .config import DataSettings
from .corruptions import CORRUPTION_NAMES, CorruptedManifestDataset
from .data import build_dataloader, verify_manifest_files
from .evaluate import predict
from .metrics import compute_metrics, prediction_table
from .models import load_checkpoint_model
from .reproducibility import resolve_device, seed_everything, write_json

ROBUSTNESS_AGGREGATION_FILENAMES: tuple[str, ...] = (
    "robustness_scores_by_run.csv",
    "robustness_scores_by_configuration.csv",
    "robustness_registered_contrasts.csv",
    "robustness_aggregation_metadata.json",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_robustness(results: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    clean_score = float(results.loc[results["corruption"] == "clean", "macro_f1"].iloc[0])
    for corruption, rows in results.loc[results["corruption"] != "clean"].groupby(
        "corruption", observed=True
    ):
        axis.plot(rows["severity"], rows["macro_f1"], marker="o", label=corruption)
    axis.axhline(clean_score, color="black", linestyle="--", label="clean test")
    axis.set(
        xlabel="Synthetic corruption severity",
        ylabel="Macro-F1",
        title="Synthetic acquisition-shift robustness",
        xticks=[1, 2, 3],
        ylim=(0, 1.02),
    )
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _expected_robustness_conditions() -> set[tuple[str, int]]:
    return {("clean", 0)} | {
        (corruption, severity) for corruption in CORRUPTION_NAMES for severity in (1, 2, 3)
    }


def _read_robustness_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"corruption", "severity", "macro_f1"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    numeric_severity = pd.to_numeric(frame["severity"], errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric_severity).all()
        or not np.equal(numeric_severity, np.floor(numeric_severity)).all()
    ):
        raise ValueError(f"{path} severity must contain finite integers")
    frame = frame.copy()
    frame["corruption"] = frame["corruption"].astype(str)
    frame["severity"] = numeric_severity.astype(np.int64)
    if frame.duplicated(["corruption", "severity"]).any():
        raise ValueError(f"{path} contains duplicate corruption/severity rows")

    actual_conditions = set(frame[["corruption", "severity"]].itertuples(index=False, name=None))
    expected_conditions = _expected_robustness_conditions()
    if actual_conditions != expected_conditions:
        missing = sorted(expected_conditions - actual_conditions)
        extra = sorted(actual_conditions - expected_conditions)
        raise ValueError(
            f"{path} does not match the registered robustness condition grid; "
            f"missing={missing}, extra={extra}"
        )

    macro_f1 = pd.to_numeric(frame["macro_f1"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(macro_f1).all() or ((macro_f1 < 0) | (macro_f1 > 1)).any():
        raise ValueError(f"{path} macro_f1 values must be finite and in [0, 1]")
    frame["macro_f1"] = macro_f1
    return frame


def _write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _write_json_exclusive(value: dict, path: Path) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def aggregate_registered_robustness(
    outputs_root: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Aggregate the exact 4 x 3 registered robustness result matrix."""
    destination = Path(output_dir)
    existing = [
        destination / filename
        for filename in ROBUSTNESS_AGGREGATION_FILENAMES
        if (destination / filename).exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing robustness aggregation artifacts: "
            f"{[str(path) for path in existing]}"
        )

    root = Path(outputs_root)
    run_rows: list[dict] = []
    for configuration in REGISTERED_CONFIGURATIONS:
        for training_seed in REGISTERED_SEEDS:
            metrics_path = (
                root
                / configuration
                / f"seed_{training_seed}"
                / "evaluation"
                / "robustness"
                / "robustness_metrics.csv"
            )
            if not metrics_path.is_file():
                raise FileNotFoundError(
                    f"Missing registered robustness metrics for {configuration}, "
                    f"seed={training_seed}: {metrics_path}"
                )
            metrics = _read_robustness_metrics(metrics_path)
            clean = float(
                metrics.loc[
                    (metrics["corruption"] == "clean") & (metrics["severity"] == 0),
                    "macro_f1",
                ].iloc[0]
            )
            corrupted = metrics.loc[metrics["corruption"] != "clean"]
            mean_corrupted = float(corrupted["macro_f1"].mean())
            worst = corrupted.sort_values(
                ["macro_f1", "corruption", "severity"],
                ascending=[True, True, True],
            ).iloc[0]
            run_rows.append(
                {
                    "configuration": configuration,
                    "seed": training_seed,
                    "clean_macro_f1": clean,
                    "mean_corrupted_macro_f1": mean_corrupted,
                    "mean_absolute_macro_f1_drop": clean - mean_corrupted,
                    "worst_corruption": worst["corruption"],
                    "worst_severity": int(worst["severity"]),
                    "worst_macro_f1": float(worst["macro_f1"]),
                    "worst_absolute_macro_f1_drop": clean - float(worst["macro_f1"]),
                    "conditions": len(metrics),
                    "metrics_path": str(metrics_path),
                }
            )

    scores_by_run = pd.DataFrame(run_rows)
    scores_by_configuration = (
        scores_by_run.groupby("configuration", observed=True)
        .agg(
            runs=("seed", "count"),
            clean_macro_f1_mean=("clean_macro_f1", "mean"),
            clean_macro_f1_sd=("clean_macro_f1", "std"),
            mean_corrupted_macro_f1_mean=("mean_corrupted_macro_f1", "mean"),
            mean_corrupted_macro_f1_sd=("mean_corrupted_macro_f1", "std"),
            mean_absolute_macro_f1_drop_mean=(
                "mean_absolute_macro_f1_drop",
                "mean",
            ),
            mean_absolute_macro_f1_drop_sd=(
                "mean_absolute_macro_f1_drop",
                "std",
            ),
        )
        .reindex(REGISTERED_CONFIGURATIONS)
        .reset_index()
    )

    corrupted_pivot = scores_by_run.pivot(
        index="seed",
        columns="configuration",
        values="mean_corrupted_macro_f1",
    ).reindex(index=REGISTERED_SEEDS, columns=REGISTERED_CONFIGURATIONS)
    if corrupted_pivot.isna().any().any():
        raise ValueError("Registered robustness matrix is incomplete after validation")

    contrast_rows: list[dict] = []
    for name, expression, coefficients in REGISTERED_CONTRASTS:
        seed_estimates = sum(
            coefficient * corrupted_pivot[configuration]
            for configuration, coefficient in coefficients.items()
        )
        values = seed_estimates.to_numpy(dtype=np.float64)
        contrast_rows.append(
            {
                "contrast": name,
                "expression": expression,
                "metric": "mean_corrupted_macro_f1",
                **{
                    f"seed_{training_seed}": float(seed_estimates.loc[training_seed])
                    for training_seed in REGISTERED_SEEDS
                },
                "mean": float(values.mean()),
                "seed_sd": float(values.std(ddof=1)),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)

    destination.mkdir(parents=True, exist_ok=True)
    _write_csv_exclusive(
        scores_by_run,
        destination / "robustness_scores_by_run.csv",
    )
    _write_csv_exclusive(
        scores_by_configuration,
        destination / "robustness_scores_by_configuration.csv",
    )
    _write_csv_exclusive(
        contrasts,
        destination / "robustness_registered_contrasts.csv",
    )
    _write_json_exclusive(
        {
            "configurations": list(REGISTERED_CONFIGURATIONS),
            "seeds": list(REGISTERED_SEEDS),
            "runs": len(scores_by_run),
            "corruption_suite": "corruption_suite_v1",
            "conditions_per_run": len(_expected_robustness_conditions()),
            "condition_grid": [
                {"corruption": corruption, "severity": severity}
                for corruption, severity in sorted(_expected_robustness_conditions())
            ],
            "summary_metric": (
                "unweighted mean macro-F1 over the "
                f"{len(CORRUPTION_NAMES) * 3} corrupted conditions"
            ),
            "contrast_count": len(contrasts),
        },
        destination / "robustness_aggregation_metadata.json",
    )
    return destination


def evaluate_robustness(
    checkpoint_path: str | Path,
    *,
    acknowledge_test_use: bool,
    device_name: str = "auto",
    data_root: str | None = None,
    manifest_path: str | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    output_dir: str | Path | None = None,
) -> Path:
    if not acknowledge_test_use:
        raise PermissionError(
            "Robustness evaluation reuses locked test images. Pass --acknowledge-test-use only "
            "after the protocol is frozen."
        )
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint_model(str(checkpoint_path), device=device)
    config = checkpoint["config"]
    data = DataSettings(**config["data"])
    root = data_root or data.root
    manifest = manifest_path or data.manifest
    verify_manifest_files(manifest, root)
    if _sha256(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError(
            "Manifest differs from training; robustness scores would not be comparable"
        )
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    effective_batch_size = batch_size or int(config["training"]["batch_size"])

    destination = (
        Path(output_dir)
        if output_dir is not None
        else checkpoint_path.parent / "evaluation" / "robustness"
    )
    if (destination / "robustness_metrics.csv").exists():
        raise FileExistsError(f"Robustness results already exist at {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []
    per_class_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    clean_loader = build_dataloader(
        manifest_path=manifest,
        data_root=root,
        split="test",
        image_size=data.image_size,
        augmentation="minimal",
        batch_size=effective_batch_size,
        num_workers=num_workers,
        seed=seed,
        return_path=True,
    )
    paths, targets, probabilities = predict(model, clean_loader, device)
    metrics, per_class, _, _ = compute_metrics(targets, probabilities)
    result_rows.append(
        {
            "corruption": "clean",
            "severity": 0,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
    )
    clean_predictions = prediction_table(paths, targets, probabilities)
    clean_predictions.insert(0, "severity", 0)
    clean_predictions.insert(0, "corruption", "clean")
    prediction_frames.append(clean_predictions)
    for row in per_class.to_dict("records"):
        per_class_rows.append({"corruption": "clean", "severity": 0, **row})
    clean_macro_f1 = metrics["macro_f1"]

    for corruption in CORRUPTION_NAMES:
        for severity in (1, 2, 3):
            dataset = CorruptedManifestDataset(
                manifest,
                root,
                split="test",
                image_size=data.image_size,
                corruption=corruption,
                severity=severity,
            )
            loader = DataLoader(
                dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            paths, targets, probabilities = predict(model, loader, device)
            metrics, per_class, _, _ = compute_metrics(targets, probabilities)
            result_rows.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1_absolute_drop": clean_macro_f1 - metrics["macro_f1"],
                    "macro_f1_relative_drop": (
                        (clean_macro_f1 - metrics["macro_f1"]) / clean_macro_f1
                        if clean_macro_f1
                        else 0.0
                    ),
                }
            )
            frame = prediction_table(paths, targets, probabilities)
            frame.insert(0, "severity", severity)
            frame.insert(0, "corruption", corruption)
            prediction_frames.append(frame)
            for row in per_class.to_dict("records"):
                per_class_rows.append({"corruption": corruption, "severity": severity, **row})

    results = pd.DataFrame(result_rows)
    results.to_csv(destination / "robustness_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        destination / "robustness_predictions.csv", index=False
    )
    pd.DataFrame(per_class_rows).to_csv(
        destination / "robustness_per_class_metrics.csv", index=False
    )
    _plot_robustness(results, destination / "robustness_macro_f1.png")
    corrupted_results = results.loc[results["corruption"] != "clean"]
    worst_condition = corrupted_results.sort_values("macro_f1").iloc[0]
    summary = {
        "corruption_suite": "corruption_suite_v1",
        "scope": (
            "Synthetic shifts applied to the same locked NEU-CLS test images; this is not "
            "external-domain validation."
        ),
        "clean_macro_f1": clean_macro_f1,
        "mean_corrupted_macro_f1": float(corrupted_results["macro_f1"].mean()),
        "mean_absolute_macro_f1_drop": float(corrupted_results["macro_f1_absolute_drop"].mean()),
        "worst_absolute_macro_f1_drop": float(corrupted_results["macro_f1_absolute_drop"].max()),
        "worst_condition": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in worst_condition.to_dict().items()
        },
        "conditions": len(results),
        "test_use_acknowledged": True,
    }
    write_json(destination / "summary.json", summary)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--acknowledge-test-use", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = evaluate_robustness(
        args.checkpoint,
        acknowledge_test_use=args.acknowledge_test_use,
        device_name=args.device,
        data_root=args.data_root,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    print(f"Robustness artifacts: {destination}")


if __name__ == "__main__":
    main()
