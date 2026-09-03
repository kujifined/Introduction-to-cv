"""Aggregate multi-seed runs and compute paired stratified bootstrap contrasts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .constants import CLASS_NAMES
from .reproducibility import write_json

REGISTERED_CONFIGURATIONS: tuple[str, ...] = (
    "resnet18_minimal",
    "resnet18_camera",
    "efficientnet_b0_minimal",
    "efficientnet_b0_camera",
)
REGISTERED_SEEDS: tuple[int, ...] = (42, 43, 44)
REGISTERED_CELL_METADATA: dict[str, tuple[str, str]] = {
    "resnet18_minimal": ("resnet18", "minimal"),
    "resnet18_camera": ("resnet18", "camera_aware"),
    "efficientnet_b0_minimal": ("efficientnet_b0", "minimal"),
    "efficientnet_b0_camera": ("efficientnet_b0", "camera_aware"),
}
COMPARISON_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "scores_by_run.csv",
    "scores_by_configuration.csv",
    "paired_bootstrap_contrasts.csv",
    "macro_f1_by_configuration.png",
    "comparison_metadata.json",
)
REGISTERED_CONTRASTS: tuple[tuple[str, str, dict[str, float]], ...] = (
    (
        "architecture_under_minimal",
        "efficientnet_b0_minimal - resnet18_minimal",
        {"efficientnet_b0_minimal": 1.0, "resnet18_minimal": -1.0},
    ),
    (
        "architecture_under_camera",
        "efficientnet_b0_camera - resnet18_camera",
        {"efficientnet_b0_camera": 1.0, "resnet18_camera": -1.0},
    ),
    (
        "augmentation_within_resnet18",
        "resnet18_camera - resnet18_minimal",
        {"resnet18_camera": 1.0, "resnet18_minimal": -1.0},
    ),
    (
        "augmentation_within_efficientnet_b0",
        "efficientnet_b0_camera - efficientnet_b0_minimal",
        {"efficientnet_b0_camera": 1.0, "efficientnet_b0_minimal": -1.0},
    ),
    (
        "architecture_x_augmentation_interaction",
        "(efficientnet_b0_camera - efficientnet_b0_minimal) - (resnet18_camera - resnet18_minimal)",
        {
            "efficientnet_b0_camera": 1.0,
            "efficientnet_b0_minimal": -1.0,
            "resnet18_camera": -1.0,
            "resnet18_minimal": 1.0,
        },
    ),
)


@dataclass(frozen=True)
class PredictionRun:
    configuration: str
    seed: int
    path: Path
    targets: np.ndarray
    predictions: np.ndarray
    sample_paths: tuple[str, ...]
    evaluation_split: str | None
    model_name: str | None
    augmentation: str | None
    manifest_sha256: str | None


def _parse_specification(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Prediction inputs must use CONFIGURATION=path/to/predictions.csv")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError(f"Invalid prediction specification: {value!r}")
    return label.strip(), Path(path)


def _validated_class_indices(frame: pd.DataFrame, column: str, path: Path) -> np.ndarray:
    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{path} column {column} must contain integer class indices")
    indices = numeric.astype(np.int64)
    invalid = (indices < 0) | (indices >= len(CLASS_NAMES))
    if invalid.any():
        invalid_values = sorted(set(indices[invalid].tolist()))
        raise ValueError(
            f"{path} column {column} contains values outside valid range "
            f"0..{len(CLASS_NAMES) - 1}: {invalid_values}"
        )
    return indices


def _read_run(
    configuration: str,
    path: Path,
    fallback_seed: int,
    *,
    require_seed_metadata: bool = False,
) -> PredictionRun:
    frame = pd.read_csv(path)
    required = {"path", "true_index", "predicted_index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["path"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample paths")
    frame = frame.sort_values("path").reset_index(drop=True)
    metrics_path = path.parent / "metrics.json"
    seed = fallback_seed
    metadata: dict = {}
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if require_seed_metadata and "seed" not in metadata:
            raise ValueError(f"{metrics_path} does not record the training seed")
        seed = int(metadata.get("seed", fallback_seed))
    elif require_seed_metadata:
        raise ValueError(f"Registered comparison requires metadata: {metrics_path}")
    return PredictionRun(
        configuration=configuration,
        seed=seed,
        path=path,
        targets=_validated_class_indices(frame, "true_index", path),
        predictions=_validated_class_indices(frame, "predicted_index", path),
        sample_paths=tuple(frame["path"].astype(str)),
        evaluation_split=metadata.get("split"),
        model_name=metadata.get("model_name"),
        augmentation=metadata.get("augmentation_training"),
        manifest_sha256=metadata.get("manifest_sha256"),
    )


def _validate_alignment(runs: list[PredictionRun]) -> tuple[np.ndarray, tuple[str, ...]]:
    reference = runs[0]
    for run in runs[1:]:
        if run.sample_paths != reference.sample_paths:
            raise ValueError(f"Sample paths differ between {reference.path} and {run.path}")
        if not np.array_equal(run.targets, reference.targets):
            raise ValueError(f"Ground-truth labels differ between {reference.path} and {run.path}")
    return reference.targets, reference.sample_paths


def _stratified_resample_indices(targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for class_index in np.unique(targets):
        indices = np.flatnonzero(targets == class_index)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled)


def _macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.unique(targets)
    return float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0))


def _pair_runs(
    left: list[PredictionRun], right: list[PredictionRun]
) -> tuple[list[tuple[PredictionRun, PredictionRun]], str]:
    left_by_seed = {run.seed: run for run in left}
    right_by_seed = {run.seed: run for run in right}
    left_seeds = set(left_by_seed)
    right_seeds = set(right_by_seed)
    if left_seeds != right_seeds:
        raise ValueError(
            "Paired comparison requires identical seed sets; "
            f"left={sorted(left_seeds)}, right={sorted(right_seeds)}"
        )
    seeds = sorted(left_seeds)
    return [(left_by_seed[seed], right_by_seed[seed]) for seed in seeds], "matched_seed"


def _validate_registered_design(grouped: dict[str, list[PredictionRun]]) -> None:
    expected_configurations = set(REGISTERED_CONFIGURATIONS)
    actual_configurations = set(grouped)
    if actual_configurations != expected_configurations:
        missing = sorted(expected_configurations - actual_configurations)
        extra = sorted(actual_configurations - expected_configurations)
        raise ValueError(
            "Registered comparison requires exactly the four factorial cells; "
            f"missing={missing}, extra={extra}"
        )
    expected_seeds = set(REGISTERED_SEEDS)
    for configuration in REGISTERED_CONFIGURATIONS:
        runs = grouped[configuration]
        actual_seeds = {run.seed for run in runs}
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"Registered configuration {configuration!r} requires seeds "
                f"{list(REGISTERED_SEEDS)}; found {sorted(actual_seeds)}"
            )
        expected_model, expected_augmentation = REGISTERED_CELL_METADATA[configuration]
        for run in runs:
            if run.model_name != expected_model or run.augmentation != expected_augmentation:
                raise ValueError(
                    f"Registered run {run.path} claims model={run.model_name!r}, "
                    f"augmentation={run.augmentation!r}; expected model={expected_model!r}, "
                    f"augmentation={expected_augmentation!r}"
                )
            if run.evaluation_split not in {"val", "test"}:
                raise ValueError(f"Registered run {run.path} does not record split=val or test")
            if not run.manifest_sha256:
                raise ValueError(f"Registered run {run.path} does not record manifest_sha256")
    all_runs = [run for runs in grouped.values() for run in runs]
    splits = {run.evaluation_split for run in all_runs}
    if len(splits) != 1:
        raise ValueError(f"Registered runs mix evaluation splits: {sorted(splits)}")
    manifest_hashes = {run.manifest_sha256 for run in all_runs}
    if len(manifest_hashes) != 1:
        raise ValueError("Registered runs were evaluated against different split manifests")


def _registered_bootstrap_contrast(
    grouped: dict[str, list[PredictionRun]],
    targets: np.ndarray,
    *,
    name: str,
    expression: str,
    coefficients: dict[str, float],
    repetitions: int,
    seed: int,
) -> dict:
    runs_by_configuration_and_seed = {
        configuration: {run.seed: run for run in runs} for configuration, runs in grouped.items()
    }

    def contrast_for_indices(indices: np.ndarray, training_seed: int) -> float:
        sampled_targets = targets[indices]
        return float(
            sum(
                coefficient
                * _macro_f1(
                    sampled_targets,
                    runs_by_configuration_and_seed[configuration][training_seed].predictions[
                        indices
                    ],
                )
                for configuration, coefficient in coefficients.items()
            )
        )

    all_indices = np.arange(len(targets))
    seed_estimates = [
        contrast_for_indices(all_indices, training_seed) for training_seed in REGISTERED_SEEDS
    ]
    estimate = float(np.mean(seed_estimates))
    rng = np.random.default_rng(seed)
    bootstrap_estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sample_indices = _stratified_resample_indices(targets, rng)
        bootstrap_estimates[repetition] = np.mean(
            [
                contrast_for_indices(sample_indices, training_seed)
                for training_seed in REGISTERED_SEEDS
            ]
        )
    lower, upper = np.quantile(bootstrap_estimates, [0.025, 0.975])
    return {
        "contrast": name,
        "expression": expression,
        "estimate": estimate,
        "seed_mean": estimate,
        "seed_sd": float(np.std(seed_estimates, ddof=1)),
        "seed_estimates": json.dumps(seed_estimates),
        "paired_runs": len(REGISTERED_SEEDS),
        "pairing": "matched_seed",
        "bootstrap_mean_estimate": float(bootstrap_estimates.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_repetitions": repetitions,
        "bootstrap_unit": "stratified test image; fixed mean across matched training seeds",
    }


def _bootstrap_contrast(
    left: list[PredictionRun],
    right: list[PredictionRun],
    targets: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    pairs, pairing = _pair_runs(left, right)
    if not pairs:
        raise ValueError("Each configuration needs at least one prediction run")
    observed_left = np.mean([_macro_f1(targets, run.predictions) for run, _ in pairs])
    observed_right = np.mean([_macro_f1(targets, run.predictions) for _, run in pairs])
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sample_indices = _stratified_resample_indices(targets, rng)
        seed_indices = rng.integers(0, len(pairs), size=len(pairs))
        left_scores = []
        right_scores = []
        for pair_index in seed_indices:
            left_run, right_run = pairs[int(pair_index)]
            sampled_targets = targets[sample_indices]
            left_scores.append(_macro_f1(sampled_targets, left_run.predictions[sample_indices]))
            right_scores.append(_macro_f1(sampled_targets, right_run.predictions[sample_indices]))
        differences[repetition] = np.mean(left_scores) - np.mean(right_scores)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "left_configuration": left[0].configuration,
        "right_configuration": right[0].configuration,
        "paired_runs": len(pairs),
        "pairing": pairing,
        "left_mean_macro_f1": float(observed_left),
        "right_mean_macro_f1": float(observed_right),
        "difference_left_minus_right": float(observed_left - observed_right),
        "bootstrap_mean_difference": float(differences.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_repetitions": repetitions,
        "bootstrap_unit": "stratified test image with seed-level resampling",
    }


def _plot_summary(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("mean_macro_f1", ascending=True)
    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.65 * len(ordered))))
    axis.barh(
        ordered["configuration"],
        ordered["mean_macro_f1"],
        xerr=ordered["sd_macro_f1"].fillna(0),
        color="#4C78A8",
        alpha=0.9,
    )
    axis.set(
        xlabel="Macro-F1 (mean +/- SD across seeds)",
        xlim=(0, 1.02),
        title="Planned experiment comparison",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def compare(
    specifications: list[str],
    *,
    output_dir: str | Path,
    repetitions: int = 5000,
    seed: int = 42,
    registered_design: bool = False,
) -> Path:
    destination = Path(output_dir)
    existing = [
        destination / filename
        for filename in COMPARISON_ARTIFACT_FILENAMES
        if (destination / filename).exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing comparison artifacts: "
            f"{[str(path) for path in existing]}"
        )

    grouped: dict[str, list[PredictionRun]] = defaultdict(list)
    for fallback_seed, specification in enumerate(specifications):
        configuration, path = _parse_specification(specification)
        run = _read_run(
            configuration,
            path,
            fallback_seed,
            require_seed_metadata=registered_design,
        )
        if any(existing.seed == run.seed for existing in grouped[configuration]):
            raise ValueError(
                f"Configuration {configuration!r} has more than one run with seed {run.seed}"
            )
        grouped[configuration].append(run)
    if len(grouped) < 2:
        raise ValueError("At least two configurations are required")
    if registered_design:
        _validate_registered_design(grouped)
    all_runs = [run for runs in grouped.values() for run in runs]
    targets, sample_paths = _validate_alignment(all_runs)

    run_rows: list[dict] = []
    for run in all_runs:
        run_rows.append(
            {
                "configuration": run.configuration,
                "seed": run.seed,
                "macro_f1": _macro_f1(targets, run.predictions),
                "predictions": str(run.path),
            }
        )
    run_scores = pd.DataFrame(run_rows)
    summary = (
        run_scores.groupby("configuration", observed=True)["macro_f1"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "runs",
                "mean": "mean_macro_f1",
                "std": "sd_macro_f1",
            }
        )
    )

    contrast_rows: list[dict] = []
    if registered_design:
        for contrast_index, (name, expression, coefficients) in enumerate(REGISTERED_CONTRASTS):
            contrast_rows.append(
                _registered_bootstrap_contrast(
                    grouped,
                    targets,
                    name=name,
                    expression=expression,
                    coefficients=coefficients,
                    repetitions=repetitions,
                    seed=seed + contrast_index,
                )
            )
    else:
        for contrast_index, (left_name, right_name) in enumerate(combinations(sorted(grouped), 2)):
            contrast_rows.append(
                _bootstrap_contrast(
                    grouped[left_name],
                    grouped[right_name],
                    targets,
                    repetitions=repetitions,
                    seed=seed + contrast_index,
                )
            )
    contrasts = pd.DataFrame(contrast_rows)

    destination.mkdir(parents=True, exist_ok=True)
    run_scores.to_csv(destination / "scores_by_run.csv", index=False)
    summary.to_csv(destination / "scores_by_configuration.csv", index=False)
    contrasts.to_csv(destination / "paired_bootstrap_contrasts.csv", index=False)
    _plot_summary(summary, destination / "macro_f1_by_configuration.png")
    write_json(
        destination / "comparison_metadata.json",
        {
            "configurations": sorted(grouped),
            "runs": len(all_runs),
            "test_images": len(sample_paths),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": seed,
            "registered_design": registered_design,
            "registered_seeds": list(REGISTERED_SEEDS) if registered_design else None,
            "contrast_count": len(contrasts),
            "interpretation": (
                "A CI spanning zero means the available test data and seeds do not distinguish "
                "the configurations; it does not prove equivalence."
            ),
        },
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Repeat CONFIGURATION=path/to/predictions.csv for every seed/run.",
    )
    parser.add_argument("--output-dir", default="reports/tables/generated/comparison")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--registered-design",
        action="store_true",
        help="Require the four preregistered cells, seeds 42/43/44, and five planned contrasts.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = compare(
        args.prediction,
        output_dir=args.output_dir,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
        registered_design=args.registered_design,
    )
    print(f"Comparison artifacts: {destination}")


if __name__ == "__main__":
    main()
