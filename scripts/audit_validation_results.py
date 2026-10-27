#!/usr/bin/env python3
"""Independently audit the 40-file validation pilot from serialized artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGURATIONS = (
    "resnet18_minimal",
    "resnet18_camera",
    "efficientnet_b0_minimal",
    "efficientnet_b0_camera",
)
PILOT_CORRUPTIONS = ("brightness_dark", "gaussian_blur", "shot_read_noise")
CONDITIONS = ("clean",) + tuple(
    f"{corruption}_s{severity}"
    for corruption in PILOT_CORRUPTIONS
    for severity in (1, 2, 3)
)
TOLERANCE = 1e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_correct(values: pd.Series) -> np.ndarray:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = values.astype(str).str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError("The correct column contains non-boolean values")
    return parsed.to_numpy(dtype=bool)


def calculate_metrics(
    table: pd.DataFrame,
    *,
    class_count: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    targets = table["true_index"].to_numpy(dtype=int)
    predictions = table["predicted_index"].to_numpy(dtype=int)
    confusion = np.zeros((class_count, class_count), dtype=int)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    class_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(class_count, dtype=float),
        where=denominator != 0,
    )
    accuracy = float((targets == predictions).mean())
    return accuracy, float(class_f1.mean()), confusion, class_f1


def cross_split_overlaps(rows: pd.DataFrame, column: str) -> list[dict[str, object]]:
    grouped = rows.groupby(column, observed=True)["split"].agg(lambda values: sorted(set(values)))
    return [
        {column: str(value), "splits": splits}
        for value, splits in grouped.items()
        if len(splits) > 1
    ]


def compare_aggregate_tables(
    root: Path,
    metric_rows: list[dict[str, object]],
) -> list[str]:
    failures: list[str] = []
    metric_by_key = {
        (str(row["configuration"]), str(row["condition"])): row for row in metric_rows
    }
    comparison_path = root / "comparison_all_conditions.csv"
    if not comparison_path.is_file():
        return [f"Missing aggregate table: {comparison_path}"]
    comparison = pd.read_csv(comparison_path).set_index(["configuration", "condition"])
    if set(comparison.index) != set(metric_by_key):
        failures.append("comparison_all_conditions.csv key set differs")
    for key, independent in metric_by_key.items():
        if key not in comparison.index:
            continue
        saved = comparison.loc[key]
        if abs(float(saved.accuracy) - float(independent["accuracy_recomputed"])) > TOLERANCE:
            failures.append(f"Comparison accuracy differs: {key}")
        if abs(float(saved.macro_f1) - float(independent["macro_f1_recomputed"])) > TOLERANCE:
            failures.append(f"Comparison macro-F1 differs: {key}")

    clean_path = root / "comparison_clean.csv"
    if not clean_path.is_file():
        failures.append(f"Missing aggregate table: {clean_path}")
    else:
        clean = pd.read_csv(clean_path)
        expected = {(configuration, "clean") for configuration in CONFIGURATIONS}
        obtained = set(zip(clean.configuration, clean.condition, strict=True))
        if obtained != expected:
            failures.append("comparison_clean.csv key set differs")

    effects_path = root / "augmentation_effects.csv"
    if not effects_path.is_file():
        failures.append(f"Missing aggregate table: {effects_path}")
        return failures
    effects = pd.read_csv(effects_path)
    if len(effects) != 20:
        failures.append(f"augmentation_effects.csv rows={len(effects)}")
    for row in effects.itertuples(index=False):
        minimal = metric_by_key[(f"{row.architecture}_minimal", row.condition)]
        camera = metric_by_key[(f"{row.architecture}_camera", row.condition)]
        clean_minimal = metric_by_key[(f"{row.architecture}_minimal", "clean")]
        clean_camera = metric_by_key[(f"{row.architecture}_camera", "clean")]
        expected = {
            "minimal_macro_f1": float(minimal["macro_f1_recomputed"]),
            "camera_macro_f1": float(camera["macro_f1_recomputed"]),
            "camera_minus_minimal_macro_f1": (
                float(camera["macro_f1_recomputed"])
                - float(minimal["macro_f1_recomputed"])
            ),
            "minimal_drop_from_clean": (
                float(clean_minimal["macro_f1_recomputed"])
                - float(minimal["macro_f1_recomputed"])
            ),
            "camera_drop_from_clean": (
                float(clean_camera["macro_f1_recomputed"])
                - float(camera["macro_f1_recomputed"])
            ),
        }
        for column, expected_value in expected.items():
            if abs(float(getattr(row, column)) - expected_value) > TOLERANCE:
                failures.append(
                    f"Augmentation effect differs: {row.architecture}/{row.condition}/{column}"
                )
    return failures


def audit_validation_results(
    *,
    root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    data_root: str | Path | None = None,
    reference_manifest: str | Path | None = None,
) -> dict[str, object]:
    root = Path(root)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "independent_audit_results.json"
    metric_path = output_dir / "independent_metric_recalculation.csv"
    if result_path.exists() or metric_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit artifacts in {output_dir}")

    manifest = pd.read_csv(manifest_path)
    active = manifest.loc[manifest.status == "ok"].copy()
    validation = active.loc[active.split == "val"].copy()
    validation_by_path = validation.set_index("path", verify_integrity=True)
    class_mapping = (
        active[["class_index", "class_name"]]
        .drop_duplicates()
        .sort_values("class_index")
    )
    class_names = class_mapping.class_name.tolist()
    class_index_to_name = dict(
        zip(class_mapping.class_index.astype(int), class_names, strict=True)
    )
    probability_columns = [f"prob_{name}" for name in class_names]
    manifest_failures: list[str] = []
    if len(manifest) != 1800:
        manifest_failures.append(f"manifest rows={len(manifest)}")
    if len(active) != 1799:
        manifest_failures.append(f"active rows={len(active)}")
    expected_splits = {"train": 1259, "val": 180, "test": 360}
    split_counts = active.split.value_counts().to_dict()
    if split_counts != expected_splits:
        manifest_failures.append(f"split counts differ: {split_counts}")
    validation_class_counts = validation.class_name.value_counts().sort_index().to_dict()
    if set(validation_class_counts.values()) != {30}:
        manifest_failures.append(f"validation class counts differ: {validation_class_counts}")
    validation_groups = validation.groupby("group_id", observed=True).path.agg(list)
    if len(validation_groups) != 176:
        manifest_failures.append(f"validation source groups={len(validation_groups)}")
    path_overlaps = cross_split_overlaps(active, "path")
    hash_overlaps = cross_split_overlaps(active, "sha256")
    group_overlaps = cross_split_overlaps(active, "group_id")
    if path_overlaps:
        manifest_failures.append("Cross-split path overlap detected")
    if hash_overlaps:
        manifest_failures.append("Cross-split SHA-256 overlap detected")
    if group_overlaps:
        manifest_failures.append("Cross-split group overlap detected")

    reference_identical: bool | None = None
    if reference_manifest is not None:
        reference = pd.read_csv(reference_manifest)
        reference_identical = manifest.equals(reference)
        if not reference_identical:
            manifest_failures.append("Reference manifest rows differ")

    raw_missing: list[str] = []
    raw_hash_mismatches: list[str] = []
    if data_root is not None:
        raw_root = Path(data_root)
        for row in manifest.itertuples(index=False):
            source = raw_root / row.path
            if not source.is_file():
                raw_missing.append(row.path)
            elif sha256_file(source) != row.sha256:
                raw_hash_mismatches.append(row.path)
        if raw_missing:
            manifest_failures.append(f"Missing raw files: {len(raw_missing)}")
        if raw_hash_mismatches:
            manifest_failures.append(f"Raw hash mismatches: {len(raw_hash_mismatches)}")

    evaluation_root = root / "evaluation"
    prediction_files = sorted(evaluation_root.glob("*/*/predictions.csv"))
    expected_keys = {
        (configuration, condition)
        for configuration in CONFIGURATIONS
        for condition in CONDITIONS
    }
    actual_keys = {
        (path.parents[1].name, path.parent.name) for path in prediction_files
    }
    prediction_failures: list[str] = []
    if actual_keys != expected_keys:
        prediction_failures.append(
            f"Prediction key set differs; missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    metric_rows: list[dict[str, object]] = []
    error_keys: list[tuple[str, str, str, str, str]] = []
    reference_paths = set(validation_by_path.index)
    for predictions_path in prediction_files:
        configuration = predictions_path.parents[1].name
        condition = predictions_path.parent.name
        table = pd.read_csv(predictions_path)
        path_set = set(table.path)
        if len(table) != 180 or table.path.nunique() != 180:
            prediction_failures.append(
                f"{configuration}/{condition}: rows={len(table)}, unique={table.path.nunique()}"
            )
        if path_set != reference_paths:
            prediction_failures.append(f"{configuration}/{condition}: path set differs")
        expected_rows = validation_by_path.reindex(table.path)
        if expected_rows.class_name.isna().any():
            prediction_failures.append(f"{configuration}/{condition}: unknown path")
            continue
        if table.true_class.tolist() != expected_rows.class_name.tolist():
            prediction_failures.append(f"{configuration}/{condition}: true class differs")
        if table.true_index.astype(int).tolist() != expected_rows.class_index.astype(int).tolist():
            prediction_failures.append(f"{configuration}/{condition}: true index differs")

        predicted_names = [
            class_index_to_name[int(index)] for index in table.predicted_index
        ]
        if predicted_names != table.predicted_class.tolist():
            prediction_failures.append(f"{configuration}/{condition}: predicted class differs")
        probabilities = table[probability_columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            prediction_failures.append(f"{configuration}/{condition}: invalid probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0):
            prediction_failures.append(
                f"{configuration}/{condition}: probabilities do not sum to one"
            )
        predicted_indices = probabilities.argmax(axis=1)
        if not np.array_equal(predicted_indices, table.predicted_index.to_numpy(dtype=int)):
            prediction_failures.append(f"{configuration}/{condition}: argmax differs")
        if not np.allclose(
            probabilities.max(axis=1),
            table.confidence.to_numpy(dtype=float),
            atol=1e-6,
            rtol=0,
        ):
            prediction_failures.append(f"{configuration}/{condition}: confidence differs")
        independently_correct = (
            table.true_index.to_numpy(dtype=int) == table.predicted_index.to_numpy(dtype=int)
        )
        if not np.array_equal(parse_correct(table.correct), independently_correct):
            prediction_failures.append(f"{configuration}/{condition}: correct flag differs")

        accuracy, macro_f1, confusion, class_f1 = calculate_metrics(
            table, class_count=len(class_names)
        )
        metrics_path = predictions_path.parent / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        accuracy_delta = accuracy - float(metrics["accuracy"])
        macro_f1_delta = macro_f1 - float(metrics["macro_f1"])
        if abs(accuracy_delta) > TOLERANCE:
            prediction_failures.append(f"{configuration}/{condition}: accuracy differs")
        if abs(macro_f1_delta) > TOLERANCE:
            prediction_failures.append(f"{configuration}/{condition}: macro-F1 differs")
        saved_confusion = pd.read_csv(
            predictions_path.parent / "confusion_matrix.csv", index_col=0
        )
        matrix_match = (
            saved_confusion.index.tolist() == class_names
            and saved_confusion.columns.tolist() == class_names
            and np.array_equal(saved_confusion.to_numpy(dtype=int), confusion)
        )
        if not matrix_match:
            prediction_failures.append(f"{configuration}/{condition}: confusion differs")
        metric_rows.append(
            {
                "configuration": configuration,
                "condition": condition,
                "accuracy_recomputed": accuracy,
                "accuracy_saved": float(metrics["accuracy"]),
                "accuracy_delta": accuracy_delta,
                "macro_f1_recomputed": macro_f1,
                "macro_f1_saved": float(metrics["macro_f1"]),
                "macro_f1_delta": macro_f1_delta,
                "confusion_matrix_match": matrix_match,
                "errors": int((~independently_correct).sum()),
                **{
                    f"f1_{class_name}": float(class_f1[index])
                    for index, class_name in enumerate(class_names)
                },
            }
        )
        for row in table.loc[~independently_correct].itertuples(index=False):
            error_keys.append(
                (
                    configuration,
                    condition,
                    row.path,
                    row.true_class,
                    row.predicted_class,
                )
            )

    aggregate_failures = compare_aggregate_tables(root, metric_rows) if metric_rows else []
    observed_path = root / "observed_validation_errors.csv"
    observed_matches: bool | None = None
    if observed_path.is_file():
        observed = pd.read_csv(observed_path)
        observed_keys = [
            (
                row.configuration,
                row.condition,
                row.path,
                row.true_class,
                row.predicted_class,
            )
            for row in observed.itertuples(index=False)
        ]
        observed_matches = Counter(error_keys) == Counter(observed_keys)
        if not observed_matches:
            aggregate_failures.append("observed_validation_errors.csv differs")

    pd.DataFrame(metric_rows).sort_values(["configuration", "condition"]).to_csv(
        metric_path, index=False
    )
    result: dict[str, object] = {
        "status": "pass"
        if not (manifest_failures or prediction_failures or aggregate_failures)
        else "fail",
        "manifest": {
            "rows": len(manifest),
            "active_rows": len(active),
            "split_counts": split_counts,
            "validation_rows": len(validation),
            "validation_class_counts": validation_class_counts,
            "validation_source_groups": len(validation_groups),
            "cross_split_path_overlaps": path_overlaps,
            "cross_split_sha256_overlaps": hash_overlaps,
            "cross_split_group_overlaps": group_overlaps,
            "reference_manifest_identical": reference_identical,
            "raw_files_checked": len(manifest) if data_root is not None else 0,
            "missing_raw_files": raw_missing,
            "raw_hash_mismatches": raw_hash_mismatches,
            "failures": manifest_failures,
        },
        "evaluation": {
            "prediction_files": len(prediction_files),
            "metric_rows": len(metric_rows),
            "max_abs_accuracy_delta": max(
                (abs(float(row["accuracy_delta"])) for row in metric_rows), default=None
            ),
            "max_abs_macro_f1_delta": max(
                (abs(float(row["macro_f1_delta"])) for row in metric_rows), default=None
            ),
            "failures": prediction_failures,
            "aggregate_failures": aggregate_failures,
        },
        "errors": {
            "recomputed_error_rows": len(error_keys),
            "unique_validation_paths": len({key[2] for key in error_keys}),
            "observed_error_key_multiset_matches": observed_matches,
        },
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if result["status"] != "pass":
        raise ValueError(f"Validation audit failed; inspect {result_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Pilot artifact root containing evaluation/ and aggregate CSV files",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = audit_validation_results(
        root=args.root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        data_root=args.data_root,
        reference_manifest=args.reference_manifest,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
