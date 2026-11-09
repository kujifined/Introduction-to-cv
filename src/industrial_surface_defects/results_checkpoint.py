"""Complete the registered matrix, freeze it, and evaluate the fixed test suite."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .compare import REGISTERED_CONFIGURATIONS, REGISTERED_SEEDS, compare
from .constants import CLASS_NAMES
from .corruptions import (
    CORRUPTION_NAMES,
    SEVERITY_VALUES,
    SHOT_READ_NOISE_SETTINGS,
    CorruptedManifestDataset,
    apply_corruption,
)
from .data import ManifestDataset, validate_manifest
from .evaluate import benchmark_latency
from .metrics import compute_metrics, confusion_pairs, prediction_table
from .models import count_parameters, load_checkpoint_model
from .plots import plot_confusion_matrix
from .reproducibility import runtime_metadata, seed_everything, write_json
from .robustness import aggregate_registered_robustness
from .transforms import build_transform
from .weekly_validation import (
    CORE_REVISION,
    MANIFEST_SHA256,
    audit_predictions,
    environment_differences,
    predict_and_fingerprint,
    verify_training,
)
from .weekly_validation import (
    build_configs as build_seed43_configs,
)

CONFIGURATIONS = tuple(REGISTERED_CONFIGURATIONS)
SEEDS = tuple(REGISTERED_SEEDS)
TEST_CONDITIONS = (("clean", 0),) + tuple(
    (corruption, severity) for corruption in CORRUPTION_NAMES for severity in (1, 2, 3)
)
SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "negative_log_likelihood",
    "expected_calibration_error_10_bins",
    "brier_score",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def build_matrix_configs(project, output_root):
    base = build_seed43_configs(project, output_root)
    return {
        (name, seed): replace(config, experiment=replace(config.experiment, seed=seed))
        for name, config in base.items()
        for seed in SEEDS
    }


def matrix_checkpoints(root):
    return {
        (name, seed): Path(root) / "outputs" / name / f"seed_{seed}" / "best.pt"
        for name in CONFIGURATIONS
        for seed in SEEDS
    }


def fixed_protocol():
    return {
        "name": "results_checkpoint_v1",
        "core_revision": CORE_REVISION,
        "configurations": list(CONFIGURATIONS),
        "training_seeds": list(SEEDS),
        "manifest_sha256": MANIFEST_SHA256,
        "training_workers": 2,
        "pretrained_weights": {"resnet18": "IMAGENET1K_V1", "efficientnet_b0": "IMAGENET1K_V1"},
        "checkpoint_selection": "earliest strict maximum clean validation macro-F1; patience 10",
        "augmentation_version": "camera_aware_v1; registered settings unchanged",
        "primary_metric": "six-class macro-F1",
        "summary_metrics": list(SUMMARY_METRICS),
        "test_images": 360,
        "test_classes": 6,
        "test_support_per_class": 60,
        "conditions": [{"corruption": c, "severity": s} for c, s in TEST_CONDITIONS],
        "severity_values": SEVERITY_VALUES,
        "shot_read_noise_settings": SHOT_READ_NOISE_SETTINGS,
        "noise_seed": "SHA256(path + NUL + corruption + NUL + severity), independent of model seed",
        "evaluation_precision": "FP32",
        "evaluation_batch_size": 32,
        "evaluation_workers": 2,
        "latency": {"batch_size": 1, "warmup_runs": 10, "timed_runs": 50, "precision": "FP32"},
        "latency_excludes": ["loading", "preprocessing", "host_to_device_transfer"],
        "aggregation": (
            "per seed, then mean and sample SD across 42/43/44; no best seed or ensemble"
        ),
        "robustness_mean": "unweighted mean of 27 condition-level macro-F1 scores, excluding clean",
        "bootstrap": {
            "repetitions": 5000,
            "base_seed": 42,
            "method": "registered paired stratified test-image bootstrap, fixed seed mean",
            "scope": "five preregistered clean macro-F1 contrasts; training-seed SD is separate",
        },
        "failure_selection": {
            "priority": "clean errors first, then corrupted errors",
            "order": "confidence descending, then configuration, seed, corruption, severity, path",
            "maximum_candidates": 12,
            "unique_source_paths": True,
            "prefer_diverse_true_classes_and_corruption_families": True,
            "all_errors_retained": True,
            "explanations": "human visual hypotheses; no invented mechanisms or modified severity",
        },
        "after_test_policy": "no changes to training, labels, split, checkpoints or severity",
        "interpretation": "360 paired source crops, not 12*28*360 independent observations",
    }


def _copy_verified(content, target, digest):
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Historical artifact hash mismatch: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(content)


def import_seed43(input_root, artifact_root, expectations):
    """Read known files from attached notebook output or an allowlisted input ZIP."""
    input_root, root = Path(input_root), Path(artifact_root)
    originals = root / "historical_source"
    if originals.exists():
        raise FileExistsError("Historical import already exists; start a fresh run")
    files = expectations["files"]
    candidates = set()
    for marker in input_root.rglob("checkpoint_provenance.json"):
        candidates.add(marker.parent)
    source_label = None
    for source in sorted(candidates):
        if all(
            (source / p).is_file() and sha256(source / p) == digest for p, digest in files.items()
        ):
            for relative, digest in files.items():
                _copy_verified((source / relative).read_bytes(), originals / relative, digest)
            source_label = str(source)
            break
    if source_label is None:
        for filename in ("icv_seed43_controls.zip", "icv_weekly_results.zip"):
            for source in sorted(input_root.rglob(filename)):
                with zipfile.ZipFile(source) as archive:
                    names = archive.namelist()
                    if len(names) != len(set(names)) or not set(files).issubset(names):
                        continue
                    for relative in files:
                        if archive.getinfo(relative).file_size > 60_000_000:
                            raise ValueError("Unexpected historical file size")
                        if hashlib.sha256(archive.read(relative)).hexdigest() != files[relative]:
                            raise ValueError(f"Historical archive content differs: {relative}")
                    for relative, digest in files.items():
                        _copy_verified(archive.read(relative), originals / relative, digest)
                source_label = str(source)
                break
            if source_label is not None:
                break
    if source_label is None:
        raise FileNotFoundError(
            "No verified seed-43 results in /kaggle/input. Attach the OUTPUT of "
            "datastr3am/icv-progress-check version 362621632 via Add Input, or upload "
            "icv_seed43_controls.zip as a private input dataset. No training started."
        )
    provenance = {}
    for name, item in expectations["runs"].items():
        source_dir = originals / item["source_directory"]
        target_dir = root / "outputs" / name / "seed_43"
        target_dir.mkdir(parents=True, exist_ok=False)
        for filename in item["run_files"]:
            shutil.copyfile(source_dir / filename, target_dir / filename)
        provenance[f"{name}/43"] = {
            "origin": "historical_seed43",
            "source_version": 362621632,
            "original_directory": item["source_directory"],
            "original_training_runtime": read_json(source_dir / "runtime.json"),
            "checkpoint_sha256": sha256(target_dir / "best.pt"),
            "trained_in_current_session": False,
        }
    write_json(root / "control_import.json", {"source": source_label, "all_hashes_verified": True})
    return provenance


def check_control_environment(root, current_runtime, current_packages, expectations):
    differences = {}
    for name, item in expectations["runs"].items():
        historical = read_json(Path(root) / "outputs" / name / "seed_43/runtime.json")
        mismatch = environment_differences(current_runtime, historical)
        for package in ("pillow", "pandas"):
            old = item["training_packages"][package]
            if old != current_packages[package]:
                mismatch[package] = {"historical": old, "current": current_packages[package]}
        if mismatch:
            differences[name] = mismatch
    return differences


def audit_validation(root, configs, manifest):
    manifest = Path(manifest)
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen manifest changed")
    rows = validate_manifest(manifest).query("split == 'val'").reset_index(drop=True)
    if len(rows) != 180 or rows.class_name.value_counts().tolist() != [30] * 6:
        raise ValueError("Wrong validation split")
    checked = []
    for key, checkpoint_path in matrix_checkpoints(root).items():
        name, seed = key
        summary = verify_training(checkpoint_path, configs[key])
        destination = checkpoint_path.parent / "evaluation/val"
        metrics = read_json(destination / "metrics.json")
        manual, confusion = audit_predictions(
            pd.read_csv(destination / "predictions.csv"), rows, metrics
        )
        if metrics["split"] != "val" or int(metrics["seed"]) != seed:
            raise ValueError("Wrong validation metadata")
        if not np.isclose(manual["macro_f1"], summary["best_validation_macro_f1"], atol=1e-10):
            raise ValueError("Fresh validation differs from the selected checkpoint")
        stored = pd.read_csv(destination / "confusion_matrix.csv", index_col=0).to_numpy()
        if not np.array_equal(stored, confusion):
            raise ValueError("Validation confusion differs")
        checked.append({"configuration": name, "seed": seed, **manual})
    write_json(Path(root) / "validation_audit.json", {"status": "pass", "runs": checked})
    return checked


def write_freeze(root, project, configs, manifest):
    root, project = Path(root), Path(project)
    if (root / "protocol_freeze.json").exists():
        raise FileExistsError("Existing freeze is immutable")
    audit_validation(root, configs, manifest)
    checkpoint_files = matrix_checkpoints(root)
    files = {}
    for checkpoint in checkpoint_files.values():
        if (checkpoint.parent / "evaluation/test").exists():
            raise ValueError("Test artifacts exist before freeze")
        for filename in (
            "best.pt",
            "config.resolved.yaml",
            "history.csv",
            "training_summary.json",
            "runtime.json",
        ):
            path = checkpoint.parent / filename
            files[path.relative_to(root).as_posix()] = sha256(path)
        for filename in ("predictions.csv", "metrics.json"):
            path = checkpoint.parent / "evaluation/val" / filename
            files[path.relative_to(root).as_posix()] = sha256(path)
    for filename in (
        "experiment_protocol.json",
        "checkpoint_provenance.json",
        "validation_audit.json",
    ):
        files[filename] = sha256(root / filename)
    source_hashes = {
        path.relative_to(project).as_posix(): sha256(path)
        for folder in ("src", "configs")
        for path in sorted((project / folder).rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml"}
    }
    freeze = {
        "status": "frozen_before_test",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": sha256(manifest),
        "protocol": fixed_protocol(),
        "files": files,
        "source_files": source_hashes,
        "checkpoint_count": len(checkpoint_files),
        "test_evaluation_started": False,
        "human_post_run_reviews_complete": False,
    }
    write_json(root / "protocol_freeze.json", freeze)
    (root / "protocol_freeze.sha256").write_text(sha256(root / "protocol_freeze.json") + "\n")
    return freeze


def assert_frozen(root, project, manifest, *, acknowledge_test_use):
    if not acknowledge_test_use:
        raise PermissionError("Final test requires explicit notebook acknowledgement")
    root, project = Path(root), Path(project)
    freeze_path = root / "protocol_freeze.json"
    if sha256(freeze_path) != (root / "protocol_freeze.sha256").read_text().strip():
        raise ValueError("Freeze record changed")
    freeze = read_json(freeze_path)
    # JSON round-trip canonicalises tuples in the severity settings.
    if freeze["protocol"] != json.loads(json.dumps(fixed_protocol())):
        raise ValueError("Evaluation protocol changed after freeze")
    if freeze["checkpoint_count"] != 12 or sha256(manifest) != freeze["manifest_sha256"]:
        raise ValueError("Checkpoint matrix or manifest changed")
    for relative, expected in freeze["files"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
    for relative, expected in freeze["source_files"].items():
        if sha256(project / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    return freeze


def test_dataset(manifest, data_root, corruption, severity):
    if (corruption, severity) not in TEST_CONDITIONS:
        raise ValueError("Unregistered test condition")
    if corruption == "clean":
        return ManifestDataset(
            manifest,
            data_root,
            "test",
            build_transform(224, "minimal", train=False),
            return_path=True,
        )
    return CorruptedManifestDataset(
        manifest,
        data_root,
        split="test",
        image_size=224,
        corruption=corruption,
        severity=severity,
    )


def _save_evaluation(destination, paths, targets, scores, expected, metadata, *, latency=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Evaluation preserved: {destination}")
    destination.mkdir(parents=True)
    metrics, per_class, confusion, normalised = compute_metrics(targets, scores)
    table = prediction_table(paths, targets, scores)
    table.to_csv(destination / "predictions.csv", index=False)
    manual, independent_confusion = audit_predictions(
        pd.read_csv(destination / "predictions.csv"),
        expected,
        metrics,
    )
    if not np.array_equal(confusion, independent_confusion):
        raise ValueError("Independent confusion mismatch")
    per_class.to_csv(destination / "per_class_metrics.csv", index=False)
    confusion_pairs(confusion).to_csv(destination / "confusion_pairs.csv", index=False)
    pd.DataFrame(confusion, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        destination / "confusion_matrix.csv"
    )
    pd.DataFrame(normalised, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        destination / "confusion_matrix_normalised.csv"
    )
    metrics.update(metadata)
    if latency is not None:
        metrics["latency"] = latency
        plot_confusion_matrix(confusion, destination / "confusion_matrix.png", normalised=False)
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "independent_audit.json", {"status": "pass", **manual})
    return metrics, table, per_class


def evaluate_test_matrix(root, project, manifest, data_root, *, acknowledge_test_use):
    root, project = Path(root), Path(project)
    assert_frozen(root, project, manifest, acknowledge_test_use=acknowledge_test_use)
    if (root / "test_started.json").exists():
        raise FileExistsError("Test stage already started; preserve evidence and inspect the run")
    expected = validate_manifest(manifest).query("split == 'test'").reset_index(drop=True)
    if len(expected) != 360 or expected.class_name.value_counts().tolist() != [60] * 6:
        raise ValueError("Wrong test split")
    write_json(
        root / "test_started.json",
        {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "freeze_sha256": sha256(root / "protocol_freeze.json"),
            "acknowledge_test_use": True,
            "acknowledgement_source": "explicit configuration in the user-launched Kaggle notebook",
        },
    )
    device = torch.device("cuda")
    all_scores, all_errors, fingerprints = [], [], {}
    provenance = read_json(root / "checkpoint_provenance.json")
    for (name, seed), checkpoint_path in matrix_checkpoints(root).items():
        assert_frozen(root, project, manifest, acknowledge_test_use=True)
        gc.collect()
        torch.cuda.empty_cache()
        seed_everything(seed)
        model, checkpoint = load_checkpoint_model(str(checkpoint_path), device=device)
        model.eval()
        training_summary = read_json(checkpoint_path.parent / "training_summary.json")
        runtime = runtime_metadata(device)
        run_scores, run_predictions, run_classes = [], [], []
        clean_f1 = None
        for corruption, severity in TEST_CONDITIONS:
            condition = "clean" if corruption == "clean" else f"{corruption}_s{severity}"
            print(f"TEST {name} seed={seed} {condition}", flush=True)
            dataset = test_dataset(manifest, data_root, corruption, severity)
            loader = DataLoader(
                dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
            )
            paths, targets, scores, fingerprint = predict_and_fingerprint(model, loader, device)
            if condition in fingerprints and fingerprint != fingerprints[condition]:
                raise ValueError("Models/seeds received different test tensors")
            fingerprints[condition] = fingerprint
            latency = None
            if corruption == "clean":
                sample, _, _ = next(iter(loader))
                latency = benchmark_latency(model, sample, device, warmup=10, repetitions=50)
            evaluation_dir = checkpoint_path.parent / "evaluation"
            destination = evaluation_dir / (
                "test" if corruption == "clean" else f"corruptions/{condition}"
            )
            metrics, table, per_class = _save_evaluation(
                destination,
                paths,
                targets,
                scores,
                expected,
                {
                    "configuration": name,
                    "model_name": checkpoint["config"]["model"]["name"],
                    "augmentation_training": checkpoint["config"]["data"]["augmentation"],
                    "seed": seed,
                    "split": "test",
                    "corruption": corruption,
                    "severity": severity,
                    "manifest_sha256": MANIFEST_SHA256,
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "input_tensor_sha256": fingerprint,
                    "runtime": runtime,
                    "test_use_acknowledged": True,
                    "freeze_sha256": sha256(root / "protocol_freeze.json"),
                },
                latency=latency,
            )
            if clean_f1 is None:
                clean_f1 = metrics["macro_f1"]
            numeric = {
                key: value for key, value in metrics.items() if isinstance(value, (int, float))
            }
            row = {
                "configuration": name,
                "seed": seed,
                "corruption": corruption,
                "severity": severity,
                **numeric,
            }
            row["macro_f1_absolute_drop"] = clean_f1 - metrics["macro_f1"]
            row["macro_f1_relative_drop"] = (
                row["macro_f1_absolute_drop"] / clean_f1 if clean_f1 else 0.0
            )
            run_scores.append(row)
            if corruption == "clean":
                write_json(
                    checkpoint_path.parent / "runtime_measurements.json",
                    {
                        "parameters": count_parameters(model),
                        "checkpoint_bytes": checkpoint_path.stat().st_size,
                        "selected_epoch": training_summary["best_epoch"],
                        "epochs_completed": training_summary["epochs_completed"],
                        "training_stage_seconds": training_summary["elapsed_seconds"],
                        "training_origin": provenance[f"{name}/{seed}"]["origin"],
                        "training_runtime": provenance[f"{name}/{seed}"][
                            "original_training_runtime"
                        ],
                        "inference_runtime": runtime,
                        "latency": latency,
                    },
                )
            table.insert(0, "severity", severity)
            table.insert(0, "corruption", corruption)
            run_predictions.append(table)
            per_class.insert(0, "severity", severity)
            per_class.insert(0, "corruption", corruption)
            run_classes.append(per_class)
            errors = table.loc[~table.correct].copy()
            errors.insert(0, "seed", seed)
            errors.insert(0, "configuration", name)
            all_errors.append(errors)
        robust_dir = checkpoint_path.parent / "evaluation/robustness"
        robust_dir.mkdir(parents=True)
        pd.DataFrame(run_scores).to_csv(robust_dir / "robustness_metrics.csv", index=False)
        pd.concat(run_predictions, ignore_index=True).to_csv(
            robust_dir / "robustness_predictions.csv", index=False
        )
        pd.concat(run_classes, ignore_index=True).to_csv(
            robust_dir / "robustness_per_class_metrics.csv", index=False
        )
        all_scores.extend(run_scores)
        del model, checkpoint
    results = pd.DataFrame(all_scores)
    results.to_csv(root / "test_scores_all_conditions.csv", index=False)
    pd.concat(all_errors, ignore_index=True).to_csv(root / "observed_test_errors.csv", index=False)
    write_json(
        root / "test_evaluation_audit.json",
        {
            "status": "pass",
            "checkpoints": 12,
            "conditions_per_checkpoint": 28,
            "condition_rows": len(results),
            "prediction_rows": 12 * 28 * 360,
            "unique_source_images": 360,
            "identical_input_hashes": fingerprints,
            "independent_prediction_audits": len(results),
            "human_post_run_reviews_complete": False,
        },
    )
    assert_frozen(root, project, manifest, acknowledge_test_use=True)
    return results


def aggregate_results(root):
    root = Path(root)
    scores = pd.read_csv(root / "test_scores_all_conditions.csv")
    if (
        len(scores) != 336
        or scores.duplicated(["configuration", "seed", "corruption", "severity"]).any()
    ):
        raise ValueError("Incomplete final evaluation")
    clean = scores.loc[scores.corruption == "clean"]
    clean.to_csv(root / "clean_scores_by_run.csv", index=False)
    summary = clean.groupby("configuration")[list(SUMMARY_METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reindex(CONFIGURATIONS).to_csv(root / "clean_scores_by_configuration.csv")
    specifications = [
        f"{name}={checkpoint.parent / 'evaluation/test/predictions.csv'}"
        for (name, _), checkpoint in matrix_checkpoints(root).items()
    ]
    compare(
        specifications,
        output_dir=root / "comparison_clean",
        repetitions=5000,
        seed=42,
        registered_design=True,
    )
    aggregate_registered_robustness(root / "outputs", output_dir=root / "comparison_robustness")
    runtime_rows = []
    for (name, seed), checkpoint in matrix_checkpoints(root).items():
        info = read_json(checkpoint.parent / "runtime_measurements.json")
        runtime_rows.append(
            {
                "configuration": name,
                "seed": seed,
                **{
                    key: value
                    for key, value in info.items()
                    if key not in {"training_runtime", "inference_runtime", "latency"}
                },
                "median_latency_ms": info["latency"]["median_latency_ms"],
                "p90_latency_ms": info["latency"]["p90_latency_ms"],
                "inference_device": info["inference_runtime"]["device_name"],
            }
        )
    pd.DataFrame(runtime_rows).to_csv(root / "runtime_summary.csv", index=False)
    mean_curves = scores.groupby(["configuration", "corruption", "severity"]).macro_f1.agg(
        ["mean", "std"]
    )
    mean_curves.to_csv(root / "severity_curve_summary.csv")
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)
    for axis, corruption in zip(axes.flat, CORRUPTION_NAMES, strict=True):
        for name in CONFIGURATIONS:
            clean_scores = scores.loc[
                (scores.configuration == name) & (scores.corruption == "clean"), "macro_f1"
            ]
            rows = mean_curves.loc[(name, corruption)].sort_index()
            means = np.r_[clean_scores.mean(), rows["mean"].to_numpy()]
            deviations = np.r_[clean_scores.std(ddof=1), rows["std"].to_numpy()]
            axis.plot(range(4), means, marker="o", label=name)
            axis.fill_between(
                range(4),
                np.maximum(0, means - deviations),
                np.minimum(1, means + deviations),
                alpha=0.12,
            )
        axis.set(title=corruption, xlabel="Severity (0 = clean)", ylim=(0, 1.02), xticks=range(4))
    axes[0, 0].set_ylabel("Test macro-F1, mean +/- seed SD")
    axes[-1, -1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "test_severity_curves.png", dpi=160)
    plt.close(figure)


def select_failure_candidates(errors, maximum=12):
    if errors.empty:
        return errors.copy()
    ordered = errors.copy()
    ordered["clean_priority"] = ordered.corruption.ne("clean").astype(int)
    ordered = ordered.sort_values(
        ["clean_priority", "confidence", "configuration", "seed", "corruption", "severity", "path"],
        ascending=[True, False, True, True, True, True, True],
        kind="stable",
    )
    selected, paths, classes, families = [], set(), set(), set()
    for require_diversity in (True, False):
        for index, row in ordered.iterrows():
            if row.path in paths:
                continue
            if require_diversity and row.true_class in classes and row.corruption in families:
                continue
            selected.append(index)
            paths.add(row.path)
            classes.add(row.true_class)
            families.add(row.corruption)
            if len(selected) == maximum:
                return ordered.loc[selected].drop(columns="clean_priority")
    return ordered.loc[selected].drop(columns="clean_priority")


def failure_materials(root, data_root):
    root = Path(root)
    errors = pd.read_csv(root / "observed_test_errors.csv")
    selected = select_failure_candidates(errors)
    selected.to_csv(root / "failure_candidates.csv", index=False)
    write_json(
        root / "failure_selection.json",
        {
            "error_rows": len(errors),
            "unique_error_source_paths": int(errors.path.nunique()),
            "selected_distinct_images": len(selected),
            "minimum_three_available": len(selected) >= 3,
            "human_explanations_complete": False,
            "zero_error_policy": "report zero honestly; do not change severity or invent examples",
        },
    )
    if selected.empty:
        return
    figure, axes = plt.subplots(len(selected), 2, figsize=(8, 3 * len(selected)), squeeze=False)
    for index, row in enumerate(selected.itertuples(index=False)):
        with Image.open(Path(data_root) / row.path) as source:
            original = source.convert("L")
        evaluated = (
            original
            if row.corruption == "clean"
            else apply_corruption(
                original,
                row.corruption,
                int(row.severity),
                sample_key=row.path,
            )
        )
        for axis, image in zip(axes[index], (original, evaluated), strict=True):
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            axis.axis("off")
        axes[index, 0].set_title(f"{row.path}\ntrue: {row.true_class}", fontsize=9)
        axes[index, 1].set_title(
            f"{row.configuration}, seed {row.seed}, {row.corruption} s{row.severity}\n"
            f"predicted: {row.predicted_class}, confidence {row.confidence:.3f}",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(root / "failure_candidates.png", dpi=110)
    plt.close(figure)


def export_bundle(root, zip_path, *, status):
    root, zip_path = Path(root), Path(zip_path)
    if zip_path.exists():
        raise FileExistsError("Existing archive is preserved")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Results may not contain symlinks")
        if path.is_file():
            if path.suffix.lower() in {".bmp", ".rar"}:
                raise ValueError("Raw dataset may not enter results bundle")
            if path.name != "bundle_manifest.json":
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    write_json(
        root / "bundle_manifest.json",
        {"status": status, "files": entries, "human_post_run_reviews_complete": False},
    )
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Output ZIP failed CRC verification")
    return sha256(zip_path)
