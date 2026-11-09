"""Four fixed seed-43 conditions and a separate, validation-only development pilot."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from .config import load_config
from .constants import CLASS_NAMES
from .corruptions import (
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
from .transforms import CameraAwareAugmentation, build_transform

SEED = 43
CORE_REVISION = "760889adc87264122efe67d04e966dfeaaa319d8"
MANIFEST_SHA256 = "df784b6590f90947db6fd40c8d2e9650ab168744a0bd400e13dc535927f8232d"
RUN_SUFFIX = "weekly_augmentation_20261026"
CONFIGURATIONS = (
    "resnet18_minimal",
    "resnet18_camera",
    "efficientnet_b0_minimal",
    "efficientnet_b0_camera",
)
PILOT_CORRUPTIONS = ("brightness_dark", "gaussian_blur", "shot_read_noise")
CONDITIONS = ("clean",) + tuple(
    f"{name}_s{severity}" for name in PILOT_CORRUPTIONS for severity in (1, 2, 3)
)
PARAMETERS = {"resnet18": 11179590, "efficientnet_b0": 4015234}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pilot_protocol():
    return {
        "name": "validation_camera_pilot_v1",
        "split": "val",
        "seed": SEED,
        "conditions": list(CONDITIONS),
        "brightness_factors": list(SEVERITY_VALUES["brightness_dark"]),
        "gaussian_blur_radius_pixels_before_resize": list(SEVERITY_VALUES["gaussian_blur"]),
        "shot_read_noise_peak_sigma": [list(item) for item in SHOT_READ_NOISE_SETTINGS],
        "corruption_seed": "SHA256(relative_path + NUL + corruption + NUL + severity)",
        "training_augmentation": "camera_aware_v1; unchanged registered source",
        "checkpoint_selection": "earliest strict maximum clean validation macro-F1",
        "pilot_used_for_checkpoint_or_hyperparameter_selection": False,
        "evaluation_precision": "FP32",
        "latency": {"batch_size": 1, "warmup_runs": 10, "timed_runs": 50},
        "test_evaluation_enabled": False,
        "interpretation": (
            "Development analysis on the same 180 validation crops, not independent test evidence. "
            "Transformed views are paired and are not additional independent observations. "
            "Do not tune severity, seeds, or training after viewing these results."
        ),
    }


def build_configs(project_root, output_root):
    result = {}
    for name in CONFIGURATIONS:
        config = load_config(Path(project_root) / f"configs/{name}.yaml", seed=SEED, device="cuda")
        config = replace(
            config,
            experiment=replace(config.experiment, output_root=str(output_root)),
            data=replace(
                config.data,
                root=str(Path(project_root) / "data/raw/NEU-CLS"),
                manifest=str(Path(project_root) / "data/manifests/neu_cls_v2.csv"),
            ),
            training=replace(config.training, num_workers=2, amp=True),
        )
        result[name] = config
    reference = result["resnet18_minimal"].training
    if any(config.training != reference for config in result.values()):
        raise ValueError("Training settings differ between conditions")
    return result


def environment_differences(current, historical):
    differences = {}
    for key in ("torch", "torchvision", "cuda", "device_name", "numpy", "scikit_learn"):
        if current.get(key) != historical.get(key):
            differences[key] = {"current": current.get(key), "historical": historical.get(key)}
    # Patch versions of Python may change without changing the experiment runtime series.
    if current["python"].split(".")[:2] != historical["python"].split(".")[:2]:
        differences["python_series"] = {
            "current": current["python"],
            "historical": historical["python"],
        }
    return differences


def import_controls(input_root, destination, expectations):
    """Import only the byte-verified historical artifacts, never arbitrary archive members."""
    input_root, destination = Path(input_root), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Controls destination exists: {destination}")
    expected_files = expectations["files"]
    # A Kaggle notebook-output input exposes the existing outputs/ tree directly.
    roots = set()
    for checkpoint in input_root.rglob("best.pt"):
        if checkpoint.parent.parent.name == "resnet18_minimal":
            roots.add(checkpoint.parent.parent.parent.parent)
    for root in sorted(roots):
        if all(
            (root / relative).is_file() and sha256(root / relative) == digest
            for relative, digest in expected_files.items()
        ):
            for relative in expected_files:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, target)
            return {
                "source": str(root),
                "kind": "attached_notebook_output",
                "hashes_verified": True,
            }
    # A small supplied control bundle is an alternative to attaching the old notebook output.
    candidates = sorted(input_root.rglob("icv_minimal_controls_seed43.zip"))
    candidates += sorted(input_root.rglob("icv_pair_results.zip"))
    for archive_path in candidates:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                continue
            if not set(expected_files).issubset(names):
                continue
            contents = {}
            for relative, digest in expected_files.items():
                info = archive.getinfo(relative)
                if info.file_size > 60_000_000:
                    raise ValueError("Unexpected control artifact size")
                content = archive.read(relative)
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"Historical control hash mismatch: {relative}")
                contents[relative] = content
            for relative, content in contents.items():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            return {
                "source": str(archive_path),
                "kind": "attached_control_zip",
                "hashes_verified": True,
            }
    raise FileNotFoundError(
        "Attach the output of datastr3am/icv-second-approach using Add Input, or upload "
        "icv_minimal_controls_seed43.zip as a private Kaggle dataset and attach it. "
        "No training has started. See RUN_KAGGLE.md."
    )


def verify_training(checkpoint_path, expected_config):
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    expected = expected_config.to_dict()
    for section, keys in {
        "experiment": ("name", "seed"),
        "data": ("image_size", "augmentation"),
        "model": ("name", "pretrained"),
        "training": tuple(expected["training"]),
    }.items():
        for key in keys:
            if config[section][key] != expected[section][key]:
                raise ValueError(f"Unexpected checkpoint setting: {section}.{key}")
    resolved = yaml.safe_load((path.parent / "config.resolved.yaml").read_text())
    if resolved != config:
        raise ValueError("Resolved configuration and checkpoint differ")
    if checkpoint["class_names"] != list(CLASS_NAMES) or checkpoint["num_classes"] != 6:
        raise ValueError("Unexpected output classes")
    if checkpoint["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Unexpected checkpoint manifest")
    if any(not torch.isfinite(value).all() for value in checkpoint["model_state"].values()):
        raise ValueError("Non-finite checkpoint tensors")
    history = pd.read_csv(path.parent / "history.csv")
    numeric = history.select_dtypes(include="number").to_numpy()
    if history.empty or not np.isfinite(numeric).all():
        raise ValueError("Empty or non-finite history")
    if history.epoch.tolist() != list(range(1, len(history) + 1)):
        raise ValueError("Non-consecutive training epochs")
    best_score, best_epoch, waiting = -1.0, 0, 0
    for row in history.itertuples(index=False):
        if row.val_macro_f1 > best_score:
            best_score, best_epoch, waiting = row.val_macro_f1, row.epoch, 0
        else:
            waiting += 1
        if waiting >= 10 and row.epoch != len(history):
            raise ValueError("Training continued beyond early stopping")
    if len(history) > 50 or (len(history) < 50 and waiting < 10):
        raise ValueError("Incomplete training history")
    summary = json.loads((path.parent / "training_summary.json").read_text())
    if (
        summary.get("status") != "complete"
        or summary["best_epoch"] != best_epoch
        or checkpoint["best_epoch"] != best_epoch
        or summary["epochs_completed"] != len(history)
        or summary["test_split_accessed"] is not False
        or not np.isclose(summary["best_validation_macro_f1"], best_score, atol=1e-10)
        or not np.isclose(checkpoint["validation_metrics"]["macro_f1"], best_score, atol=1e-10)
        or summary["elapsed_seconds"] <= 0
        or summary["model_parameters"] != PARAMETERS[config["model"]["name"]]
    ):
        raise ValueError("Checkpoint selection or summary disagrees with history")
    del checkpoint
    return summary


def make_dataset(manifest, data_root, condition):
    if condition not in CONDITIONS:
        raise ValueError("Only the fixed validation pilot conditions are allowed")
    if condition == "clean":
        return ManifestDataset(
            manifest,
            data_root,
            "val",
            build_transform(224, "minimal", train=False),
            return_path=True,
        )
    corruption, severity = condition.rsplit("_s", 1)
    return CorruptedManifestDataset(
        manifest,
        data_root,
        split="val",
        image_size=224,
        corruption=corruption,
        severity=int(severity),
    )


@torch.inference_mode()
def predict_and_fingerprint(model, loader, device):
    paths, targets, scores = [], [], []
    digest = hashlib.sha256()
    for images, labels, batch_paths in loader:
        for image, label, path in zip(images, labels, batch_paths, strict=True):
            digest.update(str(path).encode() + b"\0" + str(int(label)).encode() + b"\0")
            digest.update(image.contiguous().numpy().tobytes())
        probabilities = model(images.to(device, non_blocking=True)).softmax(dim=1)
        paths.extend(batch_paths)
        targets.extend(labels.tolist())
        scores.append(probabilities.cpu().numpy())
    return paths, np.asarray(targets), np.concatenate(scores), digest.hexdigest()


def audit_predictions(table, expected_rows, stored_metrics):
    """Independently recalculate key metrics without sklearn or compute_metrics."""
    if len(table) != len(expected_rows) or table.path.duplicated().any():
        raise ValueError("Missing or duplicate validation predictions")
    if table.path.tolist() != expected_rows.path.tolist():
        raise ValueError("Wrong validation membership or order")
    targets = table.true_index.to_numpy(dtype=int)
    if not np.array_equal(targets, expected_rows.class_index.to_numpy()):
        raise ValueError("Incorrect validation labels")
    if table.true_class.tolist() != [CLASS_NAMES[index] for index in targets]:
        raise ValueError("Incorrect validation class names")
    probabilities = table[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Invalid probabilities")
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    if not np.array_equal(predictions, table.predicted_index.to_numpy()):
        raise ValueError("Stored predictions disagree with probabilities")
    if table.predicted_class.tolist() != [CLASS_NAMES[index] for index in predictions]:
        raise ValueError("Incorrect predicted class names")
    correctness = table.correct.astype(str).str.lower().to_numpy() == "true"
    if not np.array_equal(correctness, predictions == targets):
        raise ValueError("Incorrect correctness flags")
    confusion = np.zeros((6, 6), dtype=int)
    np.add.at(confusion, (targets, predictions), 1)
    tp = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(6), where=denominator != 0)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for index in range(10):
        selected = (confidence > index / 10) & (confidence <= (index + 1) / 10)
        if index == 0:
            selected |= confidence == 0
        if selected.any():
            ece += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    manual = {
        "accuracy": float(correctness.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(np.mean(tp / confusion.sum(axis=1))),
        "negative_log_likelihood": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(targets)), targets],
                    np.finfo(float).eps,
                    1,
                )
            ).mean()
        ),
        "expected_calibration_error_10_bins": float(ece),
    }
    if stored_metrics["n_samples"] != len(expected_rows):
        raise ValueError("Wrong sample count in metrics")
    for key, value in manual.items():
        if not np.isclose(value, stored_metrics[key], atol=1e-8, rtol=0):
            raise ValueError(f"Independent metric mismatch: {key}")
    return manual, confusion


def evaluate_four(checkpoints, configs, artifact_root, manifest, data_root, provenance):
    if set(checkpoints) != set(CONFIGURATIONS):
        raise ValueError("Exactly four checkpoints are required")
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("Manifest changed")
    expected_rows = validate_manifest(manifest).query("split == 'val'").reset_index(drop=True)
    if len(expected_rows) != 180 or expected_rows.class_name.value_counts().tolist() != [30] * 6:
        raise ValueError("Unexpected validation set")
    destination = Path(artifact_root) / "evaluation"
    if destination.exists():
        raise FileExistsError("Existing evaluation is preserved")
    fingerprints, comparison, error_tables = {}, [], []
    device = torch.device("cuda")
    for name in CONFIGURATIONS:
        summary = verify_training(checkpoints[name], configs[name])
        gc.collect()
        torch.cuda.empty_cache()
        seed_everything(SEED)
        model, checkpoint = load_checkpoint_model(str(checkpoints[name]), device=device)
        if count_parameters(model) != PARAMETERS[configs[name].model.name]:
            raise ValueError("Unexpected parameter count")
        clean_f1 = None
        for condition in CONDITIONS:
            print(f"Evaluating {name}: {condition}", flush=True)
            dataset = make_dataset(manifest, data_root, condition)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
            paths, targets, probabilities, fingerprint = predict_and_fingerprint(
                model, loader, device
            )
            if condition in fingerprints and fingerprints[condition] != fingerprint:
                raise ValueError("Models received different validation tensors")
            fingerprints[condition] = fingerprint
            metrics, per_class, confusion, _ = compute_metrics(targets, probabilities)
            table = prediction_table(paths, targets, probabilities)
            current = destination / name / condition
            current.mkdir(parents=True)
            table.to_csv(current / "predictions.csv", index=False)
            # Audit the serialized file, not just the in-memory dataframe.
            manual, audited_confusion = audit_predictions(
                pd.read_csv(current / "predictions.csv"),
                expected_rows,
                metrics,
            )
            if not np.array_equal(confusion, audited_confusion):
                raise ValueError("Confusion matrix mismatch")
            per_class.to_csv(current / "per_class_metrics.csv", index=False)
            confusion_pairs(confusion).to_csv(current / "confusion_pairs.csv", index=False)
            pd.DataFrame(confusion, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
                current / "confusion_matrix.csv",
            )
            plot_confusion_matrix(confusion, current / "confusion_matrix.png", normalised=False)
            metrics.update(
                {
                    "configuration": name,
                    "condition": condition,
                    "split": "val",
                    "seed": SEED,
                    "checkpoint_sha256": sha256(checkpoints[name]),
                    "input_tensor_sha256": fingerprint,
                    "runtime": runtime_metadata(device),
                    "manifest_sha256": MANIFEST_SHA256,
                    "test_evaluation_performed": False,
                }
            )
            row = {"configuration": name, "condition": condition, **manual}
            if condition == "clean":
                clean_f1 = manual["macro_f1"]
                if not np.isclose(
                    clean_f1,
                    checkpoint["validation_metrics"]["macro_f1"],
                    atol=1e-10,
                ):
                    raise ValueError("Fresh clean macro-F1 differs from the selected checkpoint")
                images, _, _ = next(iter(loader))
                latency = benchmark_latency(model, images, device, warmup=10, repetitions=50)
                metrics["latency"] = latency
                row.update(
                    {
                        "parameters": count_parameters(model),
                        "best_epoch": summary["best_epoch"],
                        "epochs_completed": summary["epochs_completed"],
                        "training_seconds": summary["elapsed_seconds"],
                        "training_origin": provenance[name]["training_origin"],
                        "median_latency_ms": latency["median_latency_ms"],
                        "p90_latency_ms": latency["p90_latency_ms"],
                    }
                )
            row["macro_f1_drop_from_clean"] = clean_f1 - manual["macro_f1"]
            write_json(current / "metrics.json", metrics)
            write_json(current / "independent_audit.json", {"status": "pass", **manual})
            errors = table.loc[~table.correct].copy()
            errors.insert(0, "condition", condition)
            errors.insert(0, "configuration", name)
            error_tables.append(errors)
            comparison.append(row)
        del model, checkpoint
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(Path(artifact_root) / "comparison_all_conditions.csv", index=False)
    comparison.query("condition == 'clean'").to_csv(
        Path(artifact_root) / "comparison_clean.csv",
        index=False,
    )
    effects = []
    indexed = comparison.set_index(["configuration", "condition"])
    for architecture in ("resnet18", "efficientnet_b0"):
        for condition in CONDITIONS:
            minimal = indexed.loc[(f"{architecture}_minimal", condition)]
            camera = indexed.loc[(f"{architecture}_camera", condition)]
            effects.append(
                {
                    "architecture": architecture,
                    "condition": condition,
                    "minimal_macro_f1": minimal.macro_f1,
                    "camera_macro_f1": camera.macro_f1,
                    "camera_minus_minimal_macro_f1": camera.macro_f1 - minimal.macro_f1,
                    "minimal_drop_from_clean": minimal.macro_f1_drop_from_clean,
                    "camera_drop_from_clean": camera.macro_f1_drop_from_clean,
                }
            )
    pd.DataFrame(effects).to_csv(Path(artifact_root) / "augmentation_effects.csv", index=False)
    errors = pd.concat(error_tables, ignore_index=True)
    errors.to_csv(Path(artifact_root) / "observed_validation_errors.csv", index=False)
    write_json(
        Path(artifact_root) / "validation_audit.json",
        {
            "status": "pass",
            "model_conditions": 40,
            "unique_validation_images": 180,
            "independent_metric_checks": 40,
            "input_tensor_sha256_by_condition": fingerprints,
            "all_models_received_identical_tensors": True,
            "test_evaluation_performed": False,
            "human_post_run_reviews_complete": False,
        },
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, corruption in zip(axes, PILOT_CORRUPTIONS, strict=True):
        for name in CONFIGURATIONS:
            values = comparison.loc[comparison.configuration == name].set_index("condition")
            axis.plot(
                range(4),
                [values.loc["clean", "macro_f1"]]
                + [values.loc[f"{corruption}_s{severity}", "macro_f1"] for severity in (1, 2, 3)],
                marker="o",
                label=name,
            )
        axis.set(title=corruption, xlabel="Severity (0 = clean)", ylim=(0, 1.02), xticks=range(4))
    axes[0].set_ylabel("Absolute validation macro-F1")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(Path(artifact_root) / "validation_pilot_macro_f1.png", dpi=160)
    plt.close(figure)
    return comparison


def preview_augmentation(manifest, data_root, destination):
    """Save training-only examples without looking at validation or test outcomes."""
    rows = validate_manifest(manifest).query("split == 'train'")
    selected = rows.groupby("class_name", sort=False).head(2).reset_index(drop=True)
    seed_everything(SEED)
    figure, axes = plt.subplots(len(selected), 4, figsize=(9, 2 * len(selected)))
    transform = CameraAwareAugmentation()
    for index, row in selected.iterrows():
        with Image.open(Path(data_root) / row.path) as source:
            original = source.convert("L")
        for column in range(4):
            image = original if column == 0 else transform(original.copy())
            array = np.asarray(image)
            if image.size != original.size or not np.isfinite(array).all():
                raise ValueError("Augmentation changed dimensions or produced invalid pixels")
            axes[index, column].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[index, column].axis("off")
            axes[index, column].set_title(
                f"{row.class_name}: original" if column == 0 else f"draw {column}",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(destination, dpi=110)
    plt.close(figure)
    write_json(
        Path(destination).with_suffix(".json"),
        {
            "split": "train",
            "paths": selected.path.tolist(),
            "seed": SEED,
            "augmentation_version": CameraAwareAugmentation.version,
            "human_visual_review_complete": False,
            "note": "Mechanical checks do not certify preservation of defect visibility.",
        },
    )


def preview_failures(errors_path, data_root, destination, maximum=12):
    errors = pd.read_csv(errors_path)
    # Deterministic shortlist, not a manufactured explanation or a count of unique images.
    selected = errors.drop_duplicates(["path", "condition"]).head(maximum)
    selected.to_csv(Path(destination).with_suffix(".csv"), index=False)
    if selected.empty:
        write_json(
            Path(destination).with_suffix(".json"),
            {
                "observed_errors": 0,
                "note": "No errors observed; no failure examples invented.",
            },
        )
        return
    figure, axes = plt.subplots(len(selected), 2, figsize=(7, 3 * len(selected)), squeeze=False)
    for index, row in enumerate(selected.itertuples(index=False)):
        with Image.open(Path(data_root) / row.path) as source:
            original = source.convert("L")
        transformed = original
        if row.condition != "clean":
            corruption, severity = row.condition.rsplit("_s", 1)
            transformed = apply_corruption(original, corruption, int(severity), sample_key=row.path)
        for axis, image in zip(axes[index], (original, transformed), strict=True):
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            axis.axis("off")
        axes[index, 0].set_title(f"{row.path}: {row.true_class}", fontsize=9)
        axes[index, 1].set_title(
            f"{row.configuration}\n{row.condition} -> {row.predicted_class}",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(destination, dpi=110)
    plt.close(figure)
