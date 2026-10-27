#!/usr/bin/env python3
"""Run the fixed four-checkpoint validation corruption pilot.

This is a development-set analysis, not test evaluation. It evaluates four
named checkpoints on the same 180 validation images under clean, darkened,
Gaussian-blurred, and shot/read-noise conditions.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from industrial_surface_defects.constants import CLASS_NAMES
from industrial_surface_defects.corruptions import (
    SEVERITY_VALUES,
    SHOT_READ_NOISE_SETTINGS,
    CorruptedManifestDataset,
)
from industrial_surface_defects.data import (
    ManifestDataset,
    validate_manifest,
    verify_manifest_files,
)
from industrial_surface_defects.evaluate import benchmark_latency
from industrial_surface_defects.metrics import compute_metrics, confusion_pairs, prediction_table
from industrial_surface_defects.models import count_parameters, load_checkpoint_model
from industrial_surface_defects.plots import plot_confusion_matrix
from industrial_surface_defects.reproducibility import (
    resolve_device,
    runtime_metadata,
    seed_everything,
    write_json,
)
from industrial_surface_defects.transforms import CameraAwareAugmentation, build_transform

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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path: str | Path) -> str:
    """Hash manifest text after normalising Git's platform line endings."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def parse_checkpoint_specs(specifications: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("Checkpoint must use CONFIGURATION=PATH syntax")
        name, raw_path = specification.split("=", 1)
        if name in checkpoints:
            raise ValueError(f"Duplicate checkpoint configuration: {name}")
        checkpoints[name] = Path(raw_path)
    missing = set(CONFIGURATIONS) - set(checkpoints)
    extra = set(checkpoints) - set(CONFIGURATIONS)
    if missing or extra:
        raise ValueError(
            f"Exactly four configurations are required; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    absent = [str(path) for path in checkpoints.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing checkpoints: {absent}")
    return checkpoints


def build_dataset(
    manifest: str | Path,
    data_root: str | Path,
    condition: str,
    *,
    image_size: int,
):
    if condition == "clean":
        return ManifestDataset(
            manifest,
            data_root,
            "val",
            build_transform(image_size, "minimal", train=False),
            return_path=True,
        )
    corruption, severity = condition.rsplit("_s", 1)
    return CorruptedManifestDataset(
        manifest,
        data_root,
        split="val",
        image_size=image_size,
        corruption=corruption,
        severity=int(severity),
    )


@torch.inference_mode()
def predict_and_fingerprint(model, loader, device: torch.device):
    paths: list[str] = []
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    digest = hashlib.sha256()
    for images, labels, batch_paths in loader:
        for image, label, path in zip(images, labels, batch_paths, strict=True):
            digest.update(str(path).encode() + b"\0" + str(int(label)).encode() + b"\0")
            digest.update(image.contiguous().numpy().tobytes())
        scores = model(images.to(device, non_blocking=True)).softmax(dim=1)
        paths.extend(str(path) for path in batch_paths)
        targets.extend(int(label) for label in labels)
        probabilities.append(scores.cpu().numpy())
    return paths, np.asarray(targets), np.concatenate(probabilities), digest.hexdigest()


def independent_accuracy_macro_f1(table: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    targets = table["true_index"].to_numpy(dtype=int)
    predictions = table["predicted_index"].to_numpy(dtype=int)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    class_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(len(CLASS_NAMES), dtype=float),
        where=denominator != 0,
    )
    return float((targets == predictions).mean()), float(class_f1.mean()), confusion


def save_augmentation_preview(
    manifest: str | Path,
    data_root: str | Path,
    destination: Path,
    *,
    seed: int,
) -> None:
    rows = validate_manifest(manifest).query("split == 'train'")
    selected = rows.groupby("class_name", sort=False).head(2).reset_index(drop=True)
    seed_everything(seed)
    transform = CameraAwareAugmentation()
    figure, axes = plt.subplots(len(selected), 4, figsize=(9, 2 * len(selected)))
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
        destination.with_suffix(".json"),
        {
            "split": "train",
            "paths": selected.path.tolist(),
            "seed": seed,
            "augmentation_version": CameraAwareAugmentation.version,
            "human_visual_review_complete": False,
            "note": "Mechanical checks do not certify preservation of defect visibility.",
        },
    )


def run_pilot(
    checkpoints: dict[str, Path],
    *,
    manifest: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    device_name: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    skip_latency: bool,
) -> Path:
    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.mkdir(parents=True)

    verify_manifest_files(manifest, data_root)
    manifest_rows = validate_manifest(manifest)
    validation_rows = manifest_rows.query("split == 'val'").reset_index(drop=True)
    if len(validation_rows) != 180:
        raise ValueError(f"Expected 180 validation rows, found {len(validation_rows)}")
    if set(validation_rows["class_name"].value_counts()) != {30}:
        raise ValueError("Expected exactly 30 validation images per class")
    current_manifest_hash = manifest_sha256(manifest)
    device = resolve_device(device_name)

    save_augmentation_preview(
        manifest,
        data_root,
        destination / "training_augmentation_preview.png",
        seed=seed,
    )
    protocol = {
        "name": "validation_camera_pilot_v1",
        "split": "val",
        "seed": seed,
        "conditions": list(CONDITIONS),
        "brightness_factors": list(SEVERITY_VALUES["brightness_dark"]),
        "gaussian_blur_radius_pixels_before_resize": list(
            SEVERITY_VALUES["gaussian_blur"]
        ),
        "shot_read_noise_peak_sigma": [list(item) for item in SHOT_READ_NOISE_SETTINGS],
        "corruption_seed": "SHA256(relative_path + NUL + corruption + NUL + severity)",
        "pilot_used_for_checkpoint_or_hyperparameter_selection": False,
        "test_evaluation_enabled": False,
    }
    write_json(destination / "validation_pilot_protocol.json", protocol)

    fingerprints: dict[str, str] = {}
    comparison: list[dict[str, object]] = []
    error_tables: list[pd.DataFrame] = []
    for configuration in CONFIGURATIONS:
        checkpoint_path = checkpoints[configuration]
        seed_everything(seed)
        model, checkpoint = load_checkpoint_model(str(checkpoint_path), device=device)
        config = checkpoint["config"]
        if int(config["experiment"]["seed"]) != seed:
            raise ValueError(f"Unexpected seed in {checkpoint_path}")
        if config["experiment"]["name"] != configuration:
            raise ValueError(f"Checkpoint configuration mismatch: {checkpoint_path}")
        expected_augmentation = "camera_aware" if configuration.endswith("_camera") else "minimal"
        if config["data"]["augmentation"] != expected_augmentation:
            raise ValueError(f"Checkpoint augmentation mismatch: {checkpoint_path}")
        checkpoint_manifest_hash = checkpoint.get("manifest_sha256")
        if checkpoint_manifest_hash != current_manifest_hash:
            raise ValueError(
                f"Checkpoint manifest hash differs for {configuration}: "
                f"{checkpoint_manifest_hash} != {current_manifest_hash}"
            )
        image_size = int(config["data"]["image_size"])
        clean_macro_f1: float | None = None
        for condition in CONDITIONS:
            print(f"Evaluating {configuration}: {condition}", flush=True)
            dataset = build_dataset(manifest, data_root, condition, image_size=image_size)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            paths, targets, probabilities, fingerprint = predict_and_fingerprint(
                model, loader, device
            )
            previous_fingerprint = fingerprints.setdefault(condition, fingerprint)
            if previous_fingerprint != fingerprint:
                raise ValueError(f"Models received different tensors for {condition}")

            metrics, per_class, raw_confusion, _normalised = compute_metrics(
                targets,
                probabilities,
                class_names=checkpoint["class_names"],
            )
            predictions = prediction_table(
                paths,
                targets,
                probabilities,
                class_names=checkpoint["class_names"],
            )
            accuracy, macro_f1, audited_confusion = independent_accuracy_macro_f1(predictions)
            if not np.array_equal(raw_confusion, audited_confusion):
                raise ValueError("Independent confusion matrix differs")
            if not np.isclose(accuracy, metrics["accuracy"], atol=1e-12, rtol=0):
                raise ValueError("Independent accuracy differs")
            if not np.isclose(macro_f1, metrics["macro_f1"], atol=1e-12, rtol=0):
                raise ValueError("Independent macro-F1 differs")

            current = destination / "evaluation" / configuration / condition
            current.mkdir(parents=True)
            predictions.to_csv(current / "predictions.csv", index=False)
            per_class.to_csv(current / "per_class_metrics.csv", index=False)
            confusion_pairs(raw_confusion).to_csv(current / "confusion_pairs.csv", index=False)
            pd.DataFrame(
                raw_confusion,
                index=CLASS_NAMES,
                columns=CLASS_NAMES,
            ).to_csv(current / "confusion_matrix.csv")
            plot_confusion_matrix(
                raw_confusion,
                current / "confusion_matrix.png",
                normalised=False,
            )

            metrics.update(
                {
                    "configuration": configuration,
                    "condition": condition,
                    "split": "val",
                    "seed": seed,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "input_tensor_sha256": fingerprint,
                    "manifest_sha256": current_manifest_hash,
                    "runtime": runtime_metadata(device),
                    "test_evaluation_performed": False,
                }
            )
            comparison_row: dict[str, object] = {
                "configuration": configuration,
                "condition": condition,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "negative_log_likelihood": metrics["negative_log_likelihood"],
                "expected_calibration_error_10_bins": metrics[
                    "expected_calibration_error_10_bins"
                ],
            }
            if condition == "clean":
                clean_macro_f1 = macro_f1
                checkpoint_f1 = checkpoint.get("validation_metrics", {}).get("macro_f1")
                if checkpoint_f1 is not None and not np.isclose(
                    macro_f1, checkpoint_f1, atol=1e-10, rtol=0
                ):
                    raise ValueError(f"Clean macro-F1 differs for {configuration}")
                comparison_row["parameters"] = count_parameters(model)
                if not skip_latency:
                    images, _, _ = next(iter(loader))
                    metrics["latency"] = benchmark_latency(model, images, device)
                    comparison_row.update(metrics["latency"])
            if clean_macro_f1 is None:
                raise AssertionError("Clean condition must be evaluated first")
            comparison_row["macro_f1_drop_from_clean"] = clean_macro_f1 - macro_f1
            write_json(current / "metrics.json", metrics)
            write_json(
                current / "independent_audit.json",
                {"accuracy": accuracy, "macro_f1": macro_f1, "status": "pass"},
            )
            errors = predictions.loc[~predictions["correct"]].copy()
            errors.insert(0, "condition", condition)
            errors.insert(0, "configuration", configuration)
            error_tables.append(errors)
            comparison.append(comparison_row)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(destination / "comparison_all_conditions.csv", index=False)
    comparison_frame.query("condition == 'clean'").to_csv(
        destination / "comparison_clean.csv", index=False
    )
    indexed = comparison_frame.set_index(["configuration", "condition"])
    effects: list[dict[str, object]] = []
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
    pd.DataFrame(effects).to_csv(destination / "augmentation_effects.csv", index=False)
    observed_errors = pd.concat(error_tables, ignore_index=True)
    observed_errors.to_csv(destination / "observed_validation_errors.csv", index=False)
    write_json(
        destination / "validation_audit.json",
        {
            "status": "pass",
            "model_conditions": len(comparison_frame),
            "unique_validation_images": len(validation_rows),
            "independent_metric_checks": len(comparison_frame),
            "input_tensor_sha256_by_condition": fingerprints,
            "all_models_received_identical_tensors": True,
            "test_evaluation_performed": False,
        },
    )

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, corruption in zip(axes, PILOT_CORRUPTIONS, strict=True):
        for configuration in CONFIGURATIONS:
            values = comparison_frame.loc[
                comparison_frame.configuration == configuration
            ].set_index("condition")
            axis.plot(
                range(4),
                [values.loc["clean", "macro_f1"]]
                + [
                    values.loc[f"{corruption}_s{severity}", "macro_f1"]
                    for severity in (1, 2, 3)
                ],
                marker="o",
                label=configuration,
            )
        axis.set(
            title=corruption,
            xlabel="Severity (0 = clean)",
            ylim=(0, 1.02),
            xticks=range(4),
        )
    axes[0].set_ylabel("Absolute validation macro-F1")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(destination / "validation_pilot_macro_f1.png", dpi=160)
    plt.close(figure)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat four times using CONFIGURATION=PATH",
    )
    parser.add_argument("--manifest", default="data/manifests/neu_cls_v2.csv")
    parser.add_argument("--data-root", default="data/raw/NEU-CLS")
    parser.add_argument("--output-root", default="validation_pilot")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-latency", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = run_pilot(
        parse_checkpoint_specs(args.checkpoint),
        manifest=args.manifest,
        data_root=args.data_root,
        output_root=args.output_root,
        device_name=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip_latency=args.skip_latency,
    )
    print(f"Validation pilot artifacts: {destination}")


if __name__ == "__main__":
    main()
