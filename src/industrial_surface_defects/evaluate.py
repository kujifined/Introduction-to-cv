"""Evaluate a frozen checkpoint and create auditable prediction artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import DataSettings
from .constants import CLASS_NAMES
from .data import build_dataloader, verify_manifest_files
from .metrics import compute_metrics, confusion_pairs, prediction_table
from .models import count_parameters, load_checkpoint_model
from .plots import plot_confusion_matrix, plot_reliability_diagram
from .reproducibility import resolve_device, runtime_metadata, seed_everything, write_json


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def predict(model, loader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    all_paths: list[str] = []
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for images, targets, paths in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_paths.extend(paths)
        all_targets.append(targets.numpy())
        all_probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return all_paths, np.concatenate(all_targets), np.concatenate(all_probabilities)


@torch.inference_mode()
def benchmark_latency(
    model,
    sample: torch.Tensor,
    device: torch.device,
    *,
    warmup: int = 10,
    repetitions: int = 50,
) -> dict[str, float | int | str]:
    model.eval()
    sample = sample[:1].to(device)
    for _ in range(warmup):
        model(sample)
    _synchronise(device)
    timings: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        model(sample)
        _synchronise(device)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "device": str(device),
        "batch_size": 1,
        "warmup_runs": warmup,
        "timed_runs": repetitions,
        "median_latency_ms": float(median(timings)),
        "p90_latency_ms": float(np.percentile(timings, 90)),
    }


def evaluate(
    checkpoint_path: str | Path,
    *,
    split: str,
    acknowledge_test_use: bool,
    device_name: str = "auto",
    data_root: str | None = None,
    manifest_path: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_dir: str | Path | None = None,
    skip_latency: bool = False,
) -> Path:
    if split == "test" and not acknowledge_test_use:
        raise PermissionError(
            "The test split is locked. Re-run with --acknowledge-test-use only after model and "
            "analysis choices are frozen."
        )
    if split not in {"val", "test"}:
        raise ValueError("Only val and test evaluation are supported")

    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint_model(str(checkpoint_path), device=device)
    config = checkpoint["config"]
    configured_data = DataSettings(**config["data"])
    root = data_root or configured_data.root
    manifest = manifest_path or configured_data.manifest
    verify_manifest_files(manifest, root)
    current_manifest_hash = _sha256(manifest)
    if current_manifest_hash != checkpoint["manifest_sha256"]:
        raise ValueError(
            "Manifest differs from the one used for training. Refusing an incomparable evaluation."
        )

    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    loader = build_dataloader(
        manifest_path=manifest,
        data_root=root,
        split=split,
        image_size=configured_data.image_size,
        augmentation="minimal",
        batch_size=batch_size or int(config["training"]["batch_size"]),
        num_workers=(
            int(config["training"]["num_workers"]) if num_workers is None else num_workers
        ),
        seed=seed,
        return_path=True,
    )

    paths, targets, probabilities = predict(model, loader, device)
    metrics, per_class, raw_confusion, normalised_confusion = compute_metrics(
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
    pairs = confusion_pairs(raw_confusion, class_names=checkpoint["class_names"])

    destination = (
        Path(output_dir)
        if output_dir is not None
        else checkpoint_path.parent / "evaluation" / split
    )
    if (destination / "metrics.json").exists():
        raise FileExistsError(
            f"Evaluation already exists at {destination}; choose --output-dir to preserve "
            "provenance."
        )
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / "predictions.csv", index=False)
    per_class.to_csv(destination / "per_class_metrics.csv", index=False)
    pairs.to_csv(destination / "confusion_pairs.csv", index=False)
    pd.DataFrame(raw_confusion, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        destination / "confusion_matrix.csv"
    )
    pd.DataFrame(normalised_confusion, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        destination / "confusion_matrix_normalised.csv"
    )
    plot_confusion_matrix(raw_confusion, destination / "confusion_matrix.png", normalised=False)
    plot_confusion_matrix(
        normalised_confusion,
        destination / "confusion_matrix_normalised.png",
        normalised=True,
    )
    plot_reliability_diagram(targets, probabilities, destination / "reliability_diagram.png")

    metrics.update(
        {
            "split": split,
            "model_name": config["model"]["name"],
            "augmentation_training": config["data"]["augmentation"],
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "model_parameters": count_parameters(model),
            "manifest_sha256": current_manifest_hash,
            "test_use_acknowledged": bool(split == "test" and acknowledge_test_use),
            "runtime": runtime_metadata(device),
        }
    )
    if not skip_latency:
        first_images, _, _ = next(iter(loader))
        metrics["latency"] = benchmark_latency(model, first_images, device)
    write_json(destination / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--acknowledge-test-use", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-latency", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(
        args.checkpoint,
        split=args.split,
        acknowledge_test_use=args.acknowledge_test_use,
        device_name=args.device,
        data_root=args.data_root,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        skip_latency=args.skip_latency,
    )


if __name__ == "__main__":
    main()
