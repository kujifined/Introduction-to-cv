"""Train one preregistered experiment without accessing the locked test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from .config import ExperimentConfig, load_config
from .constants import CLASS_NAMES
from .data import build_dataloader, verify_manifest_files
from .metrics import compute_metrics
from .models import build_model, count_parameters
from .reproducibility import resolve_device, runtime_metadata, seed_everything, write_json


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = batch
    return (
        images.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images, targets = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.size(0)
        running_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum())
        count += batch_size
    return running_loss / count, correct / count


@torch.inference_mode()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    running_loss = 0.0
    count = 0
    targets_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    for batch in tqdm(loader, desc="val", leave=False):
        images, targets = _move_batch(batch, device)
        logits = model(images)
        loss = criterion(logits, targets)
        probabilities = logits.softmax(dim=1)
        batch_size = targets.size(0)
        running_loss += float(loss) * batch_size
        count += batch_size
        targets_all.append(targets.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    metrics, _, _, _ = compute_metrics(
        np.concatenate(targets_all),
        np.concatenate(probabilities_all),
    )
    return running_loss / count, metrics


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _plot_history(history: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train accuracy")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="validation macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Learning curves", ylim=(0, 1.02))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train(config: ExperimentConfig, *, run_suffix: str | None = None) -> Path:
    """Train a model and return the best checkpoint path."""
    verify_manifest_files(config.data.manifest, config.data.root)
    seed_everything(config.experiment.seed)
    device = resolve_device(config.training.device)
    run_name = f"seed_{config.experiment.seed}"
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"
    run_dir = Path(config.experiment.output_root) / config.experiment.name / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use a new seed or --run-suffix."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(
        manifest_path=config.data.manifest,
        data_root=config.data.root,
        split="train",
        image_size=config.data.image_size,
        augmentation=config.data.augmentation,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.experiment.seed,
    )
    val_loader = build_dataloader(
        manifest_path=config.data.manifest,
        data_root=config.data.root,
        split="val",
        image_size=config.data.image_size,
        augmentation="minimal",
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.experiment.seed,
    )

    model = build_model(config.model.name, len(CLASS_NAMES), pretrained=config.model.pretrained)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    warmup_epochs = min(config.training.warmup_epochs, max(0, config.training.epochs - 1))
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.training.epochs - warmup_epochs),
        eta_min=config.training.min_learning_rate,
    )
    if warmup_epochs:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.10, end_factor=1.0, total_iters=warmup_epochs),
                cosine,
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine
    train_criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    evaluation_criterion = nn.CrossEntropyLoss()
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    resolved_config_path = run_dir / "config.resolved.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False)
    write_json(run_dir / "runtime.json", runtime_metadata(device))

    history_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = run_dir / "best.pt"
    started = time.perf_counter()

    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            train_criterion,
            device,
            scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, val_metrics = evaluate_epoch(model, val_loader, evaluation_criterion, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "class_names": list(CLASS_NAMES),
                    "num_classes": len(CLASS_NAMES),
                    "best_epoch": best_epoch,
                    "validation_metrics": val_metrics,
                    "manifest_sha256": _sha256(config.data.manifest),
                },
            )
        else:
            epochs_without_improvement += 1

        scheduler.step()
        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    history = pd.DataFrame(history_rows)
    _plot_history(history, run_dir / "learning_curves.png")
    summary = {
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "epochs_completed": len(history_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "model_parameters": count_parameters(model),
        "checkpoint": str(checkpoint_path),
        "test_split_accessed": False,
    }
    write_json(run_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a YAML experiment configuration")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, help="Override epochs (useful for smoke tests)")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--num-workers", type=int, help="Override data-loader workers")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--run-suffix",
        help="Append a label instead of overwriting an existing run",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(
        args.config,
        seed=args.seed,
        device=args.device,
        pretrained=False if args.no_pretrained else None,
    )
    training = config.training
    if args.epochs is not None:
        training = replace(training, epochs=args.epochs)
    if args.batch_size is not None:
        training = replace(training, batch_size=args.batch_size)
    if args.num_workers is not None:
        training = replace(training, num_workers=args.num_workers)
    config = replace(config, training=training)
    train(config, run_suffix=args.run_suffix)


if __name__ == "__main__":
    main()
