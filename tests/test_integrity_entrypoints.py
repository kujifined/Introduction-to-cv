"""Verify that every data-consuming entry point performs one early byte audit."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import industrial_surface_defects.evaluate as evaluate_module
import industrial_surface_defects.robustness as robustness_module
import industrial_surface_defects.train as train_module
from industrial_surface_defects.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IntegrityProbeReached(RuntimeError):
    """Stop a pipeline immediately after recording its integrity call."""


def _checkpoint_metadata() -> dict:
    return {
        "config": {
            "experiment": {"name": "probe", "seed": 42, "output_root": "outputs"},
            "data": {
                "root": "configured-root",
                "manifest": "configured-manifest.csv",
                "image_size": 32,
                "augmentation": "minimal",
            },
            "model": {"name": "resnet18", "pretrained": False},
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "label_smoothing": 0.0,
                "early_stopping_patience": 1,
                "warmup_epochs": 0,
                "min_learning_rate": 1e-6,
                "amp": False,
                "device": "cpu",
            },
        }
    }


def test_train_calls_integrity_check_once_before_training(monkeypatch) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "resnet18_minimal.yaml")
    calls: list[tuple[str, str]] = []

    def probe(manifest: str, data_root: str) -> None:
        calls.append((manifest, data_root))
        raise IntegrityProbeReached

    monkeypatch.setattr(train_module, "verify_manifest_files", probe)

    with pytest.raises(IntegrityProbeReached):
        train_module.train(config)
    assert calls == [(config.data.manifest, config.data.root)]


def test_evaluate_calls_integrity_check_once_before_dataloader(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def probe(manifest: str, data_root: str) -> None:
        calls.append((manifest, data_root))
        raise IntegrityProbeReached

    monkeypatch.setattr(evaluate_module, "resolve_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(
        evaluate_module,
        "load_checkpoint_model",
        lambda *_args, **_kwargs: (object(), _checkpoint_metadata()),
    )
    monkeypatch.setattr(evaluate_module, "verify_manifest_files", probe)

    with pytest.raises(IntegrityProbeReached):
        evaluate_module.evaluate(
            "unused.pt",
            split="val",
            acknowledge_test_use=False,
            data_root="override-root",
            manifest_path="override-manifest.csv",
        )
    assert calls == [("override-manifest.csv", "override-root")]


def test_robustness_calls_integrity_check_once_before_dataloader(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def probe(manifest: str, data_root: str) -> None:
        calls.append((manifest, data_root))
        raise IntegrityProbeReached

    monkeypatch.setattr(
        robustness_module,
        "resolve_device",
        lambda _name: torch.device("cpu"),
    )
    monkeypatch.setattr(
        robustness_module,
        "load_checkpoint_model",
        lambda *_args, **_kwargs: (object(), _checkpoint_metadata()),
    )
    monkeypatch.setattr(robustness_module, "verify_manifest_files", probe)

    with pytest.raises(IntegrityProbeReached):
        robustness_module.evaluate_robustness(
            "unused.pt",
            acknowledge_test_use=True,
            data_root="override-root",
            manifest_path="override-manifest.csv",
        )
    assert calls == [("override-manifest.csv", "override-root")]
