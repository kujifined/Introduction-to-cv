"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    seed: int = 42
    output_root: str = "outputs"


@dataclass(frozen=True)
class DataSettings:
    root: str
    manifest: str
    image_size: int = 224
    augmentation: str = "minimal"


@dataclass(frozen=True)
class ModelSettings:
    name: str
    pretrained: bool = True


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stopping_patience: int = 10
    warmup_epochs: int = 3
    min_learning_rate: float = 1e-6
    amp: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: ExperimentSettings
    data: DataSettings
    model: ModelSettings
    training: TrainingSettings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(
    path: str | Path,
    *,
    seed: int | None = None,
    device: str | None = None,
    pretrained: bool | None = None,
) -> ExperimentConfig:
    """Load and validate one experiment configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    if raw.get("schema_version") != 1:
        raise ValueError("Only configuration schema_version=1 is supported")

    experiment_raw = dict(raw.get("experiment", {}))
    model_raw = dict(raw.get("model", {}))
    training_raw = dict(raw.get("training", {}))
    if seed is not None:
        experiment_raw["seed"] = seed
    if device is not None:
        training_raw["device"] = device
    if pretrained is not None:
        model_raw["pretrained"] = pretrained

    config = ExperimentConfig(
        schema_version=1,
        experiment=ExperimentSettings(**experiment_raw),
        data=DataSettings(**raw.get("data", {})),
        model=ModelSettings(**model_raw),
        training=TrainingSettings(**training_raw),
    )
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    if config.model.name not in {"resnet18", "efficientnet_b0"}:
        raise ValueError(f"Unsupported model: {config.model.name}")
    if config.data.augmentation not in {"minimal", "camera_aware"}:
        raise ValueError(f"Unsupported augmentation: {config.data.augmentation}")
    if config.data.image_size < 32:
        raise ValueError("image_size must be at least 32 pixels")
    if config.training.epochs < 1 or config.training.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.training.warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative")
    if not 0.0 <= config.training.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
