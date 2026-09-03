"""Configuration loading and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from industrial_surface_defects.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "model_name", "augmentation"),
    [
        ("resnet18_minimal.yaml", "resnet18", "minimal"),
        ("resnet18_camera.yaml", "resnet18", "camera_aware"),
        ("efficientnet_b0_minimal.yaml", "efficientnet_b0", "minimal"),
        ("efficientnet_b0_camera.yaml", "efficientnet_b0", "camera_aware"),
    ],
)
def test_checked_in_configs_load(
    filename: str,
    model_name: str,
    augmentation: str,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / filename)

    assert config.schema_version == 1
    assert config.model.name == model_name
    assert config.data.augmentation == augmentation
    assert config.data.image_size == 224
    assert config.training.epochs > 0


def test_load_config_applies_explicit_runtime_overrides() -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "resnet18_minimal.yaml",
        seed=7,
        device="cpu",
        pretrained=False,
    )

    assert config.experiment.seed == 7
    assert config.training.device == "cpu"
    assert config.model.pretrained is False
    assert config.to_dict()["experiment"]["seed"] == 7


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (("model", "name"), "unknown", "Unsupported model"),
        (("data", "augmentation"), "destructive", "Unsupported augmentation"),
        (("data", "image_size"), 16, "image_size must be at least 32"),
        (("training", "epochs"), 0, "epochs and batch_size must be positive"),
        (("training", "warmup_epochs"), -1, "warmup_epochs cannot be negative"),
        (("training", "label_smoothing"), 1.0, "label_smoothing must be"),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path,
    field: tuple[str, str],
    value: object,
    message: str,
) -> None:
    raw = load_config(PROJECT_ROOT / "configs" / "resnet18_minimal.yaml").to_dict()
    raw[field[0]][field[1]] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_load_config_rejects_unknown_schema(tmp_path: Path) -> None:
    raw = load_config(PROJECT_ROOT / "configs" / "resnet18_minimal.yaml").to_dict()
    raw["schema_version"] = 2
    path = tmp_path / "future.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version=1"):
        load_config(path)
