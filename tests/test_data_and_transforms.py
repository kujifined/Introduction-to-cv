"""Manifest-backed dataset and preprocessing tests."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from industrial_surface_defects.data import (
    ManifestDataset,
    build_dataloader,
    validate_manifest,
    verify_manifest_files,
)
from industrial_surface_defects.transforms import build_transform, denormalize


def test_validate_manifest_accepts_generated_split(synthetic_manifest) -> None:
    actual = validate_manifest(synthetic_manifest.path)
    pd.testing.assert_frame_equal(actual, synthetic_manifest.frame)


def test_validate_manifest_rejects_missing_columns(tmp_path: Path, synthetic_manifest) -> None:
    broken = synthetic_manifest.frame.drop(columns="sha256")
    path = tmp_path / "missing-column.csv"
    broken.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        validate_manifest(path)


def test_validate_manifest_rejects_sha_leakage(tmp_path: Path, synthetic_manifest) -> None:
    broken = synthetic_manifest.frame.copy()
    train_index = broken.index[broken["split"] == "train"][0]
    test_index = broken.index[broken["split"] == "test"][0]
    broken.loc[test_index, "sha256"] = broken.loc[train_index, "sha256"]
    path = tmp_path / "leaking.csv"
    broken.to_csv(path, index=False)

    with pytest.raises(ValueError, match="more than one split"):
        validate_manifest(path)


def test_verify_manifest_files_accepts_untouched_dataset(synthetic_manifest) -> None:
    verify_manifest_files(synthetic_manifest.path, synthetic_manifest.root)


def test_verify_manifest_files_rejects_missing_image(synthetic_manifest) -> None:
    relative_path = synthetic_manifest.frame.iloc[0]["path"]
    (synthetic_manifest.root / relative_path).unlink()

    with pytest.raises(ValueError, match=r"file integrity check failed.*1 row"):
        verify_manifest_files(synthetic_manifest.path, synthetic_manifest.root)


def test_verify_manifest_files_rejects_sha_mismatch(synthetic_manifest) -> None:
    relative_path = synthetic_manifest.frame.iloc[0]["path"]
    (synthetic_manifest.root / relative_path).write_bytes(b"tampered image bytes")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest_files(synthetic_manifest.path, synthetic_manifest.root)


def test_manifest_dataset_loads_grayscale_as_normalized_rgb(synthetic_manifest) -> None:
    transform = build_transform(48, "minimal", train=False)
    dataset = ManifestDataset(
        synthetic_manifest.path,
        synthetic_manifest.root,
        "val",
        transform,
        return_path=True,
    )
    image, target, relative_path = dataset[0]

    assert len(dataset) == 6
    assert image.shape == (3, 48, 48)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert target == int(dataset.rows.iloc[0]["class_index"])
    assert relative_path == dataset.rows.iloc[0]["path"]
    assert (synthetic_manifest.root / relative_path).is_file()

    restored = denormalize(image)
    assert restored.min() >= 0
    assert restored.max() <= 1
    torch.testing.assert_close(restored[0], restored[1])
    torch.testing.assert_close(restored[1], restored[2])


@pytest.mark.parametrize("augmentation", ["minimal", "camera_aware"])
def test_evaluation_transform_is_deterministic(
    augmentation: str,
    synthetic_manifest,
) -> None:
    relative_path = synthetic_manifest.frame.iloc[0]["path"]
    with Image.open(synthetic_manifest.root / relative_path) as source:
        image = source.copy()
    transform = build_transform(40, augmentation, train=False)

    torch.testing.assert_close(transform(image), transform(image))


def test_camera_aware_training_transform_has_expected_shape(synthetic_manifest) -> None:
    relative_path = synthetic_manifest.frame.iloc[0]["path"]
    with Image.open(synthetic_manifest.root / relative_path) as source:
        image = source.copy()
    random.seed(9)
    torch.manual_seed(9)
    transformed = build_transform(40, "camera_aware", train=True)(image)

    assert transformed.shape == (3, 40, 40)
    assert torch.isfinite(transformed).all()


def test_transform_rejects_unknown_augmentation() -> None:
    with pytest.raises(ValueError, match="Unsupported augmentation"):
        build_transform(40, "unknown", train=True)


def test_seeded_train_dataloader_order_is_reproducible(synthetic_manifest) -> None:
    arguments = {
        "manifest_path": synthetic_manifest.path,
        "data_root": synthetic_manifest.root,
        "split": "train",
        "image_size": 32,
        "augmentation": "minimal",
        "batch_size": 8,
        "num_workers": 0,
        "seed": 77,
        "return_path": True,
    }

    def paths_from_one_epoch():
        paths: list[str] = []
        for _, _, batch_paths in build_dataloader(**arguments):
            paths.extend(batch_paths)
        return paths

    first = paths_from_one_epoch()
    second = paths_from_one_epoch()
    assert first == second
    assert len(first) == 42
    assert len(set(first)) == 42
