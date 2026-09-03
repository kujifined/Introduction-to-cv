"""Deterministic synthetic acquisition-corruption tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from industrial_surface_defects.corruptions import (
    CORRUPTION_NAMES,
    CorruptedManifestDataset,
    apply_corruption,
)


@pytest.fixture
def gradient_image() -> Image.Image:
    row = np.linspace(0, 255, 64, dtype=np.uint8)
    pixels = np.tile(row, (64, 1))
    pixels[16:48, 24:40] = 190
    return Image.fromarray(pixels)


@pytest.mark.parametrize("corruption", CORRUPTION_NAMES)
def test_corruptions_are_deterministic_per_sample(
    corruption: str,
    gradient_image: Image.Image,
) -> None:
    first = apply_corruption(gradient_image, corruption, 2, sample_key="Cr/sample.png")
    second = apply_corruption(gradient_image, corruption, 2, sample_key="Cr/sample.png")

    assert first.mode == "L"
    assert first.size == gradient_image.size
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_noise_seed_depends_on_sample_key(gradient_image: Image.Image) -> None:
    first = apply_corruption(
        gradient_image,
        "shot_read_noise",
        3,
        sample_key="Cr/first.png",
    )
    second = apply_corruption(
        gradient_image,
        "shot_read_noise",
        3,
        sample_key="Cr/second.png",
    )

    assert not np.array_equal(np.asarray(first), np.asarray(second))


@pytest.mark.parametrize(
    ("corruption", "severity", "message"),
    [("unknown", 1, "Unknown corruption"), ("gamma", 0, "severity must be")],
)
def test_corruption_rejects_invalid_requests(
    corruption: str,
    severity: int,
    message: str,
    gradient_image: Image.Image,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_corruption(gradient_image, corruption, severity, sample_key="sample")


def test_corrupted_manifest_dataset_is_repeatable(synthetic_manifest) -> None:
    dataset = CorruptedManifestDataset(
        synthetic_manifest.path,
        synthetic_manifest.root,
        split="test",
        image_size=32,
        corruption="shot_read_noise",
        severity=2,
    )
    first_image, first_target, first_path = dataset[0]
    second_image, second_target, second_path = dataset[0]

    torch.testing.assert_close(first_image, second_image)
    assert (first_target, first_path) == (second_target, second_path)
    assert first_image.shape == (3, 32, 32)
    assert len(dataset) == 12
