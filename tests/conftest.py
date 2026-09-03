"""Shared synthetic NEU-like data for fast, network-free tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from industrial_surface_defects.constants import CLASS_NAMES
from industrial_surface_defects.prepare import ImageRecord, assign_splits, scan_dataset


@dataclass(frozen=True)
class SyntheticDataset:
    root: Path
    records: list[ImageRecord]


@dataclass(frozen=True)
class SyntheticManifest:
    root: Path
    records: list[ImageRecord]
    frame: pd.DataFrame
    path: Path


FOLDER_ALIASES = dict(zip(CLASS_NAMES, ("Cr", "In", "Pa", "PS", "RS", "Sc"), strict=True))


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> SyntheticDataset:
    """Create ten unique, nonstandard-size grayscale PNGs for each class."""
    root = tmp_path / "NEU-CLS"
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / FOLDER_ALIASES[class_name]
        class_dir.mkdir(parents=True)
        for image_index in range(10):
            value = class_index * 35 + image_index
            pixels = np.full((36, 40), value, dtype=np.uint8)
            pixels[image_index % pixels.shape[0], :] = value + 50
            pixels[:, (class_index + image_index) % pixels.shape[1]] = value + 60
            Image.fromarray(pixels).save(class_dir / f"sample_{image_index:02d}.png")
    return SyntheticDataset(root=root, records=scan_dataset(root))


@pytest.fixture
def synthetic_manifest(
    tmp_path: Path,
    synthetic_dataset: SyntheticDataset,
) -> SyntheticManifest:
    frame = assign_splits(
        synthetic_dataset.records,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=42,
    )
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    return SyntheticManifest(
        root=synthetic_dataset.root,
        records=synthetic_dataset.records,
        frame=frame,
        path=path,
    )
