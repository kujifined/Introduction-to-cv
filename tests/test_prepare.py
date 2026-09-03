"""Dataset discovery, audit, split, and leakage tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from industrial_surface_defects.constants import CLASS_NAMES, CLASS_TO_INDEX
from industrial_surface_defects.prepare import (
    ImageRecord,
    assign_splits,
    infer_class,
    validate_neu_cls,
)


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("Cr", "crazing"),
        ("In", "inclusion"),
        ("Pa", "patches"),
        ("pitted surface", "pitted_surface"),
        ("rolled-in-scale", "rolled_in_scale"),
        ("Sc", "scratches"),
    ],
)
def test_infer_class_accepts_neu_folder_aliases(
    tmp_path: Path,
    folder: str,
    expected: str,
) -> None:
    root = tmp_path / "dataset"
    assert infer_class(root / folder / "001.png", root) == expected


def test_infer_class_rejects_an_unknown_layout(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="Cannot infer"):
        infer_class(root / "mystery" / "001.png", root)


def test_scan_finds_all_six_classes(synthetic_dataset) -> None:
    counts = pd.Series(record.class_name for record in synthetic_dataset.records).value_counts()

    assert len(synthetic_dataset.records) == 60
    assert set(counts.index) == set(CLASS_NAMES)
    assert set(counts.to_list()) == {10}
    assert all(record.mode == "L" for record in synthetic_dataset.records)


def _canonical_records() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_name in CLASS_NAMES:
        for index in range(300):
            relative_path = f"{class_name}/{index:03d}.bmp"
            digest = hashlib.sha256(relative_path.encode()).hexdigest()
            records.append(
                ImageRecord(
                    path=relative_path,
                    class_name=class_name,
                    class_index=CLASS_TO_INDEX[class_name],
                    width=200,
                    height=200,
                    mode="L",
                    file_size=40_000,
                    sha256=digest,
                    dhash=digest[:16],
                )
            )
    return records


def test_strict_audit_accepts_canonical_shape_and_counts() -> None:
    audit = validate_neu_cls(_canonical_records(), strict=True)

    assert audit["image_count"] == 1800
    assert audit["class_counts"] == {name: 300 for name in CLASS_NAMES}
    assert audit["size_counts"] == {"200x200": 1800}
    assert audit["mode_counts"] == {"L": 1800}
    assert audit["strict_neu_cls"] is True
    assert len(audit["dataset_fingerprint_sha256"]) == 64


def test_nonstandard_audit_accepts_small_balanced_fixture(synthetic_dataset) -> None:
    audit = validate_neu_cls(synthetic_dataset.records, strict=False)

    assert audit["image_count"] == 60
    assert audit["class_counts"] == {name: 10 for name in CLASS_NAMES}
    assert audit["strict_neu_cls"] is False

    with pytest.raises(ValueError, match="expected 300 images"):
        validate_neu_cls(synthetic_dataset.records, strict=True)


def test_audit_rejects_cross_class_byte_duplicates(synthetic_dataset) -> None:
    records = list(synthetic_dataset.records)
    records[-1] = replace(records[-1], sha256=records[0].sha256)

    with pytest.raises(ValueError, match="conflicting labels"):
        validate_neu_cls(records, strict=False)


def test_split_is_deterministic_stratified_70_10_20(synthetic_dataset) -> None:
    arguments = {
        "train_ratio": 0.70,
        "val_ratio": 0.10,
        "test_ratio": 0.20,
        "seed": 123,
    }
    first = assign_splits(synthetic_dataset.records, **arguments)
    second = assign_splits(synthetic_dataset.records, **arguments)
    pd.testing.assert_frame_equal(first, second)

    counts = first.groupby(["class_name", "split"], observed=True).size()
    for class_name in CLASS_NAMES:
        assert counts[class_name, "train"] == 7
        assert counts[class_name, "val"] == 1
        assert counts[class_name, "test"] == 2
    assert first.groupby("sha256", observed=True)["split"].nunique().max() == 1


def test_duplicate_sha_group_never_crosses_splits(synthetic_dataset) -> None:
    duplicate = replace(synthetic_dataset.records[0], path="Cr/copied_sample.png")
    manifest = assign_splits(
        [*synthetic_dataset.records, duplicate],
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=42,
    )
    duplicate_rows = manifest.loc[manifest["sha256"] == duplicate.sha256]

    assert len(duplicate_rows) == 2
    assert duplicate_rows["split"].nunique() == 1
    assert manifest.groupby("sha256", observed=True)["split"].nunique().max() == 1


@pytest.mark.parametrize(
    ("ratios", "message"),
    [((0.7, 0.1, 0.3), "sum to 1"), ((0.8, 0.2, 0.0), "positive")],
)
def test_split_rejects_invalid_ratios(synthetic_dataset, ratios, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        assign_splits(
            synthetic_dataset.records,
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
            seed=42,
        )
