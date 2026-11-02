"""Independent manifest and split checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import Audit, json_load, sha256_file


def audit_manifest(
    results_root: Path,
    failure_root: Path,
    expectations: dict[str, Any],
    audit: Audit,
) -> tuple[pd.DataFrame, list[str]]:
    manifest_path = results_root / "neu_cls_v2.csv"
    manifest = pd.read_csv(manifest_path)
    protocol = json_load(results_root / "experiment_protocol.json")
    audit.equal(
        "manifest",
        "neu_cls_v2.csv",
        "manifest SHA-256",
        protocol["manifest_sha256"],
        sha256_file(manifest_path),
    )
    audit.equal(
        "manifest",
        "neu_cls_v2.csv",
        "total rows",
        int(expectations["manifest_rows"]),
        len(manifest),
    )
    audit.equal(
        "manifest",
        "neu_cls_v2.csv",
        "unique paths",
        len(manifest),
        manifest["path"].nunique(),
    )
    split_counts = manifest["split"].value_counts().to_dict()
    for split, expected in expectations["splits"].items():
        audit.equal(
            "split",
            "neu_cls_v2.csv",
            f"{split} rows",
            int(expected),
            int(split_counts.get(split, 0)),
        )
    active = manifest.loc[manifest["split"] != "excluded"].copy()
    audit.equal(
        "split",
        "neu_cls_v2.csv",
        "active rows",
        int(expectations["active_rows"]),
        len(active),
    )
    audit.equal(
        "split",
        "neu_cls_v2.csv",
        "unique active paths",
        len(active),
        active["path"].nunique(),
    )
    audit.equal(
        "split",
        "neu_cls_v2.csv",
        "unique active SHA-256",
        len(active),
        active["sha256"].nunique(),
    )
    hash_split_max = active.groupby("sha256")["split"].nunique().max()
    group_split_max = active.groupby("group_id")["split"].nunique().max()
    audit.equal(
        "split",
        "neu_cls_v2.csv",
        "maximum splits per SHA-256",
        1,
        int(hash_split_max),
    )
    audit.equal(
        "split",
        "neu_cls_v2.csv",
        "maximum splits per group_id",
        1,
        int(group_split_max),
    )
    mapping_counts = manifest.groupby("class_index")["class_name"].nunique()
    audit.equal(
        "manifest",
        "neu_cls_v2.csv",
        "class-index mappings are one-to-one",
        True,
        bool((mapping_counts == 1).all()),
    )
    class_mapping = (
        manifest[["class_index", "class_name"]]
        .drop_duplicates()
        .sort_values("class_index")
    )
    audit.equal(
        "manifest",
        "neu_cls_v2.csv",
        "class count",
        int(expectations["class_count"]),
        len(class_mapping),
    )
    class_names = class_mapping["class_name"].astype(str).tolist()
    target_split = str(expectations["target_split"])
    target_rows = active.loc[active["split"] == target_split]
    target_support = target_rows.groupby("class_name").size()
    support = int(expectations["target_support_per_class"])
    audit.equal(
        "split",
        "neu_cls_v2.csv",
        f"{target_split} support is {support} for every class",
        True,
        bool((target_support == support).all()),
    )

    supplement_images = sorted((failure_root / "images").rglob("*.bmp"))
    verified = 0
    mismatch = 0
    for image_path in supplement_images:
        relative = image_path.relative_to(failure_root / "images").as_posix()
        row = manifest.loc[manifest["path"] == relative]
        if len(row) == 1 and sha256_file(image_path) == row.iloc[0]["sha256"]:
            verified += 1
        else:
            mismatch += 1
    expected_supplement = int(expectations["supplement_images"])
    audit.equal(
        "manifest",
        "clean_failure_materials/images",
        "supplement BMP files",
        expected_supplement,
        len(supplement_images),
    )
    audit.equal(
        "manifest",
        "clean_failure_materials/images",
        "verified BMP hashes",
        expected_supplement,
        verified,
    )
    audit.equal(
        "manifest",
        "clean_failure_materials/images",
        "BMP hash mismatches",
        0,
        mismatch,
    )
    return manifest, class_names

