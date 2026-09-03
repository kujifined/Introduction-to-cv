"""Audit NEU-CLS and create a locked, duplicate-aware split manifest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .constants import CLASS_ALIASES, CLASS_NAMES, CLASS_TO_INDEX, IMAGE_EXTENSIONS
from .reproducibility import write_json


@dataclass(frozen=True)
class ImageRecord:
    path: str
    class_name: str
    class_index: int
    width: int
    height: int
    mode: str
    sha256: str
    dhash: str
    file_size: int = 0


def _normalise_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", token.strip().lower()).strip("_")


def infer_class(path: Path, data_root: Path) -> str:
    candidates: list[str] = []
    try:
        relative = path.relative_to(data_root)
    except ValueError:
        relative = path
    candidates.extend(part for part in reversed(relative.parts[:-1]))
    candidates.extend(
        [
            path.stem,
            re.split(r"[_\-\s]", path.stem, maxsplit=1)[0],
            re.match(r"[A-Za-z]+", path.stem).group(0) if re.match(r"[A-Za-z]+", path.stem) else "",
        ]
    )
    for candidate in candidates:
        normalised = _normalise_token(candidate)
        if normalised in CLASS_ALIASES:
            return CLASS_ALIASES[normalised]
    raise ValueError(f"Cannot infer an NEU-CLS class from {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image, hash_size: int = 8, high_frequency_factor: int = 4) -> str:
    """Return a 64-bit DCT perceptual hash for near-duplicate triage."""
    size = hash_size * high_frequency_factor
    pixels = np.asarray(image.convert("L").resize((size, size)), dtype=np.float64)
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    basis = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    basis[0] *= np.sqrt(1 / size)
    basis[1:] *= np.sqrt(2 / size)
    coefficients = basis @ pixels @ basis.T
    low_frequency = coefficients[:hash_size, :hash_size]
    median = np.median(low_frequency.ravel()[1:])
    bits = low_frequency > median
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def scan_dataset(data_root: str | Path) -> list[ImageRecord]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found below {root}")

    records: list[ImageRecord] = []
    failures: list[str] = []
    for path in paths:
        try:
            class_name = infer_class(path, root)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                dhash = perceptual_hash(image)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            failures.append(f"{path}: {error}")
            continue
        records.append(
            ImageRecord(
                path=path.relative_to(root).as_posix(),
                class_name=class_name,
                class_index=CLASS_TO_INDEX[class_name],
                width=width,
                height=height,
                mode=mode,
                file_size=path.stat().st_size,
                sha256=sha256_file(path),
                dhash=dhash,
            )
        )
    if failures:
        preview = "\n".join(failures[:10])
        suffix = "" if len(failures) <= 10 else f"\n... and {len(failures) - 10} more"
        raise ValueError(f"Failed to audit {len(failures)} image(s):\n{preview}{suffix}")
    return records


def validate_neu_cls(records: list[ImageRecord], *, strict: bool) -> dict:
    class_counts = Counter(record.class_name for record in records)
    unexpected_classes = set(class_counts) - set(CLASS_NAMES)
    if unexpected_classes:
        raise ValueError(f"Unexpected classes: {sorted(unexpected_classes)}")

    digest_to_classes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        digest_to_classes[record.sha256].add(record.class_name)
    cross_class_duplicates = {
        digest: sorted(classes) for digest, classes in digest_to_classes.items() if len(classes) > 1
    }
    if cross_class_duplicates:
        raise ValueError(
            "Byte-identical images have conflicting labels; resolve them before splitting: "
            f"{cross_class_duplicates}"
        )

    size_counts = Counter((record.width, record.height) for record in records)
    mode_counts = Counter(record.mode for record in records)
    duplicate_group_sizes = Counter(record.sha256 for record in records)
    duplicate_images = sum(size - 1 for size in duplicate_group_sizes.values() if size > 1)

    if strict:
        expected_counts = {class_name: 300 for class_name in CLASS_NAMES}
        if dict(class_counts) != expected_counts:
            raise ValueError(
                "Strict NEU-CLS audit expected 300 images for each of six classes; "
                f"found {dict(sorted(class_counts.items()))}"
            )
        if set(size_counts) != {(200, 200)}:
            raise ValueError(f"Strict NEU-CLS audit expected only 200x200 images: {size_counts}")
        if any(mode not in {"1", "L", "P"} for mode in mode_counts):
            raise ValueError(
                f"Strict NEU-CLS audit expected grayscale-compatible images: {mode_counts}"
            )

    fingerprint = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.path):
        fingerprint.update(f"{record.path}\0{record.class_name}\0{record.sha256}\n".encode())

    return {
        "image_count": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "size_counts": {
            f"{width}x{height}": count for (width, height), count in size_counts.items()
        },
        "mode_counts": dict(sorted(mode_counts.items())),
        "exact_duplicate_extra_images": duplicate_images,
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "strict_neu_cls": strict,
    }


def quarantine_exact_duplicates(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Keep one canonical file per byte identity and explicitly quarantine the rest."""
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)
    active: list[ImageRecord] = []
    quarantined: list[ImageRecord] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda record: record.path)
        active.append(ordered[0])
        quarantined.extend(ordered[1:])
    return sorted(active, key=lambda record: record.path), sorted(
        quarantined, key=lambda record: record.path
    )


def _integer_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    targets = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(targets.values())
    order = sorted(ratios, key=lambda name: (raw[name] - targets[name], ratios[name]), reverse=True)
    for name in order[:remainder]:
        targets[name] += 1
    return targets


def assign_splits(
    records: list[ImageRecord],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_ids: dict[str, str] | None = None,
) -> pd.DataFrame:
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(ratio <= 0 for ratio in ratios.values()) or not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("train/val/test ratios must be positive and sum to 1")

    class_totals = Counter(record.class_name for record in records)
    for class_name in CLASS_NAMES:
        if not class_totals[class_name]:
            raise ValueError(f"Class {class_name!r} has no images")

    group_for_record = {
        record.path: (
            group_ids[record.path]
            if group_ids is not None and record.path in group_ids
            else f"sha256:{record.sha256}"
        )
        for record in records
    }
    groups_by_id: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups_by_id[group_for_record[record.path]].append(record)
    groups = list(groups_by_id.items())
    rng = random.Random(f"neu-cls:{seed}:global-groups")
    rng.shuffle(groups)
    groups.sort(
        key=lambda item: (len(item[1]), len({record.class_name for record in item[1]})),
        reverse=True,
    )

    targets = {
        class_name: _integer_targets(class_totals[class_name], ratios) for class_name in CLASS_NAMES
    }
    counts = {class_name: {split: 0 for split in ratios} for class_name in CLASS_NAMES}
    assigned_groups: dict[str, str] = {}
    split_order = list(ratios)
    for group_id, group in groups:
        contribution = Counter(record.class_name for record in group)

        def objective(
            split: str,
            group_contribution: Counter = contribution,
        ) -> tuple[float, int, int]:
            squared_relative_error = 0.0
            for class_name in CLASS_NAMES:
                for candidate_split in ratios:
                    updated = counts[class_name][candidate_split]
                    if candidate_split == split:
                        updated += group_contribution[class_name]
                    target = targets[class_name][candidate_split]
                    squared_relative_error += ((updated - target) / max(1, target)) ** 2
            remaining_capacity = sum(
                targets[class_name][split] - counts[class_name][split] for class_name in CLASS_NAMES
            )
            return squared_relative_error, -remaining_capacity, split_order.index(split)

        assigned_split = min(ratios, key=objective)
        assigned_groups[group_id] = assigned_split
        for class_name, increment in contribution.items():
            counts[class_name][assigned_split] += increment

    assigned_rows: list[dict] = []
    for record in records:
        group_id = group_for_record[record.path]
        row = asdict(record)
        row["relative_path"] = record.path
        row["label"] = record.class_name
        row["perceptual_hash"] = record.dhash
        row["group_id"] = group_id
        row["status"] = "ok"
        row["split"] = assigned_groups[group_id]
        assigned_rows.append(row)

    manifest = pd.DataFrame(assigned_rows)
    split_order = pd.Categorical(manifest["split"], ["train", "val", "test"], ordered=True)
    manifest = (
        manifest.assign(split=split_order)
        .sort_values(["split", "class_index", "path"])
        .reset_index(drop=True)
    )
    _assert_no_leakage(manifest)
    manifest["split"] = manifest["split"].astype(str)
    return manifest


def _assert_no_leakage(manifest: pd.DataFrame) -> None:
    if manifest["path"].duplicated().any():
        raise AssertionError("A path appears more than once in the manifest")
    splits_per_digest = manifest.groupby("sha256", observed=True)["split"].nunique()
    leaking = splits_per_digest[splits_per_digest > 1]
    if not leaking.empty:
        raise AssertionError(f"Exact duplicate groups cross splits: {leaking.index.tolist()}")
    if "group_id" in manifest:
        splits_per_group = manifest.groupby("group_id", observed=True)["split"].nunique()
        leaking_groups = splits_per_group[splits_per_group > 1]
        if not leaking_groups.empty:
            raise AssertionError(f"Source groups cross splits: {leaking_groups.index.tolist()}")


def read_group_map(path: str | Path | None, records: list[ImageRecord]) -> dict[str, str] | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"path", "group_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Group map is missing columns: {sorted(missing)}")
    if frame["path"].duplicated().any() or frame["group_id"].isna().any():
        raise ValueError("Group map paths must be unique and every group_id must be present")
    known_paths = {record.path for record in records}
    unknown_paths = set(frame["path"]) - known_paths
    if unknown_paths:
        raise ValueError(f"Group map contains unknown paths: {sorted(unknown_paths)[:10]}")
    mapping = dict(
        zip(
            frame["path"].astype(str),
            frame["group_id"].astype(str),
            strict=True,
        )
    )
    return mapping


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def near_duplicate_candidates(manifest: pd.DataFrame, threshold: int) -> pd.DataFrame:
    candidates: list[dict] = []
    active = manifest.loc[manifest["status"] == "ok"]
    for class_name, rows in active.groupby("class_name", observed=True):
        values = rows[["path", "split", "sha256", "dhash"]].to_dict("records")
        for left, right in itertools.combinations(values, 2):
            if left["sha256"] == right["sha256"]:
                continue
            distance = hamming_distance(left["dhash"], right["dhash"])
            if distance <= threshold:
                candidates.append(
                    {
                        "class_name": class_name,
                        "left_path": left["path"],
                        "right_path": right["path"],
                        "left_split": left["split"],
                        "right_split": right["split"],
                        "perceptual_hash_distance": distance,
                        "cross_split": left["split"] != right["split"],
                    }
                )
    columns = [
        "class_name",
        "left_path",
        "right_path",
        "left_split",
        "right_split",
        "perceptual_hash_distance",
        "cross_split",
        "review_status",
        "review_notes",
    ]
    frame = pd.DataFrame(candidates)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["review_status"] = ""
    frame["review_notes"] = ""
    return frame[columns].sort_values(
        ["cross_split", "perceptual_hash_distance"], ascending=[False, True]
    )


def write_manifest(path: str | Path, manifest: pd.DataFrame) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace frozen manifest: {destination}. Use a new versioned filename."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/NEU-CLS")
    parser.add_argument("--manifest", default="data/manifests/neu_cls_v2.csv")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=8,
        help="64-bit pHash Hamming threshold; 8 yields a small review set on official NEU-CLS.",
    )
    parser.add_argument(
        "--group-map",
        help="Optional reviewed CSV with path,group_id columns; related images stay together.",
    )
    parser.add_argument(
        "--allow-nonstandard",
        action="store_true",
        help="Permit subsets or alternate dimensions; leakage checks still run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = scan_dataset(args.data_root)
    audit = validate_neu_cls(records, strict=not args.allow_nonstandard)
    active_records, quarantined_records = quarantine_exact_duplicates(records)
    group_ids = read_group_map(args.group_map, records)
    manifest = assign_splits(
        active_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_ids=group_ids,
    )
    if quarantined_records:
        quarantine_rows = []
        for record in quarantined_records:
            row = asdict(record)
            row.update(
                {
                    "relative_path": record.path,
                    "label": record.class_name,
                    "perceptual_hash": record.dhash,
                    "group_id": f"sha256:{record.sha256}",
                    "status": "duplicate_quarantined",
                    "split": "excluded",
                }
            )
            quarantine_rows.append(row)
        manifest = pd.concat([manifest, pd.DataFrame(quarantine_rows)], ignore_index=True)
    manifest_path = Path(args.manifest)
    candidates = near_duplicate_candidates(manifest, args.near_duplicate_threshold)
    candidate_path = manifest_path.with_name(f"{manifest_path.stem}_near_duplicates.csv")
    audit_path = manifest_path.with_suffix(".audit.json")
    existing = [path for path in (manifest_path, candidate_path, audit_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to replace frozen preparation artifacts: "
            f"{[str(path) for path in existing]}. Use a new versioned manifest name."
        )

    counts = (
        manifest.groupby(["split", "class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "val", "test"], columns=CLASS_NAMES, fill_value=0)
    )
    audit.update(
        {
            "split_seed": args.seed,
            "split_ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "split_counts": {
                split: {name: int(value) for name, value in row.items()}
                for split, row in counts.iterrows()
            },
            "near_duplicate_hash": "64-bit DCT pHash",
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "near_duplicate_candidates": len(candidates),
            "near_duplicate_cross_split_candidates": int(candidates["cross_split"].sum())
            if not candidates.empty
            else 0,
            "near_duplicate_warning": (
                "pHash pairs are review candidates, not automatically confirmed duplicates. "
                "Visually inspect cross-split candidates before training."
            ),
            "manual_group_map": args.group_map,
            "effective_image_count": len(active_records),
            "effective_class_counts": dict(
                sorted(Counter(record.class_name for record in active_records).items())
            ),
            "quarantined_exact_duplicates": [
                {
                    "path": record.path,
                    "class_name": record.class_name,
                    "sha256": record.sha256,
                    "reason": "byte-identical duplicate; lexicographically first path retained",
                }
                for record in quarantined_records
            ],
        }
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    candidates.to_csv(candidate_temporary, index=False)
    candidate_temporary.replace(candidate_path)
    write_json(audit_path, audit)
    # The manifest is the final commit marker: training cannot observe it before the
    # candidate-review file and audit record exist.
    write_manifest(manifest_path, manifest)
    print(counts.to_string())
    print(f"\nManifest: {manifest_path}")
    print(f"Audit: {audit_path}")
    print(f"Near-duplicate candidates: {candidate_path}")


if __name__ == "__main__":
    main()
