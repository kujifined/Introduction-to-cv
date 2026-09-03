"""Manifest-backed datasets and deterministic dataloaders."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .constants import CLASS_NAMES, CLASS_TO_INDEX
from .reproducibility import seed_worker
from .transforms import build_transform

REQUIRED_MANIFEST_COLUMNS = {
    "path",
    "class_name",
    "class_index",
    "split",
    "sha256",
    "group_id",
    "status",
}


def validate_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["path"].duplicated().any():
        raise ValueError("A path appears more than once in the manifest")
    unknown_splits = set(manifest["split"]) - {"train", "val", "test", "excluded"}
    if unknown_splits:
        raise ValueError(f"Manifest contains unknown splits: {sorted(unknown_splits)}")
    if not {"train", "val", "test"}.issubset(set(manifest["split"])):
        raise ValueError("Manifest must contain train, val, and test rows")
    allowed_statuses = {"ok", "duplicate_quarantined"}
    if not set(manifest["status"]).issubset(allowed_statuses):
        raise ValueError(f"Manifest contains unknown status values: {set(manifest['status'])}")
    excluded = manifest["split"] == "excluded"
    if not (manifest.loc[excluded, "status"] == "duplicate_quarantined").all():
        raise ValueError("Only explicitly quarantined duplicates may use split=excluded")
    if not (manifest.loc[~excluded, "status"] == "ok").all():
        raise ValueError("Train/val/test rows must have status=ok")
    expected_indices = manifest["class_name"].map(CLASS_TO_INDEX)
    if expected_indices.isna().any() or not (expected_indices == manifest["class_index"]).all():
        raise ValueError("Manifest class_name/class_index mapping is inconsistent")
    active = manifest.loc[manifest["status"] == "ok"]
    splits_per_digest = active.groupby("sha256", observed=True)["split"].nunique()
    if (splits_per_digest > 1).any():
        raise ValueError("Byte-identical images occur in more than one split")
    if "group_id" in manifest.columns:
        splits_per_group = active.groupby("group_id", observed=True)["split"].nunique()
        if (splits_per_group > 1).any():
            raise ValueError("A reviewed source group occurs in more than one split")
    candidates_path = manifest_path.with_name(f"{manifest_path.stem}_near_duplicates.csv")
    if candidates_path.exists():
        candidates = pd.read_csv(candidates_path)
        if not candidates.empty and "cross_split" in candidates:
            cross_split = candidates.loc[
                candidates["cross_split"].astype(str).str.lower().isin({"true", "1"})
            ]
            if not cross_split.empty:
                if "review_status" not in cross_split:
                    raise ValueError(
                        f"Cross-split near-duplicate candidates need review: {candidates_path}"
                    )
                accepted = cross_split["review_status"].astype(str).str.strip().str.lower()
                unresolved = ~accepted.isin({"not_duplicate"})
                if unresolved.any():
                    raise ValueError(
                        "Cross-split near-duplicate candidates are unresolved. Mark visually "
                        f"distinct pairs as not_duplicate in {candidates_path}; group confirmed "
                        "duplicates and create a new versioned manifest."
                    )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_files(manifest_path: str | Path, data_root: str | Path) -> None:
    """Verify that every manifest path exists below data_root and matches its SHA-256."""
    manifest = validate_manifest(manifest_path)
    root = Path(data_root).resolve()
    failures: list[str] = []

    for row in manifest.itertuples(index=False):
        relative_path = Path(str(row.path))
        if relative_path.is_absolute():
            failures.append(f"{row.path}: path must be relative to data_root")
            continue

        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            failures.append(f"{row.path}: path escapes data_root")
            continue

        if not candidate.is_file():
            failures.append(f"{row.path}: file is missing")
            continue

        expected = str(row.sha256).strip().lower()
        actual = _sha256_file(candidate)
        if actual != expected:
            failures.append(
                f"{row.path}: SHA-256 mismatch (expected {expected}, computed {actual})"
            )

    if failures:
        preview = "\n".join(failures[:10])
        suffix = "" if len(failures) <= 10 else f"\n... and {len(failures) - 10} more"
        raise ValueError(
            f"Manifest file integrity check failed for {len(failures)} row(s):\n{preview}{suffix}"
        )


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str,
        transform: Callable | None = None,
        *,
        return_path: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        manifest = validate_manifest(self.manifest_path)
        self.rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Manifest contains no rows for split={split!r}")
        if not set(self.rows["class_name"]).issubset(CLASS_TO_INDEX):
            unknown = sorted(set(self.rows["class_name"]) - set(CLASS_TO_INDEX))
            raise ValueError(f"Manifest contains unknown classes: {unknown}")
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = self.data_root / str(row["path"])
        with Image.open(path) as source:
            image = source.convert("L")
        if self.transform is not None:
            image = self.transform(image)
        target = int(row["class_index"])
        if self.return_path:
            return image, target, str(row["path"])
        return image, target


def build_dataloader(
    *,
    manifest_path: str | Path,
    data_root: str | Path,
    split: str,
    image_size: int,
    augmentation: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    return_path: bool = False,
) -> DataLoader:
    train = split == "train"
    dataset = ManifestDataset(
        manifest_path,
        data_root,
        split,
        build_transform(image_size, augmentation, train=train),
        return_path=return_path,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def class_counts(manifest_path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    return (
        manifest.groupby(["split", "class_name"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_NAMES, fill_value=0)
    )
