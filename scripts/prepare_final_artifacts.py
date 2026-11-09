"""Restore hash-verified final weights from the downloaded Kaggle results ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(archive_path, root, *, all_checkpoints=False):
    root, archive_path = Path(root), Path(archive_path)
    metadata = json.loads((root / "results/final/ARTIFACTS.json").read_text())
    if sha256(archive_path) != metadata["archive_sha256"]:
        raise ValueError("Results ZIP SHA-256 mismatch; use the recorded Kaggle version")
    selected = [item for item in metadata["checkpoints"] if all_checkpoints or item["seed"] == 43]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        for item in selected:
            for key in ("local_path", "path_in_results_archive"):
                p = PurePosixPath(item[key])
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("Unsafe artifact path")
            destination = root / item["local_path"]
            if destination.exists() and sha256(destination) != item["sha256"]:
                raise FileExistsError(f"Existing checkpoint differs: {destination}")
            info = archive.getinfo(item["path_in_results_archive"])
            if info.file_size > 60_000_000:
                raise ValueError("Unexpected checkpoint size")
        for item in selected:
            destination = root / item["local_path"]
            if destination.exists():
                continue
            content = archive.read(item["path_in_results_archive"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Checkpoint SHA-256 mismatch: {item['configuration']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(destination)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--all-checkpoints", action="store_true")
    args = parser.parse_args()
    count = restore(
        args.archive, Path(__file__).resolve().parents[1], all_checkpoints=args.all_checkpoints
    )
    print(f"Verified {count} checkpoints. Dataset images are obtained separately.")


if __name__ == "__main__":
    main()
