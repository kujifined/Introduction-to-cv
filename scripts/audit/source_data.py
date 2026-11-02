"""Verify the official source archive and extracted BMP files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import Audit, sha256_file


def audit_source_data(
    source_archive: Path | None,
    data_root: Path | None,
    manifest: pd.DataFrame,
    expectations: dict[str, Any],
    audit: Audit,
) -> dict[str, int | bool]:
    archive_verified = False
    if source_archive is None:
        audit.warn(
            "source_archive",
            "NEU-CLS.rar",
            "official archive fingerprint",
            "paths.source_archive is not configured.",
        )
    else:
        archive_exists = source_archive.is_file()
        audit.equal(
            "source_archive",
            str(source_archive),
            "archive exists",
            True,
            archive_exists,
        )
        if archive_exists:
            audit.equal(
                "source_archive",
                str(source_archive),
                "archive size",
                int(expectations["source_archive_bytes"]),
                source_archive.stat().st_size,
            )
            actual_hash = sha256_file(source_archive)
            audit.equal(
                "source_archive",
                str(source_archive),
                "archive SHA-256",
                str(expectations["source_archive_sha256"]),
                actual_hash,
            )
            archive_verified = (
                source_archive.stat().st_size
                == int(expectations["source_archive_bytes"])
                and actual_hash == str(expectations["source_archive_sha256"])
            )

    raw_files_checked = 0
    if data_root is None:
        audit.warn(
            "source_images",
            "raw NEU-CLS directory",
            "all source BMP hashes",
            "paths.data_root is not configured.",
        )
        return {
            "archive_verified": archive_verified,
            "raw_files_checked": raw_files_checked,
            "missing_raw_files": int(expectations["raw_file_count"]),
            "raw_hash_mismatches": 0,
            "unexpected_raw_files": 0,
        }

    expected_paths = set(manifest["path"].astype(str))
    actual_files = sorted(data_root.rglob("*.bmp")) if data_root.is_dir() else []
    actual_paths = {
        path.relative_to(data_root).as_posix(): path
        for path in actual_files
    }
    missing = sorted(expected_paths - set(actual_paths))
    unexpected = sorted(set(actual_paths) - expected_paths)
    mismatches: list[str] = []
    manifest_hashes = manifest.set_index("path")["sha256"].astype(str)
    for relative_path in sorted(expected_paths & set(actual_paths)):
        raw_files_checked += 1
        if sha256_file(actual_paths[relative_path]) != manifest_hashes.loc[relative_path]:
            mismatches.append(relative_path)

    audit.equal(
        "source_images",
        str(data_root),
        "raw BMP files found",
        int(expectations["raw_file_count"]),
        len(actual_files),
    )
    audit.equal(
        "source_images",
        str(data_root),
        "missing manifest BMP files",
        0,
        len(missing),
        note=", ".join(missing[:10]),
    )
    audit.equal(
        "source_images",
        str(data_root),
        "raw BMP SHA-256 mismatches",
        0,
        len(mismatches),
        note=", ".join(mismatches[:10]),
    )
    audit.equal(
        "source_images",
        str(data_root),
        "unexpected BMP files",
        0,
        len(unexpected),
        note=", ".join(unexpected[:10]),
    )
    return {
        "archive_verified": archive_verified,
        "raw_files_checked": raw_files_checked,
        "missing_raw_files": len(missing),
        "raw_hash_mismatches": len(mismatches),
        "unexpected_raw_files": len(unexpected),
    }

