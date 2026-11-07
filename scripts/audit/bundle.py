"""Post-extraction integrity checks for the frozen result bundle."""

from __future__ import annotations

from pathlib import Path

from .common import Audit, json_load, sha256_file


def audit_bundle(
    results_root: Path,
    expected_file_count: int,
    audit: Audit,
) -> dict[str, int]:
    manifest_path = results_root / "bundle_manifest.json"
    bundle = json_load(manifest_path)
    missing = 0
    hash_mismatch = 0
    byte_mismatch = 0
    for item in bundle["files"]:
        path = results_root / item["path"]
        if not path.is_file():
            missing += 1
            continue
        if path.stat().st_size != int(item["bytes"]):
            byte_mismatch += 1
        if sha256_file(path) != item["sha256"]:
            hash_mismatch += 1
    audit.equal(
        "bundle",
        "bundle_manifest.json",
        "listed file count",
        expected_file_count,
        len(bundle["files"]),
    )
    audit.equal("bundle", "bundle_manifest.json", "missing listed files", 0, missing)
    audit.equal(
        "bundle",
        "bundle_manifest.json",
        "byte-size mismatches",
        0,
        byte_mismatch,
    )
    audit.equal(
        "bundle",
        "bundle_manifest.json",
        "SHA-256 mismatches",
        0,
        hash_mismatch,
    )
    freeze = results_root / "protocol_freeze.json"
    expected_freeze_hash = (results_root / "protocol_freeze.sha256").read_text(
        encoding="utf-8"
    ).strip()
    audit.equal(
        "bundle",
        "protocol_freeze.json",
        "frozen protocol SHA-256",
        expected_freeze_hash,
        sha256_file(freeze),
    )
    return {
        "listed_files": len(bundle["files"]),
        "missing_files": missing,
        "byte_mismatches": byte_mismatch,
        "hash_mismatches": hash_mismatch,
    }

