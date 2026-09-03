#!/usr/bin/env python3
"""Download the official NEU-CLS archive without redistributing it through Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

OFFICIAL_LANDING_PAGE = "https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm"
OFFICIAL_DRIVE_FILE_ID = "1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc"
OFFICIAL_DRIVE_PAGE = f"https://drive.google.com/file/d/{OFFICIAL_DRIVE_FILE_ID}/view"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: Path, destination: Path) -> tuple[str, list[str]]:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"Extraction destination is not empty: {destination}")
    candidates = [
        ("unar", ["unar", "-output-directory", str(destination), str(archive)]),
        ("7z", ["7z", "x", str(archive), f"-o{destination}"]),
        ("unrar", ["unrar", "x", str(archive), str(destination)]),
    ]
    tar_path = shutil.which("tar")
    if tar_path:
        version = subprocess.run(
            [tar_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if "bsdtar" in f"{version.stdout}\n{version.stderr}".lower():
            candidates.append(("tar", [tar_path, "-xf", str(archive), "-C", str(destination)]))
    failures: list[str] = []
    for executable, command in candidates:
        if shutil.which(executable):
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                failures.append(f"{executable}: exit status {error.returncode}")
                continue
            return executable, [str(part) for part in command]
    failure_note = f" Tried but failed: {', '.join(failures)}." if failures else ""
    raise RuntimeError(
        "No working RAR extractor found. Install one of unar, 7z, unrar, or bsdtar, or "
        f"extract the archive manually into data/raw/NEU-CLS.{failure_note}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/archives/NEU-CLS.rar")
    parser.add_argument("--extract-to", default="data/raw/NEU-CLS")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--acknowledge-source-terms",
        action="store_true",
        help=(
            "Confirm that you reviewed the official dataset page and will not commit or "
            "redistribute the downloaded images without permission."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.acknowledge_source_terms:
        raise PermissionError(
            "Review the official source and re-run with --acknowledge-source-terms."
        )
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Install the data extra first: uv sync --extra data") from error

    archive = Path(args.archive)
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(id=OFFICIAL_DRIVE_FILE_ID, output=str(archive), quiet=False)
    if downloaded is None or not archive.is_file():
        raise RuntimeError("Google Drive download did not produce the expected archive")

    metadata: dict[str, object] = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "landing_page_url": OFFICIAL_LANDING_PAGE,
        "download_url_used": OFFICIAL_DRIVE_PAGE,
        "archive_filename": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "gdown_version": getattr(gdown, "__version__", "unknown"),
        "redistributed_in_repository": False,
        "licence_note": (
            "The official page does not state a standard machine-readable licence; review "
            "the source terms before any redistribution."
        ),
    }
    if args.extract:
        extractor, command = _extract(archive, Path(args.extract_to))
        metadata["extraction_tool"] = extractor
        metadata["extraction_command"] = command
        metadata["extracted_to"] = str(Path(args.extract_to))

    metadata_path = archive.with_suffix(archive.suffix + ".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Archive: {archive}")
    print(f"SHA-256: {metadata['archive_sha256']}")
    print(f"Acquisition record: {metadata_path}")


if __name__ == "__main__":
    main()
