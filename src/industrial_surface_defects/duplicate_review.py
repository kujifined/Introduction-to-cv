"""Reproduce translation-based review of near-duplicate image pairs."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class TranslationMatch:
    """Best overlap found after translating the right image relative to the left."""

    dy: int
    dx: int
    overlap_height: int
    overlap_width: int
    correlation: float
    mean_absolute_error: float
    max_absolute_error: int
    pixel_identical: bool


class ImageSource(Protocol):
    def read(self, relative_path: str) -> bytes: ...


class DirectoryImageSource:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def read(self, relative_path: str) -> bytes:
        return (self.root / relative_path).read_bytes()


class ZipImageSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self, relative_path: str) -> bytes:
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(relative_path)


def decode_grayscale(content: bytes) -> np.ndarray:
    """Decode an image without changing its pixels."""
    with Image.open(io.BytesIO(content)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def overlapping_views(
    left: np.ndarray,
    right: np.ndarray,
    *,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlap after translating ``right`` by ``(dy, dx)`` over ``left``.

    A negative ``dy`` moves the right image upward; a negative ``dx`` moves it
    leftward. For example, ``dy=-2, dx=-2`` compares ``left[:-2, :-2]`` with
    ``right[2:, 2:]`` for equally sized inputs.
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Translation review expects two-dimensional grayscale arrays")

    if dy >= 0:
        left_y, right_y = dy, 0
    else:
        left_y, right_y = 0, -dy
    if dx >= 0:
        left_x, right_x = dx, 0
    else:
        left_x, right_x = 0, -dx

    height = min(left.shape[0] - left_y, right.shape[0] - right_y)
    width = min(left.shape[1] - left_x, right.shape[1] - right_x)
    if height <= 0 or width <= 0:
        raise ValueError(f"Shift dy={dy}, dx={dx} leaves no overlapping pixels")

    return (
        left[left_y : left_y + height, left_x : left_x + width],
        right[right_y : right_y + height, right_x : right_x + width],
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.array_equal(left, right):
        return 1.0
    left_values = left.astype(np.float64, copy=False).ravel()
    right_values = right.astype(np.float64, copy=False).ravel()
    left_values = left_values - left_values.mean()
    right_values = right_values - right_values.mean()
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_values, right_values) / denominator)


def measure_translation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    dy: int,
    dx: int,
) -> TranslationMatch:
    left_overlap, right_overlap = overlapping_views(left, right, dy=dy, dx=dx)
    absolute_difference = np.abs(
        left_overlap.astype(np.int16) - right_overlap.astype(np.int16)
    )
    return TranslationMatch(
        dy=dy,
        dx=dx,
        overlap_height=int(left_overlap.shape[0]),
        overlap_width=int(left_overlap.shape[1]),
        correlation=_correlation(left_overlap, right_overlap),
        mean_absolute_error=float(absolute_difference.mean()),
        max_absolute_error=int(absolute_difference.max()),
        pixel_identical=bool(np.count_nonzero(absolute_difference) == 0),
    )


def find_best_translation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_shift: int = 12,
) -> TranslationMatch:
    """Exhaustively search all ``(2 * max_shift + 1) ** 2`` translations."""
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    candidates = (
        measure_translation(left, right, dy=dy, dx=dx)
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
    )
    return max(
        candidates,
        key=lambda match: (
            match.pixel_identical,
            match.correlation,
            -match.mean_absolute_error,
            match.overlap_height * match.overlap_width,
            -(abs(match.dy) + abs(match.dx)),
            -match.dy,
            -match.dx,
        ),
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _recorded_shift(notes: str) -> tuple[int | None, int | None]:
    match = re.search(r"dy=(-?\d+) dx=(-?\d+)", notes)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def audit_candidates(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    source: ImageSource,
    *,
    max_shift: int,
) -> pd.DataFrame:
    """Verify source hashes and reproduce the best translation for each pair."""
    manifest_by_path = manifest.set_index("path", verify_integrity=True)
    results: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        left_content = source.read(row.left_path)
        right_content = source.read(row.right_path)
        for path, content in (
            (row.left_path, left_content),
            (row.right_path, right_content),
        ):
            if path not in manifest_by_path.index:
                raise ValueError(f"Candidate path is absent from the manifest: {path}")
            expected_hash = str(manifest_by_path.loc[path, "sha256"])
            if _sha256(content) != expected_hash:
                raise ValueError(f"Image hash does not match the manifest: {path}")

        best = find_best_translation(
            decode_grayscale(left_content),
            decode_grayscale(right_content),
            max_shift=max_shift,
        )
        recorded_dy, recorded_dx = _recorded_shift(str(row.review_notes))
        results.append(
            {
                "class_name": row.class_name,
                "left_path": row.left_path,
                "right_path": row.right_path,
                "review_status": row.review_status,
                "translations_checked": (2 * max_shift + 1) ** 2,
                **asdict(best),
                "recorded_dy": recorded_dy,
                "recorded_dx": recorded_dx,
                "matches_recorded_shift": (
                    recorded_dy is None
                    or (best.dy == recorded_dy and best.dx == recorded_dx)
                ),
            }
        )
    return pd.DataFrame(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="data/manifests/neu_cls_v2_near_duplicates.csv",
    )
    parser.add_argument("--manifest", default="data/manifests/neu_cls_v2.csv")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-root", default="data/raw/NEU-CLS")
    source.add_argument("--archive", help="ZIP containing manifest-relative image paths")
    parser.add_argument("--max-shift", type=int, default=12)
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Audit all candidate rows instead of confirmed shifted duplicates only",
    )
    parser.add_argument("--output", help="Optional CSV destination")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    candidates = pd.read_csv(args.candidates)
    if not args.all_candidates:
        candidates = candidates.loc[
            candidates["review_status"] == "confirmed_shifted_duplicate"
        ].copy()
    if candidates.empty:
        raise ValueError("No candidate pairs selected")

    manifest = pd.read_csv(args.manifest)
    image_source: ImageSource
    if args.archive:
        image_source = ZipImageSource(args.archive)
    else:
        image_source = DirectoryImageSource(args.data_root)

    results = audit_candidates(
        candidates,
        manifest,
        image_source,
        max_shift=args.max_shift,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
