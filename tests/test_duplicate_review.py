"""Tests for exhaustive translation-based duplicate review."""

from __future__ import annotations

import numpy as np

from industrial_surface_defects.duplicate_review import (
    find_best_translation,
    overlapping_views,
)


def test_overlapping_views_uses_documented_shift_direction() -> None:
    left = np.arange(25, dtype=np.uint8).reshape(5, 5)
    right = np.zeros((5, 5), dtype=np.uint8)
    right[1:, 2:] = left[:-1, :-2]

    left_overlap, right_overlap = overlapping_views(left, right, dy=-1, dx=-2)

    assert left_overlap.shape == (4, 3)
    assert np.array_equal(left_overlap, right_overlap)


def test_find_best_translation_checks_multiple_offsets() -> None:
    generator = np.random.default_rng(42)
    left = generator.integers(0, 256, size=(30, 32), dtype=np.uint8)
    right = generator.integers(0, 256, size=(30, 32), dtype=np.uint8)
    right[2:, 3:] = left[:-2, :-3]

    match = find_best_translation(left, right, max_shift=4)

    assert (match.dy, match.dx) == (-2, -3)
    assert (match.overlap_height, match.overlap_width) == (28, 29)
    assert match.pixel_identical is True
    assert match.max_absolute_error == 0
    assert match.mean_absolute_error == 0.0
    assert match.correlation == 1.0


def test_find_best_translation_finds_zero_shift_for_exact_copy() -> None:
    image = np.arange(120, dtype=np.uint8).reshape(10, 12)

    match = find_best_translation(image, image.copy(), max_shift=3)

    assert (match.dy, match.dx) == (0, 0)
    assert (match.overlap_height, match.overlap_width) == image.shape
    assert match.pixel_identical is True
