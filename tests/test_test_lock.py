"""Regression tests for the locked-test acknowledgement boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from industrial_surface_defects.evaluate import evaluate
from industrial_surface_defects.robustness import evaluate_robustness


def test_evaluate_refuses_locked_test_before_opening_checkpoint(tmp_path: Path) -> None:
    nonexistent_checkpoint = tmp_path / "does-not-exist.pt"

    with pytest.raises(PermissionError, match="test split is locked"):
        evaluate(
            nonexistent_checkpoint,
            split="test",
            acknowledge_test_use=False,
            device_name="cpu",
        )


def test_robustness_refuses_locked_test_before_opening_checkpoint(tmp_path: Path) -> None:
    nonexistent_checkpoint = tmp_path / "does-not-exist.pt"

    with pytest.raises(PermissionError, match="locked test images"):
        evaluate_robustness(
            nonexistent_checkpoint,
            acknowledge_test_use=False,
            device_name="cpu",
        )
