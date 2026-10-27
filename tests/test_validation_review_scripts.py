"""Focused tests for the reproducible validation-review CLIs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_validation_results import calculate_metrics, parse_correct
from scripts.build_failure_review import as_bool, load_cases
from scripts.run_validation_pilot import (
    CONFIGURATIONS,
    manifest_sha256,
    parse_checkpoint_specs,
)


def test_independent_accuracy_and_macro_f1() -> None:
    table = pd.DataFrame(
        {
            "true_index": [0, 0, 1, 2, 3, 4, 5],
            "predicted_index": [0, 1, 1, 2, 3, 4, 5],
        }
    )

    accuracy, macro_f1, confusion, class_f1 = calculate_metrics(table, class_count=6)

    assert accuracy == pytest.approx(6 / 7)
    assert macro_f1 == pytest.approx(8 / 9)
    assert confusion.sum() == 7
    assert class_f1.tolist() == pytest.approx([2 / 3, 2 / 3, 1, 1, 1, 1])


def test_boolean_parsers_reject_ambiguous_values() -> None:
    assert parse_correct(pd.Series(["True", "0", 1])).tolist() == [True, False, True]
    assert as_bool("false") is False
    with pytest.raises(ValueError, match="non-boolean"):
        parse_correct(pd.Series(["maybe"]))
    with pytest.raises(ValueError, match="Expected a boolean"):
        as_bool("maybe")


def test_default_failure_cases_cover_a_through_d() -> None:
    cases = load_cases(None)

    assert [case["case"] for case in cases] == ["A", "B", "C", "D"]
    assert len({case["path"] for case in cases}) == 4
    assert len({case["condition"].split("_s", 1)[0] for case in cases[1:]}) == 3


def test_manifest_hash_normalises_windows_line_endings(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"path,split\r\nNEU-CLS/Cr_1.bmp,val\r\n")

    expected = hashlib.sha256(b"path,split\nNEU-CLS/Cr_1.bmp,val\n").hexdigest()

    assert manifest_sha256(manifest) == expected


def test_checkpoint_specs_require_all_four_configurations(tmp_path: Path) -> None:
    specifications = []
    for configuration in CONFIGURATIONS:
        checkpoint = tmp_path / f"{configuration}.pt"
        checkpoint.touch()
        specifications.append(f"{configuration}={checkpoint}")

    parsed = parse_checkpoint_specs(specifications)

    assert set(parsed) == set(CONFIGURATIONS)
    assert all(path.is_file() for path in parsed.values())
