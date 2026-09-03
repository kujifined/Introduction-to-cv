"""Metrics, false-positive semantics, and prediction artifact tests."""

from __future__ import annotations

import numpy as np
import pytest

from industrial_surface_defects.constants import CLASS_NAMES
from industrial_surface_defects.metrics import (
    compute_metrics,
    confusion_pairs,
    prediction_table,
)


def _probabilities_for(predictions: np.ndarray, classes: int = 6) -> np.ndarray:
    probabilities = np.full((len(predictions), classes), 0.04, dtype=np.float64)
    probabilities[np.arange(len(predictions)), predictions] = 0.80
    return probabilities


def test_compute_metrics_uses_one_vs_rest_false_positive_definition() -> None:
    targets = np.repeat(np.arange(6), 2)
    predictions = np.array([0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 4])
    probabilities = _probabilities_for(predictions)

    metrics, per_class, raw, normalized = compute_metrics(targets, probabilities)

    expected_raw = np.zeros((6, 6), dtype=int)
    for target, prediction in zip(targets, predictions, strict=True):
        expected_raw[target, prediction] += 1
    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
    assert metrics["n_samples"] == 12
    assert 0 <= metrics["macro_f1"] <= 1
    assert "true defect type is not c" in metrics["false_positive_definition"]
    assert "no normal/no-defect class" in metrics["false_positive_definition"]

    inclusion = per_class.set_index("class_name").loc["inclusion"]
    assert inclusion["tp"] == 2
    assert inclusion["fp"] == 1
    assert inclusion["fn"] == 0
    assert inclusion["tn"] == 9
    assert inclusion["false_positive_rate"] == pytest.approx(0.1)


def test_prediction_table_contains_ranked_scores_and_error_flag() -> None:
    scores = np.array(
        [
            [0.70, 0.20, 0.04, 0.03, 0.02, 0.01],
            [0.10, 0.55, 0.25, 0.05, 0.03, 0.02],
        ]
    )
    table = prediction_table(["a.png", "b.png"], [0, 0], scores)

    assert table["predicted_class"].to_list() == ["crazing", "inclusion"]
    assert table["top2_class"].to_list() == ["inclusion", "patches"]
    assert table["correct"].to_list() == [True, False]
    assert table["margin"].to_list() == pytest.approx([0.50, 0.30])
    assert set(f"prob_{name}" for name in CLASS_NAMES).issubset(table.columns)


def test_confusion_pairs_ranks_mutual_errors() -> None:
    raw = np.eye(6, dtype=int)
    raw[0, 1] = 3
    raw[1, 0] = 2
    raw[2, 3] = 4
    pairs = confusion_pairs(raw)

    assert len(pairs) == 15
    first = pairs.iloc[0]
    assert (first["class_a"], first["class_b"]) == ("crazing", "inclusion")
    assert first["a_predicted_as_b"] == 3
    assert first["b_predicted_as_a"] == 2
    assert first["mutual_confusions"] == 5


def test_compute_metrics_rejects_non_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        compute_metrics([0], np.ones((1, 6)))


def test_prediction_table_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="equal sample counts"):
        prediction_table(["only-one.png"], [0, 1], np.eye(2))
