"""Shared primitives used by the independent validation and final audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class Audit:
    """Collect check-level evidence without stopping at the first mismatch."""

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        category: str,
        artifact: str,
        check: str,
        expected: Any,
        actual: Any,
        status: str,
        *,
        difference: float | None = None,
        tolerance: float | None = None,
        note: str = "",
    ) -> None:
        self.rows.append(
            {
                "category": category,
                "artifact": artifact,
                "check": check,
                "expected": expected,
                "actual": actual,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "status": status,
                "note": note,
            }
        )

    def equal(
        self,
        category: str,
        artifact: str,
        check: str,
        expected: Any,
        actual: Any,
        *,
        note: str = "",
    ) -> None:
        status = "PASS" if actual == expected else "FAIL"
        self.add(category, artifact, check, expected, actual, status, note=note)

    def close(
        self,
        category: str,
        artifact: str,
        check: str,
        expected: float,
        actual: float,
        *,
        tolerance: float | None = None,
        note: str = "",
    ) -> None:
        selected_tolerance = self.tolerance if tolerance is None else tolerance
        difference = abs(float(actual) - float(expected))
        status = "PASS" if difference <= selected_tolerance else "FAIL"
        self.add(
            category,
            artifact,
            check,
            expected,
            actual,
            status,
            difference=difference,
            tolerance=selected_tolerance,
            note=note,
        )

    def warn(self, category: str, artifact: str, check: str, note: str) -> None:
        self.add(
            category,
            artifact,
            check,
            "independent verification",
            "not available",
            "WARN",
            note=note,
        )


def manual_confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: int,
) -> np.ndarray:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    return confusion


def manual_macro_f1(
    targets: np.ndarray,
    predictions: np.ndarray,
    classes: int,
) -> float:
    confusion = manual_confusion(targets, predictions, classes)
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(classes, dtype=np.float64),
        where=denominator != 0,
    )
    return float(f1.mean())


def manual_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    """Compute all registered metrics without importing production helpers."""
    probabilities = probabilities.astype(np.float64, copy=True)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    classes = probabilities.shape[1]
    predictions = probabilities.argmax(axis=1)
    confusion = manual_confusion(targets, predictions, classes)
    support = confusion.sum(axis=1).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_support = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(classes, dtype=np.float64),
        where=support != 0,
    )
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(classes, dtype=np.float64),
        where=predicted_support != 0,
    )
    f1_denominator = precision + recall
    per_class_f1 = np.divide(
        2 * precision * recall,
        f1_denominator,
        out=np.zeros(classes, dtype=np.float64),
        where=f1_denominator != 0,
    )
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    true_probability = probabilities[np.arange(len(targets)), targets]
    epsilon = np.finfo(probabilities.dtype).eps
    true_probability = np.clip(true_probability, epsilon, 1.0 - epsilon)
    one_hot = np.eye(classes, dtype=np.float64)[targets]
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        include = (confidence > boundaries[index]) & (
            confidence <= boundaries[index + 1]
        )
        if index == 0:
            include |= confidence == 0.0
        if include.any():
            ece += float(
                include.mean()
                * abs(correct[include].mean() - confidence[include].mean())
            )
    return {
        "n_samples": int(len(targets)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(per_class_f1.mean()),
        "weighted_f1": float(np.average(per_class_f1, weights=support)),
        "top2_accuracy": float((top2 == targets[:, None]).any(axis=1).mean()),
        "negative_log_likelihood": float(-np.log(true_probability).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "expected_calibration_error_10_bins": float(ece),
    }

