"""Classification metrics with an explicit multiclass false-positive definition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

from .constants import CLASS_NAMES


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == y_true
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        include = (confidences > lower) & (confidences <= upper)
        if index == 0:
            include |= confidences == 0.0
        if include.any():
            value += include.mean() * abs(correct[include].mean() - confidences[include].mean())
    return float(value)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    targets = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != len(targets):
        raise ValueError("probabilities must have shape [n_samples, n_classes]")
    if scores.shape[1] != len(class_names):
        raise ValueError("probability columns do not match class_names")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    probability_sums = scores.sum(axis=1, keepdims=True)
    if not np.allclose(probability_sums, 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    # Float32 softmax output can miss scikit-learn's tighter float64 sum
    # tolerance after conversion. Renormalising preserves rankings while
    # preventing a misleading validation warning in log_loss.
    scores = scores / probability_sums

    predictions = scores.argmax(axis=1)
    labels = np.arange(len(class_names))
    raw_confusion = confusion_matrix(targets, predictions, labels=labels)
    row_sums = raw_confusion.sum(axis=1, keepdims=True)
    normalised_confusion = np.divide(
        raw_confusion,
        row_sums,
        out=np.zeros_like(raw_confusion, dtype=float),
        where=row_sums != 0,
    )

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    total = len(targets)
    for index, class_name in enumerate(class_names):
        tp = int(raw_confusion[index, index])
        fn = int(raw_confusion[index, :].sum() - tp)
        fp = int(raw_confusion[:, index].sum() - tp)
        tn = total - tp - fn - fp
        rows.append(
            {
                "class_index": index,
                "class_name": class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            }
        )

    metrics: dict[str, Any] = {
        "n_samples": total,
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "top2_accuracy": float(
            top_k_accuracy_score(targets, scores, k=min(2, len(class_names)), labels=labels)
        ),
        "negative_log_likelihood": float(log_loss(targets, scores, labels=labels)),
        "brier_score": multiclass_brier_score(targets, scores),
        "expected_calibration_error_10_bins": expected_calibration_error(targets, scores),
        "false_positive_definition": (
            "For class c: an image whose true defect type is not c but whose predicted type is c. "
            "NEU-CLS has no normal/no-defect class, so this is not an industrial false-alarm rate."
        ),
    }
    return metrics, pd.DataFrame(rows), raw_confusion, normalised_confusion


def prediction_table(
    paths: Sequence[str],
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
) -> pd.DataFrame:
    targets = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(paths) != len(targets) or scores.shape[0] != len(targets):
        raise ValueError("paths, targets, and probabilities must have equal sample counts")
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1 = order[:, 0]
    top2 = order[:, 1] if scores.shape[1] > 1 else order[:, 0]
    table = pd.DataFrame(
        {
            "path": list(paths),
            "true_index": targets,
            "true_class": [class_names[index] for index in targets],
            "predicted_index": top1,
            "predicted_class": [class_names[index] for index in top1],
            "confidence": scores[np.arange(len(scores)), top1],
            "top2_index": top2,
            "top2_class": [class_names[index] for index in top2],
            "top2_probability": scores[np.arange(len(scores)), top2],
            "margin": scores[np.arange(len(scores)), top1] - scores[np.arange(len(scores)), top2],
            "correct": top1 == targets,
        }
    )
    for index, class_name in enumerate(class_names):
        table[f"prob_{class_name}"] = scores[:, index]
    return table


def confusion_pairs(
    raw_confusion: np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
) -> pd.DataFrame:
    pairs: list[dict[str, Any]] = []
    for left in range(len(class_names)):
        for right in range(left + 1, len(class_names)):
            left_to_right = int(raw_confusion[left, right])
            right_to_left = int(raw_confusion[right, left])
            pairs.append(
                {
                    "class_a": class_names[left],
                    "class_b": class_names[right],
                    "a_predicted_as_b": left_to_right,
                    "b_predicted_as_a": right_to_left,
                    "mutual_confusions": left_to_right + right_to_left,
                }
            )
    return pd.DataFrame(pairs).sort_values(
        ["mutual_confusions", "class_a", "class_b"],
        ascending=[False, True, True],
    )
