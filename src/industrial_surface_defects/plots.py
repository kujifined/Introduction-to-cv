"""Publication-ready diagnostic plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .constants import CLASS_NAMES


def plot_confusion_matrix(
    matrix: np.ndarray,
    path: str | Path,
    *,
    normalised: bool,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    format_code = ".1%" if normalised else "d"
    sns.heatmap(
        matrix,
        annot=True,
        fmt=format_code,
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Row fraction" if normalised else "Images"},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(
        xlabel="Predicted defect type",
        ylabel="True defect type",
        title="Row-normalised confusion matrix" if normalised else "Confusion matrix",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_reliability_diagram(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    path: str | Path,
    *,
    bins: int = 10,
) -> None:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    boundaries = np.linspace(0, 1, bins + 1)
    centres = (boundaries[:-1] + boundaries[1:]) / 2
    accuracies = np.full(bins, np.nan)
    mean_confidence = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for index in range(bins):
        mask = (confidence > boundaries[index]) & (confidence <= boundaries[index + 1])
        if index == 0:
            mask |= confidence == 0
        counts[index] = int(mask.sum())
        if mask.any():
            accuracies[index] = correct[mask].mean()
            mean_confidence[index] = confidence[mask].mean()

    figure, (calibration_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(6.5, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    calibration_axis.plot([0, 1], [0, 1], "--", color="0.45", label="perfect calibration")
    available = ~np.isnan(accuracies)
    calibration_axis.plot(
        mean_confidence[available],
        accuracies[available],
        marker="o",
        linewidth=2,
        label="model",
    )
    calibration_axis.set(ylabel="Empirical accuracy", ylim=(0, 1.02), title="Reliability diagram")
    calibration_axis.legend()
    count_axis.bar(centres, counts, width=0.09, color="#4C78A8")
    count_axis.set(xlabel="Top-1 confidence", ylabel="Count", xlim=(0, 1))
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
