"""Independent clean and corruption-robustness aggregation checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import Audit


def compare_frame_values(
    audit: Audit,
    category: str,
    artifact: str,
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    keys: list[str],
    values: list[str],
) -> None:
    expected_indexed = expected.set_index(keys).sort_index()
    actual_indexed = actual.set_index(keys).sort_index()
    audit.equal(
        category,
        artifact,
        "row keys",
        list(expected_indexed.index),
        list(actual_indexed.index),
    )
    for index in actual_indexed.index:
        label = index if isinstance(index, tuple) else (index,)
        for value in values:
            audit.close(
                category,
                artifact,
                f"{'/'.join(map(str, label))}/{value}",
                float(expected_indexed.loc[index, value]),
                float(actual_indexed.loc[index, value]),
            )


def audit_aggregations(
    results_root: Path,
    scores: pd.DataFrame,
    experiment: dict[str, Any],
    summary_metrics: list[str],
    contrasts: list[dict[str, Any]],
    audit: Audit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = scores.loc[scores["corruption"] == "clean"].copy()
    clean_summary_rows: list[dict[str, Any]] = []
    for configuration, group in clean.groupby("configuration", sort=False):
        row: dict[str, Any] = {"configuration": configuration}
        for metric in summary_metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        clean_summary_rows.append(row)
    clean_summary = pd.DataFrame(clean_summary_rows)
    compare_frame_values(
        audit,
        "aggregation_clean",
        "clean_scores_by_configuration.csv",
        pd.read_csv(results_root / "clean_scores_by_configuration.csv"),
        clean_summary,
        ["configuration"],
        [
            f"{metric}_{suffix}"
            for metric in summary_metrics
            for suffix in ("mean", "std")
        ],
    )

    robustness_rows: list[dict[str, Any]] = []
    for (configuration, seed), group in scores.groupby(
        ["configuration", "seed"],
        sort=False,
    ):
        clean_f1 = float(
            group.loc[group["corruption"] == "clean", "macro_f1"].iloc[0]
        )
        corrupted = group.loc[group["corruption"] != "clean"].copy()
        worst = corrupted.sort_values(
            ["macro_f1", "corruption", "severity"]
        ).iloc[0]
        mean_corrupted = float(corrupted["macro_f1"].mean())
        robustness_rows.append(
            {
                "configuration": configuration,
                "seed": int(seed),
                "clean_macro_f1": clean_f1,
                "mean_corrupted_macro_f1": mean_corrupted,
                "mean_absolute_macro_f1_drop": clean_f1 - mean_corrupted,
                "worst_corruption": worst["corruption"],
                "worst_severity": int(worst["severity"]),
                "worst_macro_f1": float(worst["macro_f1"]),
                "worst_absolute_macro_f1_drop": clean_f1
                - float(worst["macro_f1"]),
                "conditions": len(group),
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    saved_runs = pd.read_csv(
        results_root
        / "comparison_robustness"
        / "robustness_scores_by_run.csv"
    )
    compare_frame_values(
        audit,
        "aggregation_robustness",
        "comparison_robustness/robustness_scores_by_run.csv",
        saved_runs,
        robustness,
        ["configuration", "seed"],
        [
            "clean_macro_f1",
            "mean_corrupted_macro_f1",
            "mean_absolute_macro_f1_drop",
            "worst_severity",
            "worst_macro_f1",
            "worst_absolute_macro_f1_drop",
            "conditions",
        ],
    )
    saved_text_index = saved_runs.set_index(["configuration", "seed"])
    actual_text_index = robustness.set_index(["configuration", "seed"])
    for index in actual_text_index.index:
        audit.equal(
            "aggregation_robustness",
            "comparison_robustness/robustness_scores_by_run.csv",
            f"{index}/worst_corruption",
            str(saved_text_index.loc[index, "worst_corruption"]),
            str(actual_text_index.loc[index, "worst_corruption"]),
        )

    configuration_rows: list[dict[str, Any]] = []
    for configuration, group in robustness.groupby("configuration", sort=False):
        configuration_rows.append(
            {
                "configuration": configuration,
                "runs": len(group),
                "clean_macro_f1_mean": group["clean_macro_f1"].mean(),
                "clean_macro_f1_sd": group["clean_macro_f1"].std(ddof=1),
                "mean_corrupted_macro_f1_mean": group[
                    "mean_corrupted_macro_f1"
                ].mean(),
                "mean_corrupted_macro_f1_sd": group[
                    "mean_corrupted_macro_f1"
                ].std(ddof=1),
                "mean_absolute_macro_f1_drop_mean": group[
                    "mean_absolute_macro_f1_drop"
                ].mean(),
                "mean_absolute_macro_f1_drop_sd": group[
                    "mean_absolute_macro_f1_drop"
                ].std(ddof=1),
            }
        )
    robustness_summary = pd.DataFrame(configuration_rows)
    compare_frame_values(
        audit,
        "aggregation_robustness",
        "comparison_robustness/robustness_scores_by_configuration.csv",
        pd.read_csv(
            results_root
            / "comparison_robustness"
            / "robustness_scores_by_configuration.csv"
        ),
        robustness_summary,
        ["configuration"],
        [
            column
            for column in robustness_summary.columns
            if column != "configuration"
        ],
    )

    pivot = robustness.pivot(
        index="seed",
        columns="configuration",
        values="mean_corrupted_macro_f1",
    )
    contrast_rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        estimates = sum(
            coefficient * pivot[configuration]
            for configuration, coefficient in contrast["coefficients"].items()
        )
        seed_values = experiment["seeds"]
        values = estimates.loc[seed_values].to_numpy(dtype=float)
        contrast_rows.append(
            {
                "contrast": contrast["name"],
                "expression": contrast["expression"],
                "metric": "mean_corrupted_macro_f1",
                **{
                    f"seed_{seed}": float(estimates.loc[seed])
                    for seed in seed_values
                },
                "mean": float(values.mean()),
                "seed_sd": float(values.std(ddof=1)),
            }
        )
    robustness_contrasts = pd.DataFrame(contrast_rows)
    compare_frame_values(
        audit,
        "aggregation_robustness",
        "comparison_robustness/robustness_registered_contrasts.csv",
        pd.read_csv(
            results_root
            / "comparison_robustness"
            / "robustness_registered_contrasts.csv"
        ),
        robustness_contrasts,
        ["contrast"],
        [f"seed_{seed}" for seed in experiment["seeds"]]
        + ["mean", "seed_sd"],
    )
    return clean_summary, robustness_summary

