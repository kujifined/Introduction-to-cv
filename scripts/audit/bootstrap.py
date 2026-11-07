"""Independent registered paired stratified bootstrap audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import Audit, manual_macro_f1
from .predictions import CleanPredictions


def stratified_resample_indices(
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = []
    for class_index in np.unique(targets):
        indices = np.flatnonzero(targets == class_index)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled)


def audit_bootstrap(
    results_root: Path,
    clean_predictions: CleanPredictions,
    experiment: dict[str, Any],
    bootstrap_config: dict[str, Any],
    contrasts: list[dict[str, Any]],
    class_count: int,
    audit: Audit,
) -> pd.DataFrame:
    configurations = experiment["configurations"]
    seeds = experiment["seeds"]
    reference_targets, _, reference_paths = clean_predictions[
        (configurations[0], seeds[0])
    ]
    for targets, _, paths in clean_predictions.values():
        if not np.array_equal(targets, reference_targets) or paths != reference_paths:
            raise ValueError("Clean prediction tables are not aligned for paired bootstrap")

    def contrast_value(
        indices: np.ndarray,
        training_seed: int,
        coefficients: dict[str, float],
    ) -> float:
        sampled_targets = reference_targets[indices]
        return float(
            sum(
                coefficient
                * manual_macro_f1(
                    sampled_targets,
                    clean_predictions[(configuration, training_seed)][1][indices],
                    class_count,
                )
                for configuration, coefficient in coefficients.items()
            )
        )

    saved = pd.read_csv(
        results_root / "comparison_clean" / "paired_bootstrap_contrasts.csv"
    ).set_index("contrast")
    rows: list[dict[str, Any]] = []
    all_indices = np.arange(len(reference_targets))
    repetitions = int(bootstrap_config["repetitions"])
    base_seed = int(bootstrap_config["base_seed"])
    for contrast_index, contrast in enumerate(contrasts):
        seed_estimates = [
            contrast_value(all_indices, seed, contrast["coefficients"])
            for seed in seeds
        ]
        rng = np.random.default_rng(base_seed + contrast_index)
        estimates = np.empty(repetitions, dtype=np.float64)
        for repetition in range(repetitions):
            indices = stratified_resample_indices(reference_targets, rng)
            estimates[repetition] = np.mean(
                [
                    contrast_value(indices, seed, contrast["coefficients"])
                    for seed in seeds
                ]
            )
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        row = {
            "contrast": contrast["name"],
            "expression": contrast["expression"],
            "estimate": float(np.mean(seed_estimates)),
            "seed_mean": float(np.mean(seed_estimates)),
            "seed_sd": float(np.std(seed_estimates, ddof=1)),
            "seed_estimates": json.dumps(seed_estimates),
            "paired_runs": len(seeds),
            "pairing": "matched_seed",
            "bootstrap_mean_estimate": float(estimates.mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "bootstrap_repetitions": repetitions,
            "bootstrap_unit": (
                "stratified test image; fixed mean across matched training seeds"
            ),
        }
        rows.append(row)
        for metric in (
            "estimate",
            "seed_mean",
            "seed_sd",
            "bootstrap_mean_estimate",
            "ci95_lower",
            "ci95_upper",
        ):
            audit.close(
                "bootstrap",
                "comparison_clean/paired_bootstrap_contrasts.csv",
                f"{contrast['name']}/{metric}",
                float(saved.loc[contrast["name"], metric]),
                float(row[metric]),
            )
        audit.equal(
            "bootstrap",
            "comparison_clean/paired_bootstrap_contrasts.csv",
            f"{contrast['name']}/repetitions",
            int(saved.loc[contrast["name"], "bootstrap_repetitions"]),
            repetitions,
        )
    return pd.DataFrame(rows)

