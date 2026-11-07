"""Cross-check selected clean failure cases against direct prediction tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import Audit
from .predictions import CleanPredictions, prediction_path


def audit_failure_materials(
    results_root: Path,
    failure_root: Path,
    clean_predictions: CleanPredictions,
    selected_paths: list[str],
    experiment: dict[str, Any],
    audit: Audit,
) -> pd.DataFrame:
    selected = set(selected_paths)
    supplied = pd.read_csv(failure_root / "clean_failure_cross_model.csv")
    supplied_index = supplied.set_index(["path", "configuration", "seed"])
    rows: list[dict[str, Any]] = []
    mismatches = 0
    for (configuration, seed), (_, _, paths) in clean_predictions.items():
        direct = pd.read_csv(
            prediction_path(results_root, configuration, seed, "clean")
        )
        direct_index = direct.set_index("path")
        for path in sorted(selected):
            if path not in paths:
                raise ValueError(f"Selected failure path is absent from clean predictions: {path}")
            source = supplied_index.loc[(path, configuration, seed)]
            observed = direct_index.loc[path]
            same = (
                str(source["true_class"]) == str(observed["true_class"])
                and str(source["predicted_class"])
                == str(observed["predicted_class"])
                and bool(source["correct"]) == bool(observed["correct"])
                and abs(float(source["confidence"]) - float(observed["confidence"]))
                <= audit.tolerance
            )
            if not same:
                mismatches += 1
            rows.append(
                {
                    "path": path,
                    "configuration": configuration,
                    "seed": seed,
                    "true_class": observed["true_class"],
                    "predicted_class": observed["predicted_class"],
                    "confidence": float(observed["confidence"]),
                    "correct": bool(observed["correct"]),
                }
            )
    expected_rows = (
        len(selected)
        * len(experiment["configurations"])
        * len(experiment["seeds"])
    )
    audit.equal(
        "failure_cases",
        "clean_failure_cross_model.csv",
        "selected cross-model rows",
        expected_rows,
        len(rows),
    )
    audit.equal(
        "failure_cases",
        "clean_failure_cross_model.csv",
        "prediction mismatches versus clean outputs",
        0,
        mismatches,
    )
    return pd.DataFrame(rows).sort_values(["path", "configuration", "seed"])

