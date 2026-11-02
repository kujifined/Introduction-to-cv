"""YAML configuration loading and validation for audit profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL = {
    "version",
    "profile",
    "paths",
    "expectations",
    "experiment",
    "metrics",
    "bootstrap",
    "failure_cases",
}


def load_config(path: Path, project_root: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Audit config must contain a YAML mapping: {path}")
    missing = REQUIRED_TOP_LEVEL - set(config)
    if missing:
        raise ValueError(f"Audit config is missing sections: {sorted(missing)}")
    if config["version"] != 1:
        raise ValueError(f"Unsupported audit config version: {config['version']}")

    paths = config["paths"]
    for key in ("results_root", "failure_root", "output_dir", "contrasts"):
        value = paths.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"paths.{key} must be a non-empty string")
        candidate = Path(value)
        paths[key] = candidate if candidate.is_absolute() else project_root / candidate
    for key in (
        "manifest",
        "reference_manifest",
        "data_root",
        "source_archive",
    ):
        value = paths.get(key)
        if value in (None, ""):
            paths[key] = None
            continue
        candidate = Path(value)
        paths[key] = candidate if candidate.is_absolute() else project_root / candidate

    experiment = config["experiment"]
    if not experiment.get("configurations") or not experiment.get("seeds"):
        raise ValueError("experiment.configurations and experiment.seeds cannot be empty")
    experiment["seeds"] = [int(seed) for seed in experiment["seeds"]]
    config["metrics"]["tolerance"] = float(config["metrics"]["tolerance"])
    config["metrics"]["probability_sum_tolerance"] = float(
        config["metrics"]["probability_sum_tolerance"]
    )
    return config


def load_contrasts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("contrasts"), list):
        raise ValueError(f"Contrast config must define a contrasts list: {path}")
    contrasts = payload["contrasts"]
    for contrast in contrasts:
        if not {"name", "expression", "coefficients"}.issubset(contrast):
            raise ValueError(f"Invalid contrast entry: {contrast}")
        contrast["coefficients"] = {
            str(name): float(coefficient)
            for name, coefficient in contrast["coefficients"].items()
        }
    return contrasts

