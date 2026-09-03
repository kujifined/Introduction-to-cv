"""Registered 4 x 3 robustness aggregation tests using synthetic CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from industrial_surface_defects.compare import (
    REGISTERED_CONFIGURATIONS,
    REGISTERED_CONTRASTS,
    REGISTERED_SEEDS,
)
from industrial_surface_defects.corruptions import CORRUPTION_NAMES
from industrial_surface_defects.robustness import (
    ROBUSTNESS_AGGREGATION_FILENAMES,
    aggregate_registered_robustness,
)

BASE_SCORES = {
    "resnet18_minimal": 0.50,
    "resnet18_camera": 0.55,
    "efficientnet_b0_minimal": 0.60,
    "efficientnet_b0_camera": 0.70,
}


def _write_registered_robustness_matrix(root: Path) -> None:
    for configuration in REGISTERED_CONFIGURATIONS:
        for seed_index, training_seed in enumerate(REGISTERED_SEEDS):
            directory = root / configuration / f"seed_{training_seed}" / "evaluation" / "robustness"
            directory.mkdir(parents=True)
            seed_offset = 0.01 * seed_index
            rows = [
                {
                    "corruption": "clean",
                    "severity": 0,
                    "macro_f1": BASE_SCORES[configuration] + seed_offset + 0.10,
                }
            ]
            for corruption_index, corruption in enumerate(CORRUPTION_NAMES):
                for severity in (1, 2, 3):
                    common_condition_offset = 0.001 * (3 * corruption_index + severity)
                    rows.append(
                        {
                            "corruption": corruption,
                            "severity": severity,
                            "macro_f1": (
                                BASE_SCORES[configuration] + seed_offset - common_condition_offset
                            ),
                        }
                    )
            pd.DataFrame(rows).to_csv(
                directory / "robustness_metrics.csv",
                index=False,
            )


def test_registered_robustness_aggregation_writes_scores_and_five_contrasts(
    tmp_path: Path,
) -> None:
    outputs_root = tmp_path / "outputs"
    _write_registered_robustness_matrix(outputs_root)

    destination = aggregate_registered_robustness(
        outputs_root,
        output_dir=tmp_path / "aggregate",
    )

    assert {path.name for path in destination.iterdir()} == set(ROBUSTNESS_AGGREGATION_FILENAMES)
    scores_by_run = pd.read_csv(destination / "robustness_scores_by_run.csv")
    scores_by_configuration = pd.read_csv(destination / "robustness_scores_by_configuration.csv")
    contrasts = pd.read_csv(destination / "robustness_registered_contrasts.csv")
    assert len(scores_by_run) == 12
    expected_conditions = 1 + 3 * len(CORRUPTION_NAMES)
    assert set(scores_by_run["conditions"]) == {expected_conditions}
    assert scores_by_configuration["runs"].to_list() == [3, 3, 3, 3]
    assert contrasts["contrast"].to_list() == [name for name, _, _ in REGISTERED_CONTRASTS]

    expected_means = {
        "architecture_under_minimal": 0.10,
        "architecture_under_camera": 0.15,
        "augmentation_within_resnet18": 0.05,
        "augmentation_within_efficientnet_b0": 0.10,
        "architecture_x_augmentation_interaction": 0.05,
    }
    for row in contrasts.itertuples(index=False):
        assert row.mean == pytest.approx(expected_means[row.contrast])
        assert row.seed_sd == pytest.approx(0.0, abs=1e-12)

    metadata = json.loads(
        (destination / "robustness_aggregation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["runs"] == 12
    assert metadata["conditions_per_run"] == expected_conditions
    assert {condition["corruption"] for condition in metadata["condition_grid"]} >= {
        "clean",
        "shot_read_noise",
    }

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        aggregate_registered_robustness(outputs_root, output_dir=destination)


def test_registered_robustness_aggregation_rejects_an_incomplete_grid(
    tmp_path: Path,
) -> None:
    outputs_root = tmp_path / "outputs"
    _write_registered_robustness_matrix(outputs_root)
    broken_path = (
        outputs_root
        / "resnet18_minimal"
        / "seed_42"
        / "evaluation"
        / "robustness"
        / "robustness_metrics.csv"
    )
    broken = pd.read_csv(broken_path)
    broken = broken.loc[~((broken["corruption"] == "shot_read_noise") & (broken["severity"] == 3))]
    broken.to_csv(broken_path, index=False)

    with pytest.raises(ValueError, match="registered robustness condition grid"):
        aggregate_registered_robustness(
            outputs_root,
            output_dir=tmp_path / "aggregate",
        )
