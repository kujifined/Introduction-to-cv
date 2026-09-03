"""Public comparison pipeline tests with synthetic predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from industrial_surface_defects.compare import (
    COMPARISON_ARTIFACT_FILENAMES,
    REGISTERED_CELL_METADATA,
    REGISTERED_CONFIGURATIONS,
    REGISTERED_SEEDS,
    compare,
)


def _write_prediction_run(
    root: Path,
    configuration: str,
    seed: int,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> Path:
    run_dir = root / f"{configuration}-{seed}"
    run_dir.mkdir(parents=True)
    path = run_dir / "predictions.csv"
    pd.DataFrame(
        {
            "path": [f"sample-{index:03d}.png" for index in range(len(targets))],
            "true_index": targets,
            "predicted_index": predictions,
        }
    ).to_csv(path, index=False)
    metadata: dict[str, object] = {"seed": seed}
    if configuration in REGISTERED_CELL_METADATA:
        model_name, augmentation = REGISTERED_CELL_METADATA[configuration]
        metadata.update(
            {
                "split": "test",
                "model_name": model_name,
                "augmentation_training": augmentation,
                "manifest_sha256": "synthetic-manifest-sha256",
            }
        )
    (run_dir / "metrics.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_compare_bootstrap_is_deterministic_and_detects_large_difference(
    tmp_path: Path,
) -> None:
    targets = np.repeat(np.arange(6), 8)
    perfect = targets.copy()
    cyclic_errors = (targets + 1) % 6
    specifications: list[str] = []
    for seed in (11, 22):
        better = _write_prediction_run(tmp_path, "better", seed, targets, perfect)
        worse = _write_prediction_run(tmp_path, "worse", seed, targets, cyclic_errors)
        specifications.extend((f"better={better}", f"worse={worse}"))

    first = compare(specifications, output_dir=tmp_path / "comparison-1", repetitions=128, seed=7)
    second = compare(specifications, output_dir=tmp_path / "comparison-2", repetitions=128, seed=7)

    first_contrast = pd.read_csv(first / "paired_bootstrap_contrasts.csv")
    second_contrast = pd.read_csv(second / "paired_bootstrap_contrasts.csv")
    pd.testing.assert_frame_equal(first_contrast, second_contrast)
    contrast = first_contrast.iloc[0]
    assert contrast["left_configuration"] == "better"
    assert contrast["right_configuration"] == "worse"
    assert contrast["pairing"] == "matched_seed"
    assert contrast["paired_runs"] == 2
    assert contrast["difference_left_minus_right"] == 1.0
    assert contrast["ci95_lower"] > 0
    assert contrast["ci95_upper"] > 0

    expected_artifacts = {
        "scores_by_run.csv",
        "scores_by_configuration.csv",
        "paired_bootstrap_contrasts.csv",
        "macro_f1_by_configuration.png",
        "comparison_metadata.json",
    }
    assert expected_artifacts.issubset(path.name for path in first.iterdir())
    metadata = json.loads((first / "comparison_metadata.json").read_text(encoding="utf-8"))
    assert metadata["configurations"] == ["better", "worse"]
    assert metadata["runs"] == 4
    assert metadata["test_images"] == len(targets)


def test_compare_rejects_misaligned_sample_sets(tmp_path: Path) -> None:
    targets = np.arange(6)
    first = _write_prediction_run(tmp_path, "first", 1, targets, targets)
    second = _write_prediction_run(tmp_path, "second", 1, targets, targets)
    frame = pd.read_csv(second)
    frame.loc[0, "path"] = "different.png"
    frame.to_csv(second, index=False)

    try:
        compare(
            [f"first={first}", f"second={second}"],
            output_dir=tmp_path / "comparison",
            repetitions=16,
        )
    except ValueError as error:
        assert "Sample paths differ" in str(error)
    else:
        raise AssertionError("compare accepted prediction runs for different samples")


def test_compare_rejects_unmatched_training_seeds(tmp_path: Path) -> None:
    targets = np.arange(6)
    first = _write_prediction_run(tmp_path, "first", 1, targets, targets)
    second = _write_prediction_run(tmp_path, "second", 2, targets, targets)

    with pytest.raises(ValueError, match="requires identical seed sets"):
        compare(
            [f"first={first}", f"second={second}"],
            output_dir=tmp_path / "comparison",
            repetitions=8,
        )


def test_registered_compare_emits_only_five_preregistered_contrasts(tmp_path: Path) -> None:
    targets = np.repeat(np.arange(6), 4)
    specifications: list[str] = []
    for configuration in REGISTERED_CONFIGURATIONS:
        for seed in REGISTERED_SEEDS:
            predictions = targets.copy()
            if configuration == "resnet18_minimal":
                predictions[::4] = (predictions[::4] + 1) % 6
            path = _write_prediction_run(
                tmp_path,
                configuration,
                seed,
                targets,
                predictions,
            )
            specifications.append(f"{configuration}={path}")

    destination = compare(
        specifications,
        output_dir=tmp_path / "registered-comparison",
        repetitions=32,
        seed=9,
        registered_design=True,
    )
    contrasts = pd.read_csv(destination / "paired_bootstrap_contrasts.csv")

    assert contrasts["contrast"].to_list() == [
        "architecture_under_minimal",
        "architecture_under_camera",
        "augmentation_within_resnet18",
        "augmentation_within_efficientnet_b0",
        "architecture_x_augmentation_interaction",
    ]
    assert set(contrasts["paired_runs"]) == {3}
    assert set(contrasts["pairing"]) == {"matched_seed"}
    interaction = contrasts.loc[
        contrasts["contrast"] == "architecture_x_augmentation_interaction"
    ].iloc[0]
    assert interaction["estimate"] < 0


def test_registered_compare_rejects_an_incomplete_seed_matrix(tmp_path: Path) -> None:
    targets = np.repeat(np.arange(6), 2)
    specifications: list[str] = []
    for configuration in REGISTERED_CONFIGURATIONS:
        seeds = REGISTERED_SEEDS
        if configuration == "efficientnet_b0_camera":
            seeds = REGISTERED_SEEDS[:-1]
        for seed in seeds:
            path = _write_prediction_run(
                tmp_path,
                configuration,
                seed,
                targets,
                targets,
            )
            specifications.append(f"{configuration}={path}")

    with pytest.raises(ValueError, match=r"requires seeds \[42, 43, 44\]"):
        compare(
            specifications,
            output_dir=tmp_path / "incomplete-comparison",
            repetitions=8,
            registered_design=True,
        )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("true_index", -1),
        ("true_index", 6),
        ("predicted_index", -1),
        ("predicted_index", 6),
    ],
)
def test_compare_rejects_class_indices_outside_zero_to_five(
    tmp_path: Path,
    column: str,
    invalid_value: int,
) -> None:
    targets = np.arange(6)
    first = _write_prediction_run(tmp_path, "first", 1, targets, targets)
    second = _write_prediction_run(tmp_path, "second", 1, targets, targets)
    frame = pd.read_csv(first)
    frame.loc[0, column] = invalid_value
    frame.to_csv(first, index=False)

    with pytest.raises(ValueError, match=rf"{column}.*0\.\.5"):
        compare(
            [f"first={first}", f"second={second}"],
            output_dir=tmp_path / "comparison",
            repetitions=16,
        )


@pytest.mark.parametrize("artifact_name", COMPARISON_ARTIFACT_FILENAMES)
def test_compare_refuses_to_overwrite_any_expected_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    targets = np.arange(6)
    first = _write_prediction_run(tmp_path, "first", 1, targets, targets)
    second = _write_prediction_run(tmp_path, "second", 1, targets, targets)
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    existing = output_dir / artifact_name
    existing.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        compare(
            [f"first={first}", f"second={second}"],
            output_dir=output_dir,
            repetitions=16,
        )
    assert existing.read_text(encoding="utf-8") == "keep me"
