"""Focused checks for preregistered false-positive and similarity outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from industrial_surface_defects.constants import CLASS_NAMES
from industrial_surface_defects.error_analysis import (
    REGISTERED_SIMILARITY_PAIRS,
    analyze,
)


def test_error_analysis_writes_fp_composition_and_registered_pair_galleries(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "images"
    data_root.mkdir()
    image_paths: list[str] = []
    for index in range(len(CLASS_NAMES)):
        relative_path = f"sample_{index}.png"
        Image.fromarray(np.full((16, 16), index * 30, dtype=np.uint8)).save(
            data_root / relative_path
        )
        image_paths.append(relative_path)

    targets = np.arange(len(CLASS_NAMES))
    predicted = targets.copy()
    predicted[0] = 1
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "path": image_paths,
            "true_index": targets,
            "true_class": [CLASS_NAMES[index] for index in targets],
            "predicted_index": predicted,
            "predicted_class": [CLASS_NAMES[index] for index in predicted],
            "confidence": np.linspace(0.95, 0.70, len(CLASS_NAMES)),
            "correct": targets == predicted,
        }
    ).to_csv(predictions_path, index=False)

    destination = analyze(
        predictions_path,
        data_root=data_root,
        output_dir=tmp_path / "analysis",
        top_n=1,
        top_pairs=1,
    )

    composition = pd.read_csv(destination / "false_positive_source_composition.csv")
    assert composition.to_dict("records") == [
        {
            "predicted_class": "inclusion",
            "true_class": "crazing",
            "false_positive_count": 1,
            "false_positives_for_predicted_class": 1,
            "share_within_predicted_class_false_positives": 1.0,
        }
    ]
    for class_a, class_b in REGISTERED_SIMILARITY_PAIRS:
        assert (destination / f"pair_{class_a}__{class_b}.png").is_file()
