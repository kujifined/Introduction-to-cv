#!/usr/bin/env python3
"""Build a verified original/transformed gallery for selected validation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from industrial_surface_defects.corruptions import apply_corruption
from industrial_surface_defects.reproducibility import write_json

DEFAULT_CASES: tuple[dict[str, str], ...] = (
    {
        "case": "A",
        "label": "corrected by camera-aware training",
        "path": "NEU-CLS/Cr_106.bmp",
        "condition": "gaussian_blur_s2",
        "configuration": "resnet18_minimal",
        "comparison_configuration": "resnet18_camera",
    },
    {
        "case": "B",
        "label": "persistent camera-aware error",
        "path": "NEU-CLS/Cr_53.bmp",
        "condition": "brightness_dark_s3",
        "configuration": "resnet18_camera",
    },
    {
        "case": "C",
        "label": "persistent camera-aware error",
        "path": "NEU-CLS/Pa_125.bmp",
        "condition": "gaussian_blur_s2",
        "configuration": "efficientnet_b0_camera",
    },
    {
        "case": "D",
        "label": "persistent camera-aware error",
        "path": "NEU-CLS/Sc_87.bmp",
        "condition": "shot_read_noise_s2",
        "configuration": "efficientnet_b0_camera",
    },
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}")


def load_cases(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return [dict(item) for item in DEFAULT_CASES]
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases JSON must be a non-empty list")
    required = {"case", "label", "path", "condition", "configuration"}
    for index, case in enumerate(cases):
        missing = required - set(case)
        if missing:
            raise ValueError(f"Case {index} is missing fields: {sorted(missing)}")
    return cases


def load_prediction(
    root: Path,
    *,
    configuration: str,
    condition: str,
    relative_path: str,
) -> pd.Series:
    prediction_path = root / "evaluation" / configuration / condition / "predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    table = pd.read_csv(prediction_path)
    selected = table.loc[table.path == relative_path]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {configuration}/{condition}/{relative_path}, "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def build_failure_review(
    *,
    root: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    cases_path: str | Path | None = None,
) -> Path:
    root = Path(root)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    image_path = output_dir / "failure_cases.png"
    csv_path = output_dir / "selected_failure_cases.csv"
    json_path = output_dir / "selected_failure_cases.json"
    if any(path.exists() for path in (image_path, csv_path, json_path)):
        raise FileExistsError(f"Refusing to overwrite failure-review artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path).set_index("path", verify_integrity=True)
    cases = load_cases(cases_path)
    figure, axes = plt.subplots(len(cases), 2, figsize=(8.2, 3.1 * len(cases)), squeeze=False)
    selected_rows: list[dict[str, object]] = []
    structured_cases: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        relative_path = case["path"]
        if relative_path not in manifest.index:
            raise ValueError(f"Case path is absent from manifest: {relative_path}")
        manifest_row = manifest.loc[relative_path]
        if manifest_row.status != "ok" or manifest_row.split != "val":
            raise ValueError(f"Case is not an active validation image: {relative_path}")
        source_path = data_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        digest = sha256_file(source_path)
        if digest != manifest_row.sha256:
            raise ValueError(f"Source hash differs from manifest: {relative_path}")

        with Image.open(source_path) as source:
            original = source.convert("L")
        condition = case["condition"]
        if condition == "clean":
            transformed = original.copy()
        else:
            corruption, severity = condition.rsplit("_s", 1)
            transformed = apply_corruption(
                original.copy(),
                corruption,
                int(severity),
                sample_key=relative_path,
            )
        prediction = load_prediction(
            root,
            configuration=case["configuration"],
            condition=condition,
            relative_path=relative_path,
        )
        if as_bool(prediction.correct):
            raise ValueError(
                f"Primary selected case is not an error: {case['case']} {relative_path}"
            )

        axes[index, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
        axes[index, 0].set_title(
            f"{case['case']} — {case['label']}\n"
            f"{Path(relative_path).name}: original ({prediction.true_class})",
            fontsize=9,
        )
        axes[index, 1].imshow(transformed, cmap="gray", vmin=0, vmax=255)
        transformed_title = (
            f"{condition}\n{case['configuration']}: "
            f"{prediction.true_class} → {prediction.predicted_class} "
            f"(p={float(prediction.confidence):.3f})"
        )
        comparison: dict[str, object] | None = None
        comparison_configuration = case.get("comparison_configuration")
        if comparison_configuration:
            comparison_prediction = load_prediction(
                root,
                configuration=comparison_configuration,
                condition=condition,
                relative_path=relative_path,
            )
            transformed_title += (
                f"\n{comparison_configuration}: {comparison_prediction.true_class} → "
                f"{comparison_prediction.predicted_class} "
                f"(p={float(comparison_prediction.confidence):.3f})"
            )
            comparison = {
                "configuration": comparison_configuration,
                "predicted_class": comparison_prediction.predicted_class,
                "confidence": float(comparison_prediction.confidence),
                "correct": as_bool(comparison_prediction.correct),
            }
        axes[index, 1].set_title(transformed_title, fontsize=8.5)
        for axis in axes[index]:
            axis.axis("off")

        probability_columns = [
            column for column in prediction.index if str(column).startswith("prob_")
        ]
        selected_rows.append(
            {
                "case": case["case"],
                "configuration": case["configuration"],
                "condition": condition,
                "path": relative_path,
                "true_class": prediction.true_class,
                "predicted_class": prediction.predicted_class,
                "confidence": float(prediction.confidence),
                "top2_class": prediction.top2_class,
                "top2_probability": float(prediction.top2_probability),
                **{column: float(prediction[column]) for column in probability_columns},
            }
        )
        structured_cases.append(
            {
                **case,
                "source_sha256": digest,
                "true_class": prediction.true_class,
                "predicted_class": prediction.predicted_class,
                "confidence": float(prediction.confidence),
                "probabilities": {
                    column.removeprefix("prob_"): float(prediction[column])
                    for column in probability_columns
                },
                "comparison": comparison,
                "interpretation": (
                    "Visual cause requires human review; no causal claim is generated."
                ),
            }
        )

    figure.suptitle(
        "Checked validation cases: exact deterministic transformed views",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(selected_rows).to_csv(csv_path, index=False)
    write_json(
        json_path,
        {
            "cases": structured_cases,
            "source_hashes_match_manifest": True,
            "transforms_reproduced_from_registered_code": True,
            "human_interpretation_required": True,
        },
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Pilot artifact root containing evaluation/CONFIGURATION/CONDITION",
    )
    parser.add_argument("--manifest", default="data/manifests/neu_cls_v2.csv")
    parser.add_argument("--data-root", default="data/raw/NEU-CLS")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", help="Optional JSON list overriding the default A-D cases")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = build_failure_review(
        root=args.root,
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        cases_path=args.cases,
    )
    print(f"Failure-review artifacts: {destination}")


if __name__ == "__main__":
    main()
