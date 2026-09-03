"""Turn final predictions into false-positive tables and inspectable image galleries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .constants import CLASS_NAMES
from .metrics import confusion_pairs
from .reproducibility import write_json

REQUIRED_COLUMNS = {
    "path",
    "true_index",
    "true_class",
    "predicted_index",
    "predicted_class",
    "confidence",
    "correct",
}
REGISTERED_SIMILARITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("crazing", "rolled_in_scale"),
    ("crazing", "pitted_surface"),
    ("rolled_in_scale", "pitted_surface"),
)


def _load_predictions(path: str | Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing)}")
    predictions["correct"] = (
        predictions["correct"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if predictions["correct"].isna().any():
        raise ValueError("The correct column must contain boolean values")
    return predictions


def _show_image(axis, path: Path, title: str) -> None:
    try:
        with Image.open(path) as source:
            axis.imshow(source.convert("L"), cmap="gray", vmin=0, vmax=255)
    except OSError:
        axis.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def _class_gallery(
    errors: pd.DataFrame,
    data_root: Path,
    path: Path,
    *,
    group_column: str,
    top_n: int,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        len(CLASS_NAMES),
        top_n,
        figsize=(2.5 * top_n, 2.15 * len(CLASS_NAMES)),
        squeeze=False,
    )
    for row_index, class_name in enumerate(CLASS_NAMES):
        selected = (
            errors.loc[errors[group_column] == class_name]
            .sort_values("confidence", ascending=False)
            .head(top_n)
        )
        for column_index, axis in enumerate(axes[row_index]):
            if column_index >= len(selected):
                axis.axis("off")
                continue
            item = selected.iloc[column_index]
            _show_image(
                axis,
                data_root / item["path"],
                f"{Path(str(item['path'])).name}\n"
                f"true={item['true_class']}  pred={item['predicted_class']}\n"
                f"p={item['confidence']:.3f}",
            )
        axes[row_index, 0].text(
            -0.12,
            0.5,
            class_name,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0.02, 0, 1, 0.98))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _pair_gallery(
    errors: pd.DataFrame,
    data_root: Path,
    path: Path,
    class_a: str,
    class_b: str,
    *,
    top_n: int,
) -> None:
    selected = errors.loc[
        ((errors["true_class"] == class_a) & (errors["predicted_class"] == class_b))
        | ((errors["true_class"] == class_b) & (errors["predicted_class"] == class_a))
    ].sort_values("confidence", ascending=False)
    selected = selected.head(top_n)
    columns = min(5, max(1, len(selected)))
    rows = max(1, math.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.7 * columns, 2.5 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for index, axis in enumerate(flat_axes):
        if index >= len(selected):
            axis.axis("off")
            continue
        item = selected.iloc[index]
        _show_image(
            axis,
            data_root / item["path"],
            f"{Path(str(item['path'])).name}\n"
            f"{item['true_class']} -> {item['predicted_class']}  "
            f"p={item['confidence']:.3f}",
        )
    figure.suptitle(f"Mutual errors: {class_a} vs {class_b}", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_confidence(predictions: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    correct = predictions.loc[predictions["correct"], "confidence"]
    incorrect = predictions.loc[~predictions["correct"], "confidence"]
    bins = [index / 20 for index in range(21)]
    axis.hist(correct, bins=bins, alpha=0.7, label=f"correct (n={len(correct)})")
    axis.hist(incorrect, bins=bins, alpha=0.7, label=f"wrong (n={len(incorrect)})")
    axis.set(
        xlabel="Top-1 confidence",
        ylabel="Images",
        title="Confidence does not guarantee correctness",
        xlim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def analyze(
    predictions_path: str | Path,
    *,
    data_root: str | Path,
    output_dir: str | Path | None = None,
    top_n: int = 5,
    top_pairs: int = 3,
) -> Path:
    predictions_path = Path(predictions_path)
    predictions = _load_predictions(predictions_path)
    errors = predictions.loc[~predictions["correct"]].copy()
    destination = Path(output_dir) if output_dir else predictions_path.parent / "error_analysis"
    destination.mkdir(parents=True, exist_ok=True)

    false_positives = errors.sort_values(["predicted_class", "confidence"], ascending=[True, False])
    false_negatives = errors.sort_values(["true_class", "confidence"], ascending=[True, False])
    false_positives.to_csv(destination / "false_positives.csv", index=False)
    false_negatives.to_csv(destination / "false_negatives.csv", index=False)
    if false_positives.empty:
        fp_source_composition = pd.DataFrame(
            columns=[
                "predicted_class",
                "true_class",
                "false_positive_count",
                "false_positives_for_predicted_class",
                "share_within_predicted_class_false_positives",
            ]
        )
    else:
        fp_source_composition = (
            false_positives.groupby(["predicted_class", "true_class"], observed=True)
            .size()
            .rename("false_positive_count")
            .reset_index()
        )
        fp_source_composition["false_positives_for_predicted_class"] = (
            fp_source_composition.groupby("predicted_class", observed=True)[
                "false_positive_count"
            ].transform("sum")
        )
        fp_source_composition["share_within_predicted_class_false_positives"] = (
            fp_source_composition["false_positive_count"]
            / fp_source_composition["false_positives_for_predicted_class"]
        )
        fp_source_composition = fp_source_composition.sort_values(
            ["predicted_class", "false_positive_count", "true_class"],
            ascending=[True, False, True],
        )
    fp_source_composition.to_csv(
        destination / "false_positive_source_composition.csv",
        index=False,
    )

    raw_confusion = pd.crosstab(
        predictions["true_index"],
        predictions["predicted_index"],
    ).reindex(index=range(len(CLASS_NAMES)), columns=range(len(CLASS_NAMES)), fill_value=0)
    pairs = confusion_pairs(raw_confusion.to_numpy())
    pairs.to_csv(destination / "confusion_pairs.csv", index=False)

    taxonomy = errors[["path", "true_class", "predicted_class", "confidence"]].copy()
    for column in [
        "low_contrast",
        "mixed_pattern",
        "orientation_or_scale",
        "blur_or_noise",
        "crop_or_truncation",
        "brightness_or_material_shift",
        "suspected_label_ambiguity",
        "reviewer_notes",
    ]:
        taxonomy[column] = ""
    taxonomy.to_csv(destination / "manual_error_taxonomy_template.csv", index=False)

    data_root_path = Path(data_root)
    _class_gallery(
        false_positives,
        data_root_path,
        destination / "top_false_positives.png",
        group_column="predicted_class",
        top_n=top_n,
        title="Highest-confidence false positives by predicted class",
    )
    _class_gallery(
        false_negatives,
        data_root_path,
        destination / "top_false_negatives.png",
        group_column="true_class",
        top_n=top_n,
        title="Highest-confidence false negatives by true class",
    )
    _plot_confidence(predictions, destination / "confidence_correct_vs_wrong.png")

    pair_rows = pairs.loc[pairs["mutual_confusions"] > 0].head(top_pairs)
    pairs_to_plot = list(REGISTERED_SIMILARITY_PAIRS)
    for row in pair_rows.itertuples(index=False):
        empirical_pair = (row.class_a, row.class_b)
        if empirical_pair not in pairs_to_plot and empirical_pair[::-1] not in pairs_to_plot:
            pairs_to_plot.append(empirical_pair)
    for class_a, class_b in pairs_to_plot:
        filename = f"pair_{class_a}__{class_b}.png"
        _pair_gallery(
            errors,
            data_root_path,
            destination / filename,
            class_a,
            class_b,
            top_n=top_n * 2,
        )

    summary = {
        "images": len(predictions),
        "errors": len(errors),
        "correct": int(predictions["correct"].sum()),
        "highest_confidence_error": float(errors["confidence"].max()) if not errors.empty else None,
        "top_mutual_confusion_pairs": pair_rows.to_dict("records"),
        "registered_similarity_pair_galleries": [
            {"class_a": class_a, "class_b": class_b}
            for class_a, class_b in REGISTERED_SIMILARITY_PAIRS
        ],
        "false_positive_scope": (
            "One-vs-rest among six defect types; NEU-CLS contains no normal/no-defect images."
        ),
        "manual_review_required": (
            "The taxonomy CSV is intentionally blank. Two reviewers should code visible causes "
            "without changing primary labels or metrics."
        ),
    }
    write_json(destination / "summary.json", summary)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--data-root", default="data/raw/NEU-CLS")
    parser.add_argument("--output-dir")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-pairs", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = analyze(
        args.predictions,
        data_root=args.data_root,
        output_dir=args.output_dir,
        top_n=args.top_n,
        top_pairs=args.top_pairs,
    )
    print(f"Error-analysis artifacts: {destination}")


if __name__ == "__main__":
    main()
