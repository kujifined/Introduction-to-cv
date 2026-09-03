#!/usr/bin/env python3
"""Evaluate every completed run, then aggregate the registered comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_surface_defects.compare import (
    REGISTERED_CONFIGURATIONS,
    REGISTERED_SEEDS,
    compare,
)
from industrial_surface_defects.error_analysis import analyze
from industrial_surface_defects.evaluate import evaluate
from industrial_surface_defects.robustness import (
    aggregate_registered_robustness,
    evaluate_robustness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--acknowledge-test-use", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--with-error-analysis", action="store_true")
    parser.add_argument("--with-robustness", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser


def registered_checkpoints(outputs_root: str | Path) -> list[tuple[str, int, Path]]:
    root = Path(outputs_root)
    planned = [
        (
            configuration,
            seed,
            root / configuration / f"seed_{seed}" / "best.pt",
        )
        for configuration in REGISTERED_CONFIGURATIONS
        for seed in REGISTERED_SEEDS
    ]
    missing = [str(checkpoint) for _, _, checkpoint in planned if not checkpoint.is_file()]
    if missing:
        formatted = "\n- ".join(missing)
        raise FileNotFoundError(
            "The registered evaluation requires all 12 primary checkpoints. Missing:\n- "
            f"{formatted}"
        )
    return planned


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.split == "test" and not args.acknowledge_test_use:
        raise PermissionError("Pass --acknowledge-test-use only after the analysis is frozen")
    if args.with_robustness and args.split != "test":
        raise ValueError("Robustness is part of the final locked-test evaluation; use --split test")
    checkpoints = registered_checkpoints(args.outputs_root)

    specifications: list[str] = []
    for configuration, _seed, checkpoint in checkpoints:
        evaluation_dir = checkpoint.parent / "evaluation" / args.split
        predictions = evaluation_dir / "predictions.csv"
        if predictions.exists() and args.skip_existing:
            print(f"Using existing evaluation: {predictions}")
        else:
            evaluate(
                checkpoint,
                split=args.split,
                acknowledge_test_use=args.acknowledge_test_use,
                device_name=args.device,
                num_workers=args.num_workers,
                skip_latency=args.skip_latency,
            )
        specifications.append(f"{configuration}={predictions}")

        if args.with_error_analysis:
            error_dir = evaluation_dir / "error_analysis"
            if not (args.skip_existing and (error_dir / "summary.json").exists()):
                analyze(
                    predictions,
                    data_root="data/raw/NEU-CLS",
                    output_dir=error_dir,
                )
        if args.with_robustness:
            robustness_dir = checkpoint.parent / "evaluation" / "robustness"
            if not (args.skip_existing and (robustness_dir / "summary.json").exists()):
                evaluate_robustness(
                    checkpoint,
                    acknowledge_test_use=args.acknowledge_test_use,
                    device_name=args.device,
                    num_workers=0 if args.num_workers is None else args.num_workers,
                )

    if args.with_robustness:
        robustness_comparison_dir = Path("reports/tables/generated") / "robustness_comparison_test"
        aggregate_registered_robustness(
            args.outputs_root,
            output_dir=robustness_comparison_dir,
        )
        print(f"Aggregated robustness comparison: {robustness_comparison_dir}")

    comparison_dir = Path("reports/tables/generated") / f"comparison_{args.split}"
    compare(
        specifications,
        output_dir=comparison_dir,
        repetitions=args.bootstrap_repetitions,
        seed=42,
        registered_design=True,
    )
    print(f"Aggregated comparison: {comparison_dir}")


if __name__ == "__main__":
    main()
