#!/usr/bin/env python3
"""Run the four registered configurations for all registered training seeds."""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path

import torch

from industrial_surface_defects.config import load_config
from industrial_surface_defects.train import train

DEFAULT_CONFIGS = (
    "configs/resnet18_minimal.yaml",
    "configs/resnet18_camera.yaml",
    "configs/efficientnet_b0_minimal.yaml",
    "configs/efficientnet_b0_camera.yaml",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--run-suffix")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configs = args.configs or list(DEFAULT_CONFIGS)
    planned = [(config_path, seed) for config_path in configs for seed in args.seeds]
    print(f"Planned runs: {len(planned)}")
    for config_path, seed in planned:
        config = load_config(
            config_path,
            seed=seed,
            device=args.device,
            pretrained=False if args.no_pretrained else None,
        )
        training = config.training
        if args.epochs is not None:
            training = replace(training, epochs=args.epochs)
        if args.num_workers is not None:
            training = replace(training, num_workers=args.num_workers)
        config = replace(config, training=training)
        run_name = f"seed_{seed}" + (f"_{args.run_suffix}" if args.run_suffix else "")
        run_dir = Path(config.experiment.output_root) / config.experiment.name / run_name
        print(f"- {config.experiment.name}, seed={seed} -> {run_dir}")
        if args.dry_run:
            continue
        if args.skip_existing and run_dir.exists() and any(run_dir.iterdir()):
            completed_artifacts = (
                run_dir / "best.pt",
                run_dir / "training_summary.json",
            )
            if all(path.is_file() for path in completed_artifacts):
                print("  skipped (completed run artifacts exist)")
                continue
            raise RuntimeError(
                f"Refusing to skip incomplete run directory: {run_dir}. "
                "Inspect it and resume under a new run suffix."
            )
        train(config, run_suffix=args.run_suffix)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
