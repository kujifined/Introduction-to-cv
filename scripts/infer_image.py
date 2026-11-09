"""Classify an image with all four verified seed-43 checkpoints, without training."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from PIL import Image

from industrial_surface_defects.inference import file_sha256, load_models, predict_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--corruption", choices=("clean", "gaussian_blur"), default="clean")
    parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[1]
    models = load_models(root, device=args.device)
    with Image.open(args.image) as image:
        predictions = predict_image(
            models, image, corruption=args.corruption, severity=args.severity
        )
    result = {
        "input": str(args.image),
        "input_sha256": file_sha256(args.image),
        "corruption": args.corruption,
        "severity": args.severity,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": args.device,
        },
        "predictions": predictions,
    }
    content = json.dumps(result, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n")


if __name__ == "__main__":
    main()
