"""Verified checkpoint inference shared by the CLI and presentation demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from .constants import CLASS_NAMES
from .corruptions import apply_corruption
from .models import build_model
from .transforms import build_transform


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_models(repository: str | Path, *, device: str = "cpu") -> dict:
    """Load the four frozen seed-43 weights without downloading initialization."""
    repository = Path(repository)
    metadata = json.loads((repository / "results/final/ARTIFACTS.json").read_text())
    selected = [item for item in metadata["checkpoints"] if item["seed"] == 43]
    if len(selected) != 4 or len({item["configuration"] for item in selected}) != 4:
        raise ValueError("Expected four distinct seed-43 checkpoints")
    result = {}
    for item in selected:
        path = repository / item["local_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/prepare_final_artifacts.py")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if (
            checkpoint["class_names"] != list(CLASS_NAMES)
            or checkpoint["num_classes"] != len(CLASS_NAMES)
            or checkpoint["manifest_sha256"] != metadata["manifest_sha256"]
            or config["experiment"]["name"] != item["configuration"]
            or config["experiment"]["seed"] != 43
            or config["data"]["image_size"] != 224
        ):
            raise ValueError(f"Checkpoint metadata mismatch: {path}")
        model = build_model(config["model"]["name"], len(CLASS_NAMES), pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        result[item["configuration"]] = model
    return result


@torch.inference_mode()
def predict_image(
    models: dict,
    image: Image.Image,
    *,
    corruption: str = "clean",
    severity: int = 0,
    sample_key: str = "uploaded_image",
) -> list[dict]:
    """Run actual forward passes; no filenames, labels or reference CSV are features."""
    if corruption == "clean":
        if severity != 0:
            raise ValueError("Clean inference requires severity 0")
        transformed = image.convert("L")
    else:
        transformed = apply_corruption(image, corruption, severity, sample_key=sample_key)
    tensor = build_transform(224, "minimal", train=False)(transformed).unsqueeze(0)
    results = []
    for name, model in models.items():
        device = next(model.parameters()).device
        probabilities = model(tensor.to(device)).softmax(dim=1)[0].cpu()
        if probabilities.shape != (6,) or not torch.isfinite(probabilities).all():
            raise ValueError("Invalid model probabilities")
        indices = probabilities.argsort(descending=True)
        results.append(
            {
                "configuration": name,
                "seed": 43,
                "predicted_class": CLASS_NAMES[indices[0].item()],
                "confidence": probabilities[indices[0]].item(),
                "top2_class": CLASS_NAMES[indices[1].item()],
                "top2_probability": probabilities[indices[1]].item(),
                "probabilities": dict(zip(CLASS_NAMES, probabilities.tolist(), strict=True)),
            }
        )
    return results
