"""Create Grad-CAM diagnostics for frozen classification checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional
from PIL import Image

from .config import DataSettings
from .models import gradcam_target_layer, load_checkpoint_model
from .reproducibility import resolve_device, write_json
from .transforms import build_transform, denormalize


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, module, inputs, output: torch.Tensor) -> None:
        del module, inputs
        self.activations = output
        output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def close(self) -> None:
        self.handle.remove()

    def __call__(self, image: torch.Tensor, target_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[0, target_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0]
        minimum, maximum = cam.min(), cam.max()
        cam = (cam - minimum) / (maximum - minimum).clamp_min(1e-12)
        return cam.detach(), logits.detach()


def explain(
    checkpoint_path: str | Path,
    image_path: str | Path,
    *,
    output_path: str | Path | None = None,
    target_index: int | None = None,
    device_name: str = "auto",
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    image_path = Path(image_path)
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint_model(str(checkpoint_path), device=device)
    data = DataSettings(**checkpoint["config"]["data"])
    transform = build_transform(data.image_size, "minimal", train=False)
    with Image.open(image_path) as source:
        source_image = source.convert("L")
        tensor = transform(source_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        preview_logits = model(tensor)
        probabilities = preview_logits.softmax(dim=1)[0]
    predicted_index = int(probabilities.argmax())
    selected_index = predicted_index if target_index is None else target_index
    if not 0 <= selected_index < len(checkpoint["class_names"]):
        raise ValueError(f"target_index must be in [0, {len(checkpoint['class_names']) - 1}]")

    model.eval()
    cam_calculator = GradCAM(
        model,
        gradcam_target_layer(model, checkpoint["config"]["model"]["name"]),
    )
    try:
        cam, logits = cam_calculator(tensor, selected_index)
    finally:
        cam_calculator.close()
    probabilities = logits.softmax(dim=1)[0].cpu().numpy()
    display_image = denormalize(tensor[0]).mean(dim=0).detach().cpu().numpy()
    cam_array = cam.cpu().numpy()

    destination = (
        Path(output_path)
        if output_path is not None
        else checkpoint_path.parent / "explanations" / f"{image_path.stem}_gradcam.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Model input")
    axes[1].imshow(cam_array, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM: {checkpoint['class_names'][selected_index]}")
    axes[2].imshow(display_image, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(cam_array, cmap="inferno", alpha=0.48, vmin=0, vmax=1)
    axes[2].set_title(
        f"pred={checkpoint['class_names'][predicted_index]} ({probabilities[predicted_index]:.3f})"
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)
    write_json(
        destination.with_suffix(".json"),
        {
            "image": str(image_path),
            "checkpoint": str(checkpoint_path),
            "predicted_class": checkpoint["class_names"][predicted_index],
            "predicted_probability": float(probabilities[predicted_index]),
            "gradcam_target_class": checkpoint["class_names"][selected_index],
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(
                    checkpoint["class_names"], probabilities, strict=True
                )
            },
            "interpretation_warning": (
                "Grad-CAM visualises model sensitivity; it does not prove a causal human "
                "explanation."
            ),
        },
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("image")
    parser.add_argument("--output")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = explain(
        args.checkpoint,
        args.image,
        output_path=args.output,
        target_index=args.target_index,
        device_name=args.device,
    )
    print(f"Grad-CAM artifact: {destination}")


if __name__ == "__main__":
    main()
