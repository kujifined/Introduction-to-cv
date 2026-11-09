"""Offline CPU inference for the three frozen ICV defense examples."""

import base64
import hashlib
import io
import json
import os
from contextlib import nullcontext
from html import escape
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image, ImageFilter
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

DISPLAY_NAMES = {
    "resnet18_minimal": "ResNet-18 · minimal",
    "resnet18_camera": "ResNet-18 · camera-aware",
    "efficientnet_b0_minimal": "EfficientNet-B0 · minimal",
    "efficientnet_b0_camera": "EfficientNet-B0 · camera-aware",
}


def verified_bytes(content, expected_sha, name):
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError(f"SHA-256 mismatch: {name}")
    return content


def clean_preprocessing():
    # Frozen core build_transform(224, "minimal", train=False), revision 760889a.
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def create_model(architecture, num_classes):
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def checkpoint_source(directory):
    override = os.environ.get("ICV_DEMO_WEIGHTS")
    if override:
        return Path(override).expanduser()
    root = directory.parent
    candidates = (
        root / "results_download" / "icv_final_results.zip",
        root / "results_download" / "icv_final_results",
        root / "checkpoints",
        root / "week9" / "icv_final_results.zip",
        root / "week9" / "icv_final_results",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


class LiveDemo:
    def __init__(self, assets, source):
        self.spec = json.loads((assets / "demo_cases.json").read_text(encoding="utf-8"))
        self.seed = self.spec["training_seed"]
        self.classes = self.spec["class_names"]
        self.cases = {item["id"]: item for item in self.spec["cases"]}
        self.preprocess = clean_preprocessing()
        self.models = {}
        self.input_bytes = {}
        if self.seed != 43 or len(self.classes) != 6:
            raise ValueError("The demo requires the four registered six-class seed-43 runs.")
        if [row["configuration"] for row in self.spec["checkpoints"]] != list(DISPLAY_NAMES):
            raise ValueError("Unexpected checkpoint configurations in demo_cases.json.")
        if not source.exists():
            raise FileNotFoundError(
                f"Weights not found at {source}. Set ICV_DEMO_WEIGHTS to the local "
                "icv_final_results.zip or its extracted directory (with outputs/)."
            )
        for case in self.cases.values():
            path = case["path"]
            if path not in self.input_bytes:
                self.input_bytes[path] = (assets / "demo_inputs" / path).read_bytes()
            verified_bytes(self.input_bytes[path], case["raw_sha256"], path)
            if self.classes[case["true_index"]] != case["true_class"]:
                raise ValueError(f"Unexpected label index: {case['id']}")
        with ZipFile(source) if source.is_file() else nullcontext() as archive:
            for entry in self.spec["checkpoints"]:
                name = entry["configuration"]
                path = entry["path_in_results_archive"]
                if path != f"outputs/{name}/seed_{self.seed}/best.pt":
                    raise ValueError(f"Unexpected checkpoint path: {path}")
                if archive is not None:
                    raw = archive.read(path)
                else:
                    checkpoint_path = source / path
                    if not checkpoint_path.is_file():
                        checkpoint_path = source / Path(path).relative_to("outputs")
                    raw = checkpoint_path.read_bytes()
                verified_bytes(raw, entry["sha256"], path)
                checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                config = checkpoint["config"]
                architecture, mode = name.rsplit("_", 1)
                augmentation = "camera_aware" if mode == "camera" else "minimal"
                if (
                    config["model"]["name"] != architecture
                    or config["experiment"]["name"] != name
                    or config["experiment"]["seed"] != self.seed
                    or config["data"]["augmentation"] != augmentation
                    or config["data"]["image_size"] != 224
                    or checkpoint["num_classes"] != len(self.classes)
                    or list(checkpoint["class_names"]) != self.classes
                ):
                    raise ValueError(f"Unexpected checkpoint metadata: {path}")
                model = create_model(architecture, len(self.classes))
                model.load_state_dict(checkpoint["model_state"], strict=True)
                self.models[name] = model.eval()

    def case_images(self, case_id):
        case = self.cases[case_id]
        with Image.open(io.BytesIO(self.input_bytes[case["path"]])) as opened:
            original = opened.convert("L")
        if (case["corruption"], case["severity"]) == ("clean", 0):
            return original, original
        if (case["corruption"], case["severity"], case["radius_pixels_before_resize"]) == (
            "gaussian_blur",
            3,
            2.0,
        ):
            return original, original.filter(ImageFilter.GaussianBlur(radius=2.0))
        raise ValueError(f"Unknown demo corruption: {case_id}")

    @torch.inference_mode()
    def predict_case(self, case_id):
        original, image = self.case_images(case_id)
        inputs = self.preprocess(image).unsqueeze(0)
        predictions = {}
        for name, model in self.models.items():
            probabilities = model(inputs).softmax(dim=1)[0]
            indices = probabilities.topk(2).indices.tolist()
            predictions[name] = {
                "probabilities": dict(zip(self.classes, probabilities.tolist(), strict=True)),
                "top1": self.classes[indices[0]],
                "top2": self.classes[indices[1]],
            }
        # The ground truth is joined only after the forward passes, for display.
        return {
            "case": self.cases[case_id],
            "original": original,
            "input": image,
            "predictions": predictions,
        }


def image_html(image, label):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<figure style="display:inline-block;margin:4px 16px 12px 0;text-align:center">'
        f'<img src="data:image/png;base64,{data}" alt="{escape(label)}" width="176" height="176">'
        f"<figcaption>{escape(label)}</figcaption></figure>"
    )


def render_case(result):
    case = result["case"]
    is_clean = case["corruption"] == "clean"
    condition = "Clean" if is_clean else "Gaussian blur s3 (radius 2 px)"
    images = image_html(result["original"], "Input crop" if is_clean else "Original crop")
    if not is_clean:
        images += image_html(result["input"], "Blurred crop before resize")
    rows = []
    for name, prediction in result["predictions"].items():
        top1, top2 = prediction["top1"], prediction["top2"]
        probabilities = prediction["probabilities"]
        verdict = "Correct" if top1 == case["true_class"] else "Incorrect"
        rows.append(
            f"<tr><td>{DISPLAY_NAMES[name]}</td><td><b>{escape(top1)}</b></td>"
            f"<td>{probabilities[top1]:.1%}</td><td>{escape(top2)}</td>"
            f"<td>{probabilities[top2]:.1%}</td><td>{verdict}</td></tr>"
        )
    return (
        f"<h3>{escape(Path(case['path']).name)} — {condition}</h3>"
        f"<p>Ground truth: <strong>{escape(case['true_class'])}</strong> "
        "· fresh CPU inference · seed 43</p>"
        + images
        + "<table><thead><tr><th>Model</th><th>Top 1</th><th>Probability</th>"
        "<th>Top 2</th><th>Probability</th><th>Result</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
