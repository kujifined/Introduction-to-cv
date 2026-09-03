"""Deterministic synthetic acquisition shifts for robustness evaluation."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .constants import IMAGENET_MEAN, IMAGENET_STD
from .data import validate_manifest

CORRUPTION_NAMES: tuple[str, ...] = (
    "brightness_dark",
    "brightness_bright",
    "contrast",
    "gamma",
    "shot_read_noise",
    "gaussian_blur",
    "motion_blur_horizontal",
    "motion_blur_vertical",
    "jpeg",
)

SEVERITY_VALUES: dict[str, tuple[float, float, float]] = {
    "brightness_dark": (0.80, 0.65, 0.55),
    "brightness_bright": (1.20, 1.50, 1.80),
    "contrast": (0.80, 0.60, 0.40),
    "gamma": (1.20, 1.50, 2.00),
    "gaussian_blur": (0.50, 1.00, 2.00),
    "motion_blur_horizontal": (3.0, 5.0, 7.0),
    "motion_blur_vertical": (3.0, 5.0, 7.0),
    "jpeg": (70.0, 40.0, 20.0),
}

SHOT_READ_NOISE_SETTINGS: tuple[tuple[float, float], ...] = (
    (120.0, 0.005),
    (70.0, 0.0125),
    (40.0, 0.025),
)


def _motion_blur(array: np.ndarray, length: int, *, horizontal: bool) -> np.ndarray:
    padding = length // 2
    if horizontal:
        padded = np.pad(array, ((0, 0), (padding, padding)), mode="reflect")
        return sum(padded[:, offset : offset + array.shape[1]] for offset in range(length)) / length
    padded = np.pad(array, ((padding, padding), (0, 0)), mode="reflect")
    return sum(padded[offset : offset + array.shape[0], :] for offset in range(length)) / length


def _seed_for(path: str, corruption: str, severity: int) -> int:
    digest = hashlib.sha256(f"{path}\0{corruption}\0{severity}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def apply_corruption(
    image: Image.Image,
    corruption: str,
    severity: int,
    *,
    sample_key: str,
) -> Image.Image:
    if corruption not in CORRUPTION_NAMES:
        raise ValueError(f"Unknown corruption: {corruption}")
    if severity not in {1, 2, 3}:
        raise ValueError("severity must be 1, 2, or 3")
    grayscale = image.convert("L")

    if corruption == "shot_read_noise":
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
        peak, read_sigma = SHOT_READ_NOISE_SETTINGS[severity - 1]
        rng = np.random.default_rng(_seed_for(sample_key, corruption, severity))
        shot_noise = rng.poisson(array * peak) / peak
        transformed = np.clip(
            shot_noise + rng.normal(0.0, read_sigma, array.shape),
            0,
            1,
        )
        return Image.fromarray((transformed * 255.0).round().astype(np.uint8), mode="L")

    value = SEVERITY_VALUES[corruption][severity - 1]

    if corruption in {"brightness_dark", "brightness_bright"}:
        return ImageEnhance.Brightness(grayscale).enhance(value)
    if corruption == "contrast":
        return ImageEnhance.Contrast(grayscale).enhance(value)
    if corruption == "gamma":
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
        transformed = np.clip(array**value * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(transformed, mode="L")
    if corruption == "gaussian_blur":
        return grayscale.filter(ImageFilter.GaussianBlur(radius=value))
    if corruption in {"motion_blur_horizontal", "motion_blur_vertical"}:
        array = np.asarray(grayscale, dtype=np.float32)
        transformed = _motion_blur(
            array,
            int(value),
            horizontal=corruption == "motion_blur_horizontal",
        )
        return Image.fromarray(np.clip(transformed, 0, 255).round().astype(np.uint8), mode="L")
    if corruption == "jpeg":
        buffer = io.BytesIO()
        grayscale.save(buffer, format="JPEG", quality=int(value), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("L")
    raise AssertionError("Unreachable corruption branch")


class CorruptedManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        *,
        split: str,
        image_size: int,
        corruption: str,
        severity: int,
    ) -> None:
        manifest = validate_manifest(manifest_path)
        self.rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Manifest contains no rows for split={split!r}")
        self.data_root = Path(data_root)
        self.corruption = corruption
        self.severity = severity
        self.postprocess = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        relative_path = str(row["path"])
        with Image.open(self.data_root / relative_path) as source:
            image = apply_corruption(
                source,
                self.corruption,
                self.severity,
                sample_key=relative_path,
            )
        return self.postprocess(image), int(row["class_index"]), relative_path
