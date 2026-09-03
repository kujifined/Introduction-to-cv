"""Minimal and physically motivated image transformations."""

from __future__ import annotations

import io
import math
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .constants import IMAGENET_MEAN, IMAGENET_STD


def _motion_blur(image: Image.Image, length: int, *, horizontal: bool) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    padding = length // 2
    if horizontal:
        padded = np.pad(array, ((0, 0), (padding, padding)), mode="reflect")
        blurred = sum(padded[:, offset : offset + array.shape[1]] for offset in range(length))
    else:
        padded = np.pad(array, ((padding, padding), (0, 0)), mode="reflect")
        blurred = sum(padded[offset : offset + array.shape[0], :] for offset in range(length))
    return Image.fromarray(np.clip(blurred / length, 0, 255).astype(np.uint8), mode="L")


class CameraAwareAugmentation:
    """Versioned acquisition-inspired transform with at most two photometric effects."""

    version = "camera_aware_v1"
    probabilities = {
        "exposure": 0.50,
        "contrast": 0.30,
        "gamma": 0.30,
        "noise": 0.35,
        "blur": 0.25,
        "jpeg": 0.25,
    }

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        if random.random() < 0.50:
            angle = random.uniform(-10.0, 10.0)
            maximum_translation = round(0.05 * min(image.size))
            translate = (
                random.randint(-maximum_translation, maximum_translation),
                random.randint(-maximum_translation, maximum_translation),
            )
            scale = random.uniform(0.90, 1.10)
            fill = int(np.median(np.asarray(image)))
            image = vision_functional.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=fill,
            )

        selected = [
            name
            for name, probability in self.probabilities.items()
            if random.random() < probability
        ]
        if len(selected) > 2:
            selected = random.sample(selected, 2)
        selected_set = set(selected)

        if "exposure" in selected_set:
            factor = math.exp(random.uniform(math.log(0.70), math.log(1.40)))
            image = ImageEnhance.Brightness(image).enhance(factor)
        if "contrast" in selected_set:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.20))
        if "gamma" in selected_set:
            gamma = random.uniform(0.80, 1.25)
            array = np.asarray(image, dtype=np.float32) / 255.0
            image = Image.fromarray(
                np.clip(array**gamma * 255.0, 0, 255).round().astype(np.uint8), mode="L"
            )
        if "noise" in selected_set:
            array = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0)
            peak = math.exp(random.uniform(math.log(40.0), math.log(120.0)))
            sigma = random.uniform(0.005, 0.025)
            array = torch.poisson(array * peak) / peak + torch.randn_like(array) * sigma
            image = Image.fromarray(
                (array.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8), mode="L"
            )
        if "blur" in selected_set:
            if random.random() < 0.50:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.20, 1.20)))
            else:
                length = random.choice((3, 5, 7))
                image = _motion_blur(image, length, horizontal=random.random() < 0.50)
        if "jpeg" in selected_set:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(50, 95), optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert("L")
        return image


def build_transform(image_size: int, augmentation: str, *, train: bool) -> transforms.Compose:
    """Build preprocessing; stochastic operations are enabled only for training."""
    if augmentation not in {"minimal", "camera_aware"}:
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    operations: list = []
    if train and augmentation == "camera_aware":
        operations.append(CameraAwareAugmentation())
    operations.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )
    operations.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(operations)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0, 1)
