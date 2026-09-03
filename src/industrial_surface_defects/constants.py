"""Dataset constants shared by training, evaluation, and reporting."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
)

CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}

CLASS_ALIASES: dict[str, str] = {
    "cr": "crazing",
    "crazing": "crazing",
    "cracks": "crazing",
    "in": "inclusion",
    "inclusion": "inclusion",
    "inclusions": "inclusion",
    "pa": "patches",
    "patch": "patches",
    "patches": "patches",
    "ps": "pitted_surface",
    "pi": "pitted_surface",
    "pitted": "pitted_surface",
    "pitted_surface": "pitted_surface",
    "pittedsurface": "pitted_surface",
    "rs": "rolled_in_scale",
    "rolled_in_scale": "rolled_in_scale",
    "rolledinscale": "rolled_in_scale",
    "sc": "scratches",
    "scratch": "scratches",
    "scratches": "scratches",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
