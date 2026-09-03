"""Model factory for the preregistered architecture comparison."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(name: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable_only:
        parameters = model.parameters()
    return sum(parameter.numel() for parameter in parameters)


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    if name == "resnet18":
        return model.layer4[-1].conv2
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported model: {name}")


def load_checkpoint_model(
    checkpoint_path: str,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["config"]["model"]["name"]
    model = build_model(model_name, checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
