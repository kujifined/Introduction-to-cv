"""Network factory smoke tests that never download pretrained weights."""

from __future__ import annotations

import pytest
import torch

from industrial_surface_defects.models import build_model, count_parameters


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0"])
def test_model_output_shape_without_pretrained_download(name: str) -> None:
    model = build_model(name, num_classes=6, pretrained=False).eval()
    inputs = torch.zeros(1, 3, 64, 64)

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (1, 6)
    assert torch.isfinite(outputs).all()
    assert count_parameters(model) > 1_000_000
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


def test_model_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("mystery_net", num_classes=6, pretrained=False)
