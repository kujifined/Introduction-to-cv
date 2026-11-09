"""Check artifact trust boundaries and the inference-only presentation entry point."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from industrial_surface_defects import inference
from industrial_surface_defects.constants import CLASS_NAMES


def _importer():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_final_artifacts.py"
    spec = importlib.util.spec_from_file_location("artifact_importer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restore_verifies_zip_and_never_overwrites_different_weights(tmp_path):
    archive = tmp_path / "results.zip"
    content = b"trusted checkpoint"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("outputs/test/seed_43/best.pt", content)
    item = {
        "seed": 43,
        "configuration": "test",
        "path_in_results_archive": "outputs/test/seed_43/best.pt",
        "local_path": "checkpoints/test/seed_43/best.pt",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    metadata = {"archive_sha256": inference.file_sha256(archive), "checkpoints": [item]}
    target = tmp_path / "results/final/ARTIFACTS.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(metadata))
    importer = _importer()
    assert importer.restore(archive, tmp_path) == 1
    assert importer.restore(archive, tmp_path) == 1
    (tmp_path / item["local_path"]).write_bytes(b"different")
    with pytest.raises(FileExistsError, match="differs"):
        importer.restore(archive, tmp_path)
    metadata["archive_sha256"] = "0" * 64
    target.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="ZIP SHA-256"):
        importer.restore(archive, tmp_path)


def test_restore_rejects_path_traversal(tmp_path):
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("dummy", b"")
    metadata = {
        "archive_sha256": inference.file_sha256(archive),
        "checkpoints": [
            {"seed": 43, "local_path": "../escape", "path_in_results_archive": "dummy"}
        ],
    }
    target = tmp_path / "results/final/ARTIFACTS.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsafe"):
        _importer().restore(archive, tmp_path)


def test_prediction_depends_on_forward_pass_not_reference_data():
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.copy_(torch.arange(6.0))
    outputs = inference.predict_image({"test": model.eval()}, Image.new("L", (200, 200)))
    assert outputs[0]["predicted_class"] == "scratches"
    assert outputs[0]["top2_class"] == "rolled_in_scale"
    assert sum(outputs[0]["probabilities"].values()) == pytest.approx(1)
    with torch.no_grad():
        model[-1].bias[0] = 20
    assert (
        inference.predict_image({"test": model}, Image.new("L", (200, 200)))[0]["predicted_class"]
        == "crazing"
    )
    with pytest.raises(ValueError, match="severity 0"):
        inference.predict_image({"test": model}, Image.new("L", (200, 200)), severity=3)


def test_checkpoint_hash_and_class_order_are_required_before_model_build(tmp_path, monkeypatch):
    classes = list(CLASS_NAMES)
    classes.reverse()
    checkpoint = {"class_names": classes, "config": {}, "num_classes": 6}
    items = []
    for index in range(4):
        path = tmp_path / f"checkpoints/{index}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, path)
        items.append(
            {
                "configuration": str(index),
                "seed": 43,
                "local_path": str(path.relative_to(tmp_path)),
                "sha256": inference.file_sha256(path),
            }
        )
    target = tmp_path / "results/final/ARTIFACTS.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"checkpoints": items, "manifest_sha256": "test"}))

    def forbidden(*args, **kwargs):
        pytest.fail("Model must not be built from invalid checkpoint metadata")

    monkeypatch.setattr(inference, "build_model", forbidden)
    with pytest.raises(ValueError, match="metadata"):
        inference.load_models(tmp_path)
    (tmp_path / items[0]["local_path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        inference.load_models(tmp_path)
