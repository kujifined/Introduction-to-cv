"""Ensure the standalone presentation uses the frozen inputs and preprocessing."""

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from PIL import Image, ImageFilter

from industrial_surface_defects.transforms import build_transform

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("live_demo", ROOT / "notebooks/live_demo.py")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_demo_assets_and_checkpoint_hashes_match_frozen_experiment():
    assets = ROOT / "notebooks/demo_assets"
    spec = json.loads((assets / "demo_cases.json").read_text())
    freeze = json.loads((ROOT / "results/final/protocol_freeze.json").read_text())
    manifest_sha = hashlib.sha256((ROOT / "data/manifests/neu_cls_v2.csv").read_bytes()).hexdigest()
    assert manifest_sha == freeze["manifest_sha256"]
    for checkpoint in spec["checkpoints"]:
        assert checkpoint["sha256"] == freeze["files"][checkpoint["path_in_results_archive"]]
    for case in spec["cases"]:
        content = (assets / "demo_inputs" / case["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == case["raw_sha256"]


def test_standalone_preprocessing_matches_shared_pipeline():
    spec = json.loads((ROOT / "notebooks/demo_assets/demo_cases.json").read_text())
    for case in spec["cases"]:
        with Image.open(ROOT / "notebooks/demo_assets/demo_inputs" / case["path"]) as opened:
            image = opened.convert("L")
        if case["corruption"] == "gaussian_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        actual = DEMO.clean_preprocessing()(image)
        expected = build_transform(224, "minimal", train=False)(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weight_discovery_uses_documented_locations(tmp_path, monkeypatch):
    monkeypatch.delenv("ICV_DEMO_WEIGHTS", raising=False)
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    assert DEMO.checkpoint_source(notebooks) == checkpoints
    download = tmp_path / "results_download/icv_final_results.zip"
    download.parent.mkdir()
    download.write_bytes(b"placeholder")
    assert DEMO.checkpoint_source(notebooks) == download
    monkeypatch.setenv("ICV_DEMO_WEIGHTS", str(tmp_path / "custom.zip"))
    assert DEMO.checkpoint_source(notebooks) == tmp_path / "custom.zip"
