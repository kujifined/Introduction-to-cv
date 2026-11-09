"""Synthetic checks for the weekly notebook, without training or network access."""

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from industrial_surface_defects import weekly_validation as weekly
from industrial_surface_defects.constants import CLASS_NAMES
from industrial_surface_defects.metrics import compute_metrics, prediction_table


def test_four_configs_change_only_architecture_and_augmentation():
    root = Path(__file__).resolve().parents[1]
    configs = weekly.build_configs(root, "/unused/outputs")
    assert len(configs) == 4
    assert all(item.experiment.seed == 43 for item in configs.values())
    assert all(item.training == configs["resnet18_minimal"].training for item in configs.values())
    for name, config in configs.items():
        assert config.data.augmentation == (
            "camera_aware" if name.endswith("camera") else "minimal"
        )
        assert config.model.pretrained


def test_pilot_has_ten_predefined_validation_conditions():
    protocol = weekly.pilot_protocol()
    assert len(protocol["conditions"]) == 10
    assert protocol["split"] == "val"
    assert protocol["brightness_factors"] == [0.8, 0.65, 0.55]
    assert protocol["gaussian_blur_radius_pixels_before_resize"] == [0.5, 1.0, 2.0]
    assert not protocol["test_evaluation_enabled"]
    assert not protocol["pilot_used_for_checkpoint_or_hyperparameter_selection"]


@pytest.mark.parametrize("condition", weekly.CONDITIONS)
def test_dataset_never_reads_test_and_views_repeat(synthetic_manifest, condition):
    rows = synthetic_manifest.frame
    for path in rows.loc[rows.split == "test", "path"]:
        (synthetic_manifest.root / path).unlink()
    dataset = weekly.make_dataset(synthetic_manifest.path, synthetic_manifest.root, condition)
    assert set(dataset.rows.split) == {"val"}
    first, target, path = dataset[0]
    second, other_target, other_path = dataset[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert (target, path) == (other_target, other_path)


@pytest.mark.parametrize("condition", ["test", "jpeg_s1", "gaussian_blur_s4", "brightness_dark_s0"])
def test_pilot_rejects_unregistered_conditions(condition):
    with pytest.raises(ValueError, match="fixed validation"):
        weekly.make_dataset("unused", "unused", condition)


def predictions_fixture():
    targets = np.repeat(np.arange(6), 2)
    scores = np.full((12, 6), 0.02)
    predicted = targets.copy()
    predicted[0] = 1
    scores[np.arange(12), predicted] = 0.9
    paths = [f"example_{index}.bmp" for index in range(12)]
    table = prediction_table(paths, targets, scores)
    expected = pd.DataFrame({"path": paths, "class_index": targets})
    metrics, _, _, _ = compute_metrics(targets, scores)
    return table, expected, metrics


def test_manual_metrics_and_class_confusion():
    table, expected, metrics = predictions_fixture()
    manual, confusion = weekly.audit_predictions(table, expected, metrics)
    assert manual["accuracy"] == pytest.approx(11 / 12)
    assert manual["macro_f1"] == pytest.approx((2 / 3 + 4 / 5 + 4) / 6)
    assert confusion[0, 1] == 1


@pytest.mark.parametrize("problem", ["duplicate", "label", "prediction", "nan", "metric", "flag"])
def test_audit_detects_corrupted_predictions(problem):
    table, expected, metrics = predictions_fixture()
    if problem == "duplicate":
        table.loc[0, "path"] = table.loc[1, "path"]
    elif problem == "label":
        table.loc[0, "true_index"] = 2
    elif problem == "prediction":
        table.loc[0, "predicted_index"] = 3
    elif problem == "nan":
        table.loc[0, "prob_crazing"] = np.nan
    elif problem == "metric":
        metrics["macro_f1"] = 0.4
    else:
        table.loc[0, "correct"] = True
    with pytest.raises(ValueError):
        weekly.audit_predictions(table, expected, metrics)


def test_fingerprint_changes_when_input_tensor_changes():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 6)).eval()
    images = torch.zeros(2, 3, 2, 2)
    loader = [(images, torch.tensor([0, 1]), ["a", "b"])]
    first = weekly.predict_and_fingerprint(model, loader, torch.device("cpu"))
    second = weekly.predict_and_fingerprint(model, loader, torch.device("cpu"))
    assert first[-1] == second[-1]
    images[0, 0, 0, 0] = 1
    third = weekly.predict_and_fingerprint(model, loader, torch.device("cpu"))
    assert first[-1] != third[-1]


def test_runtime_comparison_allows_python_patch_but_rejects_cuda_changes():
    historical = {
        "python": "3.13.15",
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "cuda": "12.8",
        "device_name": "Tesla T4",
        "numpy": "2.1.3",
        "scikit_learn": "1.6.1",
    }
    current = {**historical, "python": "3.13.16"}
    assert weekly.environment_differences(current, historical) == {}
    current["cuda"] = "13.0"
    assert "cuda" in weekly.environment_differences(current, historical)
    current["python"] = "3.12.8"
    assert "python_series" in weekly.environment_differences(current, historical)


def test_control_import_checks_hashes_and_ignores_unknown_members(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    with zipfile.ZipFile(source / "icv_minimal_controls_seed43.zip", "w") as archive:
        archive.writestr("outputs/a/b/best.pt", b"expected")
        archive.writestr("../../escape", b"ignored")
    expected = {"files": {"outputs/a/b/best.pt": hashlib.sha256(b"expected").hexdigest()}}
    destination = tmp_path / "controls"
    result = weekly.import_controls(source, destination, expected)
    assert result["hashes_verified"]
    assert (destination / "outputs/a/b/best.pt").read_bytes() == b"expected"
    assert not (tmp_path / "escape").exists()
    expected["files"]["outputs/a/b/best.pt"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        weekly.import_controls(source, tmp_path / "bad", expected)


def training_fixture(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = weekly.build_configs(root, tmp_path)["resnet18_camera"]
    config = replace(config, data=replace(config.data, root="unused"))
    scores = [0.5, 0.8, 1.0] + [1.0] * 10
    pd.DataFrame({"epoch": range(1, 14), "val_macro_f1": scores}).to_csv(
        tmp_path / "history.csv",
        index=False,
    )
    (tmp_path / "config.resolved.yaml").write_text(yaml.safe_dump(config.to_dict()))
    summary = {
        "status": "complete",
        "best_epoch": 3,
        "epochs_completed": 13,
        "test_split_accessed": False,
        "best_validation_macro_f1": 1.0,
        "elapsed_seconds": 50,
        "model_parameters": 11179590,
    }
    (tmp_path / "training_summary.json").write_text(json.dumps(summary))
    checkpoint = {
        "config": config.to_dict(),
        "class_names": list(CLASS_NAMES),
        "num_classes": 6,
        "manifest_sha256": weekly.MANIFEST_SHA256,
        "model_state": {"weight": torch.ones(1)},
        "best_epoch": 3,
        "validation_metrics": {"macro_f1": 1.0},
    }
    path = tmp_path / "best.pt"
    torch.save(checkpoint, path)
    return path, config, checkpoint


def test_training_audit_accepts_earliest_tie(tmp_path):
    path, config, _ = training_fixture(tmp_path)
    assert weekly.verify_training(path, config)["best_epoch"] == 3


@pytest.mark.parametrize("problem", ["tie", "manifest", "augmentation", "nan", "short_history"])
def test_training_audit_rejects_wrong_checkpoint_and_incomplete_run(tmp_path, problem):
    path, config, checkpoint = training_fixture(tmp_path)
    if problem == "tie":
        checkpoint["best_epoch"] = 4
    elif problem == "manifest":
        checkpoint["manifest_sha256"] = "bad"
    elif problem == "augmentation":
        checkpoint["config"]["data"]["augmentation"] = "minimal"
    elif problem == "nan":
        checkpoint["model_state"]["weight"][0] = float("nan")
    else:
        frame = pd.read_csv(tmp_path / "history.csv").head(4)
        frame.to_csv(tmp_path / "history.csv", index=False)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        weekly.verify_training(path, config)


def mock_four_way_evaluation(monkeypatch, synthetic_manifest, tmp_path, *, mismatch=False):
    frame = synthetic_manifest.frame
    rows = frame.loc[frame.split == "val"]
    expanded = pd.concat([rows] * 30, ignore_index=True)
    expanded["path"] = [f"validation/{index}.png" for index in range(180)]
    frame = pd.concat([frame.loc[frame.split != "val"], expanded], ignore_index=True)
    manifest = tmp_path / "expanded.csv"
    frame.to_csv(manifest, index=False)
    monkeypatch.setattr(weekly, "MANIFEST_SHA256", weekly.sha256(manifest))

    class ToyModel(torch.nn.Module):
        def forward(self, images):
            indices = images[:, 0, 0, 0].round().long()
            return torch.nn.functional.one_hot(indices, num_classes=6).float() * 6

    dataset_calls = []

    def dataset_factory(_manifest, _root, condition):
        dataset_calls.append(condition)
        data = []
        for index, row in enumerate(expanded.itertuples(index=False)):
            image = torch.full((3, 2, 2), float(row.class_index))
            image[0, 0, 1] = weekly.CONDITIONS.index(condition)
            if mismatch and len(dataset_calls) == 11 and index == 0:
                image[0, 1, 1] += 0.25
            data.append((image, row.class_index, row.path))
        return data

    monkeypatch.setattr(weekly, "make_dataset", dataset_factory)
    monkeypatch.setattr(
        weekly,
        "DataLoader",
        lambda dataset, **kwargs: torch.utils.data.DataLoader(
            dataset,
            **{**kwargs, "num_workers": 0},
        ),
    )
    monkeypatch.setattr(
        weekly,
        "torch",
        SimpleNamespace(
            device=lambda _: torch.device("cpu"),
            cuda=SimpleNamespace(empty_cache=lambda: None),
        ),
    )
    monkeypatch.setattr(weekly, "runtime_metadata", lambda _: {"test_runtime": True})
    monkeypatch.setattr(
        weekly,
        "load_checkpoint_model",
        lambda *args, **kwargs: (
            ToyModel().eval(),
            {"validation_metrics": {"macro_f1": 1.0}},
        ),
    )
    monkeypatch.setattr(
        weekly,
        "verify_training",
        lambda *args: {
            "best_epoch": 3,
            "epochs_completed": 13,
            "elapsed_seconds": 1,
        },
    )
    monkeypatch.setattr(weekly, "count_parameters", lambda _: 11179590)
    monkeypatch.setattr(weekly, "PARAMETERS", {"resnet18": 11179590, "efficientnet_b0": 11179590})
    monkeypatch.setattr(
        weekly,
        "benchmark_latency",
        lambda *args, **kwargs: {
            "median_latency_ms": 1,
            "p90_latency_ms": 2,
        },
    )
    monkeypatch.setattr(
        weekly,
        "plot_confusion_matrix",
        lambda matrix, path, **kwargs: Path(path).write_bytes(b"synthetic plot"),
    )
    root = Path(__file__).resolve().parents[1]
    configs = weekly.build_configs(root, tmp_path / "unused")
    checkpoints, provenance = {}, {}
    for name in weekly.CONFIGURATIONS:
        checkpoints[name] = tmp_path / f"{name}.pt"
        checkpoints[name].write_bytes(b"synthetic checkpoint")
        provenance[name] = {"training_origin": "synthetic_test_only"}
    destination = tmp_path / "artifacts"
    destination.mkdir()
    return (
        destination,
        dataset_calls,
        (
            checkpoints,
            configs,
            destination,
            manifest,
            "unused",
            provenance,
        ),
    )


def test_all_forty_evaluations_export_and_audit(monkeypatch, synthetic_manifest, tmp_path):
    destination, calls, arguments = mock_four_way_evaluation(
        monkeypatch,
        synthetic_manifest,
        tmp_path,
    )
    comparison = weekly.evaluate_four(*arguments)
    assert len(comparison) == 40
    assert len(calls) == 40
    assert (comparison.macro_f1 == 1).all()
    assert (comparison.macro_f1_drop_from_clean == 0).all()
    assert len(list(destination.rglob("predictions.csv"))) == 40
    assert len(list(destination.rglob("independent_audit.json"))) == 40
    effects = pd.read_csv(destination / "augmentation_effects.csv")
    assert len(effects) == 20
    assert (effects.camera_minus_minimal_macro_f1 == 0).all()
    assert pd.read_csv(destination / "observed_validation_errors.csv").empty
    assert json.loads((destination / "validation_audit.json").read_text())["status"] == "pass"


def test_four_way_evaluation_blocks_changed_input_tensors(
    monkeypatch, synthetic_manifest, tmp_path
):
    destination, _, arguments = mock_four_way_evaluation(
        monkeypatch,
        synthetic_manifest,
        tmp_path,
        mismatch=True,
    )
    with pytest.raises(ValueError, match="different validation tensors"):
        weekly.evaluate_four(*arguments)
    assert not (destination / "validation_audit.json").exists()
