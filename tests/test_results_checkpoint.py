"""Synthetic final-stage checks: no training, network, or real test inference."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from industrial_surface_defects import results_checkpoint as final


def test_registered_matrix_and_fixed_protocol(tmp_path):
    project = Path(__file__).resolve().parents[1]
    configs = final.build_matrix_configs(project, tmp_path)
    assert len(configs) == 12
    assert {key[1] for key in configs} == {42, 43, 44}
    assert all(config.experiment.seed == seed for (_, seed), config in configs.items())
    assert len({config.training for config in configs.values()}) == 1
    assert all(config.training.num_workers == 2 for config in configs.values())
    protocol = final.fixed_protocol()
    assert len(protocol["conditions"]) == 28
    assert protocol["bootstrap"]["repetitions"] == 5000
    assert protocol["failure_selection"]["unique_source_paths"]


def frozen_fixture(tmp_path, monkeypatch):
    root, project = tmp_path / "artifacts", tmp_path / "project"
    root.mkdir()
    (project / "src").mkdir(parents=True)
    (project / "src/example.py").write_text("UNCHANGED = True\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("frozen manifest\n")
    for checkpoint in final.matrix_checkpoints(root).values():
        checkpoint.parent.mkdir(parents=True)
        for name in (
            "best.pt",
            "config.resolved.yaml",
            "history.csv",
            "training_summary.json",
            "runtime.json",
        ):
            (checkpoint.parent / name).write_text("fixture\n")
        (checkpoint.parent / "evaluation/val").mkdir(parents=True)
        for name in ("metrics.json", "predictions.csv"):
            (checkpoint.parent / "evaluation/val" / name).write_text("fixture\n")
    for name in ("experiment_protocol.json", "checkpoint_provenance.json", "validation_audit.json"):
        (root / name).write_text("{}\n")
    monkeypatch.setattr(final, "audit_validation", lambda *args: None)
    final.write_freeze(root, project, {}, manifest)
    return root, project, manifest


def test_freeze_and_explicit_test_gate(tmp_path, monkeypatch):
    root, project, manifest = frozen_fixture(tmp_path, monkeypatch)
    with pytest.raises(PermissionError):
        final.assert_frozen(root, project, manifest, acknowledge_test_use=False)
    assert (
        final.assert_frozen(root, project, manifest, acknowledge_test_use=True)["checkpoint_count"]
        == 12
    )
    with pytest.raises(FileExistsError):
        final.write_freeze(root, project, {}, manifest)


@pytest.mark.parametrize("changed", ["checkpoint", "source", "manifest", "record"])
def test_freeze_detects_mutation(tmp_path, monkeypatch, changed):
    root, project, manifest = frozen_fixture(tmp_path, monkeypatch)
    target = {
        "checkpoint": next(iter(final.matrix_checkpoints(root).values())),
        "source": project / "src/example.py",
        "manifest": manifest,
        "record": root / "protocol_freeze.json",
    }[changed]
    target.write_text("changed\n")
    with pytest.raises(ValueError):
        final.assert_frozen(root, project, manifest, acknowledge_test_use=True)


def test_missing_validation_prevents_freeze(tmp_path):
    with pytest.raises(FileNotFoundError):
        final.write_freeze(tmp_path, tmp_path, {}, tmp_path / "missing_manifest")
    assert not (tmp_path / "protocol_freeze.json").exists()


@pytest.mark.parametrize("corruption,severity", final.TEST_CONDITIONS)
def test_final_dataset_is_test_only_and_deterministic(synthetic_manifest, corruption, severity):
    for path in synthetic_manifest.frame.query("split=='train'").path:
        (synthetic_manifest.root / path).unlink()
    dataset = final.test_dataset(
        synthetic_manifest.path, synthetic_manifest.root, corruption, severity
    )
    assert set(dataset.rows.split) == {"test"}
    torch.testing.assert_close(dataset[0][0], dataset[0][0], atol=0, rtol=0)


def test_unregistered_test_condition_blocked():
    with pytest.raises(ValueError):
        final.test_dataset("unused", "unused", "gaussian_blur", 4)


def test_failure_selection_keeps_unique_images_and_diversity():
    errors = pd.DataFrame(
        [
            {
                "configuration": "a",
                "seed": 42,
                "corruption": "clean",
                "severity": 0,
                "path": "a",
                "true_class": "crazing",
                "confidence": 0.6,
            },
            {
                "configuration": "b",
                "seed": 43,
                "corruption": "blur",
                "severity": 1,
                "path": "a",
                "true_class": "crazing",
                "confidence": 0.99,
            },
            {
                "configuration": "a",
                "seed": 42,
                "corruption": "noise",
                "severity": 1,
                "path": "b",
                "true_class": "scratches",
                "confidence": 0.9,
            },
            {
                "configuration": "a",
                "seed": 42,
                "corruption": "dark",
                "severity": 1,
                "path": "c",
                "true_class": "patches",
                "confidence": 0.8,
            },
        ]
    )
    selected = final.select_failure_candidates(errors, maximum=3)
    assert selected.path.tolist() == ["a", "b", "c"]
    assert selected.iloc[0].corruption == "clean"


def test_export_preserves_prior_archive_and_excludes_raw(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "metrics.json").write_text("{}")
    archive = tmp_path / "results.zip"
    assert len(final.export_bundle(root, archive, status="synthetic")) == 64
    with pytest.raises(FileExistsError):
        final.export_bundle(root, archive, status="synthetic")
    (root / "raw.bmp").write_bytes(b"raw")
    with pytest.raises(ValueError, match="Raw dataset"):
        final.export_bundle(root, tmp_path / "forbidden.zip", status="synthetic")


def matrix_evaluation_fixture(tmp_path, monkeypatch, synthetic_manifest, *, mismatch=False):
    root, project, manifest = frozen_fixture(tmp_path, monkeypatch)
    base = synthetic_manifest.frame
    rows = pd.concat([base.query("split=='val'")] * 60, ignore_index=True)
    rows["path"] = [f"test/{index}.png" for index in range(360)]
    rows["split"] = "test"
    # For this integration test the manifest validator is separately covered by dataset tests.
    monkeypatch.setattr(final, "validate_manifest", lambda _: rows)
    calls = []

    class ToyModel(torch.nn.Module):
        def forward(self, images):
            indices = images[:, 0, 0, 0].round().long()
            return torch.nn.functional.one_hot(indices, 6).float() * 6

    def dataset_factory(_manifest, _root, corruption, severity):
        calls.append((corruption, severity))
        data = []
        for index, row in enumerate(rows.itertuples(index=False)):
            image = torch.full((3, 2, 2), float(row.class_index))
            image[0, 0, 1] = final.TEST_CONDITIONS.index((corruption, severity))
            if mismatch and len(calls) == 29 and index == 0:
                image[0, 1, 1] += 0.25
            data.append((image, row.class_index, row.path))
        return data

    monkeypatch.setattr(final, "test_dataset", dataset_factory)
    monkeypatch.setattr(
        final,
        "DataLoader",
        lambda dataset, **kwargs: torch.utils.data.DataLoader(
            dataset,
            **{**kwargs, "num_workers": 0, "pin_memory": False},
        ),
    )
    monkeypatch.setattr(
        final,
        "torch",
        SimpleNamespace(
            device=lambda _: torch.device("cpu"),
            cuda=SimpleNamespace(empty_cache=lambda: None),
        ),
    )
    monkeypatch.setattr(final, "runtime_metadata", lambda _: {"device_name": "synthetic CPU"})

    def load_model(path, **kwargs):
        name = Path(path).parent.parent.name
        return ToyModel().eval(), {
            "config": {
                "model": {"name": "resnet18" if name.startswith("resnet18") else "efficientnet_b0"},
                "data": {"augmentation": "camera_aware" if name.endswith("camera") else "minimal"},
            }
        }

    monkeypatch.setattr(final, "load_checkpoint_model", load_model)
    monkeypatch.setattr(
        final,
        "benchmark_latency",
        lambda *args, **kwargs: {
            "median_latency_ms": 1,
            "p90_latency_ms": 2,
            "batch_size": 1,
            "warmup_runs": 10,
            "timed_runs": 50,
        },
    )
    monkeypatch.setattr(final, "plot_confusion_matrix", lambda *args, **kwargs: None)
    provenance = {}
    for (name, seed), checkpoint in final.matrix_checkpoints(root).items():
        (checkpoint.parent / "training_summary.json").write_text(
            json.dumps(
                {
                    "best_epoch": 3,
                    "epochs_completed": 13,
                    "elapsed_seconds": 1,
                }
            )
        )
        provenance[f"{name}/{seed}"] = {
            "origin": "synthetic_test_only",
            "original_training_runtime": {},
        }
    (root / "checkpoint_provenance.json").write_text(json.dumps(provenance))
    # Changes to the fixture are still frozen before the synthetic test evaluator starts.
    (root / "protocol_freeze.json").unlink()
    (root / "protocol_freeze.sha256").unlink()
    final.write_freeze(root, project, {}, manifest)
    return root, project, manifest, calls


def test_all_336_test_conditions_are_audited_and_exported(
    tmp_path, monkeypatch, synthetic_manifest
):
    root, project, manifest, calls = matrix_evaluation_fixture(
        tmp_path, monkeypatch, synthetic_manifest
    )
    result = final.evaluate_test_matrix(
        root, project, manifest, "unused", acknowledge_test_use=True
    )
    assert len(result) == len(calls) == 336
    assert (result.macro_f1 == 1).all()
    audit = final.read_json(root / "test_evaluation_audit.json")
    assert audit["prediction_rows"] == 120960
    assert len(list(root.glob("outputs/*/*/evaluation/**/independent_audit.json"))) == 336
    observed = {}
    monkeypatch.setattr(
        final,
        "compare",
        lambda specs, **kwargs: observed.update({"specifications": len(specs), **kwargs}),
    )
    final.aggregate_results(root)
    assert observed["specifications"] == 12 and observed["repetitions"] == 5000
    assert observed["registered_design"]
    assert len(pd.read_csv(root / "runtime_summary.csv")) == 12
    assert len(pd.read_csv(root / "clean_scores_by_configuration.csv")) == 4
    final.failure_materials(root, "unused")
    assert final.read_json(root / "failure_selection.json")["selected_distinct_images"] == 0
    with pytest.raises(FileExistsError):
        final.evaluate_test_matrix(root, project, manifest, "unused", acknowledge_test_use=True)


def test_changed_tensor_blocks_final_completion(tmp_path, monkeypatch, synthetic_manifest):
    root, project, manifest, _ = matrix_evaluation_fixture(
        tmp_path, monkeypatch, synthetic_manifest, mismatch=True
    )
    with pytest.raises(ValueError, match="different test tensors"):
        final.evaluate_test_matrix(root, project, manifest, "unused", acknowledge_test_use=True)
    assert not (root / "test_evaluation_audit.json").exists()
