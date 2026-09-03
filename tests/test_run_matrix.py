"""Safety checks for resuming the registered training matrix."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from industrial_surface_defects.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_run_matrix_module() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_matrix.py"
    specification = importlib.util.spec_from_file_location("run_matrix_script", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


run_matrix = _load_run_matrix_module()


def _temporary_config(tmp_path: Path):
    config = load_config(PROJECT_ROOT / "configs" / "resnet18_minimal.yaml")
    experiment = replace(
        config.experiment,
        name="resume_probe",
        output_root=str(tmp_path / "outputs"),
    )
    return replace(config, experiment=experiment)


def test_skip_existing_accepts_only_a_completed_run(tmp_path: Path, monkeypatch) -> None:
    config = _temporary_config(tmp_path)
    run_dir = Path(config.experiment.output_root) / config.experiment.name / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "training_summary.json").write_text("{}\n", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(run_matrix, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(run_matrix, "train", lambda *args, **kwargs: calls.append((args, kwargs)))

    run_matrix.main(["--config", "unused.yaml", "--seeds", "42", "--skip-existing"])

    assert calls == []


def test_skip_existing_rejects_a_partial_run(tmp_path: Path, monkeypatch) -> None:
    config = _temporary_config(tmp_path)
    run_dir = Path(config.experiment.output_root) / config.experiment.name / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "runtime.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(run_matrix, "load_config", lambda *_args, **_kwargs: config)

    with pytest.raises(RuntimeError, match="Refusing to skip incomplete run"):
        run_matrix.main(["--config", "unused.yaml", "--seeds", "42", "--skip-existing"])
