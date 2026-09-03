"""The matrix evaluator must only consume the preregistered 4 x 3 runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from industrial_surface_defects.compare import REGISTERED_CONFIGURATIONS, REGISTERED_SEEDS

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_matrix.py"
SPEC = importlib.util.spec_from_file_location("evaluate_matrix_script", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
EVALUATE_MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE_MATRIX)
main = EVALUATE_MATRIX.main
registered_checkpoints = EVALUATE_MATRIX.registered_checkpoints


def test_registered_checkpoints_ignores_smoke_and_suffix_runs(tmp_path: Path) -> None:
    expected: set[Path] = set()
    for configuration in REGISTERED_CONFIGURATIONS:
        for seed in REGISTERED_SEEDS:
            checkpoint = tmp_path / configuration / f"seed_{seed}" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            expected.add(checkpoint)

    smoke = tmp_path / "resnet18_minimal" / "seed_42_smoke" / "best.pt"
    smoke.parent.mkdir(parents=True)
    smoke.touch()
    unrelated = tmp_path / "smallcnn" / "seed_42" / "best.pt"
    unrelated.parent.mkdir(parents=True)
    unrelated.touch()

    planned = registered_checkpoints(tmp_path)

    assert len(planned) == 12
    assert {checkpoint for _, _, checkpoint in planned} == expected


def test_registered_checkpoints_requires_the_complete_matrix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="requires all 12 primary checkpoints"):
        registered_checkpoints(tmp_path)


def test_matrix_rejects_robustness_during_validation_before_reading_checkpoints(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="use --split test"):
        main(
            [
                "--outputs-root",
                str(tmp_path),
                "--split",
                "val",
                "--with-robustness",
            ]
        )


def test_test_matrix_runs_registered_robustness_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for configuration in REGISTERED_CONFIGURATIONS:
        for seed in REGISTERED_SEEDS:
            checkpoint = tmp_path / configuration / f"seed_{seed}" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

    calls = {"evaluate": 0, "robustness": 0, "aggregate": 0, "compare": 0}

    def fake_evaluate(*_args, **_kwargs):
        calls["evaluate"] += 1

    def fake_evaluate_robustness(*_args, **_kwargs):
        calls["robustness"] += 1

    def fake_aggregate(outputs_root, *, output_dir):
        assert Path(outputs_root) == tmp_path
        assert Path(output_dir).name == "robustness_comparison_test"
        calls["aggregate"] += 1

    def fake_compare(*_args, **kwargs):
        assert kwargs["registered_design"] is True
        calls["compare"] += 1

    monkeypatch.setattr(EVALUATE_MATRIX, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        EVALUATE_MATRIX,
        "evaluate_robustness",
        fake_evaluate_robustness,
    )
    monkeypatch.setattr(
        EVALUATE_MATRIX,
        "aggregate_registered_robustness",
        fake_aggregate,
    )
    monkeypatch.setattr(EVALUATE_MATRIX, "compare", fake_compare)

    main(
        [
            "--outputs-root",
            str(tmp_path),
            "--split",
            "test",
            "--acknowledge-test-use",
            "--with-robustness",
            "--skip-latency",
        ]
    )

    assert calls == {"evaluate": 12, "robustness": 12, "aggregate": 1, "compare": 1}
