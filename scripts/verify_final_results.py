"""Run the YAML-configured independent audit of the frozen final bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from audit.aggregation import audit_aggregations
from audit.bootstrap import audit_bootstrap
from audit.bundle import audit_bundle
from audit.common import Audit
from audit.config import load_config, load_contrasts
from audit.failure_cases import audit_failure_materials
from audit.manifest import audit_manifest
from audit.predictions import audit_predictions_and_metrics
from audit.source_data import audit_source_data


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "audit" / "final_audit.yaml",
        help="YAML audit profile (default: configs/audit/final_audit.yaml)",
    )
    parser.add_argument("--results-root", type=Path, help="Override paths.results_root")
    parser.add_argument("--failure-root", type=Path, help="Override paths.failure_root")
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="Override paths.source_archive",
    )
    parser.add_argument("--data-root", type=Path, help="Override paths.data_root")
    parser.add_argument("--output-dir", type=Path, help="Override paths.output_dir")
    parser.add_argument("--tolerance", type=float, help="Override metrics.tolerance")
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    for argument, key in (
        (args.results_root, "results_root"),
        (args.failure_root, "failure_root"),
        (args.source_archive, "source_archive"),
        (args.data_root, "data_root"),
        (args.output_dir, "output_dir"),
    ):
        if argument is not None:
            config["paths"][key] = argument.resolve()
    if args.tolerance is not None:
        config["metrics"]["tolerance"] = float(args.tolerance)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_config(config_path, project_root)
    apply_cli_overrides(config, args)
    contrasts = load_contrasts(config["paths"]["contrasts"])

    paths = config["paths"]
    expectations = config["expectations"]
    experiment = config["experiment"]
    metrics = config["metrics"]
    results_root = paths["results_root"].resolve()
    failure_root = paths["failure_root"].resolve()
    output_dir = paths["output_dir"].resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_checkpoints = len(experiment["configurations"]) * len(experiment["seeds"])
    if expected_checkpoints != int(expectations["checkpoints"]):
        raise ValueError(
            "YAML checkpoint expectation does not match configurations x seeds: "
            f"{expectations['checkpoints']} != {expected_checkpoints}"
        )

    audit = Audit(float(metrics["tolerance"]))
    print("[1/7] Verifying bundle integrity", flush=True)
    bundle_stats = audit_bundle(
        results_root,
        int(expectations["bundle_files"]),
        audit,
    )
    print("[2/7] Verifying manifest and split", flush=True)
    manifest, class_names = audit_manifest(
        results_root,
        failure_root,
        expectations,
        audit,
    )
    print("[3/7] Verifying source archive and extracted BMP files", flush=True)
    source_stats = audit_source_data(
        paths.get("source_archive"),
        paths.get("data_root"),
        manifest,
        expectations,
        audit,
    )
    print("[4/7] Recomputing prediction metrics", flush=True)
    scores, clean_predictions = audit_predictions_and_metrics(
        results_root,
        manifest,
        class_names,
        experiment,
        expectations,
        metrics,
        audit,
    )
    print("[5/7] Recomputing clean and robustness aggregations", flush=True)
    clean_summary, robustness_summary = audit_aggregations(
        results_root,
        scores,
        experiment,
        list(metrics["summary_names"]),
        contrasts,
        audit,
    )
    print("[6/7] Recomputing registered paired bootstrap", flush=True)
    bootstrap = audit_bootstrap(
        results_root,
        clean_predictions,
        experiment,
        config["bootstrap"],
        contrasts,
        int(expectations["class_count"]),
        audit,
    )
    print("[7/7] Cross-checking selected failure cases", flush=True)
    failure_predictions = audit_failure_materials(
        results_root,
        failure_root,
        clean_predictions,
        list(config["failure_cases"]["paths"]),
        experiment,
        audit,
    )

    checks = pd.DataFrame(audit.rows)
    checks.to_csv(output_dir / "final_verification_results.csv", index=False)
    scores.to_csv(output_dir / "recomputed_condition_metrics.csv", index=False)
    clean_summary.to_csv(output_dir / "recomputed_clean_summary.csv", index=False)
    robustness_summary.to_csv(
        output_dir / "recomputed_robustness_summary.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "recomputed_bootstrap_contrasts.csv",
        index=False,
    )
    failure_predictions.to_csv(
        output_dir / "failure_case_predictions.csv",
        index=False,
    )

    counts = Counter(checks["status"])
    failures = checks.loc[checks["status"] == "FAIL"]
    numeric_differences = pd.to_numeric(
        checks["absolute_difference"],
        errors="coerce",
    )
    summary = {
        "status": "PASS" if failures.empty else "FAIL",
        "profile": config["profile"],
        "config": str(config_path),
        "checks": {key: int(value) for key, value in sorted(counts.items())},
        "failed_checks": int(len(failures)),
        "warning_checks": int((checks["status"] == "WARN").sum()),
        "maximum_absolute_numeric_difference": (
            float(numeric_differences.max())
            if numeric_differences.notna().any()
            else None
        ),
        "metric_tolerance": float(metrics["tolerance"]),
        "bundle_files_hashed": bundle_stats["listed_files"],
        "source_archive_verified": source_stats["archive_verified"],
        "raw_files_hashed": source_stats["raw_files_checked"],
        "raw_hash_mismatches": source_stats["raw_hash_mismatches"],
        "test_prediction_files": int(expectations["prediction_files"]),
        "test_prediction_rows": int(expectations["prediction_rows"]),
        "bootstrap_repetitions_per_contrast": int(
            config["bootstrap"]["repetitions"]
        ),
        "bootstrap_contrasts": len(contrasts),
        "limitation": (
            None
            if source_stats["archive_verified"]
            and source_stats["raw_files_checked"] == int(expectations["raw_file_count"])
            and source_stats["raw_hash_mismatches"] == 0
            else "The source archive or extracted BMP set was not fully verified."
        ),
    }
    with (output_dir / "final_verification_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

