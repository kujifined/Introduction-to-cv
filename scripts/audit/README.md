# Independent audit modules

`scripts/verify_final_results.py` is the single final-test entry point. It loads
`configs/audit/final_audit.yaml` and calls the stages in this package in order:

1. `bundle.py` — post-extraction result-bundle size and SHA-256 verification;
2. `manifest.py` — manifest, split, group isolation, and supplied failure BMP hashes;
3. `source_data.py` — official RAR fingerprint and all extracted BMP hashes;
4. `predictions.py` — table structure and independent metric recomputation;
5. `aggregation.py` — clean mean/sample SD and corruption robustness;
6. `bootstrap.py` — registered paired stratified bootstrap;
7. `failure_cases.py` — selected clean-error cross-model verification.

`common.py` contains only independent audit implementations. It does not import
the production metric, comparison, or robustness modules under `src/`.

Run the final profile from the project root:

```powershell
python scripts/verify_final_results.py --config configs/audit/final_audit.yaml
```

Paths and the metric tolerance can be overridden without editing YAML:

```powershell
python scripts/verify_final_results.py `
  --config configs/audit/final_audit.yaml `
  --results-root path/to/icv_final_results `
  --failure-root path/to/clean_failure_materials `
  --source-archive path/to/NEU-CLS.rar `
  --data-root path/to/extracted/data/root `
  --output-dir reports/final_verification `
  --tolerance 1e-12
```

The previous validation-pilot wrapper remains compatible with its original CLI
and also accepts a YAML profile. Copy and edit
`configs/audit/validation_audit.example.yaml` when the extracted weekly bundle
and full raw dataset are available.

