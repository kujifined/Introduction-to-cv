# Final experimental evidence

These records come from the [saved final Kaggle run](https://www.kaggle.com/code/datastr3am/icv-final-project-results?scriptVersionId=364524496). Original file content is preserved. `INVENTORY.json` maps each included artifact to its SHA-256; `ARTIFACTS.json` records the source ZIP, dataset manifest and checkpoint hashes. New documentation and inventory metadata are not claims about new training.

| File or directory | Contents |
|---|---|
| `clean_scores_by_configuration.csv`, `clean_scores_by_run.csv` | Clean-test means/SD and all 12 runs. |
| `test_scores_all_conditions.csv` | All 336 model/seed/condition metric records. |
| `comparison_robustness/` | Mean corrupted F1, clean-to-corrupted drops and registered contrasts. |
| `comparison_clean/` | Five paired stratified image-bootstrap contrasts and metadata. |
| `runtime_summary.csv` | Parameters, checkpoint bytes, selected/completed epochs, training origin/time and T4 latency for all 12 models. |
| `outputs/<configuration>/seed_<seed>/` | Original resolved configuration, history, training summary, runtime and learning curves. `evaluation/test` and `evaluation/corruptions` contain predictions, metrics, per-class results and confusion tables. |
| `protocol_freeze.json`, `experiment_protocol.json` | Protocol and pre-test file/source hashes. |
| `figures/` | Severity curves and three clean failure cases. |

There are 12 checkpoints, 28 conditions per checkpoint and 360 original test images. The 120,960 prediction rows are paired views of those 360 images. They are not independent new samples. Corrupted F1 is averaged over 27 condition-level scores within each seed, then summarized over three seeds. SD uses `ddof=1`. A zero-spanning image-bootstrap interval does not establish equivalence.

Training times for seed 43 come from earlier runs; all inference benchmarks were measured in the final common T4 session. Timing uses FP32, batch one, ten warm-up and 50 synchronized timed forwards, excluding load, preprocessing and host-to-device transfer. These values are not end-to-end application latency.

The complete external archive also includes raw input fingerprints, additional plots, logs, historical provenance and weights. Run `uv run --no-sync python scripts/verify_saved_results.py` to check the included records. No images or model weights are required for that verification.
