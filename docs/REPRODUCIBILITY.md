# Reproduce the results and run inference

All commands below start at the repository root. Full training is run on Kaggle; local commands cover installation, saved-result verification and inference on existing weights.

## Install and check

```bash
uv sync --locked --python 3.11 --extra dev --extra data --extra notebook
uv run --no-sync ruff check .
uv run --no-sync pytest
uv run --no-sync python scripts/verify_saved_results.py
```

The last command uses the included predictions, the fixed manifest and independent NumPy/scikit-learn calculations. It verifies all 336 condition results and the clean/corrupted macro-F1 aggregation without accessing images or weights. The tolerance for probability metrics is 2e-6; accuracy and F1 should agree to floating-point precision. The result inventory retains original file hashes.

## Obtain the dataset

Review the source terms on the [official NEU dataset page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm), then run:

```bash
uv run --no-sync python scripts/download_neu_cls.py --acknowledge-source-terms --extract
uv run --no-sync python -c "from industrial_surface_defects.data import verify_manifest_files; verify_manifest_files('data/manifests/neu_cls_v2.csv', 'data/raw/NEU-CLS'); print('manifest and files: OK')"
```

The manifest paths include `NEU-CLS/`, so an example resolves to `data/raw/NEU-CLS/NEU-CLS/Cr_107.bmp`. All 1,800 original files are checked; one duplicate remains excluded from the active 1,799-image dataset. Do not regenerate the frozen manifest. [The data notes](../data/README.md) record the reviewed groups and acquisition metadata.

## Restore ready-to-use weights

Download `icv_final_results.zip` from the [executed Kaggle version](https://www.kaggle.com/code/datastr3am/icv-final-project-results?scriptVersionId=364524496), using its Output tab. Keep the archive outside Git, for example under `results_download/`. The complete archive SHA-256 and all 12 checkpoint hashes are in [ARTIFACTS.json](../results/final/ARTIFACTS.json).

```bash
uv run --no-sync python scripts/prepare_final_artifacts.py --archive results_download/icv_final_results.zip
```

This verifies the complete ZIP and extracts only the four seed-43 demo checkpoints to `checkpoints/<configuration>/seed_43/best.pt`. Existing identical weights are accepted; different files are preserved and reported as an error. Add `--all-checkpoints` to restore all 12 models. The complete raw dataset and weights are excluded from Git; only the two verified demo crops are included.

## Real image inference

```bash
uv run --no-sync python scripts/infer_image.py --image data/raw/NEU-CLS/NEU-CLS/Cr_107.bmp --device cpu
uv run --no-sync python scripts/infer_image.py --image data/raw/NEU-CLS/NEU-CLS/Cr_107.bmp --corruption gaussian_blur --severity 3 --device cpu
uv run --no-sync python scripts/infer_image.py --image data/raw/NEU-CLS/NEU-CLS/Sc_69.bmp --device cpu
```

Each command loads hash-verified weights, builds the networks without downloading ImageNet initialization, and performs four model forward passes. It prints top-1, top-2 and all six probabilities. After initial setup, inputs and weights are local and inference requires no network. `--output validation/example.json` saves the computed result. CPU timings are not the published T4 benchmark.

The [presentation notebook](../notebooks/09_live_demo.ipynb) and its [setup guide](../notebooks/README_LIVE_DEMO.md) provide a standalone offline interface for the same four models and preprocessing. It accepts the original ZIP or restored checkpoints and displays the actual input images and model outputs. [demo/cases.json](../demo/cases.json) defines the three scenarios; [reference outputs](../demo/reference_outputs.csv) are original Kaggle predictions for checking the implementation, not a substitute for inference. The four models use the same historical seed 43 throughout the illustration. The cases were chosen for explanation, not for estimating performance.

## Check and present the demo

```bash
uv run --no-sync python -B notebooks/check_live_demo.py
uv run --no-sync jupyter lab notebooks/09_live_demo.ipynb
```

The checker executes the actual notebook cells offline and saves a fresh verification under `validation/`. Load and warm up the four models before the two-minute recording or presentation. Run the three scenes individually to show the clean input, corruption comparison and remaining error. The [checked recording script](DEMO_RECORDING_SCRIPT.md) gives four-speaker notes.

## Full independent archive audit

The team's YAML-configured `scripts/verify_final_results.py` remains unchanged. It verifies the complete extracted Kaggle bundle and can recompute the registered bootstrap. This is distinct from the quick included-artifact check above. See [the audit guide](../scripts/audit/README.md) for its expected full-bundle and raw-data paths.

## Full Kaggle reproduction

Import [08_kaggle_results_checkpoint.ipynb](../notebooks/08_kaggle_results_checkpoint.ipynb) into a fresh Kaggle notebook. Enable GPU T4 and Internet. Two provisioned T4 devices are not required; the code uses GPU 0.

The notebook is the unchanged runner used for the recorded final evaluation. It checks out the pinned historical core and applies its hash-verified extension payload. Those extension files are also readable under `src/` and `tests/` in this repository. The retained historical revision is intentional: updating this README does not change the experiment source.

Choose one mode before starting the session:

| Mode | Setup | Work performed |
|---|---|---|
| Recorded workflow | `RETRAIN_SEED43=False`; attach the output of [icv-progress-check version 362621632](https://www.kaggle.com/code/datastr3am/icv-progress-check/output?scriptVersionId=362621632) through Add Input | Verify and reuse four historical seed-43 checkpoints; train eight runs at seeds 42 and 44. |
| Complete fresh matrix | Set `RETRAIN_SEED43=True` in the first code cell; no historical input required | Train all 12 registered architecture/augmentation/seed combinations in one fresh session. This creates a new experiment, not the original checkpoint bytes. |

Keep `ACKNOWLEDGE_FINAL_TEST=True` for the complete fixed test evaluation, or set it to `False` before running for training/validation only. The test is opened after all training and validation checks and the recorded protocol freeze. The full run writes `icv_training_checkpoint.zip` before test and `icv_final_results.zip` after successful aggregation. Use **Save Version → Save & Run All**. Do not retune settings, labels or corruption severity using the observed test results.

## Result locations

- [results/final](../results/final/README.md): aggregate tables, 336 item-level prediction files, per-class metrics, configurations, histories, runtime and protocol metadata.
- [Environment notes](../environment/README.md): exact recorded Kaggle packages and separate local environment.
- [Last checked experimental report](../reports/final/results_checkpoint_20261103.md): full analysis and three real failure cases.
- [Team contributions](CONTRIBUTIONS.md): completed work, distinct from outstanding defense assignments.
