# Live demo: steel defect classification

The notebook runs actual CPU inference using the four frozen seed-43 checkpoints. It displays the input crop, the dataset label, and the two leading classes with their probabilities for each model. Reference CSV values are used only by the checker, not by the demo.

## Setup

Run these commands from the repository root. Python 3.11 and uv use the repository lock; Python 3.12 and 3.13 are also within the declared package range.

```bash
uv sync --locked --python 3.11 --extra dev --extra data --extra notebook
```

Download `icv_final_results.zip` from the [final Kaggle output](https://www.kaggle.com/code/datastr3am/icv-final-project-results?scriptVersionId=364524496) into `results_download/`. Then either use that ZIP directly or restore the four demo checkpoints:

```bash
uv run --no-sync python scripts/prepare_final_artifacts.py --archive results_download/icv_final_results.zip
```

The demo automatically checks `results_download/icv_final_results.zip`, an extracted `results_download/icv_final_results/`, and `checkpoints/`. Legacy `week9/` locations remain accepted. For a different location, set `ICV_DEMO_WEIGHTS` to the ZIP, an extracted result root containing `outputs/`, or the restored `checkpoints/` directory. No absolute personal path is embedded in the notebook.

## Open and demonstrate

```bash
uv run --no-sync jupyter lab notebooks/09_live_demo.ipynb
```

In VS Code, select this repository's `.venv` as the notebook kernel instead. Keep `09_live_demo.ipynb`, `live_demo.py` and `demo_assets/` together under `notebooks/`.

Before recording or presenting, run the first code cell and wait for `Ready`. It verifies image and checkpoint hashes, loads the models without downloading ImageNet weights, sets evaluation mode and performs a warm-up. During the demo, execute the remaining three code cells one at a time:

1. `clean_reference`: Cr_107, true class crazing; all four models predict crazing.
2. `same_image_blurred`: the same crop with registered Gaussian blur severity 3, radius 2 before resize; minimal models fail and camera-trained models retain crazing.
3. `observed_failure`: clean Sc_69, true class scratches; minimal EfficientNet predicts inclusion.

Each scene performs new forward passes. Once dependencies and weights are present, no network is required. Ground-truth labels are joined for display after inference. All models use the same grayscale-to-three-channel, 224-pixel bilinear/antialias and ImageNet-normalized preprocessing. These curated examples illustrate the experiment; final metrics use the complete three-seed test evaluation.

## Check the actual notebook cells offline

```bash
uv run --no-sync python -B notebooks/check_live_demo.py
```

To select another archive explicitly:

```bash
uv run --no-sync python -B notebooks/check_live_demo.py --weights results_download/icv_final_results.zip --output validation/live_demo_smoke_test.json
```

The checker executes all four notebook code cells in a fresh IPython process with network and external-process attempts blocked. It checks all 12 top-1/top-2 pairs against the original Kaggle predictions, validates the six probabilities and confirms the HTML contains the expected images and model rows. Fresh output is saved under ignored `validation/`. A [recorded verification](../environment/live_demo_verified.json) describes the tested code and runtime; repeat the check and rehearse in the notebook interface on the presentation device.

The two verified example crops are included under `demo_assets/`; the complete dataset and model weights are distributed separately. Published T4 latency is a separate benchmark and is not the current CPU demo time.
