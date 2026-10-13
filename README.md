# P03 — Industrial Surface Defect Classification

This repository contains the reproducible implementation for project P03 in
the **Introduction to Computer Vision** course. The task is six-class
classification of already cropped hot-rolled steel surface defects from
NEU-CLS.

> **Evidence status.** Data acquisition, audit, splitting, training,
> evaluation, robustness, comparison, error-analysis, and explanation code is
> implemented. The reviewed `neu_cls_v2` manifest is the fixed dataset
> authority. One ResNet-18 baseline run has completed on Kaggle, and its saved
> validation predictions and checkpoint have been independently checked.
> The full architecture/augmentation comparison and test evaluation remain
> pending.

## Initial baseline: 13 October 2026

The ImageNet-pretrained ResNet-18 baseline used basic preprocessing without
random augmentation, training seed `43`, and the fixed split with seed `42`.
It ran on one Tesla T4 for 13 epochs before early stopping, with the epoch-3
checkpoint selected by validation macro-F1.

| Validation measure | Result |
|---|---:|
| Images | 180, with 30 per class |
| Accuracy | 100.00% (180/180) |
| Macro-F1 | 1.0000 |
| Cross-entropy | 0.0458 |

The perfect validation score was reproduced from the saved checkpoint using
independent inference code. It is a checkpoint-selection result on a small,
same-dataset validation set, not evidence of industrial generalisation. The
reserved test set has not been evaluated, and no robustness or multi-seed
results are claimed.

- [Public Kaggle notebook and run output](https://www.kaggle.com/code/datastr3am/icv-p03-resnet18-baseline-seed-43)
- [Milestone report](reports/initial_baseline_20261013.md)
- [Verified configuration, logs, predictions and figures](reports/kaggle_baseline_20261013/)
- [Working Kaggle notebook source](notebooks/05_kaggle_baseline_v3.ipynb)

Raw images and model weights are not committed. The checkpoint is available
in the downloadable Kaggle output, with its SHA-256 recorded in the audit.

## Team

- Team leader: Danil Khasanshin (`@undefinedkuji`)
- T. Kagarmanov — `t.kagarmanov@innopolis.university`
- Danil Khasanshin — `d.khasanshin@innopolis.university`
- R. Nasibullin — `ru.nasibullin@innopolis.university`
- N. Selezenev — `n.selezenev@innopolis.university`

## Project requirements and scope

P03 requires the team to:

1. classify common surface defects with NEU Surface Defect or another approved
   industrial dataset;
2. compare two architectures or augmentation strategies;
3. analyse false positives; and
4. examine visually similar defect types.

NEU-CLS contains **defect crops only**. It has no `normal` or defect-free
class, so this project asks which known defect type a crop belongs to. It does
not detect or localise defects, estimate a production false-alarm rate on
normal steel, or decide whether a surface is defective. In this multiclass
setting, a false positive for class `c` is defined one-vs-rest: the ground-truth
class is not `c`, but the model predicts `c`.

## Registered experiment

The primary comparison is a complete **2 × 2 factorial design**. ImageNet
pretraining is held constant so that architecture is not confounded with
initialisation.

| Cell | Architecture | Initialisation | Training augmentation |
|---|---|---|---|
| R18-MIN | ResNet-18 | ImageNet pretrained | minimal |
| R18-CAM | ResNet-18 | ImageNet pretrained | camera-aware |
| ENB0-MIN | EfficientNet-B0 | ImageNet pretrained | minimal |
| ENB0-CAM | EfficientNet-B0 | ImageNet pretrained | camera-aware |

Each cell is trained with seeds `42`, `43`, and `44`: 12 registered runs in
total. A small randomly initialised CNN is allowed only as an optional sanity
baseline; it is not part of the primary contrast.

The fixed training settings are 50 epochs maximum, AdamW, learning rate
`1e-4`, weight decay `1e-4`, three warm-up epochs followed by cosine decay,
early-stopping patience 10 on validation macro-F1, and cross-entropy label
smoothing `0.05`. All four checked-in YAML configurations use the final
manifest and seed `42`; the matrix runner overrides the seed with `42`, `43`,
or `44`.

## Dataset authority and audit

Download the image-only **NEU-CLS** package from the
[official Northeastern University page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm).
Do not silently substitute NEU-DET, which is a detection dataset.

The official archive audit found 1,800 readable grayscale images of size
200 × 200, nominally 300 per class. The reviewed audit then established:

- `Pa_101.bmp` and `Pa_105.bmp` are byte-identical; `Pa_105.bmp` is retained in
  the manifest as `duplicate_quarantined` with split `excluded`;
- the effective corpus has 1,799 samples: 299 `patches` images and 300 in each
  other class;
- a 64-bit DCT pHash review used Hamming-distance threshold 12, producing 13
  candidates; all were resolved;
- four shifted-duplicate pairs are grouped before splitting:
  `PS_171`/`PS_292`, `Pa_199`/`Pa_201`, `Pa_2`/`Pa_5`, and
  `In_60`/`In_62`;
- `PS_171` and `PS_292` have a pixel-identical 198 × 198 overlap after a
  two-pixel diagonal shift and therefore share a group and split;
- `In_188` and `In_210` were reviewed as visually related but distinct. They
  remain separate groups and happen to be in the same training split.

The fixed, stratified, duplicate-aware split uses seed `42` and proportions
70%/10%/20%:

| Split | Total active images | Patches | Each other class |
|---|---:|---:|---:|
| train | 1,259 | 209 | 210 |
| validation | 180 | 30 | 30 |
| test | 360 | 60 | 60 |

The authoritative artifacts are:

- `data/manifests/neu_cls_source_v1.json` — official archive provenance and
  acquired archive SHA-256;
- `data/manifests/neu_cls_v2.csv` — rows, hashes, groups, statuses, and split;
- `data/manifests/neu_cls_v2.audit.json` — corpus and audit summary;
- `data/manifests/neu_cls_v2_near_duplicates.csv` — reviewed candidates; and
- `data/manifests/neu_cls_reviewed_groups_v2.csv` — confirmed source groups.

See [`data/README.md`](data/README.md) for full provenance and audit decisions.

## Evaluation contract

- Use the same `neu_cls_v2.csv` manifest for every run.
- Use training data to fit parameters and validation macro-F1 for checkpoint
  selection. Keep test access locked until the protocol and analysis choices
  are frozen.
- Primary metric: macro-F1. Also report accuracy, balanced accuracy, per-class
  precision/recall/F1/support, raw and normalised confusion matrices, NLL, ECE,
  latency, and model size.
- Report every registered seed and mean ± standard deviation across seeds; do
  not select the most favourable seed.
- Analyse class-specific one-vs-rest false positives, false negatives,
  high-confidence errors, and directional confusions between similar classes.
- Evaluate the selected checkpoints on clean test images and on the implemented
  deterministic robustness suite: nine corruption types at severity levels
  1, 2, and 3 (27 corrupted conditions per checkpoint).
- Treat corrupted views as paired transformations of the same 360 test images,
  not as additional independent samples or external-domain validation.

The complete registered procedure is in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Reproducible runbook

Run all commands from the repository root. Python 3.11 or 3.12 and
[uv](https://docs.astral.sh/uv/) are required.

For a remote GPU run, use
[`notebooks/02_cloud_runner.ipynb`](notebooks/02_cloud_runner.ipynb). Its
training and evaluation cells are disabled by default, it can persist outputs
to mounted storage, and it requires a human protocol-freeze record before the
locked test stage. The shell commands below describe the same workflow for any
managed GPU environment.

For the single baseline on Kaggle, use the verified
[`notebooks/05_kaggle_baseline_v3.ipynb`](notebooks/05_kaggle_baseline_v3.ipynb).
It retains Kaggle's CUDA packages, imports pinned project source directly,
extracts the official archive with `unar`, and requires data verification and
all tests to pass before training. It evaluates validation only. The recorded
Kaggle run used Python 3.13 outside the package's declared Python range; its
environment metadata documents this source-import execution mode rather than
claiming an installation from `uv.lock`.

### 1. Create the locked environment

```bash
uv sync --extra data --extra dev --extra notebook
uv run ruff check .
uv run pytest
```

`uv.lock` is the dependency authority. Record the Git revision, runtime, and
hardware before registered runs.

### 2. Acquire the official archive once

Review the source terms first. The downloader refuses to overwrite an existing
archive or extract into a non-empty destination.

```bash
uv run python scripts/download_neu_cls.py \
  --acknowledge-source-terms \
  --extract
```

This writes the local archive and acquisition metadata under `data/archives/`
and extracts beneath `data/raw/NEU-CLS/`. Neither raw data nor the archive is
redistributed through Git.

### 3. Verify the final reviewed manifest

The checked-in `neu_cls_v2.csv` is already frozen; do not regenerate it under
the same name. Verify that the local raw files match every recorded SHA-256:

```bash
uv run python -c "from industrial_surface_defects.data import verify_manifest_files; verify_manifest_files('data/manifests/neu_cls_v2.csv', 'data/raw/NEU-CLS'); print('manifest and raw files: OK')"
```

To reproduce preparation independently, write to a new temporary filename and
apply the reviewed group map before splitting:

```bash
audit_dir="$(mktemp -d)"
uv run neu-prepare \
  --data-root data/raw/NEU-CLS \
  --manifest "${audit_dir}/neu_cls_v2.csv" \
  --seed 42 \
  --near-duplicate-threshold 12 \
  --group-map data/manifests/neu_cls_reviewed_groups_v2.csv
cmp "${audit_dir}/neu_cls_v2.csv" data/manifests/neu_cls_v2.csv
```

The preparation command deliberately refuses to overwrite frozen artifacts.
The committed audit and reviewed-candidate CSV remain the reporting authority.

### 4. Inspect training-only EDA and augmentation QA

```bash
uv run jupyter lab notebooks/01_eda.ipynb
```

The notebook asserts the final counts and reviewed grouping decisions. Its
visual and intensity analyses use the training split only.

### 5. Confirm and run the 4 × 3 training matrix

```bash
uv run python scripts/run_matrix.py --dry-run
uv run python scripts/run_matrix.py --device auto
```

The second command launches four configurations × three seeds. Run directories
are immutable once populated; use `--skip-existing` only to resume a partially
completed matrix without replacing evidence.

### 6. Evaluate validation artifacts first

After all 12 checkpoints exist:

```bash
uv run python scripts/evaluate_matrix.py \
  --split val \
  --with-error-analysis \
  --device auto
```

Review validation outputs, finish all model-selection and analysis decisions,
and record any protocol deviation. Do not use robustness or test metrics for
selection.

### 7. Unlock the final test once

Only after the protocol and decisions are frozen:

```bash
uv run python scripts/evaluate_matrix.py \
  --split test \
  --acknowledge-test-use \
  --with-error-analysis \
  --with-robustness \
  --device auto
```

This produces item-level clean predictions, metrics, confusion and calibration
artifacts, false-positive/error galleries, 27 corrupted robustness conditions
plus the clean baseline, and the registered cross-run comparison. Populate the
report only from these generated artifacts; keep every unmeasured field as
`TBD`.

## Repository map

```text
.
├── configs/                 # four registered YAML experiment cells
├── data/
│   ├── README.md
│   ├── manifests/           # final v2 audit, groups, hashes, split
│   └── raw/                 # local only; never committed
├── docs/                    # protocol and report template
├── notebooks/
│   ├── 01_eda.ipynb        # training-only EDA and visual QA
│   ├── 02_cloud_runner.ipynb # opt-in Colab/Kaggle orchestration
│   └── 05_kaggle_baseline_v3.ipynb # verified single-baseline Kaggle runner
├── reports/
│   ├── initial_baseline_20261013.md
│   └── kaggle_baseline_20261013/ # checked baseline evidence, no weights
├── scripts/                 # download and 12-run orchestration
├── src/industrial_surface_defects/
├── tests/
├── pyproject.toml
└── uv.lock
```

## Reproducibility and evidence policy

Every reported number must be traceable to the final manifest and its hash,
resolved configuration, training seed, package/runtime metadata, checkpoint,
and item-level prediction file. Never copy reference output from a course
notebook, infer a value from expectations, or report an unlogged interactive
run. The report template intentionally contains `TBD` placeholders.

## References

- Kechen Song and Yunhui Yan, “A noise robust method based on completed local
  binary patterns for hot-rolled steel strip surface defects,” *Applied Surface
  Science*, vol. 285, pp. 858–864, 2013.
  [doi:10.1016/j.apsusc.2013.09.002](https://doi.org/10.1016/j.apsusc.2013.09.002)
- [NEU surface defect database — official Northeastern University faculty
  portal](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
