# Experiment Protocol

## 1. Purpose, status, and task boundary

This document registers the primary study for P03, **Industrial Surface Defect
Classification**. The implementation and reviewed dataset artifacts exist, but
this protocol does not claim that training or evaluation has been run. A model
result becomes evidence only when it can be regenerated from the fixed
manifest, a resolved configuration, a checkpoint, and item-level predictions.

NEU-CLS contains cropped images from six known defect classes and **no normal
surface class**. The task is therefore closed-set defect-type classification,
not defect detection, localisation, anomaly detection, or normal-versus-defect
screening. A production false-alarm rate on normal steel is outside the scope
of this dataset.

Any change made after registered training begins must be recorded in the
report's deviation log. Model, augmentation, checkpoint-selection, and analysis
choices must be frozen before the locked test command is acknowledged.

## 2. Research questions

### RQ1 — Architecture

With ImageNet pretraining, split, input processing, optimisation budget, and
evaluation held constant, how do ResNet-18 and EfficientNet-B0 differ in clean
macro-F1 and class-specific errors?

### RQ2 — Augmentation

Within each architecture, does `camera_aware_v1` improve robustness to
synthetic acquisition and image-pipeline shifts without an unacceptable loss
on clean images?

### RQ3 — Interaction

Does the augmentation effect differ between architectures?

### RQ4 — Failure modes

Which predicted defect types accumulate one-vs-rest false positives, and which
directional confusions dominate among visually similar classes?

Hypotheses may be stated before the runs, but must not be rewritten as findings
until measured artifacts exist.

## 3. Dataset contract

### 3.1 Source

Use the image-only NEU-CLS archive linked from the
[official Northeastern University dataset page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm).
The nominal archive contains 1,800 grayscale, 200 × 200 images: 300 each for
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled_in_scale`, and
`scratches`. NEU-DET must not be substituted without changing the task and
obtaining instructor approval.

### 3.2 Completed integrity and duplicate audit

The official archive was decoded and inventoried before splitting. The final
audit records 1,800 readable `L`-mode images at 200 × 200. SHA-256 identified
one extra exact copy:

- retained active row: `NEU-CLS/Pa_101.bmp`;
- quarantined row: `NEU-CLS/Pa_105.bmp`;
- disposition: `status=duplicate_quarantined`, `split=excluded`.

The manifest intentionally keeps the quarantined row for provenance. It is not
loaded by the train, validation, or test datasets. The effective corpus has
1,799 active samples: 299 `patches` images and 300 in each other class.

Near-duplicate screening used a 64-bit DCT perceptual hash with Hamming-distance
threshold 12. It produced 13 candidate pairs. Manual review resolved all 13:
nine as distinct and four as shifted duplicates. Confirmed shifted pairs share
a `group_id` before split assignment:

| Pair | Review evidence | Final handling |
|---|---|---|
| `PS_171` / `PS_292` | pixel-identical 198 × 198 overlap after `dy=-2`, `dx=-2` | grouped in validation |
| `Pa_199` / `Pa_201` | pixel-identical 194 × 189 overlap after `dy=-6`, `dx=-11` | grouped in validation |
| `Pa_2` / `Pa_5` | pixel-identical 191 × 198 overlap after `dy=-9`, `dx=2` | grouped in validation |
| `In_60` / `In_62` | pixel-identical 193 × 190 overlap after `dy=-7`, `dx=-10` | grouped in validation |

In particular, `PS_171` and `PS_292` are the same content under a two-pixel
diagonal crop/shift and must never cross splits. `In_188` and `In_210` are a
pHash-distance-8 candidate but were judged to contain different defect
structures. They remain distinct groups; both happen to be in training, so
their review does not create cross-split leakage.

The final review artifacts are:

```text
data/manifests/neu_cls_v2.audit.json
data/manifests/neu_cls_v2_near_duplicates.csv
data/manifests/neu_cls_reviewed_groups_v2.csv
```

### 3.3 Fixed split

`data/manifests/neu_cls_v2.csv` is the sole split authority for primary runs.
It was built with the reviewed group map, split seed `42`, stratification by
class, and targets 70% train / 10% validation / 20% test. Do not regenerate a
more favourable split after viewing model behaviour.

| Class | Train | Validation | Test | Active total |
|---|---:|---:|---:|---:|
| crazing | 210 | 30 | 60 | 300 |
| inclusion | 210 | 30 | 60 | 300 |
| patches | 209 | 30 | 60 | 299 |
| pitted surface | 210 | 30 | 60 | 300 |
| rolled-in scale | 210 | 30 | 60 | 300 |
| scratches | 210 | 30 | 60 | 300 |
| **Total** | **1,259** | **180** | **360** | **1,799** |

The CSV has 1,800 rows because it also preserves the one excluded duplicate.
Its required loader fields are:

```text
path,class_name,class_index,split,sha256,group_id,status
```

Additional dimensions, mode, file-size, source-label, and perceptual-hash
columns support provenance. Every active SHA-256 and every reviewed group must
occur in exactly one of train, validation, or test.

### 3.4 Roles of the splits

- **Train:** parameter fitting and training-only visual diagnostics.
- **Validation:** checkpoint selection by macro-F1, any permitted model choice,
  and analysis development.
- **Test:** one final clean evaluation and its paired robustness derivatives
  after all choices are frozen.

Test labels must not guide preprocessing, augmentation ranges, hyperparameters,
checkpoint selection, class-pair selection, or manual relabelling.

## 4. Factorial design

Run every registered cell:

| ID / config | Architecture | Pretraining | Training augmentation |
|---|---|---|---|
| `resnet18_minimal` | ResNet-18 | torchvision default ImageNet weights | minimal |
| `resnet18_camera` | ResNet-18 | torchvision default ImageNet weights | camera-aware |
| `efficientnet_b0_minimal` | EfficientNet-B0 | torchvision default ImageNet weights | minimal |
| `efficientnet_b0_camera` | EfficientNet-B0 | torchvision default ImageNet weights | camera-aware |

Replace the classifier with a six-logit head and fine-tune the entire network.
Record the exact torchvision and weight-enum versions from each run's runtime
and resolved configuration. A randomly initialised SmallCNN may be reported as
an optional sanity check, but cannot replace a registered cell or be mixed into
the factorial contrasts.

Run each cell with training seeds `42`, `43`, and `44`. The YAML base seed is
`42`; `scripts/run_matrix.py` passes each registered override. The primary
study therefore consists of 12 runs, all of which must be reported. Do not
select the best seed.

## 5. Input representation

NEU-CLS is grayscale while the pretrained backbones expect three channels.
Both conditions use this common order:

1. decode the source as one-channel grayscale;
2. apply a training augmentation only when assigned to the cell;
3. replicate grayscale to three channels;
4. resize to 224 × 224 with bilinear interpolation and antialiasing; and
5. convert to tensor and apply ImageNet mean/std normalisation.

Validation and test preprocessing is deterministic and identical for all
cells: grayscale replication, resize, tensor conversion, and normalisation.
No statistics are fitted on validation or test data.

## 6. Augmentation conditions

### 6.1 Minimal control

`minimal` contains no stochastic training transform. It uses only the common
input representation above.

### 6.2 `camera_aware_v1`

Stochastic transformations are applied only to training images and before
channel replication. The implemented parameters are:

| Stage | Transform | Parameter distribution | Probability |
|---|---|---|---:|
| geometry | affine | rotation ±10°; x/y translation ≤5%; scale 0.90–1.10; bilinear; median fill | 0.50 |
| exposure | brightness | factor log-uniform in 0.70–1.40 | 0.50 |
| rendering | contrast | factor uniform in 0.80–1.20 | 0.30 |
| rendering | gamma | exponent uniform in 0.80–1.25 | 0.30 |
| sensor | Poisson + Gaussian noise | peak log-uniform in 40–120; σ uniform in 0.005–0.025 | 0.35 |
| optics/motion | Gaussian or motion blur | radius 0.20–1.20, or horizontal/vertical length 3/5/7 | 0.25 |
| file pipeline | JPEG round trip | integer quality 50–95 | 0.25 |

The geometry draw is independent. Among the six exposure/rendering/sensor/
optics/file effects, at most two are retained for one image. Selected effects
are then applied in the fixed order exposure → contrast → gamma → noise → blur
→ JPEG. Gaussian versus motion blur and horizontal versus vertical motion are
each equiprobable in their branch.

Horizontal/vertical flips, hue, saturation, and white-balance jitter are not
part of this grayscale protocol. A seeded augmentation grid must be visually
reviewed before registered training. If transforms erase or change a defect,
version and document the revised augmentation before any test access.

## 7. Training and checkpoint selection

The four YAML configurations register a common optimisation budget:

| Setting | Value |
|---|---|
| objective | six-class cross-entropy |
| label smoothing | `0.05` during training; off for validation loss |
| optimiser | AdamW |
| initial learning rate | `1e-4` |
| weight decay | `1e-4` |
| batch size | 32 |
| maximum epochs | 50 |
| schedule | linear warm-up from 0.1× LR for 3 epochs, then cosine decay |
| minimum learning rate | `1e-6` |
| checkpoint score | validation macro-F1 |
| early stopping | patience 10 validation epochs without improvement |
| tie behaviour | keep the earlier checkpoint because replacement requires a strict improvement |
| mixed precision | enabled on CUDA; runtime behaviour recorded |

Each training call verifies every manifest path and SHA-256 before fitting. A
run writes its resolved configuration, runtime metadata, epoch history,
learning curves, best checkpoint, manifest hash, and training summary into an
immutable experiment/seed directory.

No architecture receives a larger search budget. Any post-registration change
must be applied symmetrically where appropriate, assigned a new run suffix,
and disclosed. Clean or corrupted test performance must never motivate
training changes.

## 8. Clean evaluation and uncertainty

### 8.1 Metrics

The primary metric is macro-F1 over the six classes. Secondary outputs are:

- accuracy, balanced accuracy, weighted F1, and top-2 accuracy;
- per-class precision, recall, F1, support, TP, FP, FN, TN, and false-positive
  rate;
- raw-count and row-normalised confusion matrices;
- negative log-likelihood, multiclass Brier score, and 10-bin expected
  calibration error;
- checkpoint size, parameter count, and batch-one latency with recorded
  device, warm-up count, and repetition count.

Save item-level paths, true/predicted classes, confidence, and all six class
probabilities. These rows, not copied table values, are the source for derived
comparisons and figures.

### 8.2 Seed and sample uncertainty

Report each seed and mean ± standard deviation across the three registered
seeds. The comparison utility additionally uses 5,000 paired bootstrap
resamples with seed `42` for registered contrasts. Training-seed variation and
finite-test-sample bootstrap uncertainty answer different questions and must
be labelled separately. With three seeds, emphasise effect size and observed
variability rather than overstate significance.

## 9. Implemented robustness suite

`corruption_suite_v1` is evaluated only after the test lock is explicitly
acknowledged. Each corruption is applied independently to the decoded 200 × 200
test image, before resizing and normalisation. There are nine corruption types
at severity levels 1, 2, and 3: **27 corrupted conditions plus one clean
condition per checkpoint**.

| Corruption key | Severity 1 | Severity 2 | Severity 3 |
|---|---:|---:|---:|
| `brightness_dark` (brightness factor) | 0.80 | 0.65 | 0.55 |
| `brightness_bright` (brightness factor) | 1.20 | 1.50 | 1.80 |
| `contrast` (contrast factor) | 0.80 | 0.60 | 0.40 |
| `gamma` (exponent) | 1.20 | 1.50 | 2.00 |
| `shot_read_noise` (`peak`, read σ) | (120, 0.005) | (70, 0.0125) | (40, 0.025) |
| `gaussian_blur` (PIL radius) | 0.50 | 1.00 | 2.00 |
| `motion_blur_horizontal` (pixels) | 3 | 5 | 7 |
| `motion_blur_vertical` (pixels) | 3 | 5 | 7 |
| `jpeg` (quality; lower is stronger) | 70 | 40 | 20 |

`shot_read_noise` applies `Poisson(peak*x)/peak + Normal(0, read_sigma²)` on
normalised intensity and then clips to `[0, 1]`. Its draws are deterministically
seeded from `(path, corruption, severity)`. The other transforms are
deterministic by construction. Brightness and gamma operations clip through the
8-bit image representation where applicable.

For every checkpoint and condition, store accuracy, macro-F1, balanced
accuracy, per-class metrics, and item-level predictions. Report absolute and
relative macro-F1 drops from the same checkpoint's clean test score, mean
corrupted macro-F1, mean drop, and the worst condition. Severity curves must be
shown rather than collapsing the suite to one arbitrary fixed setting.

These are paired synthetic derivatives of the same 360 test samples. They do
not increase the effective test size, demonstrate robustness on another
factory/camera, or reproduce a calibrated industrial acquisition process.

## 10. Factorial comparisons

Report all four cells before interpreting these registered contrasts for a
metric `M`:

- architecture under minimal augmentation:
  `M(ENB0-MIN) - M(R18-MIN)`;
- architecture under camera-aware augmentation:
  `M(ENB0-CAM) - M(R18-CAM)`;
- augmentation within ResNet-18:
  `M(R18-CAM) - M(R18-MIN)`;
- augmentation within EfficientNet-B0:
  `M(ENB0-CAM) - M(ENB0-MIN)`; and
- interaction:
  `[M(ENB0-CAM)-M(ENB0-MIN)] - [M(R18-CAM)-M(R18-MIN)]`.

Apply the same logic to clean macro-F1 and explicitly defined robustness
summaries. Do not infer a universal model ranking from one small benchmark.

## 11. False-positive and visually similar-class analysis

### 11.1 Definitions

For class `c`, reduce the six-way prediction to one-vs-rest:

- TP: `true=c` and `predicted=c`;
- FP: `true!=c` and `predicted=c`;
- FN: `true=c` and `predicted!=c`;
- TN: neither true nor predicted is `c`.

Thus every multiclass mistake is a false positive for the predicted defect type
and a false negative for the true defect type. Because there is no normal
class, this FP count is **not** a normal-surface false alarm.

### 11.2 Required analyses

For each cell and seed, then for the aggregate:

1. rank classes by FP count and one-vs-rest FP rate `FP/(FP+TN)`;
2. identify the true-class sources of each predicted class's FPs;
3. inspect highest-confidence FPs and FNs with path, labels, and confidence;
4. calculate directional confusion `a→b`, reverse `b→a`, and their sum;
5. start with the author-noted `Cr`/`RS`/`PS` similarity cluster, but also
   report any empirically dominant pair outside it;
6. compare whether the same images/pairs fail across architectures,
   augmentations, seeds, and corruption severity; and
7. use Grad-CAM only as qualitative supporting evidence, never as proof of a
   causal visual mechanism.

Discuss texture scale, defect orientation, spatial extent, local/global
contrast, blur, acquisition shortcuts, and ambiguous crops. Separate direct
observation from causal interpretation.

## 12. Execution order and lock

1. Acquire the official data and retain acquisition metadata.
2. Verify raw files against `neu_cls_v2.csv`; optionally reproduce preparation
   to a new temporary manifest with threshold 12 and the reviewed group map.
3. Run the training-only EDA and augmentation visual QA.
4. Run lint/tests and the 12-run dry run.
5. Train all four cells for seeds 42/43/44.
6. Evaluate and analyse the validation split for all 12 checkpoints.
7. Freeze selections, hypotheses, class-pair analysis rules, and deviations.
8. Run the acknowledged clean test, error analysis, robustness suite, and
   registered comparison once.
9. Populate the report from generated files and archive hashes.

The exact commands are in the root [`README.md`](../README.md).

## 13. Reproducibility checklist

Record for every result:

- Git revision and dirty-worktree status;
- `uv.lock` hash, Python, PyTorch, torchvision, CUDA/MPS, and OS versions;
- hardware, device, and runtime nondeterminism notes;
- `neu_cls_v2.csv` hash and dataset fingerprint;
- resolved YAML, architecture, augmentation version, and seed;
- exact pretrained weight identifiers;
- best epoch and validation macro-F1 used for selection;
- checkpoint and prediction-file hashes;
- evaluation command and test-access acknowledgement; and
- figure/table generation path.

## 14. Required limitations

The report must acknowledge at least:

- no normal class and therefore no production defect-detection false-alarm
  estimate;
- classification of pre-cropped known-defect images, not localisation;
- a small, balanced benchmark unlike production class prevalence;
- unknown source-strip identities beyond recoverable duplicate groups;
- grayscale data and limited camera/acquisition metadata;
- synthetic corruptions rather than calibrated hardware or external-domain
  validation;
- only two compact pretrained CNN families; and
- uncertainty from three training seeds and 360 locked test items.

## Reference

Song, K., & Yan, Y. (2013). A noise robust method based on completed local
binary patterns for hot-rolled steel strip surface defects. *Applied Surface
Science, 285*, 858–864.
[https://doi.org/10.1016/j.apsusc.2013.09.002](https://doi.org/10.1016/j.apsusc.2013.09.002)
