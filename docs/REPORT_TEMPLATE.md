# P03 Final Report Template

> Replace bracketed prompts and `TBD` values only with verified information
> traceable to repository artifacts. The dataset-audit facts below are fixed;
> model results are intentionally blank until registered runs are completed.

## Title page

**Industrial Surface Defect Classification: A Factorial Comparison of
Architecture and Camera-Aware Augmentation**

- Course: Introduction to Computer Vision
- Project: P03 — Industrial Surface Defect Classification
- Team leader: Danil Khasanshin
- Team members: T. Kagarmanov, Danil Khasanshin, R. Nasibullin, N. Selezenev
- Code revision: `[commit hash]`
- Report date: `[date]`

## Abstract

[In 150–250 words, state the closed-set six-class task, the reviewed 1,799-image
effective corpus, the ResNet-18/EfficientNet-B0 × minimal/camera-aware design,
three seeds, primary macro-F1 metric, and clean plus synthetic robustness
evaluation. Add only verified numerical results. End with the main limitation:
NEU-CLS contains no normal class and cannot estimate production false alarms.]

## 1. Introduction

### 1.1 Motivation

[Explain why reliable classification of surface-defect types can support
inspection and diagnosis. Distinguish a benchmark classifier from a deployed
inspection system.]

### 1.2 Task boundary

NEU-CLS contains pre-cropped images from six known defect classes and no
normal/defect-free class. This study classifies the defect type of a supplied
crop. It does not localise defects, decide whether an arbitrary surface is
defective, or estimate false alarms on normal production material.

For defect class `c`, a false positive is defined one-vs-rest: `true != c` and
`prediction = c`. This is a wrong defect-type assignment, not a normal-surface
false alarm.

### 1.3 Research questions

- RQ1: How do ImageNet-pretrained ResNet-18 and EfficientNet-B0 compare when
  data, training budget, and preprocessing are controlled?
- RQ2: Does camera-aware augmentation improve synthetic acquisition-shift
  robustness without unacceptable clean-performance loss?
- RQ3: Does the augmentation effect depend on architecture?
- RQ4: Which predicted classes attract false positives, and which visually
  similar class pairs dominate errors?

### 1.4 Contributions

[Describe the implemented reproducible pipeline, leakage-aware audit, complete
2 × 2 comparison, paired robustness suite, and class-specific error analysis.
Do not list unmeasured performance as a contribution.]

## 2. Related work and course concepts

[Briefly cover industrial texture classification, transfer learning, data
augmentation, confusion/calibration analysis, and the course's image-formation
pipeline. Cite the NEU paper, pretrained architectures, and any other methods
actually used.]

## 3. Data

### 3.1 Source and nominal composition

NEU-CLS was obtained from the
[official Northeastern University dataset page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm).
The official archive contains 1,800 grayscale 200 × 200 images, nominally 300
for each class.

### 3.2 Integrity and duplicate audit

Report the checked-in `data/manifests/neu_cls_v2.audit.json` and review files:

- all 1,800 source images decoded as 200 × 200 `L`-mode images;
- `Pa_101.bmp` and `Pa_105.bmp` were byte-identical;
- `Pa_105.bmp` was quarantined as `duplicate_quarantined/excluded`;
- 1,799 active images remained, including 299 patches and 300 in each other
  class;
- 64-bit DCT pHash screening at Hamming distance ≤12 produced 13 candidates;
- manual review resolved nine as distinct and four as shifted duplicates;
- grouped shifted pairs were `PS_171`/`PS_292`, `Pa_199`/`Pa_201`,
  `Pa_2`/`Pa_5`, and `In_60`/`In_62`;
- `PS_171`/`PS_292` had a pixel-identical 198 × 198 overlap after a two-pixel
  diagonal shift;
- `In_188`/`In_210` were treated as distinct after review and both are in the
  training split.

[Include a small audit table or montage only if it adds evidence. Do not expose
test-image content in a way that affected model design before the lock.]

### 3.3 Fixed split

Manifest: `data/manifests/neu_cls_v2.csv`
Split seed: `42`
Policy: stratified, duplicate/group-aware 70%/10%/20%

| Class | Source | Excluded | Train | Validation | Test | Active |
|---|---:|---:|---:|---:|---:|---:|
| crazing | 300 | 0 | 210 | 30 | 60 | 300 |
| inclusion | 300 | 0 | 210 | 30 | 60 | 300 |
| patches | 300 | 1 | 209 | 30 | 60 | 299 |
| pitted surface | 300 | 0 | 210 | 30 | 60 | 300 |
| rolled-in scale | 300 | 0 | 210 | 30 | 60 | 300 |
| scratches | 300 | 0 | 210 | 30 | 60 | 300 |
| **Total** | **1,800** | **1** | **1,259** | **180** | **360** | **1,799** |

Manifest SHA-256: `[compute from the exact reviewed artifact]`
Dataset fingerprint from audit: `[copy and verify from audit JSON]`

### 3.4 Training-only exploratory analysis

[Report class examples, within-class diversity, and training-only intensity
distributions from `notebooks/01_eda.ipynb`. Explain any suspected shortcut as
a hypothesis, not a demonstrated causal mechanism.]

## 4. Methods

### 4.1 Factorial design

| Cell | Architecture | ImageNet pretraining | Training augmentation | Seeds |
|---|---|---|---|---|
| R18-MIN | ResNet-18 | yes | minimal | 42, 43, 44 |
| R18-CAM | ResNet-18 | yes | `camera_aware_v1` | 42, 43, 44 |
| ENB0-MIN | EfficientNet-B0 | yes | minimal | 42, 43, 44 |
| ENB0-CAM | EfficientNet-B0 | yes | `camera_aware_v1` | 42, 43, 44 |

[State that all 12 registered runs are included. If a SmallCNN sanity baseline
was run, put it in a separate exploratory subsection and do not use it as a
substitute for a factorial cell.]

### 4.2 Input and augmentation

All images are decoded as grayscale, optionally augmented on training only,
replicated to three channels, resized to 224 × 224, converted to tensors, and
normalised with ImageNet mean and standard deviation. Validation and test
pipelines are deterministic.

[Summarise the exact `camera_aware_v1` affine, exposure, contrast, gamma,
shot/read-noise, blur, and JPEG ranges from the protocol. Include the seeded
visual-QA grid and state whether the team accepted it before training.]

### 4.3 Training and selection

| Setting | Registered value |
|---|---|
| loss | six-class cross-entropy, label smoothing `0.05` |
| optimiser | AdamW |
| learning rate | `1e-4` |
| weight decay | `1e-4` |
| batch size | 32 |
| maximum epochs | 50 |
| schedule | 3-epoch linear warm-up, then cosine to `1e-6` |
| early stopping | patience 10 on validation macro-F1 |
| checkpoint | strict best validation macro-F1; earlier epoch wins a tie |
| seeds | 42, 43, 44 |

Actual runtime devices and pretrained weight identifiers: `[from runtime and
resolved-config artifacts]`.

### 4.4 Evaluation

Primary metric: macro-F1. Secondary metrics: accuracy, balanced accuracy,
weighted F1, top-2 accuracy, per-class precision/recall/F1/support, raw and
normalised confusion matrices, NLL, multiclass Brier score, 10-bin ECE, model
size, and measured batch-one latency.

Report per-seed values and mean ± standard deviation across seeds. Describe the
implemented 5,000-repetition paired bootstrap (seed `42`) separately from
training-seed variability.

### 4.5 Locked robustness evaluation

The implemented suite contains nine corruptions × three severity levels. Each
condition is a paired derivative of the same 360 test images.

| Key | Severity 1 | Severity 2 | Severity 3 |
|---|---:|---:|---:|
| brightness dark | factor 0.80 | factor 0.65 | factor 0.55 |
| brightness bright | factor 1.20 | factor 1.50 | factor 1.80 |
| contrast | factor 0.80 | factor 0.60 | factor 0.40 |
| gamma | 1.20 | 1.50 | 2.00 |
| shot/read noise | peak 120, read σ 0.005 | peak 70, read σ 0.0125 | peak 40, read σ 0.025 |
| Gaussian blur | radius 0.50 | radius 1.00 | radius 2.00 |
| horizontal motion blur | 3 px | 5 px | 7 px |
| vertical motion blur | 3 px | 5 px | 7 px |
| JPEG | quality 70 | quality 40 | quality 20 |

The suite is synthetic sensitivity analysis, not an external-dataset or
calibrated industrial-camera test.

## 5. Results

> Keep every cell as `TBD` until generated artifacts have been checked.

### 5.1 Run completeness

| Cell | Seed 42 checkpoint | Seed 43 checkpoint | Seed 44 checkpoint | All artifacts verified? |
|---|---|---|---|---|
| R18-MIN | TBD | TBD | TBD | TBD |
| R18-CAM | TBD | TBD | TBD | TBD |
| ENB0-MIN | TBD | TBD | TBD | TBD |
| ENB0-CAM | TBD | TBD | TBD | TBD |

### 5.2 Clean-test primary comparison

Values should be mean ± SD across registered seeds. Link the generated table.

| Cell | Macro-F1 ↑ | Accuracy ↑ | Balanced accuracy ↑ | NLL ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| R18-MIN | TBD | TBD | TBD | TBD | TBD |
| R18-CAM | TBD | TBD | TBD | TBD | TBD |
| ENB0-MIN | TBD | TBD | TBD | TBD | TBD |
| ENB0-CAM | TBD | TBD | TBD | TBD | TBD |

Generated artifact: `reports/tables/generated/comparison_test/[path]`

[Insert learning curves before interpreting test metrics. Describe validation
behaviour separately from locked-test behaviour.]

### 5.3 Registered factorial contrasts

| Contrast | Clean macro-F1 difference | 95% paired-bootstrap interval | Robustness summary difference |
|---|---:|---:|---:|
| ENB0 − R18 under MIN | TBD | TBD | TBD |
| ENB0 − R18 under CAM | TBD | TBD | TBD |
| CAM − MIN for R18 | TBD | TBD | TBD |
| CAM − MIN for ENB0 | TBD | TBD | TBD |
| architecture × augmentation interaction | TBD | TBD | TBD |

[Interpret magnitude and uncertainty. An interval containing zero does not
prove equivalence.]

### 5.4 Per-class performance

| Cell | Class | Precision | Recall | F1 | Support | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| [ID] | crazing | TBD |  TBD | TBD | 60 | TBD | TBD |
| [ID] | inclusion | TBD | TBD | TBD | 60 | TBD | TBD |
| [ID] | patches | TBD | TBD | TBD | 60 | TBD | TBD |
| [ID] | pitted surface | TBD | TBD | TBD | 60 | TBD | TBD |
| [ID] | rolled-in scale | TBD | TBD | TBD | 60 | TBD | TBD |
| [ID] | scratches | TBD | TBD | TBD | 60 | TBD | TBD |

[Generate rows for each cell/seed or clearly identified aggregate. Insert raw
and row-normalised confusion matrices using the same class order and scale.]

### 5.5 Robustness results

Use long format so severity trends remain visible. Repeat for all cells/seeds
or state the aggregation explicitly.

| Cell | Corruption | Severity | Macro-F1 | Absolute drop from same clean checkpoint | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| [ID] | [key] | 1 | TBD | TBD | TBD |
| [ID] | [key] | 2 | TBD | TBD | TBD |
| [ID] | [key] | 3 | TBD | TBD | TBD |

| Cell | Clean macro-F1 | Mean over 27 corrupted conditions | Mean absolute drop | Worst condition | Worst drop |
|---|---:|---:|---:|---|---:|
| R18-MIN | TBD | TBD | TBD | TBD | TBD |
| R18-CAM | TBD | TBD | TBD | TBD | TBD |
| ENB0-MIN | TBD | TBD | TBD | TBD | TBD |
| ENB0-CAM | TBD | TBD | TBD | TBD | TBD |

[Include severity curves. Avoid calling the 27 paired derivatives 9,720 new
test samples; the underlying sample count remains 360.]

### 5.6 Efficiency

| Architecture | Parameters | Checkpoint bytes | Median latency | P90 latency | Device |
|---|---:|---:|---:|---:|---|
| ResNet-18 | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | TBD | TBD | TBD | TBD | TBD |

[Report batch size 1, 10 warm-up calls, 50 timed calls, software stack, and
hardware. Do not compare latencies measured on different devices.]

## 6. Error analysis

### 6.1 One-vs-rest false positives and false negatives

| Cell | Predicted class | FP | FP rate `FP/(FP+TN)` | Main true-class source | FN |
|---|---|---:|---:|---|---:|
| [ID] | [class] | TBD | TBD | TBD | TBD |

[For each defect type, show representative high-confidence FPs and FNs with
filename, true class, prediction, and confidence. Note whether errors recur
across seeds/cells. Do not describe these as alarms on normal steel.]

### 6.2 Visually similar defect pairs

| Pair | `a→b` | `b→a` | Symmetric total | Evidence-based interpretation |
|---|---:|---:|---:|---|
| crazing / rolled-in scale | TBD | TBD | TBD | TBD |
| crazing / pitted surface | TBD | TBD | TBD | TBD |
| rolled-in scale / pitted surface | TBD | TBD | TBD | TBD |
| [largest empirical pair] | TBD | TBD | TBD | TBD |

[Discuss texture scale, line orientation, spatial extent, local contrast,
background/exposure cues, and effects of blur/noise. Distinguish observation
from a causal claim.]

### 6.3 Confidence, calibration, and explanations

[Compare confidence distributions for correct and incorrect predictions,
include the reliability diagram, and discuss confident errors. Add a small
Grad-CAM case study only as qualitative evidence. Softmax confidence and
Grad-CAM are not guarantees or causal explanations.]

## 7. Discussion

### 7.1 Answers to RQ1–RQ4

[Answer each question using the registered contrasts, seed variation, paired
bootstrap intervals, and error evidence.]

### 7.2 Relation to image formation

[Connect observed sensitivity to illumination/exposure, tone mapping, optics,
motion, sensor noise, and compression. The classifier observes an image-pipeline
output, not the physical surface directly.]

### 7.3 Practical implications

[Explain which wrong defect-type assignments matter and what additional data
would be required for normal-versus-defect screening, localisation, threshold
selection, new-camera validation, and deployment.]

### 7.4 Limitations

Address at minimum:

- no normal class and no production false-alarm estimate;
- pre-cropped known-defect classification rather than detection/localisation;
- small balanced benchmark versus real prevalence;
- one quarantined exact duplicate and only partially recoverable source groups;
- grayscale data with limited camera metadata;
- synthetic corruptions rather than calibrated hardware or external-domain
  validation;
- only two compact pretrained CNN families; and
- uncertainty from three seeds and 360 test items.

## 8. Conclusion

[Summarise the verified architecture effect, augmentation effect, most
important confusion, robustness limitation, and defensible next step. Include
numbers only when linked to generated evidence.]

## 9. Reproducibility statement

- Git revision and dirty status: `[value]`
- `uv.lock` SHA-256: `[value]`
- Python/PyTorch/torchvision/CUDA or MPS/OS: `[runtime artifacts]`
- Raw archive acquisition record: `data/archives/NEU-CLS.rar.metadata.json`
- Final manifest: `data/manifests/neu_cls_v2.csv`
- Final manifest SHA-256: `[value]`
- Dataset fingerprint: `[verified audit value]`
- Reviewed group map: `data/manifests/neu_cls_reviewed_groups_v2.csv`
- Config paths and hashes: `[values]`
- Training seeds: `42`, `43`, `44`
- Pretrained weight identifiers: `[values]`
- Checkpoint/prediction hashes: `[values]`
- Validation command and date: `[value]`
- Locked-test command, acknowledgement, and date: `[value]`
- Known nondeterministic operations: `[value]`

## 10. Team contributions

| Contributor | Data/audit | Models | Experiments | Evaluation | Writing/review |
|---|---|---|---|---|---|
| [name] | [work] | [work] | [work] | [work] | [work] |

## References

Song, K., & Yan, Y. (2013). A noise robust method based on completed local
binary patterns for hot-rolled steel strip surface defects. *Applied Surface
Science, 285*, 858–864.
[https://doi.org/10.1016/j.apsusc.2013.09.002](https://doi.org/10.1016/j.apsusc.2013.09.002)

[Add official ResNet, EfficientNet, PyTorch/torchvision, metric, calibration,
and explanation references actually used.]

## Appendix A — Protocol deviations

| Date/time | Decision | Reason | Made before test access? | Affected runs | Evidence |
|---|---|---|---|---|---|
| [date] | [change or “none”] | [reason] | [yes/no] | [IDs] | [path] |

## Appendix B — Artifact inventory

| Artifact | Path | SHA-256 | Generated by | Verified by |
|---|---|---|---|---|
| final manifest | `data/manifests/neu_cls_v2.csv` | TBD | `neu-prepare` + reviewed map | TBD |
| resolved config | TBD | TBD | `neu-train` | TBD |
| checkpoint | TBD | TBD | `neu-train` | TBD |
| clean predictions | TBD | TBD | `neu-evaluate` | TBD |
| robustness predictions | TBD | TBD | `neu-robustness` | TBD |
| registered comparison | TBD | TBD | `neu-compare` | TBD |
