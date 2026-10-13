# Industrial Surface Defect Classification

Introduction to Computer Vision | Team P03 | Detailed implementation notes | 13 October 2026

Submission summary: [concise baseline milestone report](https://github.com/kujifined/Introduction-to-cv/blob/main/reports/baseline_milestone_summary_20261013.md).

## 1. Current implementation

We have implemented the NEU-CLS data loading and preprocessing pipeline and completed a ResNet-18 baseline run on Kaggle. The task is six-class classification of industrial surface defects. Our baseline uses ImageNet-pretrained ResNet-18 with basic preprocessing and no random augmentation. Macro-F1 is our primary evaluation measure.

This report covers one training seed. It demonstrates a working baseline, not the completed architecture and augmentation comparison from our project plan.

Repository: [kujifined/Introduction-to-cv](https://github.com/kujifined/Introduction-to-cv).

Kaggle notebook and run output: [ResNet-18 baseline, seed 43](https://www.kaggle.com/code/datastr3am/icv-p03-resnet18-baseline-seed-43).

The Kaggle run used the project code from our GitHub repository. All experiments share the same data loaders, training loop and evaluation code, with the model and augmentation settings selected through configuration files.

## 2. Data loading and preprocessing

NEU-CLS contains six defect categories: crazing, inclusion, patches, pitted surface, rolled-in scale and scratches. The original dataset has 1,800 grayscale images at 200 x 200 pixels. It has no normal-surface class, so we classify defect types rather than detect whether a surface is defective.

Our audit excluded one extra exact copy, leaving 1,799 active images. We also reviewed near-duplicate candidates and grouped four confirmed shifted duplicate pairs before splitting. Exact duplicate hashes and reviewed groups cannot appear in different active splits.

The fixed, class-stratified split uses seed 42:

| Split | Images | Images per class |
|---|---:|---|
| Training | 1,259 | 210, except patches: 209 |
| Validation | 180 | 30 |
| Test | 360 | 60 |

The split is stored in `data/manifests/neu_cls_v2.csv`. Each row records the image path, class, class index, split, SHA-256 hash, group and status. The loader checks the manifest structure, class mapping and cross-split duplicate constraints. Training and evaluation entry points also verify that the files exist and match their recorded hashes. All 1,800 original files, including the excluded copy, passed this integrity check on Kaggle. The split seed remains 42; the training seed for this run is 43.

`ManifestDataset` reads images with Pillow and converts them to grayscale. Baseline preprocessing replicates the grayscale channel into three channels, resizes to 224 x 224 with bilinear interpolation and antialiasing, converts pixels to a float tensor, and applies ImageNet normalization. The channel means are `(0.485, 0.456, 0.406)` and standard deviations are `(0.229, 0.224, 0.225)`.

The baseline uses no random augmentation. Training batches are shuffled with a seeded generator; validation batches are not shuffled. The loader returns image tensors and integer labels, and evaluation additionally returns image paths for item-level predictions. Camera-aware augmentation is implemented as a separate training-only option but was not used in this run.

## 3. Baseline configuration and training setup

We used torchvision ResNet-18 initialized with ImageNet weights (`ResNet18_Weights.IMAGENET1K_V1`). Its final layer was replaced with a linear layer mapping 512 features to six class logits. All layers were trained. The model has 11,179,590 parameters.

| Setting | Actual value for this run |
|---|---|
| Training seed | 43 |
| Maximum epochs / completed epochs | 50 / 13 |
| Input shape | 3 x 224 x 224 |
| Batch size | 32 |
| Data-loader workers | 2 |
| Optimizer | AdamW |
| Base learning rate | 0.0001 |
| Schedule | Three warm-up epochs, then cosine decay; minimum LR 0.000001 |
| Weight decay | 0.0001 |
| Training loss | Cross-entropy with label smoothing 0.05 |
| Validation loss | Cross-entropy without label smoothing |
| Training device | One NVIDIA Tesla T4 on Kaggle |
| Mixed precision | Enabled for CUDA training |
| Early stopping | 10 epochs without a strict improvement in validation macro-F1 |
| Selected checkpoint | Epoch 3 |

The recorded learning rates were 0.00001, 0.00004 and 0.00007 during the first three epochs, reaching 0.0001 at epoch 4. Validation macro-F1 first reached 1.0000 at epoch 3. Later epochs matched this score but did not improve it, so the earlier checkpoint was retained and training stopped after epoch 13. Checkpoint selection uses macro-F1, not validation loss.

The timed training/validation stage and output generation took 46.76 seconds, excluding environment setup, dataset verification and model initialization. The saved runtime records Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, NumPy 2.1.3 and scikit-learn 1.6.1. The run saved its resolved configuration, code revision, dataset-manifest hash, epoch history and best checkpoint.

## 4. Initial quantitative results

We evaluated the selected epoch-3 checkpoint on all 180 validation images. The saved prediction file contains one row per image, including the true class, predicted class and six class probabilities.

| Measure | Result |
|---|---:|
| Training accuracy accumulated during epoch 3 | 99.68% |
| Training loss during epoch 3, with label smoothing | 0.2817 |
| Validation accuracy | 100.00% (180/180) |
| Validation macro-F1 | 1.0000 |
| Validation balanced accuracy | 1.0000 |
| Validation cross-entropy, without label smoothing | 0.0458 |
| Validation expected calibration error, 10 bins | 0.0440 |

Each class had precision, recall and F1 of 1.0000 on its 30 validation images. The confusion matrix contains 30 correct predictions per class and no off-diagonal entries.

The learning history shows how the baseline reached this result:

| Epoch | Training accuracy | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|
| 1 | 58.86% | 77.22% | 0.7616 |
| 2 | 97.93% | 99.44% | 0.9944 |
| 3, selected checkpoint | 99.68% | 100.00% | 1.0000 |
| 13, final training epoch | 100.00% | 100.00% | 1.0000 |

Training accuracy is accumulated while model parameters change during the epoch, whereas validation uses the checkpoint in evaluation mode. Training loss also includes label smoothing, unlike validation loss. These values are not directly comparable.

We independently recomputed accuracy, macro-F1 and cross-entropy from the exported predictions, checked every validation path and label against the fixed manifest, and verified the checkpoint SHA-256. We also loaded the downloaded checkpoint on CPU and repeated validation inference using separate Pillow/NumPy preprocessing, without the project data loader or metric implementation. All 180 predicted classes matched the Kaggle output. This was an inference check only, not another training run.

On the Tesla T4, model-forward latency at batch size 1 was 3.06 ms per image at the median and 3.20 ms at the 90th percentile, after 10 warm-up passes and 50 timed passes. This excludes loading, preprocessing and model initialization, and is not an end-to-end deployment benchmark.

The test set remains reserved for final evaluation. Its raw files were checked for integrity, but it was not used for training, checkpoint selection or performance evaluation. We report no test accuracy, robustness scores or multi-seed averages at this stage.

## 5. Problems and observations

- **Duplicates required attention before splitting.** An exact copy and shifted duplicate pairs could otherwise inflate validation or test results. We excluded the extra exact copy, grouped confirmed pairs and saved the review decisions. These checks reduce known leakage; they do not prove independence of all underlying acquisition sources.
- **Grayscale inputs needed adaptation.** ImageNet-pretrained ResNet-18 expects three channels. Replicating the grayscale channel keeps the original defect texture while making the input compatible with the model.
- **Perfect validation performance needs careful interpretation.** The result was reproduced from the saved checkpoint, and prediction paths match the validation manifest with no train or test path overlap. However, validation contains only 30 images per class from the same dataset and was used to select the checkpoint. A score of 1.0000 does not establish performance on new acquisition conditions or industrial data. Independent inference confirms the reported result, not the absence of every possible source of dataset bias.
- **The baseline does not cover the full experiment matrix.** One seed and one model/augmentation combination cannot establish differences between architectures or robustness under changed acquisition conditions. An earlier one-epoch local pipeline check used a different training budget and runtime, so we do not combine it with this run as a multi-seed experiment.
- **No validation failure examples occurred in this checkpoint evaluation.** We will not invent error cases. The planned failure analysis will be based on observed mistakes in later registered evaluations.

All 88 automated tests passed in the actual Kaggle environment. The test log contains 14 dependency deprecation warnings and no test failures. Every raw image matched the saved manifest hash. These implementation checks are separate from model-performance results.

## 6. Team contributions and milestone status

The work followed the division of responsibilities in our previous project plan:

- **Danil Khasanshin, team leader** (d.khasanshin@innopolis.university): coordinated the work, integrated the shared pipeline and evaluation code, consolidated the baseline results and prepared this report.
- **Timur Kagarmanov** (t.kagarmanov@innopolis.university): prepared and audited the dataset, checked exact and near-duplicate images, verified the split and carried out exploratory data analysis.
- **Ruslan Nasibullin** (ru.nasibullin@innopolis.university): ran the initial ResNet-18 baseline with basic preprocessing, analysed its validation results and documented the training setup. He remains responsible for the ResNet-18 experiments with both augmentation settings.
- **Nikita Selezenev** (n.selezenev@innopolis.university): prepared the EfficientNet-B0 configurations for the shared pipeline. Its training and result analysis are planned for the next stage; he remains responsible for both augmentation settings.

At this milestone, the shared data loading, preprocessing and evaluation pipeline is implemented and verified. A Kaggle GPU run of ResNet-18 with basic preprocessing is complete, and its validation results have been independently rechecked. EfficientNet-B0 and camera-aware augmentation configurations are implemented, but their planned training experiments have not yet been run.

Architecture-specific responsibilities refer to running and analysing experiments through the same shared code, not maintaining separate training implementations. All team members share responsibility for reviewing the results, discussing errors and limitations, and preparing the final report and presentation.

## 7. Next steps

We will complete the registered ResNet-18 and EfficientNet-B0 experiments with both augmentation settings and seeds 42, 43 and 44 on a cloud GPU, retaining this seed-43 baseline as one completed configuration. After validation-based decisions are frozen, we will evaluate the selected checkpoints on the reserved test set. We will report both absolute macro-F1 on corrupted images and its drop from clean-image performance, then analyse observed false positives and visually similar defect types. The final failure analysis will inspect at least three observed failure cases and explain their likely causes.
