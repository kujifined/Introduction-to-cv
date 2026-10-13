# Industrial Surface Defect Classification

Introduction to Computer Vision | Team P03 | Baseline milestone | 13 October 2026

Repository: [kujifined/Introduction-to-cv](https://github.com/kujifined/Introduction-to-cv).

Kaggle notebook and run output: [ResNet-18 baseline, seed 43](https://www.kaggle.com/code/datastr3am/icv-p03-resnet18-baseline-seed-43).

## 1. Task and implemented pipeline

We classify six types of steel surface defects using NEU-CLS. This milestone presents our first working baseline: ImageNet-pretrained ResNet-18 with basic preprocessing. Macro-F1 is the primary metric. The architecture and augmentation comparison is planned for the next stage.

The repository implements dataset acquisition and verification, manifest-based loading, preprocessing, training, checkpoint selection and evaluation. Experiments share the same data loaders, training loop and evaluation code; model and augmentation settings are selected through configuration files.

## 2. Data loading and preprocessing

NEU-CLS contains 1,800 grayscale images of size 200 x 200: crazing, inclusion, patches, pitted surface, rolled-in scale and scratches. It contains defect crops only, with no normal-surface class.

We checked all 1,800 file hashes, excluded one extra exact duplicate and grouped four confirmed shifted duplicate pairs before splitting. The active dataset has 1,799 images. The fixed, class-stratified split uses seed 42: 1,259 training, 180 validation and 360 test images. Validation has 30 images per class, test has 60, and training has 210 except patches, which has 209. Exact duplicates and reviewed source groups cannot cross active splits.

The loader reads paths and labels from the saved manifest, verifies file hashes and class mappings, and opens images with Pillow. Preprocessing replicates grayscale into three channels, resizes to 224 x 224 using antialiased bilinear interpolation, converts to float tensors and applies ImageNet normalization: mean (0.485, 0.456, 0.406), standard deviation (0.229, 0.224, 0.225). Training batches are shuffled with a seeded generator. Validation is deterministic. This baseline uses no random augmentation.

## 3. Model configuration and training setup

We used torchvision ResNet-18 with ImageNet weights, replaced its 512-feature classifier with a six-class linear layer and fine-tuned all 11,179,590 parameters. Training used seed 43, batch size 32, two loader workers, AdamW, learning rate 0.0001 and weight decay 0.0001. The training loss was cross-entropy with label smoothing 0.05; validation used unsmoothed cross-entropy.

The schedule allowed 50 epochs, with three warm-up epochs followed by cosine decay to a minimum learning rate of 0.000001. Early stopping used validation macro-F1 with patience 10. Training stopped after 13 epochs; epoch 3 was retained as the earliest checkpoint achieving the best score.

Training ran on one NVIDIA Tesla T4 on Kaggle with CUDA mixed precision, using Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA 12.8. The execution procedure is provided in the linked Kaggle notebook. The timed training/validation stage and output generation took 46.76 seconds, excluding setup, data verification and model initialization.

## 4. Initial results

The selected checkpoint was evaluated on all 180 validation images.

| Measure | Result |
|---|---:|
| Training accuracy accumulated during epoch 3 | 99.68% |
| Validation accuracy | 100.00% (180/180) |
| Validation macro-F1 | 1.0000 |
| Validation balanced accuracy | 1.0000 |
| Validation cross-entropy | 0.0458 |
| Validation expected calibration error, 10 bins | 0.0440 |

Each class had precision, recall and F1 of 1.0000 on 30 images. Validation macro-F1 progressed from 0.7616 at epoch 1 to 0.9944 at epoch 2 and 1.0000 at epoch 3. Exported predictions matched the validation manifest, and independent CPU inference from the saved checkpoint reproduced all 180 predicted classes. All 88 implementation tests passed on Kaggle.

Median model-forward latency was 3.06 ms per image on the T4 at batch size 1, after 10 warm-up and 50 timed passes. This excludes data loading and preprocessing.

## 5. Problems, interpretation and next steps

Duplicate images required exclusion or grouping to prevent known cross-split leakage. Grayscale inputs required channel replication for the pretrained model. Training and validation losses are not directly comparable because only training uses label smoothing.

The perfect score needs caution: validation contains only 30 images per class from the same dataset and was used for checkpoint selection. It does not establish performance under new acquisition conditions. There were no validation errors for this checkpoint, so no failure examples are claimed. The test set has not been evaluated.

Next, we will compare ResNet-18 and EfficientNet-B0 with and without camera-aware augmentation across the planned seeds. After fixing experimental decisions, we will evaluate clean and corrupted test images and analyse at least three observed failure cases. These comparisons and robustness results are not part of this milestone.

## 6. Team contributions

- **Danil Khasanshin, team leader** (d.khasanshin@innopolis.university): coordination, shared-pipeline integration, evaluation and report assembly.
- **Timur Kagarmanov** (t.kagarmanov@innopolis.university): dataset preparation, duplicate audit, split verification and exploratory analysis.
- **Ruslan Nasibullin** (ru.nasibullin@innopolis.university): ResNet-18 baseline execution and analysis; responsible for its next-stage experiments.
- **Nikita Selezenev** (n.selezenev@innopolis.university): EfficientNet-B0 configurations; its training and analysis remain planned.
