# Environments

## Recorded experiment

[kaggle_recorded.json](kaggle_recorded.json) is the original runtime record. The experiment used Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA 12.8 on one Tesla T4. Training used CUDA mixed precision; test inference and latency measurements used FP32. [kaggle_recorded.txt](kaggle_recorded.txt) lists the recorded packages and is not a portable CUDA installation recipe.

The executed Kaggle notebook retains compatible preinstalled CUDA packages, pins the reviewed data packages and checks compatibility before reusing historical seed-43 checkpoints. If the hosted image changes incompatibly, the notebook stops. A new supported environment does not retroactively replace the recorded experiment.

## Local inference and checks

`uv.lock` is the dependency lock for repository installation. From the repository root:

```bash
uv sync --locked --python 3.11 --extra dev --extra data --extra notebook
uv run --no-sync python --version
uv run --no-sync python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

The [local inference verification record](local_inference_verified.json) describes the machine and packages used for the repository smoke-test. It is separate from Kaggle training metadata and from the presentation laptop smoke-test. CPU inference does not require CUDA. Package metadata accepts Python 3.11 through 3.13; the exact environment tested locally is recorded rather than assuming identical results on every platform.

For complete corruption replay, compare saved input fingerprints as well as predictions. A prior macOS replay differed from Kaggle in the two weaker Pillow contrast transforms by at most one grayscale level; this can affect exact input hashes. The clean and Gaussian-blur presentation examples avoid those contrast transforms. Do not replace frozen Kaggle metrics with a cross-platform replay without checking the inputs.
