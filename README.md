# Wild Me ScoutBot

[![Tests](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml)

YOLOv8-based machine learning pipeline for detecting and localising wildlife in aerial survey imagery.

## How to Install

### Linux / macOS

```bash
git clone https://github.com/SurveyTools/scoutbot-2026
cd scoutbot-2026
pip install -e .
```

### Windows (NVIDIA GPU)

The default install pulls in a CPU-only PyTorch build on Windows. After cloning, run these steps in order:

```powershell
git clone https://github.com/SurveyTools/scoutbot-2026
cd scoutbot-2026
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .

# Replace CPU-only PyTorch with the CUDA build (requires NVIDIA GPU + driver >= 525)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# Pin numpy to a version compatible with ultralytics
pip install "numpy<2.0.0" --force-reinstall

# Required for large image handling (SAHI tiling)
pip install scikit-image imagecodecs
```

Verify the GPU is detected before running:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# Expected: True  2.x.x+cu124
```

**Running batch on Windows** — PowerShell does not expand glob patterns for external commands, so use `Get-ChildItem` to pass files:

```powershell
$imgs = (Get-ChildItem "C:\path\to\images" -Filter "*.JPG").FullName
scoutbot batch @imgs --output results.json
```

For low-VRAM GPUs (e.g. 4 GB), reduce the tile batch size:

```powershell
$env:DETECTOR_BATCH_SIZE = "16"
scoutbot batch @imgs --output results.json
```

Models are downloaded automatically on first use to `~/.cache/scoutbot/` (Windows: `C:\Users\<user>\.cache\scoutbot\`). To pre-fetch explicitly:

```bash
scoutbot fetch --config v3
scoutbot fetch --config v3-cls
```

## How to Run

### CLI — Single Image

```bash
scoutbot pipeline /path/to/image.jpg --config v3 --device mps

# Save results to a JSON file
scoutbot pipeline /path/to/image.jpg --config v3-cls --output results.json
```

### CLI — Batch

```bash
# Process a directory of images, print results to stdout
scoutbot batch --config v3 --device mps images/*.jpg

# Save to JSON
scoutbot batch --config v3 --device mps --output results.json images/*.jpg
```

### CLI — Resumable Batch

For large surveys. Results are written incrementally to a JSONL file. Re-running the same command resumes from where it stopped:

```bash
scoutbot batch-resumable --output survey.jsonl --config v3 --device mps images/*.jpg
```

### CLI — Other Commands

```bash
scoutbot fetch --config v3            # download model
scoutbot get_classes --config v3-cls  # list species labels
scoutbot example                      # run on a built-in test image
```

### CLI — Core ML (Apple Silicon, macOS)

On M-series Macs, install the `coreml` extra and use the `v3-coreml` or `v3-cls-coreml` config to route inference through Apple's Neural Engine (~7–13× faster than CPU):

```bash
uv sync --extra coreml

scoutbot pipeline image.jpg --config v3-coreml
scoutbot batch *.jpg --config v3-coreml --no-progress
```

The `.mlpackage` bundle is exported on first use (~10–30 s) and cached at `~/.cache/scoutbot/`. See [docs/coreml.rst](docs/coreml.rst) for benchmarks and full details.

### CLI Options

| Option | Values | Default |
|---|---|---|
| `--config` | `v3`, `v3-cls`, `v3-coreml`, `v3-cls-coreml` | `v3` |
| `--device` | `mps`, `cpu`, `cuda`, `cuda:0` | `cuda:0` |
| `--loc-thresh` | 0–100 (integer) | `45` |
| `--output` | file path | stdout |

The default device can be set via the `SCOUTBOT_DEVICE` environment variable:

```bash
export SCOUTBOT_DEVICE=mps   # Apple Silicon
scoutbot batch --config v3 images/*.jpg
```

## Python API

```python
import scoutbot

# Single image
wic_score, detections = scoutbot.pipeline('/path/to/image.jpg', config='v3')

# Batch (reuses loaded model)
wic_list, detects_list = scoutbot.batch(filepaths, config='v3', backend_device='mps')

# Resumable batch — writes JSONL incrementally
summary = scoutbot.batch_resumable(
    filepaths=filepaths,
    output_path='survey.jsonl',
    config='v3',
    backend_device='mps',
)

# List supported species
classes = scoutbot.get_classes(config='v3-cls')
```

Each detection in the output list is a dict:

```python
{
    'l': 'elephant',   # label (str)
    'c': 0.832,        # confidence (float, 0–1)
    'x': 4596,         # left edge (pixels)
    'y': 2322,         # top edge (pixels)
    'w': 77,           # width (pixels)
    'h': 148,          # height (pixels)
}
```

The `wic` score is the maximum detection confidence across all detections in the image (0.0 if no detections).

## Models

Two models are available, both trained on KAZA aerial survey data:

| Config | File | Description |
|---|---|---|
| `v3` | yolov8-onecls.kaza.pt | Single-class detector (label: `detection`) |
| `v3-cls` | yolov8-cls.kaza.pt | Multi-class species classifier (47 species) |

Models are fetched from Azure Blob Storage with MD5 integrity verification and cached at `~/.cache/scoutbot/`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SCOUTBOT_DEVICE` | `cuda:0` | Inference device: `mps`, `cpu`, `cuda`, `cuda:0` |
| `LOC_CONFIG` | `v3` | Model config: `v3` or `v3-cls`. Alias: `CONFIG` |
| `DETECTOR_BATCH_SIZE` | `128` | Slices per GPU batch. Reduce for low-memory environments (e.g. `8` for CI) |
| `VERBOSE` | unset | Set to any value to enable debug logging |

## Tests and Coverage

```bash
pip install -r requirements.optional.txt
pytest
```

For a coverage report:

```bash
coverage run -m pytest
coverage html
open coverage/html/index.html
```

## Building Documentation

```bash
pip install -r requirements.optional.txt
cd docs/
sphinx-build -M html . build/
open build/html/index.html
```

## Code Formatting

```bash
pip install -r requirements.optional.txt
pre-commit install          # run linting on every commit
pre-commit run --all-files  # run manually
```

The codebase uses [Brunette](https://pypi.org/project/brunette/) (a configurable Black fork) with `flake8` for linting. See `setup.cfg` for style rules.
