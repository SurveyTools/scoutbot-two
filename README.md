# Wild Me ScoutBot

[![Tests](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml)

YOLOv8-based machine learning pipeline for detecting and localising wildlife in aerial survey imagery.

## How to Install

From source:

```bash
git clone https://github.com/SurveyTools/scoutbot-2026
cd scoutbot-2026
pip install -e .
```

Models are downloaded automatically on first use to `~/.cache/scoutbot/`. To pre-fetch explicitly:

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

### CLI Options

| Option | Values | Default |
|---|---|---|
| `--config` | `v3`, `v3-cls` | `v3` |
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
