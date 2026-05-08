# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScoutBot is a Python package for detecting wildlife in high-resolution aerial imagery. It wraps YOLOv8 with SAHI (Slicing Aided Hyper Inference) to handle large images by slicing them into 512×512 tiles (25% overlap), running batched YOLOv8 inference, then merging results with NMS.

Public API lives in `scoutbot/__init__.py`: `pipeline()`, `batch()`, `batch_resumable()`, `fetch()`, `get_classes()`.
CLI is defined in `scoutbot/scoutbot.py` using Click.

## Commands

```bash
# Install (editable, with dev/test extras)
pip install -e ".[test]"
# or
pip install -r requirements.txt && pip install -r requirements.optional.txt

# Run all tests
pytest

# Run a single test
pytest tests/test_loc.py::TestLoc::test_configs -v

# Run tests matching a pattern
pytest -k "test_pipeline" -v

# With coverage
pytest --cov=scoutbot

# Pre-commit linting (also runs on git commit)
pre-commit run --all-files
```

## Architecture

### Pipeline data flow
1. `loc.fetch(config)` — downloads the YOLOv8 `.pt` model from Azure Blob Storage to `~/.cache/scoutbot/`, verified by MD5.
2. `Yolov8DetectionModel` (`tile_batched/main.py`) — SAHI wrapper; slices the image, runs batched YOLOv8, merges detections.
3. `pipeline()` returns `(wic_score: float, detections: list[dict])` where each detection has keys `l`, `c`, `x`, `y`, `w`, `h`.

### Module map
- `scoutbot/loc/__init__.py` — `CONFIGS` dict (model URLs, thresholds, slice params) and `fetch()`.
- `scoutbot/tile_batched/main.py` — `Yolov8DetectionModel`, the SAHI `DetectionModel` subclass.
- `scoutbot/utils.py` — Rich + file logging setup.

### Model configs (`CONFIGS` keys)
- `v3` — single-class KAZA detector (`yolov8-onecls.kaza.pt`)
- `v3-cls` — classification variant

Default config is `v3`; overridable via `LOC_CONFIG` or `CONFIG` env vars, or the `--config` CLI flag.

## Key Env Vars

| Variable | Default | Effect |
|---|---|---|
| `VERBOSE` | unset | Enable debug logging |
| `LOC_CONFIG` / `CONFIG` | `v3` | Select model config |
| `DETECTOR_BATCH_SIZE` | `128` | YOLOv8 inference batch size |

## Code Style

Formatting: Brunette (Black fork). Linting: flake8. Imports sorted by isort. All enforced via pre-commit hooks — run `pre-commit run --all-files` before committing. Flake8 config and ignore list are in `setup.cfg`.
