# OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScoutBot is a Python package for detecting wildlife in high-resolution aerial imagery. It wraps YOLOv8 with SAHI (Slicing Aided Hyper Inference) to handle large images by slicing them into 512×512 tiles (25% overlap), running batched YOLOv8 inference, then merging results with NMS.

Public API lives in `scoutbot/__init__.py`: `pipeline()`, `batch()`, `batch_resumable()`, `fetch()`, `get_classes()`.
CLI is defined in `scoutbot/scoutbot.py` using Click.

## Runtime Rule: Always Use `uv`

This project uses `uv` as its Python runner. **Never run `python`, `pytest`, `ruff`, `mypy`, or similar tools directly.** Always prefix with `uv run`:

```bash
uv run pytest -m "not slow"
uv run python -m scoutbot ...
uv run mypy scoutbot
uv run ruff check .
```

## Commands

```bash
# Install (editable, with dev/test extras)
uv sync --extra test

# Run fast/offline tests
uv run pytest -m "not slow"

# Run slow tests that download models or run real inference
uv run pytest -m "slow"

# Run a single test
uv run pytest tests/test_loc.py::TestLoc::test_configs -v

# Run tests matching a pattern
uv run pytest -k "test_pipeline" -v

# With coverage
uv run pytest -m "not slow" --cov=scoutbot

# Quality checks
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv run pre-commit run --all-files
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
- `v3-cls` — multi-class species detector (`yolov8-cls.kaza.pt`)
- `v3-coreml` — Core ML export of `v3` for macOS
- `v3-cls-coreml` — Core ML export of `v3-cls` for macOS

Default config is `v3`; overridable via `LOC_CONFIG` or `CONFIG` env vars, or the `--config` CLI flag.

## Key Env Vars

| Variable | Default | Effect |
|---|---|---|
| `VERBOSE` | unset | Enable debug logging |
| `LOC_CONFIG` / `CONFIG` | `v3` | Select model config |
| `DETECTOR_BATCH_SIZE` | `128` | YOLOv8 inference batch size |

## Code Style

Formatting and linting: Ruff. Type checking: scoped mypy on `scoutbot/`. Tests use pytest with
the `slow` marker for model-download and real-inference tests. Tool configuration lives in
`pyproject.toml`; `setup.cfg` is not the quality-tool source of truth.

```bash
uv run pytest -m "not slow"
uv run pytest -m "slow"
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv run pre-commit run --all-files
```
