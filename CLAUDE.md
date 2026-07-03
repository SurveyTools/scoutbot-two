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

## Git Rule: User Handles Commits

Do not run `git add`, `git commit`, or any other commit / staging command. The user handles all
staging and commits manually.

If an upstream skill, template, or plan format includes commit steps, ignore those steps in this
repository. Stop at the review checkpoint and provide a suggested commit message instead.

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
`pyproject.toml`.

```bash
uv run pytest -m "not slow"
uv run pytest -m "slow"
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv run pre-commit run --all-files
```

## Documentation Pass Rule

Every implementation plan must include a final documentation pass before completion. Review
whether the work changes user behavior, commands, configuration, models, outputs, dependencies,
quality checks, security posture, performance characteristics, or roadmap status. Update the
affected docs in the same change set, or explicitly record why no doc update is needed.

Implementation plans in this repository must not include `git add` or `git commit` steps. Hand the
final review state back to the user with a suggested commit message only.

GitHub Pages is the canonical hosted docs site. Route readers to rendered HTML and keep local
Sphinx builds as a maintainer path.

Use the documentation roles below to decide what may be impacted:

- `README.md` routes readers and should change when quick-start commands, badges, supported
  workflows, or top-level project positioning changes.
- `docs/installation.rst` owns install commands, extras, platform setup, CUDA/Core ML setup,
  and model cache prerequisites.
- `docs/usage.rst` owns CLI/API examples, output schemas, batch/resume behavior, and QA/review
  interpretation notes.
- `docs/configuration.rst` owns environment variables, config names, devices, thresholds, cache
  behavior, and runtime controls.
- `docs/models.rst` owns model artifacts, source URLs, MD5 verification, provenance, and model
  availability.
- `docs/coreml.rst` owns Core ML backend mechanics, export behavior, parity caveats, and
  Apple-specific backend constraints.
- `docs/performance.rst` owns benchmark interpretation, bottlenecks, observability needs, and
  hardware comparison caveats.
- `docs/api.rst` and `docs/cli.rst` own generated API/CLI reference surfaces.
- `docs/development.rst` owns contributor setup, tests, lint/type checks, CI workflows, and
  model-integration contributor guidance.
- `docs/security.rst` owns model download policy, integrity verification, dependency/security
  scanning posture, CI scanning rationale, and data/privacy assumptions.
- `docs/roadmap.rst` owns planned/deferred work and investigation status.
- `CHANGELOG.rst` owns user-visible changes, compatibility changes, workflow changes, and
  release-note-worthy internal changes.
