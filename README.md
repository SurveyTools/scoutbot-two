# Scoutbot

[![Tests](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml)
[![Quality](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/quality.yml)

YOLOv8-based machine learning pipeline for detecting and localising wildlife in aerial survey imagery.

## Acknowledgement

Scoutbot-2026 is a continuation of the work started in the original
[WildMeOrg/scoutbot](https://github.com/WildMeOrg/scoutbot) repository. Scoutbot was originally
developed for the [Scout](https://github.com/WildMeOrg/scout) project by the Wild Me team, now
part of Conservation X Labs, and was funded by the Paul G. Allen Family Foundation.

## Quick Start

```bash
git clone https://github.com/SurveyTools/scoutbot-2026
cd scoutbot-2026
uv pip install -e .

uv run scoutbot fetch --config v3
uv run scoutbot pipeline /path/to/image.jpg --config v3 --device mps
```

Scoutbot downloads models on first use and caches them in `~/.cache/scoutbot/`.

## Where to Go Next

| I want to... | Start here |
|---|---|
| Run Scoutbot directly on images | [Installation](docs/installation.rst), [Usage](docs/usage.rst), [Configuration](docs/configuration.rst) |
| Integrate Scoutbot into a larger pipeline | [Usage](docs/usage.rst), [API reference](docs/api.rst), [CLI reference](docs/cli.rst) |
| Understand outputs and analysis limits | [Usage](docs/usage.rst), [Configuration](docs/configuration.rst), [Performance](docs/performance.rst) |
| Contribute code or prepare releases | [Development](docs/development.rst), [Roadmap](docs/roadmap.rst), [Changelog](CHANGELOG.rst) |
| Add models or investigate performance | [Models](docs/models.rst), [Core ML](docs/coreml.rst), [Performance](docs/performance.rst), [Roadmap](docs/roadmap.rst) |

## Core ML on Apple Silicon

On M-series Macs, install the Core ML extra and use `v3-coreml` or `v3-cls-coreml`:

```bash
uv sync --extra coreml
uv run scoutbot pipeline image.jpg --config v3-coreml
```

The `.mlpackage` bundle is exported on first use and cached at `~/.cache/scoutbot/`.
See [Core ML](docs/coreml.rst) and [Performance](docs/performance.rst) for backend details and benchmark caveats.

## Documentation

Build the Sphinx docs locally:

```bash
uv sync --extra docs
uv run sphinx-build -M html docs docs/build
```

Open `docs/build/html/index.html` in a browser.

## Development

Contributor setup and quality checks:

```bash
uv sync --extra test
uv run pytest -m "not slow"
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
```

See [Development](docs/development.rst) for the full contributor workflow.
