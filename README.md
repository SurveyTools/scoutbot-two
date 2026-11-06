# Scoutbot

[![Tests](https://github.com/SurveyTools/scoutbot-two/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-two/actions/workflows/testing.yml)
[![Quality](https://github.com/SurveyTools/scoutbot-two/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-two/actions/workflows/quality.yml)

YOLOv8-based machine learning pipeline for detecting and localising wildlife in aerial survey imagery.

## Acknowledgement

Scoutbot-two is a continuation of the work started in the original
[WildMeOrg/scoutbot](https://github.com/WildMeOrg/scoutbot) repository. Scoutbot was originally
developed for the [Scout](https://github.com/WildMeOrg/scout) project by the Wild Me team of Conservation X Labs with funding from the Paul G. Allen Family Foundation.

Starting with 0.5.0, the [`scoutbot` package on PyPI](https://pypi.org/project/scoutbot/) is
published from scoutbot-two by the Overhead Survey Workflow (OSW) team, with Wild Me's support.
The `0.1.x` releases on PyPI are the legacy Wild Me line and are no longer maintained.

## Quick Start

```bash
pip install scoutbot        # or: uv pip install scoutbot

scoutbot fetch --config v3
scoutbot pipeline /path/to/image.jpg --config v3 --device mps
```

From source:

```bash
git clone https://github.com/SurveyTools/scoutbot-two
cd scoutbot-two
uv pip install -e .
uv run scoutbot fetch --config v3
```

Scoutbot downloads models on first use and caches them in `~/.osw/models/`.

## Where to Go Next

| I want to... | Start here |
|---|---|
| Run Scoutbot directly on images | [Scoutbot docs](https://surveytools.github.io/scoutbot-two/), [Installation](https://surveytools.github.io/scoutbot-two/installation.html), [Usage](https://surveytools.github.io/scoutbot-two/usage.html), [Configuration](https://surveytools.github.io/scoutbot-two/configuration.html) |
| Integrate Scoutbot into a larger pipeline | [Usage](https://surveytools.github.io/scoutbot-two/usage.html), [API reference](https://surveytools.github.io/scoutbot-two/api.html), [CLI reference](https://surveytools.github.io/scoutbot-two/cli.html) |
| Understand outputs and analysis limits | [Usage](https://surveytools.github.io/scoutbot-two/usage.html), [Configuration](https://surveytools.github.io/scoutbot-two/configuration.html), [Performance](https://surveytools.github.io/scoutbot-two/performance.html) |
| Contribute code or prepare releases | [Development](https://surveytools.github.io/scoutbot-two/development.html), [Roadmap](https://surveytools.github.io/scoutbot-two/roadmap.html), [Changelog](https://surveytools.github.io/scoutbot-two/changelog.html) |
| Add models or investigate performance | [Models](https://surveytools.github.io/scoutbot-two/models.html), [Core ML](https://surveytools.github.io/scoutbot-two/coreml.html), [Performance](https://surveytools.github.io/scoutbot-two/performance.html), [Roadmap](https://surveytools.github.io/scoutbot-two/roadmap.html) |

## Core ML on Apple Silicon

On M-series Macs, install the Core ML extra and use `v3-coreml` or `v3-cls-coreml`:

```bash
pip install "scoutbot[coreml]"   # from PyPI
uv sync --extra coreml           # or, from a source checkout
uv run scoutbot pipeline image.jpg --config v3-coreml
```

The `.mlpackage` bundle is exported on first use and cached at `~/.osw/models/`.
See [Core ML](https://surveytools.github.io/scoutbot-two/coreml.html) and [Performance](https://surveytools.github.io/scoutbot-two/performance.html) for backend details and benchmark caveats.

## Documentation

Read the rendered Scoutbot docs at https://surveytools.github.io/scoutbot-two/.

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

See [Development](https://surveytools.github.io/scoutbot-two/development.html) for the full contributor workflow.
