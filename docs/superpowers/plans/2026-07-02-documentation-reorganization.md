# Documentation Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh ScoutBot's public and maintainer documentation so it accurately reflects the v3/Core ML codebase, the completed quality workflow modernization, and the next performance/observability investigation.

**Architecture:** Keep the README as a concise entry point and make Sphinx the durable reference. Split user-facing docs, maintainer docs, performance diagnostics, and planning artifacts into clear sections. Quality-tool documentation should now use the finalized branch state: Python 3.10-3.12, `uv sync --extra test`, `testing.yml`, `quality.yml`, Ruff, scoped mypy, and pytest slow markers.

**Tech Stack:** Markdown for `README.md` and internal plan files, reStructuredText for Sphinx pages and the converted roadmap, Sphinx + sphinx-click for generated CLI docs.

**Quality baseline now in force:** `docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md` has been implemented on branch `quality`. The finalized state is:

- `pyproject.toml` declares `requires-python = ">=3.10,<3.13"`.
- The `test` and `all` extras include `pytest`, `pytest-cov`, `pytest-random-order`, `ruff`, and `mypy`.
- `.github/workflows/testing.yml` runs on PRs and pushes to `main`, uses Python 3.10/3.11/3.12, installs with `uv sync --extra test`, and runs `uv run pytest -m "not slow" --cov=./ --cov-append --random-order-seed=1`.
- `.github/workflows/quality.yml` runs on PRs and pushes to `main`, installs with `uv sync --extra test`, then runs `uv run ruff format --check .`, `uv run ruff check .`, and `uv run mypy scoutbot`.
- `.pre-commit-config.yaml` uses Ruff and `ruff-format`; Brunette, flake8, isort, and pyupgrade hooks are gone.
- `setup.cfg` is still present for package metadata, but it is no longer the tool-config source of truth.

---

## Current Findings

- `README.md` is too broad. It contains installation, Windows CUDA notes, CLI examples, Core ML usage, API shape, model table, environment variables, tests, coverage, docs build, and formatting. This makes it easy for stale development commands to persist.
- `README.md` still shows only the Testing badge. It should add the Quality workflow badge now that `.github/workflows/quality.yml` exists.
- `README.md` still uses `uv pip install -e ".[test]"` and `uv pip install -r requirements.optional.txt` for development/setup commands. Current contributor setup should use `uv sync --extra test`.
- `README.md` still says the project uses Brunette + flake8 and points at `setup.cfg` for style rules. That is stale; the current tools are Ruff and scoped mypy from `pyproject.toml`.
- `README.md` still says `uv run pytest` for tests. Current default local/CI fast path should be `uv run pytest -m "not slow"`; slow/model/inference tests should be documented separately.
- `CLAUDE.md` still names Brunette, flake8, isort, and `setup.cfg`. It should become an agent/maintainer quick reference for the finalized Ruff/mypy/pytest workflow.
- `CLAUDE.md` model config notes still mention only `v3` and `v3-cls`; they should include `v3-coreml` and `v3-cls-coreml`.
- `docs/coreml.rst` is the strongest current page. It includes benchmark caveats, but the performance concern is buried in a backend page rather than promoted into a clear performance/diagnostics follow-up.
- `docs/environment.rst` still says `LOC_CONFIG` must be one of `v3` or `v3-cls`; it omits `v3-coreml` and `v3-cls-coreml`.
- `docs/coreml.rst` contains a bare `pip install -e ".[coreml]"` example. Project guidance prefers `uv`, so the pip command should either be removed or explicitly labeled as an alternative after the uv command.
- `docs/scoutbot.rst` includes `overview.rst` and `environment.rst`; `docs/cli.rst` also includes `environment.rst`. This duplicates content across generated pages and increases drift.
- `docs/index.rst` includes the changelog inline before the table of contents. That pushes navigation below historical release notes and makes the docs homepage less useful.
- `CHANGELOG.rst` is useful history, but it is stale for the Core ML work and quality workflow changes.
- `docs/roadmap.md` now exists. It tracks benchmark hardening, quality workflow v2 scope, and CoreML/performance investigations. The docs reorganization should convert it to `docs/roadmap.rst` for Sphinx, then delete the Markdown source.
- `docs/models.rst` has durable reference content for ops, security/governance, and future model contributors. It should remain as a focused model reference instead of being folded into `configuration.rst`.
- `docs/colors.rst` defines color roles and is excluded from Sphinx. It appears unused by current docs and should be removed if no references exist.
- `docs/superpowers/plans/` contains active implementation history mixed with published docs source. These plans should remain available, but they should be treated as maintainer artifacts and excluded from primary user navigation.
- `.readthedocs.yaml` installs `requirements.txt` and the package with pip. It must be updated to install a dedicated `docs` extra so hosted docs build with the same Sphinx dependencies as local docs.

---

## User Groups and Routing

The documentation should be organized around clear reader intent, not just repository file
structure. These are the primary groups the README and Sphinx landing page must route:

| User group | Description | Primary doc needs | Route to |
|------------|-------------|-------------------|----------|
| Raw ScoutBot users | Run ScoutBot directly for image analysis | install, quick start, CLI examples, batch/resume, output schema, troubleshooting | `installation.rst`, `usage.rst`, `configuration.rst` |
| Pipeline ScoutBot users | Use ScoutBot as part of a larger workflow, possibly indirectly; `../pipeline-design` is an example integration | Python API, stable CLI contracts, JSON/JSONL schema, environment/config controls, exit/failure behavior | `usage.rst`, `configuration.rst`, `api.rst`, `cli.rst` |
| Survey/data managers | Own getting survey imagery through analysis and into usable deliverables | input expectations, output interpretation, resumable runs, runtime/storage expectations, downstream handoff | `usage.rst`, `configuration.rst`, `performance.rst` |
| Deployment/ops users | Run ScoutBot on workstations, shared machines, cloud runners, or scheduled jobs | dependency constraints, device selection, cache paths, model prefetching, logging, offline/restricted-network operation | `installation.rst`, `configuration.rst`, `development.rst` as needed |
| Research/scientific consumers | Need to judge whether ScoutBot outputs are suitable for analysis or publication | model provenance, supported classes, confidence meaning, limitations, reproducibility, versioning | `overview.rst`, `models.rst`, `configuration.rst`, `roadmap.rst`, `changelog.rst` |
| Review/QA annotators | Inspect or validate ScoutBot outputs without necessarily running the tool | output schema, confidence/threshold interpretation, common mismatch patterns, mapping results back to images | `usage.rst`, `configuration.rst` |
| Code contributors | Open source contributors changing code | local setup, tests, quality checks, architecture, CI, contribution workflow | `development.rst`, `api.rst`, `roadmap.rst` |
| Release maintainers | Maintainers preparing public releases | release checklist, changelog policy, versioning, CI expectations, public repo hygiene | `development.rst`, `changelog.rst`, `roadmap.rst` |
| Model contributors/integrators | Future contributors adding model updates or new model integrations; connects to `2026-05-09-scoutbot-registry.md` | model registry design, artifact publishing, config schema, validation, compatibility guarantees | `configuration.rst`, future `models.rst` or `model-registry.rst`, registry plan |
| Benchmark/performance investigators | Diagnose Core ML and pipeline performance | reproducible benchmarks, benchmark limitations, phase timings, hardware comparison guidance, observability needs | `performance.rst`, `coreml.rst` |
| Security/governance reviewers | Review dependency, model download, and publication posture | model download sources, MD5 verification, dependency/update policy, CodeQL replacement rationale, data/privacy assumptions | `security.rst`, `models.rst`, `development.rst`, `roadmap.rst` |

README routing should group these into five top-level paths:

1. **Run ScoutBot directly** — raw users, survey/data managers, review/QA readers
2. **Integrate ScoutBot** — pipeline users, deployment/ops users
3. **Understand results** — research consumers, survey managers, QA readers
4. **Improve ScoutBot** — code contributors, release maintainers, security/governance reviewers
5. **Extend models and performance** — model contributors, benchmark/performance investigators

---

## Target Information Architecture

Primary Sphinx navigation should be:

1. `overview.rst` — what ScoutBot does, pipeline shape, supported backends
2. `installation.rst` — install paths for Linux/macOS, Windows CUDA, Apple Silicon Core ML
3. `usage.rst` — CLI and Python API examples, output schema, resumable batch
4. `configuration.rst` — model configs, environment variables, thresholds, devices
5. `models.rst` — model artifact URLs, cache behavior, MD5 verification, provenance notes
6. `coreml.rst` — Core ML backend details, export behavior, parity caveats
7. `performance.rst` — benchmarks, why M1/M4 results are close, planned diagnostics and observability
8. `api.rst` — autodoc API reference
9. `cli.rst` — sphinx-click generated CLI reference
10. `development.rst` — setup, tests, quality workflow, style, type checking
11. `security.rst` — security/governance posture for downloads, dependencies, CI scanning, and data handling
12. `roadmap.rst` — current workstreams and deferred v2 scope
13. `changelog.rst` — release history

Root `README.md` should be:

- one-paragraph project summary
- Testing and Quality status badges
- 5-minute quick start
- minimal CLI example for raw users
- explicit "Where to go next" routing table for the five top-level paths above
- current development commands from the finalized quality baseline

---

## File Map

| Action | Path | What changes |
|--------|------|-------------|
| Modify | `README.md` | Slim down to quick start and links; add Quality badge; remove duplicated reference content |
| Modify | `CLAUDE.md` | Refresh agent commands and architecture notes for finalized quality workflow |
| Modify | `docs/index.rst` | Replace inline changelog-first homepage with navigable landing page |
| Modify | `docs/conf.py` | Keep Sphinx extension config aligned with generated CLI docs; no MyST needed if roadmap is converted to RST |
| Create | `docs/installation.rst` | Install, extras, Windows CUDA, Apple Silicon Core ML setup |
| Create | `docs/usage.rst` | CLI/Python examples, output schema, resumable batch behavior |
| Create | `docs/configuration.rst` | Model configs, env vars, devices, thresholds, cache paths |
| Modify | `docs/models.rst` | Keep as focused model artifact/provenance/integrity reference; move contributor extension notes out |
| Delete | `docs/environment.rst` | Content is migrated to `configuration.rst`; do not leave as an orphan page |
| Modify | `docs/coreml.rst` | Keep backend details; link to installation/performance pages |
| Create | `docs/performance.rst` | Benchmark interpretation and observability investigation plan |
| Rename | `docs/scoutbot.rst` -> `docs/api.rst` | API reference without duplicated overview/environment includes |
| Modify | `docs/cli.rst` | CLI reference only; remove environment include |
| Create | `docs/development.rst` | uv setup, test markers, Ruff, mypy, CI workflows, docs build |
| Create | `docs/security.rst` | Security/governance posture for model downloads, dependency scanning, CodeQL removal, and local data handling |
| Rename | `docs/roadmap.md` -> `docs/roadmap.rst` | Convert existing roadmap to Sphinx-native RST; refine current workstreams and v2 quality/doc/perf scope |
| Optional create | `docs/model-registry.rst` | Future model contributor/integrator route if `models.rst` is not enough |
| Modify | `CHANGELOG.rst` | Add an Unreleased section for Core ML, quality workflow, and documentation reorganization |
| Delete | `docs/colors.rst` | Remove if `rg ":[a-z]+:" docs README.md` confirms no role usage |
| Optional | `docs/superpowers/plans/README.md` | Explain that plan files are internal maintainer artifacts |
| Modify | `pyproject.toml` | Add a `docs` extra for Sphinx, sphinx-rtd-theme, and sphinx-click |
| Modify | `.readthedocs.yaml` | Install the project with the `docs` extra so hosted docs build consistently |

---

## Task 0: Confirm the finalized quality command surface

**Files:**
- Read: `docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md`
- Read: `pyproject.toml`, `.github/workflows/testing.yml`, `.github/workflows/quality.yml`, `.pre-commit-config.yaml`, `setup.cfg`, `CLAUDE.md`
- Read: `docs/conf.py`, `.readthedocs.yaml`

**Background:** The quality implementation is complete. Before editing docs, confirm the final branch state so documentation reflects the actual files, not earlier plan drafts.

- [x] **Step 1: Re-read the quality files**

```bash
git status --short --branch
sed -n '1,220p' docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md
sed -n '1,180p' pyproject.toml
sed -n '1,180p' .github/workflows/testing.yml
sed -n '1,180p' .github/workflows/quality.yml
sed -n '1,180p' .pre-commit-config.yaml
sed -n '1,140p' setup.cfg
sed -n '1,140p' CLAUDE.md
sed -n '1,180p' docs/conf.py
sed -n '1,120p' .readthedocs.yaml
```

- [x] **Step 2: Record the final command surface for docs**

Use these finalized commands unless the files have changed again:

```bash
uv sync --extra test
uv run pytest -m "not slow"
uv run pytest -m "slow"
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv run pre-commit run --all-files
```

- [x] **Step 3: Confirm Sphinx extension and dependency baseline**

`docs/conf.py` must keep `sphinx_click` enabled because `docs/cli.rst` uses the
`.. click::` directive. Do not add `myst-parser`; this plan converts the existing
`docs/roadmap.md` to `docs/roadmap.rst` instead.

---

## Task 1: Build a precise staleness matrix

**Files:**
- Read all docs: `README.md`, `CLAUDE.md`, `CHANGELOG.rst`, `docs/*.rst`, `docs/superpowers/plans/*.md`

- [x] **Step 1: Inventory every public command and config value**

Run stale-tool searches separately from valid current-tool searches to avoid false positives.

Stale terms:

```bash
rg -n "Brunette|flake8|isort|CodeQL|pip install -e|uv pip install -r requirements.optional|Python ≥ 3.8|>=3.8|v3 or v3-cls" README.md CLAUDE.md CHANGELOG.rst docs
```

Current-tool/config verification:

```bash
rg -n "uv sync --extra test|uv run pytest -m \"not slow\"|uv run ruff|uv run mypy scoutbot|quality.yml|testing.yml|v3-coreml|v3-cls-coreml|LOC_CONFIG|SCOUTBOT_DEVICE|DETECTOR_BATCH_SIZE" README.md CLAUDE.md CHANGELOG.rst docs
```

- [x] **Step 2: Create a short audit note inside the PR description or working notes**

Record each stale item as:

```text
path:line — current text — replacement/source of truth
```

Do not edit plan files solely to modernize historical instructions unless they are active plans. Historical plans can remain as records.

---

## Task 2: Restructure Sphinx navigation

**Files:**
- Modify: `docs/index.rst`, `docs/conf.py`
- Create: `docs/installation.rst`, `docs/usage.rst`, `docs/configuration.rst`, `docs/performance.rst`, `docs/development.rst`
- Create: `docs/security.rst`, `docs/changelog.rst`
- Rename/modify: `docs/scoutbot.rst` -> `docs/api.rst`, `docs/roadmap.md` -> `docs/roadmap.rst`

- [x] **Step 1: Replace `docs/index.rst` with a navigation-first homepage**

Use this structure:

```rst
================
Wild Me ScoutBot
================

ScoutBot is a YOLOv8-based pipeline for detecting and localising wildlife in
high-resolution aerial survey imagery.

Start with :doc:`installation` and :doc:`usage`. See :doc:`coreml` and
:doc:`performance` for Apple Silicon acceleration and benchmark interpretation.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   overview
   installation
   usage
   configuration
   models
   coreml
   performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   cli

.. toctree::
   :maxdepth: 1
   :caption: Project

   development
   security
   roadmap
   changelog
```

- [x] **Step 2: Stop including changelog inline**

Create `docs/changelog.rst` as a thin include:

```rst
Changelog
=========

.. include:: ../CHANGELOG.rst
```

- [x] **Step 3: Remove duplicate includes**

`docs/api.rst` should contain API reference only. Do not include `overview.rst` or `environment.rst`.

`docs/cli.rst` should contain sphinx-click output only. Do not include `environment.rst`.

- [x] **Step 4: Rename replacement pages and delete orphaned pages**

Use git-aware moves/deletes:

```bash
git mv docs/scoutbot.rst docs/api.rst
git mv docs/roadmap.md docs/roadmap.rst
git rm docs/environment.rst
```

Do not delete `docs/models.rst`; it remains a first-class model reference page.

- [x] **Step 5: Keep Sphinx config RST-only and verify sphinx-click**

Confirm `docs/conf.py` includes `sphinx_click` in `extensions`. Do not add MyST unless the
roadmap remains Markdown; the chosen path is to convert the roadmap to RST.

- [x] **Step 6: Add docs dependencies and RTD alignment**

Add a `docs` optional dependency group to `pyproject.toml`:

```toml
docs = [
    "Sphinx>=5,<6",
    "sphinx-rtd-theme",
    "sphinx-click",
]
```

Update `.readthedocs.yaml` so hosted docs install the project with the docs extra rather than
only `requirements.txt`. Keep this explicit enough that RTD can build `docs/cli.rst` with
`sphinx_click`.

---

## Task 3: Rewrite the user-facing docs

**Files:**
- Modify: `README.md`
- Modify: `docs/overview.rst`, `docs/models.rst`
- Create: `docs/installation.rst`
- Create: `docs/usage.rst`
- Create/modify: `docs/configuration.rst`
- Delete: `docs/environment.rst`

- [x] **Step 1: Slim `README.md`**

Use sections:

```md
# Wild Me ScoutBot

[Testing badge] [Quality badge]

YOLOv8-based machine learning pipeline for detecting and localising wildlife in aerial survey imagery.

## Quick Start

## Where to Go Next

## Core ML on Apple Silicon

## Documentation

## Development
```

Keep only the shortest working examples. Move detailed Windows CUDA, environment variables, model table, API schema, and docs-build instructions into Sphinx.

The badge block should include:

```md
[![Tests](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/testing.yml)
[![Quality](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/SurveyTools/scoutbot-2026/actions/workflows/quality.yml)
```

- [x] **Step 1a: Add README routing table**

Add a compact routing table under `Where to Go Next`:

```md
| I want to... | Start here |
|---|---|
| Run ScoutBot directly on images | Installation, Usage, Configuration |
| Integrate ScoutBot into a larger pipeline | Usage, API reference, CLI reference |
| Understand outputs and analysis limits | Usage, Configuration, Performance |
| Contribute code or prepare releases | Development, Roadmap, Changelog |
| Add models or investigate performance | Configuration, Core ML, Performance, Roadmap |
```

Use relative links to the final Sphinx source files or published docs URL, depending on the
project's preferred README convention.

- [x] **Step 2: Update install docs**

`docs/installation.rst` must cover:

- contributor setup with `uv sync --extra test`
- package install for direct users with `uv pip install -e .`
- Apple Silicon Core ML with `uv sync --extra coreml`
- Windows CUDA replacement of CPU PyTorch with CUDA wheel
- model cache path and first-run download behavior
- Core ML setup currently duplicated in `docs/coreml.rst`; move install/setup commands here and leave `docs/coreml.rst` with a link back to installation

- [x] **Step 3: Update usage docs**

`docs/usage.rst` must include:

- single-image CLI
- batch CLI
- resumable batch CLI
- `fetch`, `get_classes`, `example`, `export-coreml`
- Python API examples for `pipeline`, `batch`, `batch_resumable`, `fetch`, `get_classes`
- detection output schema `{l, c, x, y, w, h}`
- `wic` definition
- QA/review notes:
  - bounding boxes use source-image pixel coordinates with `x`, `y`, `w`, `h`
  - `x`, `y` are the top-left corner; `w`, `h` are width and height in pixels
  - confidence `c` and `--loc-thresh` affect which detections are returned
  - reviewers should map detections back to original images by filename/order in JSON or JSONL output
  - common review caveats are threshold-borderline detections, duplicate/merged boxes near tile boundaries, and class-label uncertainty for multi-class configs

- [x] **Step 3a: Update overview docs**

`docs/overview.rst` must mention:

- supported configs: `v3`, `v3-cls`, `v3-coreml`, `v3-cls-coreml`
- the common pipeline shape: model fetch/cache, SAHI slicing, backend inference, NMS/merge, output formatting
- Core ML uses the same SAHI/output path but a different backend/export/cache path
- direct CLI users and integration/API users are both supported

- [x] **Step 4: Consolidate model and environment docs**

`docs/configuration.rst` must include:

- model config table with `v3`, `v3-cls`, `v3-coreml`, `v3-cls-coreml`
- environment variables `SCOUTBOT_DEVICE`, `LOC_CONFIG`/`CONFIG`, `DETECTOR_BATCH_SIZE`, `VERBOSE`
- device guidance for `cpu`, `cuda`, `cuda:0`, `mps`, and Core ML
- cache location `~/.cache/scoutbot/`
- threshold and slice-size behavior at a high level

`docs/environment.rst` is deleted after this content is migrated. Do not leave it as a Sphinx orphan.

- [x] **Step 5: Keep `docs/models.rst` as the model artifact reference**

`docs/models.rst` remains in the toctree. Update it to cover:

- model config table for `v3`, `v3-cls`, `v3-coreml`, `v3-cls-coreml`
- source `.pt` model URLs and MD5 sidecar URLs for PyTorch-backed configs
- MD5 verification behavior and cache path
- Core ML bundles are exported locally from source `.pt` models, not hosted
- model provenance notes currently available for KAZA aerial survey data

Move "Adding Future Models" contributor guidance out of `models.rst` and into
`docs/development.rst` or a future `docs/model-registry.rst` section. In `models.rst`, replace it
with a short link to the roadmap/model-registry plan so the page does not promise a mature public
extension API.

---

## Task 4: Promote Core ML performance and diagnostics

**Files:**
- Modify: `docs/coreml.rst`
- Create: `docs/performance.rst`

- [ ] **Step 1: Keep `coreml.rst` backend-focused**

Keep:

- configs
- export behavior
- `.mlpackage` cache behavior
- pipeline integration details
- parity/caveats

Task 3 owns moving Core ML install/setup commands from `coreml.rst` to `installation.rst`.
Task 4 owns benchmark/performance content only: move benchmark interpretation to
`performance.rst`, leaving a short link from `coreml.rst`.

- [ ] **Step 2: Create `performance.rst`**

Include:

- current benchmark table or link to benchmark section
- explicit caveat that M1/16 GB and M4/48 GB numbers are too close to represent raw hardware capacity
- current suspected bottleneck: serial per-tile SAHI/Ultralytics/Core ML dispatch overhead
- current benchmark limitation: full-pipeline wall time only
- next diagnostics needed:
  - image discovery and validation time
  - image decode/load time
  - slice generation time
  - per-tile model invocation count
  - per-tile model wall time
  - post-processing/NMS time
  - JSON/result serialization time
  - Core ML compute-unit selection
  - direct `coremltools` prediction timing
  - concurrent/batched tile inference experiment

- [ ] **Step 3: Link future observability spec**

Reserve a section:

```rst
Observability Follow-up
~~~~~~~~~~~~~~~~~~~~~~~

The next investigation should add structured benchmark phase timings before
claiming hardware-specific Core ML throughput. See the forthcoming observability
spec for implementation details.
```

Replace this with an actual link once the observability spec exists.

- [ ] **Step 4: Keep benchmark code changes out of this documentation plan**

Do not modify `benchmarks/benchmark_coreml.py` in this documentation reorganization. Benchmark
output cleanup is tracked in `docs/roadmap.rst` under "Benchmark hardening" and should be handled
as a separate implementation task.

---

## Task 5: Rewrite maintainer/development docs for the finalized quality workflow

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/development.rst`
- Create: `docs/security.rst`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Replace stale style section with the final quality workflow. Expected content:

````md
## Code Quality

Formatting and linting: Ruff.
Type checking: scoped mypy on `scoutbot/`.
Tests: pytest with `slow` marker for network/model/inference tests.

Commands:

```bash
uv run pytest -m "not slow"
uv run pytest -m "slow"
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv run pre-commit run --all-files
```
````

- [ ] **Step 2: Create `docs/development.rst`**

Include:

- `uv` rule
- supported Python versions
- local setup
- fast/offline tests vs slow tests
- coverage command
- Ruff format/lint commands
- scoped mypy command
- pre-commit command
- CI workflow names and purpose: `testing.yml` and `quality.yml`
- docs build command
- model contributor/integrator note:
  - current public extension API is not mature
  - current model configs live in `scoutbot/loc/__init__.py`
  - future registry work is tracked in `docs/superpowers/plans/2026-05-09-scoutbot-registry.md`
  - new model integration expectations should include artifact provenance, integrity verification, config schema tests, CLI choice tests, and output compatibility checks

- [ ] **Step 3: Create `docs/security.rst`**

Own the security/governance reviewer content here. Include:

- model download source policy: hosted model artifacts are fetched from the configured Azure Blob Storage URLs in `scoutbot.loc.CONFIGS`
- integrity policy: `.pt` models use MD5 sidecar verification before cache reuse/download acceptance
- Core ML artifact policy: `.mlpackage` bundles are exported locally from verified source `.pt` models and are not hosted as separate remote artifacts
- dependency/update posture: dependencies are managed through `pyproject.toml`; quality v2 roadmap tracks `pip-audit`/Bandit-style scanning
- CodeQL removal rationale: CodeQL upload required GitHub Advanced Security/code scanning features unavailable for this repository; project-owned quality checks now run in `quality.yml`
- data/privacy assumption: ScoutBot inference runs locally; image data is not uploaded by ScoutBot itself, though users are responsible for their surrounding pipeline/storage environment
- CI posture: `testing.yml` runs fast/offline tests, `quality.yml` runs Ruff and scoped mypy

- [ ] **Step 4: Update README development section**

Keep it short and link to `docs/development.rst`.

---

## Task 6: Integrate roadmap, changelog, and planning docs

**Files:**
- Rename/modify: `docs/roadmap.md` -> `docs/roadmap.rst`
- Modify: `CHANGELOG.rst`
- Optional create: `docs/superpowers/plans/README.md`

- [ ] **Step 1: Convert and refine existing roadmap**

Convert `docs/roadmap.md` to `docs/roadmap.rst` with equivalent content. Keep its lightweight
tracking format and ensure it covers:

- benchmark hardening for `benchmarks/benchmark_coreml.py`, including missing `coremltools` handling and third-party warning/log noise
- quality workflow modernization complete; quality v2 scope remains deferred
- documentation reorganization planned or in progress
- future community contribution docs should include the same documentation-pass expectation for external contributors, likely in a future `CONTRIBUTING.md` or equivalent contributor guide
- CoreML/performance investigations, including phase timing, Core ML dispatch overhead, tile batching/parallelism, ANE confirmation, and M4 follow-up benchmark
- performance observability spec as the next investigation if it is tracked separately
- model registry/model integration future work, linking to `docs/superpowers/plans/2026-05-09-scoutbot-registry.md`
- quality v2 deferred scope:
  - expand mypy to tests/benchmarks
  - coverage enforcement
  - security/dependency scanning
  - macOS Core ML CI
  - Ruff rule expansion

- [ ] **Step 2: Update changelog**

Add an `Unreleased` section to `CHANGELOG.rst` that covers:

- Core ML backend/config documentation and benchmark caveat additions
- CodeQL workflow removal and replacement with `testing.yml`/`quality.yml` quality gates
- Python support raised to `>=3.10,<3.13`
- documentation reorganization if this plan is implemented in the same change set

- [ ] **Step 3: Add a plans README**

If keeping plan files under `docs/superpowers/plans/`, add:

```md
# Implementation Plans

These files are maintainer/agent implementation plans, not user documentation.
They are retained for project history and handoff context.
```

Do not add this directory to the main Sphinx user-guide navigation.

---

## Task 7: Validate docs

**Files:**
- Read/build docs output
- Read: `.readthedocs.yaml`, `pyproject.toml`, `docs/conf.py`

- [ ] **Step 1: Build Sphinx docs**

Use the docs extra:

```bash
uv sync --extra docs
uv run sphinx-build -M html docs docs/build
```

- [ ] **Step 2: Run quality checks**

Use the final quality commands:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv run pytest -m "not slow" -v
```

- [ ] **Step 3: Search for stale terms**

```bash
rg -n "Brunette|flake8|isort|CodeQL|pip install -e|v3 or v3-cls|Python ≥ 3.8|>=3.8" README.md CLAUDE.md CHANGELOG.rst docs
```

Expected:

- no current docs claim Brunette/flake8/isort are active tools
- CodeQL only appears in historical plan context or a short note explaining it was removed
- current setup commands use `uv`
- config lists include Core ML variants where appropriate
- Python support is documented as `>=3.10,<3.13`

- [ ] **Step 4: Verify no orphaned replacement docs remain**

```bash
test ! -e docs/environment.rst
test ! -e docs/scoutbot.rst
test ! -e docs/roadmap.md
test -e docs/api.rst
test -e docs/roadmap.rst
test -e docs/models.rst
test -e docs/security.rst
```

- [ ] **Step 5: Verify hosted-docs configuration**

Confirm `.readthedocs.yaml` installs the project with the `docs` extra and that `pyproject.toml`
defines that extra with `Sphinx`, `sphinx-rtd-theme`, and `sphinx-click`.

---

## Task 8: Add the ongoing documentation-pass rule for agents

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/development.rst`
- Modify: `docs/roadmap.rst`

**Background:** The reorganization reduces current drift, but future work can reintroduce it.
Every future implementation plan should include an explicit documentation pass before completion.

- [ ] **Step 1: Add a Documentation Pass rule to `CLAUDE.md`**

Add a rule for all agents:

```md
## Documentation Pass Rule

Every implementation plan must include a final documentation pass before completion. Review
whether the work changes user behavior, commands, configuration, models, outputs, dependencies,
quality checks, security posture, performance characteristics, or roadmap status. Update the
affected docs in the same change set, or explicitly record why no doc update is needed.

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
```

- [ ] **Step 2: Add the same rule to `docs/development.rst`**

Include it in the contributor workflow section so external contributors see the requirement.

- [ ] **Step 3: Update future-plan expectations**

Any future plan under `docs/superpowers/plans/` should include a final checklist item:

```md
- [ ] Documentation pass completed: reviewed README/Sphinx docs/changelog/roadmap/security docs
      relevant to this change, updated them or recorded why no doc update was needed.
```

This plan should not retroactively edit historical completed plans; the rule applies to new plans
after the documentation reorganization lands.

- [ ] **Step 4: Track public contributor equivalent in the roadmap**

Add a roadmap item for the future public contribution surface:

```rst
Community contribution docs
---------------------------

Status: planned

When the project opens for community contributions, add the same documentation-pass requirement
to the public contributor guide, likely `CONTRIBUTING.md` or an equivalent contributor page.
Contributors should be expected to review and update affected docs, changelog, and roadmap entries
as part of any code, model, CLI, configuration, workflow, security, or performance change.
```

Do not create `CONTRIBUTING.md` in this reorganization unless the maintainers decide to expand
the scope. `docs/development.rst` remains the current contributor-facing location.

---

## Acceptance Criteria

- [ ] README is a short entry point, not the full reference manual.
- [ ] README routes the five top-level reader paths to the right docs.
- [ ] Sphinx docs explicitly cover raw users, pipeline users, survey/data managers, deployment/ops users, research consumers, review/QA annotators, code contributors, release maintainers, model contributors/integrators, benchmark investigators, and security/governance reviewers.
- [ ] Sphinx docs have clear user guide, reference, project, and development sections.
- [ ] `docs/models.rst` remains a focused model artifact/provenance/integrity reference.
- [ ] `docs/environment.rst`, `docs/scoutbot.rst`, and `docs/roadmap.md` do not remain as orphaned replacement files.
- [ ] `docs/roadmap.rst` is Sphinx-native RST and appears in the Project toctree.
- [ ] `docs/security.rst` owns security/governance content: model download policy, MD5 verification, dependency/security scanning posture, CodeQL removal rationale, and local data/privacy assumptions.
- [ ] `CHANGELOG.rst` has an `Unreleased` section covering Core ML, quality workflow modernization, and documentation reorganization.
- [ ] `.readthedocs.yaml` and the `docs` extra in `pyproject.toml` are aligned so hosted docs can build.
- [ ] No current user-facing docs describe Brunette, flake8, or isort as active tooling.
- [ ] Environment/config docs include `v3-coreml` and `v3-cls-coreml`.
- [ ] Core ML docs keep backend details while performance interpretation is easy to find.
- [ ] Performance docs clearly state that current M1/M4 benchmark numbers are bottlenecked by the current pipeline and should not be read as raw hardware capacity.
- [ ] Development docs reflect `testing.yml`, `quality.yml`, Ruff, scoped mypy, pytest markers, and `uv`.
- [ ] `CLAUDE.md` and `docs/development.rst` require a final documentation pass in future implementation work and define which doc roles map to which kinds of changes.
- [ ] `docs/roadmap.rst` tracks adding the same documentation-pass expectation to a future public `CONTRIBUTING.md` or equivalent contributor guide.
- [ ] Historical plan files remain available without being treated as primary user docs.
- [ ] Sphinx docs build cleanly.
- [ ] Stale-term search is clean except for intentional historical references.
