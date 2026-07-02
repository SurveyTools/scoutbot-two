# Quality Workflow Modernization — v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the removed CodeQL workflow with project-owned quality checks; modernize `testing.yml` to use `uv`; replace flake8 + isort + Brunette with Ruff in one clean cutover; add scoped mypy on the library code.

**Architecture:** Four independent tasks that together produce two CI workflows (`testing.yml` updated, `quality.yml` new) and a clean tool consolidation. Config lives entirely in `pyproject.toml` after the migration — `setup.cfg` tool sections are removed. Tasks should be committed in order (Task 2 reformats files that Task 3 may annotate), but each produces a working, reviewable state on its own.

**Tech Stack:** Python 3.10–3.12, uv, Ruff, mypy, pytest, GitHub Actions.

**Python version policy:** This PR also raises the declared package support from `>=3.8,<3.13` to `>=3.10,<3.13`, aligning `pyproject.toml` with the CI matrix. Python 3.8 and 3.9 are end-of-life and no longer tested.

**Rationale:** CodeQL cannot upload results without GitHub Advanced Security enabled, so it was dropped. This project needs stronger Python quality gates more than a half-enabled code scanning workflow.

## Global Constraints

- Python ≥ 3.10 in CI (matrix: 3.10, 3.11, 3.12); `pyproject.toml` `requires-python` raised to `>=3.10,<3.13` in this PR
- All tool invocations use `uv run` prefix — never bare `pytest`, `ruff`, `mypy`
- `pyproject.toml` is the single source of truth for tool config; `setup.cfg` tool sections (`[flake8]`, `[isort]`, `[brunette]`) will be removed — the file itself is NOT deleted (it still contains package metadata, entry points, and extras)
- Single-quote style must be preserved across the Ruff migration (`quote-style = "single"`)
- Do NOT run `git add` or `git commit` — user handles all commits
- Line length: 90 (matches current Brunette config)

---

## File Map

| Action | Path | What changes |
|--------|------|-------------|
| Modify | `tests/test_pipeline.py` | Task 1 — add `@pytest.mark.slow` to all tests |
| Modify | `tests/test_batch.py` | Task 1 — add `@pytest.mark.slow` to TestBatch, TestBatchResumable |
| Modify | `tests/test_loc.py` | Task 1 — add `@pytest.mark.slow` to TestLocFetch |
| Modify | `pyproject.toml` | Task 1 — add `[tool.pytest.ini_options] markers`; Tasks 2+3 — add Ruff + mypy deps and config |
| Modify | `.github/workflows/testing.yml` | Task 1 — uv, PR trigger, drop inline flake8, add `-m "not slow"` |
| Modify | `.pre-commit-config.yaml` | Task 2 — replace pyupgrade+isort+brunette+flake8 with ruff hooks |
| Modify | `setup.cfg` | Task 2 — remove `[flake8]`, `[isort]`, `[brunette]` sections (and `[tool:pytest]` if already owned by pyproject.toml); do NOT delete the file |
| Create | `.github/workflows/quality.yml` | Task 4 — ruff format check, ruff lint, mypy |

---

## Task 1: Pytest markers + modernize `testing.yml`

**Files:**
- Modify: `tests/test_pipeline.py`, `tests/test_batch.py`, `tests/test_loc.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/testing.yml`

**Background:** The current `testing.yml` runs `pip` (not `uv`), triggers on every push to every branch, and has flake8 inline. Tests that download models or run real inference run in CI, making the suite slow and fragile on the default runner. Adding `@pytest.mark.slow` to those tests lets CI run `pytest -m "not slow"` by default — fast, offline, deterministic.

**Slow tests to mark:**
- `tests/test_pipeline.py` — all tests (use `sample_image` fixture which downloads model)
- `tests/test_batch.py` — `TestBatch` class (4 tests) and `TestBatchResumable` class (4 tests)
- `tests/test_loc.py` — `TestLocFetch` class (2 tests)

- [ ] **Step 1: Raise `requires-python` in `pyproject.toml`**

Find the `requires-python` line in `[project]` and change it from `>=3.8,<3.13` to `>=3.10,<3.13`:

```toml
[project]
requires-python = ">=3.10,<3.13"
```

- [ ] **Step 2: Read the three test files**

Read `tests/test_pipeline.py`, `tests/test_batch.py`, and `tests/test_loc.py` in full to confirm which test classes/methods need the marker and where imports are.

- [ ] **Step 3: Add `@pytest.mark.slow` to slow test classes**

In `tests/test_pipeline.py`, add `@pytest.mark.slow` above each test class (or above the class definition if all tests in the class are slow). Pattern:


```python
@pytest.mark.slow
class TestPipeline:
    ...
```

Add `import pytest` at the top of the file if not already present.

In `tests/test_batch.py`, add `@pytest.mark.slow` to `TestBatch` and `TestBatchResumable` only — leave `TestBatchResumableProgressBar`, `TestBatchResumableErrorResume`, and `TestApiEnvVarConfig` unmarked (they are offline/fast).

In `tests/test_loc.py`, add `@pytest.mark.slow` to `TestLocFetch` only — leave `TestLocConfig` and `TestExportCoreml` unmarked.

- [ ] **Step 4: Register the marker in `pyproject.toml`**

In the `[tool.pytest.ini_options]` section, add a `markers` entry:

```toml
[tool.pytest.ini_options]
minversion = "5.4"
addopts = "-v --durations-min=1.0 --color=yes -ra"
testpaths = ["tests"]
filterwarnings = ["default"]
markers = [
    "slow: marks tests that download models or run real inference (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 5: Verify marker coverage**

Check that slow-marked tests are complete:

```bash
uv run pytest -m "slow" --collect-only -q
```

Expected output includes tests from `test_pipeline.py`, `TestBatch`, `TestBatchResumable`, `TestLocFetch`.

Then run the negative check — confirm no inference or model-download tests leaked into the fast suite:

```bash
uv run pytest -m "not slow" --collect-only -q
```

Scan the output and confirm none of `TestPipeline`, `TestBatch`, `TestBatchResumable`, or `TestLocFetch` appear. If any do, add the missing `@pytest.mark.slow` and re-run both checks.

- [ ] **Step 6: Verify fast suite runs offline**

```bash
uv run pytest -m "not slow" -v
```

Expected: all non-slow tests pass without network access, completing in under 30 seconds.

- [ ] **Step 7: Update `.github/workflows/testing.yml`**

Replace the full file content with:

```yaml
name: Testing

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-22.04
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    env:
      OS: ubuntu-22.04
      PYTHON: ${{ matrix.python-version }}
      DETECTOR_BATCH_SIZE: 8

    steps:
      - uses: actions/checkout@v5
        with:
          lfs: true
          fetch-depth: 0

      - name: Free Disk Space
        run: |
          sudo rm -rf /usr/share/dotnet /opt/ghc "/usr/local/share/boost" "$AGENT_TOOLSDIRECTORY"
          sudo apt-get clean

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --extra test

      - name: Run fast/offline tests
        run: uv run pytest -m "not slow" --cov=./ --cov-append --random-order-seed=1

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: false
```

- [ ] **Step 8: Run flake8 on changed files to confirm no regressions before the Ruff migration**

```bash
uv run flake8 tests/test_pipeline.py tests/test_batch.py tests/test_loc.py
```

Expected: no output (clean). This is the last flake8 check before Task 2 replaces it.

---

## Task 2: Replace flake8 + isort + Brunette with Ruff

**Files:**
- Modify: `pyproject.toml`
- Modify: `.pre-commit-config.yaml`
- Modify: `setup.cfg`
- All Python source files (reformatted by `ruff format`)

**Background:** Ruff replaces three tools (flake8, isort, Brunette/Black) with one. The critical config detail is `quote-style = "single"` — without it, Ruff's formatter will switch the codebase to double quotes. The ignore list is translated from flake8 codes to Ruff codes; `W503` (line break before binary operator) has no Ruff equivalent and is simply dropped. `pyupgrade` is intentionally not replaced in v1 — Ruff's `UP` rule set covers the same modernizations but will be enabled as a separate expansion step in v2 once the baseline is stable.

- [ ] **Step 1: Add Ruff to `pyproject.toml` dependencies**

In the `[project.optional-dependencies]` section, update the `test` extra. **Note: this intentionally changes the existing `pytest >= 6.2.2` pin to an unpinned `pytest` (current compat is fine; the floor was conservative) and adds `pytest-random-order` (already used in CI via `--random-order-seed` but not declared). Call these out in the commit message.**

```toml
[project.optional-dependencies]
test = [
    "pytest",
    "pytest-cov",
    "pytest-random-order",
    "ruff",
]
```

- [ ] **Step 2: Add Ruff config to `pyproject.toml`**

Append to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 90
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = [
    "E127",  # continuation line over-indented
    "E201",  # whitespace after '('
    "E202",  # whitespace before ')'
    "E203",  # whitespace before ':'
    "E221",  # multiple spaces before operator
    "E222",  # multiple spaces after operator
    "E241",  # multiple spaces after ','
    "E265",  # block comment should start with '# '
    "E271",  # whitespace around keyword
    "E272",  # whitespace around keyword
    "E301",  # expected 1 blank line
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "single"
```

- [ ] **Step 3: Install updated dependencies**

```bash
uv sync --extra test
```

- [ ] **Step 4: Run Ruff lint with auto-fix**

```bash
uv run ruff check . --fix
```

Review the output. If any fixes are made, inspect them with `git diff`. Mechanical import-order and unused-import fixes are expected and fine. Anything touching logic should be reviewed carefully before proceeding.

- [ ] **Step 5: Run Ruff formatter**

```bash
uv run ruff format .
```

This replaces Brunette. With `quote-style = "single"` the diff should be minimal (line-length and spacing adjustments only, no quote flips).

- [ ] **Step 6: Verify Ruff checks pass cleanly**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: both return exit code 0.

- [ ] **Step 7: Update `.pre-commit-config.yaml`**

Replace the full file with:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-ast
      - id: check-yaml
      - id: check-toml
      - id: check-executables-have-shebangs
      - id: check-docstring-first
      - id: trailing-whitespace
      - id: mixed-line-ending
      - id: end-of-file-fixer
```

Notes on removals:
- `pyupgrade` — **intentionally removed without replacement in v1.** Ruff's `UP` rules are the equivalent; they will be enabled in v2 as a deliberate expansion step, not automatically.
- `isort` — removed; Ruff's `I` rules handle import sorting
- `brunette` — removed; replaced by `ruff-format`
- `flake8` — removed; replaced by `ruff check`
- `double-quote-string-fixer` — removed; `ruff-format` with `quote-style = "single"` handles this
- `fix-encoding-pragma` — removed; Python 3 does not require `# -*- coding: utf-8 -*-` headers
- `requirements-txt-fixer` — removed; project uses `pyproject.toml` / `uv.lock`
- Hook versions bumped to current stable

- [ ] **Step 8: Remove obsolete tool sections from `setup.cfg`**

Read `setup.cfg` in full first — it contains more than just old tool config. Remove only the `[flake8]`, `[isort]`, and `[brunette]` sections. Do NOT delete the file; it still holds package metadata, entry points, and extras that are not duplicated in `pyproject.toml`.

Also check `[tool:pytest]` (the setuptools-style pytest section). If all its settings are already captured in `pyproject.toml`'s `[tool.pytest.ini_options]`, remove it. If any settings are unique to `[tool:pytest]`, migrate them to `pyproject.toml` first, then remove the section.

- [ ] **Step 9: Run the fast test suite to confirm no regressions**

```bash
uv run pytest -m "not slow" -v
```

Expected: all pass.

---

## Task 3: Add scoped mypy on `scoutbot/`

**Files:**
- Modify: `pyproject.toml`

**Background:** Starting with `ignore_missing_imports = true` avoids noise from third-party libraries (sahi, ultralytics, coremltools) that lack complete stubs. Tests and benchmarks are excluded from v1 scope — they use MagicMock patterns and dynamic imports that generate noise before the config is tuned. The goal is a clean baseline, not strict typing.

- [ ] **Step 1: Add mypy to `pyproject.toml` dependencies**

Add `"mypy"` to the `test` extra:

```toml
[project.optional-dependencies]
test = [
    "pytest",
    "pytest-cov",
    "pytest-random-order",
    "ruff",
    "mypy",
]
```

- [ ] **Step 2: Add mypy config to `pyproject.toml`**

```toml
[tool.mypy]
python_version = "3.12"
warn_unused_configs = true
ignore_missing_imports = true
```

- [ ] **Step 3: Install updated dependencies**

```bash
uv sync --extra test
```

- [ ] **Step 4: Run mypy and capture baseline output**

```bash
uv run mypy scoutbot 2>&1 | tee /tmp/mypy-baseline.txt
```

Review the output. For each error:
- **Type annotation issues** on owned code: fix them inline (add return type, fix wrong type).
- **Third-party type errors** that `ignore_missing_imports` didn't suppress: add a targeted `# type: ignore[import]` comment on that line.
- **Legitimate bugs** surfaced by mypy: fix the bug.

Do not add broad `# type: ignore` without a specific error code. Do not add `disallow_untyped_defs = true` until annotations exist.

- [ ] **Step 5: Run mypy until clean**

```bash
uv run mypy scoutbot
```

Expected: `Success: no issues found in N source files`.

- [ ] **Step 6: Run full fast suite to confirm no regressions**

```bash
uv run pytest -m "not slow" -v
```

---

## Task 4: Add `quality.yml` CI workflow

**Files:**
- Create: `.github/workflows/quality.yml`

**Background:** Quality checks (formatting, linting, type checking) run once per PR and on pushes to main. They are separate from `testing.yml` so a formatting failure doesn't block the test result and vice versa.

- [ ] **Step 1: Create `.github/workflows/quality.yml`**

```yaml
name: Quality

on:
  push:
    branches: [main]
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-22.04

    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --extra test

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: Ruff lint
        run: uv run ruff check .

      - name: Mypy
        run: uv run mypy scoutbot
```

- [ ] **Step 2: Verify all three quality checks pass locally**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
```

Expected: all exit 0.

- [ ] **Step 3: Run fast test suite one final time**

```bash
uv run pytest -m "not slow" -v
```

Expected: all pass. This is the final local verification before the PR.

---

## Acceptance Criteria

- [ ] CodeQL workflow remains deleted.
- [ ] `testing.yml` uses `uv`, triggers on PR + push-to-main only, runs `pytest -m "not slow"`.
- [ ] `quality.yml` runs Ruff format check, Ruff lint, and `mypy scoutbot` — all passing.
- [ ] Pre-commit config uses Ruff hooks only (no flake8, isort, Brunette, pyupgrade).
- [ ] `setup.cfg` tool sections removed; `pyproject.toml` is single source of truth for tool config.
- [ ] `@pytest.mark.slow` marker applied to all model-downloading tests.
- [ ] Fast suite (`-m "not slow"`) completes offline in under 30 seconds.
- [ ] README and CLAUDE.md command examples stay consistent with `uv run`.

---

---

# V2 Scope (deferred — do not implement in this PR)

These items are captured here to avoid losing them. Create the roadmap document (V2a below) first so every subsequent item has a home.

## V2a: Roadmap document

Start `docs/roadmap.md` to consolidate deferred work before it gets lost across plan files:
- This V2 list
- The Group 4 performance/cleanup items from the CoreML PR (double model load, single-image benchmark nan, `batch_resumable` s/img display)
- Future CoreML investigations (parallel tile inference, M4 parallelisation benchmark)
- Any other planned work surfaced during PR reviews

A lightweight format — dated sections, one-liner per item, status tags (`planned` / `in-progress` / `done`) — works better than GitHub issues for items that don't yet have a clear owner or timeline.

## V2b: Coverage enforcement

Add a minimum coverage threshold so regressions are caught:

```toml
[tool.pytest.ini_options]
addopts = "... --cov-fail-under=70"
```

Also consider `coverage.py` branch coverage (`--cov-branch`) and an HTML report artifact uploaded to CI. Current gap: `scoutbot/example()`, the `VERBOSE` logging path, and the Windows PowerShell path are untested.

## V2c: Expand mypy to tests and benchmarks

Once the `scoutbot/` baseline is clean, extend coverage:

```bash
uv run mypy scoutbot tests benchmarks
```

Likely requires:
- Adding `pytest-stub` or `types-*` packages for pytest fixtures
- Annotating `conftest.py` fixtures
- Suppressing MagicMock-related noise with targeted ignores
- Deciding on `disallow_untyped_defs` per-module via `[[tool.mypy.overrides]]`

Note: `ignore_missing_imports = true` is a pragmatic v1 baseline. V2c is the right time to replace it with targeted `[[tool.mypy.overrides]]` per-library if the project leans on types more heavily.

## V2d: Ruff rule expansion

Once the codebase is Ruff-clean on the conservative v1 rule set, consider enabling:

- `UP` (pyupgrade) — modernize Python syntax to 3.10+ idioms; this restores the behavior that was intentionally deferred when `pyupgrade` was removed in v1
- `B` (flake8-bugbear) — opinionated but low-noise bug patterns
- `C4` (flake8-comprehensions) — prefer list/dict/set comprehensions
- `SIM` (flake8-simplify) — simplifiable if/else patterns

Expand one rule group per PR so regressions are easy to bisect.

## V2e: Security / dependency scanning

- `pip-audit` — scans `uv.lock` for known CVEs. Fast, low-noise, fits in `quality.yml`.
- `bandit` — Python AST security linter. Useful but needs a tuned config to avoid false positives on the benchmark script's subprocess patterns.

Add to `quality.yml` as separate steps once the Ruff + mypy baseline is stable.

## V2f: macOS CI for CoreML path

The CoreML backend (`v3-coreml`, `v3-cls-coreml`) is macOS-only and untested in CI. A separate workflow on `macos-latest` could run:

```bash
uv sync --extra coreml
uv run pytest -m "not slow" -v   # offline CoreML config/schema tests
uv run scoutbot export-coreml --config v3-coreml  # smoke test export
```

Cost: GitHub-hosted macOS runners are ~10× more expensive than ubuntu. Consider running only on pushes to main (not every PR) or using a self-hosted M-series runner. Address after V2e — this is the most expensive item and benefits least from CI automation given deterministic hardware requirements.
