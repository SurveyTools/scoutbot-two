# Documentation Reorganization Task 1 Audit

Working note for Task 0 and Task 1 of `2026-07-02-documentation-reorganization.md`.

## Task 0 Confirmation

Task 0 files were re-read on branch `documentation-reorg`.

Confirmed quality command surface:

```bash
uv sync --extra test
uv run pytest -m "not slow"
uv run pytest -m "slow"
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv run pre-commit run --all-files
```

Confirmed repo state:

- `pyproject.toml` declares `requires-python = ">=3.10,<3.13"`.
- `pyproject.toml` `test` and `all` extras include `pytest`, `pytest-cov`, `pytest-random-order`, `ruff`, and `mypy`.
- `.github/workflows/testing.yml` uses Python 3.10/3.11/3.12, installs with `uv sync --extra test`, and runs `uv run pytest -m "not slow" --cov=./ --cov-append --random-order-seed=1`.
- `.github/workflows/quality.yml` uses Python 3.12, installs with `uv sync --extra test`, and runs Ruff format check, Ruff lint, and `uv run mypy scoutbot`.
- `.pre-commit-config.yaml` uses Ruff hooks and no longer uses Brunette, flake8, isort, or pyupgrade.
- `docs/conf.py` already includes `sphinx_click`, and no MyST parser is configured. This matches the plan's decision to convert `docs/roadmap.md` to `docs/roadmap.rst`.

Confirmed follow-up items for later tasks:

- `.readthedocs.yaml` still installs `requirements.txt` and the project via pip; Task 2/7 should align it with the planned `docs` extra.
- `setup.cfg` still contains legacy package metadata (`python_requires = >=3.7`) and old extras (`pycodestyle`, old pytest floor). This is not fixed by Task 0, but later docs/dependency work should avoid treating `setup.cfg` as the quality-tool source of truth and should decide whether the remaining setup metadata needs cleanup.
- `CLAUDE.md` remains stale and is scheduled for Task 5.

## Scope

Commands run:

```bash
rg -n "Brunette|flake8|isort|CodeQL|pip install -e|uv pip install -r requirements.optional|Python >= 3.8|>=3.8|v3 or v3-cls" README.md CLAUDE.md CHANGELOG.rst docs
rg -n "uv sync --extra test|uv run pytest -m \"not slow\"|uv run ruff|uv run mypy scoutbot|quality.yml|testing.yml|v3-coreml|v3-cls-coreml|LOC_CONFIG|SCOUTBOT_DEVICE|DETECTOR_BATCH_SIZE" README.md CLAUDE.md CHANGELOG.rst docs
```

The stale-term search also matched historical plan files. Those are intentionally retained as historical implementation records and should not be rewritten solely to modernize old instructions.

## Actionable Current-Doc Findings

`README.md:3` — README only shows the Testing badge.

Replacement/source of truth: add the Quality badge for `.github/workflows/quality.yml` next to the Testing badge.

`README.md:14` — `uv pip install -e ".[test]"` appears in the top-level Linux/macOS install path.

Replacement/source of truth: README quick start should be short and route detailed install modes to `docs/installation.rst`. Contributor setup should use `uv sync --extra test`; direct editable package install can be documented in installation as `uv pip install -e .`.

`README.md:26` — Windows setup uses `uv pip install -e .`.

Replacement/source of truth: this is acceptable for direct-user package install, but it should move out of README into `docs/installation.rst` with the rest of Windows CUDA guidance.

`README.md:193` — Tests section uses `uv pip install -e ".[test]"`.

Replacement/source of truth: contributor/test setup should use `uv sync --extra test`.

`README.md:194` — Tests section uses `uv run pytest`.

Replacement/source of truth: default fast/offline test command should be `uv run pytest -m "not slow"`. Slow/model/inference tests should be documented separately as `uv run pytest -m "slow"`.

`README.md:200` — Coverage command uses `uv run pytest --cov=scoutbot --cov-report=html`.

Replacement/source of truth: keep coverage guidance in `docs/development.rst`; if retained, make clear whether it runs all tests or should use `-m "not slow"`.

`README.md:207` — Documentation build setup uses `uv pip install -r requirements.optional.txt`.

Replacement/source of truth: docs setup should use the planned docs extra, `uv sync --extra docs`.

`README.md:215` — Formatting setup uses `uv pip install -r requirements.optional.txt`.

Replacement/source of truth: development setup should use `uv sync --extra test`; formatting/linting commands are Ruff-based.

`README.md:220` — README says the codebase uses Brunette with flake8 and points at `setup.cfg`.

Replacement/source of truth: current tools are Ruff format/lint and scoped mypy configured in `pyproject.toml`, with pre-commit using Ruff hooks.

`CLAUDE.md:26` — command uses `uv run mypy scoutbot/`.

Replacement/source of truth: finalized quality workflow uses `uv run mypy scoutbot`.

`CLAUDE.md:34` — install command uses `uv pip install -e ".[test]"`.

Replacement/source of truth: contributor setup should use `uv sync --extra test`.

`CLAUDE.md:37` — run-all-tests command uses `uv run pytest`.

Replacement/source of truth: default fast/offline command is `uv run pytest -m "not slow"`; slow tests are run with `uv run pytest -m "slow"`.

`CLAUDE.md:68` and `CLAUDE.md:75` — model config notes do not list `v3-coreml` or `v3-cls-coreml`.

Replacement/source of truth: update model config guidance to include `v3`, `v3-cls`, `v3-coreml`, and `v3-cls-coreml`.

`CLAUDE.md:80` — code style section says Brunette, flake8, isort, and `setup.cfg`.

Replacement/source of truth: replace with Ruff format/lint, scoped mypy, pytest markers, `quality.yml`, and `pyproject.toml` as tool config source.

`docs/coreml.rst:11` — Core ML setup shows `pip install -e ".[coreml]"` before `uv sync --extra coreml`.

Replacement/source of truth: installation docs should prefer `uv sync --extra coreml`; `coreml.rst` should link to `docs/installation.rst` for setup.

`docs/environment.rst:10` — `LOC_CONFIG` says configs must be only `v3` or `v3-cls`.

Replacement/source of truth: migrate environment content into `docs/configuration.rst`, include Core ML configs, then delete `docs/environment.rst`.

`CHANGELOG.rst` — no current `Unreleased` entry for Core ML docs, quality workflow modernization, Python support change, or documentation reorganization.

Replacement/source of truth: add an `Unreleased` section during Task 6.

`.readthedocs.yaml` — not matched by the stale search, but current docs plan requires RTD alignment.

Replacement/source of truth: update RTD config to install the project with the planned `docs` extra.

`requirements.optional.txt` — not directly in the rg target set, but README currently relies on it and it still contains obsolete quality tools (`brunette`, `flake8`).

Replacement/source of truth: after adding `pyproject.toml` docs/test extras as the install source of truth, either update or retire `requirements.optional.txt` references so docs do not direct contributors to obsolete tools.

## Intentional Historical Hits

These files contain obsolete commands or pre-quality tool references as historical implementation records. Do not rewrite them solely for staleness cleanup:

- `docs/superpowers/plans/2026-07-01-pr2-group1-fixes.md`
- `docs/superpowers/plans/2026-07-01-pr2-group2-docs-tests.md`
- `docs/superpowers/plans/2026-07-01-pr2-group3-preexisting-bugs.md`
- `docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md`

The active documentation reorganization plan may continue to mention stale terms when describing what must be fixed or when specifying validation searches.

## Current-Tool Verification Notes

Current quality/config references appear mainly in the active documentation plan and should be propagated into user-facing docs during later tasks:

- `uv sync --extra test`
- `uv run pytest -m "not slow"`
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy scoutbot`
- `testing.yml` and `quality.yml`
- `v3-coreml` and `v3-cls-coreml`
- `SCOUTBOT_DEVICE`, `LOC_CONFIG`/`CONFIG`, and `DETECTOR_BATCH_SIZE`
