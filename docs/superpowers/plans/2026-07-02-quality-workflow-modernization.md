# Quality Workflow Modernization Plan

**Goal:** Replace the removed CodeQL workflow with simpler, project-owned quality checks that run in CI and can be mirrored locally.

**Rationale:** CodeQL cannot upload results unless GitHub Advanced Security/code scanning is enabled for the repository. Running it with upload disabled hides the main value while keeping a fragile CI moving part. This project needs stronger Python quality gates more than a half-enabled code scanning workflow.

## Proposed CI Shape

### `testing.yml`

Keep this focused on runtime behavior:

- Run on pull requests and pushes to `main`.
- Use `uv` for dependency setup.
- Test supported Python versions.
- Run the fast/offline test suite by default.
- Keep network/model-download tests out of default CI unless explicitly marked and gated.

Suggested checks:

```bash
uv run pytest
```

If full-suite runtime remains too slow because existing tests exercise real model inference, split tests into:

- default fast/offline tests
- opt-in integration/performance tests

## New `quality.yml`

Run deterministic static checks once per PR:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tests benchmarks
```

Start with scoped mypy on owned code. If third-party library typing or old tests are noisy, use a narrow config and expand intentionally rather than disabling broad error classes.

## Tooling Changes

### Ruff

Add `ruff` as a dev/test dependency and configure it in `pyproject.toml`.

Initial scope:

- formatter replacement for Brunette where practical
- import sorting
- pyflakes/pycodestyle-equivalent linting
- selected bugbear-style rules if low-noise

Migration path:

1. Add Ruff with a conservative rule set matching current style.
2. Run `uv run ruff check .` and fix only mechanical violations.
3. Add `uv run ruff format --check .` to CI.
4. Later decide whether to remove Brunette/isort/flake8 from pre-commit.

### Mypy

Add `mypy` as a dev/test dependency.

Initial target:

```bash
uv run mypy scoutbot
```

Then expand to:

```bash
uv run mypy scoutbot tests benchmarks
```

Likely starter config:

```toml
[tool.mypy]
python_version = "3.12"
warn_unused_configs = true
ignore_missing_imports = true
```

Avoid claiming strict typing until errors are resolved without broad ignores.

## Pre-commit

Update `.pre-commit-config.yaml` after Ruff is configured:

- `ruff check --fix`
- `ruff format`
- existing YAML/trailing whitespace/end-of-file hooks
- mypy only if local runtime is acceptable

## Deferred Optional Checks

Consider later, not in the first modernization PR:

- `pip-audit` for dependency vulnerabilities
- `bandit` for Python security linting
- separate benchmark/observability workflow for Core ML performance

## Acceptance Criteria

- CodeQL workflow remains deleted.
- `testing.yml` uses `uv` and has a clear fast/offline default.
- New or updated quality workflow runs Ruff format, Ruff lint, and scoped mypy.
- Pre-commit matches the CI quality checks where practical.
- README/CLAUDE command examples stay consistent with `uv`.
