# Contributing to Scoutbot

Thank you for your interest in contributing. This guide covers everything you need to get a
development environment running, follow the project's coding and testing standards, and submit
changes.

---

## Ways to contribute

- Report a bug by opening a GitHub issue
- Propose a feature or improvement in a GitHub issue before implementing it
- Fix a bug or implement an approved feature via a pull request
- Improve documentation or tests

## Before you start

- Check the open issues and pull requests to avoid duplicating work in progress.
- For non-trivial changes, open an issue first to discuss the approach before writing code.
- By contributing you agree that your changes will be licensed under the project's Apache-2.0 license.
- These standards implement the family-wide engineering practices maintained in the **`pipeline-design`** repo at `docs/engineering-practices.md` (the shared rulebook); this guide adds only Scoutbot specifics. Referenced by repo name rather than linked, since a sibling checkout layout is not assumed.

---

## Development setup

Install [uv](https://docs.astral.sh/uv/) if you do not already have it, then:

```bash
git clone https://github.com/SurveyTools/scoutbot-2026
cd scoutbot-2026
uv sync --extra test
```

This installs the project in editable mode with pytest, pytest-cov, pytest-random-order, Ruff,
mypy, and Bandit. For a full maintainer environment that also includes the Sphinx doc toolchain
and Core ML support (macOS only), use `--extra all` instead.

**Runtime rule:** always prefix Python tool invocations with `uv run`. Never call `python`,
`pytest`, `ruff`, `mypy`, or similar tools directly — they may resolve to a different environment.

```bash
uv run pytest -m "not slow"   # fast suite
uv run ruff check .
uv run mypy scoutbot
```

---

## Testing

The test suite is split by the `slow` marker:

```bash
# Fast / offline — the default local and CI path
uv run pytest -m "not slow"

# Slow — downloads models or runs real inference; skipped in CI
uv run pytest -m "slow"

# Single test or pattern
uv run pytest tests/test_loc.py::TestLocConfig::test_configs_exist -v
uv run pytest -k "test_pipeline" -v

# With coverage
uv run pytest -m "not slow" --cov=scoutbot --cov-report=html
```

### Test design

New tests must cover meaningful code branches — exception handlers, guard clauses, and
behavioural forks — not just add lines to a coverage metric.

Standard patterns used throughout:

- `unittest.mock.patch` for network and model dependencies
- `tmp_path` for filesystem fixtures
- `CliRunner` for CLI commands
- `@pytest.mark.slow` only when a test actually hits the network or loads a real model file

Branches that exist solely to satisfy a metric (e.g. `if __name__ == '__main__':`) are
excluded from coverage expectations.

---

## Code style

Formatting and linting are handled by [Ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff format --check .   # check formatting
uv run ruff format .            # apply formatting
uv run ruff check .             # lint
```

Key style settings (from `pyproject.toml`):

- Line length: 90 characters
- Quote style: single quotes
- Import ordering: Ruff `I` rules

**Comments:** write no comments by default. Add one only when the *why* is non-obvious — a
hidden constraint, a subtle invariant, or a workaround for a specific bug. Do not explain what
the code does; well-named identifiers already do that.

---

## Type checking

Scoped mypy runs against the `scoutbot/` package and tools:

```bash
uv run mypy scoutbot tools/visualise.py
```

---

## Dependency security

Audit the locked dependency graph before submitting a change that touches `pyproject.toml` or
`uv.lock`:

```bash
uv audit --locked --preview-features audit-command
```

This runs automatically in CI. Any new vulnerability finding blocks the quality gate. If an
advisory has no available fix, document the waiver in `[tool.uv.audit]` in `pyproject.toml`
with a comment explaining why and when to revisit.

---

## Documentation

Every change that affects user behaviour, commands, configuration, models, outputs,
dependencies, quality checks, security posture, performance characteristics, or roadmap status
must include a documentation update in the same pull request.

Use these pages as your guide for which file owns which topic:

| Topic | File |
|---|---|
| Install commands, extras, platform setup | `docs/installation.rst` |
| CLI / API examples, output schema | `docs/usage.rst` |
| Env vars, config names, runtime controls | `docs/configuration.rst` |
| Model artifacts, URLs, MD5 hashes | `docs/models.rst` |
| Core ML backend, export, Apple caveats | `docs/coreml.rst` |
| Benchmark interpretation, performance | `docs/performance.rst` |
| Contributor setup, tests, CI workflows | `docs/development.rst` |
| Security / dependency scanning posture | `docs/security.rst` |
| Planned and deferred work | `docs/roadmap.rst` |
| User-visible and workflow changes | `CHANGELOG.rst` |

Build and preview Sphinx docs locally:

```bash
uv sync --extra docs
uv run sphinx-build -M html docs docs/build
open docs/build/html/index.html
```

[GitHub Pages](https://surveytools.github.io/scoutbot-2026/) is the canonical hosted docs
site. Local builds are a maintainer preview path.

---

## Quality checklist

Run these before opening a pull request:

```bash
uv run pytest -m "not slow"
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv audit --locked --preview-features audit-command
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
```

All six must pass. CI enforces the same checks.

---

## Submitting a pull request

1. Fork the repository and create a branch from `main`.
2. Write tests for any new behaviour before writing the implementation.
3. Run the full quality checklist above and ensure everything passes.
4. Include any required documentation updates in the same PR.
5. Write a clear PR description: what changed, why, and how to test it.
6. Open the pull request against `main`. A maintainer will review it.

Branch naming convention: `<type>/<short-description>` — for example,
`feat/batch-progress-bar` or `fix/coreml-input-size`.

---

## Reporting bugs

Open a GitHub issue and include:

- Scoutbot version (`uv run scoutbot --version`)
- Python version and platform
- Minimal reproduction steps
- Actual vs expected behaviour
- Any relevant log output

**Windows and cross-platform issues:** The project has a Windows smoke CI job that runs the
fast/offline test suite on Python 3.13, but this does not constitute full platform parity. Windows
issues are welcome — please include the exact command you ran, your Python version, and details
about your operating system and runner (WSL, native Windows, etc.) to help isolate platform-specific
problems.

**GPU and CUDA issues:** GPU support is best-effort for NVIDIA systems on Windows and is not
CI-tested. Before reporting GPU issues, consult the verification steps in [installation docs
(Windows with NVIDIA GPU)](https://surveytools.github.io/scoutbot-2026/installation.html#windows-with-nvidia-gpu)
and the contributor guidance in [development docs (GPU and CUDA)](https://surveytools.github.io/scoutbot-2026/development.html#gpu-and-cuda).
If you have confirmed the GPU is correctly detected and the issue persists, please include your
PyTorch and CUDA versions, GPU model, driver version, and the error output.

---

## Proposing features

Open a GitHub issue describing the feature, the problem it solves, and any constraints or
tradeoffs you are aware of. Discuss the approach before starting an implementation — this
avoids building something that conflicts with the roadmap or that requires significant rework.

---

## Code of conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of
conduct. Be respectful, constructive, and focused on the work. Maintainers reserve the right to
remove contributions or participants that violate these standards.

---

## License

Scoutbot is released under the [Apache-2.0 License](LICENSE). By submitting a pull request you
confirm that you have the right to license your contribution under the same terms.
