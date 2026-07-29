# uv Audit Quality Suite Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the roadmap's dependency-scanning placeholder with `uv audit` as the project's
quality-suite vulnerability check, keeping the quality workflow fast, deterministic, and
lockfile-based.

**Architecture:** The quality workflow already installs the test environment with `uv sync`; this
plan adds a lockfile-aware `uv audit` gate to the same job so dependency scanning runs against the
same resolved graph that CI is already using. If the audit command proves sufficient, the roadmap
should stop naming `pip-audit` and instead track `uv audit` as the project's canonical dependency
scanner. Any durable ignore policy should live in `pyproject.toml` only if the audit step exposes a
real, documented waiver that we intend to keep.

**Tech Stack:** uv, GitHub Actions, `pyproject.toml`, Markdown/RST docs.

## Global Constraints

- Use `uv` commands directly for audit and dependency checks; do not add `pip-audit`
- Preserve the repo's existing `uv`-first workflow style
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Pause after each task for review before moving on

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `.github/workflows/quality.yml` | Add `uv audit` as a required quality gate |
| Modify | `docs/development.rst` | Document how contributors run the new audit check locally |
| Modify | `docs/roadmap.rst` | Replace the `pip-audit` roadmap note with `uv audit` if the integration covers the same policy |
| Modify | `CHANGELOG.rst` | Record the quality-suite scanner swap as a workflow change |
| Modify | `pyproject.toml` | Only if a durable `[tool.uv.audit]` ignore policy is required |

---

## Task 1: Validate `uv audit` against the current quality policy

**Files:**
- Read: `.github/workflows/quality.yml`
- Read: `docs/roadmap.rst`
- Read: `docs/development.rst`
- Read: `pyproject.toml`

**Background:** `uv audit` is available in the installed `uv` binary and supports `locked`,
`frozen`, `json`, `sarif`, `ignore`, and `ignore-until-fixed`. Before wiring it into CI, verify
which flags match the repo's current policy: audit the same locked project graph that `quality.yml`
installs, and decide whether any optional dependency groups should be excluded explicitly or left
in scope.

- [ ] Run `uv audit --help` and confirm the command is available without a preview-only gate in
  the local `uv` binary
- [ ] Record the local `uv` version and the fact that `uv audit` is available without a preview
  flag so Task 2 can describe the CI floor explicitly
- [ ] Run `uv sync --extra test` followed by `uv audit --locked` from the repo root and confirm
  the command inspects the project graph cleanly
- [ ] If the audit output shows that the installed CI environment should exclude a specific group
  or extra, record the exact `--no-group`, `--only-group`, or `--no-extra` flag needed
- [ ] Decide whether any vulnerability IDs need a durable exception. If yes, capture the exact
  ID and whether it belongs in `ignore` or `ignore-until-fixed`

**Exit criteria:**
- We know the exact `uv audit` invocation that should run in CI
- We know whether `pyproject.toml` needs an audit-configuration stanza or can stay untouched

---

## Task 2: Add `uv audit` to the quality workflow

**Files:**
- Modify: `.github/workflows/quality.yml`
- Modify: `pyproject.toml` only if Task 1 established a durable audit-ignore policy

**Background:** This task makes dependency scanning part of the main quality gate. The workflow
should fail on any vulnerability finding and should use the same lockfile-backed dependency graph
that the repository already relies on for CI reproducibility.

- [ ] Add a `uv audit` step to `quality.yml` after dependencies are installed and before the job
  completes
- [ ] Use `uv audit --locked` rather than `--frozen`; `--locked` should fail if the lockfile is
  stale, which is the correct behavior for a quality gate
- [ ] Use the exact locked audit invocation established in Task 1 so the job checks the project
  graph consistently
- [ ] If Task 1 identified a permanent or until-fixed exception, add the smallest possible
  `[tool.uv.audit]` configuration in `pyproject.toml` and keep the exception list explicit
- [ ] Keep the rest of the workflow unchanged unless `uv audit` requires a small ordering tweak
  to run after `uv sync`

**Verification:**
- Run the quality-job command sequence locally, ending with `uv audit` against the locked graph
- Confirm the workflow fails cleanly if `uv audit` is forced to report a vulnerability

---

## Task 3: Update docs and roadmap to match the new scanner

**Files:**
- Modify: `docs/development.rst`
- Modify: `docs/roadmap.rst`
- Modify: `CHANGELOG.rst`

**Background:** Contributors need a single, current command for dependency scanning, and the roadmap
should stop planning for a `pip-audit` implementation if `uv audit` now covers the same policy
surface.

- [ ] Add `uv audit` to the quality-checks section in `docs/development.rst`, including the exact
  local command contributors should run
- [ ] Update the CI workflow description in `docs/development.rst` so the quality job mentions
  Ruff, mypy, and `uv audit`
- [ ] Replace the roadmap's `pip-audit` wording with `uv audit` if Task 2 fully covers the
  dependency-scanning need; otherwise leave a clear note that the roadmap item remains open
- [ ] Mention `--output-format sarif` as a follow-on option in the roadmap note so a later GitHub
  Code Scanning upload path is easy to add if the team wants machine-readable results
- [ ] Add an unreleased changelog bullet for the workflow change so the audit tool swap is
  recorded alongside the other quality-workflow updates

**Exit criteria:**
- Docs tell contributors to use `uv audit`, not `pip-audit`
- The roadmap no longer presents `pip-audit` as the planned implementation if the new workflow
  already satisfies the dependency-scanning requirement

---

## Task 4: Final consistency pass

**Files:**
- Re-read: `.github/workflows/quality.yml`, `docs/development.rst`, `docs/roadmap.rst`,
  `CHANGELOG.rst`, and `pyproject.toml` if it changed

**Background:** This pass verifies that the workflow, docs, and roadmap all describe the same
scanner and the same policy boundaries.

- [ ] Re-read the touched files and confirm they all name the same dependency-scanning command
- [ ] Confirm the roadmap does not still imply that `pip-audit` is the next required step if the
  workflow has already been switched
- [ ] Pause for review with a concise summary of the final scanner choice and any residual risk
