# Bandit Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Bandit to the quality workflow as a source-code security gate, while keeping
`uv audit` as the separate dependency-vulnerability check already in place.

**Architecture:** The repository already has a consolidated `quality.yml` job for Ruff, mypy, and
dependency scanning. Bandit should join that job as a second security layer that scans the source
tree for common Python security mistakes. Keep the scope narrow: this plan should add Bandit, wire
the local command into contributor docs, and update the roadmap so it no longer says Bandit is
simply deferred.

**Tech Stack:** Bandit, uv, GitHub Actions, `pyproject.toml`, Markdown/RST docs.

## Global Constraints

- Use `uv run` for all Python/tool commands
- Keep `uv audit` in the workflow; do not replace it
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Pause after each task for review before moving on

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `pyproject.toml` | Add Bandit as a development/quality dependency and, if needed, a `[tool.bandit]` config block |
| Modify | `.github/workflows/quality.yml` | Run Bandit after Ruff/mypy and after `uv audit` |
| Modify | `docs/development.rst` | Document the local Bandit command and explain how it fits with `uv audit` |
| Modify | `docs/security.rst` | Document the source-code-vs-dependency scanning split and any Bandit exclusions |
| Modify | `CONTRIBUTING.md` | Add Bandit to the contributor quality checklist |
| Modify | `docs/roadmap.rst` | Replace the “Bandit remains deferred” note with the new planned/complete state |
| Modify | `CHANGELOG.rst` | Record the quality-suite security scan addition as a workflow change |

---

## Task 1: Decide Bandit scope and config shape

**Files:**
- Read: `.github/workflows/quality.yml`
- Read: `pyproject.toml`
- Read: `docs/development.rst`
- Read: `docs/roadmap.rst`

**Background:** Bandit is source-code security scanning, not dependency scanning. `uv audit`
already covers the dependency side, so Bandit should focus on the Python package source tree.
Before editing the workflow, decide the exact target and exclusions so CI and docs say the same
thing on day one.

- [ ] Run a quick Bandit help/config check and decide the exact command shape for this repo
- [ ] Record the Bandit target path(s) that should be scanned in CI
- [ ] Decide whether `tests/` should be excluded from the scan
- [ ] Decide whether `benchmarks/` should be excluded from the scan
- [ ] Confirm whether `pyproject.toml` needs a `[tool.bandit]` block for the chosen excludes or
  skips, or whether the workflow can start with a simple CLI invocation
- [ ] Confirm that `uv audit` stays separate and does not need to change for this task

**Exit criteria:**
- We know the exact Bandit invocation to add to CI
- We know whether Bandit config belongs in `pyproject.toml`

---

## Task 2: Add Bandit to the quality workflow

**Files:**
- Modify: `.github/workflows/quality.yml`
- Modify: `pyproject.toml` if Task 1 requires Bandit config or a new dependency declaration

**Background:** This task adds Bandit as a second security check in the existing `quality.yml`
job. The workflow should continue to run Ruff, mypy, and `uv audit`, then run Bandit against the
chosen source path(s).

- [ ] Add a Bandit step to `quality.yml` after the existing quality gates
- [ ] Use the exact Bandit command from Task 1, including any config file if one is needed
- [ ] Add Bandit to the `test` extra in `pyproject.toml` so `uv sync --extra test` makes the
  command available in CI
- [ ] Keep the `uv audit` step intact so dependency scanning remains covered separately

**Verification:**
- Run the Bandit command locally through `uv run` against the chosen source path(s)
- Confirm the workflow fails if Bandit reports a real issue and passes cleanly on the current tree

---

## Task 3: Update contributor docs and roadmap

**Files:**
- Modify: `docs/development.rst`
- Modify: `docs/security.rst`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/roadmap.rst`
- Modify: `CHANGELOG.rst`

**Background:** Contributors should be able to run the same security checks locally that CI runs.
The roadmap should stop treating Bandit as simply deferred once the workflow has it.

- [ ] Add the local Bandit command to the quality-checks section in `docs/development.rst`
- [ ] Explain in `docs/development.rst` that `uv audit` handles dependency vulnerabilities while
  Bandit handles source-code security checks
- [ ] Update `docs/security.rst` so the canonical security page reflects the Bandit/`uv audit`
  split and the chosen exclusion policy
- [ ] Add Bandit to the contributor quality checklist in `CONTRIBUTING.md` so it matches CI
- [ ] Update `docs/roadmap.rst` so the security scanning note reflects the new Bandit state and
  no longer says Bandit is just deferred
- [ ] Add an unreleased changelog note for the Bandit addition as a workflow/security change

**Exit criteria:**
- Docs tell contributors how to run Bandit locally
- The roadmap matches the implemented security-scanning split between `uv audit` and Bandit

---

## Task 4: Final consistency pass

**Files:**
- Re-read: `.github/workflows/quality.yml`, `docs/development.rst`, `docs/roadmap.rst`,
  `CHANGELOG.rst`, and `pyproject.toml` if it changed

**Background:** This pass verifies that the workflow, docs, and roadmap all describe the same
two security layers: `uv audit` for dependencies and Bandit for source code.

- [ ] Re-read the touched files and confirm the Bandit command, dependency scanning, and roadmap
  wording are consistent
- [ ] Confirm the roadmap no longer implies Bandit is deferred if the workflow now runs it
- [ ] Pause for review with a concise summary of the final security-scanning split
