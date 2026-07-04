# Python Interpreter Readiness — 3.13 Support, 3.14 Awareness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise ScoutBot's declared support to Python 3.13 and document a clear readiness
checklist for Python 3.14 without claiming support for 3.14 yet.

**Architecture:** This work has two layers:
1. make Python 3.13 an explicit supported target in packaging, CI, and user-facing docs
2. record Python 3.14 as an awareness-only readiness item so the next interpreter bump is
   already scoped before it becomes urgent

The plan should stay tight: only change files that communicate or enforce interpreter support.
Do not widen the scope into unrelated dependency churn unless a Python-version blocker forces it.

**Tech Stack:** Python, uv, pytest, GitHub Actions, Sphinx docs.

## Global Constraints

- Use `uv run` for all Python/tool commands; do not invoke `pytest`, `ruff`, `mypy`, or
  `sphinx-build` directly
- Preserve Ruff conventions already in the repo: single quotes and 90-character line length
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Pause after each task for review before moving on
- Treat Python 3.14 as awareness-only in this plan; do not claim active support for 3.14

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `pyproject.toml` | Raise `requires-python` to keep 3.13 supported while excluding 3.14 |
| Modify | `.github/workflows/testing.yml` | Add Python 3.13 to the test matrix |
| Modify | `docs/development.rst` | Update the supported-version text and local setup guidance |
| Modify | `docs/installation.rst` | Update the user-facing supported-version text |
| Modify | `CHANGELOG.rst` | Record the supported-version bump in the unreleased notes |
| Modify | `docs/roadmap.rst` | Refine the interpreter-readiness section if the current wording is stale or incomplete |

---

## Task 1: Audit current Python-version gates and compatibility surface

**Files:**
- Read: `pyproject.toml`
- Read: `.github/workflows/testing.yml`
- Read: `.github/workflows/quality.yml`
- Read: `docs/development.rst`
- Read: `docs/roadmap.rst`

**Background:** The repo still declares `>=3.10,<3.13` and CI only exercises Python 3.10,
3.11, and 3.12 in the test matrix. Before changing the gates, confirm where the support text
is repeated and whether any workflow or dependency is already Python-3.13-sensitive.

- [ ] Read the files above and note every Python-version declaration, CI matrix, and doc
  sentence that still says `<3.13` or omits 3.13 entirely
- [ ] Check whether `quality.yml` can remain on Python 3.12 or should also move to 3.13;
  only change it if the audit shows a real compatibility reason
- [ ] Check whether `pyproject.toml`'s `python_version` for mypy should move from `3.12` to
  `3.13` alongside the packaging support bump
- [ ] Record any dependency or toolchain blockers that would prevent 3.13 support from being
  advertised yet

**Exit criteria:**
- We know exactly which files need to change for the 3.13 support bump
- We have a short list of any blockers that need follow-up

---

## Task 2: Promote Python 3.13 to a supported target

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/testing.yml`
- Modify: `docs/development.rst`
- Modify: `docs/installation.rst`
- Modify: `CHANGELOG.rst`

**Background:** This task makes the active support story match reality: package metadata,
CI, and contributor-facing docs should all agree that Python 3.13 is supported, while Python
3.14 remains excluded for now.

- [ ] Raise `requires-python` in `pyproject.toml` from `>=3.10,<3.13` to `>=3.10,<3.14`
- [ ] Bump `tool.mypy.python_version` in `pyproject.toml` from `3.12` to `3.13` so local
  type checking follows the supported interpreter baseline
- [ ] Add Python 3.13 to the `testing.yml` matrix while keeping the existing supported
  versions in place
- [ ] Update `docs/development.rst` so it names Python 3.13 as supported and no longer presents
  `<3.13` as the declared upper bound
- [ ] Update `docs/installation.rst` so the user-facing support statement matches the new
  packaging and CI policy
- [ ] Add an unreleased changelog note for the Python 3.13 support bump
- [ ] Keep the rest of the release/quality guidance unchanged unless the audit from Task 1
  found a concrete need to update it

**Verification:**
- Confirm the docs and metadata now agree on 3.13 support and 3.14 exclusion
- If Python 3.13 is available locally, run the fast CI-equivalent test path against it; if it
  is not available, stop at metadata consistency and note CI as the real gating check

---

## Task 3: Document Python 3.14 readiness without claiming support

**Files:**
- Modify: `docs/roadmap.rst`
- Modify: `docs/development.rst` if the local contributor guidance needs a short readiness note

**Background:** 3.14 should be treated as a planning item, not a supported runtime. The docs
should say what would have to change before 3.14 can be promoted: test matrix, packaging
metadata, and any dependency compatibility work discovered during the audit.

- [ ] Add or refine the roadmap entry so Python 3.14 is listed as readiness-only, not active
  support
- [ ] Spell out the concrete readiness checklist: CI matrix, package metadata, and the
  specific dependencies most likely to lag 3.14 support (`torch`, `ultralytics`, `sahi`, and
  `coremltools`)
- [ ] Add a short note that 3.14 readiness also needs a dependency-stub/type-check pass once
  the matrix and package metadata are ready
- [ ] Make sure the wording distinguishes "supported now" from "needed later" so readers do not
  mistake awareness for commitment

**Exit criteria:**
- The roadmap clearly tracks 3.14 as a future readiness item
- No doc still implies that 3.14 is supported today

---

## Task 4: Final consistency pass

**Files:**
- Re-read: `pyproject.toml`, `.github/workflows/testing.yml`, `docs/development.rst`,
  `docs/roadmap.rst`

**Background:** This last pass is a documentation and config sanity check. It should catch any
leftover `<3.13` references, any accidental 3.14 support claims, and any mismatch between the
packaging metadata and the docs.

- [ ] Re-read the touched files for stale version text and mismatched support statements
- [ ] Confirm the final state still matches the intended policy: 3.13 supported, 3.14 awareness
  only
- [ ] Pause for review with a concise summary of what changed and any follow-up risk that still
  remains
