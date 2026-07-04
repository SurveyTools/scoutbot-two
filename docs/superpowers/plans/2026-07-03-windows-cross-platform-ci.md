# Windows Cross-Platform CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight Windows smoke path to CI so the non-Core ML pipeline is verified on
another platform without turning the workflow into a full cross-platform matrix.

**Architecture:** Keep the existing Linux test matrix intact and add one Windows job that runs the
fast/offline suite only. The Windows path should stay focused on the base PyTorch pipeline and
should not pull Core ML or other Apple-only concerns into scope. This plan fixes the Windows scope
up front: one `windows-latest` smoke job on Python 3.13, no full Windows matrix, no Core ML, and
no GPU-specific setup. Documentation should describe the support level plainly: Windows is
covered by a smoke path, not a separate full product matrix.

**Tech Stack:** GitHub Actions, uv, pytest, Sphinx docs.

## Global Constraints

- Use `uv run` for Python/tool commands
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Pause after each task for review before moving on

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `.github/workflows/testing.yml` | Add a Windows smoke job that runs the fast/offline tests |
| Modify | `docs/development.rst` | Document the Windows smoke path and how it fits the CI story |
| Modify | `CONTRIBUTING.md` | Add a short note about Windows bug reports and the limited CI coverage |
| Modify | `docs/roadmap.rst` | Mark Windows / cross-platform CI as active work once the smoke job exists |
| Modify | `CHANGELOG.rst` | Record the new Windows CI coverage as a workflow change |

---

## Task 1: Audit platform assumptions and collect the reusable pieces

**Files:**
- Read: `.github/workflows/testing.yml`
- Read: `docs/development.rst`
- Read: `docs/roadmap.rst`

**Background:** The project already has a stable fast/offline test path, and that is the
appropriate starting point for Windows coverage. The Windows scope is already fixed in the
Architecture section; this task only records the exact Linux steps that will be copied or adapted
and the Windows-specific caveats the docs need to mention.

- [ ] Read the current Linux workflow and confirm which steps should be reused on Windows
- [ ] Identify any platform-sensitive spots that the smoke job is expected to catch
- [ ] Record any known Windows-specific exclusions or caveats so the docs can state them plainly

**Exit criteria:**
- The smoke-job scope is explicit and intentionally narrow
- We know which docs need wording updates after the workflow changes land

---

## Task 2: Add the Windows smoke job to `testing.yml`

**Files:**
- Modify: `.github/workflows/testing.yml`

**Background:** Add a separate Windows job rather than expanding the existing Linux matrix. That
keeps runtime cost down and makes it clear that the goal is platform coverage, not a new supported
runner matrix everywhere. Reuse the same action versions that the current Linux workflow already
uses in `testing.yml` so the Windows job stays aligned with the repo's existing CI style.
Keep the Windows smoke run on the same `--random-order-seed=1` ordering as Linux so failures are
comparable across runners.

- [ ] Add a `windows-smoke` job that uses `runs-on: windows-latest`
- [ ] Reuse the same `uv sync --extra test` setup and `uv run pytest -m "not slow"` command
- [ ] Keep the job limited to the fast/offline suite; do not add Core ML or GPU-specific steps
- [ ] Preserve the existing Linux test matrix unchanged

**Target YAML:**

```yaml
  windows-smoke:
    runs-on: windows-latest
    env:
      PYTHON: "3.13"
      DETECTOR_BATCH_SIZE: 8
    steps:
      - uses: actions/checkout@v7
        with:
          lfs: true
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v8.2.0

      - name: Set up Python 3.13
        uses: actions/setup-python@v6
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: uv sync --extra test

      - name: Run fast/offline tests
        run: uv run pytest -m "not slow" --random-order-seed=1
```

**Verification:**
- Confirm the workflow YAML is syntactically valid and the job shape matches the repo's existing
  `uv`-based CI style
- Confirm the job name and summary make it obvious that this is a smoke path, not a full Windows
  support matrix

---

## Task 3: Update docs and roadmap

**Files:**
- Modify: `docs/development.rst`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/roadmap.rst`
- Modify: `CHANGELOG.rst`

**Background:** Contributors should be able to tell that Windows is now covered by CI, but only by
the fast/offline smoke path. The roadmap should move the item out of “future idea” territory once
the job exists.

- [ ] Add a short CI note to `docs/development.rst` that names the Windows smoke job and its
  narrow scope
- [ ] Add a short contributor note to `CONTRIBUTING.md` that says Windows issues are welcome,
  that the project has Windows smoke coverage but not full parity, and that bug reports should
  include the exact command, Python version, and runner details
- [ ] Update the roadmap's Windows / cross-platform CI item so it reflects the new smoke coverage
- [ ] Add a changelog bullet that records the Windows coverage addition as a workflow change

**Exit criteria:**
- The docs explain the new Windows coverage without overstating support
- The roadmap and changelog match the workflow that was actually added

---

## Task 4: Final consistency pass

**Files:**
- Re-read: `.github/workflows/testing.yml`, `docs/development.rst`, `docs/roadmap.rst`,
  `CHANGELOG.rst`

**Background:** This final pass checks that the workflow, docs, and roadmap all say the same thing
about Windows: the fast suite is covered, and the project is not claiming a broader Windows
guarantee than it has verified.

- [ ] Re-read the touched files and confirm the smoke-job scope is consistent everywhere
- [ ] Confirm the roadmap wording does not imply full Windows support if only smoke coverage was
  added
- [ ] Run `uv run pytest -m "not slow" -q` to confirm the existing fast suite still passes after
  the workflow and doc changes
- [ ] Pause for review with a concise summary of the final platform-support story
