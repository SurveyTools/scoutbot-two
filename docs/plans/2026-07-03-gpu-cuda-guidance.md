# GPU / CUDA Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ScoutBot's CUDA support story explicit for contributors and users, including what is
supported, what is best-effort, and how to verify that a GPU is actually being used.

**Architecture:** This is primarily a documentation update. The current installation page already
describes Windows NVIDIA GPU setup, but the support level and verification story are still fuzzy.
The plan sets the policy up front: CUDA is documented as a best-effort runtime path for NVIDIA
systems, not a CI guarantee, and the docs should give users one exact verification command so they
can confirm whether `torch.cuda.is_available()` is true on their machine. The plan then tightens
that language across installation, contributor guidance, and the roadmap so the project says one
clear thing about CUDA instead of several implied ones.

**Tech Stack:** Sphinx docs, GitHub-flavored Markdown, uv, pytest.

## Global Constraints

- Use `uv run` for Python/tool commands
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Pause after each task for review before moving on

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `docs/installation.rst` | State the CUDA support level and show the verification command users should run |
| Modify | `docs/development.rst` | Document the maintainer-facing support posture and verification guidance |
| Modify | `docs/roadmap.rst` | Update the GPU / CUDA roadmap note so it matches the documented policy |
| Modify | `CHANGELOG.rst` | Record the docs/support clarification as a release-note-worthy workflow/docs change |

---

## Task 1: Audit the current CUDA wording against the chosen support statement

**Files:**
- Read: `docs/installation.rst`
- Read: `docs/development.rst`
- Read: `docs/roadmap.rst`

**Background:** The repository already explains how to install CUDA wheels on Windows, but it does
not clearly say whether CUDA is supported, best-effort, or merely documented. The support
statement is already chosen in the Architecture section; this task checks the current wording
against that decision and records the exact verification command that should accompany it.

- [ ] Read the current GPU/CUDA sections and note every place where support level is implied
- [ ] Confirm the verification command that should be shown to users for checking GPU availability
- [ ] Record any wording that must stay platform-specific (for example, Windows NVIDIA guidance)

**Exit criteria:**
- The support statement is explicit and small enough to repeat consistently in docs
- The verification command is agreed before any file changes

---

## Task 2: Update installation and contributor docs

**Files:**
- Modify: `docs/installation.rst`
- Modify: `docs/development.rst`

**Background:** The installation page owns the user-facing CUDA setup path, while the development
page should tell contributors and maintainers how to verify GPU execution. Keep the wording crisp and
non-ambiguous.

- [ ] Add a short support statement to `docs/installation.rst` under the Windows GPU section
- [ ] Add the verification command block below and explain that `False` on a CPU-only machine is
  expected, not a failure
- [ ] Use this exact snippet:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Use the same one-line command in the Windows NVIDIA section so it runs in PowerShell and Command
Prompt without requiring a bash-compatible shell.

- [ ] Update `docs/development.rst` so the contributor guidance matches the same support posture
- [ ] Make sure the docs say what is supported and what remains best-effort, if anything

**Verification:**
- Build the docs locally with `uv sync --extra docs` and `uv run sphinx-build -M html docs docs/build`
- Confirm the new text does not contradict the existing Windows installation instructions

---

## Task 3: Align the roadmap and changelog

**Files:**
- Modify: `docs/roadmap.rst`
- Modify: `CHANGELOG.rst`

**Background:** The roadmap currently tracks GPU / CUDA guidance as deferred work. Once the docs are
explicit, the roadmap should reflect the updated status instead of leaving the wording vague.

- [ ] Update the roadmap GPU / CUDA note so it matches the support statement chosen in Task 1
- [ ] Add a changelog bullet that records the CUDA support clarification as a user-visible docs
  change

**Exit criteria:**
- The roadmap and docs tell the same CUDA story
- The changelog records the support clarification explicitly

---

## Task 4: Final consistency pass

**Files:**
- Re-read: `docs/installation.rst`, `docs/development.rst`, `docs/roadmap.rst`, `CHANGELOG.rst`

**Background:** This final pass checks that the support statement, verification command, and roadmap
status are consistent and that the docs do not overstate CUDA guarantees.

- [ ] Re-read the touched files and confirm the CUDA wording is consistent everywhere
- [ ] Confirm the support statement does not promise more than the project is willing to verify
- [ ] Run `uv run sphinx-build -M html docs docs/build` and confirm the docs still build cleanly
- [ ] Pause for review with a concise summary of the final CUDA guidance
