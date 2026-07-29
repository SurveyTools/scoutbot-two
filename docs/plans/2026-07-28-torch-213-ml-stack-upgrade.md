# Torch 2.13 ML-Stack Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Scoutbot's locked ML stack to torch 2.13, clear the temporary `uv audit` ignores, and validate the model-loading, MPS, Core ML, and CUDA/multiprocess paths affected by the upgrade.

**Architecture:** Treat the dependency bump as a lockfile/security change plus explicit ML behavior validation. Keep the default install path on PyPI-resolved wheels, avoid unrelated SAHI/Ultralytics/Core ML changes unless validation forces them, and keep model re-export/dill removal as a bounded companion task rather than an implicit side effect.

**Tech Stack:** `uv`, PyTorch, torchvision, Ultralytics, SAHI, coremltools, pytest, Ruff, mypy, Bandit, Sphinx docs.

**Spec:** `docs/specs/2026-07-28-torch-213-ml-stack-upgrade-design.md`

---

## Global Constraints

- Use `uv run ...` for project Python commands. Do not use bare `pytest`, bare `ruff`, bare `mypy`, bare `bandit`, or bare `scoutbot`.
- Use `uv ...` directly for `uv lock`, `uv sync`, `uv pip install`, and `uv audit`.
- Do not run `pre-commit`.
- Do not run `git add`, `git commit`, or `git push`. The user owns staging and commits. Each task ends with a suggested commit message for the user only.
- Do not broaden this plan into unrelated registry, `loc`, output-schema, or public-release work.
- Ruff formatting uses single quotes and 90-character lines.
- Mypy scope is `uv run mypy scoutbot tools/visualise.py`, matching current CI.
- If a command needs network access and fails from sandbox or network restrictions, rerun it with escalation rather than silently substituting a weaker check.
- Pause after each task for user review.

## Prerequisite

This plan assumes the Windows-smoke numpy fix has already landed in the working
tree: `pyproject.toml` must include `numpy>=2.1`, and the lockfile must resolve
numpy 2.x. That work is complementary, not overlapping: the numpy floor protects
Python 3.13 / Windows import behavior, while this plan clears the torch and
setuptools audit findings.

## File Map

| Path | Action | Purpose |
| --- | --- | --- |
| `uv.lock` | Modify | Lock torch 2.13 / torchvision 0.28 / setuptools >=83 and compatible transitive packages. |
| `.github/workflows/quality.yml` | Modify | Remove the temporary `uv audit --ignore` flags and stale comments. |
| `pyproject.toml` | Possibly modify | Only if the resolver does not pull `setuptools>=83.0.0` without an explicit bound, or if dill removal is approved and verified. |
| `scoutbot/model_store.py` | Possibly modify | Only if the optional model re-export/dill-removal companion task is approved and new hashes/URLs are available. |
| `docs/roadmap.rst` | Modify | Mark the torch 2.13 blocker done, blocked, or narrowed based on validation. Update the model re-export/dill item only if that companion work lands. |
| `CHANGELOG.rst` | Modify | Record the ML-stack upgrade and audit-ignore removal; record model re-export/dill removal only if it lands. |
| `docs/security.rst` | Modify | Fix stale `[tool.uv.audit]` guidance and document the final dependency-audit posture. |
| `CONTRIBUTING.md` | Modify | Fix stale `[tool.uv.audit]` contributor guidance. |
| `docs/coreml.rst` | Modify | Update only with observed torch/coremltools validation results. |
| `docs/performance.rst` | Possibly modify | Update only if new benchmark data is actually generated and accepted. |
| `notes/*.json` | Possibly create | Optional raw benchmark evidence, if the user wants it retained in-tree. |

## Task 1: Preflight Resolution and Wheel Support

**Files:**
- Read: `pyproject.toml`
- Read: `uv.lock`
- Read: `docs/specs/2026-07-28-torch-213-ml-stack-upgrade-design.md`

**Purpose:** Prove the target torch/torchvision pair can support Scoutbot's declared Python/platform envelope before touching the lockfile.

- [ ] **Step 1: Confirm current locked baseline**

Run:

```bash
uv tree --locked | rg '^(torch|torchvision|triton|setuptools|ultralytics|sahi|coremltools|numpy|pillow)'
```

Expected: output includes current versions similar to:

```text
torch v2.11.0
torchvision v0.26.0
setuptools v81.0.0
ultralytics v8.4.x
sahi v0.11.36
numpy v2.x
```

If `uv tree --locked` output formatting changes, record the equivalent exact versions from `uv.lock` with:

```bash
rg -n '^name = "(torch|torchvision|triton|setuptools|ultralytics|sahi|coremltools|numpy)"|^version = ' uv.lock
```

- [ ] **Step 2: Verify Python 3.10 wheel availability for target packages**

Run all three dry runs. These must select binary wheels and must not attempt source builds:

```bash
uv pip install --dry-run --only-binary :all: --python-version 3.10 \
  --python-platform linux "torch==2.13.0" "torchvision==0.28.0"
uv pip install --dry-run --only-binary :all: --python-version 3.10 \
  --python-platform macos "torch==2.13.0" "torchvision==0.28.0"
uv pip install --dry-run --only-binary :all: --python-version 3.10 \
  --python-platform windows "torch==2.13.0" "torchvision==0.28.0"
```

Expected: each command resolves successfully. If any command fails because no Python 3.10 wheel exists, stop the plan and report the decision fork:

- defer the torch upgrade and keep documented audit ignores; or
- approve a separate compatibility-breaking Python 3.10 support removal plan.

- [ ] **Step 3: Check coremltools mitigation availability**

Run:

```bash
uv pip install --dry-run --only-binary :all: --python-version 3.13 \
  --python-platform macos "coremltools>=9.0"
```

Expected: resolution identifies the newest compatible `coremltools` wheel for Python 3.13/macOS. Record the version in the task notes. Do not bump `coremltools` yet; this is mitigation information for Task 4 if Core ML export fails.

- [ ] **Step 4: Record Task 1 outcome**

Add a short note to your handoff message with:

- current locked torch/torchvision/setuptools versions;
- current locked numpy version and confirmation that `pyproject.toml` has `numpy>=2.1`;
- whether Python 3.10 target wheels resolved on Linux/macOS/Windows;
- newest available compatible `coremltools` version seen by the dry run;
- whether the plan can continue.

Suggested commit message for user after review, if they commit this task's notes separately:

```text
docs: record torch 2.13 upgrade preflight
```

## Task 2: Lockfile Upgrade and Audit Cleanup

**Files:**
- Modify: `uv.lock`
- Modify: `.github/workflows/quality.yml`
- Possibly modify: `pyproject.toml`

**Purpose:** Move the locked dependency graph to torch 2.13 and remove the temporary dependency-audit ignores.

- [ ] **Step 1: Update the lockfile through uv**

Run:

```bash
uv lock \
  --upgrade-package torch \
  --upgrade-package torchvision \
  --upgrade-package setuptools
```

Expected: `uv.lock` changes and the resolver completes without conflict.

- [ ] **Step 2: Inspect the resolved package versions**

Run:

```bash
rg -n '^name = "(torch|torchvision|triton|setuptools|ultralytics|sahi|coremltools|numpy)"|^version = ' uv.lock
```

Expected:

- `torch` is `2.13.0`;
- `torchvision` is `0.28.0`;
- `setuptools` is `83.0.0` or newer;
- `numpy` remains `2.1` or newer;
- `ultralytics` remains within `>=8.3,<9`;
- `sahi` remains `0.11.36`;
- `coremltools` remains compatible with `coremltools>=9.0 ; sys_platform == 'darwin'`.

If `setuptools` remains below `83.0.0`, add an explicit dependency bound in `pyproject.toml`:

```toml
dependencies = [
    ...
    "setuptools>=83.0.0",
    ...
]
```

Place it near the other runtime dependencies with a short comment only if needed:

```toml
# Required until the ML stack pulls setuptools >=83 transitively; clears
# GHSA-h35f-9h28-mq5c / PYSEC-2026-3447 in the locked graph.
"setuptools>=83.0.0",
```

Then rerun:

```bash
uv lock
```

- [ ] **Step 3: Remove `uv audit` ignore flags from quality workflow**

In `.github/workflows/quality.yml`, replace the current dependency audit step:

```yaml
      - name: Dependency audit
        # `uv audit` reads ignores from CLI flags only (not pyproject config).
        # Each accepted CVE is build-time or an unused codepath; the durable fix
        # is the torch 2.13 ML-stack upgrade (docs/roadmap.rst, top priority),
        # which clears all three and lets these flags be removed.
        #   GHSA-rrmf-rvhw-rf47  torch.jit.script memory corruption — unused in ScoutBot
        #   GHSA-h35f-9h28-mq5c  setuptools MANIFEST.in sdist bypass — build-time only
        #   PYSEC-2026-3447      setuptools (same, fixed in 83.0.0; gated behind torch 2.13)
        run: >
          uv audit --locked --preview-features audit-command
          --ignore GHSA-rrmf-rvhw-rf47
          --ignore GHSA-h35f-9h28-mq5c
          --ignore PYSEC-2026-3447
```

with:

```yaml
      - name: Dependency audit
        run: uv audit --locked --preview-features audit-command
```

- [ ] **Step 4: Run the audit gate**

Run:

```bash
uv audit --locked --preview-features audit-command
```

Expected: pass with zero advisories and no ignore flags.

If the audit still fails, do not add new ignores in this task. Record the advisory IDs and stop for review.

- [ ] **Step 5: Sync the upgraded environment**

Run:

```bash
uv sync --extra test
```

Expected: environment sync completes. This sync intentionally installs the test
environment only; it may prune docs/coreml extras from the local venv. Later
tasks re-sync with `--extra coreml` and `--extra docs` before running those
checks.

- [ ] **Step 6: Record Task 2 outcome**

Report:

- exact resolved torch, torchvision, setuptools, numpy, ultralytics, sahi, and coremltools versions;
- whether `pyproject.toml` needed an explicit setuptools bound;
- whether `uv audit` passed without ignores.

Suggested commit message for user:

```text
build: upgrade torch stack and clear audit ignores
```

## Task 3: Fast Quality and Offline Regression Gate

**Files:**
- Possibly modify: source or tests only if a regression is directly caused by the upgrade

**Purpose:** Prove the normal CI-quality path still passes after the lockfile change.

- [ ] **Step 1: Run formatter check**

Run:

```bash
uv run ruff format --check .
```

Expected: pass. If it fails, run `uv run ruff format .`, inspect the diff, and keep only formatting changes.

- [ ] **Step 2: Run Ruff lint**

Run:

```bash
uv run ruff check .
```

Expected: pass. If it fails, fix only issues introduced or surfaced by the upgrade.

- [ ] **Step 3: Run mypy**

Run:

```bash
uv run mypy scoutbot tools/visualise.py
```

Expected: pass.

- [ ] **Step 4: Run Bandit**

Run:

```bash
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
```

Expected: pass.

- [ ] **Step 5: Run fast tests**

Run:

```bash
uv run pytest -m "not slow"
```

Expected: pass.

- [ ] **Step 6: If regressions occur, classify them before fixing**

Use this classification in the handoff:

- **Lockfile-only issue:** resolver selected an incompatible transitive package.
- **Torch behavior issue:** import/model/deserialization/backend behavior changed.
- **Test assumption issue:** test patched or asserted a version-specific behavior that should be updated.
- **Pre-existing issue surfaced by dependency change:** not caused by the upgrade; stop for review before broadening scope.

Do not make broad refactors in this task.

- [ ] **Step 7: Record Task 3 outcome**

Report the exact commands run and pass/fail status.

Suggested commit message for user:

```text
test: validate fast suite on torch 2.13 stack
```

## Task 4: Slow PyTorch, MPS, and Deserialization Validation

**Files:**
- Possibly modify: source/tests only for direct torch 2.13 regressions
- Possibly modify: `docs/roadmap.rst` only if validation reveals a blocker that changes the plan

**Purpose:** Prove the real `.pt` model loading path still works, including the old Ultralytics/dill serialization edge and MPS.

- [ ] **Step 1: Run slow tests on the default device**

Run:

```bash
uv run pytest -m slow
```

Expected: pass. In the task notes, explicitly state whether the run fetched/loaded the real `v3` and `v3-cls` `.pt` files.

If this fails with `torch.load`, `weights_only`, `safe_globals`, missing class/global, or `dill` errors, stop and move to the Task 5 decision point before making code changes.

- [ ] **Step 2: Run the authoritative MPS smoke on Apple Silicon**

The slow pytest suite is not sufficient proof of MPS coverage because some slow
tests pass an explicit `backend_device='cpu'`, bypassing `SCOUTBOT_DEVICE`.
On Apple Silicon macOS, run a direct pipeline smoke that explicitly passes
`backend_device='mps'`:

```bash
uv run python -c "import scoutbot; print(scoutbot.pipeline('examples/0d4e4df2-7b69-91b1-1985-c8421f2f3253.jpg', config='v3', backend_device='mps')[0])"
```

Expected: command completes and prints the `wic` value for the image, or fails
with a clear MPS/backend error. If that example file is unavailable locally, use
one existing `examples/*.jpg` image and record the exact path.

Optionally, run the environment-variable form afterward as supporting context,
but do not count it as the MPS acceptance gate unless output or instrumentation
proves MPS was actually used:

```bash
CONFIG=v3 SCOUTBOT_DEVICE=mps uv run pytest -m slow
```

- [ ] **Step 3: Confirm no hidden `dill` dependency decision is being made**

Run:

```bash
uv run python -c "import dill, torch, ultralytics; print('dill', dill.__version__); print('torch', torch.__version__); print('ultralytics', ultralytics.__version__)"
```

Expected: confirms `dill` is still present unless Task 5 later removes it. This task should not remove `dill`.

- [ ] **Step 4: Record Task 4 outcome**

Report:

- default slow test result;
- MPS validation result;
- whether any model load/deserialization warning or failure appeared;
- whether Task 5 should be executed now or deferred.

Suggested commit message for user:

```text
test: validate slow model inference on torch 2.13
```

## Task 5: Model Re-export / Dill Removal Checkpoint

**Files:**
- Read: `pyproject.toml`
- Read: `docs/roadmap.rst`
- Read: `scoutbot/model_store.py`

**Purpose:** Decide whether model re-export/dill removal should become its own follow-up plan now. This task does not edit model artifacts or dependencies.

- [ ] **Step 1: Check whether this branch needs model re-export**

Use Task 4 results to classify the need:

- **Required now:** slow tests fail because torch 2.13 cannot load the current `.pt` files due to `dill`, `torch.load`, `weights_only`, or safe-globals behavior.
- **Recommended but separable:** slow tests pass with `dill` present, but the team wants to remove the temporary runtime dependency before public release.
- **Deferred:** slow tests pass and public-release timing favors clearing the audit blocker first.

- [ ] **Step 2: Record the exact prerequisites for a separate re-export plan**

If model re-export should proceed, record these facts in the task handoff:

- local paths to the current source `.pt` files;
- whether the replacement artifacts should overwrite the current Azure/GCS URLs or publish under new URLs;
- which model source is canonical when Azure and GCS both exist;
- whether old MD5s must remain documented for rollback;
- whether Core ML `.mlpackage` files should be regenerated in the same follow-up or only on demand via `export-coreml`;
- whether `dill` removal is required in the same follow-up or only after re-export validation.

- [ ] **Step 3: Run only a non-mutating model-load probe if local source paths are known**

If source model paths are available locally, run the same non-mutating class-name load against each current `.pt` path. Use only real local files supplied by the user. Example command shape, with the final argument replaced by a real path before running:

```bash
uv run python -c "from ultralytics import YOLO; import sys; m=YOLO(sys.argv[1]); print(m.names)" REAL_LOCAL_MODEL_PATH.pt
```

Run it once for the one-class model and once for the class-aware model. Expected: both models load and print classes under torch 2.13 with `dill` still installed. If local source paths are not available, skip this step and record that the follow-up plan needs exact local model paths before it can be executed.

- [ ] **Step 4: Decide whether to write a separate model re-export plan**

If the classification is **Required now** or the user wants the companion work before public release, stop this plan after Task 5 and write a dedicated model re-export / dill-removal implementation plan. That follow-up plan must include exact source paths, publishing destinations, MD5 generation, config updates, slow inference validation without `dill`, and docs updates.

If the classification is **Deferred**, continue to Task 6 with `dill>=0.4.0` unchanged.

Suggested commit message for user if Task 5 records a deferral:

```text
docs: defer dill removal after torch upgrade
```

## Task 6: Core ML Export and Benchmark Validation

**Files:**
- Possibly modify: `docs/coreml.rst`
- Possibly modify: `docs/performance.rst`
- Possibly create: `notes/benchmark-coreml-torch213.json`

**Purpose:** Validate the highest-risk Apple/Core ML path under the upgraded torch stack.

- [ ] **Step 1: Sync with Core ML extra**

On Apple Silicon macOS, run:

```bash
uv sync --extra test --extra coreml
```

Expected: sync completes.

- [ ] **Step 2: Export Core ML packages from scratch**

Run:

```bash
uv run scoutbot export-coreml --force
```

Expected: both `v3-coreml` and `v3-cls-coreml` export successfully.

If export fails with a torch/coremltools compatibility error, do not disable Core ML in code. First test whether a newer `coremltools` release identified in Task 1 fixes the export. If it does, update the `coreml` extra in `pyproject.toml` to the minimum working version and rerun:

```bash
uv lock
uv sync --extra test --extra coreml
uv run scoutbot export-coreml --force
```

If no `coremltools` version fixes export, stop for review and use the spec's failure policy.

- [ ] **Step 3: Run Core ML benchmark**

Run:

```bash
uv run python benchmarks/benchmark_coreml.py --summary-json notes/benchmark-coreml-torch213.json
```

Expected:

- benchmark completes;
- human output is not polluted by known third-party warning noise;
- JSON summary is written;
- parity remains within the existing benchmark tolerance envelope.

If the user does not want raw benchmark JSON retained in git, use a temporary path instead:

```bash
uv run python benchmarks/benchmark_coreml.py --summary-json /private/tmp/benchmark-coreml-torch213.json
```

- [ ] **Step 4: Inspect benchmark metadata**

Run:

```bash
uv run python -c "import json; p='notes/benchmark-coreml-torch213.json'; data=json.load(open(p)); print(type(data).__name__); print(data.keys() if isinstance(data, dict) else 'list-items', len(data) if isinstance(data, list) else '')"
```

If using `/private/tmp`, replace the path in the command.

Expected: JSON is parseable. Record where the benchmark captured version metadata; if it does not record torch/torchvision/ultralytics/coremltools versions, note that as a follow-up rather than inventing benchmark schema changes in this task.

- [ ] **Step 5: Record Task 6 outcome**

Report:

- Core ML export result;
- coremltools version used;
- benchmark command/path;
- whether docs can make new performance claims or should only record validation.

Suggested commit message for user:

```text
test: validate core ml export on torch 2.13
```

## Task 7: CUDA Multiprocessing Validation

**Files:**
- Possibly create: `notes/benchmark-concurrency-torch213.json`
- Possibly modify: `docs/performance.rst`
- Possibly modify: `docs/roadmap.rst`

**Purpose:** Validate that the new `batch_resumable(processes=N)` path still behaves under the upgraded stack.

- [ ] **Step 1: Run concurrency benchmark on Linux CUDA**

On a Linux CUDA machine, run:

```bash
uv run python benchmarks/benchmark_concurrency.py \
  --modes sequential multiprocess \
  --processes 2 4 8 \
  --summary-json notes/benchmark-concurrency-torch213.json
```

Expected: benchmark completes and writes JSON.

If no Linux CUDA machine is available in this task, skip to Step 4 and record the hardware limitation. Do not make new performance claims.

- [ ] **Step 2: Run a real resumable multiprocess smoke**

Use a small local image directory and write outputs to `/private/tmp`:

```bash
uv run scoutbot batch-resumable examples/*.jpg \
  -o /private/tmp/scoutbot-torch213-mp.jsonl \
  --processes 2 \
  --summary-json /private/tmp/scoutbot-torch213-mp-summary.json
```

The `batch-resumable` command takes image files as positional arguments and a required
`-o/--output` JSONL path — not a directory plus a positional output path.

Expected:

- command completes or fails with a clear, resume-safe error;
- `/private/tmp/scoutbot-torch213-mp.jsonl` contains valid JSONL;
- summary JSON is written if the command completes.

- [ ] **Step 3: Validate JSONL and summary files**

Run:

```bash
uv run python -c "import json, pathlib; p=pathlib.Path('/private/tmp/scoutbot-torch213-mp.jsonl'); lines=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; print(len(lines)); s=json.load(open('/private/tmp/scoutbot-torch213-mp-summary.json')); print(s.get('mode'), s.get('processes'), s.get('processed'), s.get('skipped'), s.get('errors'))"
```

Expected: JSON parses and summary accounting is coherent.

- [ ] **Step 4: Record Task 7 outcome**

Report:

- whether CUDA hardware was available;
- benchmark result path, if run;
- `batch-resumable --processes` smoke result, if run;
- whether any public performance guidance should change.

Suggested commit message for user:

```text
test: validate cuda multiprocess path on torch 2.13
```

## Task 8: Documentation and Roadmap Pass

**Files:**
- Modify: `CHANGELOG.rst`
- Modify: `docs/roadmap.rst`
- Modify: `docs/security.rst`
- Modify: `CONTRIBUTING.md`
- Modify: `.github/workflows/quality.yml`
- Possibly modify: `docs/coreml.rst`
- Possibly modify: `docs/performance.rst`
- Possibly modify: `docs/models.rst`

**Purpose:** Make contributor, security, and user-facing docs match the final state.

- [ ] **Step 1: Fix stale uv-audit waiver guidance**

In `docs/security.rst`, replace wording that says durable exceptions live in `[tool.uv.audit]` with wording matching the actual project behavior:

```rst
Temporary audit exceptions, if any are accepted, live directly on the
``uv audit`` command in ``.github/workflows/quality.yml`` with comments naming
the advisory, rationale, and planned removal path. ``uv audit`` does not read a
project ``[tool.uv.audit]`` table in this repository.
```

In `CONTRIBUTING.md`, replace the stale sentence:

```md
If an advisory has no available fix, document the waiver in `[tool.uv.audit]` in `pyproject.toml`
with a rationale and removal condition before opening a PR.
```

with:

```md
If an advisory has no available fix, document any temporary waiver directly on the
`uv audit` command in `.github/workflows/quality.yml`, including the advisory ID,
rationale, and removal condition before opening a PR.
```

- [ ] **Step 2: Update changelog**

Add an Unreleased bullet to `CHANGELOG.rst` describing the final state.

If the upgrade fully lands:

```rst
* Upgraded the locked ML stack to ``torch==2.13.0`` / ``torchvision==0.28.0``
  and removed the temporary ``uv audit`` ignores for the torch and setuptools
  advisories after the locked graph audited cleanly.
```

If model re-export/dill removal also lands, add a second bullet:

```rst
* Re-exported the bundled KAZA model artifacts with the current ML stack and
  removed the temporary runtime ``dill`` dependency after slow inference passed
  without it.
```

If validation blocks the upgrade, instead add a short note only if the user wants the blocker recorded in the changelog; otherwise keep it in the roadmap.

- [ ] **Step 3: Update roadmap**

In `docs/roadmap.rst`, update `Dependency security — torch 2.13 ML-stack upgrade`:

- mark it done if all acceptance criteria passed;
- keep it in-progress and name the exact blocker if Core ML, MPS, CUDA, or slow model validation failed;
- remove the interim `pillow`/ignore wording only if the ignores are gone.

If Task 5 removed dill, mark `Model re-export / dill removal` done or move it to the completed archive. If Task 5 was deferred, leave the roadmap item planned and optionally add a sentence that it remains separate from the torch 2.13 upgrade.

- [ ] **Step 4: Update Core ML docs only with observed results**

In `docs/coreml.rst`, update the Torch/coremltools caveat based on Task 6.

If export passed under torch 2.13:

```rst
* **Torch / coremltools version pairing.** The export and inference paths have
  been validated with Torch 2.13.0 and the locked ``coremltools`` version listed
  in ``uv.lock``. Future Torch or ``coremltools`` bumps should re-run
  ``scoutbot export-coreml --force`` and the Core ML benchmark before updating
  performance claims.
```

If export failed, preserve the caveat and add the specific blocker instead of claiming validation.

- [ ] **Step 5: Update performance docs only if benchmark data was accepted**

If Task 6 or Task 7 generated accepted benchmark data and the user wants docs updated, update `docs/performance.rst` with the new results and mention the torch/coremltools/CUDA stack used.

If no accepted benchmark data was generated, do not change performance tables. Add no speculative performance wording.

- [ ] **Step 6: Run docs build**

Run:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Expected: Sphinx build succeeds.

- [ ] **Step 7: Final full local gate**

Run the local gate that is feasible on the current machine:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv audit --locked --preview-features audit-command
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
uv run pytest -m "not slow"
```

Expected: all pass.

- [ ] **Step 8: Documentation pass completed**

Review:

- `README.md`
- `docs/installation.rst`
- `docs/configuration.rst`
- `docs/coreml.rst`
- `docs/performance.rst`
- `docs/models.rst`
- `docs/security.rst`
- `docs/development.rst`
- `CONTRIBUTING.md`
- `CHANGELOG.rst`
- `docs/roadmap.rst`

Update any stale torch/audit/Core ML/dill wording, or record in the task handoff why no update was needed.

Suggested commit message for user:

```text
docs: close torch 2.13 upgrade documentation
```

## Final Handoff Checklist

- [ ] Python 3.10 target wheel preflight passed.
- [ ] `uv.lock` resolves to `torch==2.13.0`, `torchvision==0.28.0`, and `setuptools>=83.0.0`.
- [ ] `.github/workflows/quality.yml` has no `uv audit --ignore` flags.
- [ ] `uv audit --locked --preview-features audit-command` passes with zero advisories.
- [ ] Fast quality gates pass.
- [ ] Slow model tests pass or blocker is documented.
- [ ] MPS validation ran or hardware unavailability is documented.
- [ ] Core ML export/benchmark ran or blocker is documented.
- [ ] CUDA multiprocessing validation ran or hardware unavailability is documented.
- [ ] Dill removal either landed with re-exported models or remains explicitly deferred.
- [ ] Docs and roadmap reflect the final state.
- [ ] User reviewed task outputs before staging/committing.
