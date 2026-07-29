# Torch 2.13 ML-Stack Upgrade Spec

## Problem

The locked Scoutbot dependency graph currently resolves `torch==2.11.0`,
`torchvision==0.26.0`, `triton==3.6.0`, and `setuptools==81.0.0`. The
quality workflow has a temporary `uv audit` exception list for:

- `GHSA-rrmf-rvhw-rf47` / `CVE-2025-3000`: PyTorch memory corruption through
  `torch.jit.script`, fixed in `torch==2.13.0`.
- `GHSA-h35f-9h28-mq5c` / `PYSEC-2026-3447`: setuptools source-distribution
  exclusion bypass on macOS normalization-preserving filesystems, fixed in
  `setuptools==83.0.0`.

The roadmap records that a torch 2.13 stack resolution clears the current
audit failures with zero ignores. This is still not a routine dependency bump:
Scoutbot's model loading, PyTorch inference, SAHI slicing/NMS path, Core ML
export, MPS behavior, and CUDA multiprocessing validation all sit downstream of
the ML stack.

## Research Snapshot

As of 2026-07-28:

- `torch==2.13.0` was published to PyPI on 2026-07-08 with wheels for Python
  3.10, 3.11, 3.12, 3.13, and 3.14 across macOS arm64, Linux, and Windows.
  Because Python 3.10 is Scoutbot's oldest supported interpreter and PyTorch
  routinely drops old CPython wheels over time, implementation must verify the
  `cp310` wheels directly before touching the lockfile.
- PyTorch's 2.13 release candidate announcement pairs `torch==2.13.0` with
  `torchvision==0.28.0` and lists CPU, CUDA 12.6, CUDA 13.0, CUDA 13.2, ROCm,
  and XPU wheel channels.
- `torchvision==0.28.0` was published on 2026-07-08 and its package metadata
  requires Python `>=3.10, !=3.14.1`.
- `coremltools==9.0` remains the latest stable Core ML toolchain in the current
  lock. It supports Python 3.13, but Apple release notes say Core ML conversion
  support is validated against PyTorch 2.7, not 2.13. That makes Core ML export
  a required validation gate rather than an assumed pass.
- The current project docs already warn that `coremltools` may lag installed
  torch versions. This spec keeps that warning and adds a concrete validation
  requirement for the torch 2.13 upgrade.
- `ultralytics` is currently bounded as `>=8.3,<9` and resolves to 8.4.x. The
  upgrade should avoid changing that bound unless resolution or tests prove it
  necessary.
- `sahi` is pinned exactly at `0.11.36`. The upgrade should avoid changing this
  pin unless resolution or tests prove it necessary.

Sources used:

- PyPI torch 2.13.0: https://pypi.org/project/torch/2.13.0/
- PyPI torchvision: https://pypi.org/project/torchvision/
- PyTorch 2.13 release candidate announcement:
  https://dev-discuss.pytorch.org/t/pytorch-2-13-rc1-produced-for-pytorch-torchvision/3401
- OSV advisory for `GHSA-rrmf-rvhw-rf47`:
  https://osv.dev/vulnerability/GHSA-rrmf-rvhw-rf47
- OSV advisory for `PYSEC-2026-3447` / `GHSA-h35f-9h28-mq5c`:
  https://osv.dev/vulnerability/PYSEC-2026-3447
  https://osv.dev/vulnerability/GHSA-h35f-9h28-mq5c
- coremltools PyPI and release notes:
  https://pypi.org/project/coremltools/
  https://github.com/apple/coremltools/releases

## Goals

1. Upgrade the locked ML stack so `uv audit --locked --preview-features
   audit-command` passes with zero ignore flags.
2. Keep Scoutbot's declared Python support at `>=3.10,<3.14` unless the lock or
   CI proves a supported interpreter is no longer viable. If torch 2.13 cannot
   support Python 3.10, the implementation must stop for a project decision:
   either defer the torch upgrade or separately approve dropping Python 3.10
   support as a compatibility break.
3. Preserve the existing public API and output schemas.
4. Preserve default installation behavior for CPU/MPS users.
5. Validate that real model loading and inference still work for:
   - `v3`
   - `v3-cls`
   - `v3-coreml`
   - `v3-cls-coreml`
6. Validate that CUDA multiprocessing still behaves correctly enough for the
   newly-landed `batch_resumable(processes=N)` path to remain documented.
7. Update the security docs and roadmap so the temporary audit-ignore state is
   removed cleanly.

## Non-goals

1. Do not claim Python 3.14 support in this upgrade, even though torch and
   torchvision now publish Python 3.14 wheels.
2. Do not redesign the model registry, local model cache, or `loc` compatibility
   wrapper.
3. Do not require model re-export for the torch upgrade unless validation proves
   existing `.pt` artifacts fail under torch 2.13. A planned, explicit
   re-export/dill-removal slice may still run immediately before or alongside
   this upgrade if the project wants to reduce deserialization risk.
4. Do not switch Scoutbot to a custom PyTorch wheel index by default. The normal
   locked environment should continue resolving from PyPI unless a platform
   validation result proves otherwise.
5. Do not add new benchmark claims to the user docs until benchmark data has
   actually been regenerated under torch 2.13.

## Upgrade Scope

The intended minimal dependency movement is:

| Package | Current locked version | Target |
| --- | ---: | ---: |
| `torch` | `2.11.0` | `2.13.0` |
| `torchvision` | `0.26.0` | `0.28.0` |
| `triton` | `3.6.0` | resolver-selected torch 2.13 compatible version |
| `setuptools` | `81.0.0` | `>=83.0.0`, preferably resolver-selected `83.0.0` |
| CUDA runtime packages | current resolver-selected cu13 packages | resolver-selected torch 2.13 compatible set |

These packages should move only through lockfile resolution. Do not hard-pin
transitive CUDA packages by hand.

The project should continue to leave `ultralytics>=8.3,<9`,
`sahi==0.11.36`, and `coremltools>=9.0 ; sys_platform == 'darwin'` unchanged
unless the resolver or tests force a narrower decision.

## Design Decisions

### D1: Treat this as a security upgrade with behavior validation

The immediate driver is removing `uv audit` ignores. The risk profile is still
behavioral: torch changes can alter model deserialization, inference numerics,
MPS/CUDA backend behavior, and Ultralytics export paths. The implementation plan
must therefore pair the lockfile update with slow model tests and benchmark
checks.

The model-deserialization check must specifically cover PyTorch's `torch.load`
safe-loading behavior (`weights_only` defaults and safe-globals allowlists).
Scoutbot's current `.pt` files were serialized by an older Ultralytics stack and
already require `dill`; if torch 2.13 rejects an embedded class or global, the
implementation must report that directly and decide whether model re-export is
needed rather than treating it as a generic inference failure.

If the project has the source `.pt` files available, advancing the roadmap's
model re-export / dill-removal item is a strong companion task: load the current
artifacts with `dill` present, re-save them with the upgraded torch/Ultralytics
stack, republish the model files and MD5s, then remove `dill` from runtime
dependencies only after slow inference proves the new files load without it.
Keep that as an explicit task boundary so a failed re-export does not obscure
whether the base torch 2.13 dependency graph is otherwise viable.

### D2: Use the lockfile as the source of truth

The package metadata in `pyproject.toml` does not directly declare `torch` or
`torchvision`; they arrive through SAHI/Ultralytics. The implementation should
first verify that `torch==2.13.0` and `torchvision==0.28.0` publish compatible
Python 3.10 wheels for the supported platforms. Only after that should it use
`uv lock --upgrade-package torch --upgrade-package torchvision` and allow the
resolver to pull the compatible transitive stack. If setuptools remains below
83.0.0 after that, add an explicit lower bound for the build/runtime graph only
as a second step, with a documented reason.

### D3: Keep CUDA installation guidance separate from the default lock

The public `pyproject.toml`/`uv.lock` path should remain the default install
path. CUDA-specific users can still reinstall PyTorch wheels from a matching
PyTorch CUDA index as documented in `docs/installation.rst`. The validation
matrix should record which CUDA wheel family was used for any benchmark or
`processes=N` performance claim.

### D4: Core ML export is the highest-risk compatibility check

`coremltools==9.0` advertises Python 3.13 support, but Apple release notes for
9.0 name PyTorch 2.7 support. That does not prove torch 2.13 fails, but it does
mean `scoutbot export-coreml --force` and Core ML parity checks are required
before removing the roadmap blocker.

If Core ML export fails under torch 2.13, the first mitigation is to check
whether a newer `coremltools` release exists with torch 2.13 conversion support.
Only if no compatible `coremltools` upgrade is available should the project
choose between deferring the torch upgrade, accepting a documented Core ML
blocker, or considering a split platform stack.

### D5: No benchmark table update without regenerated data

Docs may say the stack was upgraded and validation passed. They must not update
Core ML or CUDA performance claims unless `benchmarks/benchmark_coreml.py` or
`benchmarks/benchmark_concurrency.py` has been rerun under the new stack and the
new input data is recorded.

## Validation Matrix

### Required fast checks

Before changing the lockfile, verify Python 3.10 wheel availability for the
target pair across Scoutbot's supported operating-system classes. Use
`--only-binary :all:` so this check fails if uv would need to build from an
sdist:

```bash
uv pip install --dry-run --only-binary :all: --python-version 3.10 \
  --python-platform linux "torch==2.13.0" "torchvision==0.28.0"
uv pip install --dry-run --only-binary :all: --python-version 3.10 \
  --python-platform macos "torch==2.13.0" "torchvision==0.28.0"
uv pip install --dry-run --only-binary :all: --python-version 3.10 \
  --python-platform windows "torch==2.13.0" "torchvision==0.28.0"
```

Expected result: resolution selects binary wheels, not an sdist build. If this
fails because Python 3.10 wheels are unavailable, stop and resolve the support
policy before proceeding.

Run these after the lockfile update:

```bash
uv sync --extra test
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv audit --locked --preview-features audit-command
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
uv run pytest -m "not slow"
```

Expected result: all pass, and `uv audit` uses no `--ignore` flags.

### Required slow checks

Run at least once on a machine with access to the example data and model cache:

```bash
uv run pytest -m slow
```

Expected result: real model fetch/load/inference tests pass for the PyTorch
configs. The validation notes must call out whether model loading exercised the
legacy `.pt` deserialization path successfully or failed on a `torch.load`
safe-globals / `weights_only` restriction. If a slow test is not runnable
because the environment lacks Git LFS or model access, record that as a blocker
rather than treating fast tests as sufficient.

### Required MPS checks

On Apple Silicon macOS, run at least one slow inference path with the PyTorch
MPS device selected:

```bash
CONFIG=v3 SCOUTBOT_DEVICE=mps uv run pytest -m slow
```

If the existing slow suite does not exercise MPS under those environment
settings, add a narrow manual smoke using `scoutbot.pipeline(...,
backend_device='mps')` over one example image and record the command/output in
the PR notes. Core ML validation does not replace this check: MPS and Core ML
use different execution paths.

### Required Core ML checks

On Apple Silicon macOS with the `coreml` extra:

```bash
uv sync --extra test --extra coreml
uv run scoutbot export-coreml --force
uv run python benchmarks/benchmark_coreml.py --summary-json notes/benchmark-coreml-torch213.json
```

Required observations:

- both Core ML configs export successfully or fail with a documented
  `coremltools`/torch incompatibility;
- parity remains within the benchmark's existing tolerance envelope;
- the benchmark output is noise-controlled and records the new torch,
  torchvision, ultralytics, and coremltools versions;
- docs are updated only with actual regenerated data.

### Required CUDA / multiprocessing checks

On a Linux CUDA machine where the `processes=N` path is expected to matter:

```bash
uv run python benchmarks/benchmark_concurrency.py \
  --modes sequential multiprocess \
  --processes 2 4 8 \
  --summary-json notes/benchmark-concurrency-torch213.json
```

Also run at least one real `batch_resumable(..., processes=N)` smoke over a
small non-production image set and confirm:

- output JSONL remains valid;
- `summary_json` is written;
- `processed + skipped + errors` accounting is coherent;
- no new worker-startup or CUDA-context regressions appear.

### Supported-interpreter checks

CI remains the canonical gate for Python 3.10, 3.11, 3.12, 3.13, and Windows
3.13 smoke. If local validation is limited to one interpreter, the PR must say
which checks were local and which are deferred to CI.

## Implementation Outline

The eventual implementation plan should be split into four reviewable tasks:

1. **Lockfile upgrade and audit cleanup**
   - verify `torch==2.13.0` / `torchvision==0.28.0` Python 3.10 wheel
     availability before changing the lock;
   - update `uv.lock` through the resolver;
   - remove the three `--ignore` flags from `.github/workflows/quality.yml`;
   - confirm `uv audit --locked --preview-features audit-command` passes.
2. **Fast quality and slow inference validation**
   - run and record fast quality gates;
   - run and record slow model tests;
   - fix only regressions directly caused by the upgrade.
3. **Core ML and CUDA validation**
   - run the explicit Apple Silicon MPS check;
   - run Core ML export/parity on macOS;
   - run CUDA concurrency checks where hardware is available;
   - capture benchmark JSON under `notes/` only if the project wants to keep
     the raw evidence in-tree.
4. **Documentation pass**
   - update `CHANGELOG.rst`;
   - update `docs/roadmap.rst` to mark the blocker done or to document a
     remaining upstream incompatibility;
   - update `docs/security.rst`, `CONTRIBUTING.md`, and
     `.github/workflows/quality.yml` comments so they no longer mention stale
     `[tool.uv.audit]` waiver configuration;
   - update `docs/coreml.rst` only with observed Core ML results.

Per local project rules, agents executing the plan must not run `git add`,
`git commit`, or `pre-commit`; the user owns staging and commits.

The `[tool.uv.audit]` wording in `docs/security.rst`, `CONTRIBUTING.md`, and
`CHANGELOG.rst` is already stale today because `uv audit` ignores are CLI-only
in this project. That documentation bug can be fixed before the torch upgrade;
the upgrade's documentation pass should then remove the ignore-list discussion
entirely once the audit gate is clean.

## Failure Policy

If fast tests pass but Core ML export fails, the security upgrade should not be
merged as a normal completion. First check whether a newer `coremltools` release
supports torch 2.13 conversion and can be safely adopted. If not, choose one of:

1. keep torch 2.11 with documented audit ignores until Core ML tooling catches
   up;
2. land torch 2.13 while temporarily disabling Core ML export docs/configs, only
   if the project explicitly accepts that regression;
3. pin a split platform stack only if `uv` can express it cleanly and CI can
   validate it.

Option 1 is the default recommendation unless the security posture requires
moving immediately.

If CUDA multiprocessing regresses but single-process inference and Core ML
remain healthy, the upgrade may still be acceptable, but the roadmap and
performance docs must downgrade `processes=N` guidance until a follow-up fix
lands.

## Acceptance Criteria

The upgrade is complete only when:

1. `uv.lock` resolves to `torch==2.13.0`, `torchvision==0.28.0`, and
   `setuptools>=83.0.0`.
2. `.github/workflows/quality.yml` contains no `uv audit --ignore` flags.
3. `uv audit --locked --preview-features audit-command` passes with zero
   advisories.
4. Fast quality checks pass locally or in CI.
5. Slow model inference tests pass on at least one supported Python 3.13
   environment.
6. Apple Silicon MPS inference has been explicitly run or is explicitly blocked
   by hardware availability.
7. Core ML export/parity has been explicitly run or is explicitly blocked with
   an upstream-compatibility note.
8. CUDA multiprocessing validation has been explicitly run or marked
   hardware-unavailable with no new public performance claim.
9. `CHANGELOG.rst`, `docs/roadmap.rst`, `docs/security.rst`, and
   `CONTRIBUTING.md` reflect the final state.

## Open Questions

1. Should the project keep raw benchmark JSON outputs in `notes/` for this
   upgrade, or only summarize them in `docs/performance.rst`?
2. Should CI add a scheduled macOS/Core ML validation job later, or is local
   maintainer validation sufficient for now?
