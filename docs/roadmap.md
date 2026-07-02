# Roadmap

Lightweight tracking for deferred work. Status: `planned` / `in-progress` / `done`.

---

## Benchmark hardening

**Status:** planned

Two output cleanliness issues to fix in `benchmarks/benchmark_coreml.py`:

### 1. Missing `coremltools` — not clean

When `coremltools` is not installed, each family block prints an inline `SKIPPED: ModuleNotFoundError: No module named 'coremltools'` row mixed into the results table. Should instead detect the missing module once at startup, print a single clear message (e.g. `coreml variant skipped: coremltools not installed — run: uv sync --extra coreml`), and omit the coreml column from the results table entirely.

### 2. Third-party noise in output

Several noisy lines appear during benchmark runs regardless of coremltools installation:

- **SAHI deprecation warning** — `sahi/utils/torch_utils.py:80: UserWarning: 'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'` — appears once per family run. Suppress with `warnings.filterwarnings`.
- **coremltools version warning** — `[coremltools] Torch version X.Y.Z has not been tested with coremltools` — expected and already documented in `docs/coreml.rst`; suppress in benchmark output.
- **coremltools SyntaxWarning** — `SyntaxWarning: invalid escape sequence '\_'` from `coremltools/optimize/torch/palettization/fake_palettize.py` — a coremltools packaging bug; suppress with `warnings.filterwarnings`.
- **"Loading … for CoreML inference…"** log line — emitted by `scoutbot` during model load; interrupts the per-family header. Suppress or redirect to stderr during benchmark runs.

---

## Quality workflow — v2 scope

**Status:** planned  
**Reference:** `docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md` (V2 Scope section)

1. Coverage enforcement (`--cov-fail-under`, branch coverage)
2. Expand mypy to `tests/` and `benchmarks/`; add targeted `[[tool.mypy.overrides]]` per library; revisit `ignore_missing_imports = true`
3. Ruff rule expansion — enable `UP` (restores deferred pyupgrade behaviour), `B`, `C4`, `SIM` one group per PR
4. Security / dependency scanning — `pip-audit` on `uv.lock`, tuned `bandit`
5. macOS CI runner for CoreML path (expensive; consider self-hosted M-series)

---

## CoreML / performance investigations

**Status:** planned  
**Reference:** `docs/coreml.rst` (benchmark caveats)

- Break out slicing, per-tile prediction, post-processing, and Core ML dispatch overhead in the benchmark
- Investigate parallel or batched tile inference for the CoreML path
- Confirm ANE engagement via Activity Monitor / `coremltools` compute-unit API
- M4 follow-up benchmark once pipeline overhead is reduced
