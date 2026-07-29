# Benchmark Noise Cleanup Plan

> **For agentic workers:** use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans`. Keep the scope limited to the benchmark cleanup items currently
> listed in `docs/roadmap.rst`.

**Goal:** Make `benchmarks/benchmark_coreml.py` handle missing `coremltools` cleanly and suppress
third-party noise so the benchmark output stays readable.

**Dependency:** The runtime observability plan already says it will absorb the benchmark noise
cleanup from the roadmap. If that work lands first and fully resolves these issues, this standalone
plan becomes redundant and should be closed without execution. If the issues remain after the
observability work, execute this plan as the explicit cleanup follow-up.

**Scope:** This plan only covers the two roadmap cleanup items under "Benchmark hardening":

- missing `coremltools` should be a single clean skip, not inline per-family noise
- third-party warnings and log lines should not pollute the results table

**Not in scope:** changing the structured benchmark summary format, changing runtime observability
hooks, or redesigning the performance docs.

## Global Constraints

- Use `uv run` for Python and pytest commands.
- Keep Ruff formatting on single quotes and 90-character line length.
- Keep mypy scoped to `scoutbot/` unless a task explicitly widens it.
- Do not add a blanket `pre-commit run --all-files` step.

## Task 1: Add a clean skip for missing Core ML support

**Files:**
- Modify: `benchmarks/benchmark_coreml.py`
- Test: `tests/test_benchmark_coreml.py`

**Goal:** Detect missing `coremltools` once and print one clear skip message instead of repeating an
inline `SKIPPED` row for every family.

- [x] **Step 1: Write a failing test for the missing-Core-ML path**

Add or extend a test in `tests/test_benchmark_coreml.py` that asserts:

- the benchmark prints one skip message for missing Core ML support
- the message names `coremltools` explicitly
- the results table does not grow a per-family `ModuleNotFoundError` row

- [x] **Step 2: Run the benchmark test to confirm the current behavior is still noisy**

Run:

```bash
uv run pytest tests/test_benchmark_coreml.py -v
```

Expected: the new test fails until the benchmark checks for missing `coremltools` once at startup.

- [x] **Step 3: Implement the clean skip behavior**

Update `benchmarks/benchmark_coreml.py` so it:

- checks for missing `coremltools` before the per-family loop
- prints one clear skip message such as `coreml variant skipped: coremltools not installed`
- omits the Core ML variant from the table when the dependency is unavailable

- [x] **Step 4: Re-run the benchmark test**

Run:

```bash
uv run pytest tests/test_benchmark_coreml.py -v
```

Expected: the clean-skip test passes and the benchmark output has one skip message only.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_coreml.py tests/test_benchmark_coreml.py
git commit -m "fix: clean up missing coremltools benchmark skip"
```

## Task 2: Suppress benchmark noise from third-party warnings and logs

**Files:**
- Modify: `benchmarks/benchmark_coreml.py`
- Test: `tests/test_benchmark_coreml.py`

**Goal:** Keep benchmark output readable by suppressing the known noisy warnings and the Core ML
load log line.

Specific noise to handle:

- SAHI deprecation warning: `sahi/utils/torch_utils.py:80: UserWarning: 'has_mps' is deprecated,
  please use 'torch.backends.mps.is_built()'`
- coremltools version warning: `[coremltools] Torch version X.Y.Z has not been tested with
  coremltools`
- coremltools syntax warning: `SyntaxWarning: invalid escape sequence '\_'` from
  `coremltools/optimize/torch/palettization/fake_palettize.py`
- Scoutbot Core ML load log line: `Loading ... for CoreML inference...`

- [x] **Step 1: Write a failing test for output cleanliness**

Add or extend a test in `tests/test_benchmark_coreml.py` that verifies:

- the known warnings are filtered or redirected away from the results table
- the Core ML load log line does not interrupt the family header
- the benchmark output remains readable even when those warnings are emitted upstream

- [x] **Step 2: Run the benchmark test to confirm the noise is still present**

Run:

```bash
uv run pytest tests/test_benchmark_coreml.py -v
```

Expected: the new cleanliness test fails until the warning filters and log redirection are added.

- [x] **Step 3: Implement the warning and log filtering**

Update `benchmarks/benchmark_coreml.py` so it suppresses or redirects the specific warnings and
log lines above without changing benchmark behavior.

- [x] **Step 4: Re-run the benchmark test**

Run:

```bash
uv run pytest tests/test_benchmark_coreml.py -v
```

Expected: the benchmark output stays clean and the warning/log filtering test passes.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_coreml.py tests/test_benchmark_coreml.py
git commit -m "fix: suppress benchmark output noise"
```

## Task 3: Update the roadmap note to match the cleanup

**Files:**
- Modify: `docs/roadmap.rst`

**Goal:** Mark the benchmark hardening note as implemented once the code cleanup is in place.

- [x] **Step 1: Update the roadmap entry**

Update `docs/roadmap.rst` so the "Benchmark hardening" section reflects the actual benchmark
behavior after the cleanup.

- [x] **Step 2: Verify the docs render**

Run:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Expected: the docs build succeeds and the roadmap note reads as current state, not future work.

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap.rst
git commit -m "docs: update benchmark hardening roadmap note"
```
