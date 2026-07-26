# Core ML Compute-Unit Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `requested=unknown actual=unknown` in Core ML benchmark rows with explicit
requested / actual compute-unit metadata, or `null` when the runtime cannot know the actual target.

**Architecture:** This is a focused follow-up to the performance observability work. Core ML config
owns the requested value, the Core ML runtime wrapper emits the best available actual value, and the
benchmark prints exactly what the event stream reports. No filename heuristics, no platform
guessing, and no duplicate benchmark-hardening work in this plan.

**Tech Stack:** Python, pytest, `scoutbot._observability`, `scoutbot.loc`,
`scoutbot.tile_batched.main`, `benchmarks/benchmark_coreml.py`, Sphinx docs.

---

## Global Constraints

- Use `uv run` for Python, pytest, Ruff, and mypy commands.
- Keep Ruff formatting on single quotes and 90-character line length.
- Keep mypy scoped to `scoutbot/` unless a task explicitly widens it.
- Do not add a blanket `pre-commit run --all-files` step.
- Preserve the existing no-op behavior when observability is disabled.

---

## Interfaces

- `loc.fetch(..., observer=...)` and `loc.export_coreml(..., observer=...)` should emit
  `compute_units_requested` for Core ML configs only.
- `tile_batched.Yolov8DetectionModel(..., observer=...)` should emit `compute_units_actual` for
  Core ML inference paths only.
- `compute_units_requested` is sourced from the Core ML config, not inferred from the model name.
- `compute_units_actual` is sourced from the Core ML runtime / compute-plan probe; if the runtime
  cannot determine it reliably, emit `null`.
- Canonical compute-unit strings should be lower-case enum names: `cpu_only`, `cpu_and_gpu`,
  `cpu_and_ne`, and `all`.
- `benchmarks/benchmark_coreml.py` should render those values directly from the observability
  stream. Non-Core ML rows should omit the compute-unit suffix entirely rather than printing
  placeholder values.

---

## Task 1: Pin the compute-unit contract in tests

**Files:**
- Modify: `tests/test_loc.py`
- Modify: `tests/test_tile_batched.py`
- Modify: `tests/test_benchmark_coreml.py`
- Modify: `tests/test_benchmark_coreml_observability.py`

**Goal:** Write tests that fail until Core ML configs and runtime events carry explicit compute-unit
metadata.

- [ ] **Step 1: Add failing tests for Core ML metadata emission**

Add tests that assert:

- `loc.CONFIGS['v3-coreml']['compute_units_requested']` and
  `loc.CONFIGS['v3-cls-coreml']['compute_units_requested']` are explicit values, not missing
- `loc.fetch(..., observer=...)` emits `compute_units_requested` for Core ML configs
- `loc.fetch(..., observer=...)` does not emit `compute_units_requested` for `v3` or `v3-cls`
- `Yolov8DetectionModel(..., observer=...)` emits `compute_units_actual` on Core ML inference
  paths
- benchmark summaries collapse those fields into real values, and render `null` instead of
  `unknown` when a field is explicitly absent

Use concrete assertions, not guessed value sets. For example:

```python
requested = loc.CONFIGS['v3-coreml']['compute_units_requested']
assert fetch_event['compute_units_requested'] == requested
assert event['compute_units_actual'] == 'cpu_and_gpu'
assert summary['compute_units_requested'] == requested
assert summary['compute_units_actual'] == 'cpu_and_gpu'
```

Patch the private Core ML probe in the `tile_batched` test so the actual value stays synthetic.
Use a fixed return value like `cpu_and_gpu` there; do not require real Apple Silicon hardware for
the contract test. The test should patch `_resolve_compute_units_actual`, not guess at a lower-level
probe name.

- [ ] **Step 2: Run the targeted tests to confirm they fail**

Run:

```bash
uv run pytest tests/test_loc.py tests/test_tile_batched.py tests/test_benchmark_coreml.py tests/test_benchmark_coreml_observability.py -v
```

Expected: the new assertions fail until the runtime and benchmark emit the compute-unit fields.

---

## Task 2: Emit requested and actual compute units from the runtime

**Files:**
- Modify: `scoutbot/loc/__init__.py`
- Modify: `scoutbot/tile_batched/main.py`

**Goal:** Populate the compute-unit metadata at the source so benchmark output can stop reporting
`unknown`.

- [ ] **Step 1: Run the failing metadata tests**

Run:

```bash
uv run pytest tests/test_loc.py tests/test_tile_batched.py tests/test_benchmark_coreml.py tests/test_benchmark_coreml_observability.py -v
```

Expected: the metadata assertions fail because the runtime is not yet emitting the fields.

- [ ] **Step 2: Record the requested compute units in the Core ML configs and fetch events**

Update `scoutbot/loc/__init__.py` so the Core ML config entries carry an explicit requested
compute-unit value. Thread that value into the `fetch()` and `export_coreml()` observability events
as `compute_units_requested`.

Keep non-Core ML configs unchanged. If a path is not Core ML, the field should stay absent or
`null` rather than being guessed.

- [ ] **Step 3: Add a Core ML runtime probe for the actual compute units**

Update `scoutbot/tile_batched/main.py` so the Core ML load path resolves the runtime compute-units
result with a small private helper, then emits it as `compute_units_actual` on the timed inference
event.

Use a helper like `_resolve_compute_units_actual(model_path) -> str | None` that loads the compiled
Core ML bundle with `ct.models.MLModel(str(model_path))`, reads its `compute_unit`, and maps the
enum to the canonical strings above with `.name.lower()`. The helper may return `None` if
`coremltools` cannot report a compute unit. Do not add a separate `ImportError` guard; the helper
runs only on the Core ML path after the model has already been identified as Core ML. The helper
runs in `load_model()` so the extra probe cost is paid once per model load, not once per tile.
Store the result on the model instance as `self._compute_units_actual` in `load_model()`, then read
that instance variable back inside `perform_inference()` when emitting the timed inference event.
Initialize `self._compute_units_actual = None` in `__init__` alongside `self._coreml = False` so
non-Core ML models never raise `AttributeError` when the inference event reads the field.

The implementation should:

- report the resolved runtime target when it is known
- emit `null` when the runtime cannot determine the actual target
- keep non-Core ML inference behavior unchanged

- [ ] **Step 4: Re-run the targeted metadata tests**

Run:

```bash
uv run pytest tests/test_loc.py tests/test_tile_batched.py tests/test_benchmark_coreml.py tests/test_benchmark_coreml_observability.py -v
```

Expected: the tests pass and the emitted metadata is no longer `unknown` on Core ML runs.

---

## Task 3: Teach the benchmark and docs to surface the new metadata

**Files:**
- Modify: `benchmarks/benchmark_coreml.py`
- Modify: `docs/coreml.rst`
- Modify: `docs/performance.rst`
- Modify: `docs/roadmap.rst`
- Modify: `CHANGELOG.rst`

**Goal:** Replace the `requested=unknown actual=unknown` output with values derived from the event
stream, then document the semantics.

- [ ] **Step 1: Run the benchmark tests against the current summary output**

Run:

```bash
uv run pytest tests/test_benchmark_coreml.py tests/test_benchmark_coreml_observability.py -v
```

Expected: the benchmark tests still fail because the summary currently falls back to `unknown`
when metadata is missing.

- [ ] **Step 2: Update the benchmark summary formatting**

Change `benchmarks/benchmark_coreml.py` so it:

- reads `compute_units_requested` and `compute_units_actual` from the event stream
- prints the resolved values in the phase summary
- renders an explicit `null` for Core ML rows when `compute_units_requested` is present but
  `compute_units_actual` is absent, instead of using `unknown`
- preserves the existing parity and timing output
- omits the compute-unit suffix entirely for non-Core ML variants so pt/mps rows stay quiet

If multiple events report the same compute-unit value, keep collapsing to a single scalar in the
summary. If values differ, keep the existing list behavior.

Use the exact replacement pattern `value if value is not None else 'null'` instead of a truthiness
check so empty strings or zero-like values do not fall through to the fallback.

Use this conditional shape in the renderer:

```python
if req is None:
    suffix = ''
else:
    suffix = f'requested={req} actual={actual if actual is not None else "null"}'
```

- [ ] **Step 3: Update the benchmark tests for the new output**

Extend the benchmark tests so they assert:

- a Core ML run with metadata shows the requested and actual values in the summary
- a missing actual target is printed as `null`, not `unknown`
- non-Core ML variants omit the requested/actual suffix entirely

Keep the benchmark assertions synthetic as well. The tests should exercise
`aggregate_events()` / `benchmark_family()` with mocked event streams instead of requiring a real
model download or a macOS runner.

- [ ] **Step 4: Update the docs and roadmap wording**

Update the Core ML and performance docs so they explain:

- the difference between `compute_units_requested` and `compute_units_actual`
- what `null` means when the actual runtime target cannot be determined
- how the benchmark now reports these fields

Update `docs/roadmap.rst` so this metadata gap is marked as resolved, not still speculative. Add a
short changelog note for the user-visible measurement change.

- [ ] **Step 5: Verify the docs render**

Run:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Expected: the docs build succeeds and the updated wording renders cleanly.
