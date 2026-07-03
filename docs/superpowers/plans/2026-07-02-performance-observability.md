# Performance Observability and Benchmark Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add private runtime observability hooks and structured benchmark reporting so Scoutbot can explain where Core ML performance time is going on real machines and in pipeline integrations.

**Architecture:** Keep observability internal to the `scoutbot` package as a private helper module, with opt-in hooks on the existing public pipeline and model-fetch paths. Refactor the Core ML benchmark to consume structured JSON Lines events instead of timing the whole pipeline as one opaque block, then summarize those events into human-readable output. Finish by updating the performance/coreml/roadmap docs so the benchmark strategy, the runtime hook model, and the measurement caveats stay aligned.

**Tech Stack:** Python, pytest, `json`, `time`, `scoutbot` runtime APIs, `benchmarks/benchmark_coreml.py`, Sphinx docs.

---

## Global Constraints

- Use `uv run` for Python, pytest, Ruff, and mypy commands.
- Keep Ruff formatting on single quotes and 90-character line length.
- Keep mypy scoped to `scoutbot/` unless a task explicitly widens it.
- Do not add a blanket `pre-commit run --all-files` step; use task-specific verification instead.
- Keep observability hooks opt-in and avoid changing public inference output schemas.

---

## Task 1: Add private observability primitives and hook them into the runtime

**Files:**
- Create: `scoutbot/_observability.py`
- Modify: `scoutbot/__init__.py`
- Modify: `scoutbot/loc/__init__.py`
- Modify: `scoutbot/tile_batched/main.py`
- Test: `tests/test_observability.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_batch.py`
- Test: `tests/test_loc.py`

**Interfaces:**

- `scoutbot._observability.Observer` is a private helper protocol with `emit(event: dict) -> None`.
- Every emitted event dict must include `name`, `started_ns`, `ended_ns`, `duration_ns`,
  `config`, `backend`, and `device`.
- The optional fields are `image_path`, `tile_index`, `tile_count`, `model_invocations`,
  `compute_units_requested`, `compute_units_actual`, `cache_hit`, and `error_type`.
- JSON Lines output writes one event dict per line and uses the same keys as the in-memory event.
- `compute_units_requested` is the requested / configured target. `compute_units_actual` is the
  observed runtime target or fallback path.

**Goal:** Add opt-in timing/event hooks that can be enabled without changing inference outputs or default behavior.

- [ ] **Step 1: Write failing tests for disabled and enabled observability**

Create `tests/test_observability.py` with tests that describe the new contract:

```python
def test_observability_defaults_to_noop(tmp_path):
    from unittest.mock import patch

    from scoutbot import pipeline

    sample = tmp_path / 'image.jpg'
    sample.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 64)

    with patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), patch(
        'sahi.predict.get_sliced_prediction',
        return_value=type('R', (), {'object_prediction_list': []})(),
    ):
        result = pipeline(str(sample), config='v3', batched_detection_model=object())
    assert isinstance(result, tuple)


def test_observer_receives_pipeline_events(tmp_path):
    from unittest.mock import patch

    from scoutbot import pipeline

    events = []

    class Collector:
        def emit(self, event):
            events.append(event)

    image = tmp_path / 'image.jpg'
    image.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 64)

    with patch('sahi.predict.get_sliced_prediction', return_value=type('R', (), {
        'object_prediction_list': []
    })()):
        pipeline(
            str(image),
            config='v3',
            observer=Collector(),
            batched_detection_model=object(),
        )
    assert events
    assert any(event['name'] == 'pipeline.start' for event in events)
```

Add focused tests in `tests/test_pipeline.py`, `tests/test_batch.py`, and `tests/test_loc.py` for:

- `pipeline(..., observer=...)` calling the sink
- `batch(..., observer=...)` wrapping repeated `pipeline()` calls without flattening child events
- `batch_resumable(..., observer=...)` emitting resume/batch metadata
- `loc.fetch(..., observer=...)` emitting cache-hit / fetch events
- The observer hook should be minimally asserted in the first test: prove the event fires and the
  required keys above are present, not just the event name.

- [ ] **Step 2: Run the tests to confirm they fail for missing hooks**

Run:

```bash
uv run pytest tests/test_observability.py tests/test_pipeline.py tests/test_batch.py tests/test_loc.py -v
```

Expected: failures for missing `observer=` plumbing and missing observability module / event helpers.

- [ ] **Step 3: Implement the private helper and hook plumbing**

Add `scoutbot/_observability.py` with:

- a no-op sink
- a collecting sink
- a lightweight event record helper
- a context manager for timing named phases
- JSON Lines serialization helpers for structured output

Update `scoutbot/__init__.py` so `pipeline()`, `batch()`, and `batch_resumable()` accept an optional `observer=` keyword argument, pass it through the runtime, and emit named phase events for:

- image discovery
- image decode / load
- model fetch
- model load / export
- slice generation
- per-tile prediction
- post-processing / NMS
- output serialization

Update `scoutbot/loc/__init__.py` so `fetch()` and `export_coreml()` can record cache-hit, export, and compute-unit metadata. Update `scoutbot/tile_batched/main.py` only as needed to capture Core ML-specific runtime facts such as requested vs actual compute units.
For `tile_batched/main.py`, map the requested compute unit to `compute_units_requested` and the
runtime / fallback target to `compute_units_actual`; if the actual target cannot be resolved, emit
`null` for that field.

- [ ] **Step 4: Run the targeted tests again**

Run:

```bash
uv run pytest tests/test_observability.py tests/test_pipeline.py tests/test_batch.py tests/test_loc.py -v
```

Expected: pass, with no change to the existing inference outputs.

- [ ] **Step 5: Commit**

```bash
git add scoutbot/_observability.py scoutbot/__init__.py scoutbot/loc/__init__.py scoutbot/tile_batched/main.py tests/test_observability.py tests/test_pipeline.py tests/test_batch.py tests/test_loc.py
git commit -m "feat: add runtime observability hooks"
```

---

## Task 2: Refactor the Core ML benchmark to consume structured events

**Files:**
- Modify: `benchmarks/benchmark_coreml.py`
- Modify: `pyproject.toml`
- Test: `tests/test_benchmark_coreml.py`
- Test: `tests/test_benchmark_coreml_observability.py`

**Goal:** Replace the current single wall-clock benchmark with phase-level JSON Lines collection and summary output.

- [ ] **Step 1: Write failing tests for benchmark aggregation and hardening**

Extend `tests/test_benchmark_coreml.py` and add `tests/test_benchmark_coreml_observability.py` with tests that pin the new behavior:

```python
def test_parse_event_stream_aggregates_phase_timings():
    from benchmark_coreml import aggregate_events

    events = [
        {'name': 'slice_generation', 'duration_ns': 10},
        {'name': 'per_tile_prediction', 'duration_ns': 20},
    ]
    summary = aggregate_events(events, total_wall_ns=40)
    assert summary['phase_totals_ns']['slice_generation'] == 10
    assert summary['phase_totals_ns']['per_tile_prediction'] == 20
    assert summary['unattributed_ns'] == 10


def test_missing_coremltools_is_reported_once(capsys):
    from unittest.mock import patch

    import benchmark_coreml

    with patch.object(benchmark_coreml, '_run_variant', side_effect=RuntimeError('coremltools missing')):
        benchmark_coreml.main(['--family', 'onecls', '--images', 'examples'])

    captured = capsys.readouterr()
    assert captured.out.count('SKIPPED') == 1
```

Cover these cases explicitly:

- JSON Lines event parsing
- phase-time aggregation
- warm-up exclusion
- parity summary preservation
- one-line missing-`coremltools` skip behavior
- third-party noise cleanup staying out of the result table

- [ ] **Step 2: Run the tests to confirm the current benchmark shape fails them**

Before running the benchmark tests, add `pythonpath = ["benchmarks", "tests"]` under
`[tool.pytest.ini_options]` in `pyproject.toml` so the test modules can import the benchmark
script directly.

Run:

```bash
uv run pytest tests/test_benchmark_coreml.py tests/test_benchmark_coreml_observability.py -v
```

Expected: failures because the current benchmark still measures one opaque wall-clock number and does not consume JSONL events.

- [ ] **Step 3: Refactor `benchmarks/benchmark_coreml.py` around structured collection**

Update the benchmark so it:

- reads JSON Lines event streams from the observability layer
- records phase totals, compute-unit metadata, tile counts, and warm-up timing separately
- preserves the existing human-readable table for comparison output
- emits a machine-readable summary artifact in JSON or JSON Lines, clearly separated from the human table
- reports missing `coremltools` once and suppresses the existing warning / log noise described in `docs/roadmap.rst`
- exposes its argument parsing through a testable `main(argv=None)` so the tests can exercise the
  skip path without spawning a subprocess

Keep the benchmark usable from the existing command line entry point:

```bash
uv run python benchmarks/benchmark_coreml.py
uv run python benchmarks/benchmark_coreml.py --family cls
```

- [ ] **Step 4: Run the benchmark tests and a focused smoke check**

Run:

```bash
uv run pytest tests/test_benchmark_coreml.py tests/test_benchmark_coreml_observability.py -v
uv run python benchmarks/benchmark_coreml.py --family onecls --images examples
```

Expected: tests pass. The benchmark smoke run is local-only and should be treated as a best-effort
manual check: on a machine with Git LFS and real `examples/` data, it should produce a readable
phase summary without raw traceback noise; in a fresh clone without LFS, use a synthetic-image
temporary directory or skip the smoke run explicitly.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmark_coreml.py pyproject.toml tests/test_benchmark_coreml.py tests/test_benchmark_coreml_observability.py
git commit -m "feat: add structured Core ML benchmark observability"
```

---

## Task 3: Update the performance docs and roadmap to match the new observability model

**Files:**
- Modify: `docs/performance.rst`
- Modify: `docs/coreml.rst`
- Modify: `docs/roadmap.rst`
- Modify: `CHANGELOG.rst`

**Goal:** Make the docs explain the new measurement model, the new runtime hooks, and the revised roadmap status.

- [ ] **Step 1: Rewrite the performance page around phase timings and JSONL observability**

Update `docs/performance.rst` so it explains:

- why the old benchmark numbers were too close across M1 and M4
- what phase timings the new benchmark records
- what the JSON Lines stream contains
- how to interpret human-readable versus machine-readable benchmark output
- what remains intentionally unmeasured or deferred

Preserve the existing benchmark invocation examples, but update the interpretation text so it matches the new collector.

- [ ] **Step 2: Add Core ML runtime diagnostics and compute-unit wording**

Update `docs/coreml.rst` so it names the new runtime facts:

- requested vs actual compute units
- cache-hit / export state
- direct `coremltools` timing
- how the observability hooks explain the M1 vs M4 gap

Keep the existing Core ML export and parity guidance, but make sure the new hook model is linked from the page.

- [ ] **Step 3: Mark the roadmap item as implemented and link the spec**

Update `docs/roadmap.rst` so the `Performance observability spec` entry points at the new spec and reflects the implementation status. Also update the `Benchmark hardening` and `Core ML / performance investigations` wording so it no longer reads like the work is still hypothetical.

- [ ] **Step 4: Add a changelog note for the user-visible measurement changes**

Update `CHANGELOG.rst` with a short `Unreleased` bullet that says the benchmark now reports phase-level Core ML observability and the runtime gained opt-in structured hooks.

- [ ] **Step 5: Verify the docs build cleanly**

Run:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Expected: the docs build succeeds and the updated pages render without raw directive output.

- [ ] **Step 6: Commit**

```bash
git add docs/performance.rst docs/coreml.rst docs/roadmap.rst CHANGELOG.rst
git commit -m "docs: describe performance observability model"
```

---
