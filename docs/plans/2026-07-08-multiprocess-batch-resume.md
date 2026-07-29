# Multiprocess Batch Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, crash-safe multiprocess mode to `scoutbot.batch_resumable(processes=N)` that recovers the ~2x CUDA throughput measured in the concurrency benchmark, without weakening the single-file JSONL resume guarantee.

**Architecture:** A pure orchestrator (`run_batch`) owns the output JSONL — write ordering, provenance attach, fail-fast, and the enriched summary — and is driven by an injected `submit` callable, so it is fully unit-testable with no GPU. A `forkserver`-based multiprocess submitter provides the real `submit`: workers each load one model from a parent-resolved path and stream per-image results (with piggybacked observability events) back in completion order. Resume filtering and phase aggregation are shared helpers.

**Tech Stack:** Python 3.13, `multiprocessing` (forkserver), SAHI + Ultralytics YOLOv8, `uv`, pytest, Click, Sphinx.

**Reference spec:** `docs/specs/2026-07-08-multiprocess-batch-resume-design.md`

## Global Constraints

- **Runner:** Every Python/tooling command is prefixed with `uv run`. Never invoke bare `python`, `pytest`, `ruff`, or `mypy`.
- **Git:** Do NOT run `git add`, `git commit`, or any staging/commit command. Each task ends at a **review checkpoint** with a suggested commit message; the user commits. Ignore any `git` steps an execution sub-skill would otherwise run.
- **TDD:** Write the failing test first; fast suite is `uv run pytest -m "not slow"`.
- **Quality gates (run before each review checkpoint):** `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy scoutbot`.
- **Start method:** Obtain a local context with `multiprocessing.get_context('forkserver')` and build the pool from it. NEVER call `multiprocessing.set_start_method(...)`. `set_forkserver_preload([...])` is called in exactly one place, immediately before pool creation, with an explanatory comment; keep the preload list minimal. (Linux/CUDA-first; macOS/Windows are out of scope.)
- **Resume/output invariants:** The per-image JSONL result schema is unchanged (`filepath`, `wic`, `detections`, `timestamp`, `config`, optional `model`). Only completed, flushed results are ever written. The `processes=1` processing/write loop keeps today's behavior; existing `tests/test_batch.py` must stay green.
- **No new `loc.fetch` call sites:** Resolve models through the model-store surface — `scoutbot._resolve_model` for `model_name`, and `scoutbot.model_store.fetch` for the plain `config` case (this call is expected and correct). New code and its tests must not reference the `loc` compatibility wrapper.
- **Metric honesty:** the derived CPU bucket is `sahi+post`; the worker-inference proxy is `inference_duty_per_worker` (NOT "gpu duty cycle").

---

### Task 1: Forkserver worker-death characterization spike

Runs FIRST, before any `processes>1` production code exists. Establishes empirically whether `imap_unordered` raises when a worker dies hard (`os._exit(1)`). The result decides whether Task 6 needs the bounded-wait fallback. No model, no GPU — pure `multiprocessing` behavior.

> **This task is a spike, not a feature. A *failing* probe is a valid outcome.** Unlike every other task, an initial red here does not mean the task is incomplete. Two acceptable completions: (a) the probe PASSES as written → record **CLEAN**; (b) the probe FAILS with "PROBE HUNG" or "did not raise" → record **HANG/SILENT**. In case (b), do **not** change production or multiprocessing internals to force a CLEAN result — the observed behavior *is* the finding. Instead, **rewrite the test's assertions to characterize what was actually observed** (e.g. `assert not completed` for a hang, or `assert out == 'NO_RAISE'` for a silent drop) so the committed test is green and documents reality, record the outcome in the review checkpoint, and carry **HANG/SILENT** into Task 6's conditional (bounded-wait). The only true failures of this task are an unbounded hang (prevented by the subprocess timeout) or being unable to run the probe at all.

**Files:**
- Create: `tests/test_batch_mp_faults.py`

**Interfaces:**
- Produces: a recorded finding (in this task's review checkpoint) — either "imap_unordered raises cleanly on hard death" or "imap_unordered hangs/misbehaves → Task 6 must implement bounded-wait".

- [ ] **Step 1: Write the shared subprocess probe harness + characterization test**

Every forkserver probe runs in a **fresh subprocess** with a hard, code-enforced timeout. This bounds a possible hang without manual intervention (safe for CI/subagents) AND guarantees a clean forkserver singleton per probe (a forkserver started earlier in the pytest process cannot contaminate a later probe — see Task 6). `start_new_session=True` + `os.killpg` on timeout tears down the whole forkserver process tree, leaving no orphans.

```python
# tests/test_batch_mp_faults.py
"""Characterization spike (DL-5): does a forkserver Pool.imap_unordered raise when a
worker dies hard (os._exit) mid-map? The resume-safety design assumes it raises.
Each probe runs in a fresh subprocess with a hard timeout so a hang is bounded by
code, and so the forkserver singleton is never shared across probes."""

import os
import signal
import subprocess
import sys

import pytest

SENTINEL = '__DIE__'
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _maybe_die(item):
    # Module-level so forkserver workers import it by reference (tests.test_batch_mp_faults).
    if item == SENTINEL:
        os._exit(1)  # hard death: no exception, no cleanup
    return item


def _torch_in_worker(_):
    import sys
    return 'torch' in sys.modules


def _run_probe(code, timeout=30):
    """Run a forkserver probe in a fresh subprocess.

    Returns (completed, stdout). ``completed`` is False if the probe hung past
    ``timeout``; in that case the whole process group (forkserver + workers) is
    SIGKILLed so nothing is left running.
    """
    env = {
        **os.environ,
        'PYTHONPATH': _REPO_ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''),
    }
    proc = subprocess.Popen(
        [sys.executable, '-c', code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=_REPO_ROOT, env=env, start_new_session=True,
    )
    try:
        out, _err = proc.communicate(timeout=timeout)
        return True, out.strip()
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return False, ''


_DEATH_PROBE = """
import multiprocessing
from tests.test_batch_mp_faults import _maybe_die, SENTINEL
ctx = multiprocessing.get_context('forkserver')
items = ['a', 'b', SENTINEL, 'c', 'd', 'e', 'f', 'g']
try:
    with ctx.Pool(processes=2) as pool:
        for _ in pool.imap_unordered(_maybe_die, items, chunksize=1):
            pass
    print('NO_RAISE')
except Exception as exc:
    print('RAISED:' + type(exc).__name__)
"""


@pytest.mark.slow
def test_imap_unordered_behavior_on_hard_worker_death():
    completed, out = _run_probe(_DEATH_PROBE, timeout=30)
    assert completed, (
        'PROBE HUNG: imap_unordered did not surface a hard worker death within 30s '
        '-> Task 6 must use the bounded-wait fallback'
    )
    assert out.startswith('RAISED:'), (
        f'PROBE did not raise on hard death (stdout={out!r}) '
        '-> Task 6 must use the bounded-wait fallback'
    )
```

- [ ] **Step 2: Run the test and observe behavior**

Run: `uv run pytest tests/test_batch_mp_faults.py::test_imap_unordered_behavior_on_hard_worker_death -v -m slow`
Expected: PASS in the CLEAN case (stdout `RAISED:<ExceptionType>`). If it FAILS with "PROBE HUNG" or "did not raise", that is the finding that forces the bounded-wait path in Task 6. The test never hangs the suite — the subprocess timeout bounds it.

- [ ] **Step 3: Record the finding**

In the review checkpoint below, state exactly one of:
- "CLEAN: `imap_unordered` raised `<ExceptionType>` on hard death → Task 6 uses the plain `imap_unordered`; bounded-wait stays documented-but-unused." or
- "HANG/SILENT: probe hung or printed `NO_RAISE` → Task 6 wraps results in the bounded-wait fallback."

**Review checkpoint.** Do not commit. Suggested commit message:

```
Add forkserver worker-death characterization spike

Slow test establishing whether Pool.imap_unordered raises on a hard
worker exit (os._exit), gating the fail-fast vs bounded-wait choice
for the multiprocess batch mode.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 2: Promote phase aggregation to `_observability.py` (pipeline-only `sahi+post`)

Move the benchmark's phase aggregation into shared, tested code, corrected so the CPU bucket derives from the `pipeline` wrapper only (an outer `batch`/`batch_resumable` wrapper in the stream must not inflate it).

**Files:**
- Modify: `scoutbot/_observability.py`
- Modify: `benchmarks/benchmark_concurrency.py` (import the shared helper; delete the local copy)
- Test: `tests/test_observability.py`

**Interfaces:**
- Produces: `scoutbot._observability.aggregate_phase_ns(events: list[dict]) -> dict[str, int]` and module constant `OUTER_WRAPPER_NAMES = frozenset({'batch', 'batch_resumable'})`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_observability.py  (add to existing file)
from scoutbot._observability import aggregate_phase_ns


def test_aggregate_phase_ns_sums_children_and_derives_sahi_post():
    events = [
        {'name': 'perform_inference', 'duration_ns': 600},
        {'name': 'perform_inference', 'duration_ns': 400},
        {'name': 'pipeline', 'duration_ns': 2000},
    ]
    # sahi+post = pipeline(2000) - children(1000) = 1000
    assert aggregate_phase_ns(events) == {'perform_inference': 1000, 'sahi+post': 1000}


def test_aggregate_phase_ns_ignores_outer_batch_wrapper():
    events = [
        {'name': 'perform_inference', 'duration_ns': 1000},
        {'name': 'pipeline', 'duration_ns': 2000},
        {'name': 'batch_resumable', 'duration_ns': 999999},  # must NOT affect sahi+post
    ]
    assert aggregate_phase_ns(events) == {'perform_inference': 1000, 'sahi+post': 1000}


def test_aggregate_phase_ns_negative_guard():
    events = [
        {'name': 'perform_inference', 'duration_ns': 5000},
        {'name': 'pipeline', 'duration_ns': 2000},
    ]
    assert aggregate_phase_ns(events) == {'perform_inference': 5000, 'sahi+post': 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_observability.py -k aggregate_phase_ns -v`
Expected: FAIL with `ImportError: cannot import name 'aggregate_phase_ns'`

- [ ] **Step 3: Implement the shared helper**

Add to `scoutbot/_observability.py`:

```python
from collections import defaultdict

OUTER_WRAPPER_NAMES = frozenset({'batch', 'batch_resumable'})


def aggregate_phase_ns(events):
    """Sum event durations by phase and derive the CPU pre/post bucket.

    Named child phases (perform_inference, load_model, ...) are summed
    directly. Only the ``pipeline`` wrapper defines the per-image span, so
    ``sahi+post = Sum(pipeline) - Sum(child phases)`` captures the pure-Python
    CPU work (SAHI slicing, NMS merge, serialization). Outer wrappers
    (``batch``/``batch_resumable``) are excluded from both the span and the
    child totals so a production stream cannot double-count the outer batch.
    """
    totals = defaultdict(int)
    pipeline_ns = 0
    for event in events:
        name = event.get('name')
        duration_ns = event.get('duration_ns')
        if not isinstance(duration_ns, (int, float)):
            continue
        if name == 'pipeline':
            pipeline_ns += int(duration_ns)
        elif name in OUTER_WRAPPER_NAMES:
            continue
        else:
            totals[name] += int(duration_ns)
    if pipeline_ns:
        totals['sahi+post'] = max(0, pipeline_ns - sum(totals.values()))
    return dict(sorted(totals.items()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability.py -k aggregate_phase_ns -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Point the benchmark at the shared helper**

In `benchmarks/benchmark_concurrency.py`, delete the local `_aggregate_phase_ns` function and its `WRAPPER_EVENT_NAMES` constant, add `from scoutbot._observability import aggregate_phase_ns`, and replace the three call sites `_aggregate_phase_ns(...)` with `aggregate_phase_ns(...)`.

- [ ] **Step 6: Verify the benchmark still imports and lints**

Run: `uv run python -c "import ast, importlib.util; s=importlib.util.spec_from_file_location('bc','benchmarks/benchmark_concurrency.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('ok', hasattr(m,'aggregate_phase_ns'))"`
Expected: `ok True`
Run: `uv run ruff check benchmarks/benchmark_concurrency.py scoutbot/_observability.py`
Expected: `All checks passed!`

**Review checkpoint.** Do not commit. Suggested commit message:

```
Promote phase aggregation to _observability (pipeline-only sahi+post)

Share one perf definition between the concurrency benchmark and the
upcoming batch summary; derive sahi+post from the pipeline wrapper only
so an outer batch/batch_resumable event cannot inflate it.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 3: Batch summary builder in `_observability.py`

A pure function that turns collected events + counts into the enriched, machine-readable batch summary.

**Files:**
- Modify: `scoutbot/_observability.py`
- Test: `tests/test_observability.py`

**Interfaces:**
- Consumes: `aggregate_phase_ns` (Task 2).
- Produces:
  `build_batch_summary(events, *, wall_ns, processes, mode, start_method, processed, skipped, errors, total, default_pid, base) -> dict`
  where `base` is a dict of static fields merged in (e.g. `{'name': 'batch_resumable', 'config': ..., 'device': ..., 'output_path': ...}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_observability.py  (add)
from scoutbot._observability import build_batch_summary


def test_build_batch_summary_core_fields():
    events = [
        {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 11},
        {'name': 'perform_inference', 'duration_ns': 1_000_000_000, 'worker_pid': 11},
        {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 12},
        {'name': 'perform_inference', 'duration_ns': 1_000_000_000, 'worker_pid': 12},
    ]
    s = build_batch_summary(
        events,
        wall_ns=2_000_000_000,
        processes=2,
        mode='multiprocess',
        start_method='forkserver',
        processed=2,
        skipped=0,
        errors=0,
        total=2,
        default_pid=999,
        base={'name': 'batch_resumable', 'config': 'v3'},
    )
    assert s['mode'] == 'multiprocess'
    assert s['processes'] == 2
    assert s['start_method'] == 'forkserver'
    assert s['processed'] == 2 and s['total'] == 2
    assert s['per_worker_counts'] == {11: 1, 12: 1}
    assert s['phase_totals_ns'] == {'perform_inference': 2_000_000_000, 'sahi+post': 2_000_000_000}
    # inference_duty_per_worker = inference(2e9) / (wall 2e9 * procs 2) = 0.5
    assert s['inference_duty_per_worker'] == pytest.approx(0.5)
    assert s['config'] == 'v3'


def test_build_batch_summary_defaults_missing_worker_pid():
    events = [{'name': 'pipeline', 'duration_ns': 1_000_000_000}]  # no worker_pid (sequential)
    s = build_batch_summary(
        events, wall_ns=1_000_000_000, processes=1, mode='sequential',
        start_method=None, processed=1, skipped=0, errors=0, total=1,
        default_pid=555, base={'name': 'batch_resumable'},
    )
    assert s['per_worker_counts'] == {555: 1}
    assert s['start_method'] is None
```

(`pytest` is already imported in `tests/test_observability.py`; add `import pytest` if missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_observability.py -k build_batch_summary -v`
Expected: FAIL with `ImportError: cannot import name 'build_batch_summary'`

- [ ] **Step 3: Implement the builder**

Add to `scoutbot/_observability.py`:

```python
def _per_worker_counts(events, default_pid):
    counts = defaultdict(int)
    for event in events:
        if event.get('name') == 'pipeline':
            counts[event.get('worker_pid') or default_pid] += 1
    return dict(counts)


def build_batch_summary(
    events,
    *,
    wall_ns,
    processes,
    mode,
    start_method,
    processed,
    skipped,
    errors,
    total,
    default_pid,
    base,
):
    """Assemble the enriched, machine-readable batch summary dict."""
    inference_ns = sum(
        int(e['duration_ns'])
        for e in events
        if e.get('name') == 'perform_inference'
        and isinstance(e.get('duration_ns'), (int, float))
    )
    wall_s = wall_ns / 1e9 if wall_ns else 0.0
    return {
        **base,
        'status': 'ok',
        'mode': mode,
        'processes': processes,
        'start_method': start_method,
        'wall_ns': wall_ns,
        'images_processed': processed,
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'total': total,
        'throughput_img_per_s': (processed / wall_s) if wall_s else 0.0,
        'per_worker_counts': _per_worker_counts(events, default_pid),
        'phase_totals_ns': aggregate_phase_ns(events),
        'inference_duty_per_worker': (
            inference_ns / (wall_ns * processes) if wall_ns and processes else 0.0
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability.py -k build_batch_summary -v`
Expected: PASS (2 passed)

**Review checkpoint.** Do not commit. Suggested commit message:

```
Add build_batch_summary observability helper

Pure builder for the enriched batch summary: mode/processes/start_method,
throughput, per-worker counts, shared phase totals, and the honestly
named inference_duty_per_worker proxy.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 4: `_load_processed` + `BatchWork` in `scoutbot/_batch_mp.py`

Extract the resume-filter logic into a shared, tested helper and define the work descriptor. Rewire the sequential path to use the helper (behavior-preserving).

**Files:**
- Create: `scoutbot/_batch_mp.py`
- Modify: `scoutbot/__init__.py` (replace the inline `processed` loop with `_load_processed`)
- Test: `tests/test_batch_mp.py`

**Interfaces:**
- Produces:
  - `scoutbot._batch_mp.BatchWork` — `@dataclass(frozen=True)` with `remaining: list[str]`, `skipped_count: int`, `total_count: int`.
  - `scoutbot._batch_mp._load_processed(output_path) -> set[str]`.
  - `scoutbot._batch_mp.plan_work(filepaths, output_path) -> BatchWork` (convenience: load + filter + counts).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batch_mp.py
"""Tests for the multiprocess batch orchestration helpers."""

import json

from scoutbot._batch_mp import BatchWork, _load_processed, plan_work


def _write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def test_load_processed_returns_non_error_filepaths(tmp_path):
    out = tmp_path / 'out.jsonl'
    _write_jsonl(out, [
        {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []},
        {'filepath': 'b.jpg', 'error': 'boom'},   # error -> not processed
        {'filepath': 'c.jpg', 'wic': 0.2, 'detections': []},
    ])
    assert _load_processed(out) == {'a.jpg', 'c.jpg'}


def test_load_processed_ignores_blank_and_malformed_lines(tmp_path):
    out = tmp_path / 'out.jsonl'
    out.write_text('\n{not json}\n' + json.dumps({'filepath': 'a.jpg', 'wic': 0.1}) + '\n')
    assert _load_processed(out) == {'a.jpg'}


def test_load_processed_missing_file_is_empty(tmp_path):
    assert _load_processed(tmp_path / 'nope.jsonl') == set()


def test_plan_work_filters_and_counts(tmp_path):
    out = tmp_path / 'out.jsonl'
    _write_jsonl(out, [{'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}])
    work = plan_work(['a.jpg', 'b.jpg', 'c.jpg'], out)
    assert isinstance(work, BatchWork)
    assert work.remaining == ['b.jpg', 'c.jpg']
    assert work.skipped_count == 1
    assert work.total_count == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_batch_mp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoutbot._batch_mp'`

- [ ] **Step 3: Implement the module**

Create `scoutbot/_batch_mp.py`:

```python
"""Multiprocess batch orchestration for scoutbot.batch_resumable(processes=N).

Split into a GPU-less orchestrator/resume layer (this file's top half) and a
forkserver-based inference submitter (added in a later task). Kept private.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchWork:
    """One resume-filtering result, kept coherent as a single value."""

    remaining: list  # list[str], already resume-filtered
    skipped_count: int  # len(filepaths) - len(remaining)
    total_count: int  # len(filepaths)


def _load_processed(output_path):
    """Return the set of filepaths already successfully processed.

    Reads the resumable JSONL at ``output_path``; a record counts as processed
    only if it has a ``filepath`` and no ``error`` key. Blank and malformed
    lines are ignored. A missing file yields an empty set.
    """
    output_path = Path(output_path)
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'filepath' in record and 'error' not in record:
                processed.add(record['filepath'])
    return processed


def plan_work(filepaths, output_path):
    """Load prior progress and return the BatchWork for this run."""
    processed = _load_processed(output_path)
    remaining = [fp for fp in filepaths if fp not in processed]
    return BatchWork(
        remaining=remaining,
        skipped_count=len(filepaths) - len(remaining),
        total_count=len(filepaths),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batch_mp.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Rewire the sequential path to use `_load_processed`**

In `scoutbot/__init__.py`, add near the top-level imports: `from scoutbot._batch_mp import _load_processed`. In `batch_resumable`, replace the inline `processed = set()` / `if output_path.exists(): ...` block (the loop that reads the file) with:

```python
    processed = _load_processed(output_path)
```

Leave the rest of the sequential loop untouched.

- [ ] **Step 6: Verify sequential behavior is unchanged**

Run: `uv run pytest tests/test_batch.py -v`
Expected: PASS (all existing resumable tests, including `TestBatchResumableErrorResume`, still pass)
Run: `uv run pytest tests/test_batch_mp.py -v`
Expected: PASS

**Review checkpoint.** Do not commit. Suggested commit message:

```
Extract resume filtering into _batch_mp (_load_processed, BatchWork)

Shared, unit-tested resume-set loader and work descriptor; sequential
batch_resumable now uses it (behavior-preserving).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 5: Orchestrator `run_batch` in `scoutbot/_batch_mp.py`

The pure, injectable write-integrity core: consumes a `BatchWork` and an injected `submit`, writes completion-order JSONL with per-record flush, attaches provenance, fails fast on a lost worker, and emits the enriched summary.

**Files:**
- Modify: `scoutbot/_batch_mp.py`
- Test: `tests/test_batch_mp.py`

**Interfaces:**
- Consumes: `BatchWork` (Task 4), `build_batch_summary` (Task 3), `scoutbot._observability.emit`.
- Produces:
  `run_batch(work, output_path, submit, *, summary_meta, provenance=None, on_progress=None, observer=None) -> dict`
  where `submit: Callable[[list[str]], Iterator[tuple[str, dict, list[dict]]]]` yields `(filepath, record_or_error, events)` in completion order, and `summary_meta` is a dict with keys `mode`, `start_method`, `processes`, `default_pid`, and `base`.
  Return dict has at least `processed`, `skipped`, `errors`, `total`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batch_mp.py  (add)
import json as _json

from scoutbot._batch_mp import BatchWork, run_batch


def _meta(mode='multiprocess', processes=2):
    return {
        'mode': mode,
        'start_method': 'forkserver' if mode == 'multiprocess' else None,
        'processes': processes,
        'default_pid': 999,
        'base': {'name': 'batch_resumable', 'config': 'v3'},
    }


def _fake_submit(results):
    def submit(remaining):
        yield from results
    return submit


def test_run_batch_writes_completion_order_with_flush(tmp_path):
    out = tmp_path / 'out.jsonl'
    results = [
        ('b.jpg', {'filepath': 'b.jpg', 'wic': 0.2, 'detections': []}, [
            {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 7},
            {'name': 'perform_inference', 'duration_ns': 1_000_000_000, 'worker_pid': 7},
        ]),
        ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [
            {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 8},
            {'name': 'perform_inference', 'duration_ns': 1_000_000_000, 'worker_pid': 8},
        ]),
    ]
    work = BatchWork(remaining=['a.jpg', 'b.jpg'], skipped_count=0, total_count=2)
    summary = run_batch(work, out, _fake_submit(results), summary_meta=_meta())
    lines = [_json.loads(x) for x in out.read_text().splitlines()]
    assert [r['filepath'] for r in lines] == ['b.jpg', 'a.jpg']  # completion order
    assert summary['processed'] == 2 and summary['skipped'] == 0 and summary['total'] == 2
    assert summary['errors'] == 0


def test_run_batch_reflects_injected_counts(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg'], skipped_count=3, total_count=4)
    results = [('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])]
    summary = run_batch(work, out, _fake_submit(results), summary_meta=_meta())
    assert summary['skipped'] == 3 and summary['total'] == 4 and summary['processed'] == 1


def test_run_batch_attaches_provenance(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg'], skipped_count=0, total_count=1)
    results = [('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])]
    run_batch(work, out, _fake_submit(results), summary_meta=_meta(),
              provenance={'model_name': 'scoutbot-v3'})
    rec = _json.loads(out.read_text().splitlines()[0])
    assert rec['model'] == {'model_name': 'scoutbot-v3'}


def test_run_batch_counts_error_records(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg', 'b.jpg'], skipped_count=0, total_count=2)
    results = [
        ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, []),
        ('b.jpg', {'filepath': 'b.jpg', 'error': 'boom'}, []),
    ]
    summary = run_batch(work, out, _fake_submit(results), summary_meta=_meta())
    assert summary['processed'] == 1 and summary['errors'] == 1
    # error record is not given a provenance block even if provenance is set
    lines = [_json.loads(x) for x in out.read_text().splitlines()]
    assert any('error' in r for r in lines)


def test_run_batch_fail_fast_on_lost_worker(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg', 'b.jpg', 'c.jpg'], skipped_count=0, total_count=3)

    class WorkerDied(Exception):
        pass

    def submit(remaining):
        yield ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])
        raise WorkerDied('worker 2 exited')

    import pytest
    with pytest.raises(WorkerDied):
        run_batch(work, out, submit, summary_meta=_meta())
    # partial output intact and flushed: the one completed result survived
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert _json.loads(lines[0])['filepath'] == 'a.jpg'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_batch_mp.py -k run_batch -v`
Expected: FAIL with `ImportError: cannot import name 'run_batch'`

- [ ] **Step 3: Implement `run_batch`**

Add to `scoutbot/_batch_mp.py` (add imports `import logging`, `from time import perf_counter_ns`, `from scoutbot._observability import build_batch_summary, emit`; `log = logging.getLogger(__name__)`):

```python
def run_batch(
    work,
    output_path,
    submit,
    *,
    summary_meta,
    provenance=None,
    on_progress=None,
    observer=None,
):
    """Write-integrity orchestrator: one writer, completion order, fail-fast.

    ``submit(work.remaining)`` yields ``(filepath, record_or_error, events)`` in
    completion order. Success records (those without an ``error`` key) get the
    ``provenance`` block attached. Only completed, flushed results are ever
    written, so a mid-stream exception (a lost worker) leaves a valid,
    resumable file.
    """
    output_path = Path(output_path)
    started_ns = perf_counter_ns()
    processed_count = 0
    error_count = 0
    all_events = []

    try:
        with open(output_path, 'a') as f:
            for filepath, record, events in submit(work.remaining):
                all_events.extend(events)
                is_error = 'error' in record
                if not is_error and provenance is not None:
                    record['model'] = provenance
                f.write(json.dumps(record) + '\n')
                f.flush()
                if is_error:
                    error_count += 1
                    log.error(f'Failed to process {filepath}: {record.get("error")}')
                else:
                    processed_count += 1
                if on_progress is not None:
                    on_progress(filepath, record)
    except Exception:
        log.error(
            f'Batch aborted after a lost worker: {processed_count} done, '
            f'{len(work.remaining) - processed_count - error_count} remaining '
            f'- re-run with the same output to resume'
        )
        raise

    wall_ns = perf_counter_ns() - started_ns
    summary_event = build_batch_summary(
        all_events,
        wall_ns=wall_ns,
        processes=summary_meta['processes'],
        mode=summary_meta['mode'],
        start_method=summary_meta['start_method'],
        processed=processed_count,
        skipped=work.skipped_count,
        errors=error_count,
        total=work.total_count,
        default_pid=summary_meta['default_pid'],
        base=summary_meta['base'],
    )
    emit(observer, summary_event)
    return {
        'processed': processed_count,
        'skipped': work.skipped_count,
        'errors': error_count,
        'total': work.total_count,
        'summary': summary_event,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batch_mp.py -k run_batch -v`
Expected: PASS (5 passed)

**Review checkpoint.** Do not commit. Suggested commit message:

```
Add run_batch write-integrity orchestrator

Pure, injectable single-writer: completion-order JSONL with per-record
flush, provenance attach, fail-fast on a lost worker (partial output
stays resumable), and enriched summary emission.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 6: Forkserver submitter + worker in `scoutbot/_batch_mp.py`

The real `submit`: a `forkserver` pool (local context, preload) whose workers each load one model from the parent-resolved path and process images, returning records plus worker-tagged events. Applies the bounded-wait fallback only if Task 1 found `imap_unordered` unreliable.

**Files:**
- Modify: `scoutbot/_batch_mp.py`
- Test: `tests/test_batch_mp_faults.py` (extend Task 1's file), `tests/test_batch_mp.py`

**Interfaces:**
- Consumes: `scoutbot.pipeline`, `scoutbot.tile_batched.Yolov8DetectionModel`, `scoutbot._observability.list_observer`.
- Produces: `make_mp_submit(*, model_path, effective_run_config, config_name, backend_device, processes) -> submit` where `submit(remaining)` is a generator yielding `(filepath, record_or_error, events)` in completion order; and module-level `_mp_init(...)` / `_worker_process_image(filepath)`.

- [ ] **Step 1: Write the failing orchestrator-level fault test**

```python
# tests/test_batch_mp_faults.py  (add)
import pytest

from scoutbot._batch_mp import BatchWork, run_batch


def _die_submit(remaining):
    # First image completes, then simulate the pool surfacing a hard death.
    yield ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])
    raise RuntimeError('pool worker died')


@pytest.mark.slow
def test_run_batch_fail_fast_flushes_partial(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg', 'b.jpg'], skipped_count=0, total_count=2)
    meta = {'mode': 'multiprocess', 'start_method': 'forkserver', 'processes': 2,
            'default_pid': 1, 'base': {'name': 'batch_resumable'}}
    with pytest.raises(RuntimeError):
        run_batch(work, out, _die_submit, summary_meta=meta)
    assert out.read_text().strip() != ''  # completed result flushed before abort
```

- [ ] **Step 2: Run it to verify it passes against the existing orchestrator**

Run: `uv run pytest tests/test_batch_mp_faults.py::test_run_batch_fail_fast_flushes_partial -v`
Expected: PASS (confirms `run_batch` fail-fast is submitter-agnostic; the real-pool wiring is exercised by the slow smoke in Step 6)

- [ ] **Step 3: Implement the worker and pool init**

Add to `scoutbot/_batch_mp.py` (add `import os`, `import multiprocessing`, `from datetime import datetime, timezone`):

```python
_WORKER = {}
_WORKER_EVENTS = []  # per-worker; model perform_inference + pipeline events land here


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _mp_init(model_path, effective_run_config, config_name, backend_device):
    # Runs once per worker. The model's own observer is fixed at construction,
    # so wire it to this worker's event list; the per-call observer below reuses
    # the same list. model_path was resolved once by the parent (DL-2).
    from scoutbot import tile_batched
    from scoutbot._observability import list_observer

    observer = list_observer(_WORKER_EVENTS)
    model = tile_batched.Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=effective_run_config.get('confidence_threshold', 0.45),
        device=backend_device,
        observer=observer,
    )
    _WORKER.clear()
    _WORKER.update(
        model=model,
        run_config=effective_run_config,
        config_name=config_name,
        device=backend_device,
        observer=observer,
        pid=os.getpid(),
    )


def _worker_process_image(filepath):
    import scoutbot

    rc = _WORKER['run_config']
    _WORKER_EVENTS.clear()  # bound memory: only this image's events accumulate
    try:
        wic_, detects = scoutbot.pipeline(
            filepath,
            _WORKER['config_name'],
            batched_detection_model=_WORKER['model'],
            backend_device=_WORKER['device'],
            loc_thresh=rc.get('confidence_threshold', 0.45),
            slice_height=rc.get('slice_height', 512),
            slice_width=rc.get('slice_width', 512),
            overlap_height_ratio=rc.get('overlap_height_ratio', 0.25),
            overlap_width_ratio=rc.get('overlap_width_ratio', 0.25),
            perform_standard_pred=rc.get('perform_standard_pred', False),
            postprocess_class_agnostic=rc.get('postprocess_class_agnostic', True),
            observer=_WORKER['observer'],
        )
        record = {
            'filepath': filepath,
            'wic': wic_,
            'detections': detects,
            'timestamp': _utcnow_iso(),
            'config': _WORKER['config_name'],
        }
    except Exception as exc:  # per-image failure -> error record (retried on resume)
        record = {
            'filepath': filepath,
            'error': str(exc),
            'timestamp': _utcnow_iso(),
            'config': _WORKER['config_name'],
        }
    events = list(_WORKER_EVENTS)
    for event in events:
        event['worker_pid'] = _WORKER['pid']
    return filepath, record, events
```

- [ ] **Step 4: Implement `make_mp_submit`**

Add to `scoutbot/_batch_mp.py`:

```python
def make_mp_submit(
    *, model_path, effective_run_config, config_name, backend_device, processes
):
    """Return a completion-order ``submit`` backed by a forkserver pool.

    A local forkserver context is used (never set_start_method, which would
    mutate the caller's global default). The heavy third-party imports are
    preloaded once in the server so forked workers inherit them instead of
    re-importing them per worker.
    """

    def submit(remaining):
        ctx = multiprocessing.get_context('forkserver')
        # Preload ONLY the heavy third-party imports whose per-worker cost we are
        # eliminating (this is the one place we touch preload; it configures this
        # process's forkserver). Keep the list minimal: 'scoutbot' is deliberately
        # NOT preloaded - it is imported lazily in _mp_init/_worker and is cheap
        # once torch is already loaded. Importing torch here does not initialize
        # CUDA (that stays lazy, per worker, after fork).
        ctx.set_forkserver_preload(['torch', 'ultralytics'])
        with ctx.Pool(
            processes=processes,
            initializer=_mp_init,
            initargs=(model_path, effective_run_config, config_name, backend_device),
        ) as pool:
            yield from pool.imap_unordered(
                _worker_process_image, remaining, chunksize=1
            )

    return submit
```

> **Task 1 conditional (bounded-wait fallback).** If Task 1 recorded "HANG/SILENT", wrap the iteration so a per-result deadline raises. Replace the `yield from pool.imap_unordered(...)` line with:
> ```python
>             it = pool.imap_unordered(_worker_process_image, remaining, chunksize=1)
>             while True:
>                 try:
>                     yield it.next(timeout=RESULT_TIMEOUT_S)
>                 except StopIteration:
>                     break
>                 except multiprocessing.TimeoutError as exc:
>                     raise RuntimeError('worker exceeded result deadline (treated as dead)') from exc
> ```
> and add a module constant `RESULT_TIMEOUT_S = float(os.getenv('SCOUTBOT_MP_RESULT_TIMEOUT', '600'))`. If Task 1 recorded "CLEAN", keep the plain `yield from` above and do not add the constant.

- [ ] **Step 5: Verify the module imports and lints**

Run: `uv run python -c "import scoutbot._batch_mp as m; print('ok', hasattr(m,'make_mp_submit') and hasattr(m,'_worker_process_image'))"`
Expected: `ok True`
Run: `uv run ruff check scoutbot/_batch_mp.py`
Expected: `All checks passed!`
Run: `uv run mypy scoutbot`
Expected: mypy success

- [ ] **Step 6: Write the slow real-pool smoke test**

```python
# tests/test_batch_mp.py  (add)
import sys

import pytest

from scoutbot._batch_mp import make_mp_submit, plan_work, run_batch


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux', reason='forkserver path is Linux/CUDA-first')
def test_mp_submit_end_to_end_over_examples(sample_image, tmp_path):
    import scoutbot
    from scoutbot import model_store

    out = tmp_path / 'out.jsonl'
    model_path = model_store.fetch(config='v3')
    effective = {
        'confidence_threshold': 0.45, 'slice_height': 512, 'slice_width': 512,
        'overlap_height_ratio': 0.25, 'overlap_width_ratio': 0.25,
        'perform_standard_pred': False, 'postprocess_class_agnostic': True,
    }
    submit = make_mp_submit(
        model_path=model_path, effective_run_config=effective, config_name='v3',
        backend_device='cpu', processes=2,
    )
    work = plan_work([sample_image, sample_image], out)
    meta = {'mode': 'multiprocess', 'start_method': 'forkserver', 'processes': 2,
            'default_pid': 0, 'base': {'name': 'batch_resumable', 'config': 'v3'}}
    summary = run_batch(work, out, submit, summary_meta=meta)
    assert summary['processed'] >= 1
    assert out.read_text().strip() != ''
```

- [ ] **Step 7: Validate the forkserver preload (DL-3 claim)**

Reuse the isolated subprocess harness from Task 1 (`_run_probe` / `_torch_in_worker`). Running in a **fresh** subprocess is required: if any earlier test in the pytest process already started the forkserver singleton without preload, a later `set_forkserver_preload` would not restart it, giving a false result. A clean process guarantees the preload is honored.

```python
# tests/test_batch_mp_faults.py  (add; _run_probe and _torch_in_worker already exist)
_PRELOAD_PROBE = """
import multiprocessing
from tests.test_batch_mp_faults import _torch_in_worker
ctx = multiprocessing.get_context('forkserver')
ctx.set_forkserver_preload(['torch'])
with ctx.Pool(processes=1) as pool:
    print('TORCH_PRELOADED:' + str(pool.apply(_torch_in_worker, (None,))))
"""


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux', reason='forkserver path is Linux/CUDA-first')
def test_forkserver_preload_puts_torch_in_workers():
    completed, out = _run_probe(_PRELOAD_PROBE, timeout=60)
    assert completed, 'preload probe hung'
    # With preload, torch is inherited from the server -> already in sys.modules at
    # worker start, not re-imported per worker.
    assert out == 'TORCH_PRELOADED:True', out
```

Run: `uv run pytest tests/test_batch_mp_faults.py::test_forkserver_preload_puts_torch_in_workers -v -m slow`
Expected: PASS on a Linux box with `torch` installed (validates preload — not per-worker import — places `torch` in workers).

- [ ] **Step 8: Run the fast suite (slow tests deselected) to confirm no regressions**

Run: `uv run pytest tests/test_batch_mp.py tests/test_batch_mp_faults.py -m "not slow" -v`
Expected: PASS (fast orchestrator/helper tests green; slow real-pool tests deselected)

**Review checkpoint.** Do not commit. Suggested commit message:

```
Add forkserver submitter and worker for multiprocess batch

Local forkserver context with one-time torch/ultralytics preload;
per-worker model loaded from the parent-resolved path; completion-order
imap_unordered submit yielding records + worker-tagged events.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 7: Wire `processes` and `summary_json` into `batch_resumable`

Add the public kwargs and dispatch: `processes=1` keeps today's loop; `processes>1` resolves the model once (both `config` and `model_name` cases), builds the submitter, and runs the orchestrator. Optionally writes the summary JSON in both modes.

**Files:**
- Modify: `scoutbot/__init__.py`
- Test: `tests/test_batch.py` (new class `TestBatchResumableMultiprocess`)

**Interfaces:**
- Consumes: `plan_work`, `make_mp_submit`, `run_batch` (Tasks 4-6), `_resolve_model`, `model_store.fetch`.
- Produces: `batch_resumable(..., processes: int = 1, summary_json=None)` unchanged return dict.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batch.py  (add)
import json as _json
from unittest.mock import patch


class TestBatchResumableMultiprocess:
    def test_processes_gt_1_uses_run_batch(self, tmp_path):
        import scoutbot

        out = tmp_path / 'out.jsonl'

        def fake_submit(remaining):
            for fp in remaining:
                yield (fp, {'filepath': fp, 'wic': 0.5, 'detections': [], 'config': 'v3'}, [])

        with (
            patch('scoutbot.model_store.fetch', return_value='/fake/model.pt'),
            patch('scoutbot._batch_mp.make_mp_submit', return_value=fake_submit) as mk,
        ):
            summary = scoutbot.batch_resumable(
                ['a.jpg', 'b.jpg'], output_path=str(out), config='v3',
                processes=4, show_progress=False,
            )
        assert mk.called
        assert summary['processed'] == 2 and summary['total'] == 2
        lines = [_json.loads(x) for x in out.read_text().splitlines()]
        assert {r['filepath'] for r in lines} == {'a.jpg', 'b.jpg'}

    def test_processes_1_still_sequential(self, tmp_path):
        import scoutbot

        out = tmp_path / 'out.jsonl'
        with (
            patch('scoutbot.model_store.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.5, [])),
            patch('scoutbot._batch_mp.make_mp_submit') as mk,
        ):
            scoutbot.batch_resumable(
                ['a.jpg'], output_path=str(out), config='v3',
                processes=1, show_progress=False,
            )
        assert not mk.called  # sequential path must not build a pool

    def test_summary_json_written(self, tmp_path):
        import scoutbot

        out = tmp_path / 'out.jsonl'
        sj = tmp_path / 'summary.json'

        def fake_submit(remaining):
            for fp in remaining:
                yield (fp, {'filepath': fp, 'wic': 0.5, 'detections': [], 'config': 'v3'}, [])

        with (
            patch('scoutbot.model_store.fetch', return_value='/fake/model.pt'),
            patch('scoutbot._batch_mp.make_mp_submit', return_value=fake_submit),
        ):
            scoutbot.batch_resumable(
                ['a.jpg'], output_path=str(out), config='v3',
                processes=2, summary_json=str(sj), show_progress=False,
            )
        assert sj.exists()
        data = _json.loads(sj.read_text())
        assert data['mode'] == 'multiprocess' and data['processes'] == 2

    def test_summary_json_written_sequential(self, tmp_path):
        import scoutbot

        out = tmp_path / 'out.jsonl'
        sj = tmp_path / 'summary.json'
        with (
            patch('scoutbot.model_store.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.5, [])),
        ):
            scoutbot.batch_resumable(
                ['a.jpg'], output_path=str(out), config='v3',
                processes=1, summary_json=str(sj), show_progress=False,
            )
        data = _json.loads(sj.read_text())
        assert data['mode'] == 'sequential' and data['processes'] == 1

    def test_processes_less_than_1_rejected(self, tmp_path):
        import pytest

        import scoutbot

        out = tmp_path / 'out.jsonl'
        with pytest.raises(ValueError):
            scoutbot.batch_resumable(
                ['a.jpg'], output_path=str(out), config='v3',
                processes=0, show_progress=False,
            )

    def test_model_name_provenance_preserved_on_mp_path(self, tmp_path):
        import scoutbot

        out = tmp_path / 'out.jsonl'
        provenance = {'model_name': 'scoutbot-v3', 'md5': 'abc'}
        effective = {
            'confidence_threshold': 0.45, 'slice_height': 512, 'slice_width': 512,
            'overlap_height_ratio': 0.25, 'overlap_width_ratio': 0.25,
            'perform_standard_pred': False, 'postprocess_class_agnostic': True,
        }

        def fake_submit(remaining):
            for fp in remaining:
                yield (fp, {'filepath': fp, 'wic': 0.5, 'detections': [], 'config': 'v3'}, [])

        with (
            patch(
                'scoutbot._resolve_model',
                return_value=('/fake/model.pt', effective, provenance),
            ) as rm,
            patch('scoutbot._batch_mp.make_mp_submit', return_value=fake_submit),
        ):
            scoutbot.batch_resumable(
                ['a.jpg'], output_path=str(out), model_name='scoutbot-v3',
                processes=2, show_progress=False,
            )
        assert rm.called  # registry resolution used on the mp path
        rec = _json.loads(out.read_text().splitlines()[0])
        assert rec['model'] == provenance  # provenance attached to written record
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_batch.py -k Multiprocess -v`
Expected: FAIL with `TypeError: batch_resumable() got an unexpected keyword argument 'processes'`

- [ ] **Step 3: Add the `processes` / `summary_json` dispatch**

First, add `model_store` to the top-level import at `scoutbot/__init__.py:46` so new code resolves models without the `loc` compatibility wrapper:

```python
from scoutbot import loc, model_store, tile_batched  # NOQA
```

Then migrate `batch_resumable`'s existing sequential config-path fetch off `loc`: change the line `model_path = loc.fetch(config=config, **observer_kwargs)` (the `else` branch of the sequential model resolution) to:

```python
        model_path = model_store.fetch(config=config, **observer_kwargs)
```

This is behavior-identical (`loc.fetch` is a re-export of `model_store.fetch`) and keeps both the sequential and multiprocess paths on one non-`loc` surface. Do NOT touch `batch()` or `pipeline()` — they are out of scope.

Now add `processes=1` and `summary_json=None` to the `batch_resumable` signature. Immediately after the signature (before any work), validate the count:

```python
    if processes < 1:
        raise ValueError(f'processes must be >= 1, got {processes}')
```

Then, after `remaining`/`processed` are computed and the early-return-if-empty branch, add the multiprocess branch BEFORE the existing sequential model-build:

```python
    if processes > 1:
        from scoutbot import _batch_mp

        work = _batch_mp.BatchWork(
            remaining=remaining,
            skipped_count=len(filepaths) - len(remaining),
            total_count=len(filepaths),
        )
        if model_name is not None:
            model_path, effective_run_config, model_provenance = _resolve_model(
                config=config,
                model_name=model_name,
                confidence_threshold=loc_thresh if loc_thresh != 0.45 else None,
                slice_height=slice_height if slice_height != 512 else None,
                slice_width=slice_width if slice_width != 512 else None,
                overlap_height_ratio=overlap_height_ratio
                if overlap_height_ratio != 0.25 else None,
                overlap_width_ratio=overlap_width_ratio
                if overlap_width_ratio != 0.25 else None,
            )
        else:
            # forward the observer to fetch, matching the sequential path.
            # Use the model-store surface directly (not the loc compatibility
            # wrapper) so new code does not reinforce loc.
            model_path = model_store.fetch(config=config, **observer_kwargs)
            effective_run_config = {
                'confidence_threshold': loc_thresh,
                'slice_height': slice_height,
                'slice_width': slice_width,
                'overlap_height_ratio': overlap_height_ratio,
                'overlap_width_ratio': overlap_width_ratio,
                'perform_standard_pred': perform_standard_pred,
                'postprocess_class_agnostic': postprocess_class_agnostic,
            }
            model_provenance = None

        submit = _batch_mp.make_mp_submit(
            model_path=model_path,
            effective_run_config=effective_run_config,
            config_name=config,
            backend_device=backend_device,
            processes=processes,
        )
        progress_bar = tqdm(
            total=len(remaining), desc='Processing', unit='img',
            disable=not show_progress,
        )

        def _on_progress(filepath, record):
            progress_bar.update(1)

        try:
            result = _batch_mp.run_batch(
                work, output_path, submit,
                summary_meta={
                    'mode': 'multiprocess',
                    'start_method': 'forkserver',
                    'processes': processes,
                    'default_pid': os.getpid(),
                    'base': {
                        'name': 'batch_resumable', 'config': config,
                        'backend': 'coreml' if str(model_path).endswith('.mlpackage') else 'pt',
                        'device': backend_device, 'output_path': str(output_path),
                        'file_count': len(filepaths),
                    },
                },
                provenance=model_provenance,
                on_progress=_on_progress,
                observer=observer,
            )
        finally:
            progress_bar.close()  # closed even if run_batch fails fast on worker death
        if summary_json is not None:
            Path(summary_json).write_text(
                json.dumps(result['summary'], indent=2, sort_keys=True) + '\n'
            )
        return {k: result[k] for k in ('processed', 'skipped', 'errors', 'total')}
```

Then make the sequential path emit the **same** shared summary (the spec's "both modes via one builder", so `mode` is meaningful and the schema matches). Add `from scoutbot._observability import build_batch_summary` to the imports, and replace the existing sequential summary emit (the hand-built `emit(observer, {'name': 'batch_resumable', ...})` block after `result = {...}`) with:

```python
    ended_ns = perf_counter_ns()
    summary_event = build_batch_summary(
        [],  # v1: per-phase breakdown is not collected on the sequential path
        wall_ns=ended_ns - started_ns,
        processes=1,
        mode='sequential',
        start_method=None,
        processed=processed_count,
        skipped=len(processed),
        errors=error_count,
        total=len(filepaths),
        default_pid=os.getpid(),
        base={
            'name': 'batch_resumable',
            'config': config,
            'backend': 'coreml' if str(model_path).endswith('.mlpackage') else 'pt',
            'device': backend_device,
            'output_path': str(output_path),
            'file_count': len(filepaths),
        },
    )
    # Spec: the sequential summary keys per_worker_counts by the parent pid (the
    # single process that did all the work). build_batch_summary derives counts
    # from per-image events, which the sequential path does not collect, so set
    # it explicitly here.
    summary_event['per_worker_counts'] = {os.getpid(): processed_count}
    emit(observer, summary_event)
    if summary_json is not None:
        Path(summary_json).write_text(
            json.dumps(summary_event, indent=2, sort_keys=True) + '\n'
        )
    return result
```

(`build_batch_summary` with an empty event list yields a schema-consistent summary — correct counts/wall/throughput, and (with the override above) `per_worker_counts = {parent_pid: processed_count}` per the spec. `phase_totals_ns` stays empty on the sequential path (per-phase collection is a deferred nicety). The result is a strict superset of today's sequential event, so `tests/test_batch.py::TestBatchResumable::test_batch_resumable_forwards_observer_and_emits_event` stays green.)

Also update the **early-return branch** (`if not remaining:` — the "all files already processed" case, which precedes the `processes > 1` branch and covers both modes) to emit the same shared schema and honor `summary_json`. Replace its hand-built `emit(observer, {'name': 'batch_resumable', ...})` with:

```python
        summary_event = build_batch_summary(
            [],
            wall_ns=perf_counter_ns() - started_ns,
            processes=processes,
            mode='sequential' if processes == 1 else 'multiprocess',
            start_method=None,  # no pool was started; nothing ran
            processed=0,
            skipped=len(filepaths),
            errors=0,
            total=len(filepaths),
            default_pid=os.getpid(),
            base={
                'name': 'batch_resumable', 'config': config, 'backend': 'unknown',
                'device': backend_device, 'output_path': str(output_path),
                'file_count': len(filepaths),
            },
        )
        emit(observer, summary_event)
        if summary_json is not None:
            Path(summary_json).write_text(
                json.dumps(summary_event, indent=2, sort_keys=True) + '\n'
            )
        return result
```

Also update the `batch_resumable` docstring `Args:` to document `processes` (default 1; Linux/CUDA-first; each of N processes loads its own model copy — ensure they fit in VRAM) and `summary_json`.

- [ ] **Step 4: Migrate existing `batch_resumable` test patches off `loc`**

Because the sequential `batch_resumable` fetch now resolves via `model_store.fetch`, any existing **fast** test that patches `scoutbot.loc.fetch` to exercise `batch_resumable` will no longer intercept the fetch. In `tests/test_batch.py`, repoint those patches to `scoutbot.model_store.fetch` — this affects the tests that call `batch_resumable` (e.g. the `TestBatchResumableProgressBar` and `TestBatchResumableErrorResume` classes and the resumable observer-forwarding test). Do NOT change patches in tests that exercise `batch()` or `pipeline()` — those functions still use `loc.fetch` and are out of scope.

Mechanically: for each such test, change `patch('scoutbot.loc.fetch', ...)` to `patch('scoutbot.model_store.fetch', ...)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_batch.py -k Multiprocess -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Run the full fast batch suite**

Run: `uv run pytest tests/test_batch.py -m "not slow" -v`
Expected: PASS (all, including the pre-existing resumable tests you just repointed)
Run: `uv run pytest -m "not slow" -q`
Expected: PASS (whole fast suite — catches any other test that patched `loc.fetch` while exercising the sequential `batch_resumable` path)
Run: `uv run ruff check scoutbot/__init__.py`
Expected: `All checks passed!`
Run: `uv run mypy scoutbot`
Expected: clean

**Review checkpoint.** Do not commit. Suggested commit message:

```
Wire processes=N and summary_json into batch_resumable

processes=1 keeps the sequential loop; processes>1 resolves the model
once (config and model_name cases), builds the forkserver submitter, and
runs the write-integrity orchestrator. Optional machine-readable summary.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 8: CLI `--processes` and `--summary-json`

Expose the new options on the `batch-resumable` command.

**Files:**
- Modify: `scoutbot/scoutbot.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `scoutbot.batch_resumable(processes=, summary_json=)` (Task 7).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py  (add)
from unittest.mock import patch

from click.testing import CliRunner


def test_batch_resumable_cli_passes_processes_and_summary_json(tmp_path):
    from scoutbot.scoutbot import batch_resumable as cli_batch_resumable

    out = tmp_path / 'out.jsonl'
    sj = tmp_path / 'summary.json'
    with patch('scoutbot.batch_resumable', return_value={
        'processed': 1, 'skipped': 0, 'errors': 0, 'total': 1,
    }) as mock_br:
        result = CliRunner().invoke(cli_batch_resumable, [
            'a.jpg', '-o', str(out), '--processes', '4', '--summary-json', str(sj),
        ])
    assert result.exit_code == 0, result.output
    kwargs = mock_br.call_args.kwargs
    assert kwargs['processes'] == 4
    assert kwargs['summary_json'] == str(sj)


def test_batch_resumable_cli_rejects_zero_processes(tmp_path):
    from scoutbot.scoutbot import batch_resumable as cli_batch_resumable

    out = tmp_path / 'out.jsonl'
    with patch('scoutbot.batch_resumable') as mock_br:
        result = CliRunner().invoke(cli_batch_resumable, [
            'a.jpg', '-o', str(out), '--processes', '0',
        ])
    assert result.exit_code != 0  # IntRange(min=1) rejects 0
    assert not mock_br.called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "processes_and_summary_json or rejects_zero" -v`
Expected: FAIL (`no such option: --processes`)

- [ ] **Step 3: Add the CLI options**

In `scoutbot/scoutbot.py`, add to the `batch-resumable` command (after the `--no-progress` option):

```python
@click.option(
    '--processes',
    default=1,
    type=click.IntRange(min=1),  # reject 0/negatives at the CLI boundary
    help='Number of worker processes (Linux/CUDA). 1 = sequential. Each process '
    'loads its own model copy; ensure N copies fit in VRAM.',
)
@click.option(
    '--summary-json',
    default=None,
    type=str,
    help='Optional path for a machine-readable run summary.',
)
```

Add `processes` and `summary_json` to the `def batch_resumable(...)` parameter list, and pass them through in the `scoutbot.batch_resumable(...)` call:

```python
        processes=processes,
        summary_json=summary_json,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k "processes_and_summary_json or rejects_zero" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the CLI suite**

Run: `uv run pytest tests/test_cli.py -m "not slow" -v`
Expected: PASS
Run: `uv run ruff check scoutbot/scoutbot.py`
Expected: `All checks passed!`

**Review checkpoint.** Do not commit. Suggested commit message:

```
Add --processes and --summary-json to batch-resumable CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

### Task 9: Documentation pass

Update the reader-facing docs per the repository documentation-pass rule.

**Files:**
- Modify: `docs/usage.rst`, `docs/configuration.rst`, `docs/performance.rst`, `docs/roadmap.rst`, `CHANGELOG.rst`

**Interfaces:** none (docs only).

- [ ] **Step 1: `docs/usage.rst`** — In the `batch_resumable` / batch CLI section, document `processes` and `--processes`, `summary_json` / `--summary-json`, that output is written in **completion order** (resume keys on filepath; sort by filepath if input order is needed), and that a lost worker aborts fail-fast and is recovered by re-running the same command.

- [ ] **Step 2: `docs/configuration.rst`** — Document `--processes` / `processes=N`: Linux/CUDA-first scope, that each process loads its own model copy (N copies must fit in VRAM; `v3` is ~1–2 GB/copy, ~8 on a 12 GB GPU), and that the start method is `forkserver`.

- [ ] **Step 3: `docs/performance.rst`** — In the existing "Concurrency benchmark (CUDA)" section, cross-reference the new in-band summary: `inference_duty_per_worker` (honest proxy, not GPU occupancy), `per_worker_counts`, and `phase_totals_ns` including `sahi+post`, now emitted on every `processes>1` run and writable via `--summary-json`.

- [ ] **Step 4: `docs/roadmap.rst`** — In the "Cross-image concurrency (CUDA)" subsection, change status from "productionization planned" to reflect that `batch_resumable(processes=N)` has landed (fail-fast/resume-safe, forkserver), leaving best-effort recovery and within-image concurrency deferred.

- [ ] **Step 5: `CHANGELOG.rst`** — Add an entry: opt-in multiprocess batch mode `batch_resumable(processes=N)` / `--processes` with crash-safe resume, plus a machine-readable run summary (`summary_json` / `--summary-json`).

- [ ] **Step 6: Build the docs to verify no new warnings**

Run: `uv run sphinx-build -q -b html docs /tmp/scoutbot-docbuild`
Expected: exits 0. Read the command's own warning output directly (do not pipe it through a filter that could mask the exit status). The only acceptable warning is the pre-existing `roadmap.rst` "Model-store / ``loc`` staged cleanup" title-underline warning; any NEW warning referencing `usage.rst`, `configuration.rst`, `performance.rst`, `roadmap.rst`, or `CHANGELOG.rst` must be fixed before the checkpoint.

**Review checkpoint.** Do not commit. Suggested commit message:

```
Document multiprocess batch mode and run summary

usage/configuration/performance/roadmap/CHANGELOG: processes=N,
--processes, --summary-json, completion-order output, VRAM/platform
guidance, and the landed cross-image concurrency status.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

## Final verification (after all tasks)

- [ ] `uv run pytest -m "not slow"` — full fast suite green
- [ ] `uv run pytest -m slow tests/test_batch_mp_faults.py tests/test_batch_mp.py` — on a Linux box, the real-pool fault-injection and end-to-end smoke pass
- [ ] `uv run ruff format --check .` — clean
- [ ] `uv run ruff check .` — clean
- [ ] `uv run mypy scoutbot` — clean
- [ ] Confirm the Task 1 finding and the Task 6 conditional agree (bounded-wait present iff Task 1 said HANG/SILENT)
