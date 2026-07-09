# Multiprocess Batch Resume — Write-Integrity Design Spec

## Problem

Profiling `scoutbot.pipeline()` on CUDA (RTX 5070, `v3`, real survey imagery) shows each call is
roughly 55% GPU inference and 45% single-threaded CPU work — SAHI slice generation and NMS merge,
pure Python that runs *between* GPU calls. On a full survey the GPU sits near 30% utilization while
one CPU core pins at ~90–100%: the serial CPU pre/post-processing starves the GPU. Threading is
GIL-limited (~22% gain); cross-image multiprocessing recovers ~2x (see
`benchmarks/benchmark_concurrency.py` and `docs/performance.rst`).

The measured win is real, but it lives only in a benchmark that **discards results**. Exposing it
as a first-class mode — e.g. `batch_resumable(processes=N)` — is blocked on one hard problem:
`batch_resumable` guarantees crash-safe resume by appending each result to a single JSONL file, and
several worker processes appending to one file interleave and corrupt lines (writes larger than
`PIPE_BUF` are not atomic). This spec defines how to run that workload across processes **without
losing the crash-safe resume guarantee**, and treats observability for future tuning as a
first-class part of the deliverable rather than an afterthought.

## Goals

1. Add opt-in cross-image multiprocessing to `batch_resumable` that preserves crash-safe resume.
2. Leave today's single-process **processing and JSONL-writing loop** (`processes=1`) unchanged, so
   all new resume/write risk is confined to the `processes>1` path. (The final summary *event* is
   enriched additively for both modes — see Observability — but no existing field is removed and no
   result-writing behavior changes.)
3. Guarantee no result line is ever interleaved, corrupted, double-written, or silently dropped.
4. Isolate the risky orchestration logic (resume, ordered write, fail-fast) from inference so it is
   fully unit-testable without a GPU or a real model.
5. Emit structured, machine-readable perf observability on every run so future tuning and
   perf-assessment decisions do not require re-running a dedicated benchmark.
6. Keep the observability layer consistent with the ad-hoc benchmark by sharing one phase-aggregation
   definition.

## Non-goals

1. No best-effort worker recovery / retry-on-death in v1 (deferred; see Deferred work).
2. No within-image (per-tile) concurrency — that is a separate, harder axis.
3. No `spawn`-based macOS/Windows support in v1 — Linux/CUDA first; other platforms are a later spec.
4. No change to the per-image detection output schema (records, including the model-provenance
   block, are identical to today's single-process output).
5. No remote telemetry, network reporting, or persistent analytics storage.
6. No direct GPU hardware counters or queue-depth timeseries.

## Governing decisions

These were settled during design and constrain everything below:

- **DL-1 — Parent owns the output JSONL; workers never write it.** Workers only produce records and
  hand them back; exactly one process (the parent) writes and flushes the file.
- **DL-2 — Parent resolves the model once, preserving `config` and `model_name`.** Before starting
  workers the parent resolves exactly once and passes the local `model_path` and an
  `effective_run_config` to the workers, keeping any `provenance` block in the parent. It uses the
  **existing** resolution path, and the two cases are resolved exactly as `batch_resumable` does them
  today — this spec must not invent a registry dependency for normal `--config` runs:
  - **`model_name` / `--model NAME` case:** `_resolve_model(config=, model_name=, …overrides)` returns
    `(model_path, effective_run_config, provenance)`; `effective_run_config` is the registry entry's
    default run config with the non-default call overrides applied, and `provenance` is the registry
    provenance dict.
  - **plain `config` case:** resolve `model_path` via the same `config` fetch path used today (no
    registry lookup); `effective_run_config` is assembled from the `batch_resumable()` call
    parameters themselves (`loc_thresh`, `slice_height`, `slice_width`, the overlap ratios,
    `perform_standard_pred`, `postprocess_class_agnostic`); and `provenance` is `None` (matching
    today's config path, which attaches no `model` block).

  Workers only *load* the model; they never resolve, fetch, or write the registry, so N workers
  cannot race the download or registry write. This spec introduces **no new `fetch` call sites** — it
  reuses the current resolution helpers, so `--model NAME`, effective run-config overrides, and the
  `model` provenance block all carry over unchanged (and it stays aligned with the model-store
  cleanup direction rather than reinforcing `loc.fetch`).
- **DL-3 — `forkserver` start method, with explicit preload.** Workers fork from a clean server
  process, which sidesteps the caller's CUDA state (the hazard that makes plain `fork` unsafe as a
  library). The "import once, not per worker" advantage over `spawn` is **not** automatic: plain
  `forkserver` does not inherit the parent's imports the way `fork` does, so the implementation
  **must** preload `['torch', 'ultralytics', …]` so those modules load in the server for workers to
  inherit. Without preload, each worker re-imports and the advantage over `spawn` disappears.
  Preloading `torch` is safe: importing it does not initialize CUDA (that stays lazy, per worker,
  after fork). The plan must **validate** the startup benefit (measure worker spin-up with vs.
  without preload) rather than assume it. Requires a module-level, picklable worker function and
  picklable initializer args. Linux-first.

  *Implementation constraint:* obtain a **local** context via
  `ctx = multiprocessing.get_context('forkserver')` and create the pool from it; do **not** call
  `multiprocessing.set_start_method(...)`, which mutates the process-global default and would surprise
  unrelated multiprocessing code in a caller's process. The preload is process-global (there is one
  forkserver per process), so `set_forkserver_preload([...])` must be called in exactly one
  documented, controlled place — immediately before the pool is created, with a comment stating that
  it configures this process's forkserver — and the preload list kept minimal.
- **DL-4 — Completion order, not input order.** Results are written as they complete. Resume keys on
  `filepath` (not line position), so completion order costs nothing for resume correctness and gives
  the best incremental crash safety with no buffering. Consumers needing input order sort by
  `filepath` afterward.
- **DL-5 — Fail-fast, resume-safe on hard worker death.** A worker that dies hard (CUDA context
  loss, OOM-kill, native segfault) is *expected* to surface as an exception from `imap_unordered`;
  the parent then flushes everything already written, reports `N done, M remaining — re-run to
  resume`, and exits non-zero. Recovery is a re-run, which the existing resume machinery already
  handles. **This is an assumption to be verified, not trusted:** `multiprocessing.Pool` can hang or
  enter a broken-pool state on abrupt worker exit instead of raising cleanly. The plan must prove the
  behavior with a real-pool fault-injection test (a worker calling `os._exit(1)` on a sentinel input)
  before the design is sound. If `imap_unordered` does not reliably raise, the fallback is a bounded
  per-result wait — a worker that misses the deadline is treated as dead and routed to the same
  fail-fast path. Either way, no partial or duplicate line is possible, because only completed,
  already-flushed results are ever written. **Sequencing:** the real-pool `os._exit(1)` spike is an
  *early* task in the plan — it runs before any production `processes>1` code path is exposed. If
  `imap_unordered` behaves cleanly, the bounded-wait fallback stays documented but minimal; if it
  hangs or behaves inconsistently, the implementation switches to the bounded-wait path first. The
  mechanism is chosen by that measurement, not assumed.
- **DL-6 — Separate orchestration from inference; unit-test the orchestration.** The parent-side
  logic is a pure unit driven by an injectable "submit" callable, tested with fake workers and a
  simulated worker death — no torch, no GPU, no pool.
- **DL-7 — Structured perf observability is core.** Every `processes>1` run emits an enriched batch
  summary with concurrency and phase metrics, and can write a machine-readable summary artifact.

## Architecture

Two separated units, plus a thin dispatch in the existing public function.

### Orchestrator (pure, injectable, GPU-lessly testable)

Owns the output JSONL: write ordering, provenance attach, fail-fast, and summary. It knows nothing
about torch, pools, or CUDA. Resume *filtering* is **not** its job — that is a separate pure helper
(see below); the orchestrator is handed the already-filtered `remaining` plus the counts it needs to
report. Approximate interface:

```
@dataclass(frozen=True)
class BatchWork:
    remaining: list[str]          # already resume-filtered by dispatch
    skipped_count: int            # len(filepaths) - len(remaining)
    total_count: int              # len(filepaths)

run_batch(
    work: BatchWork,
    output_path: Path,
    submit: Callable[[list[str]], Iterator[tuple[str, dict, list[dict]]]],
    *,
    provenance: dict | None = None,
    on_progress: Callable[[str, dict], None] | None = None,
    observer=None,
) -> dict            # {'processed', 'skipped', 'errors', 'total', ...summary fields}
```

`remaining`, `skipped_count`, and `total_count` are bundled in a small frozen `BatchWork` dataclass
rather than passed as loose kwargs, so they stay coherent (they are three views of one filtering step
and must not drift apart). `submit(work.remaining)` yields `(filepath, record_or_error, events)`
tuples in completion order. The
orchestrator, for each tuple: attaches the `provenance` block to a success record (as today), writes
one JSONL line, flushes, updates progress, tallies processed/errors, and folds the worker's `events`
into its running aggregation. On an exception raised out of the iterator (a lost worker, per DL-5) it
flushes, logs the resume-safe message, records the partial summary, and re-raises / returns a
resume-safe result. On normal completion it emits the enriched batch summary event (see
Observability) and returns the summary dict, deriving `skipped`/`total` from the counts it was given.

Resume filtering lives in a shared pure helper `_load_processed(output_path) -> set[str]` (the
current inline resume logic, extracted): read the existing JSONL, return the set of non-error
`filepath`s, ignoring blank/malformed lines. Dispatch calls it, builds the `BatchWork` (computing
`remaining` / `skipped_count` / `total_count`), and passes it in. This keeps resume-skipping testable
in `_load_processed` and write/order/fail-fast/summary testable in `run_batch`, with no overlap.

Because `submit` is injected, tests drive `run_batch` with a fake `submit` that yields canned
tuples — including one that raises mid-stream to exercise fail-fast — with no inference involved.

### Multiprocess submitter (the real `submit`)

Sets up a `forkserver` pool (preloaded per DL-3) sized to `processes`, with a pool initializer that
builds one `Yolov8DetectionModel` per worker from the parent-resolved `model_path` and stores it
process-global, along with the `effective_run_config`. Runs `imap_unordered(_worker_process_image,
remaining, chunksize=1)` and yields each result as it completes. `chunksize=1` keeps results
streaming and load balanced. A worker hard-death propagates as an exception out of `imap_unordered`
(subject to DL-5 verification), which is the fail-fast signal the orchestrator expects.

### Worker function

Module-level and picklable (required by `forkserver`):

```
_worker_process_image(filepath: str) -> tuple[str, dict, list[dict]]
```

It runs the existing `pipeline()` against the process-global model **using the parent-resolved
`effective_run_config`** (so registry run-config overrides apply on this path too), collects that
image's observability events into a local list (the "one pickle per image, not per event" pattern
validated in the benchmark), tags them with `worker_pid`, and returns `(filepath, record_or_error,
events)`. The record does **not** carry the `provenance` block — that is a run-level constant the
parent attaches at write time, so it never rides the per-image pickle. A catchable per-image
exception is turned into an `error` record here (mirroring today's behavior), not raised — only a
hard process death reaches the orchestrator as an exception.

### Dispatch

`batch_resumable(..., processes=1)` runs today's sequential loop — its resume set now loaded via the
shared `_load_processed` helper (a behavior-preserving extraction), the processing and JSONL-writing
loop otherwise unchanged. `processes>1`: loads the `processed` set via the same `_load_processed`
helper, builds the `BatchWork`, resolves the model once (DL-2) into `(model_path,
effective_run_config, provenance)`, constructs the multiprocess submitter, and calls `run_batch`.

## Data flow

1. Dispatch loads the `processed` set from any existing JSONL at `output_path` via the shared
   `_load_processed(output_path)` helper (the current resume logic, extracted), then computes
   `remaining`, `skipped_count`, and `total_count`.
2. Parent resolves the model **once** via the existing path — `_resolve_model(config=, model_name=,
   …)` for the registry/`--model` case, else the current `config` fetch path — yielding `(model_path,
   effective_run_config, provenance)`; resolution/download touches no CUDA in the parent.
3. Parent starts the preloaded `forkserver` pool; each worker loads its own model from `model_path`.
4. `imap_unordered` streams `(filepath, record_or_error, events)` tuples in completion order.
5. Orchestrator attaches `provenance` to each success record, writes it as one JSONL line + flush,
   updates the progress bar, emits a per-image event, and folds `events` into the aggregation.
6. On completion: emit the enriched batch summary event, optionally write the summary JSON, return
   `{processed, skipped, errors, total, ...}`.

## Error handling

- **Per-image catchable exception** (bad image, decode failure): the worker writes an `error` record
  and returns it; the parent writes it to JSONL. Because the resume `processed` set counts only
  non-error records, the image is retried on the next run — identical to current single-process
  semantics.
- **Hard worker death**: `imap_unordered` raises in the parent (or, per the DL-5 fallback, a bounded
  wait elapses). The orchestrator flushes all already-written results, logs `N done, M remaining —
  re-run to resume`, and surfaces a non-zero exit at the CLI. No partial or duplicate line is
  possible because only completed results were ever written, each followed by a flush.
- **v1 pool policy**: `maxtasksperchild=None` (no auto-recycle). Best-effort recovery is deferred.

## API and compatibility

- New keyword: `processes: int = 1` on `batch_resumable`. `processes=1` is the current behavior
  exactly; `processes>1` activates the multiprocess path.
- New keyword: `summary_json: str | Path | None = None` — when set, the run's machine-readable
  summary is written to this path for either mode (mirrors
  `benchmarks/benchmark_coreml.py --summary-json`).
- CLI: `--processes N` and `--summary-json PATH` on the batch-resumable command.
- Existing `config` / `model_name` selection (including `--model NAME`), effective run-config
  overrides, and the `model` provenance block are all preserved on the `processes>1` path (DL-2).
- Output JSONL result schema is unchanged.
- Docstring and `docs/` document: the Linux/CUDA-first scope, and VRAM guidance — `processes` model
  copies must fit in VRAM simultaneously (data point: `v3` is ~1–2 GB/copy, ~8 copies share a 12 GB
  GPU comfortably).

## Observability

### Worker → parent event transport

Worker events ride back on the result tuple as `events` (one pickle per image, no separate queue, no
per-tile IPC on the hot path). Events are tagged with `worker_pid` (available from the pool
initializer) so per-worker behavior is recoverable. If a worker dies mid-image, only that image's
events are lost, which is acceptable under fail-fast.

### Shared phase aggregation

Promote the benchmark's phase aggregation out of `benchmarks/benchmark_concurrency.py` and into
`scoutbot/_observability.py`, so the benchmark and the production batch summary compute the perf
story from one definition (the benchmark then imports it rather than defining its own).

The promoted helper must derive the CPU bucket from the **`pipeline` wrapper duration only**:

```
sahi+post = Σ(pipeline duration) − Σ(child-phase durations)
```

It must **not** treat all wrapper events as the span. In a production stream a `batch` /
`batch_resumable` wrapper event is also present, and counting it would double-count the outer batch
duration and badly inflate `sahi+post`. Only `pipeline` defines the per-image span; outer wrappers
(`batch`, `batch_resumable`) are excluded from both the span and the child-phase totals. (The
benchmark's current single-stream version happens to be correct only because it never emits outer
wrappers; the shared version must be robust to them, and a unit test asserts an injected
`batch_resumable` wrapper does not affect `sahi+post`.)

### Enriched batch summary event

A shared summary builder produces the enriched `batch_resumable` summary event, and **both** the
sequential loop and the orchestrator call it — so `mode` is meaningful and the schema is identical
across paths. In the sequential case `processes` is 1, `start_method` is `None`, and
`per_worker_counts` is keyed by the single parent pid. The builder adds, in addition to the existing
fields:

- `mode` — `'sequential'` or `'multiprocess'`
- `processes`, `start_method`
- `wall_ns`, `images_processed`, `skipped`, `errors`, `throughput_img_per_s`
- `per_worker_counts` — `{worker_pid: n}` (load balance)
- `phase_totals_ns` — the shared aggregation, including `sahi+post`
- `inference_duty_per_worker` — `Σ(perform_inference_ns) / (wall_ns × processes)`: the average
  fraction of wall time a worker spends inside inference. This is deliberately **not** called GPU
  occupancy — it does not measure the GPU, and because inference calls across workers overlap on the
  one GPU it *understates* the true GPU-busy fraction. It is a cheap, honestly-labeled proxy for "how
  inference-bound is a worker," which is enough to surface the CPU-starvation pattern the finding is
  about. A real GPU-occupancy number needs hardware counters (deferred).

### Machine-readable output

The enriched summary rides the existing `observer` channel (callers already pass an observer;
`_observability.write_jsonl` serializes events). The optional `summary_json` path additionally
serializes the run's summary as a standalone JSON artifact a downstream pipeline can consume without
scraping logs — consistent with the "audit trail travels with the data" principle.

## Testing

- **Resume-filter unit tests (`_load_processed`).** Reads an existing JSONL and returns the set of
  non-error `filepath`s; ignores blank and malformed lines; error records are *not* treated as
  processed (so they are retried). Resume-skipping is verified **here** — it happens in the helper
  before `run_batch`, not inside it.
- **Orchestrator unit tests (no GPU, fast suite).** Drive `run_batch` with a fake `submit` plus an
  injected `BatchWork` / `provenance` / events:
  - happy path writes N lines, in completion order, one flush each;
  - the `BatchWork` `skipped_count` / `total_count` are reflected in the returned summary;
  - the `provenance` block, when given, is attached to each written success record;
  - an `error` result is written and counted as an error, not a success;
  - a `submit` that raises mid-stream triggers fail-fast: partial output is intact and flushed, the
    summary reports processed/remaining correctly, and the resume-safe message is logged;
  - the enriched summary fields (including `sahi+post` and `inference_duty_per_worker`) are computed
    correctly from injected fake events.
- **Shared aggregation unit tests.** Cover the promoted aggregator directly: child-phase sums; the
  `sahi+post` derivation from `pipeline` duration only; that an injected outer `batch_resumable`
  wrapper does not affect `sahi+post`; the `max(0, …)` negative guard.
- **Real-pool fault-injection test (required; may be slow).** A real `forkserver` pool where one
  worker calls `os._exit(1)` on a sentinel input, asserting `imap_unordered` surfaces the death as an
  exception and the orchestrator's fail-fast path runs (partial output flushed, non-zero result).
  This validates the DL-5 assumption that resume safety depends on; if it fails, the bounded-wait
  fallback is implemented and this test targets that behavior instead.
- **forkserver preload validation.** Confirm the heavy imports load once in the server, not per
  worker (e.g. a worker's startup does not re-import `torch`), validating the DL-3 claim.
- **Slow smoke (`@pytest.mark.slow`).** A real `forkserver` pool over `examples/` proves the wiring
  end to end (CPU acceptable where CUDA is unavailable; the goal is wiring, not throughput).

## Deferred work

- Best-effort worker recovery / retry-on-death (respawn, re-queue, poison-image detection).
- Within-image (per-tile) concurrency or batched tile inference.
- `spawn`-based macOS/Windows support (separate spec), and re-measuring the benchmark's claimed
  `spawn` steady-state penalty (noted, not blocking the `forkserver` choice).
- Explicit SAHI/NMS phase hooks inside the pipeline (would replace the derived `sahi+post` bucket
  with directly measured phases); tracked with the broader instrumentation work.
- Automatic sequential-baseline speedup in the summary (needs a control run).

## Documentation impact

Per the repository documentation-pass rule, the implementation must update:

- `docs/usage.rst` — `batch_resumable` / batch CLI: the `processes` and `summary_json` options,
  completion-order output, and resume-after-failure behavior.
- `docs/configuration.rst` — `--processes` / VRAM guidance and the Linux/CUDA-first scope.
- `docs/performance.rst` — cross-reference the in-band `inference_duty_per_worker` and summary schema
  from the concurrency section already added.
- `docs/roadmap.rst` — move the cross-image concurrency item from "productionization planned" toward
  done as the work lands.
- `CHANGELOG.rst` — new opt-in multiprocess batch mode and machine-readable run summary.

## Cross-references

- `benchmarks/benchmark_concurrency.py` — the investigation artifact this productionizes.
- `docs/performance.rst` — "Concurrency benchmark (CUDA)" finding.
- `docs/superpowers/specs/2026-07-02-performance-observability-spec.md` — the observability hooks and
  private `_observability` module this builds on.
- `scoutbot/__init__.py` — current `batch_resumable` and resume logic.
