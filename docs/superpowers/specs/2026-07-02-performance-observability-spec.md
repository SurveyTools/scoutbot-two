# Performance Observability and Benchmark Hardening Spec

## Problem

Scoutbot's Core ML benchmark numbers are too close across hardware generations to support strong
throughput claims. The current benchmark measures the whole serial SAHI / Ultralytics / Core ML
path, but it does not separate where time is actually spent.

That creates two gaps:

1. The benchmark cannot explain why M1 and M4 results are close.
2. The runtime has no lightweight, structured observability hooks for future performance work.

This spec defines both the benchmark instrumentation and the runtime hooks needed to support a
clean follow-up investigation.

## Goals

1. Break benchmark time into named phases so the pipeline cost can be measured instead of guessed.
2. Add runtime observability hooks that are opt-in, low overhead, and safe to leave in production.
3. Make the benchmark report the expensive parts explicitly: image discovery, decode/load, slicing,
   per-tile invocation count, per-tile model wall time, post-processing, serialization, Core ML
   compute-unit selection, direct `coremltools` prediction timing, and batched or concurrent tile
   experiments.
4. Preserve normal Scoutbot behavior when observability is not enabled.
5. Allow both human-readable and machine-readable observability / benchmark outputs where that helps
   the performance investigation.
6. Keep the observability layer small enough to remain a tool, not a parallel logging system.

The machine-readable path serves two concrete use cases:

1. Scoutbot running as one step in a broader connected pipeline, where the operator may not be the
   direct human invoker.
2. Future agentic tuning and automated testing of the pipeline, where structured timing and
   diagnostic data are easier to consume than free-form text.

This spec treats the observability helpers as internal implementation detail, not a new public
package API. The runtime may expose optional hook parameters on existing public functions, but the
helper module itself should be private (`scoutbot._observability`).

## Non-goals

1. Do not add remote telemetry, network reporting, or persistent analytics storage.
2. Do not redesign the Core ML backend or tiling pipeline as part of this work.
3. Do not change public inference output schemas.
4. Do not make the benchmark depend on a special machine profile or external service.
5. Do not make signature additions to existing public APIs imply a compatibility guarantee for the
   observability hook types themselves; the promise applies to inference outputs, not hook internals.

## Priority order

1. Runtime and benchmark timing hooks
2. Benchmark output hardening and aggregation
3. Optional CLI / environment wiring for ad hoc profiling
4. Documentation updates in `docs/coreml.rst` and `docs/performance.rst`

## Proposed design

### 1. Observability primitives

Add a small internal module, `scoutbot._observability`, that exposes:

- a lightweight event / span record
- a no-op sink
- a collecting sink for benchmarks and local debugging
- a context-manager helper for timing named sections

The runtime should be able to emit structured events without importing the benchmark module. The
benchmark should subscribe to those events, not monkeypatch core code.

Suggested event shape:

- `name`: phase name
- `started_ns` / `ended_ns`
- `duration_ns`
- `config`: Scoutbot config name
- `backend`: `pt`, `mps`, or `coreml`
- `device`: requested backend device
- `image_path`: optional
- `tile_index` / `tile_count`: optional
- `model_invocations`: optional
- `compute_units_requested`: optional
- `compute_units_actual`: optional
- `cache_hit`: optional
- `error_type`: optional, only when a phase fails

The default sink should do nothing.

### 2. Runtime hooks

Add optional observability plumbing to the public entry points that already own the pipeline work:

- `scoutbot.pipeline()`
- `scoutbot.batch()`
- `scoutbot.batch_resumable()`
- `loc.fetch()`
- Core ML export or load paths if needed for the benchmark investigation

The hooks should emit phase boundaries rather than per-line logs. The key phases are:

1. image discovery
2. image decode / load
3. model fetch
4. model load / export
5. slice generation
6. per-tile prediction
7. post-processing / NMS
8. output serialization
9. Core ML compute-unit selection
10. direct `coremltools` prediction

The hook API should allow benchmark code to distinguish:

- one-time setup work
- per-image work
- per-tile work
- backend-specific work

Aggregation rule:

- `pipeline()` emits one event stream per invocation.
- `batch()` and `batch_resumable()` wrap repeated `pipeline()` calls and add batch-level start/end
  events plus a batch identifier, but they do not flatten child events into a single opaque record.
- The benchmark collector aggregates across those per-invocation streams when it needs summary
  output.

### 3. Benchmark collector

Refactor `benchmarks/benchmark_coreml.py` so it consumes observability events instead of timing the
entire `pipeline()` call as a single number.

The benchmark output should include:

- mean wall time per image
- warm-up excluded timing
- tile count per image
- per-phase timing totals
- per-tile invocation count
- parity summary against the PyTorch reference
- explicit notes when Core ML falls back or skips a phase

The benchmark should also make the noisy current behavior disappear or become explicit:

- missing `coremltools` should produce one clear skip message
- third-party warnings should be suppressed or redirected as documented
- the output table should not mix benchmark data and incidental stderr noise
- the hardening work should explicitly absorb the current benchmark noise cleanup already tracked
  in `docs/roadmap.rst`

### 4. Core ML-specific diagnostics

The Core ML path needs dedicated observability for the suspected bottlenecks:

- whether the model was exported or loaded from cache
- what input size was used
- which compute units were requested
- whether the run actually exercised the ANE or fell back to GPU / CPU
- how long direct `coremltools` prediction took apart from SAHI overhead
- `compute_units_requested` records the configured or requested compute-unit target. `compute_units_actual`
  records the observed runtime execution target or fallback path. Both fields should be emitted on
  Core ML runs; if the runtime cannot determine the actual target, it should be set to `null`.

This is the part that will explain why the M1 and M4 results are currently too close.

### 5. Optional runtime exposure

If the implementation needs user-facing hooks, prefer one of these forms:

1. an optional `observer=` / `sink=` keyword argument on public APIs
2. a context-manager wrapper for ad hoc profiling
3. an environment variable for local debugging only

The default should remain silent and cheap.

The machine-readable output format for the benchmark and structured hooks should be JSON Lines
(`.jsonl`), one event object per line, using the event shape above. Any final summary artifact may
be emitted as JSON as long as it is clearly separated from the event stream.

Sink thread-safety is out of scope for the first pass. The collecting sink only needs to be
correct for single-threaded emission, but the design should not block a later lock-protected
implementation if batch execution ever parallelizes.

## Error handling

1. Observability failures must not change inference results.
2. A broken sink must not crash the pipeline.
3. Benchmark collection failures should fail the benchmark run clearly, because those are
   diagnostic failures, not user inference failures.
4. A missing Core ML dependency should continue to be handled as a clean skip, not an exception
   dump inside the results table.

## Testing

The implementation should add tests for:

- no-op behavior when observability is disabled
- phase timing emission for a representative pipeline run
- benchmark aggregation over a synthetic event stream
- clean handling of missing `coremltools`
- correct handling of Core ML-specific metadata such as compute-unit selection

The tests should use mocks and synthetic inputs. They should not depend on real model downloads or
real Core ML hardware.

## Documentation impact

This spec implies updates to:

- `docs/performance.rst` for the benchmark strategy and observability plan
- `docs/coreml.rst` for Core ML-specific diagnostics and caveats
- `docs/roadmap.rst` to track the spec / implementation status
- `CHANGELOG.rst` once the hooks or benchmark changes land

## Acceptance criteria

1. The benchmark reports phase-level timings instead of a single opaque wall-clock number.
2. The summed phase timings account for at least 95% of the measured wall time on representative
   runs, leaving no more than 5% unattributed overhead.
3. The runtime exposes a lightweight observability hook that can be enabled without changing
   inference semantics.
4. The implementation remains silent and low overhead when hooks are not enabled.
5. The docs can point to a concrete observability model instead of a future placeholder.
