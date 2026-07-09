Performance
===========

This page explains how Scoutbot measures performance -- Core ML backend timing and cross-image
concurrency on CUDA -- and how to read the current benchmark output.

Core ML benchmark
-----------------

``benchmarks/benchmark_coreml.py`` runs the full SAHI-tiled pipeline over the images in
``examples/`` for each variant. The benchmark now produces two views of the same run:

* a human-readable comparison table for quick inspection
* an optional machine-readable JSON summary via ``--summary-json``

The benchmark is still a full pipeline test, not an isolated model microbenchmark. It measures
the serial tile path that Scoutbot actually uses, so the numbers include Python, SAHI,
Ultralytics, Core ML dispatch, and post-processing overhead.

.. note::

   ``examples/`` images are stored via Git LFS. Before running the benchmark::

       brew install git-lfs   # once per machine
       git lfs install        # once per clone
       git lfs pull           # downloads the actual images (~23 MB)

   Or point ``--images`` at your own directory of aerial survey JPEGs.

::

   uv run python benchmarks/benchmark_coreml.py
   uv run python benchmarks/benchmark_coreml.py --family cls
   uv run python benchmarks/benchmark_coreml.py --summary-json benchmark-summary.json

What the benchmark records
--------------------------

The structured benchmark summary tracks the timings that matter for the M1 / M4 gap:

* image discovery and validation
* image decode / load time
* model fetch / export / load time
* slice generation
* per-tile model invocation count
* per-tile prediction time
* post-processing / NMS time
* result serialization
* Core ML compute-unit metadata where the runtime can provide it
* direct ``coremltools`` prediction timing when the Core ML path exposes it

The summary also records warm-up timing separately so the first image does not contaminate the
timed average.

How to read the output
----------------------

The human table is for people. It shows mean milliseconds per image, relative speedup, and parity
against the PyTorch CPU baseline.

The JSON summary is for automation. It is intended for downstream analysis, pipeline diagnostics,
and future agentic tuning. The machine-readable form is useful when Scoutbot is part of a broader
pipeline and the benchmark result needs to be consumed without scraping terminal output.

The Core ML run still reflects the real serial tile path, so it should not be read as a clean
hardware throughput benchmark. Earlier M1 / M4 wall-clock numbers were too close because they
included this dispatch overhead. The structured summary exists so future investigations can see
which phase is actually expensive instead of guessing from one aggregate number.

Current M1 / M4 comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

The July 2026 M1 and M4 JSON summaries both ran the same ``examples/`` corpus with six timed
images after one warm-up image (880 timed tile invocations per variant). The M4 is consistently
faster, but the full-pipeline Core ML gap is smaller than a raw hardware comparison would suggest
because serial tile orchestration and post-processing remain part of the measured wall time.

.. list-table::
   :header-rows: 1

   * - Family / variant
     - M1 ms/img
     - M4 ms/img
     - M1 / M4
   * - ``onecls`` / ``pt-cpu``
     - 33680.4
     - 24626.9
     - 1.37x
   * - ``onecls`` / ``pt-mps``
     - 7582.2
     - 3252.9
     - 2.33x
   * - ``onecls`` / ``coreml``
     - 3618.6
     - 2191.7
     - 1.65x
   * - ``cls`` / ``pt-cpu``
     - 18096.2
     - 13851.6
     - 1.31x
   * - ``cls`` / ``pt-mps``
     - 7580.9
     - 3173.2
     - 2.39x
   * - ``cls`` / ``coreml``
     - 3147.7
     - 2196.5
     - 1.43x

For Core ML specifically, the attributed ``perform_inference`` totals were 1.74x slower on M1 for
``onecls`` and 1.44x slower for ``cls``. The unattributed full-pipeline remainder was also slower
on M1 (1.51x and 1.43x respectively), which means the comparison is not only a Core ML runtime
question. Both machines reported ``compute_units_requested=all`` and ``compute_units_actual=null``,
so the benchmark still cannot prove which physical Apple compute unit executed inference without
external Activity Monitor / Instruments observation.

Compute-unit observability
--------------------------

The benchmark phase-total output now surfaces Core ML compute-unit metadata when the runtime
provides it. The suffix on each phase-total line takes one of three forms:

* **Non-Core ML variants** (``pt-cpu``, ``pt-mps``): no suffix. Compute-unit fields are not
  applicable and are omitted entirely so the line stays uncluttered::

      pt-cpu   tiles=128 invocations=2

* **Core ML variant with known actual**: both fields are printed::

      coreml   tiles=128 invocations=2 requested=cpu_and_ne actual=cpu_and_ne

* **Core ML variant where actual cannot be determined**: the requested value is shown and
  ``actual`` is rendered as ``null`` (not ``unknown``) to make the absence explicit::

      coreml   tiles=128 invocations=2 requested=cpu_and_ne actual=null

The ``requested`` value is the compute-unit hint passed to Core ML at model load time.
The ``actual`` value is read back from the runtime after inference; ``null`` means the
runtime did not expose this information for that invocation, not that inference failed.

When ``actual=null`` appears, confirm ANE engagement via Activity Monitor's *Neural Engine*
column (0% means the model fell back to CPU/GPU), or investigate whether the ``coremltools``
version exposes the compute-unit readback API for the installed runtime.

Concurrency benchmark (CUDA)
----------------------------

``benchmarks/benchmark_concurrency.py`` measures a different axis than the Core ML benchmark: how
throughput scales when whole images are processed concurrently on a single CUDA GPU. It runs the
full ``v3`` pipeline over a directory of images in three modes -- ``sequential``, ``threaded``,
and ``multiprocess`` -- and reports mean milliseconds per image, relative speedup, and per-phase
timing totals drawn from the same observability events the rest of Scoutbot emits.

::

   uv run python benchmarks/benchmark_concurrency.py
   uv run python benchmarks/benchmark_concurrency.py --modes sequential multiprocess --processes 8

The finding
~~~~~~~~~~~

Profiling on real survey imagery (8000x5320px, ``v3``) on an RTX 5070 showed each ``pipeline()``
call splits roughly **55% GPU inference / 45% CPU**, where the CPU portion is SAHI slice
generation and NMS merge -- pure single-threaded Python that runs *between* GPU calls. On the full
Matetsi survey the GPU sat at ~30% utilization while one CPU core pinned at ~90-100% and the other
23 cores stayed idle: the serial CPU pre/post-processing was starving the GPU, not the GPU itself
being the bottleneck.

Two ways to overlap that work were measured:

* **Threading** (shared model, one process) plateaued at roughly **-22%** and stopped improving
  past ~8 workers. Python's GIL means threads only overlap the fraction of work that releases it
  (GPU kernel launches), not the CPU-bound 45%.
* **Multiprocessing** (separate processes, each with its own model, no shared GIL) reached roughly
  **-55% (about 2x)** and plateaued around 8 processes. The model is small enough (~1-2 GB VRAM)
  that several loaded copies share one 12 GB GPU comfortably. On the full Matetsi survey this is
  about **9 hours instead of ~20**.

This is a CUDA / NVIDIA observation and is distinct from the Core ML M1 / M4 gap above: the
CPU-versus-accelerator balance, and therefore the payoff from concurrency, differs by backend.

``batch_resumable(processes=N)`` (see :doc:`usage` and :doc:`configuration`) turns this finding
into a first-class, resume-safe mode. Every ``processes>1`` run -- and, when requested via
``summary_json`` / ``--summary-json``, every sequential run too -- emits a machine-readable
summary in-band with fields drawn from the same observability events described below:
``per_worker_counts`` (how evenly work was balanced across workers), ``phase_totals_ns``
(including a derived ``sahi+post`` CPU bucket), and ``inference_duty_per_worker``. The latter is
an honest proxy for how much of a worker's wall time was spent in GPU inference calls -- it is
**not** a GPU-occupancy measurement, since it cannot see time spent on the GPU by other workers.

Two axes, one measured
~~~~~~~~~~~~~~~~~~~~~~~~

There are two independent ways to parallelize this workload:

* **Cross-image** -- process several images at once, each on its own model copy. This is what the
  benchmark measures and what delivered the ~2x above.
* **Within-image** -- run the tiles of a *single* image concurrently or as larger batched
  inferences. This remains deferred (see below) and is a separate, harder change.

The cross-image win does not depend on, and does not preclude, the within-image work.

Productionization has landed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark itself proves the speedup but deliberately **discards results**; it does not append
to the resumable JSONL file. The hard part of exposing this as a first-class mode was write
integrity: ``batch_resumable`` guarantees crash-safe resume by appending each result to a single
JSONL file, and several processes appending to that file directly would interleave and corrupt
lines (writes larger than ``PIPE_BUF`` are not atomic).

``batch_resumable(processes=N)`` (see :doc:`usage` and :doc:`configuration`) now solves this with
a single-writer design: worker processes each load their own model copy and process images
independently, but only the parent process writes to the JSONL file, appending one flushed result
at a time in completion order as workers finish. Workers are launched via ``forkserver``. A lost
or hung worker is treated as dead once it misses a bounded per-result deadline
(``SCOUTBOT_MP_RESULT_TIMEOUT``, default 600 seconds) and aborts the run fail-fast; everything
already flushed to disk stays valid, so re-running the same command resumes the rest. A
machine-readable run summary (``summary_json`` / ``--summary-json``) is available in both modes.
Remaining real-survey validation at scale and per-GPU default guidance for ``processes`` are
tracked in :doc:`roadmap`.

What remains deferred
---------------------

The current benchmarks do not try to measure every possible optimization. The following remain
follow-up work:

* within-image concurrent or batched tile inference experiments (the cross-image axis is now
  measured and productionized; see `Concurrency benchmark (CUDA)`_)
* direct hardware counters for ANE utilization

Observability Follow-up
-----------------------

The next investigation is tracked in :doc:`roadmap`. The benchmark now emits structured phase
timings and a machine-readable summary, so the follow-up can focus on explaining the remaining
overhead instead of inferring it from a single opaque wall-clock number.
