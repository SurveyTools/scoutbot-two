Performance
===========

This page explains how Scoutbot measures Core ML performance and how to read the current
benchmark output.

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

What remains deferred
---------------------

The current benchmark does not try to measure every possible optimization. The following remain
follow-up work:

* concurrent or batched tile inference experiments
* direct hardware counters for ANE utilization
* more detailed attribution when a run falls back from the requested compute units

Observability Follow-up
-----------------------

The next investigation is tracked in :doc:`roadmap`. The benchmark now emits structured phase
timings and a machine-readable summary, so the follow-up can focus on explaining the remaining
overhead instead of inferring it from a single opaque wall-clock number.
