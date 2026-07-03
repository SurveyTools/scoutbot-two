Performance
===========

This page covers benchmark interpretation, known bottlenecks, and planned observability work for
Scoutbot performance investigations.

Core ML benchmark
-----------------

``benchmarks/benchmark_coreml.py`` runs the full SAHI-tiled pipeline over the images in
``examples/`` for each variant and reports wall-clock per image plus detection parity against the
PyTorch-CPU reference.

.. note::

   ``examples/`` images are stored via Git LFS. Before running the benchmark::

       brew install git-lfs   # once per machine
       git lfs install        # once per clone
       git lfs pull           # downloads the actual images (~23 MB)

   Or point ``--images`` at your own directory of aerial survey JPEGs.

::

   uv run python benchmarks/benchmark_coreml.py
   uv run python benchmarks/benchmark_coreml.py --family cls

Representative results (7 images, warm-up excluded):

**Apple M1 Pro 10/16, 16 GB**

.. list-table::
   :header-rows: 1

   * - Family
     - Variant
     - ms / image
     - Speedup vs CPU
   * - onecls
     - pt-cpu
     - 29443
     - 1.00×
   * - onecls
     - pt-mps
     - 5049
     - 5.83×
   * - onecls
     - **coreml**
     - **2705**
     - **10.88×**
   * - cls
     - pt-cpu
     - 17112
     - 1.00×
   * - cls
     - pt-mps
     - 4834
     - 3.54×
   * - cls
     - **coreml**
     - **2696**
     - **6.35×**

**Apple M4, 48 GB**  *(Torch 2.11.0, ultralytics 8.2.103)*

.. list-table::
   :header-rows: 1

   * - Family
     - Variant
     - ms / image
     - Speedup vs CPU
   * - onecls
     - pt-cpu
     - 27178
     - 1.00×
   * - onecls
     - pt-mps
     - 3204
     - 8.48×
   * - onecls
     - **coreml**
     - **2114**
     - **12.86×**
   * - cls
     - pt-cpu
     - 14368
     - 1.00×
   * - cls
     - pt-mps
     - 3201
     - 4.49×
   * - cls
     - **coreml**
     - **2049**
     - **7.01×**

Interpretation
--------------

Core ML is fastest in wall-clock on both machines, ahead of MPS as well as CPU. These numbers
should not be read as a measure of raw M4 / ANE capacity.

The current benchmark times the full SAHI pipeline. The live Core ML path runs one 512×512 tile
at a time through Ultralytics' Core ML wrapper. That serial per-tile path adds Python, PIL, SAHI,
Ultralytics, and Core ML dispatch overhead to every tile, so the M1 Pro and M4 results are much
closer than the hardware difference would suggest.

The benchmark proves the Core ML backend is useful in the existing pipeline. It does not prove
that the pipeline saturates Apple Silicon or that the M4 acceleration path is well-utilized.

Diagnostics needed
------------------

A follow-up benchmark should break out:

* image discovery and validation time
* image decode/load time
* slice generation time
* per-tile model invocation count
* per-tile model wall time
* post-processing/NMS time
* JSON/result serialization time
* Core ML compute-unit selection
* direct ``coremltools`` prediction timing
* concurrent or batched tile inference experiments

Observability Follow-up
-----------------------

The next investigation should add structured benchmark phase timings before claiming
hardware-specific Core ML throughput. The planned observability spec is tracked in
:doc:`roadmap`.
