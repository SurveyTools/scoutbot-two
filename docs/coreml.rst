Core ML (Apple Neural Engine)
-----------------------------

ScoutBot can run the v3 / v3-cls localizers as Core ML ``.mlpackage`` bundles, so inference
is dispatched through Apple's Core ML runtime (and is eligible for the Neural Engine) on
macOS. The Core ML configs mirror their PyTorch counterparts and produce detections within
sub-pixel / FP16 tolerance of the ``.pt`` models — see `Parity and caveats`_.

This is a macOS feature. It requires ``coremltools``. See :doc:`installation` for Apple
Silicon setup and explicit export commands.

Configs
~~~~~~~

Two Core ML configs are registered in ``scoutbot/loc/__init__.py`` alongside ``v3`` / ``v3-cls``:

``v3-coreml`` — single-class detector
   * Bundle: ``yolov8-onecls.kaza.mlpackage``
   * Source: ``v3``

``v3-cls-coreml`` — multi-class species classifier
   * Bundle: ``yolov8-cls.kaza.mlpackage``
   * Source: ``v3-cls``

Each carries ``backend: 'coreml'`` and a ``source`` key naming the ``.pt`` config it is exported
from. Select one the same way as any other config::

    import scoutbot
    wic, detections = scoutbot.pipeline('image.jpg', config='v3-cls-coreml')

or via the CLI / env var (``CONFIG=v3-coreml``, ``--config v3-coreml``).

Exporting the bundles
~~~~~~~~~~~~~~~~~~~~~~

Unlike the ``.pt`` models, the Core ML bundles are **not hosted** — they are exported on demand
from their source ``.pt`` model and cached at ``~/.cache/scoutbot/``. The first time a Core ML
config is used, ``loc.fetch()`` fetches the source ``.pt`` (downloading it if needed) and
converts it; subsequent runs reuse the cached bundle. Nothing extra is required to use a Core ML
config — the export happens transparently.

To build (or rebuild) the bundles explicitly::

    uv run scoutbot export-coreml                       # export all coreml configs
    uv run scoutbot export-coreml --config v3-coreml    # one config
    uv run scoutbot export-coreml --force               # re-export even if present

Or from Python::

    from scoutbot import loc
    loc.export_coreml('v3-coreml', force=True)

The export is::

    YOLO(pt_path).export(format='coreml', half=True, nms=False, imgsz=slice_width)

* ``half=True`` — FP16 weights (a Neural Engine eligibility requirement).
* ``nms=False`` — no built-in NMS; SAHI performs class-agnostic NMS downstream, so baking it
  into the model would double up.
* ``imgsz`` is taken from the config's slice size (512), giving the model a **fixed** 512×512
  input. Static shapes are an ANE requirement, and matching the slice size avoids a resize on
  every tile.

Ultralytics writes ``<pt_stem>.mlpackage`` next to the source ``.pt``; since the ``.pt`` lives in
the cache, the bundle lands exactly where ``fetch()`` expects it.

How it runs in the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Core ML path reuses the existing SAHI tiling machinery. ``scoutbot.pipeline()`` calls SAHI's
``get_sliced_prediction``, which feeds **one 512×512 tile at a time** to
``tile_batched.Yolov8DetectionModel`` (a subclass of SAHI's ``UltralyticsDetectionModel``).
That subclass is Core ML-aware, with all special handling contained in it — no call-site changes:

1. **Load with** ``task='detect'``. ``YOLO('…-cls.kaza.mlpackage')`` with no task guesses
   ``classify`` from the filename and returns ``boxes=None``; both checkpoints are detectors.
2. **Skip** ``model.to(device)``. Ultralytics dispatches Core ML through its own runtime, so a
   torch device — in particular the ``cuda:0`` default — is meaningless here and predict is run
   with ``device='cpu'``.
3. **Pass the fixed** ``imgsz``. The model's input size is read from the ``.mlpackage`` at load
   time and passed to ``predict``; otherwise Ultralytics defaults to ``imgsz=640`` and Core ML
   rejects the mismatch (*"Image size 640 x 640 not in allowed set of image sizes"*).

Everything downstream (``result.boxes.data``, SAHI's class-agnostic NMS, the
``{l, c, x, y, w, h}`` pixel-dict output) is identical to the ``.pt`` path.

Benchmark
~~~~~~~~~

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

Core ML is fastest in wall-clock on both machines, ahead of MPS as well as CPU. These numbers
should not be read as a measure of raw M4 / ANE capacity. The current benchmark times the full
SAHI pipeline and the live path runs one 512×512 tile at a time through Ultralytics' Core ML
wrapper. That serial per-tile path adds Python, PIL, SAHI, Ultralytics, and Core ML dispatch
overhead to every tile, so the M1 Pro and M4 results are much closer than the hardware difference
would suggest. This benchmark proves the Core ML backend is useful in the existing pipeline; it
does not prove that the pipeline saturates Apple Silicon or that the M4 acceleration path is
well-utilized. A follow-up benchmark should break out slicing, per-tile model prediction,
post-processing, direct ``coremltools`` prediction, compute-unit selection, and concurrent or
batched tile inference.

Parity and caveats
~~~~~~~~~~~~~~~~~~~

* **Detection parity** vs the ``.pt`` models is sub-pixel for matched boxes, with confidence
  within the expected FP16 tolerance (< 0.01). The handful of unmatched detections are borderline
  boxes the rounding nudges across the confidence threshold — the benchmark gates matches by
  centre distance so these show honestly as unmatched counts rather than inflating the worst-case
  pixel delta.
* **ANE engagement is not guaranteed.** The speedup proves the model is Neural-Engine *eligible*
  (FP16, static shapes), not that the ANE is actually carrying the load. Confirm by watching
  Activity Monitor's *Neural Engine* column during a run — 0% means it fell back to GPU/CPU
  (re-check that FP16 and the static input size survived export).
* **Re-export after changing the slice size.** ``imgsz`` is baked into the bundle at export time;
  if a config's ``slice_width`` changes, rebuild with ``scoutbot export-coreml --force`` so the
  fixed input matches the tiles.
* **Torch / coremltools version pairing.** The export and inference paths have been
  validated with Torch 2.7.0 (the most recent version tested by coremltools at writing).
  Torch 2.11.0 produces a runtime warning but inference results are correct in practice.
  If you see unexpected export failures or accuracy regressions, check whether a newer
  ``coremltools`` release supports your installed Torch version before filing a bug.
