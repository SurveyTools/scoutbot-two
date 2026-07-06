Core ML (Apple Neural Engine)
-----------------------------

Scoutbot can run the v3 / v3-cls localizers as Core ML ``.mlpackage`` bundles, so inference
is dispatched through Apple's Core ML runtime (and is eligible for the Neural Engine) on
macOS. The Core ML configs mirror their PyTorch counterparts and produce detections within
sub-pixel / FP16 tolerance of the ``.pt`` models — see `Parity and caveats`_.

This is a macOS feature. It requires ``coremltools>=9.0``. See :doc:`installation` for Apple
Silicon setup and explicit export commands.

Configs
~~~~~~~

Two Core ML configs are registered in ``scoutbot.model_store`` alongside ``v3`` / ``v3-cls``:

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
config is used, ``scoutbot.model_store.fetch()`` fetches the source ``.pt`` (downloading it if
needed) and converts it; subsequent runs reuse the cached bundle. Nothing extra is required to use
a Core ML config — the export happens transparently.

To build (or rebuild) the bundles explicitly::

    uv run scoutbot export-coreml                       # export all coreml configs
    uv run scoutbot export-coreml --config v3-coreml    # one config
    uv run scoutbot export-coreml --force               # re-export even if present

Or from Python::

    from scoutbot import loc
    loc.export_coreml('v3-coreml', force=True)

The export is::

    YOLO(pt_path).export(format='coreml', quantize=True, nms=False, imgsz=slice_width)

* ``quantize=True`` — FP16 weights (a Neural Engine eligibility requirement). ``half=True``
  was the pre-8.3 spelling and is now deprecated.
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

Runtime observability
~~~~~~~~~~~~~~~~~~~~~

Scoutbot now exposes opt-in runtime hooks for the Core ML path. The public entry points accept an
optional ``observer=`` argument, and the private observability layer emits structured timing and
state events for model fetch, export, load, batch execution, and per-image pipeline runs.

Those events carry the cache-hit / export state that explains whether the bundle was reused or
rebuilt, and the benchmark summary surfaces Core ML compute-unit metadata directly from the event
stream. The current implementation is intentionally lightweight: no observer means no extra work.

Compute-unit metadata
^^^^^^^^^^^^^^^^^^^^^

Each inference event can carry two compute-unit fields:

``compute_units_requested``
   The compute-unit hint passed to the Core ML runtime at model load time (e.g.
   ``cpu_and_ne``, ``all``). This is set by the Scoutbot runtime and reflects what was asked
   for; it is always known for Core ML variants and ``None`` for plain PyTorch variants.

``compute_units_actual``
   The physical hardware that executed inference, as reported by the Core ML runtime.
   Currently always ``null``: coremltools does not expose a public API to query which
   hardware unit (ANE, GPU, or CPU) ran a given prediction — ``MLModel.compute_unit``
   returns the loading parameter, not a runtime measurement. The field and the call site
   are retained so a future coremltools release that adds a real runtime query can be
   wired in without touching callers. For now the benchmark renders it as ``actual=null``.

   To confirm ANE engagement in practice, watch Activity Monitor's Neural Engine column
   while the benchmark runs: non-zero Neural Engine % indicates ANE utilisation.

The distinction matters when diagnosing performance: ``requested=all actual=null`` is the
expected output on current Apple Silicon — it means Core ML selected the best available
hardware but the runtime did not report which unit won.

The benchmark aggregates these fields across all timed inference events. If every event
reports the same value the summary collapses to a single scalar; if values differ across
events the summary keeps the full list so mixed-unit runs are visible.

For non-Core ML variants (``pt-cpu``, ``pt-mps``) neither field is set, so the benchmark
omits the compute-unit suffix entirely — the phase-total line shows only tile and invocation
counts.

When investigating the M1 / M4 gap, use the structured benchmark summary in :doc:`performance`
to separate the serial SAHI tile overhead from the Core ML-specific work.

Performance and benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~

Core ML benchmark numbers, interpretation caveats, and planned diagnostics live in
:doc:`performance`. The short version: the benchmark now reports phase-level timings and an
optional JSON summary, which makes the M1 / M4 comparison more useful than the old single wall-
clock number. The benchmark still measures the full serial SAHI/Ultralytics/Core ML tile path
rather than raw Apple Silicon capacity.

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
