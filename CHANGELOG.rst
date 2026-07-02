=========
Changelog
=========

Unreleased
----------

* Added Core ML documentation for ``v3-coreml`` and ``v3-cls-coreml`` configs, export behavior,
  parity caveats, and Apple Silicon benchmark interpretation.
* Fixed Core ML and YOLO loading correctness issues: macOS-only ``.mlpackage`` loading is now
  guarded, YOLO models are loaded with ``task='detect'`` to avoid classifier misidentification,
  and empty/``None`` detection boxes return an empty prediction list instead of crashing.
* Fixed type-checking issues in model fetch/export code by normalizing model config values before
  path, URL, and progress-bar use.
* Applied Ruff formatting and lint fixes as part of the quality-tool migration.
* Removed the stale CodeQL workflow and documented the replacement quality gates:
  ``testing.yml`` for fast/offline pytest coverage and ``quality.yml`` for Ruff and scoped mypy.
* Raised declared Python support to ``>=3.10,<3.13``.
* Reorganized documentation around reader roles, with separate installation, usage,
  configuration, model, performance, development, security, roadmap, API, and CLI pages.

Version 0.2.0
-------------

**Breaking changes**

* Phase 1 and MVP models removed entirely. Only ``v3`` and ``v3-cls`` (YOLOv8) are supported.
* ``pipeline_v3()`` renamed to ``pipeline()``; ``batch_v3()`` renamed to ``batch()``.
* ``wic``, ``agg``, and ``tile`` modules removed.
* ``loc`` module now exposes only ``fetch()`` and ``CONFIGS`` — the ``pre()``,
  ``predict()``, and ``post()`` functions are gone.
* CLI options ``--wic-thresh``, ``--loc-nms-thresh``, ``--agg-thresh``, ``--agg-nms-thresh``
  removed. Only ``--loc-thresh`` remains.
* Config aliases ``old`` and ``new`` removed from ``--config`` choices.

**New features**

* ``batch_resumable()`` — incremental JSONL output with automatic resume on interruption.
* Model fetching now uses MD5 sidecar verification (``requests`` + ``hashlib``);
  ``pooch`` is no longer required.
* Models are cached at ``~/.cache/scoutbot/`` instead of inside the package directory.
* ``get_classes()`` returns class labels by model introspection.

**Dependency reductions**

* Removed: ``pooch``, ``wbia-utool``, ``onnxruntime``, ``torch`` (standalone),
  ``torchvision``, ``imgaug``, ``gradio``, ``dill``.
* Added: ``requests`` (replaces ``pooch`` for model downloads).
* ``ultralytics`` and ``sahi`` are now the only ML dependencies.

**Internal**

* ``tile_batched/main.py`` reduced from ~550 lines to ~35 lines; all SAHI reimplementation
  removed. Stock ``sahi.predict.get_sliced_prediction()`` is used directly.
* Docker image no longer requires Rust toolchain or ``onnxruntime-gpu``.

Version 0.1.17
--------------

* **[BREAKING CHANGE]** Added detection label mapping for the ``phase1`` output to rename the following species labels to be consistent with the ``mvp`` output labels:

    * ``elephant_savanna`` to ``elephant``
    * ``dead_animalwhite_bones`` to ``white_bones``
    * ``deadbones`` to ``white_bones``
    * ``elecarcass_old`` to ``white_bones``
    * ``gazelle_gr`` to ``gazelle_grants``
    * ``gazelle_th`` to ``gazelle_thomsons``

* Added rounding to the WIC predicted confidence to 4 decimal points in the print and JSON outputs.
* Added to the documentation the list of supported class labels for each model configuration.
* Updated the documentation to highlight the list of recommended supported species for the MVP model
  (based on held-out validation data), and to add the build and deployment instructions.
* Added platform detection code to detect macOS and reduce the batch size of WIC models with the
  MVP model to 1 (added to Known Issues).
* Added three new environment variables to allow specifying the model configuration for the ``WIC``,
  ``LOC``, and ``AGG``, respectively: ``WIC_CONFIG``, ``LOC_CONFIG``, ``AGG_CONFIG``.  If unset, it
  uses the global config and behavior as specified by the ``CONFIG`` environment variable.  The TILE
  module does not have different settings dependent on the model configuration.
* Added a new environment variable to allow for faster but less accurate results: ``FAST``.  If unset, it
  uses the standard tile extraction behavior for grid1 and grid2.  Turning this flag on will dramatically
  speed up inference by processing approximately half of the number of tiles per image.
* Added ``CHANGELOG.rst`` and ``ISSUES.rst``.
* Modified documentation strings in a few places for clarity and correctness.
* Updated both Gradio apps to support the Phase1 and MVP models and their respective configurations.
* Updated tests to support the new ML configuration default settings.

Version 0.1.16
--------------

*Alpha version of Scoutbot, with all Phase 1 and MVP functionality and pre-trained models included*
