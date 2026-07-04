=========
Changelog
=========

Unreleased
----------

* Added a local model registry backed by ``~/.rebel-workflow/model-registry.json``.
  ``scoutbot fetch`` auto-registers downloaded models; subsequent fetches resolve
  from the registry and skip the network. New CLI commands: ``scoutbot register``
  registers any local ``.pt`` file; ``scoutbot models`` lists all registry entries;
  ``scoutbot models info NAME`` prints a model's full entry as JSON. All inference
  commands (``pipeline``, ``batch``, ``batch-resumable``) now accept a ``--model NAME``
  flag as an alternative to ``--config``; the two flags are mutually exclusive.
  JSONL output from ``batch-resumable`` gains a ``model`` block recording the
  registry name, MD5, framework, and SAHI run config used for each image.
* Added CUDA support clarification for NVIDIA systems: best-effort Windows support (not CI-tested),
  installation guide with GPU detection verification command (``torch.cuda.is_available()``), and
  development guidance for contributors testing on GPU hardware. CUDA is not required and the
  package runs normally on CPU-only machines.
* Added a ``windows-smoke`` CI job to ``.github/workflows/testing.yml`` that runs the fast/offline
  test suite on Python 3.13 under Windows. This provides basic cross-platform coverage for non-Core ML
  code without claiming full Windows parity; Core ML configs remain macOS-only.
* Moved ``visualise.py`` from the repo root to ``tools/visualise.py`` and included it
  in all quality gates (ruff, mypy, bandit). Rewrote the script: replaced
  ``cv2.imshow`` (incompatible with the headless OpenCV build) with ``webbrowser``
  for display; added ``--no-display`` and ``--image-key`` flags; fixed label rendering
  near the top edge of the image; removed the ``numpy`` dependency; added per-class
  box colours; added a test suite in ``tests/test_visualise.py``.
* Added Bandit source-code security scanning to the quality workflow (``bandit -c pyproject.toml
  -r scoutbot/ benchmarks/ tools/``). B101 (assert-used) is skipped; ``tests/`` is excluded. Fixed
  ``hashlib.md5()`` → ``hashlib.md5(usedforsecurity=False)`` to reflect its use as a file
  integrity check rather than a cryptographic hash.
* Added ``uv audit --locked`` to the quality workflow as a required dependency-scanning gate.
  Upgraded ``idna`` 3.13 → 3.18 to resolve GHSA-65pc-fj4g-8rjx / PYSEC-2026-215. Added
  ``ignore-until-fixed`` in ``[tool.uv.audit]`` for GHSA-rrmf-rvhw-rf47 (``torch`` CVE,
  no upstream fix available).
* The Core ML benchmark phase-total output now shows ``requested=<val> actual=<val>`` for
  Core ML variants (or ``actual=null`` when the runtime does not expose the actual compute
  unit). Non-Core ML variants (``pt-cpu``, ``pt-mps``) omit the suffix entirely instead of
  printing ``requested=unknown actual=unknown``.
* Added private runtime observability hooks and a structured Core ML benchmark summary so
  performance investigations can separate phase timing from serial SAHI overhead.
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
* Published rendered docs to GitHub Pages and retired Read the Docs so the README routes
  readers to HTML instead of raw ``.rst`` sources.
* Raised declared Python support to ``>=3.10,<3.14`` and added Python 3.13 to the supported
  target matrix.
* Reorganized documentation around reader roles, with separate installation, usage,
  configuration, model, performance, development, security, roadmap, API, and CLI pages.
* Raised fast-suite test coverage from 58% to 92%+ by adding unit tests with mocked
  SAHI, requests, and coremltools dependencies.

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
