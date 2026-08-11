=========
Changelog
=========

0.5.0 (unreleased)
------------------

* ``scoutbot.example()`` now downloads its demo image from the OSW public bucket
  (``osw-public-data/scoutbot/examples/scout.example.jpg``) instead of the retired Wild Me Azure
  CDN. The image bytes and pinned SHA-256 are unchanged.
* ``scoutbot.example()`` now prints the WIC score and detection count (not just the raw detection
  list), returns ``(wic_score, detections)`` instead of ``None``, and points to the docs for batch /
  multi-class / Core ML usage. It remains a single-image install smoke check, not an API tour.
* ``scoutbot.example()`` now caches its demo image under the OSW home
  (``${OSW_HOME:-~/.osw}/examples`` via a new ``get_examples_dir()`` resolver) instead of recreating
  the legacy ``~/.cache/scoutbot`` directory. This completes the move off ``~/.cache/scoutbot``
  begun in 0.5.0 for model files.
* Corrected the Windows NVIDIA GPU install docs: the pinned ``torch`` (2.13.0) is published under the
  ``cu130`` PyTorch index, not the stale ``cu124`` the docs previously showed (nor ``cu128``). Added
  guidance to match the CUDA index to the pinned Torch version so it does not silently install a
  CPU-only build.
* ``batch_resumable(processes=N)`` / ``--processes N`` now falls back to the sequential path with a
  warning on platforms without a ``forkserver`` start method (e.g. Windows, which offers only
  ``spawn``), instead of crashing with ``ValueError: cannot find context for 'forkserver'``.
  Callers that always set ``N`` (e.g. Habitat Hub) run single-process there rather than failing
  every batch.
* Fixed ``scoutbot.__version__`` reporting a stale hardcoded value (``0.2.0``) instead of the
  packaged version. ``__version__`` (and the ``VERSION`` / ``version`` aliases) now derive from
  installed package metadata (``importlib.metadata``), making ``pyproject.toml`` the single source
  of truth; a regression test guards against future drift.
* Added an internal test-corpus builder (``tools/build_test_corpus.py``) and an opt-in,
  skip-if-absent CPU drift-regression test (``tests/test_corpus_regression.py``, gated on
  ``SCOUTBOT_TEST_CORPUS``). The builder selects density-tiered frames from the internal Matetsi
  survey, GPS/EXIF-strips them, and freezes CPU inference baselines with a pinned model + dependency
  stack; the regression test replays them and asserts exact CPU parity when the stack matches. Adds
  ``piexif`` to the ``test`` extra.

* Renamed the ecosystem namespace from ``rw`` ("Rebel Workflow") to ``osw`` ("Overhead Survey
  Workflow"): environment variables ``OSW_HOME`` / ``OSW_REGISTRY_PATH`` / ``OSW_MODELS_DIR``, the
  registry directory ``~/.osw``, and the public model bucket ``osw-public-models``. The local
  model registry is now the first implementation of the shared **OSW Model Registry** standard
  with schema id ``osw:model-registry:1.0``; its default location honors ``OSW_HOME`` (default
  ``~/.osw``) per the ecosystem toolchain-directory principle. Downloaded model files and Core ML
  exports now cache under the OSW models directory (``OSW_MODELS_DIR``, default ``~/.osw/models``)
  instead of ``~/.cache/scoutbot``, keeping the whole toolchain footprint under one home. This is
  a clean break with no ``rw`` compatibility aliases; existing ``~/.rebel-workflow`` registries
  and ``~/.cache/scoutbot`` model files are orphaned and rebuilt on next fetch. A registry schema-guard now rejects foreign or incompatible-major registry files
  (including the pre-release ``rw`` and interim ``osw:local-model-cache`` families) with a clear,
  actionable ``RegistrySchemaError`` (surfaced as a clean CLI error) instead of failing
  cryptically.
* Removed the runtime ``dill`` dependency. The two bundled KAZA models were re-serialized with the
  current Ultralytics/torch stack so they load without ``dill`` (weights unchanged — CPU inference
  is bit-identical), re-hosted under date-tagged filenames
  (``yolov8-onecls.kaza.20260729.pt`` / ``yolov8-cls.kaza.20260729.pt``), and their MD5 is now
  pinned in ``scoutbot.model_store`` ``CONFIGS`` for stronger integrity and cache invalidation.
  The download source order is now **GCS primary, Azure fallback** (previously Azure primary).
  Upgrading clients automatically re-download the re-serialized models on next use. **Note:** the
  Azure fallback objects are not yet uploaded (Azure storage was unavailable when this landed), so
  GCS is the only live source until that is completed.
* Upgraded the locked ML stack to ``torch==2.13.0`` / ``torchvision==0.28.0`` (with the
  attendant ``triton``, CUDA-runtime, and ``setuptools==83.0.0`` bumps) and removed the
  temporary ``uv audit`` ignores for the torch and setuptools advisories after the locked graph
  audited cleanly. **Platform change:** torch 2.13 ships only ``macosx_14_0_arm64`` wheels, so
  Apple Silicon now requires macOS 14 or newer.
* Added an opt-in multiprocess batch mode: ``batch_resumable(processes=N)`` /
  ``scoutbot batch-resumable --processes N`` (Linux/CUDA-first, default ``processes=1`` unchanged).
  Each of the ``N`` worker processes is launched via ``forkserver`` and loads its own model copy;
  a single writer in the parent process appends completed results to the JSONL file in completion
  order, keeping the existing crash-safe resume contract intact for concurrent writers. A lost or
  hung worker aborts the run fail-fast once it misses a bounded per-result deadline (new
  ``SCOUTBOT_MP_RESULT_TIMEOUT`` env var, default 600 seconds); re-running the same command
  resumes the remaining work. Both ``batch_resumable()`` and the CLI now accept
  ``summary_json`` / ``--summary-json`` to write a machine-readable run summary (throughput,
  ``per_worker_counts``, ``phase_totals_ns``, ``inference_duty_per_worker``) in sequential and
  multiprocess modes alike. **Observability change:** the ``batch_resumable`` observer summary
  event now reports ``wall_ns`` plus these enriched fields instead of the previous
  ``started_ns`` / ``ended_ns`` / ``duration_ns``; external consumers of that event should switch
  to ``wall_ns``.
* Upgraded ``ultralytics`` from ``==8.2.103`` to ``>=8.3,<9`` (currently resolves to 8.4.89).
  The 8.3+ series removes the ``numpy<2.0`` restriction for standard inference, which was
  previously forcing a source-built numpy on Python 3.13 / Windows and causing a fatal access
  violation during test collection. numpy 2.x is now the default for Python 3.13. Core ML export
  now requires ``coremltools>=9.0`` (the previous pin was unversioned). Two new transitive
  dependencies are added: ``polars>=0.20.0`` and ``nvidia-ml-py>=12.0.0``. A minor type-annotation
  fix in ``model_store.py`` was required because ``YOLO.export()`` now carries a typed return value.
* Model downloads try multiple sources in order with automatic failover if any source is
  unreachable or resets the connection mid-transfer. The built-in order is GCS primary
  (``storage.googleapis.com/osw-public-models``), Azure fallback. Registry provenance records
  which provider delivered the file. Download chunk size increased from 8 KB to 64 KB for better
  throughput on large model files.
* Added a local model registry backed by ``~/.osw/model-registry.json``.
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
  Upgraded ``idna`` 3.13 → 3.18 to resolve GHSA-65pc-fj4g-8rjx / PYSEC-2026-215.
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
