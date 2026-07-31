Configuration
=============

Configuration controls model selection, device selection, thresholds, and tile batching.

Model configs
-------------

.. list-table::
   :header-rows: 1

   * - Config
     - Backend
     - Description
   * - ``v3``
     - PyTorch / Ultralytics
     - Single-class detector
   * - ``v3-cls``
     - PyTorch / Ultralytics
     - Multi-class species detector
   * - ``v3-coreml``
     - Core ML
     - Local Core ML export of ``v3`` for macOS
   * - ``v3-cls-coreml``
     - Core ML
     - Local Core ML export of ``v3-cls`` for macOS

Select a config with ``--config``:

.. code-block:: bash

   uv run scoutbot pipeline image.jpg --config v3-cls

Or set ``LOC_CONFIG`` / ``CONFIG`` for API calls that use the default config:

.. code-block:: bash

   export LOC_CONFIG=v3-coreml

Environment variables
---------------------

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Effect
   * - ``SCOUTBOT_DEVICE``
     - ``cuda:0``
     - Inference device for PyTorch configs. Accepted CLI values are ``cuda:0``, ``cuda``,
       ``mps``, and ``cpu``.
   * - ``LOC_CONFIG`` / ``CONFIG``
     - ``v3``
     - Default model config. ``LOC_CONFIG`` is checked first; ``CONFIG`` is used as a fallback.
   * - ``DETECTOR_BATCH_SIZE``
     - ``128``
     - Number of sliced tiles sent to the detector in one batch for PyTorch-backed inference.
       Reduce this for low-memory environments.
   * - ``VERBOSE``
     - unset
     - Enable debug logging when set to any value.
   * - ``SCOUTBOT_MP_RESULT_TIMEOUT``
     - ``600``
     - Per-result deadline in seconds for ``batch_resumable(processes=N)`` /
       ``--processes N`` when ``N`` > 1. A worker that does not return a result within
       this many seconds is treated as dead, which aborts the run fail-fast (see
       `Parallel execution`_). Re-running the same command resumes the remaining work.
   * - ``OSW_HOME``
     - ``~/.osw``
     - Shared Overhead Survey Workflow toolchain directory. The model registry and per-tool
       registry paths default to sub-directories under it.
   * - ``OSW_REGISTRY_PATH``
     - ``$OSW_HOME/model-registry.json``
     - Full path to the OSW Model Registry file. Overrides ``OSW_HOME`` for the registry location.
   * - ``OSW_MODELS_DIR``
     - ``$OSW_HOME/models``
     - Directory holding the downloaded model files and on-demand Core ML exports. Overrides
       ``OSW_HOME`` for the models location.

Device guidance
---------------

Use ``cuda`` or ``cuda:0`` for NVIDIA GPUs, ``mps`` for PyTorch acceleration on Apple Silicon,
and ``cpu`` for portable CPU inference. Core ML configs are macOS-only and use Ultralytics' Core
ML runtime internally; the command-line ``--device`` option is not meaningful for Core ML in the
same way it is for PyTorch configs.

Parallel execution
-------------------

``batch_resumable`` accepts ``processes=N`` (CLI: ``--processes N``) to process images across
``N`` worker processes on a single machine instead of the default sequential loop
(``processes=1``). This mode is **Linux/CUDA-first**; it uses the ``forkserver`` start method to
launch workers.

Each worker process loads its own copy of the model, so ``N`` copies must all fit in the GPU's
available VRAM at once. As a data point, the ``v3`` model uses roughly 1-2 GB of VRAM per copy, so
around 8 copies share a 12 GB GPU comfortably. Size ``N`` to your model and hardware accordingly,
and see :doc:`performance` for the measured throughput gains and how to read the run summary.

Thresholds and slicing
----------------------

``--loc-thresh`` is an integer percentage from 0 to 100. The CLI converts it to the detector
confidence threshold used by the pipeline. The default is ``45``.

All current configs use 512×512 slices with 25% overlap in both dimensions. SAHI merges
detections across slice boundaries with class-agnostic NMS.

Cache location
--------------

Models are cached in the OSW models directory — ``OSW_MODELS_DIR`` (default ``~/.osw/models/``,
under ``OSW_HOME``; see `Environment variables`_). On Windows this resolves under the user's home
directory, for example ``C:\Users\<user>\.osw\models\``.

PyTorch ``.pt`` models are downloaded and verified with MD5 sidecars. Core ML ``.mlpackage``
bundles are exported locally from the verified source ``.pt`` model and cached in the same
directory.

Model download sources
----------------------

Scoutbot tries download sources in order and uses the first that succeeds.
The built-in order is:

1. **Google Cloud Storage** (``storage.googleapis.com/osw-public-models``) — primary
2. **Azure Blob Storage** (``scoutmodels.blob.core.windows.net``) — fallback

If GCS is unreachable or resets the connection mid-transfer, the download automatically retries
from Azure with no user action required. A warning is logged for each skipped source. Once any
source delivers the file and the local MD5 is verified against the value pinned in ``CONFIGS``, the
model is registered in the local registry and subsequent runs resolve from cache with no network
access.

See :doc:`models` for the full URL listing per model.
