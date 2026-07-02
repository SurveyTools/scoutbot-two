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

Device guidance
---------------

Use ``cuda`` or ``cuda:0`` for NVIDIA GPUs, ``mps`` for PyTorch acceleration on Apple Silicon,
and ``cpu`` for portable CPU inference. Core ML configs are macOS-only and use Ultralytics' Core
ML runtime internally; the command-line ``--device`` option is not meaningful for Core ML in the
same way it is for PyTorch configs.

Thresholds and slicing
----------------------

``--loc-thresh`` is an integer percentage from 0 to 100. The CLI converts it to the detector
confidence threshold used by the pipeline. The default is ``45``.

All current configs use 512×512 slices with 25% overlap in both dimensions. SAHI merges
detections across slice boundaries with class-agnostic NMS.

Cache location
--------------

Models are cached at ``~/.cache/scoutbot/``. On Windows, this resolves under the user's home
directory, for example ``C:\Users\<user>\.cache\scoutbot\``.

PyTorch ``.pt`` models are downloaded and verified with MD5 sidecars. Core ML ``.mlpackage``
bundles are exported locally from the verified source ``.pt`` model and cached in the same
directory.
