Models
------

Scoutbot v3 uses YOLOv8 models trained on KAZA aerial survey data. PyTorch ``.pt`` models are
downloaded automatically on first use and cached at ``~/.cache/scoutbot/``. Core ML
``.mlpackage`` bundles are exported locally from those source ``.pt`` models.

Model configs
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Config
     - Artifact
     - Description
     - Hosted
   * - ``v3``
     - ``yolov8-onecls.kaza.pt``
     - Single-class detector
     - Yes
   * - ``v3-cls``
     - ``yolov8-cls.kaza.pt``
     - Multi-class species detector
     - Yes
   * - ``v3-coreml``
     - ``yolov8-onecls.kaza.mlpackage``
     - Core ML export of ``v3``
     - No; exported locally
   * - ``v3-cls-coreml``
     - ``yolov8-cls.kaza.mlpackage``
     - Core ML export of ``v3-cls``
     - No; exported locally

Hosted model files
~~~~~~~~~~~~~~~~~~

PyTorch-backed configs are hosted on both Azure Blob Storage (primary) and Google Cloud Storage
(fallback). Scoutbot tries each source in order and uses the first that succeeds. Both hosts carry
matching ``.md5`` sidecar files for integrity verification.

``v3`` — single-class detector

  *Azure (primary)*

  * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt
  * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5

  *GCS (fallback)*

  * Model: https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt
  * MD5:   https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt.md5

``v3-cls`` — multi-class species detector

  *Azure (primary)*

  * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt
  * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5

  *GCS (fallback)*

  * Model: https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt
  * MD5:   https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt.md5

Core ML configs do not have hosted ``.mlpackage`` artifacts. Scoutbot exports them on demand from
the verified source ``.pt`` model and stores the resulting bundle in ``~/.cache/scoutbot/``.

Integrity verification
~~~~~~~~~~~~~~~~~~~~~~

On each ``fetch()`` call for a hosted ``.pt`` model, Scoutbot:

1. Checks the local model registry; returns immediately if a valid cached entry is found
   (no network access).
2. Tries each configured source in order (Azure first, then GCS):

   a. Downloads the ``.md5`` sidecar from that source.
   b. Compares the sidecar's MD5 against the locally cached model file.
   c. If they match, returns the cached path immediately (cache hit).
   d. Otherwise, streams the model file from that source and verifies the local MD5 on
      completion.
   e. If the download or verification fails, logs a warning and moves to the next source.

3. Raises ``RuntimeError`` if all sources fail.
4. Registers the successfully downloaded model in the local registry with the provider name
   recorded in the provenance block.

Core ML bundles are directory packages, so they do not use remote MD5 sidecars. Their integrity
depends on exporting from the already-verified source ``.pt`` model.

Supported classes
~~~~~~~~~~~~~~~~~

To list the class labels supported by a model:

.. code-block:: python

   import scoutbot

   print(scoutbot.get_classes(config='v3'))       # ['animal']
   print(scoutbot.get_classes(config='v3-cls'))   # ['elephant', 'giraffe', ...]

The ``v3`` model reports a single class. The ``v3-cls`` model reports individual species labels
as trained. Core ML configs mirror the classes of their source configs.

Model provenance and future integrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current models were trained on KAZA aerial survey data. New model integration is planned but
not yet a mature public extension API. The current model catalog lives in
``scoutbot.model_store``. Future registry work is tracked in
``docs/superpowers/plans/2026-05-09-scoutbot-registry.md`` and in :doc:`roadmap`.
