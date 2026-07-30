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
     - ``yolov8-onecls.kaza.20260729.pt``
     - Single-class detector
     - Yes
   * - ``v3-cls``
     - ``yolov8-cls.kaza.20260729.pt``
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

PyTorch-backed configs are hosted on both Google Cloud Storage (primary) and Azure Blob Storage
(fallback). Scoutbot tries each source in order and uses the first that succeeds. Both hosts carry
matching ``.md5`` sidecar files, and the expected MD5 is additionally pinned in code
(``scoutbot.model_store`` ``CONFIGS``); see `Integrity verification`_ below.

The current artifacts were re-serialized on 2026-07-29 from the original v3 KAZA weights with the
current Ultralytics/torch stack. The weights are identical to the earlier ``yolov8-*.kaza.pt``
files; only the on-disk serialization changed — the older files embedded a ``dill`` dependency,
which these do not — so the filenames carry a ``20260729`` build tag.

``v3`` — single-class detector (MD5 ``c4954cd3957b286913c886c4219cd6c5``)

  *GCS (primary)*

  * Model: https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-onecls.kaza.20260729.pt
  * MD5:   https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-onecls.kaza.20260729.pt.md5

  *Azure (fallback)*

  * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.20260729.pt
  * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.20260729.pt.md5

``v3-cls`` — multi-class species detector (MD5 ``6a1565be5460d88a0e2eea2a74e66863``)

  *GCS (primary)*

  * Model: https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-cls.kaza.20260729.pt
  * MD5:   https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-cls.kaza.20260729.pt.md5

  *Azure (fallback)*

  * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.20260729.pt
  * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.20260729.pt.md5

.. note::

   The Azure fallback objects for the ``20260729`` artifacts are not yet uploaded — Azure storage
   was unavailable when this work landed — so GCS is currently the only live source. Completing the
   Azure upload and failover validation is tracked in :doc:`roadmap`.

Other artifacts in the model bucket — notably the ``.onnx`` exports (``scout.loc.*.onnx``) — are
not part of this catalog and are unaffected by the ``dill`` re-serialization: they are not loaded
through the PyTorch pickle path and keep their existing names and URLs.

Core ML configs do not have hosted ``.mlpackage`` artifacts. Scoutbot exports them on demand from
the verified source ``.pt`` model and stores the resulting bundle in ``~/.cache/scoutbot/``.

Integrity verification
~~~~~~~~~~~~~~~~~~~~~~

On each ``fetch()`` call for a hosted ``.pt`` model, Scoutbot:

1. Checks the local model registry; returns the cached file without network access only if its
   MD5 matches the value pinned for that config in ``CONFIGS`` (the ``md5`` field). A cached file
   registered under a different hash — for example a pre-``20260729`` artifact — is treated as
   stale and re-fetched.
2. Tries each configured source in order (GCS first, then Azure):

   a. Downloads the ``.md5`` sidecar from that source.
   b. Compares the sidecar's MD5 against the locally cached model file.
   c. If they match, returns the cached path immediately (cache hit).
   d. Otherwise, streams the model file from that source and verifies the local MD5 on
      completion.
   e. If the download or verification fails, logs a warning and moves to the next source.

3. Asserts the hosted sidecar's MD5 equals the pinned ``CONFIGS`` value, so a bad or partial
   re-upload fails loudly instead of being served.
4. Raises ``RuntimeError`` if all sources fail.
5. Registers the successfully downloaded model in the local registry with the provider name
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
``docs/plans/2026-05-09-scoutbot-registry.md`` and in :doc:`roadmap`.
