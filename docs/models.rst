Models
------

ScoutBot v3 uses YOLOv8 models trained on KAZA aerial survey data. PyTorch ``.pt`` models are
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

PyTorch-backed configs are hosted on Azure Blob Storage with MD5 sidecar files for integrity
verification:

``v3`` — single-class detector
   * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt
   * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5

``v3-cls`` — multi-class species detector
   * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt
   * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5

Core ML configs do not have hosted ``.mlpackage`` artifacts. ScoutBot exports them on demand from
the verified source ``.pt`` model and stores the resulting bundle in ``~/.cache/scoutbot/``.

Integrity verification
~~~~~~~~~~~~~~~~~~~~~~

On each ``fetch()`` call for a hosted ``.pt`` model, ScoutBot:

1. Downloads the ``.md5`` sidecar.
2. Compares the sidecar's MD5 against the locally cached model file.
3. If they match, returns the cached path immediately.
4. If they differ, or the local model is absent, downloads the model and verifies the MD5.
5. Raises ``AssertionError`` if the downloaded file still does not match the sidecar.

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
not yet a mature public extension API. Future registry work is tracked in
``docs/superpowers/plans/2026-05-09-scoutbot-registry.md`` and in :doc:`roadmap`.
