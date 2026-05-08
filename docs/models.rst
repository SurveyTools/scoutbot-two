Models
------

ScoutBot v3 uses YOLOv8 models trained on KAZA survey data. Models are downloaded automatically
on first use and cached at ``~/.cache/scoutbot/``.

Model Files
~~~~~~~~~~~

Both models are hosted on Azure Blob Storage with MD5 sidecar files for integrity verification:

``v3`` — single-class detector
   * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt
   * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5

``v3-cls`` — multi-class species classifier
   * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt
   * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5

Integrity Verification
~~~~~~~~~~~~~~~~~~~~~~

On each ``fetch()`` call, ScoutBot:

1. Downloads the ``.md5`` sidecar (small, always fresh).
2. Compares the sidecar's MD5 against the locally cached model file.
3. If they match, returns the cached path immediately — no download.
4. If they differ (or the file is absent), downloads the model, verifies the MD5, and
   raises ``AssertionError`` if the hash still does not match.

Supported Classes
~~~~~~~~~~~~~~~~~

To list the class labels supported by a model::

    import scoutbot
    print(scoutbot.get_classes(config='v3'))       # ['animal']
    print(scoutbot.get_classes(config='v3-cls'))   # ['elephant', 'giraffe', ...]

The ``v3`` model reports a single class (``animal``). The ``v3-cls`` model reports individual
species labels as trained.

Adding Future Models
~~~~~~~~~~~~~~~~~~~~

To register a new model, add an entry to ``CONFIGS`` in ``scoutbot/loc/__init__.py`` with
the required keys (``name``, ``url``, ``md5_url``, ``thresh``, ``slice_height``,
``slice_width``, ``overlap_height_ratio``, ``overlap_width_ratio``,
``perform_standard_pred``, ``postprocess_class_agnostic``). The ``fetch()``,
``get_classes()``, and CLI ``--config`` choices will all pick it up automatically.
