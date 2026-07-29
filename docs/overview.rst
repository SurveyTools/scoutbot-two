Overview
--------

Scoutbot is a machine learning package for aerial wildlife survey image processing. It detects
and localises animals in aerial photographs using YOLOv8 with SAHI (Slicing Aided Hyper
Inference).

Scoutbot supports direct CLI use and integration through its Python API. The same output format is
used for single-image, batch, and resumable-batch workflows.

Pipeline shape
~~~~~~~~~~~~~~

The pipeline runs in four stages:

   1. **Fetch/cache** — Download the YOLOv8 ``.pt`` model from Azure Blob Storage if it is not
      already cached locally. MD5 verification ensures the local copy matches the remote. Core ML
      configs first fetch the verified source ``.pt`` model, then export a local ``.mlpackage``.

   2. **Slicing** — The input image is divided into overlapping 512×512 slices. The default
      overlap is 25% in both dimensions.

   3. **Backend inference** — PyTorch configs run YOLOv8 through SAHI/Ultralytics on the selected
      ``cpu``, ``cuda``, or ``mps`` device. Core ML configs use the same SAHI path but dispatch
      each fixed-size tile through an exported Core ML bundle on macOS.

   4. **Merge/output** — SAHI collects and deduplicates detections across slice boundaries using
      class-agnostic NMS, then Scoutbot returns pixel-coordinate bounding boxes in the
      ``{l, c, x, y, w, h}`` format.

Model configs
~~~~~~~~~~~~~

Four configs are available:

   - ``v3`` (``yolov8-onecls.kaza.20260729.pt``) — single-class detector
   - ``v3-cls`` (``yolov8-cls.kaza.20260729.pt``) — multi-class species detector
   - ``v3-coreml`` (``yolov8-onecls.kaza.mlpackage``) — Core ML export of ``v3``
   - ``v3-cls-coreml`` (``yolov8-cls.kaza.mlpackage``) — Core ML export of ``v3-cls``

The main entry points are:

   - :func:`scoutbot.pipeline` — run on a single image
   - :func:`scoutbot.batch` — run on a list of images with model reuse
   - :func:`scoutbot.batch_resumable` — like ``batch``, but writes results incrementally to
     JSONL so processing can be resumed after interruption
   - :func:`scoutbot.fetch` — explicitly download a model
   - :func:`scoutbot.get_classes` — list the class labels supported by a model
