Overview
--------

ScoutBot is a machine learning package for aerial wildlife survey image processing. It detects
and localises animals in aerial photographs using YOLOv8 with SAHI (Slicing Aided Hyper
Inference).

The v3 pipeline runs in two steps:

   1. **Fetch** — Download the YOLOv8 model from Azure Blob Storage if it is not already cached
      locally. MD5 verification ensures the local copy matches the remote.

   2. **Sliced inference** — The input image is divided into overlapping 512×512 slices.
      YOLOv8 runs inference on all slices in configurable batches (default: 128 slices per
      batch). SAHI collects and deduplicates the resulting detections across slice boundaries
      using class-agnostic NMS, then returns the full set of bounding boxes.

Two model variants are available:

   - ``v3`` (``yolov8-onecls.kaza.pt``) — single-class (animal) detector
   - ``v3-cls`` (``yolov8-cls.kaza.pt``) — multi-class species classifier

The main entry points are:

   - :func:`scoutbot.pipeline` — run on a single image
   - :func:`scoutbot.batch` — run on a list of images with model reuse
   - :func:`scoutbot.batch_resumable` — like ``batch``, but writes results incrementally to
     JSONL so processing can be resumed after interruption
   - :func:`scoutbot.fetch` — explicitly download a model
   - :func:`scoutbot.get_classes` — list the class labels supported by a model
