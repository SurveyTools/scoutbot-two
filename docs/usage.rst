Usage
=====

Scoutbot can be run from the command line or imported as a Python package. Commands accept the
same model configs documented in :doc:`configuration`.

Single image
------------

Run one image and print JSON to stdout:

.. code-block:: bash

   uv run scoutbot pipeline /path/to/image.jpg --config v3 --device mps

Save JSON to a file:

.. code-block:: bash

   uv run scoutbot pipeline /path/to/image.jpg --config v3-cls --output results.json

Use ``--model NAME`` to select a model by registry name instead of a config alias. This
requires the model to already be in the local registry (run ``scoutbot fetch`` once first):

.. code-block:: bash

   uv run scoutbot fetch --config v3             # register on first use
   uv run scoutbot pipeline /path/to/image.jpg --model scoutbot-v3 --device mps

Batch
-----

Process multiple images and print JSON to stdout:

.. code-block:: bash

   uv run scoutbot batch --config v3 --device mps images/*.jpg

Save batch output:

.. code-block:: bash

   uv run scoutbot batch --config v3 --device mps --output results.json images/*.jpg

Use ``--no-progress`` for quiet batch runs in scripts:

.. code-block:: bash

   uv run scoutbot batch --config v3-coreml --no-progress images/*.jpg

Resumable batch
---------------

For large surveys, write each image result incrementally to JSONL. Re-running the same command
skips completed images and resumes the remaining work:

.. code-block:: bash

   uv run scoutbot batch-resumable --output survey.jsonl --config v3 --device mps images/*.jpg

Each JSONL line contains one image result, including a ``model`` provenance block with the
registry name, MD5, framework, and the effective SAHI parameters used for that run:

.. code-block:: json

   {
     "filepath": "/path/to/img.jpg",
     "wic": 0.832,
     "detections": [],
     "timestamp": "2026-05-09T12:00:00Z",
     "config": "v3",
     "model": {
       "name": "scoutbot-v3",
       "file": "yolov8-onecls.kaza.pt",
       "md5": "bb1613609235f7f868e827cc5c5bba75",
       "framework": "yolov8",
       "run_config": {
         "confidence_threshold": 0.45,
         "slice_height": 512,
         "slice_width": 512,
         "overlap_height_ratio": 0.25,
         "overlap_width_ratio": 0.25,
         "perform_standard_pred": false,
         "postprocess_class_agnostic": true
       }
     }
   }

Use ``--processes N`` to process images across multiple worker processes on a single machine
(Linux/CUDA-first; ``N`` defaults to ``1``, the unchanged sequential path). Each worker loads its
own copy of the model, so ``N`` copies must fit in available VRAM -- see
:doc:`configuration` for sizing guidance. Because workers complete at different speeds, results
are written to the JSONL file in **completion order**, not input order; resume matches on the
``filepath`` field, so completion order does not affect correctness. Sort the output file by
``filepath`` afterwards if a consumer needs input order.

If a worker process is lost or stops responding, the run aborts (fail-fast) rather than silently
dropping results. Everything already written to the JSONL file stays valid and durable, so
re-running the exact same command resumes from where it left off, processing only the remaining
images:

.. code-block:: bash

   uv run scoutbot batch-resumable --output survey.jsonl --config v3 --device cuda:0 \
     --processes 8 images/*.jpg

Pass ``--summary-json PATH`` to also write a machine-readable run summary (throughput, per-worker
image counts, and phase timing totals) alongside the JSONL output. This is written for both the
sequential and multiprocess paths:

.. code-block:: bash

   uv run scoutbot batch-resumable --output survey.jsonl --config v3 --device cuda:0 \
     --processes 8 --summary-json survey-summary.json images/*.jpg

See :doc:`performance` for how to interpret the summary fields.

Other commands
--------------

.. code-block:: bash

   uv run scoutbot fetch --config v3            # download model
   uv run scoutbot get_classes --config v3-cls  # list species labels
   uv run scoutbot export-coreml                # export Core ML bundles
   uv run scoutbot example                      # run on a built-in test image

**Registry commands**

List all models in the local registry:

.. code-block:: bash

   uv run scoutbot models

Print the full registry entry for a specific model as JSON:

.. code-block:: bash

   uv run scoutbot models info scoutbot-v3

Register a local model file (for custom or community models not in the built-in catalog):

.. code-block:: bash

   uv run scoutbot register \
     --name my-custom-model \
     --path /path/to/model.pt \
     --arch yolov8 \
     --description "Fine-tuned on XYZ dataset" \
     --threshold 0.4 \
     --slice-height 512 \
     --slice-width 512 \
     --overlap 0.25

Once registered, pass the model name to ``--model`` on any inference command instead of ``--config``.

Python API
----------

.. code-block:: python

   import scoutbot

   # Single image
   wic_score, detections = scoutbot.pipeline('/path/to/image.jpg', config='v3')

   # Batch (reuses a loaded model)
   wic_list, detects_list = scoutbot.batch(
       filepaths,
       config='v3',
       backend_device='mps',
   )

   # Resumable batch writes JSONL incrementally
   summary = scoutbot.batch_resumable(
       filepaths=filepaths,
       output_path='survey.jsonl',
       config='v3',
       backend_device='mps',
   )

   # Multiprocess resumable batch (Linux/CUDA-first): each of the N worker
   # processes loads its own model copy; results land in completion order.
   summary = scoutbot.batch_resumable(
       filepaths=filepaths,
       output_path='survey.jsonl',
       config='v3',
       backend_device='cuda:0',
       processes=8,
       summary_json='survey-summary.json',
   )

   # Download a model and list supported classes
   model_path = scoutbot.fetch(config='v3')
   classes = scoutbot.get_classes(config='v3-cls')

Output schema
-------------

The CLI returns image paths mapped to a ``wic`` score and a ``loc`` detection list:

.. code-block:: json

   {
     "/path/to/image.jpg": {
       "wic": 0.832,
       "loc": [
         {"l": "elephant", "c": 0.832, "x": 4596, "y": 2322, "w": 77, "h": 148}
       ]
     }
   }

The Python ``pipeline()`` function returns ``(wic_score, detections)`` for one image. Each
detection is a dictionary:

.. code-block:: python

   {
       'l': 'elephant',  # label
       'c': 0.832,       # confidence, 0-1
       'x': 4596,        # top-left x coordinate in source-image pixels
       'y': 2322,        # top-left y coordinate in source-image pixels
       'w': 77,          # bounding-box width in pixels
       'h': 148,         # bounding-box height in pixels
   }

The ``wic`` score is the maximum detection confidence for the image, or ``0.0`` when there are no
detections.

Review notes
------------

Bounding boxes use source-image pixel coordinates. ``x`` and ``y`` are the top-left corner;
``w`` and ``h`` are width and height in pixels. Map detections back to source imagery using the
image path keys in JSON output or the ``filepath`` field/order in JSONL output.

Confidence ``c`` and the ``--loc-thresh`` command-line option control which detections are
returned. Reviewers should pay special attention to:

* detections close to the confidence threshold
* duplicate or merged boxes near tile boundaries
* class-label uncertainty when using multi-class configs

Visualisation
-------------

A standalone visualisation tool is provided in ``tools/visualise.py``. It renders
bounding boxes and confidence scores from ScoutBot JSON output onto the source image.
By default it hands the result to the OS default handler for the file URI (Preview on
macOS, ``xdg-open`` on Linux); use ``--output`` with ``--no-display`` to save without
opening anything.

.. code-block:: bash

   uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json
   uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json --threshold 0.5
   uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json --output annotated.jpg --no-display

When the JSON file contains multiple image entries (e.g. from ``scoutbot batch``
output), specify which image to display with ``--image-key``:

.. code-block:: bash

   uv run python tools/visualise.py IMAGE.jpg batch_results.json --image-key /path/to/image.jpg

Pass ``--help`` for the full option list.
