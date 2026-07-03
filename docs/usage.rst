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

Each JSONL line contains one image result.

Other commands
--------------

.. code-block:: bash

   uv run scoutbot fetch --config v3            # download model
   uv run scoutbot get_classes --config v3-cls  # list species labels
   uv run scoutbot export-coreml                # export Core ML bundles
   uv run scoutbot example                      # run on a built-in test image

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
