Environment Variables
---------------------

The ScoutBot API and CLI have environment variables that allow you to configure global settings.

   - ``SCOUTBOT_DEVICE`` (default: ``cuda:0``)
      The inference device passed to the YOLOv8 model. Accepted values: ``cuda:0``,
      ``cuda``, ``mps``, ``cpu``. On Apple Silicon Macs, set to ``mps`` for GPU
      acceleration. Overridden by the ``--device`` CLI flag when specified.
   - ``LOC_CONFIG`` (default: ``v3``)
      The model configuration to use. Must be one of ``v3`` or ``v3-cls``.
      Aliases: ``CONFIG`` (checked if ``LOC_CONFIG`` is not set).
   - ``DETECTOR_BATCH_SIZE`` (default: ``128``)
      Number of image slices to send to the GPU in a single batch during inference.
      Increase for faster throughput (at the cost of more GPU memory); decrease if you run
      out of memory. A value of ``8`` is suitable for CI and low-memory environments.
   - ``VERBOSE`` (default: not set)
      Verbosity flag. Setting this to any value (e.g. ``VERBOSE=1``) enables debug logging.
