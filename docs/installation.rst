Installation
============

Scoutbot supports direct local runs, contributor development, and Apple Silicon Core ML runs,
with best-effort GPU support for Windows NVIDIA systems.

Python support
--------------

Scoutbot currently supports Python 3.10, 3.11, 3.12, and 3.13.

Direct local install
--------------------

Use this when you want to run Scoutbot directly from a clone:

.. code-block:: bash

   git clone https://github.com/SurveyTools/scoutbot-2026
   cd scoutbot-2026
   uv pip install -e .

Models are downloaded automatically on first use and cached at ``~/.osw/models/``. To fetch
models explicitly:

.. code-block:: bash

   uv run scoutbot fetch --config v3
   uv run scoutbot fetch --config v3-cls

Upgrading is automatic: the expected model hash is pinned in code, so after upgrading past a model
re-serialization (such as the 2026-07-29 dill-free re-export) the next run detects the cached file
is stale and re-downloads the current artifact. The previous cache file is left in place.

Contributor install
-------------------

Use the test extra for development and quality checks:

.. code-block:: bash

   uv sync --extra test

The test extra includes pytest, pytest-cov, pytest-random-order, Ruff, and mypy.

Documentation install
---------------------

Use the docs extra when building Sphinx documentation:

.. code-block:: bash

   uv sync --extra docs
   uv run sphinx-build -M html docs docs/build

Apple Silicon Core ML
---------------------

.. note::

   Apple Silicon requires **macOS 14 (Sonoma) or newer**. The locked ``torch`` 2.13 build
   publishes only ``macosx_14_0_arm64`` wheels, so macOS 13 is no longer supported on arm64.

Core ML support is macOS-only and requires ``coremltools`` through the ``coreml`` extra:

.. code-block:: bash

   uv sync --extra coreml

Then use a Core ML config:

.. code-block:: bash

   uv run scoutbot pipeline image.jpg --config v3-coreml
   uv run scoutbot batch *.jpg --config v3-coreml --no-progress

The ``.mlpackage`` bundle is exported on first use from the source ``.pt`` model and cached at
``~/.osw/models/``. Export can also be triggered explicitly:

.. code-block:: bash

   uv run scoutbot export-coreml
   uv run scoutbot export-coreml --config v3-coreml
   uv run scoutbot export-coreml --force

Windows with NVIDIA GPU
-----------------------

CUDA GPU support is best-effort for NVIDIA systems on Windows. It is not CI-tested and is not
required to run the package — CPU-only installs work on any Windows machine without a GPU.

The default install may pull in a CPU-only PyTorch build on Windows. For NVIDIA GPU use, replace
that build with the CUDA wheel after installing Scoutbot:

.. code-block:: powershell

   git clone https://github.com/SurveyTools/scoutbot-2026
   cd scoutbot-2026
   uv venv
   .\.venv\Scripts\Activate.ps1
   uv pip install -e .

   # Replace CPU-only PyTorch with the CUDA build (requires NVIDIA GPU + driver >= 525)
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

   # Required for large image handling (SAHI tiling)
   uv pip install scikit-image imagecodecs

Verify the GPU is detected before running:

.. code-block:: powershell

   uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

``True`` means PyTorch found an NVIDIA GPU and a compatible CUDA driver. ``False`` is expected on
CPU-only machines and is not an error — the package runs normally without a GPU.

PowerShell does not expand glob patterns for external commands, so pass files with
``Get-ChildItem``:

.. code-block:: powershell

   $imgs = (Get-ChildItem "C:\path\to\images" -Filter "*.JPG").FullName
   uv run scoutbot batch @imgs --output results.json

If you have a GPU and encounter out-of-memory errors, reduce the tile batch size:

.. code-block:: powershell

   $env:DETECTOR_BATCH_SIZE = "16"
   uv run scoutbot batch @imgs --output results.json
