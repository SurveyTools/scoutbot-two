Development
===========

This page covers contributor setup, tests, formatting, linting, type checking, CI workflows,
documentation builds, and model-integration contributor guidance.

Runtime rule
------------

Use ``uv`` for Python and tool commands. Do not run ``python``, ``pytest``, ``ruff``, ``mypy``,
or similar tools directly.

.. code-block:: bash

   uv run pytest -m "not slow"
   uv run ruff check .
   uv run mypy scoutbot

Python support
--------------

ScoutBot currently supports Python 3.10, 3.11, and 3.12. The package metadata declares
``>=3.10,<3.13``.

Local setup
-----------

Install development and test dependencies with the ``test`` extra:

.. code-block:: bash

   uv sync --extra test

This installs pytest, pytest-cov, pytest-random-order, Ruff, and mypy.

Testing
-------

Fast/offline tests are the default local and CI path:

.. code-block:: bash

   uv run pytest -m "not slow"

Slow tests download models or run real inference:

.. code-block:: bash

   uv run pytest -m "slow"

Run coverage on the fast suite:

.. code-block:: bash

   uv run pytest -m "not slow" --cov=scoutbot --cov-report=html

Quality checks
--------------

Ruff handles formatting, linting, and import ordering:

.. code-block:: bash

   uv run ruff format --check .
   uv run ruff check .

Scoped mypy checks library code:

.. code-block:: bash

   uv run mypy scoutbot

Run all configured pre-commit hooks:

.. code-block:: bash

   uv run pre-commit run --all-files

CI workflows
------------

``.github/workflows/testing.yml``
   Runs on pull requests and pushes to ``main``. It tests Python 3.10, 3.11, and 3.12, installs
   with ``uv sync --extra test``, and runs ``uv run pytest -m "not slow"`` with coverage.

``.github/workflows/quality.yml``
   Runs on pull requests and pushes to ``main``. It installs with ``uv sync --extra test`` and
   runs Ruff format check, Ruff lint, and ``uv run mypy scoutbot``.

Documentation builds
--------------------

Install docs dependencies and build Sphinx:

.. code-block:: bash

   uv sync --extra docs
   uv run sphinx-build -M html docs docs/build

Model contributors and integrators
----------------------------------

The current public model-extension API is not mature. Model configs live in
``scoutbot/loc/__init__.py`` today, and future registry work is tracked in
``docs/superpowers/plans/2026-05-09-scoutbot-registry.md`` and :doc:`roadmap`.

New model integration work should include:

* artifact provenance
* integrity verification
* config schema tests
* CLI choice tests
* output compatibility checks

