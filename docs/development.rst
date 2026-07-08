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

Scoutbot currently supports Python 3.10, 3.11, 3.12, and 3.13. The package metadata declares
``>=3.10,<3.14``.

Local setup
-----------

For a full maintainer environment (tests, docs, and Core ML on macOS):

.. code-block:: bash

   uv sync --extra all

For the test/quality tools only (the CI-equivalent path):

.. code-block:: bash

   uv sync --extra test

``--extra test`` installs pytest, pytest-cov, pytest-random-order, Ruff, mypy, and Bandit.
``--extra all`` is a shorthand for ``test`` + ``docs`` + ``coreml`` combined.

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

Testing philosophy
~~~~~~~~~~~~~~~~~~

New tests must cover meaningful branches — exception handlers, guard clauses, and
behavioral forks — not just add lines to a coverage metric. Branches that exist only
to satisfy a metric (e.g. ``if __name__ == '__main__':``) are explicitly excluded.

The mock pattern used throughout: ``unittest.mock.patch`` for network and model
dependencies, ``tmp_path`` for filesystem, ``CliRunner`` for CLI commands. Only add
``@pytest.mark.slow`` when a test actually hits the network or loads a real model
file from disk.

Quality checks
--------------

Ruff handles formatting, linting, and import ordering:

.. code-block:: bash

   uv run ruff format --check .
   uv run ruff check .

Scoped mypy checks library code and tools:

.. code-block:: bash

   uv run mypy scoutbot tools/visualise.py

Dependency vulnerability scanning audits the locked project graph:

.. code-block:: bash

   uv audit --locked --preview-features audit-command

Source-code security scanning checks the ``scoutbot/``, ``benchmarks/``, and ``tools/``
trees for common Python security mistakes (``tests/`` is excluded — pytest assertions are
expected there):

.. code-block:: bash

   uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/

``uv audit`` and Bandit cover different surfaces: ``uv audit`` checks installed packages
against the vulnerability database; Bandit checks your source code for insecure patterns.
Both run in CI and both should pass locally before opening a pull request.

Run all configured pre-commit hooks:

.. code-block:: bash

   uv run pre-commit run --all-files

CI workflows
------------

``.github/workflows/testing.yml``
   Runs on pull requests and pushes to ``main``. It tests Python 3.10, 3.11, 3.12, and 3.13, installs
   with ``uv sync --extra test``, and runs ``uv run pytest -m "not slow"`` with coverage.
   A ``windows-smoke`` job also runs the fast/offline suite on Windows with Python 3.13 only, providing
   smoke coverage for non-Core ML code without claiming full Windows parity.

``.github/workflows/quality.yml``
   Runs on pull requests and pushes to ``main``. It installs with ``uv sync --extra test`` and
   runs Ruff format check, Ruff lint, ``uv run mypy scoutbot tools/visualise.py``, a
   locked-graph dependency audit (``uv audit --locked``), and a Bandit source-code security
   scan (``scoutbot/``, ``benchmarks/``, ``tools/``).

GPU and CUDA
~~~~~~~~~~~~

CUDA is not required in CI and is not tested there. For local contributors with NVIDIA hardware,
CUDA support is best-effort: the package runs correctly on CPU without any GPU present. If you
are testing on a GPU-enabled Windows machine, run the verification command in
:doc:`installation` (Windows NVIDIA section) to confirm PyTorch detects the GPU before filing
bug reports related to GPU behavior.

Documentation builds
--------------------

Install docs dependencies and build Sphinx:

.. code-block:: bash

   uv sync --extra docs
   uv run sphinx-build -M html docs docs/build

GitHub Pages is the canonical hosted docs site. Route readers to rendered HTML and keep local
Sphinx builds as a maintainer path.

Documentation pass
------------------

Every implementation plan must include a final documentation pass before completion. Review
whether the work changes user behavior, commands, configuration, models, outputs, dependencies,
quality checks, security posture, performance characteristics, or roadmap status. Update the
affected docs in the same change set, or explicitly record why no doc update is needed.

This section mirrors the agent-facing rule in ``CLAUDE.md``. Keep both copies aligned until a
public contributor guide becomes the shared source for contributor documentation requirements.

Use these documentation roles to decide what may be impacted:

``README.md``
   Routes readers and should change when quick-start commands, badges, supported workflows, or
   top-level project positioning changes.

``docs/installation.rst``
   Owns install commands, extras, platform setup, CUDA/Core ML setup, and model cache
   prerequisites.

``docs/usage.rst``
   Owns CLI/API examples, output schemas, batch/resume behavior, and QA/review interpretation
   notes.

``docs/configuration.rst``
   Owns environment variables, config names, devices, thresholds, cache behavior, and runtime
   controls.

``docs/models.rst``
   Owns model artifacts, source URLs, MD5 verification, provenance, and model availability.

``docs/coreml.rst``
   Owns Core ML backend mechanics, export behavior, parity caveats, and Apple-specific backend
   constraints.

``docs/performance.rst``
   Owns benchmark interpretation, bottlenecks, observability needs, and hardware comparison
   caveats.

``docs/api.rst`` and ``docs/cli.rst``
   Own generated API/CLI reference surfaces.

``docs/development.rst``
   Owns contributor setup, tests, lint/type checks, CI workflows, and model-integration
   contributor guidance.

``docs/security.rst``
   Owns model download policy, integrity verification, dependency/security scanning posture, CI
   scanning rationale, and data/privacy assumptions.

``docs/roadmap.rst``
   Owns planned/deferred work and investigation status.

``CHANGELOG.rst``
   Owns user-visible changes, compatibility changes, workflow changes, and release-note-worthy
   internal changes.

Model contributors and integrators
----------------------------------

The current public model-extension API is not mature. Model configs live in
``scoutbot.model_store`` today, and future registry work is tracked in
``docs/superpowers/plans/2026-05-09-scoutbot-registry.md`` and :doc:`roadmap`.

New model integration work should include:

* artifact provenance
* integrity verification
* config schema tests
* CLI choice tests
* output compatibility checks
