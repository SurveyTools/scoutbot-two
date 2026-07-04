Security and Governance
=======================

This page covers model download policy, integrity verification, dependency and security scanning
posture, CI scanning rationale, and local data/privacy assumptions.

Model download policy
---------------------

Hosted model artifacts are fetched from the URLs configured in ``scoutbot.model_store.CONFIGS``. Current
PyTorch-backed models are hosted on Azure Blob Storage:

* ``v3`` uses ``yolov8-onecls.kaza.pt``
* ``v3-cls`` uses ``yolov8-cls.kaza.pt``

See :doc:`models` for the model and MD5 sidecar URLs.

Integrity verification
----------------------

Hosted ``.pt`` models use MD5 sidecar verification. Scoutbot downloads the sidecar, compares it
against any cached model, and downloads/rechecks the model when the cache is absent or mismatched.
If a downloaded model still does not match the sidecar, Scoutbot raises an error rather than using
the artifact.

Core ML artifacts
-----------------

Core ML ``.mlpackage`` bundles are not hosted as separate remote artifacts. They are exported
locally from verified source ``.pt`` models and cached in ``~/.cache/scoutbot/``.

Dependency and scanning posture
-------------------------------

Dependencies and tool configuration are managed through ``pyproject.toml``. The quality workflow
runs two complementary security checks:

``uv audit --locked``
   Checks all installed packages in the locked dependency graph against the OSV vulnerability
   database. Catches known CVEs in third-party dependencies. Waivers (e.g. CVEs with no
   available fix) are recorded in ``[tool.uv.audit]`` in ``pyproject.toml`` with an explanation.

``bandit -c pyproject.toml -r scoutbot/ benchmarks/``
   Scans the project's own source code for common Python security mistakes (insecure hash
   usage, dangerous deserialisation, shell injection, etc.). ``tests/`` is excluded because
   pytest assertions are not a security concern there. B101 (``assert_used``) is skipped
   globally — production asserts are internal invariants, not security gates.

CodeQL removal rationale
------------------------

The CodeQL workflow was removed because uploading code scanning results requires GitHub Advanced
Security/code scanning features that are unavailable for this repository. Project-owned quality
checks now run in ``.github/workflows/quality.yml``.

Data and privacy assumptions
----------------------------

Scoutbot inference runs locally. Scoutbot itself does not upload image data during inference.
Users are responsible for the privacy, storage, network, and access-control behavior of any
surrounding pipeline or infrastructure where Scoutbot is embedded.

CI posture
----------

``testing.yml`` runs fast/offline tests across supported Python versions. ``quality.yml`` runs
Ruff formatting/linting, scoped mypy, ``uv audit --locked``, and a Bandit source-code security
scan. Slow tests that download models or run real inference are kept out of the default CI test
path.
