Security and Governance
=======================

This page covers model download policy, integrity verification, dependency and security scanning
posture, CI scanning rationale, and local data/privacy assumptions.

Model download policy
---------------------

Hosted model artifacts are fetched from the sources configured in ``scoutbot.model_store.CONFIGS``.
PyTorch-backed models are hosted on two providers tried in order:

1. **Google Cloud Storage** (``storage.googleapis.com/rebel-pipeline-public-models``) — primary
2. **Azure Blob Storage** (``scoutmodels.blob.core.windows.net``) — fallback

* ``v3`` uses ``yolov8-onecls.kaza.20260729.pt``
* ``v3-cls`` uses ``yolov8-cls.kaza.20260729.pt``

If the primary source is unreachable or resets the connection mid-transfer, Scoutbot automatically
retries from the fallback with a logged warning. No user action is required. See :doc:`models` for
the full URL listing per model and provider.

Integrity verification
----------------------

Both providers carry matching ``.md5`` sidecar files, and the expected MD5 for each PyTorch model
is additionally **pinned in code** in ``scoutbot.model_store`` ``CONFIGS`` (the ``md5`` field).
On each download attempt Scoutbot fetches the sidecar from the same source as the model, verifies
the local MD5 on completion, and asserts the hosted sidecar's MD5 equals the pinned value — so a
bad or partial re-upload fails loudly rather than being served. The pin is also authoritative for
the cache: a cached file whose hash does not match the pinned value is treated as stale and
re-fetched, which is how a client upgrading past a model re-serialization is moved off the old
artifact. Pinning the hash in code is stronger than trusting a sidecar hosted next to the model;
signed manifests or SHA-256 remain possible future hardening (see :doc:`roadmap`). A source that
passes verification is registered in the local registry; subsequent runs resolve from cache with
no network access.

Core ML ``.mlpackage`` bundles are not hosted as separate remote artifacts. They are exported
locally from verified source ``.pt`` models and cached in ``~/.cache/scoutbot/``.

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
   database. Catches known CVEs in third-party dependencies. Temporary audit exceptions, if any
   are accepted, live directly on the ``uv audit`` command in
   ``.github/workflows/quality.yml`` with comments naming the advisory, rationale, and planned
   removal path. ``uv audit`` does not read a project ``[tool.uv.audit]`` table in this
   repository.

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
