Roadmap
=======

Lightweight tracking for deferred work. Status values are ``planned``,
``in-progress``, or ``done``.

Current priorities
------------------

1. **Release and contributor readiness**

   * define the release/tagging checklist and release-note hygiene
   * make the repository public and enable branch protection
   * keep the hosted docs site healthy on ``main``
   * publish the contributor-facing docs and workflow needed for external contributors

2. **Pipeline integration stability**

   * define the output schema compatibility policy for pipeline consumers
   * finish the ``loc`` wrapper removal without breaking compatibility
   * continue shaping the survey-level registry contract in ``pipeline-design``

3. **Model registry follow-on**

   * add ``scoutbot unregister NAME`` and ``scoutbot fetch --force`` (see `Registry model removal`_)
   * define the dependency currency policy for ``ultralytics``, ``sahi``, and ``torch``

4. **Quality workflow — v2 scope**

   * coverage enforcement, expanded mypy, Ruff rule expansion, macOS CI runner for Core ML

5. **Performance and parallel execution**

   * **Overall priority: high.** Finish ``batch_resumable(processes=N)`` as the next
     production performance feature, with resume-safe single-writer output and machine-readable
     run summaries (see `Linux / CUDA multiprocess batch mode`_).
   * **Internal priority 1:** validate the Linux/CUDA implementation on real survey workloads:
     sequential vs threads vs processes, ``processes=2/4/8/12``, sparse vs dense images, and
     Matetsi-scale end-to-end timing.
   * **Internal priority 2:** turn those measurements into tuning guidance: process-count sizing,
     VRAM/model-copy cost, failure/re-run behavior, and interpretation of ``sahi+post``,
     ``per_worker_counts``, and ``inference_duty_per_worker``.
   * **Internal priority 3:** compare the M4 benchmark JSON against the M1 JSON and document the
     Core ML gap once the M1 JSON is available.
   * **Internal priority 4:** validate whether the benchmark image set represents real survey
     imagery; add dense-detection and survey-subset cases if the current examples bias timings.
   * **Internal priority 5:** spec a second performance phase for queue/backpressure behavior,
     worker lifecycle and retry policy, process-count autotuning, SAHI/NMS bottlenecks, and
     possible within-image tiling concurrency.

6. **Long-term model ecosystem**

   * define survey-level model-family IDs, aliases, parent/child fine-tunes, and compatibility
     metadata in ``pipeline-design``
   * keep Scoutbot's local model registry aligned with that future survey-level contract

Work triage
-----------

Use the roadmap to decide whether a topic needs a spec, a plan, or direct implementation:

* **Write a spec** when the work defines a contract, policy, or long-lived behavior.
  That includes model registry shape, model-family IDs, aliases, parent/child fine-tunes,
  compatibility metadata, output schema stability, interpreter support policy, and any
  observability data that future tooling will consume.
* **Write a plan** when the shape is known but the work is multi-step or crosses files.
  That includes ``loc`` / ``model_store`` cleanup, docs reorganization, hosted docs workflow,
  quality workflow changes, Windows / cross-platform CI, ``visualise.py`` relocation, and
  benchmark or runtime diagnostics work.
* **Code now** when the decision is already made and the change is local.
  That includes stale docs updates, small CI YAML fixes, tests for already-specified behavior,
  and cleanup refactors with no new external contract.

If the question is "what is the contract?", write a spec.
If the question is "what files change and in what order?", write a plan.
If neither question matters, just implement the change.

Release and distribution
------------------------

**Status:** planned

Public release checklist
~~~~~~~~~~~~~~~~~~~~~~~~

The repository should move through the following release steps together:

1. choose and document the next version tag, including the tagging convention and the release
   branch / commit state that will be tagged
2. make the repository public
3. enable main-branch protection in GitHub settings at the same time as the public switch
4. verify the GitHub Pages docs site still builds on ``main``
5. confirm the release notes / changelog entry for the tag and keep the changelog entry aligned
   with the tagged release
6. publish the contributor-facing docs needed for external contributors, including the documented
   documentation-pass rule and the public contribution workflow
7. ✓ Python 3.13 support confirmed: CI runs the full fast suite on Python 3.13 under both
   macOS and Windows; the 179-test slow suite passes on macOS 3.13.
8. record the Python 3.14 readiness checklist so the next interpreter bump is scoped before it
   becomes urgent
9. revisit PyPI publishing once namespace ownership is clarified

PyPI publishing remains deferred until the old Wild Me Scoutbot namespace and ownership path are
clarified. That should stay a separate decision from the public-repo move itself.

Release-note hygiene should include a short pre-tag check that the changelog entry:

* names the release boundary clearly
* calls out user-visible changes, compatibility changes, and workflow changes
* stays synchronized with the eventual Git tag and public announcement

Python interpreter readiness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.13 is the confirmed supported target. The full fast suite and slow suite both pass on
macOS 3.13; the Windows smoke job runs on Python 3.13 and passes. Python 3.14 remains
awareness-only until a later readiness review says otherwise.

Python 3.14 readiness checklist:

* expand the CI/test matrix to include 3.14
* update packaging metadata once the interpreter bump is real
* confirm dependency compatibility for the packages most likely to lag first:
  ``torch``, ``ultralytics``, ``sahi``, and ``coremltools``
* rerun the dependency-stub / type-check pass after the matrix and metadata are ready

3.14 does not need to be supported now, but it should be explicitly tracked so the next bump is
deliberate rather than reactive.

Compatibility and platform support
-----------------------------------

Output schema stability
~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The detection dict format (``l``, ``c``, ``x``, ``y``, ``w``, ``h``) is relied on by pipeline
integrators, but the roadmap does not yet state whether that schema is stable across releases. Add
an explicit compatibility policy and versioning note before making further output changes.

Windows / cross-platform CI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** done

The ``windows-smoke`` CI job runs the full fast/offline test suite on Python 3.13 under Windows.
The package import chain (sahi, torch, numpy) is deferred via PEP 562 ``__getattr__`` in
``tile_batched/__init__.py`` so collection does not crash on Windows when numpy is not yet
initialized. An import regression test (``tests/test_imports.py``) guards this path. Core ML
configs (``v3-coreml``, ``v3-cls-coreml``) remain macOS-only.

GPU / CUDA guidance
~~~~~~~~~~~~~~~~~~~

**Status:** done

CUDA support is documented as best-effort for NVIDIA systems. Users can verify GPU detection
with the verification command in :doc:`installation` (Windows NVIDIA section). CUDA is not
required and not CI-tested; users on CPU-only machines will see ``torch.cuda.is_available()``
return ``False``, which is expected and not an error.

Windows parallel execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

**Overall priority:** medium. Windows should remain supported for base use, but it is not the
first high-performance parallel execution target.

Current support target:

* keep ``pipeline()``, ``batch()``, and ``batch_resumable(processes=1)`` working under the
  Windows smoke job
* keep Windows NVIDIA / CUDA guidance documented as best-effort
* do not block the Linux/CUDA multiprocess implementation on Windows multiprocessing semantics

Future Windows parallel work needs its own design because Windows uses ``spawn`` rather than
``forkserver``. That changes worker startup, import safety, model load cost, CUDA process
behavior, path handling, and process cleanup. Treat Windows ``processes=N`` as a separate
compatibility and performance spike before making support claims.

macOS parallel execution
~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

**Overall priority:** medium-low until Linux/CUDA measurements are complete. macOS parallelism
is worth testing, but it should not be bundled into the first production multiprocess mode.

The macOS path should be a dedicated ``spawn``-based spike covering:

* ``pt-mps`` and ``coreml`` backends
* ``processes=2/4/8`` on representative M-series machines, including low-memory and high-memory
  cases where available
* sparse and dense survey images
* startup/warmup cost separated from steady-state throughput
* Core ML / ANE, MPS, and memory-pressure behavior under multiple worker processes

Only productize macOS ``processes=N`` if the spike shows stable speedup, clean resume behavior,
and understandable resource limits. Until then, document multiprocess batch mode as
Linux/CUDA-first with macOS parallel execution planned as a separate validation track.

Dependency and maintenance policy
---------------------------------

**Status:** planned

Model re-export / dill removal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The KAZA ``.pt`` model files were serialized with an older Ultralytics version that embedded
``dill`` pickle references. Loading them at runtime requires ``dill`` as a dependency even though
the inference code itself has no need for it. ``dill>=0.4.0`` has been re-added to
``[project.dependencies]`` as a temporary shim.

To remove ``dill`` properly:

1. Download the current KAZA ``.pt`` files from Azure Blob Storage.
2. Load each model with ``dill`` available (current environment is fine).
3. Re-export using ``model.save()`` or extract and re-save the state dict with plain
   ``torch.save``, targeting a Ultralytics version that serializes without ``dill``.
4. Verify MD5s, update ``scoutbot/model_store.py`` ``CONFIGS`` with the new hashes and URLs,
   and re-upload the artifacts to Azure.
5. Remove ``dill`` from ``[project.dependencies]`` and this roadmap entry.

Dependency currency policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ultralytics``, ``sahi``, and ``torch`` move quickly. The roadmap should define how dependency
updates are absorbed, how breaking changes are evaluated, and what gets pinned versus left to
track upstream. The recent ``ultralytics`` upgrade (8.2.103 → ≥8.3) is the first instance of
a forced bump driven by a platform compatibility requirement (numpy 2.x / Python 3.13 Windows).

Quality workflow -- v2 scope
----------------------------

**Status:** planned

**Reference:** ``docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md``
(V2 Scope section)

Quality workflow v1 is complete: Ruff formatting and lint, scoped mypy on ``scoutbot/``,
``bandit`` source-code scanning, and ``uv audit --locked`` dependency scanning all run in CI.

Remaining v2 work:

1. Coverage enforcement (``--cov-fail-under`` gated on meaningful branch coverage,
   not raw line count; branch coverage via ``--cov-branch``)
2. Expand mypy to ``tests/`` and ``benchmarks/``; add targeted
   ``[[tool.mypy.overrides]]`` per library; revisit ``ignore_missing_imports = true``
3. Ruff rule expansion — enable ``UP`` (restores deferred pyupgrade behaviour),
   ``B``, ``C4``, ``SIM`` one group per PR
4. macOS CI runner for Core ML path (expensive; consider self-hosted M-series)

The quality work and the dependency currency policy above should stay aligned so breaking
dependency upgrades are caught in CI instead of surfacing in release work.

Documentation reorganization
----------------------------

**Status:** done

**Reference:** ``docs/superpowers/plans/2026-07-02-documentation-reorganization.md``

Documentation was reorganized around reader intent, separating user guide, reference, project,
development, security, roadmap, and changelog content into discrete Sphinx pages. An ongoing
documentation-pass rule was established for future agent and contributor work.

GitHub Pages is now the canonical hosted docs site. The repository keeps the Sphinx source and
builds HTML on merge to ``main``. Read the Docs has been retired and its config file removed.

Test coverage
-------------

**Status:** done

Fast-suite coverage raised from 58% to 92%+ by adding unit tests with mocked SAHI,
``requests``, and ``coremltools`` dependencies. All tests use ``unittest.mock`` to avoid
network or model-file dependencies.

Remaining uncovered lines (all intentional or deferred):

* ``example()`` -- requires network and model download; covered only by ``@pytest.mark.slow``
* ``if __name__ == '__main__':`` CLI guard -- untestable without subprocess
* ``scoutbot/scoutbot.py:205-208`` -- ``batch`` CLI ``--output`` file-write branch
* ``scoutbot/__init__.py:246-250`` -- ``batch()`` ``show_progress=True`` tqdm postfix update
  (deferred to V2 scope; requires a real or patched tqdm iterator)

Core ML / performance investigations
------------------------------------

**Status:** in-progress

**Reference:** ``docs/coreml.rst`` (benchmark caveats)

The next investigation pass should be broader than Core ML alone, but it is now an
investigation and measurement track rather than a separate observability-spec track:

* compare the latest M4 benchmark JSON against the upcoming M1 JSON and document the actual
  differences instead of relying on memory
* break out image load/decode, tiling, model fetch/load, per-tile inference, post-processing,
  serialization, and error handling so the benchmark can explain where wall time goes
* validate batching and parallelism ideas against explicit measurements, not just aggregate wall
  clock
* check whether the benchmark image set is representative of real survey imagery, or whether the
  current examples bias the timing story
* identify the minimum benchmark and runtime diagnostics needed to support future tuning work

The forward-looking runtime question is therefore:

* what needs to be measured now so future model families can be compared consistently?
* which batch, queue, and concurrency metrics are worth retaining if parallel tile processing is
  added later?
* what image-set metadata is needed so benchmark results can be compared to real survey workloads
  rather than a narrow demo corpus?

Linux / CUDA multiprocess batch mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** in-progress

**Overall priority:** high. This is the next production performance feature because the CUDA
benchmark shows material survey-scale throughput gains and the current implementation plan
directly addresses resume integrity.

``benchmarks/benchmark_concurrency.py`` now gives the parallelism question above explicit CUDA
measurements. On an RTX 5070 the ``v3`` pipeline is roughly 55% GPU inference / 45% single-threaded
CPU (SAHI tiling + NMS), which leaves the GPU near 30% utilization on a full survey. Cross-image
multiprocessing recovers about 2x (roughly 9 hours instead of ~20 on the full Matetsi survey);
threading is GIL-limited to about 22%. See :doc:`performance` for the full finding. This is a CUDA
observation, distinct from the Core ML M1 / M4 gap.

The follow-on is to finish first-class ``batch_resumable(processes=N)`` support. That work is
gated on a **write-integrity / resume design**: the current single-file JSONL append is not safe
for concurrent writers, so production mode uses a single-writer orchestration path and explicit
device / VRAM guidance for the process count.

Internal priorities:

1. land the resume-safe implementation and summary JSON output
2. validate real survey runs against sequential, threaded, and multiprocess baselines
3. document practical defaults for ``processes=N`` by GPU memory and model-copy cost
4. retain observability for worker balance, ``sahi+post``, inference duty, errors, and skipped
   resume records
5. defer within-image / per-tile concurrency to a separate experiment after cross-image mode is
   stable

Model registry / model integration
----------------------------------

**Status:** in-progress

**Reference:** ``docs/superpowers/plans/2026-05-09-scoutbot-registry.md``

The local offline-first registry is implemented and in production use. ``scoutbot fetch``
auto-registers downloaded models; ``scoutbot models`` / ``scoutbot models info NAME`` list
registry entries; ``scoutbot register`` registers any local ``.pt`` file. All inference
commands accept ``--model NAME`` as an alternative to ``--config``; JSONL output from
``batch-resumable`` includes a ``model`` provenance block. Model downloads are resilient
across multiple sources (Azure primary, GCS fallback) with automatic failover.

Remaining registry work is tracked in the sub-sections below.

Registry model removal
~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

There is currently no CLI command to remove a model from the local registry or cache. To force
a re-fetch today, a user must manually delete the cached file and the registry entry:

.. code-block:: bash

   rm ~/.cache/scoutbot/<model-name>.pt
   rm ~/.rebel-workflow/model-registry.json   # or edit the JSON to remove one entry

The cache also contains Core ML ``.mlpackage`` bundles (directory trees, not single files),
which require ``rm -rf`` rather than ``rm``:

.. code-block:: bash

   rm -rf ~/.cache/scoutbot/<model-name>.mlpackage

Two commands should be added:

* ``scoutbot unregister NAME`` — removes the named entry from the registry JSON (does not
  delete the model file or bundle, in case it was registered manually from an existing path).
* ``scoutbot fetch --force`` — re-downloads ``.pt`` models and re-exports ``.mlpackage``
  bundles from remote sources, overwriting both the cache and the registry entry regardless
  of what the registry currently records.

These two commands cover the common cases: ``unregister`` for registry-only cleanup (e.g. a
stale path after moving the model file), and ``fetch --force`` for a full cache-and-registry
reset before a re-download. Both commands must handle ``.mlpackage`` bundle directories
correctly (recursive delete / recursive copy rather than single-file operations).

Model-store / ``loc`` staged cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The model catalog and fetch/export plumbing live in ``scoutbot/model_store.py``. The
``scoutbot.loc`` module is now a thin compatibility wrapper around ``model_store`` and should
be treated as temporary technical debt, not the canonical home for model policy.

The staged cleanup should:

* move remaining internal imports to ``scoutbot.model_store`` instead of ``scoutbot.loc``
* update tests and docs that still describe ``scoutbot.loc`` as the owner of the catalog
* retire the wrapper once the remaining call sites are migrated and no public compatibility need
  remains

This cleanup should stay coupled to the broader registry/model-store work so the eventual
deprecation does not break the offline-first model path or the existing test surface prematurely.

Model registry evolution / model-family lineage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The current registry is the first offline-first implementation step, not the final registry
contract. The longer-term design work belongs in the sibling ``pipeline-design`` project, where
``stage-3-ai-detection.md`` and ``schema-family.md`` already frame ``model_registry.json`` as
survey-level infrastructure and leave the survey-level versus run-level boundary open for
refinement.

Survey-level registry contract (v3+)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The future survey-level contract should define the canonical registry shape for the whole survey,
not just the local model cache. That contract should live in ``pipeline-design`` and should cover:

* canonical model-family identifiers that stay stable across exports and retraining runs
* parent/child lineage between base models and fine-tuned derivatives
* versioned aliases so the same family can be referenced by release, training, or export variant
* compatibility metadata for Stage 3 consumers and downstream pipeline integrations
* the split between survey-level registry facts and per-run provenance captured in detection
  outputs

ScoutBot should continue to own the offline-first local registry, fetch plumbing, and CLI
resolution. The survey-level artifact schema, validation rules, and stage-boundary semantics belong
in ``pipeline-design`` so future model families can evolve without forcing another schema rewrite
in this repository.

Completed work archive
----------------------

The following completed items were removed from the active roadmap so the page can focus on open
work:

* benchmark hardening for ``benchmarks/benchmark_coreml.py``
* CI coverage reporting via the GitHub Actions job summary
* performance observability spec and Core ML compute-unit metadata follow-up
* ``visualise.py`` relocation from repo root to ``tools/visualise.py``, full rewrite, and
  wiring into all quality gates (ruff, mypy, bandit, pytest)
* Windows smoke CI job and cross-platform import-chain fix (PEP 562 lazy loader for
  ``tile_batched``, numpy/torch deferred to ``perform_inference()``, import regression test)
* Python 3.13 support: full CI matrix coverage on macOS and Windows, 179-test slow suite
  verified on macOS 3.13
* multi-source model fetch: Azure primary with automatic GCS fallback, per-provider provenance
  recording, increased chunk size
* local model registry: offline-first fetch/register/resolve flow, ``scoutbot models`` and
  ``scoutbot register`` CLI, ``--model NAME`` flag, JSONL provenance block
* model store refactor: catalog and fetch/export plumbing moved from ``scoutbot.loc`` to
  ``scoutbot/model_store.py``; ``scoutbot.loc`` is now a compatibility wrapper
* quality workflow v1: Ruff formatting and lint, scoped mypy on ``scoutbot/``, bandit
  source-code scanning, ``uv audit --locked`` dependency scanning, all running in CI
* documentation reorganization: reader-intent Sphinx structure, GitHub Pages deployment,
  Read the Docs retired
* test coverage: fast-suite coverage raised from 58% to 92%+
* GPU / CUDA guidance documented as best-effort for NVIDIA systems

These remain reflected in the project history and in the implementation notes / changelog, but they
no longer need to occupy active roadmap space.
