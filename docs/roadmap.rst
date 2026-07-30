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

2. **Dill-free model Azure fallback upload + failover validation**

   * **Overall priority: high — reliability gap in flight.** The re-serialized (dill-free)
     KAZA models are hosted and validated on **GCS (primary)**, but **Azure storage was
     unavailable** when the dill-removal work landed, so the dated artifacts
     (``yolov8-onecls.kaza.20260729.pt`` / ``yolov8-cls.kaza.20260729.pt`` + ``.md5`` sidecars)
     and ``INDEX.txt`` are **not yet uploaded** to the Azure ``localizer`` container.
   * **Consequence:** the Azure fallback URLs in ``scoutbot/model_store.py`` ``CONFIGS`` point at
     objects that do not exist yet, so failover is temporarily non-functional — redundancy is
     reduced (a GCS outage before this is closed would break model fetch), though normal
     operation on GCS is unaffected.
   * **To close:** when Azure is available, upload the four files + ``INDEX.txt`` to ``localizer``
     (same names as on GCS), then run Task 4 Step 2 in
     ``docs/plans/2026-07-29-dill-removal.md`` (isolated clean-room fetch with GCS mocked
     unreachable, asserting Azure serves both configs with the pinned hashes).

3. **Pipeline integration stability**

   * define the output schema compatibility policy for pipeline consumers
   * finish the ``loc`` wrapper removal without breaking compatibility
   * continue shaping the survey-level registry contract in ``pipeline-design``

4. **Model registry follow-on**

   * add ``scoutbot unregister NAME`` and ``scoutbot fetch --force`` (see `Registry model removal`_)
   * define the dependency currency policy for ``ultralytics``, ``sahi``, and ``torch``

5. **Quality workflow — v2 scope**

   * coverage enforcement, expanded mypy, Ruff rule expansion, macOS CI runner for Core ML

6. **Performance and parallel execution**

   * **Overall priority: high.** ``batch_resumable(processes=N)`` has landed: resume-safe
     single-writer output, ``forkserver`` workers, bounded-wait fail-fast, and machine-readable
     run summaries. The emphasis now shifts to validating it on real survey workloads (see
     `Linux / CUDA multiprocess batch mode`_).
   * **Internal priority 1:** validate the Linux/CUDA implementation on real survey workloads:
     sequential vs threads vs processes, ``processes=2/4/8/12``, sparse vs dense images, and
     Matetsi-scale end-to-end timing.
   * **Internal priority 2:** turn those measurements into tuning guidance: process-count sizing,
     VRAM/model-copy cost, failure/re-run behavior, and interpretation of ``sahi+post``,
     ``per_worker_counts``, and ``inference_duty_per_worker``.
   * **Internal priority 3:** spec and implement per-image benchmark records: phase timings,
     image metadata, tile count, detection count, density indicators, and warning counts.
   * **Internal priority 4:** validate whether the benchmark image set represents real survey
     imagery; add dense-detection and survey-subset cases if the current examples bias timings.
   * **Internal priority 5:** correlate macOS Core ML runs with external ANE / Instruments
     observations because ``compute_units_actual`` is still unavailable from the runtime.
   * **Internal priority 6:** spec a second performance phase for queue/backpressure behavior,
     worker lifecycle and retry policy, process-count autotuning, SAHI/NMS bottlenecks, and
     possible within-image tiling concurrency.

7. **Long-term model ecosystem**

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

Dependency currency policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ultralytics``, ``sahi``, and ``torch`` move quickly. The roadmap should define how dependency
updates are absorbed, how breaking changes are evaluated, and what gets pinned versus left to
track upstream. The recent ``ultralytics`` upgrade (8.2.103 → ≥8.3) is the first instance of
a forced bump driven by a platform compatibility requirement (numpy 2.x / Python 3.13 Windows).

One open tension the policy should resolve: ``ultralytics`` 8.4.90 declares a runtime
``numpy<=2.3.5`` requirement, but the lock resolves ``numpy`` 2.5.1 on Python 3.13 (pinned up by
the windows-smoke floor). ultralytics warns and attempts an auto-downgrade (which fails
harmlessly — no ``pip`` in the venv) then proceeds; export and inference succeed. A future
ultralytics bump that raises the numpy ceiling clears the warning.

Quality workflow -- v2 scope
----------------------------

**Status:** planned

**Reference:** ``docs/plans/2026-07-02-quality-workflow-modernization.md``
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

**Reference:** ``docs/plans/2026-07-02-documentation-reorganization.md``

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

The July 2026 M1 / M4 comparison is now documented in :doc:`performance`. It confirms that the
M4 is consistently faster, but also that the full-pipeline Core ML gap is moderated by serial
Scoutbot orchestration and post-processing. The next investigation pass should therefore be
broader than Core ML alone:

* break out image load/decode, tiling, model fetch/load, per-tile inference, post-processing,
  serialization, and error handling so the benchmark can explain where wall time goes
* record per-image and per-tile distributions (p50 / p90 / p95 / max), not only aggregate totals
* record image workload metadata: dimensions, file size, tile count, detection count,
  post-processing / NMS warning count, and coarse density bucket
* validate batching and parallelism ideas against explicit measurements, not just aggregate wall
  clock
* check whether the benchmark image set is representative of real survey imagery, or whether the
  current examples bias the timing story
* pair macOS Core ML runs with external hardware observations (Activity Monitor Neural Engine,
  Instruments, or ``powermetrics`` where practical), because ``compute_units_actual`` remains
  ``null`` in current benchmark output
* identify the minimum benchmark and runtime diagnostics needed to support future tuning work

The forward-looking runtime question is therefore:

* what needs to be measured now so future model families can be compared consistently?
* which batch, queue, and concurrency metrics are worth retaining if parallel tile processing is
  added later?
* what image-set metadata is needed so benchmark results can be compared to real survey workloads
  rather than a narrow demo corpus?

Per-image benchmark records
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

**Overall priority:** high within the performance track, now that the resume-safe
``processes=N`` implementation has landed.

The most useful next observability increment is a machine-readable per-image record emitted by
the benchmarks. Each record should include:

* filepath and stable image identifier
* dimensions and file size
* tile count and timed model invocation count
* detection count and coarse density bucket
* phase timings for image load/decode, slice generation, per-tile inference, post-processing /
  NMS, serialization, and unattributed remainder
* distribution-friendly timings (per-image wall time and per-tile inference samples, so p50 /
  p90 / p95 / max can be computed)
* warning counters for NMS / post-processing pressure
* backend, device, model provenance, and compute-unit fields where applicable

This should support three use cases with the same data shape: M1 / M4 Core ML comparisons,
Linux/CUDA concurrency tuning, and future agentic pipeline tuning.

Benchmark corpus design
~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The current ``examples/`` corpus is useful for smoke tests but should not be assumed
representative of production survey workloads. Define a small benchmark corpus with explicit
categories:

* current examples for continuity
* sparse real survey images
* dense real survey images
* large / high-resolution edge cases
* known NMS-heavy frames

The corpus metadata should make benchmark results comparable across machines and future model
families without relying on informal memory of which images were used.

Relatedly, ``benchmarks/benchmark_coreml.py`` and ``benchmark_concurrency.py`` do not record the
ML-stack versions (torch, torchvision, ultralytics, coremltools, numpy) in their
``--summary-json`` output, so a saved benchmark cannot be tied to the stack that produced it. Add
version metadata to the summary schema alongside the corpus metadata.

Linux / CUDA multiprocess batch mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** in-progress (implementation landed; validation remaining)

**Overall priority:** high. The CUDA benchmark showed material survey-scale throughput gains, and
the resume-safe production implementation has now landed; the remaining work is validating it on
real survey workloads and turning that into tuning guidance.

``benchmarks/benchmark_concurrency.py`` now gives the parallelism question above explicit CUDA
measurements. On an RTX 5070 the ``v3`` pipeline is roughly 55% GPU inference / 45% single-threaded
CPU (SAHI tiling + NMS), which leaves the GPU near 30% utilization on a full survey. Cross-image
multiprocessing recovers about 2x (roughly 9 hours instead of ~20 on the full Matetsi survey);
threading is GIL-limited to about 22%. See :doc:`performance` for the full finding. This is a CUDA
observation, distinct from the Core ML M1 / M4 gap.

``batch_resumable(processes=N)`` / ``--processes N`` (see :doc:`usage` and :doc:`configuration`)
now provides first-class support for this mode. It uses a **write-integrity / resume design**: a
single writer in the parent process appends each completed result to the JSONL file in completion
order, while ``forkserver``-launched workers each hold their own model copy and never write to the
file directly. A lost or hung worker aborts the run fail-fast once it misses a bounded per-result
deadline (``SCOUTBOT_MP_RESULT_TIMEOUT``), and re-running the same command resumes the remaining
work. Every run emits a machine-readable summary (``summary_json`` / ``--summary-json``) with
worker-balance and phase-timing fields.

Internal priorities:

1. land the resume-safe implementation and summary JSON output -- done
2. validate real survey runs against sequential, threaded, and multiprocess baselines
3. document practical defaults for ``processes=N`` by GPU memory and model-copy cost
4. retain observability for worker balance, ``sahi+post``, inference duty, errors, and skipped
   resume records
5. defer within-image / per-tile concurrency to a separate experiment after cross-image mode is
   stable

Model registry / model integration
----------------------------------

**Status:** in-progress

**Reference:** ``docs/plans/2026-05-09-scoutbot-registry.md``

The local offline-first registry is implemented and in production use. ``scoutbot fetch``
auto-registers downloaded models; ``scoutbot models`` / ``scoutbot models info NAME`` list
registry entries; ``scoutbot register`` registers any local ``.pt`` file. All inference
commands accept ``--model NAME`` as an alternative to ``--config``; JSONL output from
``batch-resumable`` includes a ``model`` provenance block. Model downloads are resilient
across multiple sources (GCS primary, Azure fallback) with automatic failover.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
* multi-source model fetch: two-provider hosting with automatic failover (later reordered to
  GCS-primary / Azure-fallback), per-provider provenance recording, increased chunk size
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
* torch 2.13 ML-stack upgrade: ``torch==2.13.0`` / ``torchvision==0.28.0`` / ``setuptools==83.0.0``
  locked, the temporary ``uv audit`` ignores removed, and the graph audits with zero advisories;
  validated locally against the slow inference suite, MPS, and Core ML export parity (CUDA left to
  hardware availability). Established the macOS 14+ Apple Silicon floor
* dill removal: the two KAZA models re-serialized dill-free (identical weights) with the current
  Ultralytics/torch stack, re-hosted GCS-first under date-tagged ``yolov8-*.kaza.20260729.pt``
  filenames with their MD5 pinned in ``CONFIGS``, and the runtime ``dill`` dependency dropped;
  validated by the full slow suite running with ``dill`` uninstalled. The GCS-primary upload and
  failover are validated; the **Azure fallback upload + failover validation remains open** and is
  tracked as a current-priority item above

These remain reflected in the project history and in the implementation notes / changelog, but they
no longer need to occupy active roadmap space.
