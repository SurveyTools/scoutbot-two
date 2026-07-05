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

2. **Raw user onboarding and model resolution**

   * make the first-run fetch-before-inference flow explicit
   * document the registry-backed ``--model`` path and how it differs from ``--config``
   * keep ``README.md``, ``docs/usage.rst``, ``docs/installation.rst``, and ``docs/models.rst`` aligned with the supported workflow

3. **Pipeline integration stability**

   * define the output schema compatibility policy for pipeline consumers
   * keep per-run provenance and model resolution behavior stable
   * finish the model-store / ``loc`` staged cleanup without breaking compatibility
   * continue shaping the survey-level registry contract in ``pipeline-design``

4. **Quality, compatibility, and platform coverage**

   * keep the quality workflow aligned with ``pyproject.toml`` and CI
   * maintain the dependency-currency policy and security scanning posture
   * support Python 3.13 now and keep 3.14 readiness tracked
   * expand platform coverage beyond the current macOS-focused Core ML work
   * finish the Windows and cross-platform smoke coverage

5. **Developer tooling cleanup**

   * finish the ``visualise.py`` relocation and rewrite into ``tools/``
   * keep the docs and quality gates aligned with the new tool location
   * retire the root-level utility once the new entry point is the source of truth

6. **Performance and model-ecosystem evolution**

   * compare the latest M4 benchmark JSON against the upcoming M1 JSON and explain the gap
   * validate the benchmark image set and the useful runtime diagnostics for future backends
   * keep the benchmark and runtime instrumentation minimal but useful for batching and parallelism
   * evolve the model-family / alias / lineage contract in ``pipeline-design``

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
7. confirm Python 3.13 support is actually in place, with CI and packaging aligned to the supported
   interpreter floor / ceiling
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

The public release path should treat Python 3.13 as the supported target now. Python 3.14 remains
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

**Status:** planned

Output schema stability
~~~~~~~~~~~~~~~~~~~~~~~

The detection dict format (``l``, ``c``, ``x``, ``y``, ``w``, ``h``) is relied on by pipeline
integrators, but the roadmap does not yet state whether that schema is stable across releases. Add
an explicit compatibility policy and versioning note before making further output changes.

Windows / cross-platform CI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** in-progress (smoke coverage added)

A ``windows-smoke`` CI job now runs the fast/offline test suite on Python 3.13 under Windows,
providing basic non-Core ML coverage without claiming full platform parity. Core ML configs
(``v3-coreml``, ``v3-cls-coreml``) remain macOS-only. Further cross-platform testing may expand
to additional Python versions and runners as needed.

GPU / CUDA guidance
~~~~~~~~~~~~~~~~~~~

**Status:** done

CUDA support is documented as best-effort for NVIDIA systems. Users can verify GPU detection
with the verification command in :doc:`installation` (Windows NVIDIA section). CUDA is not
required and not CI-tested; users on CPU-only machines will see ``torch.cuda.is_available()``
return ``False``, which is expected and not an error.

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
track upstream.

Quality workflow -- v2 scope
----------------------------

**Status:** planned

**Reference:** ``docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md``
(V2 Scope section)

1. Coverage enforcement (``--cov-fail-under`` gated on meaningful branch coverage,
   not raw line count; branch coverage via ``--cov-branch``)
2. Expand mypy to ``tests/`` and ``benchmarks/``; add targeted
   ``[[tool.mypy.overrides]]`` per library; revisit ``ignore_missing_imports = true``
3. Ruff rule expansion -- enable ``UP`` (restores deferred pyupgrade behaviour),
   ``B``, ``C4``, ``SIM`` one group per PR
4. Security / dependency scanning -- ``uv audit --locked`` for dependencies and
   ``bandit`` for source-code scanning, both in ``quality.yml`` (done).
5. macOS CI runner for Core ML path (expensive; consider self-hosted M-series)

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

Model registry / model integration
----------------------------------

**Status:** planned

**Reference:** ``docs/superpowers/plans/2026-05-09-scoutbot-registry.md``

Future model integration work should provide a registry mechanism for model updates and new model
integrations. Until that is mature, model configs remain internal implementation details in
``scoutbot.model_store`` and new model work should include artifact provenance, integrity
verification, config schema tests, CLI choice tests, and output compatibility checks.

Model-store / ``loc`` staged cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The current `scoutbot.loc` module is now a compatibility wrapper around
``scoutbot.model_store``. That wrapper should be treated as temporary technical debt, not a new
home for model policy.

The staged cleanup should:

* move remaining internal imports to ``scoutbot.model_store`` instead of ``scoutbot.loc``
* update tests and docs that still describe ``scoutbot.loc`` as the owner of the catalog
* keep the wrapper in place until the registry and model-store refactors no longer depend on it
* retire the wrapper once the remaining call sites are migrated and no public compatibility need
  remains

This cleanup should stay coupled to the broader registry/model-store work so the eventual
deprecation does not break the offline-first model path or the existing test surface prematurely.

Model registry evolution / model-family lineage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The current registry plan should be treated as the first offline-first implementation step, not the
final registry contract. The longer-term design work belongs in the sibling ``pipeline-design``
project, where ``stage-3-ai-detection.md`` and ``schema-family.md`` already frame
``model_registry.json`` as survey-level infrastructure and leave the survey-level versus run-level
boundary open for refinement.

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

Future registry work should define how the registry represents:

* model families and canonical family identifiers
* parent/child relationships between base models and fine-tuned variants
* versioned aliases for the same family across training or export iterations
* compatibility metadata for Stage 3 consumers and pipeline integrations
* the split between survey-level registry facts and per-run provenance in detection outputs

The current ScoutBot registry plan should continue to focus on offline fetch, local provenance, and
CLI plumbing. The richer family/lineage contract should be designed as a follow-on spec so future
model integrations can grow without reworking the registry format from scratch.

Completed work archive
----------------------

The following completed items were removed from the active roadmap so the page can focus on open
work:

* benchmark hardening for ``benchmarks/benchmark_coreml.py``
* CI coverage reporting via the GitHub Actions job summary
* performance observability spec and Core ML compute-unit metadata follow-up

These remain reflected in the project history and in the implementation notes / changelog, but they
no longer need to occupy active roadmap space.
