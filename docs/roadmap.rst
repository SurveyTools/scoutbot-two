Roadmap
=======

Lightweight tracking for deferred work. Status values are ``planned``,
``in-progress``, or ``done``.

Current priorities
------------------

1. **Release and distribution** -- define the release/tagging checklist, make the repository
   public, then support Python 3.13 and document what would be needed for Python 3.14 readiness
   before revisiting package distribution.
2. **Quality and compatibility** -- finish the quality workflow hardening, add dependency-currency
   policy, and expand platform coverage beyond the current macOS-focused Core ML work.
3. **Core ML / performance investigations** -- use the latest M4 benchmark JSON, the upcoming M1
   JSON, and broader observability work to explain the remaining gap and decide what additional
   instrumentation is worth keeping.

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

The base PyTorch pipeline should work beyond macOS, but the repository has no stated support level
or CI coverage for Windows. Add a lightweight cross-platform smoke test path so non-Core ML code
stays honest on other operating systems.

GPU / CUDA guidance
~~~~~~~~~~~~~~~~~~~

``--device`` is documented, but there is no explicit support statement or validation path for CUDA
workflows. Add guidance that states what is supported, what is best-effort, and how to verify GPU
execution when a CUDA environment is available.

Dependency and maintenance policy
---------------------------------

**Status:** planned

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
4. Security / dependency scanning -- ``uv audit --locked`` on ``uv.lock`` (done;
   ``pip-audit`` is not used). ``bandit`` remains deferred.
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

The next investigation pass should be broader than Core ML alone:

* compare the latest M4 benchmark JSON against the upcoming M1 JSON and document the actual
  differences instead of relying on memory
* break out image load/decode, tiling, model fetch/load, per-tile inference, post-processing,
  serialization, and error handling so the benchmark can explain where wall time goes
* decide which additional observability hooks should be kept for future backends beyond Core ML
* validate batching and parallelism ideas against explicit measurements, not just aggregate wall
  clock
* check whether the benchmark image set is representative of real survey imagery, or whether the
  current examples bias the timing story
* identify the minimum benchmark and runtime diagnostics needed to support future tuning work

The forward-looking observability question is therefore:

* what needs to be instrumented now so future model families can be measured consistently?
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
``scoutbot/loc/__init__.py`` and new model work should include artifact provenance, integrity
verification, config schema tests, CLI choice tests, and output compatibility checks.

Model registry evolution / model-family lineage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

The current registry plan should be treated as the first offline-first implementation step, not the
final registry contract. The longer-term design work belongs in the sibling ``pipeline-design``
project, where ``stage-3-ai-detection.md`` and ``schema-family.md`` already frame
``model_registry.json`` as survey-level infrastructure and leave the survey-level versus run-level
boundary open for refinement.

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
