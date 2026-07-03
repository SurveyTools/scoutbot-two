Roadmap
=======

Lightweight tracking for deferred work. Status values are ``planned``,
``in-progress``, or ``done``.

Current priorities
------------------

1. **Release and distribution** -- define the release/tagging checklist, resolve the Python 3.13
   ceiling, and plan package distribution carefully. PyPI publishing is likely deferred until the
   old project namespace and ownership are clarified.
2. **Quality and compatibility** -- finish the quality workflow hardening, add dependency-currency
   policy, and expand platform coverage beyond the current macOS-focused Core ML work.
3. **Core ML / performance investigations** -- now that the runtime hooks and structured benchmark
   are in place, use them to explain the remaining M1 / M4 gap and decide what further diagnostics
   are worth keeping.

Release and distribution
------------------------

**Status:** planned

PyPI publishing
~~~~~~~~~~~~~~~

There is no publish workflow yet. The old Wild Me Scoutbot project already occupied the PyPI
namespace, so reusing that name without resolving ownership would be confusing. Treat PyPI
publishing as deferred until the project name, namespace, and ownership path are clarified.

Python 3.13 support
~~~~~~~~~~~~~~~~~~~

The package metadata still declares ``requires-python = ">=3.10,<3.13"``, which blocks users whose
environments have already moved to Python 3.13. Track the compatibility work needed to lift that
ceiling, starting with the test matrix and any dependency constraints that still pin 3.12.

Release readiness
~~~~~~~~~~~~~~~~~

GitHub Pages now covers the docs site, but the project still needs a documented release path for
version tagging and release-note hygiene so the codebase can move from internal iteration to
public consumption cleanly. Package distribution details remain dependent on the PyPI namespace
decision above.

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

Benchmark hardening
-------------------

**Status:** done

``benchmarks/benchmark_coreml.py`` now emits structured phase timings, a JSON summary option,
clean skip handling when a variant cannot run, and benchmark-side suppression for the known
third-party warning / log noise described below.

Missing ``coremltools`` is not clean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``coremltools`` is not installed, the benchmark now detects the missing
module once at startup, prints a single clear message such as ``coreml variant
skipped: coremltools not installed -- run: uv sync --extra coreml``, and omits
the Core ML column from the results table entirely.

Third-party noise in output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark now suppresses the known noisy lines that appeared during runs
regardless of ``coremltools`` installation:

* **SAHI deprecation warning** -- ``sahi/utils/torch_utils.py:80: UserWarning:
  'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'`` appears
  once per family run. Suppress with ``warnings.filterwarnings``.
* **coremltools version warning** -- ``[coremltools] Torch version X.Y.Z has not
  been tested with coremltools`` is expected and already documented in
  ``docs/coreml.rst``; suppress in benchmark output.
* **coremltools SyntaxWarning** -- ``SyntaxWarning: invalid escape sequence
  '\_'`` from ``coremltools/optimize/torch/palettization/fake_palettize.py`` is
  a coremltools packaging bug; suppress with ``warnings.filterwarnings``.
* **"Loading ... for CoreML inference..."** log line -- emitted by ``scoutbot``
  during model load; interrupts the per-family header. Suppress or redirect to
  stderr during benchmark runs.

CI coverage reporting
---------------------

**Status:** in-progress

The codecov upload step was removed from ``.github/workflows/testing.yml`` because
no Codecov account exists; the upload was silently failing with a token error.

A GitHub Actions job summary step was added in its place: pytest runs with
``--cov-report=json`` and a small Python snippet writes the percentage and line
counts to ``$GITHUB_STEP_SUMMARY``, making coverage visible on every PR in the
Actions UI with no external account.

Remaining options for further coverage tooling:

1. **Codecov free tier** -- create a free account at `codecov.io`, add the repo,
   store ``CODECOV_TOKEN`` in GitHub Actions secrets, and restore the
   ``codecov/codecov-action`` step. Provides per-PR diff coverage annotations and
   a README badge. Requires an ongoing Codecov account.

2. **``--cov-fail-under``** -- enforce a minimum threshold in CI. Already listed
   in Quality workflow v2 scope below.

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
4. Security / dependency scanning -- ``pip-audit`` on ``uv.lock``, tuned
   ``bandit``
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

The structured benchmark makes the remaining work explicit:

* break out slicing, per-tile prediction, post-processing, and Core ML dispatch overhead in the
  benchmark summaries
* investigate parallel or batched tile inference for the Core ML path
* confirm ANE engagement via Activity Monitor / ``coremltools`` compute-unit API
* run the M4 follow-up benchmark once the pipeline overhead is better understood

Performance observability spec
------------------------------

**Status:** done

**Reference:** ``docs/superpowers/specs/2026-07-02-performance-observability-spec.md``

The structured observability model is now implemented in the runtime and benchmark. The spec
remains the canonical description of the event shape and the diagnostics the benchmark should
surface when Core ML work continues.

Model registry / model integration
----------------------------------

**Status:** planned

**Reference:** ``docs/superpowers/plans/2026-05-09-scoutbot-registry.md``

Future model integration work should provide a registry mechanism for model updates and new model
integrations. Until that is mature, model configs remain internal implementation details in
``scoutbot/loc/__init__.py`` and new model work should include artifact provenance, integrity
verification, config schema tests, CLI choice tests, and output compatibility checks.

Community contribution docs
---------------------------

**Status:** planned

When the project opens for community contributions, add the same documentation-pass requirement
to the public contributor guide, likely ``CONTRIBUTING.md`` or
an equivalent contributor page. Contributors should be expected to review and
update affected docs, changelog, and roadmap entries as part of any code, model,
CLI, configuration, workflow, security, or performance change.
