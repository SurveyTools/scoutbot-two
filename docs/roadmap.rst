Roadmap
=======

Lightweight tracking for deferred work. Status values are ``planned``,
``in-progress``, or ``done``.

Benchmark hardening
-------------------

**Status:** planned

Two output cleanliness issues need to be fixed in ``benchmarks/benchmark_coreml.py``.

Missing ``coremltools`` is not clean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``coremltools`` is not installed, each family block prints an inline
``SKIPPED: ModuleNotFoundError: No module named 'coremltools'`` row mixed into
the results table. The benchmark should detect the missing module once at
startup, print a single clear message such as ``coreml variant skipped:
coremltools not installed -- run: uv sync --extra coreml``, and omit the Core ML
column from the results table entirely.

Third-party noise in output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several noisy lines appear during benchmark runs regardless of ``coremltools``
installation:

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

Quality workflow -- v2 scope
----------------------------

**Status:** planned

**Reference:** ``docs/superpowers/plans/2026-07-02-quality-workflow-modernization.md``
(V2 Scope section)

1. Coverage enforcement (``--cov-fail-under``, branch coverage)
2. Expand mypy to ``tests/`` and ``benchmarks/``; add targeted
   ``[[tool.mypy.overrides]]`` per library; revisit ``ignore_missing_imports = true``
3. Ruff rule expansion -- enable ``UP`` (restores deferred pyupgrade behaviour),
   ``B``, ``C4``, ``SIM`` one group per PR
4. Security / dependency scanning -- ``pip-audit`` on ``uv.lock``, tuned
   ``bandit``
5. macOS CI runner for Core ML path (expensive; consider self-hosted M-series)

Documentation reorganization
----------------------------

**Status:** done

**Reference:** ``docs/superpowers/plans/2026-07-02-documentation-reorganization.md``

Documentation was reorganized around reader intent, separating user guide, reference, project,
development, security, roadmap, and changelog content into discrete Sphinx pages. An ongoing
documentation-pass rule was established for future agent and contributor work.

Core ML / performance investigations
------------------------------------

**Status:** planned

**Reference:** ``docs/coreml.rst`` (benchmark caveats)

* Break out slicing, per-tile prediction, post-processing, and Core ML dispatch
  overhead in the benchmark
* Investigate parallel or batched tile inference for the Core ML path
* Confirm ANE engagement via Activity Monitor / ``coremltools`` compute-unit API
* M4 follow-up benchmark once pipeline overhead is reduced

Performance observability spec
------------------------------

**Status:** planned

The next performance investigation should specify structured benchmark observability before making
hardware-specific throughput claims. The spec should define phase timings for image discovery,
decode/load, slicing, per-tile model invocation count, per-tile model wall time, post-processing,
serialization, Core ML compute-unit selection, direct ``coremltools`` prediction timing, and
concurrent or batched tile inference experiments.

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
