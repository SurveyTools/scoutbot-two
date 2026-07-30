# rw → osw Namespace Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebrand the ecosystem namespace in scoutbot-2026 from `rw` ("Rebel Workflow") to `osw` ("Overhead Survey Workflow") across env vars, the registry directory, the registry schema id, the GCS bucket, and public metadata — with a clean break and a hardened registry read.

**Architecture:** A clean rename (no backward-compat aliases) plus one net-new piece of logic: a registry **schema-guard** that validates the `schema` id family and major version on read. The GCS bucket (`osw-public-models`) is already live and verified, so the URL swap is a normal code change.

**Tech Stack:** Python, `uv`, Click CLI, pytest.

## Global Constraints

- Runtime: **always** `uv run` — never bare `python`/`pytest`/`ruff`/`mypy`.
- **No git in this plan.** Do not run `git add`/`commit`/`push`. Each task ends at a review checkpoint with a suggested commit message; the user stages and commits.
- **Pause after each task** for human review before starting the next (repo rule; overrides any skill's continuous-execution default).
- Test-first: write the failing test and run it red before implementing.
- **Clean break**: no `RW_*` / `~/.rebel-workflow` compatibility aliases; the old registry is orphaned (models remain in `~/.cache/scoutbot/`; first fetch re-registers under `~/.osw/`).
- Target names (verbatim): dir `~/.osw`; env vars `OSW_REGISTRY_PATH` / `OSW_MODELS_DIR`; schema family `osw:local-model-cache`; supported major `1`; schema id `osw:local-model-cache:1.0`; bucket `osw-public-models`; ownership `Overhead Survey Workflow Team`.
- **Audit scope (completion gate):** this must return zero after Task 4:
  ```bash
  grep -rniE "RW_|rebel[ _-]workflow|rebel-pipeline|\brw:" \
    scoutbot/ tests/ docs/*.rst README.md CONTRIBUTING.md CHANGELOG.rst pyproject.toml docs/conf.py
  ```
  `docs/plans/**` and `docs/specs/**` are archival and excluded. Test-local temp filenames like `rw-registry.json` are not brand tokens and are not matched by the audit; leaving them is fine.

---

## File Map

- `scoutbot/registry/paths.py` — env-var + default-dir rename (Task 1).
- `scoutbot/registry/__init__.py` — schema id, `RegistrySchemaError`, and the schema-guard (Task 2).
- `scoutbot/scoutbot.py` — CLI error surface for `RegistrySchemaError` (Task 2); `register` docstring rename (Task 4).
- `scoutbot/model_store.py` — GCS bucket URL swap (Task 3).
- `pyproject.toml`, `docs/conf.py` — ownership metadata (Task 4).
- `docs/security.rst`, `docs/configuration.rst`, `docs/models.rst`, `docs/roadmap.rst`, `CHANGELOG.rst` — docs (Task 4).
- Tests: `tests/test_registry.py` (Tasks 1, 2), `tests/test_loc.py`, `tests/test_model_store.py`, `tests/test_pipeline.py`, `tests/test_cli.py` (Task 1 env swaps; Task 2 CLI test; Task 3 CONFIGS test).

---

## Task 1: Env-var + registry-path rename (`OSW_` / `~/.osw`)

Rename the two env vars and the default directory, atomically with every test that references them.

**Files:**
- Modify: `scoutbot/registry/paths.py`
- Test: `tests/test_registry.py:7-44,110,120`, `tests/test_loc.py`, `tests/test_model_store.py:39-40`, `tests/test_pipeline.py:308,363`, `tests/test_cli.py:352`

**Interfaces:**
- Produces: `get_registry_path()` reading `OSW_REGISTRY_PATH` (default `~/.osw/model-registry.json`) and `get_models_dir()` reading `OSW_MODELS_DIR` (default `~/.osw/models`). `get_models_dir()` remains **API-only** — no fetch-path caller.

- [ ] **Step 1: Update the path/env unit tests (red)**

In `tests/test_registry.py`, replace the four path tests (lines 7–44) so they expect the new names:

```python
def test_get_registry_path_default():
    """Default registry path is under ~/.osw/."""
    os.environ.pop('OSW_REGISTRY_PATH', None)
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_registry_path() == Path.home() / '.osw' / 'model-registry.json'


def test_get_registry_path_from_env(tmp_path):
    """OSW_REGISTRY_PATH env var overrides default."""
    custom = str(tmp_path / 'custom-registry.json')
    os.environ['OSW_REGISTRY_PATH'] = custom
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_registry_path() == Path(custom)
    os.environ.pop('OSW_REGISTRY_PATH')


def test_get_models_dir_default():
    os.environ.pop('OSW_MODELS_DIR', None)
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_models_dir() == Path.home() / '.osw' / 'models'


def test_get_models_dir_from_env(tmp_path):
    os.environ['OSW_MODELS_DIR'] = str(tmp_path / 'models')
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_models_dir() == tmp_path / 'models'
    os.environ.pop('OSW_MODELS_DIR')
```

Also update the I/O fixture at `tests/test_registry.py:117-121`:

```python
@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    path = tmp_path / 'registry.json'
    monkeypatch.setenv('OSW_REGISTRY_PATH', str(path))
    return path
```

- [ ] **Step 2: Run the path tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -k "registry_path or models_dir" -v`
Expected: FAIL — `paths.py` still reads `RW_*` and defaults to `~/.rebel-workflow`.

- [ ] **Step 3: Rename in `scoutbot/registry/paths.py`**

Replace the file body's two functions:

```python
def get_registry_path() -> Path:
    return Path(
        os.getenv(
            'OSW_REGISTRY_PATH', Path.home() / '.osw' / 'model-registry.json'
        )
    )


def get_models_dir() -> Path:
    return Path(os.getenv('OSW_MODELS_DIR', Path.home() / '.osw' / 'models'))
```

- [ ] **Step 4: Run the path tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -k "registry_path or models_dir" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Sweep the remaining `RW_*` env-var references in the test suite**

Replace `RW_REGISTRY_PATH` → `OSW_REGISTRY_PATH` and `RW_MODELS_DIR` → `OSW_MODELS_DIR` in every remaining test fixture:
- `tests/test_loc.py` (9 occurrences of `RW_REGISTRY_PATH`)
- `tests/test_model_store.py:39-40` (`RW_REGISTRY_PATH`, `RW_MODELS_DIR`)
- `tests/test_pipeline.py:308,363` (`RW_REGISTRY_PATH`)
- `tests/test_cli.py:352` (`RW_REGISTRY_PATH`)

(Leave test-local temp filenames such as `'rw-registry.json'` as-is — they are not brand tokens and are not matched by the audit.)

- [ ] **Step 6: Verify no `RW_` env-var references remain in tests, and run the fast suite**

Run:
```bash
grep -rnE "RW_(REGISTRY|MODELS)" scoutbot/ tests/ && echo "STILL PRESENT" || echo "clean"
uv run pytest -m "not slow" -q
```
Expected: `clean`, and the full fast suite passes (189+ tests).

- [ ] **Step 7: Review checkpoint**

Pause for human review. Suggested commit message:
```
refactor: rename registry env vars/path RW_→OSW_, ~/.rebel-workflow→~/.osw
```

---

## Task 2: Schema id + `RegistrySchemaError` + schema-guard + CLI surface

Rename the schema id and add the family/major-enforcing guard, plus a clean CLI error surface.

**Files:**
- Modify: `scoutbot/registry/__init__.py`, `scoutbot/scoutbot.py`
- Test: `tests/test_registry.py:194` + new guard tests, `tests/test_cli.py`

**Interfaces:**
- Consumes: `OSW_REGISTRY_PATH` isolation from Task 1.
- Produces: `SCHEMA_ID='osw:local-model-cache:1.0'`, `SCHEMA_FAMILY='osw:local-model-cache'`, `SCHEMA_MAJOR=1`, and `class RegistrySchemaError(RuntimeError)` (all in `scoutbot.registry`). `_read_raw()` raises `RegistrySchemaError` on any incompatible/foreign/malformed file. A custom root `click.Group` (`_ScoutbotGroup`) converts it to a clean `click.ClickException` for **every** subcommand.

- [ ] **Step 1: Write the guard tests (red)**

Append to `tests/test_registry.py` (the `registry_path` fixture and `json`/`pytest` imports already exist):

```python
def _write_registry(registry_path, obj):
    import json as _json
    registry_path.write_text(obj if isinstance(obj, str) else _json.dumps(obj))


def test_guard_rejects_non_json(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, 'not json {{{')
    with pytest.raises(registry.RegistrySchemaError, match='OSW_REGISTRY_PATH'):
        registry.lookup('x')


def test_guard_rejects_non_object(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, '[1, 2, 3]')
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_missing_schema(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {'models': {}})
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_non_string_schema(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {'schema': 123, 'models': {}})
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_bare_family(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {'schema': 'osw:local-model-cache', 'models': {}})
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_old_rw_family(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {'schema': 'rw:local-model-cache:1.0', 'models': {}})
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_malformed_version(registry_path):
    from scoutbot import registry

    for bad in ('osw:local-model-cache:1', 'osw:local-model-cache:1.x',
                'osw:local-model-cache:1.2.3'):
        _write_registry(registry_path, {'schema': bad, 'models': {}})
        with pytest.raises(registry.RegistrySchemaError):
            registry.lookup('x')


def test_guard_rejects_incompatible_major(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {'schema': 'osw:local-model-cache:2.0', 'models': {}})
    with pytest.raises(registry.RegistrySchemaError, match='major'):
        registry.lookup('x')


def test_guard_accepts_same_major_newer_minor(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {
        'schema': 'osw:local-model-cache:1.9',
        'future_field': 'ignored',
        'models': {},
    })
    assert registry.lookup('missing') is None  # reads fine, ignores unknown field
```

Also update the existing schema assertion at `tests/test_registry.py:194`:

```python
    assert data['schema'] == 'osw:local-model-cache:1.0'
```

- [ ] **Step 2: Run the guard tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -k "guard or valid_schema" -v`
Expected: FAIL — `RegistrySchemaError` does not exist and no guard is applied.

- [ ] **Step 3: Implement the schema-guard in `scoutbot/registry/__init__.py`**

Replace the `SCHEMA_ID` line (line 12) and the `_read_raw()` function; add the constants, exception, and helpers:

```python
SCHEMA_FAMILY = 'osw:local-model-cache'
SCHEMA_MAJOR = 1
SCHEMA_ID = f'{SCHEMA_FAMILY}:{SCHEMA_MAJOR}.0'


class RegistrySchemaError(RuntimeError):
    """Raised when the registry file is not a compatible osw:local-model-cache registry."""


def _schema_error(path: Path, detail: str) -> str:
    return (
        f'{path} is not a compatible {SCHEMA_FAMILY}:{SCHEMA_MAJOR}.* registry ({detail}). '
        f'Set OSW_REGISTRY_PATH to point scoutbot at a different registry file.'
    )


def _check_schema(path: Path, schema: object) -> None:
    if not isinstance(schema, str):
        raise RegistrySchemaError(
            _schema_error(path, f'missing or non-string "schema": {schema!r}')
        )
    parts = schema.split(':')
    if len(parts) != 3:
        raise RegistrySchemaError(_schema_error(path, f'malformed schema id {schema!r}'))
    family = ':'.join(parts[:2])
    if family != SCHEMA_FAMILY:
        raise RegistrySchemaError(_schema_error(path, f'wrong schema family {family!r}'))
    version = parts[2].split('.')
    if len(version) != 2 or not all(v.isdigit() for v in version):
        raise RegistrySchemaError(_schema_error(path, f'malformed version {parts[2]!r}'))
    if int(version[0]) != SCHEMA_MAJOR:
        raise RegistrySchemaError(
            _schema_error(path, f'incompatible major schema version {schema!r}')
        )


def _read_raw() -> dict:
    path = _registry_path()
    if not path.exists():
        return {'schema': SCHEMA_ID, 'models': {}}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RegistrySchemaError(_schema_error(path, f'not valid JSON: {exc}')) from exc
    if not isinstance(data, dict):
        raise RegistrySchemaError(_schema_error(path, 'top-level JSON is not an object'))
    _check_schema(path, data.get('schema'))
    return data
```

- [ ] **Step 4: Run the guard tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (all guard cases + the updated schema assertion + existing round-trip tests).

- [ ] **Step 5: Write the CLI-surface test (red)**

Append to `tests/test_cli.py`. It covers a registry-management command (`models`), the model-fetch
entry point (`fetch`), and an inference command (`pipeline`) — proving the surface catches *every*
command, not just registry-management ones:

```python
import pytest
from click.testing import CliRunner


@pytest.mark.parametrize(
    'args',
    [
        ['models'],
        ['fetch', '--config', 'v3'],
        # --model forces _resolve_model() (scoutbot.py:127) — which reads the registry — to run
        # BEFORE any image work, so the guard fires deterministically regardless of the image.
        ['pipeline', '--model', 'unregistered', '__IMG__'],
    ],
)
def test_cli_reports_incompatible_registry_cleanly(args, tmp_path, monkeypatch):
    """A foreign/incompatible registry yields a clean non-zero CLI error, not a traceback."""
    from scoutbot.scoutbot import cli
    from scoutbot.registry import RegistrySchemaError

    reg = tmp_path / 'registry.json'
    reg.write_text('{"schema": "rw:local-model-cache:1.0", "models": {}}')
    monkeypatch.setenv('OSW_REGISTRY_PATH', str(reg))
    # Isolate the model cache too: model_store.fetch() creates CACHE_DIR (default
    # ~/.cache/scoutbot) before it reads the registry, so without this the `fetch` case would
    # touch the real user cache during the test.
    from scoutbot import model_store

    monkeypatch.setattr(model_store, 'CACHE_DIR', tmp_path / 'cache')

    if args[-1] == '__IMG__':
        img = tmp_path / 'img.jpg'
        img.write_bytes(b'')  # exists → passes the filepath validator; not decoded (registry raises first)
        args = args[:-1] + [str(img)]

    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    # Handled as a clean Click error, not leaked as an unhandled RegistrySchemaError traceback:
    assert not isinstance(result.exception, RegistrySchemaError)
    assert 'OSW_REGISTRY_PATH' in result.output
```

(`pytest` and `CliRunner` may already be imported at the top of `tests/test_cli.py`; if so, don't
duplicate the imports.)

- [ ] **Step 6: Run the CLI test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_cli_reports_incompatible_registry_cleanly -v`
Expected: FAIL (all 3 parametrized cases) — the command lets `RegistrySchemaError` leak unhandled, so `result.exception` is a `RegistrySchemaError` and the message is absent from `result.output`.

- [ ] **Step 7: Add a group-level CLI error surface in `scoutbot/scoutbot.py`**

Wrapping individual commands is error-prone — `pipeline`, `batch`, `batch-resumable`,
`get_classes`, `export-coreml`, and `example` all reach the registry through
`scoutbot._resolve_model()` / `scoutbot.model_store.fetch()`, so any of them can raise
`RegistrySchemaError`. Handle it **once** for every subcommand with a custom `click.Group`.

Add the import and group class near the top of `scoutbot/scoutbot.py` (after `import click`):

```python
from scoutbot.registry import RegistrySchemaError


class _ScoutbotGroup(click.Group):
    """Convert a RegistrySchemaError raised by any subcommand into a clean Click error
    (message, non-zero exit, no traceback)."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except RegistrySchemaError as exc:
            raise click.ClickException(str(exc)) from exc
```

Then change the root group definition at line 511 to use it:

```python
@click.group(cls=_ScoutbotGroup)
def cli():
    ...
```

Because the root group's `invoke()` runs each subcommand (including nested groups like
`models info`) within its own call stack, this single wrapper catches `RegistrySchemaError` from
**every** command — current and future — with no per-command decorators to keep in sync.

- [ ] **Step 8: Run the CLI test to verify it passes**

Run: `uv run pytest tests/test_cli.py::test_cli_reports_incompatible_registry_cleanly -v`
Expected: PASS (all 3 parametrized cases — exit code 1, `OSW_REGISTRY_PATH` in output, `result.exception` is a `SystemExit`, not a `RegistrySchemaError`).

- [ ] **Step 9: Full fast suite + lint/type**

Run:
```bash
uv run pytest -m "not slow" -q
uv run ruff check scoutbot/ tests/
uv run ruff format --check scoutbot/ tests/
uv run mypy scoutbot tools/visualise.py
```
Expected: all pass.

- [ ] **Step 10: Review checkpoint**

Pause for human review. Suggested commit message:
```
feat: osw registry schema id + family/major schema-guard with clean CLI error
```

---

## Task 3: GCS bucket URL swap + clean-room validation

Point `CONFIGS` at the already-verified `osw-public-models` bucket.

**Files:**
- Modify: `scoutbot/model_store.py:33-34,56-57`
- Test: `tests/test_model_store.py`

**Interfaces:**
- Consumes: the live `osw-public-models/scoutbot` bucket (content == sidecar == pinned MD5, already verified).
- Produces: `CONFIGS['v3']` / `CONFIGS['v3-cls']` GCS `url`/`md5_url` hosted on `osw-public-models`; Azure fallback URLs and pinned MD5s unchanged.

- [ ] **Step 1: Write the CONFIGS-host test (red)**

Append to `tests/test_model_store.py`:

```python
def test_pt_configs_use_osw_bucket():
    from scoutbot import model_store

    for key in ('v3', 'v3-cls'):
        gcs = model_store.CONFIGS[key]['sources'][0]
        assert gcs['provider'] == 'gcs'
        assert gcs['url'].startswith(
            'https://storage.googleapis.com/osw-public-models/scoutbot/'
        )
        assert gcs['md5_url'].startswith(
            'https://storage.googleapis.com/osw-public-models/scoutbot/'
        )
        assert 'rebel-pipeline-public-models' not in gcs['url']
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_model_store.py::test_pt_configs_use_osw_bucket -v`
Expected: FAIL — URLs still point at `rebel-pipeline-public-models`.

- [ ] **Step 3: Swap the bucket host in `scoutbot/model_store.py`**

In the `v3` and `v3-cls` GCS sources (lines 33–34 and 56–57), replace `rebel-pipeline-public-models` with `osw-public-models` (path prefix `/scoutbot/`, filenames, and `md5` pins unchanged). The four affected lines become:

```python
                'url': 'https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-onecls.kaza.20260729.pt',
                'md5_url': 'https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-onecls.kaza.20260729.pt.md5',
```
```python
                'url': 'https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-cls.kaza.20260729.pt',
                'md5_url': 'https://storage.googleapis.com/osw-public-models/scoutbot/yolov8-cls.kaza.20260729.pt.md5',
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_model_store.py -q`
Expected: PASS (including the new test; all existing model-store tests still green).

- [ ] **Step 5: Clean-room network validation against the new bucket**

Run:
```bash
uv run python - <<'PY'
import os, tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ['OSW_REGISTRY_PATH'] = str(tmp / 'registry.json')

from scoutbot import model_store
model_store.CACHE_DIR = tmp / 'cache'

for key in ('v3', 'v3-cls'):
    events = []
    path = model_store.fetch(config=key, observer=lambda e: events.append(e))
    evt = [e for e in events if e.get('name') == 'fetch'][-1]
    assert evt['cache_hit'] is False, f'{key}: expected a real download'
    assert evt.get('provider') == 'gcs', f'{key}: expected gcs, got {evt.get("provider")}'
    assert model_store._md5(Path(path)) == model_store.CONFIGS[key]['md5'], f'{key}: pinned hash mismatch'
    print(f'{key}: downloaded from osw-public-models (gcs), pinned hash verified')
PY
```
Expected: two lines confirming each config downloaded from `osw-public-models` with the pinned hash verified.

- [ ] **Step 6: Review checkpoint**

Pause for human review. Suggested commit message:
```
refactor: point model CONFIGS at osw-public-models bucket
```

---

## Task 4: Metadata, docstring, docs, CHANGELOG, and audit gate

Finish the rename on the naming surface (metadata + docstring + docs), reconcile the changelog, and prove completion with the audit grep and full gate.

**Files:**
- Modify: `pyproject.toml:13`, `docs/conf.py:23`, `scoutbot/scoutbot.py:438`
- Modify: `docs/security.rst`, `docs/configuration.rst`, `docs/models.rst`, `docs/roadmap.rst`, `CHANGELOG.rst`

- [ ] **Step 1: Rename ownership metadata + the CLI docstring**

- `pyproject.toml:13`: `{name = "Rebel Workflow Team"},` → `{name = "Overhead Survey Workflow Team"},` (leave the `{name = "Wild Me"}` line unchanged).
- `docs/conf.py:23`: `author = 'Rebel Workflow Team'` → `author = 'Overhead Survey Workflow Team'`.
- `scoutbot/scoutbot.py:438`: `"""Register a local model file in the Rebel Workflow model registry."""` → `"""Register a local model file in the Overhead Survey Workflow (OSW) model registry."""`.

- [ ] **Step 2: Update `docs/models.rst` and `docs/configuration.rst`**

- `docs/models.rst`: replace the four `rebel-pipeline-public-models` GCS URLs with `osw-public-models` (same paths/filenames).
- `docs/configuration.rst`: replace the GCS bucket URL (`rebel-pipeline-public-models` → `osw-public-models`); update the registry directory/env-var mentions to `~/.osw`, `OSW_REGISTRY_PATH`, `OSW_MODELS_DIR` if present.

- [ ] **Step 3: Update `docs/security.rst` — bucket, guard, and the versioning contract**

- Replace the `rebel-pipeline-public-models` bucket URL with `osw-public-models`.
- Add a **Registry schema versioning** subsection stating the contract verbatim from the spec:
  - Schema id is `osw:local-model-cache:MAJOR.MINOR`.
  - **MINOR** bump = additive/backward-compatible (new optional fields; older readers ignore unknown fields, newer readers tolerate missing ones — no reader breaks).
  - **MAJOR** bump = breaking (field removal/rename/semantic change); a reader built for major `N` refuses major `≠ N`; adopting a new major is coordinated across tools sharing the registry.
  - Note the read-time guard raises a clear error (naming the path and `OSW_REGISTRY_PATH`) on a foreign or incompatible-major registry file, and never overwrites an unrecognized file.

- [ ] **Step 4: Update `docs/roadmap.rst`**

Replace the registry-removal command `rm ~/.rebel-workflow/model-registry.json` with `rm ~/.osw/model-registry.json` (and any other `~/.rebel-workflow` mention in that file).

Also add a roadmap item capturing the architectural smell surfaced during plan review — registry-backed model resolution is reachable through several paths, which is *why* the schema-error surface had to be centralized at the CLI group. Add this subsection under an appropriate maintenance section (e.g. near *Model-store / `loc` staged cleanup*); the underline is intentionally longer than the title so the docs build stays warning-free:

```rst
Consolidate model resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** planned

Registry-backed model resolution is currently reachable through multiple paths —
``scoutbot._resolve_model()``, ``scoutbot.model_store.fetch()``, and direct calls from several
CLI commands (``pipeline`` / ``batch`` / ``batch-resumable`` each call ``_resolve_model`` and
also fetch). This scattering is why registry error handling had to be centralized at the CLI
group level (``_ScoutbotGroup``) rather than at the resolution boundary. Consolidate resolution
behind a single well-defined entry point so registry access, ``--model`` / ``--config``
resolution, provenance, and error handling live in one place.
```

- [ ] **Step 5: Reconcile `CHANGELOG.rst`**

These features first ship this release, so make the `Unreleased` section read in final form (do not merely append):
- The multi-source-fetch bullet naming `storage.googleapis.com/rebel-pipeline-public-models` → `osw-public-models`.
- The registry bullet naming `~/.rebel-workflow/model-registry.json` (and any `RW_*` env vars) → `~/.osw/model-registry.json` / `OSW_*`.
- Add a bullet documenting the rename itself:
  ```rst
  * Renamed the ecosystem namespace from ``rw`` ("Rebel Workflow") to ``osw`` ("Overhead Survey
    Workflow"): environment variables ``OSW_REGISTRY_PATH`` / ``OSW_MODELS_DIR``, the registry
    directory ``~/.osw``, the registry schema id ``osw:local-model-cache:1.0``, and the public
    model bucket ``osw-public-models``. This is a clean break with no ``rw`` compatibility
    aliases; existing ``~/.rebel-workflow`` registries are orphaned and rebuilt on next fetch. A
    registry schema-guard now rejects foreign or incompatible-major registry files with a clear,
    actionable error instead of failing cryptically.
  ```

- [ ] **Step 6: Run the audit-scope grep — must be zero**

Run:
```bash
grep -rniE "RW_|rebel[ _-]workflow|rebel-pipeline|\brw:" \
  scoutbot/ tests/ docs/*.rst README.md CONTRIBUTING.md CHANGELOG.rst pyproject.toml docs/conf.py \
  && echo "STILL PRESENT — fix before proceeding" || echo "AUDIT CLEAN"
```
Expected: `AUDIT CLEAN` (zero matches). Investigate and fix any hit before continuing.

- [ ] **Step 7: Docs build (clean) + full gate**

Run:
```bash
uv sync --extra docs && uv run sphinx-build -b html docs docs/_build/html
uv sync --extra test
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv audit --locked --preview-features audit-command
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
uv run pytest -m "not slow"
```
Expected: docs build succeeds with no new warnings; all gates pass.

- [ ] **Step 8: Review checkpoint**

Pause for human review. Suggested commit message:
```
docs: complete rw→osw rename (metadata, docs, changelog); audit clean
```

---

## Final Handoff Checklist

- [ ] `OSW_REGISTRY_PATH` / `OSW_MODELS_DIR` + `~/.osw` are the only registry env/path names; `get_models_dir()` remains API-only.
- [ ] Schema id is `osw:local-model-cache:1.0`; the guard enforces family + major and raises `RegistrySchemaError` on every malformed/foreign/incompatible case.
- [ ] **Every** CLI subcommand (via the `_ScoutbotGroup` root group) surfaces a clean non-zero error (no traceback) on a bad registry — verified for `models`, `fetch`, and `pipeline`.
- [ ] `CONFIGS` GCS URLs point at `osw-public-models`; clean-room fetch verified.
- [ ] Ownership metadata + CLI docstring say "Overhead Survey Workflow".
- [ ] `docs/security.rst` documents the schema versioning contract; other docs + CHANGELOG updated.
- [ ] The **Audit scope** grep returns zero; archival `docs/plans/**` / `docs/specs/**` left intact.
- [ ] Docs build clean; full non-slow gate + audit + bandit pass.
- [ ] User reviewed each task before staging/committing.
