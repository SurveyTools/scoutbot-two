# ScoutBot Model Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local model registry to scoutbot-2026 so inference runs fully offline after an explicit one-time fetch, and every `batch_resumable` output record includes full model provenance.

**Architecture:** A new `scoutbot/registry/` module owns the registry file (`~/.rebel-workflow/model-registry.json`), dataclasses, and I/O. `loc/fetch()` checks the registry before any network call. `batch_resumable()` stamps a full provenance block on every JSONL record. Four new/modified CLI commands surface the registry.

**Tech Stack:** Python stdlib only for registry core (`json`, `os`, `pathlib`, `dataclasses`). `click` for CLI. Existing `ultralytics` + `sahi` unchanged.

**Spec:** `pipeline-design/docs/superpowers/specs/2026-05-09-scoutbot-registry-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scoutbot/registry/__init__.py` | Public API: `lookup`, `register`, `unregister`, `list_models`, `load` |
| Create | `scoutbot/registry/schema.py` | `ModelEntry`, `RegistrySource` dataclasses with `to_dict`/`from_dict` |
| Create | `scoutbot/registry/paths.py` | `get_registry_path()`, `get_models_dir()` — read env at call time |
| Create | `tests/test_registry.py` | Tests for registry module |
| Modify | `scoutbot/loc/__init__.py` | `fetch()` checks registry first; registers after download |
| Modify | `scoutbot/__init__.py` | Add `_resolve_model()` helper; update `pipeline()`, `batch()`, `batch_resumable()` |
| Modify | `scoutbot/scoutbot.py` | Add `register`, `models` commands; add `--model` flag |
| Modify | `tests/test_loc.py` | Update fetch tests for offline-first behavior |
| Modify | `tests/test_cli.py` | Add tests for new CLI commands |
| Modify | `tests/test_pipeline.py` | Update for `_resolve_model` and provenance in output |

---

## Task 1: Registry paths module

**Files:**
- Create: `scoutbot/registry/__init__.py` (stub only)
- Create: `scoutbot/registry/paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import os
import importlib
from pathlib import Path


def test_get_registry_path_default():
    """Default registry path is under ~/.rebel-workflow/."""
    # Ensure env var is not set
    os.environ.pop('RW_REGISTRY_PATH', None)
    import scoutbot.registry.paths as p
    importlib.reload(p)
    assert p.get_registry_path() == Path.home() / '.rebel-workflow' / 'model-registry.json'


def test_get_registry_path_from_env(tmp_path):
    """RW_REGISTRY_PATH env var overrides default."""
    custom = str(tmp_path / 'custom-registry.json')
    os.environ['RW_REGISTRY_PATH'] = custom
    import scoutbot.registry.paths as p
    importlib.reload(p)
    assert p.get_registry_path() == Path(custom)
    os.environ.pop('RW_REGISTRY_PATH')


def test_get_models_dir_default():
    os.environ.pop('RW_MODELS_DIR', None)
    import scoutbot.registry.paths as p
    importlib.reload(p)
    assert p.get_models_dir() == Path.home() / '.rebel-workflow' / 'models'


def test_get_models_dir_from_env(tmp_path):
    os.environ['RW_MODELS_DIR'] = str(tmp_path / 'models')
    import scoutbot.registry.paths as p
    importlib.reload(p)
    assert p.get_models_dir() == tmp_path / 'models'
    os.environ.pop('RW_MODELS_DIR')
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/kirk/projects/surveying/survey-tools/scoutbot-2026
pytest tests/test_registry.py -v
```
Expected: `ModuleNotFoundError: No module named 'scoutbot.registry'`

- [ ] **Step 3: Create stub `scoutbot/registry/__init__.py`**

```python
# scoutbot/registry/__init__.py
```
(empty for now)

- [ ] **Step 4: Create `scoutbot/registry/paths.py`**

```python
# scoutbot/registry/paths.py
import os
from pathlib import Path


def get_registry_path() -> Path:
    return Path(os.getenv('RW_REGISTRY_PATH',
                          Path.home() / '.rebel-workflow' / 'model-registry.json'))


def get_models_dir() -> Path:
    return Path(os.getenv('RW_MODELS_DIR',
                          Path.home() / '.rebel-workflow' / 'models'))
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_registry.py -v
```
Expected: all 4 pass.

- [ ] **Step 6: Commit**

```bash
git add scoutbot/registry/__init__.py scoutbot/registry/paths.py tests/test_registry.py
git commit -m "feat: add registry module skeleton and paths"
```

---

## Task 2: Registry schema dataclasses

**Files:**
- Create: `scoutbot/registry/schema.py`
- Modify: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_registry.py`

```python
# --- schema tests ---

from scoutbot.registry.schema import ModelEntry, RegistrySource


def test_registry_source_to_dict_omits_nones():
    src = RegistrySource(type='manual', note='test')
    d = src.to_dict()
    assert d == {'type': 'manual', 'note': 'test'}
    assert 'url' not in d
    assert 'fetched_at' not in d


def test_registry_source_round_trip():
    src = RegistrySource(type='fetched', url='https://example.com/m.pt',
                         fetched_at='2026-05-09T00:00:00Z')
    d = src.to_dict()
    src2 = RegistrySource.from_dict(d)
    assert src2.type == 'fetched'
    assert src2.url == 'https://example.com/m.pt'
    assert src2.note is None


def test_model_entry_to_dict():
    entry = ModelEntry(
        name='test-model',
        path='/tmp/model.pt',
        md5='abc123',
        framework='yolov8',
        registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'),
        default_run_config={'confidence_threshold': 0.5},
        description='test',
    )
    d = entry.to_dict()
    assert d['name'] == 'test-model'
    assert d['md5'] == 'abc123'
    assert d['source'] == {'type': 'manual'}
    assert d['default_run_config'] == {'confidence_threshold': 0.5}


def test_model_entry_round_trip():
    entry = ModelEntry(
        name='my-model',
        path='/models/m.pt',
        md5='deadbeef',
        framework='herdnet',
        registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='local', original_path='/models/m.pt'),
        default_run_config={'patch_size': 512},
    )
    d = entry.to_dict()
    entry2 = ModelEntry.from_dict(d)
    assert entry2.name == 'my-model'
    assert entry2.framework == 'herdnet'
    assert entry2.source.type == 'local'
    assert entry2.source.original_path == '/models/m.pt'
    assert entry2.default_run_config == {'patch_size': 512}
    assert entry2.description == ''
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_registry.py -v -k "schema or source or entry"
```
Expected: `ImportError: cannot import name 'ModelEntry'`

- [ ] **Step 3: Create `scoutbot/registry/schema.py`**

```python
# scoutbot/registry/schema.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RegistrySource:
    type: str
    url: Optional[str] = None
    fetched_at: Optional[str] = None
    original_path: Optional[str] = None
    copied_from: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> RegistrySource:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ModelEntry:
    name: str
    path: str
    md5: str
    framework: str
    registered: str
    source: RegistrySource
    default_run_config: dict
    description: str = ''

    def to_dict(self) -> dict:
        d = asdict(self)
        d['source'] = self.source.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelEntry:
        return cls(
            name=d['name'],
            path=d['path'],
            md5=d['md5'],
            framework=d['framework'],
            registered=d['registered'],
            source=RegistrySource.from_dict(d['source']),
            default_run_config=d.get('default_run_config', {}),
            description=d.get('description', ''),
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_registry.py -v
```
Expected: all schema tests pass.

- [ ] **Step 5: Commit**

```bash
git add scoutbot/registry/schema.py tests/test_registry.py
git commit -m "feat: add ModelEntry and RegistrySource dataclasses"
```

---

## Task 3: Registry I/O (load, save, lookup, register, list_models)

**Files:**
- Modify: `scoutbot/registry/__init__.py`
- Modify: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_registry.py`

```python
# --- I/O tests ---
# All I/O tests use tmp_path and set RW_REGISTRY_PATH to isolate from real registry.

import json
import pytest


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    path = tmp_path / 'registry.json'
    monkeypatch.setenv('RW_REGISTRY_PATH', str(path))
    return path


def _make_entry(name='test-model', framework='yolov8'):
    return ModelEntry(
        name=name, path=f'/tmp/{name}.pt', md5=f'md5-{name}',
        framework=framework, registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'),
        default_run_config={'confidence_threshold': 0.45},
    )


def test_lookup_returns_none_when_registry_absent(registry_path):
    from scoutbot import registry
    assert registry.lookup('anything') is None


def test_register_and_lookup(registry_path):
    from scoutbot import registry
    entry = _make_entry('alpha')
    registry.register(entry)
    result = registry.lookup('alpha')
    assert result is not None
    assert result.name == 'alpha'
    assert result.md5 == 'md5-alpha'


def test_register_overwrites_existing(registry_path):
    from scoutbot import registry
    registry.register(_make_entry('m'))
    updated = ModelEntry(
        name='m', path='/new.pt', md5='new-md5', framework='yolov8',
        registered='2026-05-09T01:00:00Z',
        source=RegistrySource(type='manual'),
        default_run_config={},
    )
    registry.register(updated)
    result = registry.lookup('m')
    assert result.md5 == 'new-md5'
    assert result.path == '/new.pt'


def test_list_models_empty(registry_path):
    from scoutbot import registry
    assert registry.list_models() == []


def test_list_models_returns_all(registry_path):
    from scoutbot import registry
    for name in ('a', 'b', 'c'):
        registry.register(_make_entry(name))
    models = registry.list_models()
    assert len(models) == 3
    assert {m.name for m in models} == {'a', 'b', 'c'}


def test_registry_file_is_valid_schema(registry_path):
    from scoutbot import registry
    registry.register(_make_entry('x'))
    with open(registry_path) as f:
        data = json.load(f)
    assert data['schema'] == 'rw:model-registry:1.0'
    assert 'updated' in data
    assert 'x' in data['models']


def test_registry_write_is_atomic(registry_path):
    """No .tmp file left behind after a write."""
    from scoutbot import registry
    registry.register(_make_entry('x'))
    tmp = registry_path.with_suffix('.tmp')
    assert not tmp.exists()


def test_load_returns_raw_dict(registry_path):
    from scoutbot import registry
    registry.register(_make_entry('z'))
    raw = registry.load()
    assert isinstance(raw, dict)
    assert 'models' in raw
    assert 'z' in raw['models']
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_registry.py -v -k "lookup or register or list or schema or atomic or load"
```
Expected: failures due to empty `scoutbot/registry/__init__.py`.

- [ ] **Step 3: Implement `scoutbot/registry/__init__.py`**

```python
# scoutbot/registry/__init__.py
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scoutbot.registry.schema import ModelEntry

SCHEMA_ID = 'rw:model-registry:1.0'


def _registry_path() -> Path:
    from scoutbot.registry.paths import get_registry_path
    return get_registry_path()


def _read_raw() -> dict:
    path = _registry_path()
    if not path.exists():
        return {'schema': SCHEMA_ID, 'models': {}}
    with open(path, 'r') as f:
        return json.load(f)


def _write_raw(data: dict) -> None:
    path = _registry_path()
    data['updated'] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load() -> dict:
    """Return the raw registry dict for inspection."""
    return _read_raw()


def lookup(name: str) -> Optional[ModelEntry]:
    """Return a ModelEntry by name, or None if not found."""
    entry = _read_raw().get('models', {}).get(name)
    if entry is None:
        return None
    return ModelEntry.from_dict(entry)


def register(entry: ModelEntry) -> None:
    """Add or update a model entry. Saves atomically."""
    data = _read_raw()
    data.setdefault('models', {})[entry.name] = entry.to_dict()
    _write_raw(data)


def unregister(name: str) -> None:
    """Remove a model entry by name. No-op if absent."""
    data = _read_raw()
    data.get('models', {}).pop(name, None)
    _write_raw(data)


def list_models() -> list[ModelEntry]:
    """Return all registered ModelEntry objects."""
    return [ModelEntry.from_dict(v) for v in _read_raw().get('models', {}).values()]
```

- [ ] **Step 4: Run all registry tests**

```bash
pytest tests/test_registry.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scoutbot/registry/__init__.py tests/test_registry.py
git commit -m "feat: implement registry I/O (lookup, register, list_models)"
```

---

## Task 4: Update `fetch()` to check registry first

**Files:**
- Modify: `scoutbot/loc/__init__.py`
- Modify: `tests/test_loc.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_loc.py`

```python
# --- registry-integrated fetch tests ---

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv('RW_REGISTRY_PATH', str(tmp_path / 'registry.json'))
    monkeypatch.setenv('RW_MODELS_DIR', str(tmp_path / 'models'))
    return tmp_path


def test_fetch_uses_registry_when_model_cached(isolated_registry):
    """fetch() does not hit the network if the model is in the registry with matching MD5."""
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    from scoutbot import registry, loc

    # Create a fake model file
    model_file = isolated_registry / 'models' / 'yolov8-onecls.kaza.pt'
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b'fake model content')
    md5 = loc._md5(model_file)

    registry.register(ModelEntry(
        name='scoutbot-v3',
        path=str(model_file),
        md5=md5,
        framework='yolov8',
        registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='fetched', url='https://example.com/m.pt'),
        default_run_config={},
    ))

    with patch('requests.get') as mock_get:
        result = loc.fetch(config='v3')
        mock_get.assert_not_called()

    assert result == str(model_file)


def test_fetch_fails_clearly_when_not_registered_and_no_network(isolated_registry):
    """fetch() raises a clear error if model not in registry and network unavailable."""
    from scoutbot import loc
    import requests

    with patch('requests.get', side_effect=requests.ConnectionError('no network')):
        with pytest.raises(Exception):
            loc.fetch(config='v3')


def test_fetch_registers_model_after_download(isolated_registry, tmp_path):
    """After a successful download, fetch() writes a ModelEntry to the registry."""
    from scoutbot import registry, loc

    fake_content = b'fake model data'
    fake_md5 = loc._md5.__func__ if hasattr(loc._md5, '__func__') else None

    # Compute expected MD5
    import hashlib
    h = hashlib.md5()
    h.update(fake_content)
    expected_md5 = h.hexdigest()

    mock_md5_response = MagicMock()
    mock_md5_response.text = expected_md5
    mock_md5_response.raise_for_status = MagicMock()

    mock_model_response = MagicMock()
    mock_model_response.raise_for_status = MagicMock()
    mock_model_response.headers = {'content-length': str(len(fake_content))}
    mock_model_response.iter_content = MagicMock(return_value=[fake_content])
    mock_model_response.__enter__ = MagicMock(return_value=mock_model_response)
    mock_model_response.__exit__ = MagicMock(return_value=False)

    def mock_get(url, **kwargs):
        if 'md5' in url:
            return mock_md5_response
        return mock_model_response

    with patch('requests.get', side_effect=mock_get):
        loc.fetch(config='v3')

    entry = registry.lookup('scoutbot-v3')
    assert entry is not None
    assert entry.framework == 'yolov8'
    assert entry.source.type == 'fetched'
    assert 'confidence_threshold' in entry.default_run_config
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_loc.py -v -k "registry"
```
Expected: all 3 new tests fail.

- [ ] **Step 3: Update `scoutbot/loc/__init__.py`**

Replace the `fetch()` function body:

```python
def fetch(config=DEFAULT_CONFIG) -> str:
    if config is None:
        config = DEFAULT_CONFIG

    cfg = CONFIGS[config]
    registry_name = f'scoutbot-{config}'

    # --- offline path: check registry first ---
    from scoutbot import registry as _registry
    from scoutbot.registry.paths import get_models_dir
    entry = _registry.lookup(registry_name)
    if entry is not None:
        local_path = Path(entry.path)
        if local_path.exists() and _md5(local_path) == entry.md5:
            log.debug(f'Model found in registry: {registry_name}')
            return str(local_path)

    # --- online path: download and register ---
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    local_path = models_dir / cfg['name']

    response = requests.get(cfg['md5_url'], timeout=30)
    response.raise_for_status()
    remote_md5 = response.text.strip().split()[0]

    if not local_path.exists() or _md5(local_path) != remote_md5:
        log.info(f"Downloading {cfg['name']}...")
        with requests.get(cfg['url'], stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(local_path, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, desc=cfg['name']
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        assert _md5(local_path) == remote_md5, f"MD5 mismatch: {cfg['name']}"

    # Register / update entry
    from datetime import datetime, timezone
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    new_entry = ModelEntry(
        name=registry_name,
        path=str(local_path),
        md5=remote_md5,
        framework='yolov8',
        registered=datetime.now(timezone.utc).isoformat(),
        source=RegistrySource(
            type='fetched',
            url=cfg['url'],
            fetched_at=datetime.now(timezone.utc).isoformat(),
        ),
        default_run_config={
            'slice_height':               cfg['slice_height'],
            'slice_width':                cfg['slice_width'],
            'overlap_height_ratio':       cfg['overlap_height_ratio'],
            'overlap_width_ratio':        cfg['overlap_width_ratio'],
            'perform_standard_pred':      cfg['perform_standard_pred'],
            'postprocess_class_agnostic': cfg['postprocess_class_agnostic'],
            'confidence_threshold':       cfg['thresh'],
        },
        description=f'ScoutBot {config} model (KAZA)',
    )
    _registry.register(new_entry)
    log.info(f'Registered: {registry_name}')
    return str(local_path)
```

- [ ] **Step 4: Run all loc tests**

```bash
pytest tests/test_loc.py -v
```
Expected: all pass. Note: `test_fetch_v3_returns_path` etc. will hit the network if model not already registered — that is expected behaviour for the existing integration tests.

- [ ] **Step 5: Commit**

```bash
git add scoutbot/loc/__init__.py tests/test_loc.py
git commit -m "feat: fetch() checks registry before network; registers after download"
```

---

## Task 5: `_resolve_model()` helper and provenance in `batch_resumable`

**Files:**
- Modify: `scoutbot/__init__.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_pipeline.py`

```python
# --- _resolve_model tests ---

import os
import pytest
from unittest.mock import patch


@pytest.fixture
def registry_with_model(tmp_path, monkeypatch):
    monkeypatch.setenv('RW_REGISTRY_PATH', str(tmp_path / 'registry.json'))
    model_file = tmp_path / 'model.pt'
    model_file.write_bytes(b'fake')
    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    registry.register(ModelEntry(
        name='scoutbot-v3',
        path=str(model_file),
        md5='abc',
        framework='yolov8',
        registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'),
        default_run_config={
            'slice_height': 512, 'slice_width': 512,
            'overlap_height_ratio': 0.25, 'overlap_width_ratio': 0.25,
            'perform_standard_pred': False, 'postprocess_class_agnostic': True,
            'confidence_threshold': 0.45,
        },
    ))
    return str(model_file)


def test_resolve_model_by_config(registry_with_model):
    from scoutbot import _resolve_model
    path, run_cfg, prov = _resolve_model(config='v3')
    assert path == registry_with_model
    assert run_cfg['confidence_threshold'] == 0.45
    assert prov['name'] == 'scoutbot-v3'
    assert prov['framework'] == 'yolov8'
    assert 'run_config' in prov


def test_resolve_model_by_name(registry_with_model):
    from scoutbot import _resolve_model
    path, run_cfg, prov = _resolve_model(model_name='scoutbot-v3')
    assert prov['name'] == 'scoutbot-v3'


def test_resolve_model_override_threshold(registry_with_model):
    from scoutbot import _resolve_model
    _, run_cfg, prov = _resolve_model(config='v3', confidence_threshold=0.70)
    assert run_cfg['confidence_threshold'] == 0.70
    assert prov['run_config']['confidence_threshold'] == 0.70


def test_resolve_model_raises_when_not_registered(tmp_path, monkeypatch):
    monkeypatch.setenv('RW_REGISTRY_PATH', str(tmp_path / 'empty.json'))
    from scoutbot import _resolve_model
    with pytest.raises(RuntimeError, match="not found in registry"):
        _resolve_model(config='v3')
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_pipeline.py -v -k "resolve_model"
```
Expected: `ImportError: cannot import name '_resolve_model'`

- [ ] **Step 3: Add `_resolve_model()` to `scoutbot/__init__.py`**

Add after the imports, before `fetch()`:

```python
def _resolve_model(config=None, model_name=None, **overrides):
    """
    Resolve a registry entry and build the effective run config.

    Returns:
        tuple: (model_path: str, effective_run_config: dict, provenance: dict)

    Raises:
        RuntimeError: if the model is not in the registry.
    """
    from pathlib import Path as _Path
    from scoutbot import registry as _registry

    if model_name is None:
        if config is None:
            config = 'v3'
        model_name = f'scoutbot-{config}'

    entry = _registry.lookup(model_name)
    if entry is None:
        hint = f'scoutbot fetch --config {config}' if config else f'scoutbot register --name {model_name} ...'
        raise RuntimeError(
            f"Model '{model_name}' not found in registry. Run: {hint}"
        )

    effective = dict(entry.default_run_config)
    effective.update({k: v for k, v in overrides.items() if v is not None})

    provenance = {
        'name': entry.name,
        'file': _Path(entry.path).name,
        'md5': entry.md5,
        'framework': entry.framework,
        'run_config': dict(effective),
    }

    return entry.path, effective, provenance
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pipeline.py -v -k "resolve_model"
```
Expected: all 4 pass.

- [ ] **Step 5: Write failing test for provenance in `batch_resumable` output**

Append to `tests/test_pipeline.py`:

```python
def test_batch_resumable_output_includes_model_provenance(tmp_path, monkeypatch, registry_with_model):
    """Each JSONL record must include a 'model' provenance block."""
    import json
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import scoutbot

    output_file = tmp_path / 'results.jsonl'

    # Fake pipeline() to return predictable results
    with patch.object(scoutbot, 'pipeline', return_value=(0.9, [{'l': 'elephant', 'c': 0.9, 'x': 0, 'y': 0, 'w': 10, 'h': 10}])):
        # We need a real image file path for the filepath arg
        fake_img = tmp_path / 'img.jpg'
        fake_img.write_bytes(b'fake')
        summary = scoutbot.batch_resumable(
            filepaths=[str(fake_img)],
            output_path=str(output_file),
            config='v3',
        )

    assert summary['processed'] == 1
    with open(output_file) as f:
        record = json.loads(f.readline())

    assert 'model' in record
    assert record['model']['name'] == 'scoutbot-v3'
    assert record['model']['framework'] == 'yolov8'
    assert 'run_config' in record['model']
    assert 'confidence_threshold' in record['model']['run_config']
    assert 'file' in record['model']
    assert 'md5' in record['model']
```

- [ ] **Step 6: Run to verify failure**

```bash
pytest tests/test_pipeline.py -v -k "provenance"
```
Expected: fails — `'model'` key missing from output record.

- [ ] **Step 7: Update `batch_resumable()` in `scoutbot/__init__.py`**

Near the top of `batch_resumable()`, after the config check and before the model load, add:

```python
    model_path, effective_run_config, model_provenance = _resolve_model(
        config=config,
        model_name=model_name if 'model_name' in locals() else None,
        confidence_threshold=loc_thresh if loc_thresh != 0.45 else None,
        slice_height=slice_height if slice_height != 512 else None,
        slice_width=slice_width if slice_width != 512 else None,
        overlap_height_ratio=overlap_height_ratio if overlap_height_ratio != 0.25 else None,
        overlap_width_ratio=overlap_width_ratio if overlap_width_ratio != 0.25 else None,
    )
```

Then update the record dict inside the loop:

```python
                record = {
                    'filepath': filepath,
                    'wic': wic_,
                    'detections': detects,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'config': config,        # kept for backward compatibility
                    'model': model_provenance,
                }
```

Note: `model_name` parameter is added to `batch_resumable()` signature in Task 7. For now the override dict approach is sufficient; the `model_name` kwarg defaults to None.

- [ ] **Step 8: Run all pipeline tests**

```bash
pytest tests/test_pipeline.py -v
```
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add scoutbot/__init__.py tests/test_pipeline.py
git commit -m "feat: add _resolve_model helper; stamp provenance block in batch_resumable output"
```

---

## Task 6: `scoutbot register` CLI command

**Files:**
- Modify: `scoutbot/scoutbot.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing test** — append to `tests/test_cli.py`

```python
# --- register command tests ---

import os
import pytest
from click.testing import CliRunner
from pathlib import Path


@pytest.fixture
def isolated_registry_cli(tmp_path, monkeypatch):
    monkeypatch.setenv('RW_REGISTRY_PATH', str(tmp_path / 'registry.json'))
    return tmp_path


def test_register_adds_model_to_registry(isolated_registry_cli, tmp_path):
    from scoutbot.scoutbot import cli
    from scoutbot import registry

    model_file = tmp_path / 'custom.pt'
    model_file.write_bytes(b'fake model')

    runner = CliRunner()
    result = runner.invoke(cli, [
        'register',
        '--name', 'my-model',
        '--path', str(model_file),
        '--arch', 'yolov8',
        '--description', 'Test model',
        '--threshold', '0.60',
    ])

    assert result.exit_code == 0, result.output
    entry = registry.lookup('my-model')
    assert entry is not None
    assert entry.framework == 'yolov8'
    assert entry.description == 'Test model'
    assert entry.default_run_config['confidence_threshold'] == 0.60
    assert entry.source.type == 'local'


def test_register_fails_if_path_missing(isolated_registry_cli):
    from scoutbot.scoutbot import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        'register',
        '--name', 'x', '--path', '/does/not/exist.pt', '--arch', 'yolov8',
    ])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_cli.py -v -k "register"
```
Expected: `UsageError: No such command 'register'`

- [ ] **Step 3: Add `register` command to `scoutbot/scoutbot.py`**

Add before the `cli` group definition:

```python
@click.command('register')
@click.option('--name',        required=True,  help='Registry name for this model')
@click.option('--path',        required=True,  help='Absolute path to model file on disk')
@click.option('--arch',        required=True,  help='Framework/architecture (e.g. yolov8, herdnet, owl)')
@click.option('--description', default='',     help='Human-readable description')
@click.option('--threshold',   default=0.45,   type=float, help='Default confidence threshold')
@click.option('--slice-height',default=512,    type=int,   help='SAHI slice height in pixels')
@click.option('--slice-width', default=512,    type=int,   help='SAHI slice width in pixels')
@click.option('--overlap',     default=0.25,   type=float, help='SAHI overlap ratio (both axes)')
def register_model(name, path, arch, description, threshold, slice_height, slice_width, overlap):
    """Register a local model file in the Rebel Workflow model registry."""
    import sys
    from datetime import datetime, timezone
    from pathlib import Path as _Path
    from scoutbot import registry as _registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    from scoutbot.loc import _md5

    model_path = _Path(path).resolve()
    if not model_path.exists():
        log.error(f'File not found: {model_path}')
        sys.exit(1)

    log.info(f'Computing MD5 for {model_path.name}...')
    md5 = _md5(model_path)

    entry = ModelEntry(
        name=name,
        path=str(model_path),
        md5=md5,
        framework=arch,
        registered=datetime.now(timezone.utc).isoformat(),
        source=RegistrySource(type='local', original_path=str(model_path)),
        default_run_config={
            'confidence_threshold':       threshold,
            'slice_height':               slice_height,
            'slice_width':                slice_width,
            'overlap_height_ratio':       overlap,
            'overlap_width_ratio':        overlap,
        },
        description=description,
    )
    _registry.register(entry)
    print(f'Registered {name!r} ({arch}) — {model_path.name}')
```

Add `register_model` to the `cli` group:

```python
cli.add_command(register_model, name='register')
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v -k "register"
```
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add scoutbot/scoutbot.py tests/test_cli.py
git commit -m "feat: add 'scoutbot register' CLI command"
```

---

## Task 7: `scoutbot models` and `scoutbot models info` commands

**Files:**
- Modify: `scoutbot/scoutbot.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_cli.py`

```python
def test_models_list_empty(isolated_registry_cli):
    from scoutbot.scoutbot import cli
    runner = CliRunner()
    result = runner.invoke(cli, ['models'])
    assert result.exit_code == 0
    assert 'No models' in result.output


def test_models_list_shows_registered(isolated_registry_cli, tmp_path):
    from scoutbot.scoutbot import cli
    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource

    model_file = tmp_path / 'm.pt'
    model_file.write_bytes(b'x')
    registry.register(ModelEntry(
        name='test-yolo', path=str(model_file), md5='abc123',
        framework='yolov8', registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'), default_run_config={},
    ))

    runner = CliRunner()
    result = runner.invoke(cli, ['models'])
    assert result.exit_code == 0
    assert 'test-yolo' in result.output
    assert 'yolov8' in result.output


def test_models_info_prints_json(isolated_registry_cli, tmp_path):
    from scoutbot.scoutbot import cli
    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    import json

    model_file = tmp_path / 'm.pt'
    model_file.write_bytes(b'x')
    registry.register(ModelEntry(
        name='info-model', path=str(model_file), md5='deadbeef',
        framework='herdnet', registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual', note='test'),
        default_run_config={'patch_size': 256},
    ))

    runner = CliRunner()
    result = runner.invoke(cli, ['models', 'info', 'info-model'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['name'] == 'info-model'
    assert data['framework'] == 'herdnet'
    assert data['default_run_config'] == {'patch_size': 256}


def test_models_info_unknown_exits_nonzero(isolated_registry_cli):
    from scoutbot.scoutbot import cli
    runner = CliRunner()
    result = runner.invoke(cli, ['models', 'info', 'nonexistent'])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_cli.py -v -k "models"
```
Expected: `UsageError: No such command 'models'`

- [ ] **Step 3: Add `models` group to `scoutbot/scoutbot.py`**

Add before the main `cli` group definition:

```python
@click.group('models', invoke_without_command=True)
@click.pass_context
def models_group(ctx):
    """List and inspect registered models. Defaults to listing all models."""
    if ctx.invoked_subcommand is None:
        from scoutbot import registry as _registry
        entries = _registry.list_models()
        if not entries:
            print('No models registered. Run: scoutbot fetch --config v3')
            return
        col = '{:<25} {:<12} {:<35} {:<10} {}'
        print(col.format('NAME', 'FRAMEWORK', 'FILE', 'MD5', 'REGISTERED'))
        print('-' * 90)
        for e in entries:
            from pathlib import Path as _Path
            fname = _Path(e.path).name
            print(col.format(e.name, e.framework, fname, e.md5[:8], e.registered[:10]))


@models_group.command('info')
@click.argument('name')
def models_info(name):
    """Print the full registry entry for a named model as JSON."""
    import sys
    from scoutbot import registry as _registry
    entry = _registry.lookup(name)
    if entry is None:
        log.error(f'Model not found in registry: {name}')
        sys.exit(1)
    print(json.dumps(entry.to_dict(), indent=2))
```

Add to the `cli` group:

```python
cli.add_command(models_group, name='models')
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v -k "models"
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scoutbot/scoutbot.py tests/test_cli.py
git commit -m "feat: add 'scoutbot models' and 'scoutbot models info' commands"
```

---

## Task 8: Add `--model NAME` flag to inference commands

**Files:**
- Modify: `scoutbot/scoutbot.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests** — append to `tests/test_cli.py`

```python
def test_pipeline_accepts_model_flag(isolated_registry_cli, tmp_path):
    """--model flag routes to the correct registry entry."""
    from scoutbot.scoutbot import cli
    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    from unittest.mock import patch
    import scoutbot

    model_file = tmp_path / 'custom.pt'
    model_file.write_bytes(b'fake')
    img_file = tmp_path / 'img.jpg'
    img_file.write_bytes(b'fake image')

    registry.register(ModelEntry(
        name='my-custom',
        path=str(model_file),
        md5='abc',
        framework='yolov8',
        registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'),
        default_run_config={
            'slice_height': 512, 'slice_width': 512,
            'overlap_height_ratio': 0.25, 'overlap_width_ratio': 0.25,
            'perform_standard_pred': False, 'postprocess_class_agnostic': True,
            'confidence_threshold': 0.45,
        },
    ))

    runner = CliRunner()
    with patch.object(scoutbot, 'pipeline', return_value=(0.0, [])) as mock_p:
        result = runner.invoke(cli, ['pipeline', str(img_file), '--model', 'my-custom'])
    assert result.exit_code == 0, result.output


def test_pipeline_model_and_config_are_exclusive(isolated_registry_cli, tmp_path):
    from scoutbot.scoutbot import cli
    img_file = tmp_path / 'img.jpg'
    img_file.write_bytes(b'x')
    runner = CliRunner()
    result = runner.invoke(cli, ['pipeline', str(img_file), '--model', 'x', '--config', 'v3'])
    assert result.exit_code != 0
    assert 'mutually exclusive' in result.output.lower() or result.exit_code != 0
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_cli.py -v -k "model_flag or exclusive"
```
Expected: `Error: No such option: --model`

- [ ] **Step 3: Add `--model` option to CLI commands in `scoutbot/scoutbot.py`**

Add to the `v3_options` list:

```python
v3_options = [
    click.option(
        '--device',
        help='Specifies the device for inference (see YOLO and PyTorch documentation)',
        default=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
        type=click.Choice(['cuda:0', 'cuda', 'mps', 'cpu']),
    ),
    click.option(
        '--loc-thresh',
        help='Detection confidence threshold (0-100)',
        default=45,
        type=click.IntRange(0, 100, clamp=True),
    ),
    click.option(
        '--model',
        help='Model name from registry (alternative to --config; mutually exclusive)',
        default=None,
        type=str,
    ),
]
```

Update the `pipeline` command function signature to accept `model` and validate mutual exclusion:

```python
@click.command('pipeline')
@click.argument('filepath', nargs=1, type=str, callback=pipeline_filepath_validator)
@add_options(model_option)
@add_options(v3_options)
@add_options(output_option)
def pipeline(filepath, config, output, device, loc_thresh, model):
    if model is not None and config != 'v3':
        raise click.UsageError('--model and --config are mutually exclusive.')

    loc_thresh_float = loc_thresh / 100.0
    model_name = model  # None means derive from config

    wic_, detects = scoutbot.pipeline(
        filepath,
        config=config if model is None else None,
        model_name=model_name,
        backend_device=device,
        loc_thresh=loc_thresh_float,
        ...  # existing SAHI params from CONFIGS
    )
    ...
```

Also add `model_name=None` parameter to `scoutbot.pipeline()` and pass it through to `_resolve_model()` in `__init__.py`. Do the same for `batch` and `batch-resumable` commands.

Full updated `pipeline` function in `scoutbot/__init__.py`:

```python
def pipeline(
        filepath,
        config='v3',
        model_name=None,
        batched_detection_model=None,
        backend_device=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
        loc_thresh=0.45,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=False,
        postprocess_class_agnostic=True,
):
    from sahi.predict import get_sliced_prediction

    if batched_detection_model is None:
        model_path, effective_cfg, _ = _resolve_model(
            config=config,
            model_name=model_name,
            confidence_threshold=loc_thresh,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        batched_detection_model = tile_batched.Yolov8DetectionModel(
            model_path=model_path,
            confidence_threshold=effective_cfg.get('confidence_threshold', loc_thresh),
            device=backend_device,
        )

    # rest of function unchanged
    ...
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scoutbot/scoutbot.py scoutbot/__init__.py tests/test_cli.py
git commit -m "feat: add --model flag to inference commands; mutually exclusive with --config"
```

---

## Task 9: Run full test suite and pre-commit checks

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 2: Run pre-commit linting**

```bash
pre-commit run --all-files
```
Expected: no errors. Fix any formatting issues before committing.

- [ ] **Step 3: Smoke test against a real registered model** (requires internet for initial fetch)

```bash
# Fetch and register the v3 model
scoutbot fetch --config v3

# Confirm it appears in the registry
scoutbot models

# Run on the built-in example — should work fully offline after this point
scoutbot example

# Confirm offline: disconnect network, run again
scoutbot pipeline ~/.rebel-workflow/models/yolov8-onecls.kaza.pt --model scoutbot-v3
# (or run on any image you have)
```

- [ ] **Step 4: Final commit**

```bash
git add -p   # stage only intentional changes
git commit -m "feat: complete scoutbot model registry — offline inference, full provenance output"
```

---

## Self-Review Notes

**Spec coverage check:**
- [x] Registry file format with `default_run_config` and source types → Tasks 2–3
- [x] `~/.rebel-workflow/` paths, env var overrides → Task 1
- [x] `fetch()` offline-first, registers after download → Task 4
- [x] `_resolve_model()` with effective run config → Task 5
- [x] Provenance block in `batch_resumable` output → Task 5
- [x] `scoutbot register` command → Task 6
- [x] `scoutbot models` / `scoutbot models info` → Task 7
- [x] `--model` flag, mutual exclusion with `--config` → Task 8
- [x] Backward compat: `"config"` key kept in output → Task 5 (noted in record dict)
- [x] Atomic writes → Task 3 (`os.replace`)

**Note:** `batch()` also needs `model_name` parameter for full consistency — add it in Task 8 alongside `pipeline()` and `batch_resumable()`.
