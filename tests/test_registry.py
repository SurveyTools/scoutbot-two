# tests/test_registry.py
import importlib
from pathlib import Path


def test_get_registry_path_default(monkeypatch):
    """Default registry path is under ~/.osw/ when no env overrides are set."""
    monkeypatch.delenv('OSW_REGISTRY_PATH', raising=False)
    monkeypatch.delenv('OSW_HOME', raising=False)
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_registry_path() == Path.home() / '.osw' / 'model-registry.json'


def test_get_registry_path_from_env(tmp_path, monkeypatch):
    """OSW_REGISTRY_PATH overrides both OSW_HOME and the default."""
    custom = str(tmp_path / 'custom-registry.json')
    monkeypatch.setenv('OSW_REGISTRY_PATH', custom)
    monkeypatch.setenv('OSW_HOME', str(tmp_path / 'home'))  # explicit path must still win
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_registry_path() == Path(custom)


def test_get_registry_path_under_osw_home(tmp_path, monkeypatch):
    """OSW_HOME relocates the base; the registry file lands under it (P13 compliance)."""
    monkeypatch.delenv('OSW_REGISTRY_PATH', raising=False)
    monkeypatch.setenv('OSW_HOME', str(tmp_path / 'osw-home'))
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_registry_path() == tmp_path / 'osw-home' / 'model-registry.json'


def test_get_models_dir_default(monkeypatch):
    monkeypatch.delenv('OSW_MODELS_DIR', raising=False)
    monkeypatch.delenv('OSW_HOME', raising=False)
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_models_dir() == Path.home() / '.osw' / 'models'


def test_get_models_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv('OSW_MODELS_DIR', str(tmp_path / 'models'))
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_models_dir() == tmp_path / 'models'


def test_get_models_dir_under_osw_home(tmp_path, monkeypatch):
    """OSW_HOME relocates the base; the models dir lands under it (P13 compliance)."""
    monkeypatch.delenv('OSW_MODELS_DIR', raising=False)
    monkeypatch.setenv('OSW_HOME', str(tmp_path / 'osw-home'))
    import scoutbot.registry.paths as p

    importlib.reload(p)
    assert p.get_models_dir() == tmp_path / 'osw-home' / 'models'


# --- schema tests ---

from scoutbot.registry.schema import ModelEntry, RegistrySource


def test_registry_source_to_dict_omits_nones():
    src = RegistrySource(type='manual', note='test')
    d = src.to_dict()
    assert d == {'type': 'manual', 'note': 'test'}
    assert 'url' not in d
    assert 'fetched_at' not in d


def test_registry_source_round_trip():
    src = RegistrySource(
        type='fetched', url='https://example.com/m.pt', fetched_at='2026-05-09T00:00:00Z'
    )
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


# --- I/O tests ---
# All I/O tests use tmp_path and set OSW_REGISTRY_PATH to isolate from real registry.

import json

import pytest


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    path = tmp_path / 'registry.json'
    monkeypatch.setenv('OSW_REGISTRY_PATH', str(path))
    return path


def _make_entry(name='test-model', framework='yolov8'):
    return ModelEntry(
        name=name,
        path=f'/tmp/{name}.pt',
        md5=f'md5-{name}',
        framework=framework,
        registered='2026-05-09T00:00:00Z',
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
        name='m',
        path='/new.pt',
        md5='new-md5',
        framework='yolov8',
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
    assert data['schema'] == 'osw:model-registry:1.0'
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


# --- schema-guard tests ---


def _write_registry(registry_path, obj):
    registry_path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


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

    _write_registry(registry_path, {'schema': 'osw:model-registry', 'models': {}})
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_old_rw_family(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {'schema': 'rw:local-model-cache:1.0', 'models': {}})
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_interim_local_model_cache_family(registry_path):
    """The interim osw:local-model-cache family (superseded by osw:model-registry) is rejected."""
    from scoutbot import registry

    _write_registry(registry_path, {'schema': 'osw:local-model-cache:1.0', 'models': {}})
    with pytest.raises(registry.RegistrySchemaError):
        registry.lookup('x')


def test_guard_rejects_malformed_version(registry_path):
    from scoutbot import registry

    for bad in (
        'osw:model-registry:1',
        'osw:model-registry:1.x',
        'osw:model-registry:1.2.3',
    ):
        _write_registry(registry_path, {'schema': bad, 'models': {}})
        with pytest.raises(registry.RegistrySchemaError):
            registry.lookup('x')


def test_guard_rejects_incompatible_major(registry_path):
    from scoutbot import registry

    _write_registry(registry_path, {'schema': 'osw:model-registry:2.0', 'models': {}})
    with pytest.raises(registry.RegistrySchemaError, match='major'):
        registry.lookup('x')


def test_guard_accepts_same_major_newer_minor(registry_path):
    from scoutbot import registry

    _write_registry(
        registry_path,
        {'schema': 'osw:model-registry:1.9', 'future_field': 'ignored', 'models': {}},
    )
    assert registry.lookup('missing') is None  # reads fine, ignores unknown field


def test_get_examples_dir_default(monkeypatch):
    """Default examples dir is under ~/.osw/ when OSW_HOME is unset."""
    monkeypatch.delenv('OSW_HOME', raising=False)
    import scoutbot.registry.paths as p

    assert p.get_examples_dir() == Path.home() / '.osw' / 'examples'


def test_get_examples_dir_under_osw_home(tmp_path, monkeypatch):
    """OSW_HOME relocates the base; the examples dir lands under it (P13 compliance)."""
    monkeypatch.setenv('OSW_HOME', str(tmp_path / 'osw-home'))
    import scoutbot.registry.paths as p

    assert p.get_examples_dir() == tmp_path / 'osw-home' / 'examples'
