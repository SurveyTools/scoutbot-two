# tests/test_registry.py
import importlib
import os
from pathlib import Path


def test_get_registry_path_default():
    """Default registry path is under ~/.osw/."""
    # Ensure env var is not set
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
    assert data['schema'] == 'rw:local-model-cache:1.0'
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
