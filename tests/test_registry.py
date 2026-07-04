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
