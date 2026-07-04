# scoutbot/registry/paths.py
import os
from pathlib import Path


def get_registry_path() -> Path:
    return Path(os.getenv('RW_REGISTRY_PATH',
                          Path.home() / '.rebel-workflow' / 'model-registry.json'))


def get_models_dir() -> Path:
    return Path(os.getenv('RW_MODELS_DIR',
                          Path.home() / '.rebel-workflow' / 'models'))
