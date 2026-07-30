# scoutbot/registry/paths.py
import os
from pathlib import Path


def get_registry_path() -> Path:
    return Path(
        os.getenv('OSW_REGISTRY_PATH', Path.home() / '.osw' / 'model-registry.json')
    )


def get_models_dir() -> Path:
    return Path(os.getenv('OSW_MODELS_DIR', Path.home() / '.osw' / 'models'))
