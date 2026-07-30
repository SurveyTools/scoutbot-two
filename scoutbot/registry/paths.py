# scoutbot/registry/paths.py
import os
from pathlib import Path


def _osw_home() -> Path:
    """Ecosystem-shared user-level home (Overhead Survey Workflow), configurable via ``OSW_HOME``.

    Per the OSW toolchain-directory principle, per-tool path overrides default to sub-paths under
    this home rather than to independent locations. Downloaded model files live under
    ``get_models_dir()`` (``OSW_MODELS_DIR``, default ``<home>/models``), keeping the whole
    toolchain footprint under one home.
    """
    return Path(os.getenv('OSW_HOME', Path.home() / '.osw'))


def get_registry_path() -> Path:
    return Path(os.getenv('OSW_REGISTRY_PATH', _osw_home() / 'model-registry.json'))


def get_models_dir() -> Path:
    return Path(os.getenv('OSW_MODELS_DIR', _osw_home() / 'models'))
