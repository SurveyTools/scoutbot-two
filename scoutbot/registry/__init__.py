# scoutbot/registry/__init__.py
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scoutbot.registry.schema import ModelEntry

SCHEMA_ID = 'rw:local-model-cache:1.0'


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
