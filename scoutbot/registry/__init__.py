# scoutbot/registry/__init__.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scoutbot.registry.schema import ModelEntry

SCHEMA_FAMILY = 'osw:model-registry'
SCHEMA_MAJOR = 1
SCHEMA_ID = f'{SCHEMA_FAMILY}:{SCHEMA_MAJOR}.0'


class RegistrySchemaError(RuntimeError):
    """Raised when the registry file is not a compatible osw:local-model-cache registry."""


def _registry_path() -> Path:
    from scoutbot.registry.paths import get_registry_path

    return get_registry_path()


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
