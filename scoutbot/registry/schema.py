# scoutbot/registry/schema.py
from __future__ import annotations

from dataclasses import asdict, dataclass
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
