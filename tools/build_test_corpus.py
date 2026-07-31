"""Build a density-tiered, GPS-stripped scoutbot test corpus from the internal Matetsi survey.

Maintainer tool (run internally where Matetsi is available). It selects images by density tier
from the scoutbot detection JSONL, strips EXIF/GPS, freezes CPU inference baselines, and writes a
self-contained corpus (images/ + baselines.jsonl + manifest.json + README.md). It performs no
network I/O; the model must already be present locally.
"""

from __future__ import annotations

from pathlib import Path

TIERS = ('empty', 'typical', 'dense', 'extreme')


def resolve_local(filepath: str, jsonl_source_root: str, matetsi_root: Path) -> Path:
    """Map a JSONL record's foreign absolute ``filepath`` to the local file by full relative suffix.

    Basename matching is unsafe — ~72% of Matetsi basenames collide across left/right and flights —
    so we strip the declared source-root prefix and join the *entire* remaining suffix onto
    ``matetsi_root``. That is a single deterministic path; collisions are a non-issue by
    construction. Raises if the prefix is absent, the suffix escapes ``matetsi_root``, or the joined
    path does not exist.
    """
    prefix = jsonl_source_root.rstrip('/') + '/'
    if not filepath.startswith(prefix):
        raise ValueError(
            f'JSONL filepath {filepath!r} is not under --jsonl-source-root {jsonl_source_root!r}'
        )
    suffix = filepath[len(prefix) :]
    if '..' in Path(suffix).parts:
        raise ValueError(f'Refusing traversal in JSONL suffix: {suffix!r}')
    local = matetsi_root / suffix
    root = matetsi_root.resolve()
    if not local.resolve().is_relative_to(root):
        raise ValueError(f'Resolved path escapes --matetsi-root: {local}')
    if not local.is_file():
        raise FileNotFoundError(f'Resolved local image does not exist: {local}')
    return local
