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


def _bucket(records: list[dict], lo: int, hi: int) -> list[dict]:
    return [r for r in records if lo <= r['_count'] <= hi]


def _evenly_spaced(items: list[dict], n: int) -> list[dict]:
    """Pick ``n`` evenly-spaced items from a sorted list (not first-``n``, which clusters at an edge)."""
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def select_tiers(records: list[dict], per_tier: int) -> dict[str, list[dict]]:
    """Bucket JSONL records into density tiers and deterministically pick ``per_tier`` from each.

    Selection is over records only — never a directory walk. Bands are exact inclusive integer
    bounds derived from the Matetsi distribution. ``empty``/``typical``/``dense`` pick evenly-spaced
    across the (count, filepath)-sorted bucket; ``extreme`` (count >= 14, disjoint from ``dense``)
    takes the top-``per_tier`` descending so the global max is always included. The ``filepath``
    tiebreak orders deterministically and is never emitted. Raises if a tier is short.
    """
    recs = [{**r, '_count': len(r.get('detections', []))} for r in records]
    buckets = {
        'empty': _bucket(recs, 0, 0),
        'typical': _bucket(recs, 3, 6),
        'dense': _bucket(recs, 9, 13),
        'extreme': _bucket(recs, 14, 10**9),
    }
    out: dict[str, list[dict]] = {}
    for tier in TIERS:
        pool = sorted(buckets[tier], key=lambda r: (r['_count'], r['filepath']))
        if tier == 'extreme':
            picked = list(reversed(pool))[:per_tier]
        else:
            picked = _evenly_spaced(pool, per_tier)
        if len(picked) < per_tier:
            raise ValueError(
                f'Tier {tier!r} has only {len(picked)} candidate(s) (< --per-tier {per_tier})'
            )
        out[tier] = picked
    return out
