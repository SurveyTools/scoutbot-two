"""Build a density-tiered, GPS-stripped scoutbot test corpus from the internal Matetsi survey.

Maintainer tool (run internally where Matetsi is available). It selects images by density tier
from the scoutbot detection JSONL, strips EXIF/GPS, freezes CPU inference baselines, and writes a
self-contained corpus (images/ + baselines.jsonl + manifest.json + README.md). It performs no
network I/O; the model must already be present locally.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import piexif
from PIL import Image

import scoutbot
from scoutbot import model_store
from scoutbot.model_store import CONFIGS
from scoutbot.registry.paths import get_models_dir

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


_EXIF_IFDS = ('0th', 'Exif', 'GPS', '1st')


def _has_exif(exif: dict) -> bool:
    return any(exif.get(ifd) for ifd in _EXIF_IFDS) or exif.get('thumbnail') is not None


def strip_and_verify(src: Path, dst: Path) -> None:
    """Copy ``src`` → ``dst`` with EXIF removed, then assert no EXIF/GPS survives.

    ``piexif.remove`` rewrites metadata only; pixels are not re-encoded. Baselines are frozen from
    the stripped copy, so bit-losslessness is not load-bearing — but we still assert the stripped
    image decodes and keeps its dimensions, since a corrupted strip would silently change baselines.
    Governance-critical: a residual EXIF/GPS tag is a hard failure.
    """
    with Image.open(src) as im:
        src_size = im.size
    shutil.copy(src, dst)
    # Only call remove() when there is EXIF to remove; piexif.remove() on a segment-less JPEG raises.
    if _has_exif(piexif.load(str(dst))):
        piexif.remove(str(dst))
    residual = piexif.load(str(dst))
    if _has_exif(residual):
        raise AssertionError(f'EXIF/GPS survived strip in {dst}: {residual}')
    with Image.open(dst) as im2:
        if im2.size != src_size:
            raise AssertionError(
                f'Dimensions changed after strip: {src_size} -> {im2.size}'
            )
        im2.load()


def lockfile_sha256(lock_path: Path) -> str:
    return hashlib.sha256(Path(lock_path).read_bytes()).hexdigest()


def _pkg(name: str) -> str:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return 'unknown'


def _git_commit() -> str:
    # WHY: provenance only — a detached/non-git checkout must not fail the build.
    # nosec B603/B607: fixed argv (no shell, no user input); `git` from PATH is intended for a
    # maintainer tool run inside the repo.
    try:
        out = subprocess.run(  # nosec B603 B607
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def collect_runtime(lock_path: Path) -> dict:
    """Capture the dependency stack that makes an exact-equality baseline interpretable."""
    return {
        'scoutbot_version': scoutbot.__version__,
        'scoutbot_commit': _git_commit(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'platform_system': platform.system(),
        'platform_machine': platform.machine(),
        'torch': _pkg('torch'),
        'torchvision': _pkg('torchvision'),
        'ultralytics': _pkg('ultralytics'),
        'sahi': _pkg('sahi'),
        'numpy': _pkg('numpy'),
        'lockfile_sha256': lockfile_sha256(lock_path)
        if Path(lock_path).exists()
        else 'unknown',
    }


def run_config_for(config: str) -> dict:
    """The effective pipeline() kwargs sourced from CONFIGS (pipeline does not read these itself)."""
    cfg = CONFIGS[config]
    # CONFIGS key is 'thresh'; the pipeline() kwarg is 'loc_thresh'.
    return {
        'backend_device': 'cpu',
        'loc_thresh': cfg['thresh'],
        'slice_height': cfg['slice_height'],
        'slice_width': cfg['slice_width'],
        'overlap_height_ratio': cfg['overlap_height_ratio'],
        'overlap_width_ratio': cfg['overlap_width_ratio'],
        'perform_standard_pred': cfg['perform_standard_pred'],
        'postprocess_class_agnostic': cfg['postprocess_class_agnostic'],
    }


def assert_model_present(config: str) -> Path:
    """Offline preflight: the model must already be local, else fail with fetch instructions.

    pipeline() would fetch on a cache miss, violating the no-network constraint. We check presence
    and, when the config exposes a pinned md5, integrity — never downloading. Configs without a
    pinned hash (e.g. Core ML bundles) fall back to a presence-only check.
    """
    cfg = CONFIGS[config]
    model_path = get_models_dir() / str(cfg['name'])
    hint = f'Run: uv run python -m scoutbot fetch --config {config}'
    if not model_path.exists():
        raise SystemExit(f'Model not present locally: {model_path}\n{hint}')
    pinned = cfg.get('md5')
    if pinned is not None:
        actual = model_store._md5(model_path)
        if actual != pinned:
            raise SystemExit(
                f'Model md5 mismatch for {model_path} (got {actual}, expected {pinned}).\n{hint}'
            )
    return model_path


def build_manifest(
    config: str, per_tier: int, images: list[dict], runtime: dict, model_md5: str
) -> dict:
    cfg = CONFIGS[config]
    return {
        'schema': 'scoutbot:test-corpus:1.0',
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source': 'matetsi (internal; GPS-stripped; opaque IDs)',
        'per_tier': per_tier,
        'model': {'config': config, 'name': str(cfg['name']), 'md5': model_md5},
        'run_config': run_config_for(config),
        'runtime': runtime,
        'images': images,
    }
