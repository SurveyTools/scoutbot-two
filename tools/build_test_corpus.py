"""Build a density-tiered, GPS-stripped scoutbot test corpus from the internal Matetsi survey.

Maintainer tool (run internally where Matetsi is available). It selects images by density tier
from the scoutbot detection JSONL, strips EXIF/GPS, freezes CPU inference baselines, and writes a
self-contained corpus (images/ + baselines.jsonl + manifest.json + README.md). It performs no
network I/O; the model must already be present locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess  # nosec B404: used only for `git rev-parse` provenance (fixed argv, no shell)
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


_EMITTED_FILES = ('manifest.json', 'baselines.jsonl', 'README.md')


def source_segments(
    records: list[dict], jsonl_source_root: str, matetsi_root: Path
) -> set[str]:
    """Forbidden source-derived tokens: the roots plus every selected record's path parts + basename.

    Deliberately does NOT include the bare '/' — the README carries commands and corpus-relative
    paths — so the scan targets source tokens, not path separators. Tier labels are excluded because
    they are legitimate emitted vocabulary (opaque names, manifest 'tier').
    """
    forbidden: set[str] = {
        jsonl_source_root,
        jsonl_source_root.rstrip('/'),
        str(matetsi_root),
    }
    for r in records:
        parts = [p for p in r['filepath'].split('/') if p]
        forbidden.update(parts)
        if parts:
            forbidden.add(Path(parts[-1]).stem)  # basename without extension
    return {f for f in forbidden if f} - set(TIERS)


def assert_no_source_identifiers(corpus_dir: Path, forbidden: set[str]) -> None:
    """Fail the build if any emitted file leaks a source-derived token (governance-critical)."""
    for name in _EMITTED_FILES:
        path = corpus_dir / name
        if not path.exists():
            continue
        text = path.read_text()
        for token in forbidden:
            if token and token in text:
                raise AssertionError(f'Source identifier {token!r} leaked into {name}')


def render_readme(config: str, per_tier: int) -> str:
    """Provenance/governance note. Opaque IDs only — no source filenames or paths."""
    return (
        '# Scoutbot Test Corpus (internal)\n\n'
        'Density-tiered scoutbot regression/benchmark corpus derived from an **internal** survey.\n\n'
        '> INTERNAL USE ONLY — GPS/EXIF-stripped, opaque IDs. **Not cleared for public '
        'distribution** until redistribution permission is established.\n\n'
        f'- Model config: `{config}`\n'
        f'- Images per tier: {per_tier} (tiers: empty, typical, dense, extreme)\n'
        '- Baselines: frozen scoutbot CPU output (`baselines.jsonl`); a drift-regression anchor, '
        'not human ground truth.\n\n'
        '## Consuming it\n\n'
        'Point the regression test at this directory:\n\n'
        '```\n'
        'SCOUTBOT_TEST_CORPUS=/path/to/this/corpus uv run pytest tests/test_corpus_regression.py\n'
        '```\n'
    )


def build_corpus(
    matetsi_jsonl: Path,
    jsonl_source_root: str,
    matetsi_root: Path,
    out_dir: Path,
    per_tier: int,
    config: str,
    force: bool = False,
) -> dict:
    """Select → resolve → strip → freeze → write corpus. No network I/O; model must be local."""
    images_dir = out_dir / 'images'
    # WHY: fail fast before any expensive work. Refuse ANY non-empty output dir unless --force —
    # stale images, or a stale manifest/baselines/README left by a rebuild that failed midway, would
    # silently corrupt the corpus. With --force we clear the whole output dir, not just images/.
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(
                f'Output dir {out_dir} is non-empty; pass --force to overwrite.'
            )
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    model_path = assert_model_present(config)
    model_md5 = model_store._md5(model_path) if CONFIGS[config].get('md5') else 'unknown'

    records = [
        json.loads(line)
        for line in matetsi_jsonl.read_text().splitlines()
        if line.strip()
    ]
    selected = select_tiers(records, per_tier)
    rc = run_config_for(config)

    image_records: list[dict] = []
    baselines: list[dict] = []
    used_records: list[dict] = []
    for tier in TIERS:
        for idx, rec in enumerate(selected[tier], start=1):
            used_records.append(rec)
            name = f'{tier}_{idx:02d}.jpg'
            dst = images_dir / name
            strip_and_verify(
                resolve_local(rec['filepath'], jsonl_source_root, matetsi_root), dst
            )

            wic, detections = scoutbot.pipeline(
                str(dst),
                config=config,
                backend_device='cpu',
                loc_thresh=rc['loc_thresh'],
                slice_height=rc['slice_height'],
                slice_width=rc['slice_width'],
                overlap_height_ratio=rc['overlap_height_ratio'],
                overlap_width_ratio=rc['overlap_width_ratio'],
                perform_standard_pred=rc['perform_standard_pred'],
                postprocess_class_agnostic=rc['postprocess_class_agnostic'],
            )
            baselines.append(
                {
                    'name': name,
                    'tier': tier,
                    'wic': wic,
                    'detections': detections,
                    'baseline_detection_count': len(detections),
                }
            )
            image_records.append(
                {
                    'name': name,
                    'tier': tier,
                    'md5': model_store._md5(dst),
                    'source_density': rec['_count'],
                    'baseline_detection_count': len(detections),
                }
            )

    (out_dir / 'baselines.jsonl').write_text(
        ''.join(json.dumps(b) + '\n' for b in baselines)
    )
    runtime = collect_runtime(Path(__file__).resolve().parents[1] / 'uv.lock')
    manifest = build_manifest(config, per_tier, image_records, runtime, model_md5)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out_dir / 'README.md').write_text(render_readme(config, per_tier))

    # Governance gate — fail the build if any source token leaked into an emitted file.
    assert_no_source_identifiers(
        out_dir, source_segments(used_records, jsonl_source_root, matetsi_root)
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Build a scoutbot test corpus from Matetsi.'
    )
    parser.add_argument('--matetsi-jsonl', required=True, type=Path)
    parser.add_argument('--jsonl-source-root', required=True)
    parser.add_argument('--matetsi-root', required=True, type=Path)
    parser.add_argument('-o', '--out', required=True, type=Path)
    parser.add_argument('--per-tier', type=int, default=5)
    parser.add_argument('--config', default='v3')
    parser.add_argument(
        '--force', action='store_true', help='overwrite a non-empty output images/ dir'
    )
    args = parser.parse_args(argv)
    manifest = build_corpus(
        args.matetsi_jsonl,
        args.jsonl_source_root,
        args.matetsi_root,
        args.out,
        args.per_tier,
        args.config,
        args.force,
    )
    print(f'Wrote corpus with {len(manifest["images"])} images to {args.out}')


if __name__ == '__main__':
    main()
