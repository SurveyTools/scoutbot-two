# -*- coding: utf-8 -*-
"""
Model catalog and model fetching for ScoutBot YOLOv8 models.

Note: scoutbot.__init__ defines `log` before importing this module through
scoutbot.loc. Keep that import order intact so `from scoutbot import log`
continues to work here.
"""

import hashlib
import os
import shutil
from pathlib import Path
from time import perf_counter_ns

import requests
from tqdm import tqdm

from scoutbot import log
from scoutbot._observability import emit

CACHE_DIR = Path.home() / '.cache' / 'scoutbot'

DEFAULT_CONFIG = os.getenv('LOC_CONFIG', os.getenv('CONFIG', 'v3')).strip().lower()

CONFIGS = {
    'v3': {
        'name': 'yolov8-onecls.kaza.pt',
        'sources': [
            {
                'provider': 'azure',
                'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt',
                'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5',
            },
            {
                'provider': 'gcs',
                'url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt',
                'md5_url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt.md5',
            },
        ],
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    'v3-cls': {
        'name': 'yolov8-cls.kaza.pt',
        'sources': [
            {
                'provider': 'azure',
                'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt',
                'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5',
            },
            {
                'provider': 'gcs',
                'url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt',
                'md5_url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt.md5',
            },
        ],
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    # Core ML (.mlpackage) exports of the v3 detectors, for Apple Neural Engine
    # inference on macOS. These are local-only directory bundles (no url/md5_url):
    # fetch() short-circuits on backend == 'coreml' and, if the bundle is missing,
    # exports it on demand from the 'source' .pt config (see export_coreml). 'source'
    # also tells the exporter which .pt to convert.
    'v3-coreml': {
        'name': 'yolov8-onecls.kaza.mlpackage',
        'backend': 'coreml',
        'source': 'v3',
        'compute_units_requested': 'all',
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    'v3-cls-coreml': {
        'name': 'yolov8-cls.kaza.mlpackage',
        'backend': 'coreml',
        'source': 'v3-cls',
        'compute_units_requested': 'all',
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
}
CONFIGS[None] = CONFIGS[DEFAULT_CONFIG]  # type: ignore[index]


def _md5(filepath: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _download_from_source(source: dict, local_path: Path) -> tuple[str, bool]:
    """
    Attempt to resolve the model from one source entry.

    Fetches the remote MD5, skips the download if the local file already
    matches, otherwise streams the download and verifies locally.

    Returns:
        (remote_md5, cache_hit) on success.

    Raises:
        requests.RequestException: on any network failure.
        AssertionError: if the downloaded file fails MD5 verification.
    """
    md5_resp = requests.get(source['md5_url'], timeout=30)
    md5_resp.raise_for_status()
    remote_md5 = md5_resp.text.strip().split()[0]

    if local_path.exists() and _md5(local_path) == remote_md5:
        return remote_md5, True

    with requests.get(source['url'], stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with (
            open(local_path, 'wb') as f,
            tqdm(total=total, unit='B', unit_scale=True, desc=local_path.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

    assert _md5(local_path) == remote_md5, (
        f'MD5 mismatch after download from {source["provider"]}: {local_path.name}'
    )
    return remote_md5, False


def fetch(config=DEFAULT_CONFIG, observer=None) -> str:
    """
    Fetch the YOLOv8 model file, downloading only if absent or hash-mismatched.

    Checks the local model registry first; skips all network I/O when a valid
    cached entry is found. Registers the model after a successful download so
    subsequent calls resolve offline.

    Args:
        config (str or None): ``'v3'`` or ``'v3-cls'``. Defaults to ``'v3'``.

    Returns:
        str: local model file path.

    Raises:
        AssertionError: If the model cannot be fetched or MD5 verification fails.
    """
    if config is None:
        config = DEFAULT_CONFIG

    started_ns = perf_counter_ns()
    cfg = CONFIGS[config]
    registry_name = f'scoutbot-{config}'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / str(cfg['name'])

    # Core ML bundles are local-only directory packages: no remote URL to fetch and
    # no single file to hash. If absent, export one on demand from the source .pt.
    if cfg.get('backend') == 'coreml':
        cache_hit = local_path.exists()
        if not local_path.exists():
            log.info(f'Core ML model missing, exporting: {cfg["name"]}')
            export_coreml(config, observer=observer)
        assert local_path.exists(), f'Core ML export failed: {local_path}'
        log.debug(f'Using local Core ML model: {local_path}')
        ended_ns = perf_counter_ns()
        emit(
            observer,
            {
                'name': 'fetch',
                'status': 'ok',
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'config': config,
                'backend': 'coreml',
                'device': 'coreml',
                'cache_hit': cache_hit,
                'model_path': str(local_path),
                'compute_units_requested': cfg.get('compute_units_requested'),
            },
        )
        return str(local_path)

    from scoutbot import registry as _registry

    # A pinned md5 in CONFIGS is authoritative: a cache registered under a
    # different hash (e.g. a pre-migration artifact) must be re-fetched, not
    # served, even though it still matches its own registry entry.
    pinned_md5 = cfg.get('md5')
    entry = _registry.lookup(registry_name)
    if entry is not None:
        cached_path = Path(entry.path)
        if (
            cached_path.exists()
            and _md5(cached_path) == entry.md5
            and (pinned_md5 is None or entry.md5 == pinned_md5)
        ):
            log.debug(f'Model found in registry: {registry_name}')
            ended_ns = perf_counter_ns()
            emit(
                observer,
                {
                    'name': 'fetch',
                    'status': 'ok',
                    'started_ns': started_ns,
                    'ended_ns': ended_ns,
                    'duration_ns': ended_ns - started_ns,
                    'config': config,
                    'backend': 'pt',
                    'device': 'cpu',
                    'cache_hit': True,
                    'model_path': str(cached_path),
                },
            )
            return str(cached_path)

    last_exc: Exception | None = None
    used_source: dict | None = None
    remote_md5 = ''
    cache_hit = False
    sources: list = cfg['sources']  # type: ignore[assignment]

    for source in sources:
        try:
            remote_md5, cache_hit = _download_from_source(source, local_path)
            used_source = source
            break
        except (requests.RequestException, AssertionError) as exc:
            log.warning(
                f'Source {source["provider"]} unavailable for {cfg["name"]}: {exc!r}'
            )
            last_exc = exc
            if local_path.exists():
                local_path.unlink()

    if used_source is None:
        assert last_exc is not None
        raise RuntimeError(f'All sources failed for {cfg["name"]}') from last_exc

    # Defense in depth: the hosted sidecar hash must equal the pinned md5, so a
    # bad or partial re-upload fails loudly here instead of silently serving
    # mismatched bytes. _download_from_source already verified the downloaded
    # file against remote_md5; this ties remote_md5 to the code-pinned value.
    if pinned_md5 is not None:
        assert remote_md5 == pinned_md5, (
            f'Pinned MD5 mismatch for {cfg["name"]}: hosted sidecar {remote_md5} '
            f'!= pinned {pinned_md5}'
        )

    ended_ns = perf_counter_ns()

    if cache_hit:
        log.debug(f'Model up to date: {local_path}')
        emit(
            observer,
            {
                'name': 'fetch',
                'status': 'ok',
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'config': config,
                'backend': 'pt',
                'device': 'cpu',
                'cache_hit': True,
                'model_path': str(local_path),
                'provider': used_source['provider'],
            },
        )
        return str(local_path)

    from datetime import datetime, timezone

    from scoutbot.registry.schema import ModelEntry, RegistrySource

    _registry.register(
        ModelEntry(
            name=registry_name,
            path=str(local_path),
            md5=remote_md5,
            framework='yolov8',
            registered=datetime.now(timezone.utc).isoformat(),
            source=RegistrySource(
                type='fetched',
                url=str(used_source['url']),
                provider=used_source['provider'],
                fetched_at=datetime.now(timezone.utc).isoformat(),
            ),
            default_run_config={
                'slice_height': cfg['slice_height'],
                'slice_width': cfg['slice_width'],
                'overlap_height_ratio': cfg['overlap_height_ratio'],
                'overlap_width_ratio': cfg['overlap_width_ratio'],
                'perform_standard_pred': cfg['perform_standard_pred'],
                'postprocess_class_agnostic': cfg['postprocess_class_agnostic'],
                'confidence_threshold': cfg['thresh'],
            },
            description=f'Scoutbot {config} model (KAZA)',
        )
    )
    log.info(f'Registered: {registry_name}')
    log.info(f'Saved: {local_path} (from {used_source["provider"]})')
    emit(
        observer,
        {
            'name': 'fetch',
            'status': 'ok',
            'started_ns': started_ns,
            'ended_ns': ended_ns,
            'duration_ns': ended_ns - started_ns,
            'config': config,
            'backend': 'pt',
            'device': 'cpu',
            'cache_hit': False,
            'model_path': str(local_path),
            'provider': used_source['provider'],
        },
    )
    return str(local_path)


def export_coreml(config, force: bool = False, observer=None) -> str:
    """
    Export a Core ML ``.mlpackage`` from its source ``.pt`` model into the cache.

    The source ``.pt`` is fetched first if needed. The export is FP16
    (``half=True``) with NMS disabled (SAHI does its own NMS) and a fixed input
    size equal to the config's slice size — static shapes are an ANE requirement,
    and matching the slice size avoids a resize on every tile.

    Args:
        config (str): a Core ML config name (e.g. ``'v3-coreml'``).
        force (bool): re-export even if the bundle already exists.

    Returns:
        str: local ``.mlpackage`` path.
    """
    from scoutbot import loc as _loc

    cfg = _loc.CONFIGS[config]
    assert cfg.get('backend') == 'coreml', f'{config} is not a Core ML config'

    started_ns = perf_counter_ns()
    target = _loc.CACHE_DIR / str(cfg['name'])
    if target.exists() and not force:
        log.debug(f'Core ML model already present: {target}')
        ended_ns = perf_counter_ns()
        emit(
            observer,
            {
                'name': 'export_coreml',
                'status': 'ok',
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'config': config,
                'backend': 'coreml',
                'device': 'coreml',
                'force': force,
                'source_config': cfg['source'],
                'target_path': str(target),
                'cache_hit': True,
                'compute_units_requested': cfg.get('compute_units_requested'),
            },
        )
        return str(target)

    from ultralytics import YOLO

    if observer is None:
        pt_path = _loc.fetch(cfg['source'])
    else:
        pt_path = _loc.fetch(cfg['source'], observer=observer)
    imgsz = cfg.get('slice_width', 640)
    log.info(f'Exporting {cfg["name"]} from {Path(pt_path).name} (imgsz={imgsz})...')

    # Ultralytics writes "<pt_stem>.mlpackage" next to the .pt; since the .pt lives
    # in CACHE_DIR as e.g. yolov8-onecls.kaza.pt, the output lands exactly at target.
    exported = Path(
        # quantize='fp16' (half precision): ultralytics >=8.3 replaced the boolean
        # quantize=True with explicit precision values ('fp16'/'int8'/8/16/...).
        YOLO(pt_path).export(format='coreml', quantize='fp16', nms=False, imgsz=imgsz)
    )
    assert exported.exists(), f'Export produced no bundle for {config}'
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported), str(target))

    log.info(f'Saved: {target}')
    ended_ns = perf_counter_ns()
    emit(
        observer,
        {
            'name': 'export_coreml',
            'status': 'ok',
            'started_ns': started_ns,
            'ended_ns': ended_ns,
            'duration_ns': ended_ns - started_ns,
            'config': config,
            'backend': 'coreml',
            'device': 'coreml',
            'force': force,
            'source_config': cfg['source'],
            'target_path': str(target),
            'cache_hit': False,
            'compute_units_requested': cfg.get('compute_units_requested'),
        },
    )
    return str(target)
