# -*- coding: utf-8 -*-
"""
Localizer (LOC) configuration and model fetching for YOLOv8 models.
"""

import hashlib
import os
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

from scoutbot import log

CACHE_DIR = Path.home() / '.cache' / 'scoutbot'

DEFAULT_CONFIG = os.getenv('LOC_CONFIG', os.getenv('CONFIG', 'v3')).strip().lower()

CONFIGS = {
    'v3': {
        'name': 'yolov8-onecls.kaza.pt',
        'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt',
        'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5',
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
        'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt',
        'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5',
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
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(config=DEFAULT_CONFIG) -> str:
    """
    Fetch the YOLOv8 model file, downloading only if absent or hash-mismatched.

    Args:
        config (str or None): ``'v3'`` or ``'v3-cls'``. Defaults to ``'v3'``.

    Returns:
        str: local model file path.

    Raises:
        AssertionError: If the model cannot be fetched or MD5 verification fails.
    """
    if config is None:
        config = DEFAULT_CONFIG

    cfg = CONFIGS[config]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / str(cfg['name'])

    # Core ML bundles are local-only directory packages: no remote URL to fetch and
    # no single file to hash. If absent, export one on demand from the source .pt.
    if cfg.get('backend') == 'coreml':
        if not local_path.exists():
            log.info(f'Core ML model missing, exporting: {cfg["name"]}')
            export_coreml(config)
        assert local_path.exists(), f'Core ML export failed: {local_path}'
        log.debug(f'Using local Core ML model: {local_path}')
        return str(local_path)

    response = requests.get(str(cfg['md5_url']), timeout=30)
    response.raise_for_status()
    remote_md5 = response.text.strip().split()[0]

    if local_path.exists() and _md5(local_path) == remote_md5:
        log.debug(f'Model up to date: {local_path}')
        return str(local_path)

    log.info(f'Downloading {cfg["name"]}...')
    with requests.get(str(cfg['url']), stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with (
            open(local_path, 'wb') as f,
            tqdm(total=total, unit='B', unit_scale=True, desc=cfg['name']) as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    assert _md5(local_path) == remote_md5, f'MD5 mismatch after download: {cfg["name"]}'
    log.info(f'Saved: {local_path}')
    return str(local_path)


def export_coreml(config, force: bool = False) -> str:
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
    cfg = CONFIGS[config]
    assert cfg.get('backend') == 'coreml', f'{config} is not a Core ML config'

    target = CACHE_DIR / str(cfg['name'])
    if target.exists() and not force:
        log.debug(f'Core ML model already present: {target}')
        return str(target)

    from ultralytics import YOLO

    pt_path = fetch(cfg['source'])
    imgsz = cfg.get('slice_width', 640)
    log.info(f'Exporting {cfg["name"]} from {Path(pt_path).name} (imgsz={imgsz})...')

    # Ultralytics writes "<pt_stem>.mlpackage" next to the .pt; since the .pt lives
    # in CACHE_DIR as e.g. yolov8-onecls.kaza.pt, the output lands exactly at target.
    exported = YOLO(pt_path).export(format='coreml', half=True, nms=False, imgsz=imgsz)
    exported = Path(exported)
    assert exported.exists(), f'Export produced no bundle for {config}'
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported), str(target))

    log.info(f'Saved: {target}')
    return str(target)
