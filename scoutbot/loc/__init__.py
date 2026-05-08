# -*- coding: utf-8 -*-
'''
Localizer (LOC) configuration and model fetching for YOLOv8 models.
'''
import hashlib
import os
from pathlib import Path

import requests
from tqdm import tqdm

from scoutbot import log

CACHE_DIR = Path.home() / '.cache' / 'scoutbot'

DEFAULT_CONFIG = os.getenv('LOC_CONFIG', os.getenv('CONFIG', 'v3')).strip().lower()

CONFIGS = {
    'v3': {
        'name':                       'yolov8-onecls.kaza.pt',
        'url':                        'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt',
        'md5_url':                    'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5',
        'thresh':                     0.45,
        'slice_height':               512,
        'slice_width':                512,
        'overlap_height_ratio':       0.25,
        'overlap_width_ratio':        0.25,
        'perform_standard_pred':      False,
        'postprocess_class_agnostic': True,
    },
    'v3-cls': {
        'name':                       'yolov8-cls.kaza.pt',
        'url':                        'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt',
        'md5_url':                    'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5',
        'thresh':                     0.45,
        'slice_height':               512,
        'slice_width':                512,
        'overlap_height_ratio':       0.25,
        'overlap_width_ratio':        0.25,
        'perform_standard_pred':      False,
        'postprocess_class_agnostic': True,
    },
}
CONFIGS[None] = CONFIGS[DEFAULT_CONFIG]


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
    local_path = CACHE_DIR / cfg['name']

    response = requests.get(cfg['md5_url'], timeout=30)
    response.raise_for_status()
    remote_md5 = response.text.strip().split()[0]

    if local_path.exists() and _md5(local_path) == remote_md5:
        log.debug(f'Model up to date: {local_path}')
        return str(local_path)

    log.info(f"Downloading {cfg['name']}...")
    with requests.get(cfg['url'], stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=cfg['name']
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    assert _md5(local_path) == remote_md5, f"MD5 mismatch after download: {cfg['name']}"
    log.info(f'Saved: {local_path}')
    return str(local_path)
