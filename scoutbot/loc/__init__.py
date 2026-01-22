# -*- coding: utf-8 -*-
'''
Localizer (LOC) configuration and model fetching for YOLOv8 models.

This module defines how YOLOv8 models are downloaded from an external CDN
and provides configuration for the v3 detection pipeline.
'''
import os
from os.path import exists, join
from pathlib import Path

from scoutbot import log, utils

PWD = Path(__file__).absolute().parent

DEFAULT_CONFIG = os.getenv('LOC_CONFIG', os.getenv('CONFIG', 'v3')).strip().lower()

CONFIGS = {
    'v3': {
        'hash': None,
        'name': 'yolov8-onecls.kaza.pt',
        'path': join(PWD, 'models', 'yolo', 'yolov8-onecls.kaza.pt'),
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    'v3-cls': {
        'hash': None,
        'name': 'yolov8-cls.kaza.pt',
        'path': join(PWD, 'models', 'yolo', 'yolov8-cls.kaza.pt'),
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
}
CONFIGS[None] = CONFIGS[DEFAULT_CONFIG]


def fetch(pull=False, config=DEFAULT_CONFIG):
    """
    Fetch the YOLOv8 model file from CDN, local path, or network path.

    This function will throw an AssertionError if the download fails or the
    file otherwise does not exist locally on disk.

    Args:
        pull (bool, optional): If :obj:`True`, force using the downloaded/copied versions
            stored in the local system's cache.  Defaults to :obj:`False`.
        config (str or None, optional): the configuration to use, one of ``v3``
            or ``v3-cls``.  Defaults to ``v3``.

    Returns:
        str: local model file path.

    Raises:
        AssertionError: If the model cannot be fetched.
        FileNotFoundError: If the model file doesn't exist at the specified path.
    """
    if config is None:
        config = DEFAULT_CONFIG

    model_name = CONFIGS[config]['name']
    model_path = CONFIGS[config]['path']
    model_hash = CONFIGS[config]['hash']

    if exists(model_path) and not pull:
        if utils.check_file_integrity(model_path, model_hash):
            return model_path
        else:
            log.warning(f"Local model found at {model_path} but hash mismatch. Attempting fetch...")

    # Import the utility function from parent module
    from scoutbot import MODEL_BASE_URL, get_model_from_source

    model = get_model_from_source(
        model_name=model_name,
        model_hash=model_hash,
        source_base=MODEL_BASE_URL,
        use_cache=True
    )
    assert exists(model)

    log.debug(f'LOC Model: {model}')

    return model
