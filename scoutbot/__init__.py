# -*- coding: utf-8 -*-
'''
ScoutBot v3 Pipeline

ScoutBot provides a unified ML pipeline for wildlife detection in aerial imagery.
The pipeline uses YOLOv8 with SAHI (Slicing Aided Hyper Inference) for detecting
animals in large images.

Example usage:

.. code-block:: python

    import scoutbot

    # Process a single image
    filepath = '/path/to/image.jpg'
    wic_score, detections = scoutbot.pipeline(filepath, config='v3')

    # Process multiple images in batch
    filepaths = ['/path/to/image1.jpg', '/path/to/image2.jpg']
    wic_list, detects_list = scoutbot.batch(filepaths, config='v3')

    # Each detection is a dictionary:
    # {
    #     'l': class_label (str)
    #     'c': confidence (float)
    #     'x': x_top_left (float)
    #     'y': y_top_left (float)
    #     'w': width (float)
    #     'h': height (float)
    # }
'''
import os
import cv2
import shutil
from os.path import exists
from urllib.parse import urlparse
from pathlib import Path

import pooch

from scoutbot import utils

log = utils.init_logging()
QUIET = not utils.VERBOSE

# Base URLs for downloading models and data (can be overridden via environment variables)
MODEL_BASE_URL = (os.getenv('SCOUTBOT_MODEL_URL') or 'https://wildbookiarepository.azureedge.net/models').rstrip('/')
DATA_BASE_URL = (os.getenv('SCOUTBOT_DATA_URL') or 'https://wildbookiarepository.azureedge.net/data').rstrip('/')

# Validate MODEL_BASE_URL
parsed = urlparse(MODEL_BASE_URL)
if parsed.scheme in ('http', 'https', 'ftp'):
    # It's a URL, no further validation needed
    pass
elif parsed.scheme and parsed.scheme not in ('file', ''):
    raise ValueError(f"Unrecognized scheme in SCOUTBOT_MODEL_URL: {parsed.scheme}")
else:
    # It's a path - check if it exists
    model_path = Path(MODEL_BASE_URL)
    if not model_path.exists():
        raise FileNotFoundError(f"SCOUTBOT_MODEL_URL path does not exist: {MODEL_BASE_URL}")

# Validate DATA_BASE_URL
parsed = urlparse(DATA_BASE_URL)
if parsed.scheme in ('http', 'https', 'ftp'):
    # It's a URL, no further validation needed
    pass
elif parsed.scheme and parsed.scheme not in ('file', ''):
    raise ValueError(f"Unrecognized scheme in DATA_BASE_URL: {parsed.scheme}")
else:
    # It's a path - check if it exists
    data_path = Path(DATA_BASE_URL)
    if not data_path.exists():
        raise FileNotFoundError(f"DATA_BASE_URL path does not exist: {DATA_BASE_URL}")

from scoutbot import loc, tile_batched  # NOQA

VERSION = '3.0.0'
version = VERSION
__version__ = VERSION


def get_resource_from_source(resource_name, resource_hash, source_base, resource_type='model', use_cache=True):
    """
    Retrieve a resource (model or data) from URL, local path, or network path.

    Args:
        resource_name: Name of the resource file
        resource_hash: Expected hash of the resource (used for URL downloads)
        source_base: Base URL or path (from MODEL_BASE_URL or DATA_BASE_URL)
        resource_type: Type of resource ('model' or 'data') for cache organization
        use_cache: Whether to use cached version for URLs

    Returns:
        str: Path to the resource file

    Raises:
        FileNotFoundError: If the resource file doesn't exist at the specified path
    """
    # Parse the source to determine if it's a URL or path
    parsed = urlparse(source_base)

    if parsed.scheme in ('http', 'https', 'ftp'):
        # It's a URL - use pooch as before
        resource_url = f'{source_base}/{resource_name}'
        return pooch.retrieve(
            url=resource_url,
            known_hash=resource_hash,
            progressbar=not QUIET,
        )
    else:
        # It's a local or network path
        if source_base.startswith('file://'):
            source_base = source_base[7:]

        source_path = Path(source_base) / resource_name

        if not source_path.exists():
            raise FileNotFoundError(f"{resource_type.capitalize()} file not found: {source_path}")

        if use_cache:
            # Copy to cache directory to maintain consistency
            cache_dir = Path(pooch.os_cache(f"scoutbot/{resource_type}s"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / resource_name

            if source_path.resolve() == cache_path.resolve():
                return str(source_path)

            # Only copy if not already cached or file has changed
            if not cache_path.exists() or os.path.getmtime(source_path) > os.path.getmtime(cache_path):
                log.debug(f"Copying {resource_type} from {source_path} to cache {cache_path}")
                shutil.copy2(source_path, cache_path)

            return str(cache_path)
        else:
            # Use directly from source
            return str(source_path)


def get_model_from_source(model_name, model_hash, source_base, use_cache=True):
    """Backward compatibility wrapper for get_resource_from_source"""
    return get_resource_from_source(model_name, model_hash, source_base, 'model', use_cache)


def get_data_from_source(data_name, data_hash, source_base, use_cache=True):
    """Convenience wrapper for get_resource_from_source for data files"""
    return get_resource_from_source(data_name, data_hash, source_base, 'data', use_cache)


def fetch(pull=False, config='v3'):
    """
    Fetch the YOLOv8 model file from CDN if it does not exist locally.

    Args:
        pull (bool, optional): If :obj:`True`, force using the downloaded versions
            stored in the local system's cache.  Defaults to :obj:`False`.
        config (str, optional): the configuration to use, one of ``v3``
            or ``v3-cls``.  Defaults to ``v3``.

    Returns:
        str: Path to the model file.

    Raises:
        AssertionError: If the model cannot be fetched.
    """
    if config is None:
        config = 'v3'
    return loc.fetch(pull=pull, config=config)


def pipeline(
        filepath,
        config='v3',
        batched_detection_model=None,
        backend_device='cuda:0',
        loc_thresh=0.45,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=False,
        postprocess_class_agnostic=True,
):
    """
    Run the ML pipeline on a given image filepath and return the detections.

    The final output is a list of dictionaries, each representing a single detection.
    Each dictionary has a structure with the following keys:

        ::

            {
                'l': class_label (str)
                'c': confidence (float)
                'x': x_top_left (float)
                'y': y_top_left (float)
                'w': width (float)
                'h': height (float)
            }

    Args:
        filepath (str): image filepath (relative or absolute)
        config (str, optional): the configuration to use, one of ``v3``
            or ``v3-cls``.  Defaults to ``v3``.
        batched_detection_model: Pre-loaded YOLOv8 model (optional, for batch efficiency)
        backend_device (str): torch backend device (e.g., 'cuda:0', 'cpu', 'mps')
        loc_thresh (float): confidence threshold for detections
        slice_height (int): height of image slices for SAHI
        slice_width (int): width of image slices for SAHI
        overlap_height_ratio (float): vertical overlap ratio between slices
        overlap_width_ratio (float): horizontal overlap ratio between slices
        perform_standard_pred (bool): also run standard (non-sliced) prediction
        postprocess_class_agnostic (bool): apply class-agnostic NMS

    Returns:
        tuple ( float, list ( dict ) ): wic score (max detection confidence), list of detections
    """
    if config is None:
        config = 'v3'

    # Load model if not provided
    if batched_detection_model is None:
        yolov8_model_path = loc.fetch(config=config)

        batched_detection_model = tile_batched.Yolov8DetectionModel(
            model_path=yolov8_model_path,
            confidence_threshold=loc_thresh,
            device=backend_device,
        )

    det_result = tile_batched.get_sliced_prediction_batched(
        cv2.imread(filepath),
        batched_detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=perform_standard_pred,
        postprocess_class_agnostic=postprocess_class_agnostic,
    )

    # Postprocess detections
    coco_prediction_list = []
    for object_prediction in det_result.object_prediction_list:
        coco_prediction_list.append(
            object_prediction.to_coco_prediction(image_id=None).json
        )

    wic_score = max([item['score'] for item in coco_prediction_list], default=0)

    # Convert to output format
    detects = []
    for pred in coco_prediction_list:
        converted_pred = {
            'l': pred['category_name'],
            'c': pred['score'],
            'x': pred['bbox'][0],
            'y': pred['bbox'][1],
            'w': pred['bbox'][2],
            'h': pred['bbox'][3],
        }
        detects.append(converted_pred)

    wic_ = round(wic_score, 4)

    return wic_, detects


def batch(
        filepaths,
        config='v3',
        backend_device='cuda:0',
        loc_thresh=0.45,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=False,
        postprocess_class_agnostic=True,
        show_progress=True,
):
    """
    Run the ML pipeline on a batch of image filepaths and return detections.

    The output is a list of outputs matching the output of :func:`scoutbot.pipeline`,
    except the processing reuses the loaded model for efficiency.

    Args:
        filepaths (list): list of image filepaths (relative or absolute)
        config (str, optional): the configuration to use, one of ``v3``
            or ``v3-cls``.  Defaults to ``v3``.
        backend_device (str): torch backend device (e.g., 'cuda:0', 'cpu', 'mps')
        loc_thresh (float): confidence threshold for detections
        slice_height (int): height of image slices for SAHI
        slice_width (int): width of image slices for SAHI
        overlap_height_ratio (float): vertical overlap ratio between slices
        overlap_width_ratio (float): horizontal overlap ratio between slices
        perform_standard_pred (bool): also run standard (non-sliced) prediction
        postprocess_class_agnostic (bool): apply class-agnostic NMS
        show_progress (bool): show progress bar during processing

    Returns:
        tuple ( list ( float ), list ( list ( dict ) ) ): list of wic scores, list of detection lists
    """
    from tqdm import tqdm

    if config is None:
        config = 'v3'

    # Load model once for all images
    yolov8_model_path = loc.fetch(config=config)

    batched_detection_model = tile_batched.Yolov8DetectionModel(
        model_path=yolov8_model_path,
        confidence_threshold=loc_thresh,
        device=backend_device,
    )

    wic_list = []
    detects_list = []

    # Process with progress bar
    iterator = tqdm(
        filepaths,
        desc='Processing',
        unit='img',
        disable=not show_progress or QUIET,
    )

    for filepath in iterator:
        wic_, detects = pipeline(
            filepath,
            config,
            batched_detection_model=batched_detection_model,
            backend_device=backend_device,
            loc_thresh=loc_thresh,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=perform_standard_pred,
            postprocess_class_agnostic=postprocess_class_agnostic,
        )
        wic_list.append(wic_)
        detects_list.append(detects)

        # Update progress bar with current file info
        if show_progress:
            iterator.set_postfix({'file': Path(filepath).name[:20]})

    return wic_list, detects_list


def batch_resumable(
        filepaths,
        output_path,
        config='v3',
        backend_device='cuda:0',
        loc_thresh=0.45,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=False,
        postprocess_class_agnostic=True,
        show_progress=True,
):
    """
    Process images with automatic resume capability.

    Results are written to a JSONL file as each image completes.
    If processing is interrupted, calling again with the same output_path
    will skip already-processed images.

    Each line in the output JSONL file is a complete JSON object:

        ::

            {"filepath": "/path/to/img.jpg", "wic": 0.85, "detections": [...], "timestamp": "...", "config": "v3"}

    Args:
        filepaths (list): list of image filepaths (relative or absolute)
        output_path (str or Path): path to output JSONL file
        config (str, optional): the configuration to use, one of ``v3``
            or ``v3-cls``.  Defaults to ``v3``.
        backend_device (str): torch backend device (e.g., 'cuda:0', 'cpu', 'mps')
        loc_thresh (float): confidence threshold for detections
        slice_height (int): height of image slices for SAHI
        slice_width (int): width of image slices for SAHI
        overlap_height_ratio (float): vertical overlap ratio between slices
        overlap_width_ratio (float): horizontal overlap ratio between slices
        perform_standard_pred (bool): also run standard (non-sliced) prediction
        postprocess_class_agnostic (bool): apply class-agnostic NMS
        show_progress (bool): show progress bar during processing

    Returns:
        dict: Summary with keys 'processed', 'skipped', 'errors', 'total'
    """
    from datetime import datetime
    import json
    from tqdm import tqdm

    if config is None:
        config = 'v3'

    output_path = Path(output_path)

    # Load already-processed files
    processed = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if 'filepath' in record:
                        processed.add(record['filepath'])
                except json.JSONDecodeError:
                    continue

    # Filter to unprocessed files
    remaining = [fp for fp in filepaths if fp not in processed]

    if not remaining:
        log.info('All files already processed')
        return {
            'processed': 0,
            'skipped': len(filepaths),
            'errors': 0,
            'total': len(filepaths),
        }

    log.info(f'Resuming: {len(processed)} done, {len(remaining)} remaining')

    # Load model once
    yolov8_model_path = loc.fetch(config=config)

    batched_detection_model = tile_batched.Yolov8DetectionModel(
        model_path=yolov8_model_path,
        confidence_threshold=loc_thresh,
        device=backend_device,
    )

    processed_count = 0
    error_count = 0

    # Process and write incrementally
    with open(output_path, 'a') as f:
        iterator = tqdm(
            remaining,
            desc='Processing',
            unit='img',
            disable=not show_progress or QUIET,
        )

        for filepath in iterator:
            try:
                wic_, detects = pipeline(
                    filepath,
                    config,
                    batched_detection_model=batched_detection_model,
                    backend_device=backend_device,
                    loc_thresh=loc_thresh,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    perform_standard_pred=perform_standard_pred,
                    postprocess_class_agnostic=postprocess_class_agnostic,
                )

                record = {
                    'filepath': filepath,
                    'wic': wic_,
                    'detections': detects,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'config': config,
                }

                f.write(json.dumps(record) + '\n')
                f.flush()  # Ensure data is written immediately
                processed_count += 1

                if show_progress:
                    iterator.set_postfix({
                        'file': Path(filepath).name[:20],
                        'detects': len(detects),
                    })

            except Exception as e:
                log.error(f'Failed to process {filepath}: {e}')
                # Write error record
                error_record = {
                    'filepath': filepath,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'config': config,
                }
                f.write(json.dumps(error_record) + '\n')
                f.flush()
                error_count += 1

    return {
        'processed': processed_count,
        'skipped': len(processed),
        'errors': error_count,
        'total': len(filepaths),
    }


def get_classes(config='v3'):
    """
    Get the list of class labels supported by the model.

    Args:
        config (str, optional): the configuration to use. Defaults to ``v3``.

    Returns:
        list: list of class label strings
    """
    if config is None:
        config = 'v3'

    yolov8_model_path = loc.fetch(config=config)
    model = tile_batched.Yolov8DetectionModel(
        model_path=yolov8_model_path,
        confidence_threshold=0.5,
        device='cpu',
    )
    return list(model.category_names)


def example():
    """
    Run the pipeline on an example image with the default configuration.
    """
    import utool as ut

    TEST_IMAGE = 'scout.example.jpg'
    TEST_IMAGE_HASH = (
        '786a940b062a90961f409539292f09144c3dbdbc6b6faa64c3e764d63d55c988'  # NOQA
    )

    img_filepath = get_data_from_source(
        data_name=TEST_IMAGE,
        data_hash=TEST_IMAGE_HASH,
        source_base=DATA_BASE_URL,
        use_cache=True
    )
    assert exists(img_filepath)

    log.debug(f'Running pipeline on image: {img_filepath}')

    wic_, detects = pipeline(img_filepath, config='v3')

    log.debug(ut.repr3(detects))
