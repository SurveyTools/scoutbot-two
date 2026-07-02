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
import json
import os
from os.path import exists
from pathlib import Path

from scoutbot import utils

log = utils.init_logging()
QUIET = not utils.VERBOSE

from scoutbot import loc, tile_batched  # NOQA

VERSION = '0.2.0'
version = VERSION
__version__ = VERSION


def fetch(config=None):
    """
    Fetch the YOLOv8 model file from CDN if it does not exist locally.

    Args:
        config (str, optional): the configuration to use, one of ``v3``
            or ``v3-cls``.  Defaults to ``v3``.

    Returns:
        str: Path to the model file.
    """
    if config is None:
        config = loc.DEFAULT_CONFIG
    return loc.fetch(config=config)


def pipeline(
        filepath,
        config=None,
        batched_detection_model=None,
        backend_device=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
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
    from sahi.predict import get_sliced_prediction

    if config is None:
        config = loc.DEFAULT_CONFIG

    if batched_detection_model is None:
        yolov8_model_path = loc.fetch(config=config)

        batched_detection_model = tile_batched.Yolov8DetectionModel(
            model_path=yolov8_model_path,
            confidence_threshold=loc_thresh,
            device=backend_device,
        )

    det_result = get_sliced_prediction(
        image=filepath,
        detection_model=batched_detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=perform_standard_pred,
        postprocess_class_agnostic=postprocess_class_agnostic,
        verbose=0,
    )

    coco_prediction_list = []
    for object_prediction in det_result.object_prediction_list:
        coco_prediction_list.append(
            object_prediction.to_coco_prediction(image_id=None).json
        )

    wic_score = max([item['score'] for item in coco_prediction_list], default=0)

    detects = []
    for pred in coco_prediction_list:
        detects.append({
            'l': pred['category_name'],
            'c': pred['score'],
            'x': pred['bbox'][0],
            'y': pred['bbox'][1],
            'w': pred['bbox'][2],
            'h': pred['bbox'][3],
        })

    wic_ = round(wic_score, 4)

    return wic_, detects


def batch(
        filepaths,
        config=None,
        backend_device=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
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
    import time

    from tqdm import tqdm

    if config is None:
        config = loc.DEFAULT_CONFIG

    yolov8_model_path = loc.fetch(config=config)

    batched_detection_model = tile_batched.Yolov8DetectionModel(
        model_path=yolov8_model_path,
        confidence_threshold=loc_thresh,
        device=backend_device,
    )

    wic_list = []
    detects_list = []

    iterator = tqdm(
        filepaths,
        desc='Processing',
        unit='img',
        disable=not show_progress,
    )

    t_start = time.time()

    for i, filepath in enumerate(iterator):
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

        if show_progress:
            elapsed = time.time() - t_start
            mean_s = elapsed / (i + 1)
            iterator.set_postfix({'file': Path(filepath).name[:20], 's/img': f'{mean_s:.1f}'})

    return wic_list, detects_list


def batch_resumable(
        filepaths,
        output_path,
        config=None,
        backend_device=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
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

    from tqdm import tqdm

    if config is None:
        config = loc.DEFAULT_CONFIG

    output_path = Path(output_path)

    processed = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if 'filepath' in record and 'error' not in record:
                        processed.add(record['filepath'])
                except json.JSONDecodeError:
                    continue

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

    yolov8_model_path = loc.fetch(config=config)

    batched_detection_model = tile_batched.Yolov8DetectionModel(
        model_path=yolov8_model_path,
        confidence_threshold=loc_thresh,
        device=backend_device,
    )

    processed_count = 0
    error_count = 0

    with open(output_path, 'a') as f:
        iterator = tqdm(
            remaining,
            desc='Processing',
            unit='img',
            disable=not show_progress,
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
                f.flush()
                processed_count += 1

                if show_progress:
                    iterator.set_postfix({
                        'file': Path(filepath).name[:20],
                        'detects': len(detects),
                    })

            except Exception as e:
                log.error(f'Failed to process {filepath}: {e}')
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


def get_classes(config=None):
    """
    Get the list of class labels supported by the model.

    Args:
        config (str, optional): the configuration to use. Defaults to ``v3``.

    Returns:
        list: list of class label strings
    """
    if config is None:
        config = loc.DEFAULT_CONFIG

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
    import hashlib

    import requests

    TEST_IMAGE_URL = 'https://wildbookiarepository.azureedge.net/data/scout.example.jpg'
    TEST_IMAGE_HASH = '786a940b062a90961f409539292f09144c3dbdbc6b6faa64c3e764d63d55c988'

    cache_dir = Path.home() / '.cache' / 'scoutbot'
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_filepath = cache_dir / 'scout.example.jpg'

    def _sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    if not img_filepath.exists() or _sha256(img_filepath) != TEST_IMAGE_HASH:
        log.info('Downloading example image...')
        with requests.get(TEST_IMAGE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(img_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    assert exists(str(img_filepath))
    log.debug(f'Running pipeline on image: {img_filepath}')

    wic_, detects = pipeline(str(img_filepath), config='v3')
    print(json.dumps(detects, indent=2))
