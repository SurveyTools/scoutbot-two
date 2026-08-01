# -*- coding: utf-8 -*-
"""
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
"""

import json
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from os.path import exists
from pathlib import Path
from time import perf_counter_ns

from scoutbot import utils
from scoutbot._batch_mp import _load_processed
from scoutbot._observability import build_batch_summary, emit

log = utils.init_logging()
QUIET = not utils.VERBOSE

from scoutbot import loc, model_store, tile_batched  # NOQA

# pyproject.toml is the single source of truth for the version; read it from the installed
# package metadata rather than duplicating it here (a hardcoded copy silently drifted before).
try:
    __version__ = _package_version('scoutbot')
except PackageNotFoundError:  # running from a source tree with no installed dist metadata
    __version__ = '0.0.0+unknown'
# Backwards-compatible aliases.
VERSION = __version__
version = __version__


def _resolve_model(config=None, model_name=None, **overrides):
    """
    Resolve a registry entry and build the effective run config.

    Returns:
        tuple: (model_path: str, effective_run_config: dict, provenance: dict)

    Raises:
        RuntimeError: if the model is not in the registry.
    """
    from pathlib import Path as _Path

    from scoutbot import registry as _registry

    if model_name is None:
        if config is None:
            config = 'v3'
        model_name = f'scoutbot-{config}'

    entry = _registry.lookup(model_name)
    if entry is None:
        hint = (
            f'uv run scoutbot fetch --config {config}'
            if config
            else (
                'uv run scoutbot register --name scoutbot-v3 '
                '--path /path/to/model.pt --arch yolov8'
            )
        )
        raise RuntimeError(f"Model '{model_name}' not found in registry. Run: {hint}")

    effective = dict(entry.default_run_config)
    effective.update({k: v for k, v in overrides.items() if v is not None})

    provenance = {
        'name': entry.name,
        'file': _Path(entry.path).name,
        'md5': entry.md5,
        'framework': entry.framework,
        'run_config': dict(effective),
    }

    return entry.path, effective, provenance


def fetch(config=None, observer=None):
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
    if observer is None:
        return loc.fetch(config=config)
    return loc.fetch(config=config, observer=observer)


def pipeline(
    filepath,
    config=None,
    model_name=None,
    batched_detection_model=None,
    backend_device=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
    loc_thresh=0.45,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    perform_standard_pred=False,
    postprocess_class_agnostic=True,
    observer=None,
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

    started_ns = perf_counter_ns()
    observer_kwargs = {'observer': observer} if observer is not None else {}
    if batched_detection_model is None:
        if model_name is not None:
            yolov8_model_path, _, _ = _resolve_model(model_name=model_name)
        else:
            yolov8_model_path = loc.fetch(config=config, **observer_kwargs)

        batched_detection_model = tile_batched.Yolov8DetectionModel(
            model_path=yolov8_model_path,
            confidence_threshold=loc_thresh,
            device=backend_device,
            **observer_kwargs,
        )

    try:
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
            detects.append(
                {
                    'l': pred['category_name'],
                    'c': pred['score'],
                    'x': pred['bbox'][0],
                    'y': pred['bbox'][1],
                    'w': pred['bbox'][2],
                    'h': pred['bbox'][3],
                }
            )

        wic_ = round(wic_score, 4)
        ended_ns = perf_counter_ns()
        emit(
            observer,
            {
                'name': 'pipeline',
                'status': 'ok',
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'config': config,
                'filepath': filepath,
                'backend': 'coreml'
                if str(getattr(batched_detection_model, 'model_path', '')).endswith(
                    '.mlpackage'
                )
                else 'pt',
                'device': backend_device,
                'detections': len(detects),
                'wic': wic_,
            },
        )
        return wic_, detects
    except Exception as exc:
        ended_ns = perf_counter_ns()
        emit(
            observer,
            {
                'name': 'pipeline',
                'status': 'error',
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'config': config,
                'filepath': filepath,
                'backend': 'coreml'
                if str(getattr(batched_detection_model, 'model_path', '')).endswith(
                    '.mlpackage'
                )
                else 'pt',
                'device': backend_device,
                'error_type': type(exc).__name__,
            },
        )
        raise


def batch(
    filepaths,
    config=None,
    model_name=None,
    backend_device=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
    loc_thresh=0.45,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    perform_standard_pred=False,
    postprocess_class_agnostic=True,
    show_progress=True,
    observer=None,
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

    started_ns = perf_counter_ns()
    observer_kwargs = {'observer': observer} if observer is not None else {}

    if model_name is not None:
        model_path, _, _ = _resolve_model(
            config=config,
            model_name=model_name,
            confidence_threshold=loc_thresh if loc_thresh != 0.45 else None,
        )
    else:
        model_path = loc.fetch(config=config, **observer_kwargs)

    batched_detection_model = tile_batched.Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=loc_thresh,
        device=backend_device,
        **observer_kwargs,
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

    status = 'ok'
    try:
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
                observer=observer,
            )
            wic_list.append(wic_)
            detects_list.append(detects)

            if show_progress:
                elapsed = time.time() - t_start
                mean_s = elapsed / (i + 1)
                iterator.set_postfix(
                    {'file': Path(filepath).name[:20], 's/img': f'{mean_s:.1f}'}
                )
        return wic_list, detects_list
    except Exception as exc:
        status = 'error'
        ended_ns = perf_counter_ns()
        emit(
            observer,
            {
                'name': 'batch',
                'status': status,
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'config': config,
                'backend': 'coreml' if str(model_path).endswith('.mlpackage') else 'pt',
                'device': backend_device,
                'file_count': len(filepaths),
                'processed': len(wic_list),
                'error_type': type(exc).__name__,
            },
        )
        raise
    finally:
        if status == 'ok':
            ended_ns = perf_counter_ns()
            emit(
                observer,
                {
                    'name': 'batch',
                    'status': status,
                    'started_ns': started_ns,
                    'ended_ns': ended_ns,
                    'duration_ns': ended_ns - started_ns,
                    'config': config,
                    'backend': 'coreml'
                    if str(model_path).endswith('.mlpackage')
                    else 'pt',
                    'device': backend_device,
                    'file_count': len(filepaths),
                    'processed': len(wic_list),
                },
            )


def batch_resumable(
    filepaths,
    output_path,
    config=None,
    model_name=None,
    backend_device=os.getenv('SCOUTBOT_DEVICE', 'cuda:0'),
    loc_thresh=0.45,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    perform_standard_pred=False,
    postprocess_class_agnostic=True,
    show_progress=True,
    observer=None,
    processes=1,
    summary_json=None,
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
        processes (int): number of worker processes to use. Defaults to 1
            (the sequential loop). When greater than 1, the model is resolved
            once in the parent and each of the N worker processes loads its
            own copy, so ensure N copies fit in available VRAM. Linux/CUDA is
            the primary target for this mode.
        summary_json (str or Path, optional): if given, the machine-readable
            batch summary (counts, timing, throughput) is written to this
            path as JSON on completion, in both sequential and multiprocess
            modes.

    Returns:
        dict: Summary with keys 'processed', 'skipped', 'errors', 'total'
    """
    from datetime import datetime, timezone

    from tqdm import tqdm

    if processes < 1:
        raise ValueError(f'processes must be >= 1, got {processes}')

    if config is None:
        config = loc.DEFAULT_CONFIG

    started_ns = perf_counter_ns()
    observer_kwargs = {'observer': observer} if observer is not None else {}
    output_path = Path(output_path)

    processed = _load_processed(output_path)

    remaining = [fp for fp in filepaths if fp not in processed]

    if not remaining:
        log.info('All files already processed')
        result = {
            'processed': 0,
            'skipped': len(filepaths),
            'errors': 0,
            'total': len(filepaths),
        }
        summary_event = build_batch_summary(
            [],
            wall_ns=perf_counter_ns() - started_ns,
            processes=processes,
            mode='sequential' if processes == 1 else 'multiprocess',
            start_method=None,  # no pool was started; nothing ran
            processed=0,
            skipped=len(filepaths),
            errors=0,
            total=len(filepaths),
            default_pid=os.getpid(),
            base={
                'name': 'batch_resumable',
                'config': config,
                'backend': 'unknown',
                'device': backend_device,
                'output_path': str(output_path),
                'file_count': len(filepaths),
            },
        )
        emit(observer, summary_event)
        if summary_json is not None:
            Path(summary_json).write_text(
                json.dumps(summary_event, indent=2, sort_keys=True) + '\n'
            )
        return result

    log.info(f'Resuming: {len(processed)} done, {len(remaining)} remaining')

    if processes > 1:
        from scoutbot import _batch_mp

        work = _batch_mp.BatchWork(
            remaining=remaining,
            skipped_count=len(filepaths) - len(remaining),
            total_count=len(filepaths),
        )
        if model_name is not None:
            model_path, effective_run_config, model_provenance = _resolve_model(
                config=config,
                model_name=model_name,
                confidence_threshold=loc_thresh if loc_thresh != 0.45 else None,
                slice_height=slice_height if slice_height != 512 else None,
                slice_width=slice_width if slice_width != 512 else None,
                overlap_height_ratio=overlap_height_ratio
                if overlap_height_ratio != 0.25
                else None,
                overlap_width_ratio=overlap_width_ratio
                if overlap_width_ratio != 0.25
                else None,
            )
        else:
            # forward the observer to fetch, matching the sequential path.
            # Use the model-store surface directly (not the loc compatibility
            # wrapper) so new code does not reinforce loc.
            model_path = model_store.fetch(config=config, **observer_kwargs)
            effective_run_config = {
                'confidence_threshold': loc_thresh,
                'slice_height': slice_height,
                'slice_width': slice_width,
                'overlap_height_ratio': overlap_height_ratio,
                'overlap_width_ratio': overlap_width_ratio,
                'perform_standard_pred': perform_standard_pred,
                'postprocess_class_agnostic': postprocess_class_agnostic,
            }
            model_provenance = None

        submit = _batch_mp.make_mp_submit(
            model_path=model_path,
            effective_run_config=effective_run_config,
            config_name=config,
            backend_device=backend_device,
            processes=processes,
        )
        progress_bar = tqdm(
            total=len(remaining),
            desc='Processing',
            unit='img',
            disable=not show_progress,
        )

        def _on_progress(filepath, record):
            progress_bar.update(1)

        try:
            result = _batch_mp.run_batch(
                work,
                output_path,
                submit,
                summary_meta={
                    'mode': 'multiprocess',
                    'start_method': 'forkserver',
                    'processes': processes,
                    'default_pid': os.getpid(),
                    'base': {
                        'name': 'batch_resumable',
                        'config': config,
                        'backend': 'coreml'
                        if str(model_path).endswith('.mlpackage')
                        else 'pt',
                        'device': backend_device,
                        'output_path': str(output_path),
                        'file_count': len(filepaths),
                    },
                },
                provenance=model_provenance,
                on_progress=_on_progress,
                observer=observer,
            )
        finally:
            progress_bar.close()  # closed even if run_batch fails fast on worker death
        if summary_json is not None:
            Path(summary_json).write_text(
                json.dumps(result['summary'], indent=2, sort_keys=True) + '\n'
            )
        return {k: result[k] for k in ('processed', 'skipped', 'errors', 'total')}

    if model_name is not None:
        model_path, _, model_provenance = _resolve_model(
            config=config,
            model_name=model_name,
            confidence_threshold=loc_thresh if loc_thresh != 0.45 else None,
            slice_height=slice_height if slice_height != 512 else None,
            slice_width=slice_width if slice_width != 512 else None,
            overlap_height_ratio=overlap_height_ratio
            if overlap_height_ratio != 0.25
            else None,
            overlap_width_ratio=overlap_width_ratio
            if overlap_width_ratio != 0.25
            else None,
        )
    else:
        model_path = model_store.fetch(config=config, **observer_kwargs)
        model_provenance = None

    batched_detection_model = tile_batched.Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=loc_thresh,
        device=backend_device,
        **observer_kwargs,
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
                    observer=observer,
                )

                record = {
                    'filepath': filepath,
                    'wic': wic_,
                    'detections': detects,
                    'timestamp': datetime.now(timezone.utc)
                    .isoformat()
                    .replace('+00:00', 'Z'),
                    'config': config,
                }
                if model_provenance is not None:
                    record['model'] = model_provenance

                f.write(json.dumps(record) + '\n')
                f.flush()
                processed_count += 1

                if show_progress:
                    iterator.set_postfix(
                        {
                            'file': Path(filepath).name[:20],
                            'detects': len(detects),
                        }
                    )

            except Exception as e:
                log.error(f'Failed to process {filepath}: {e}')
                error_record = {
                    'filepath': filepath,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc)
                    .isoformat()
                    .replace('+00:00', 'Z'),
                    'config': config,
                }
                f.write(json.dumps(error_record) + '\n')
                f.flush()
                error_count += 1

    result = {
        'processed': processed_count,
        'skipped': len(processed),
        'errors': error_count,
        'total': len(filepaths),
    }
    ended_ns = perf_counter_ns()
    summary_event = build_batch_summary(
        [],  # v1: per-phase breakdown is not collected on the sequential path
        wall_ns=ended_ns - started_ns,
        processes=1,
        mode='sequential',
        start_method=None,
        processed=processed_count,
        skipped=len(processed),
        errors=error_count,
        total=len(filepaths),
        default_pid=os.getpid(),
        base={
            'name': 'batch_resumable',
            'config': config,
            'backend': 'coreml' if str(model_path).endswith('.mlpackage') else 'pt',
            'device': backend_device,
            'output_path': str(output_path),
            'file_count': len(filepaths),
        },
    )
    # Spec: the sequential summary keys per_worker_counts by the parent pid (the
    # single process that did all the work). build_batch_summary derives counts
    # from per-image events, which the sequential path does not collect, so set
    # it explicitly here.
    summary_event['per_worker_counts'] = {os.getpid(): processed_count}
    emit(observer, summary_event)
    if summary_json is not None:
        Path(summary_json).write_text(
            json.dumps(summary_event, indent=2, sort_keys=True) + '\n'
        )
    return result


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
