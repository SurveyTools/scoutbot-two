# -*- coding: utf-8 -*-
"""
Concurrency benchmark: sequential vs threaded vs multiprocess execution of
`scoutbot.pipeline()`.

Motivation: profiling on real survey imagery (8000x5320px, v3 config) showed each
pipeline() call splits roughly 55% GPU inference / 45% CPU (SAHI tiling + NMS
merge, all pure Python). Threading only overlaps the portion that releases the
GIL (GPU kernel launches) and plateaus quickly with more workers; separate
processes have no shared GIL, so they parallelize the CPU-bound 45% too and
scale further. On an RTX 5070 with the full v3 pipeline:

    sequential          ~3.5-3.9s/image  (baseline)
    threaded, 8 workers ~2.96s/image     (-22%, plateaus - GIL-limited)
    multiprocess, 8     ~1.75s/image     (-55%, plateaus around 8-16 procs)

Each model is small enough (~1-2GB VRAM) that several processes can share one
GPU comfortably.

Usage:
    uv run python benchmarks/benchmark_concurrency.py
    uv run python benchmarks/benchmark_concurrency.py --images examples --workers 8 --processes 8
    uv run python benchmarks/benchmark_concurrency.py --modes sequential multiprocess --processes 4
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import scoutbot
from scoutbot import loc, tile_batched
from scoutbot._observability import list_observer

CONFIG = 'v3'
LOC_THRESH = 0.45
WRAPPER_EVENT_NAMES = {'pipeline', 'batch', 'batch_resumable'}

_LFS_PREFIX = b'version https://git-lfs.github.com'
_VALID_SIGS = (b'\xff\xd8\xff', b'\x89PNG')  # JPEG and PNG magic bytes


def _validate_images(images):
    """Fail fast with a clear message if images are Git LFS stubs or wrong type."""
    stubs, bad = [], []
    for p in images:
        header = p.read_bytes()[: max(len(_LFS_PREFIX), 8)]
        if header[: len(_LFS_PREFIX)] == _LFS_PREFIX:
            stubs.append(p)
        elif not any(header[: len(sig)] == sig for sig in _VALID_SIGS):
            bad.append(p)
    if stubs:
        names = ', '.join(p.name for p in stubs[:3])
        raise SystemExit(
            f'{len(stubs)} image(s) are Git LFS pointer stubs ({names}...).\n'
            f'Run: git lfs install && git lfs pull'
        )
    if bad:
        names = ', '.join(p.name for p in bad[:3])
        raise SystemExit(f'{len(bad)} file(s) are not valid JPEG/PNG images: {names}')


def _build_model(device, observer=None):
    model_path = loc.fetch(config=CONFIG, observer=observer)
    return tile_batched.Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=LOC_THRESH,
        device=device,
        observer=observer,
    )


def _run_one(model, device, filepath, observer=None):
    return scoutbot.pipeline(
        str(filepath),
        config=CONFIG,
        batched_detection_model=model,
        backend_device=device,
        loc_thresh=LOC_THRESH,
        observer=observer,
    )


def _aggregate_phase_ns(events):
    """Sum durations by phase name, excluding the outer pipeline/batch wrapper events."""
    totals = defaultdict(int)
    for event in events:
        name = event.get('name')
        duration_ns = event.get('duration_ns')
        if name in WRAPPER_EVENT_NAMES or not isinstance(duration_ns, (int, float)):
            continue
        totals[name] += int(duration_ns)
    return dict(sorted(totals.items()))


def run_sequential(images, device, _args):
    events = []
    observer = list_observer(events)
    model = _build_model(device, observer=observer)
    _run_one(model, device, images[0], observer=observer)  # warm-up: lazy load, caches, allocator

    t0 = time.monotonic()
    for img in images[1:]:
        _run_one(model, device, img, observer=observer)
    wall_s = time.monotonic() - t0
    return wall_s, len(images) - 1, _aggregate_phase_ns(events)


def run_threaded(images, device, args):
    events = []
    observer = list_observer(events)
    model = _build_model(device, observer=observer)

    # Ultralytics lazily fuses conv+batchnorm in-place on the model's *first*
    # predict() call. That mutation isn't thread-safe -- two threads racing to
    # fuse the same model corrupts it (AttributeError: bn). Warm up single-
    # threaded first so every concurrent call afterward hits an already-
    # initialized predictor.
    _run_one(model, device, images[0], observer=observer)

    def _timed(img):
        _run_one(model, device, img, observer=observer)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_timed, images[1:]))  # warm-up image already run
    wall_s = time.monotonic() - t0
    return wall_s, len(images) - 1, _aggregate_phase_ns(events)


_mp_model = None
_mp_device = None
_mp_events = None


def _mp_init(device):
    # perform_inference/load_model events fire via the model's *own* self.observer
    # (fixed at construction), not whatever observer a caller passes to
    # pipeline() later - so it must be wired in here, not per-call.
    global _mp_model, _mp_device, _mp_events
    _mp_device = device
    _mp_events = []
    observer = list_observer(_mp_events)
    _mp_model = _build_model(device, observer=observer)


def _mp_run_one(filepath):
    _run_one(_mp_model, _mp_device, filepath, observer=list_observer(_mp_events))


def _mp_drain_events(_ignored):
    """Fetch this worker's accumulated events back to the main process."""
    return _mp_events


def run_multiprocess(images, device, args):
    # CUDA contexts don't survive fork(): if this (main) process has already
    # initialized CUDA - e.g. a 'sequential' or 'threaded' mode ran first in the
    # same invocation - forked workers inherit a broken copy and crash on
    # model.to(device). 'spawn' avoids that crash but re-imports torch/ultralytics
    # from scratch per worker, which is ~3.5x slower in steady state, not just at
    # startup (measured: 2.0s/img fork vs 7.2s/img spawn, identical images/procs).
    # So instead: main() always runs 'multiprocess' first, before any other mode
    # touches CUDA in this process, keeping plain fork safe *and* fast.
    with Pool(processes=args.processes, initializer=_mp_init, initargs=(device,)) as pool:
        pool.map(_mp_run_one, images[: args.processes])  # warm every worker once, untimed
        t0 = time.monotonic()
        pool.map(_mp_run_one, images)
        wall_s = time.monotonic() - t0
        # one drain task per worker to collect each worker's accumulated events
        per_worker_events = pool.map(_mp_drain_events, range(args.processes))
    all_events = [e for evs in per_worker_events for e in evs]
    return wall_s, len(images), _aggregate_phase_ns(all_events)


MODES = {
    'sequential': run_sequential,
    'threaded': run_threaded,
    'multiprocess': run_multiprocess,
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--images', default='examples', help='dir of images to benchmark')
    ap.add_argument('--device', default='cuda:0', help='torch device')
    ap.add_argument(
        '--workers', type=int, default=8, help='thread count for threaded mode'
    )
    ap.add_argument(
        '--processes', type=int, default=8, help='process count for multiprocess mode'
    )
    ap.add_argument(
        '--modes',
        nargs='+',
        choices=list(MODES),
        default=list(MODES),
        help='which execution modes to benchmark',
    )
    ap.add_argument(
        '--summary-json', default=None, help='optional path for a machine-readable JSON summary'
    )
    args = ap.parse_args(argv)

    image_dir = Path(args.images)
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    if not images:
        raise SystemExit(f'No images found in {image_dir}')
    if len(images) < 2:
        raise SystemExit('Need at least 2 images (first is discarded as warm-up)')
    _validate_images(images)

    print(f'Images: {len(images)} (first is warm-up, excluded from timing)')
    print(f'Device: {args.device}')
    print(f'Modes: {", ".join(args.modes)}\n')

    # 'multiprocess' must run before any mode that touches CUDA in this (main)
    # process - sequential/threaded both load a CUDA model here, and forking
    # afterward would inherit a broken CUDA context (see run_multiprocess).
    run_order = sorted(args.modes, key=lambda m: 0 if m == 'multiprocess' else 1)

    results = {}
    for mode in run_order:
        wall_s, count, phase_ns = MODES[mode](images, args.device, args)
        mean_ms = (wall_s * 1000 / count) if count else float('nan')
        results[mode] = {
            'mean_ms_per_image': mean_ms,
            'wall_s': wall_s,
            'images_timed': count,
            'phase_totals_ns': phase_ns,
        }
    results = {mode: results[mode] for mode in args.modes}  # restore requested order for display

    base = results.get('sequential', {}).get('mean_ms_per_image')

    header = f'  {"mode":<14} {"mean ms/img":>12} {"speedup":>9}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for mode, r in results.items():
        ms = r['mean_ms_per_image']
        speedup = f'{base / ms:.2f}x' if base and ms else '-'
        print(f'  {mode:<14} {ms:>12.1f} {speedup:>9}')

    print('\n  phase totals (timed runs, summed across all images/workers):')
    for mode, r in results.items():
        phase_bits = ', '.join(
            f'{name}={ns / 1e6:.1f}ms' for name, ns in r['phase_totals_ns'].items()
        )
        print(f'  {mode:<14} {phase_bits}')

    if args.summary_json is not None:
        Path(args.summary_json).write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')

    return results


if __name__ == '__main__':
    main()
