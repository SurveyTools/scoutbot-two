# -*- coding: utf-8 -*-
"""
Full-image benchmark: PyTorch (cpu / mps) vs Core ML localizers.

Runs the *complete* SAHI-tiled `scoutbot.pipeline()` over the images in `examples/`
for each variant, and reports:

  - wall-clock per image (mean, with a discarded warm-up run), and
  - detection parity of the Core ML run against the PyTorch-CPU reference
    (worst box center delta in px, worst confidence delta, unmatched counts).

Core ML's win on Apple Silicon is mainly ANE energy efficiency, not necessarily
raw wall-clock vs MPS -- so all three are reported side by side. While the Core ML
rows run, watch Activity Monitor's "Neural Engine" column: 0% ANE load means it fell
back to GPU/CPU (FP16 / static-shape regression).

Usage:
    uv run python benchmarks/benchmark_coreml.py
    uv run python benchmarks/benchmark_coreml.py --family cls --images examples
"""
import argparse
import time
from pathlib import Path

import scoutbot
from scoutbot import loc, tile_batched

# (label, pt_config, coreml_config)
FAMILIES = {
    'onecls': ('v3', 'v3-coreml'),
    'cls': ('v3-cls', 'v3-cls-coreml'),
}

# (variant label, config-kind, torch device). config-kind picks pt vs coreml config.
VARIANTS = [
    ('pt-cpu', 'pt', 'cpu'),
    ('pt-mps', 'pt', 'mps'),
    ('coreml', 'coreml', 'cpu'),  # device ignored; forced to cpu inside the model
]

LOC_THRESH = 0.45


def _build_model(config, device):
    model_path = loc.fetch(config=config)
    return tile_batched.Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=LOC_THRESH,
        device=device,
    )


def _run_variant(config, device, images):
    """Return (mean_ms_per_image, {image_path: detections})."""
    model = _build_model(config, device)
    timings = []
    detections = {}
    for idx, img in enumerate(images):
        t0 = time.perf_counter()
        _wic, dets = scoutbot.pipeline(
            str(img),
            config=config,
            batched_detection_model=model,
            backend_device=device,
            loc_thresh=LOC_THRESH,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        detections[str(img)] = dets
        if idx > 0:  # discard the first run (warm-up: lazy load, caches, allocator)
            timings.append(elapsed_ms)
    mean_ms = sum(timings) / len(timings) if timings else float('nan')
    return mean_ms, detections


def _center(d):
    return (d['x'] + d['w'] / 2.0, d['y'] + d['h'] / 2.0)


# A match beyond this center distance (px) is not the same detection -- it's the
# greedy matcher pairing a leftover box when the true partner crossed the FP16
# confidence threshold and dropped out. Gate so worst_px reflects real matches and
# the difference shows up honestly as an unmatched count. FP16 nudges boxes <~2px,
# so any sane gate cleanly separates true matches from artifacts.
MATCH_GATE_PX = 16.0


def _parity(ref_dets, test_dets):
    """Gated greedy center-distance match of test boxes to the reference set.

    Returns (worst_center_px, worst_conf_delta, n_unmatched_ref, n_unmatched_test)
    where worst_* are computed over matched pairs only.
    """
    worst_px = 0.0
    worst_conf = 0.0
    unmatched_ref = 0
    remaining = list(test_dets)
    for r in ref_dets:
        rx, ry = _center(r)
        best_i, best_d = None, None
        for i, t in enumerate(remaining):
            tx, ty = _center(t)
            d = ((rx - tx) ** 2 + (ry - ty) ** 2) ** 0.5
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_i is None or best_d > MATCH_GATE_PX:
            unmatched_ref += 1
            continue
        match = remaining.pop(best_i)
        worst_px = max(worst_px, best_d)
        worst_conf = max(worst_conf, abs(r['c'] - match['c']))
    return worst_px, worst_conf, unmatched_ref, len(remaining)


def benchmark_family(label, pt_config, coreml_config, images):
    print(f'\n=== Family: {label}  ({pt_config} vs {coreml_config}) ===')
    print(f'Images: {len(images)} (first is warm-up, excluded from timing)\n')

    results = {}
    detections_by_variant = {}
    for vlabel, kind, device in VARIANTS:
        config = pt_config if kind == 'pt' else coreml_config
        try:
            mean_ms, dets = _run_variant(config, device, images)
        except Exception as exc:  # noqa: BLE001 - report and continue (e.g. no MPS)
            print(f'  {vlabel:<8} SKIPPED: {type(exc).__name__}: {exc}')
            continue
        results[vlabel] = mean_ms
        detections_by_variant[vlabel] = dets

    ref = detections_by_variant.get('pt-cpu')

    header = f'  {"variant":<8} {"mean ms/img":>12} {"speedup":>9}  parity vs pt-cpu'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    base = results.get('pt-cpu')
    for vlabel, _, _ in VARIANTS:
        if vlabel not in results:
            continue
        ms = results[vlabel]
        speedup = f'{base / ms:.2f}x' if base and ms else '-'
        parity = ''
        if ref is not None and vlabel != 'pt-cpu':
            worst_px = worst_conf = 0.0
            tot_unmatched = 0
            for img in detections_by_variant[vlabel]:
                wp, wc, ur, ut = _parity(ref[img], detections_by_variant[vlabel][img])
                worst_px = max(worst_px, wp)
                worst_conf = max(worst_conf, wc)
                tot_unmatched += ur + ut
            parity = (
                f'worst {worst_px:.2f}px / {worst_conf:.4f} conf, '
                f'{tot_unmatched} unmatched'
            )
        print(f'  {vlabel:<8} {ms:>12.1f} {speedup:>9}  {parity}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', default='examples', help='dir of images to benchmark')
    ap.add_argument(
        '--family',
        choices=list(FAMILIES) + ['all'],
        default='all',
        help='which detector family to benchmark',
    )
    args = ap.parse_args()

    image_dir = Path(args.images)
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    if not images:
        raise SystemExit(f'No images found in {image_dir}')

    families = FAMILIES if args.family == 'all' else {args.family: FAMILIES[args.family]}
    for label, (pt_config, coreml_config) in families.items():
        benchmark_family(label, pt_config, coreml_config, images)


if __name__ == '__main__':
    main()
