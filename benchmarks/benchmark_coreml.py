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
import importlib.util
import io
import json
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import scoutbot
from scoutbot import loc, tile_batched
from scoutbot._observability import list_observer

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
WRAPPER_EVENT_NAMES = {'pipeline', 'batch', 'batch_resumable'}
_NOISY_WARNINGS = (
    (
        UserWarning,
        r"'has_mps' is deprecated, please use 'torch\.backends\.mps\.is_built\(\)'",
    ),
    (
        UserWarning,
        r'\[coremltools\] Torch version .* has not been tested with coremltools',
    ),
    (SyntaxWarning, r'.*invalid escape sequence.*'),
)


def read_event_stream(path):
    """Read a JSON Lines event stream into a list of dicts."""
    events = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def aggregate_events(events, total_wall_ns):
    """Aggregate observability events into a benchmark summary dict."""
    phase_totals_ns = defaultdict(int)
    phase_counts = defaultdict(int)
    compute_units_requested = set()
    compute_units_actual = set()
    tile_count = 0
    model_invocations = 0
    event_count = 0
    observed_duration_ns = 0

    for event in events:
        event_count += 1
        name = event.get('name')
        duration_ns = event.get('duration_ns')
        if name and isinstance(duration_ns, (int, float)):
            if name not in WRAPPER_EVENT_NAMES:
                phase_totals_ns[name] += int(duration_ns)
                phase_counts[name] += 1
                observed_duration_ns += int(duration_ns)
        if name in WRAPPER_EVENT_NAMES:
            continue

        if name == 'perform_inference':
            model_invocations += 1
            tile_count += int(event.get('tile_count') or 0)

        requested = event.get('compute_units_requested')
        actual = event.get('compute_units_actual')
        if requested is not None:
            compute_units_requested.add(requested)
        if actual is not None:
            compute_units_actual.add(actual)

    summary = {
        'event_count': event_count,
        'phase_totals_ns': dict(sorted(phase_totals_ns.items())),
        'phase_counts': dict(sorted(phase_counts.items())),
        'unattributed_ns': max(int(total_wall_ns) - observed_duration_ns, 0),
        'model_invocations': model_invocations,
        'tile_count': tile_count,
        'compute_units_requested': _collapse_unique_values(compute_units_requested),
        'compute_units_actual': _collapse_unique_values(compute_units_actual),
    }
    return summary


def _collapse_unique_values(values):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _coremltools_available():
    return importlib.util.find_spec('coremltools') is not None


@contextmanager
def _suppress_benchmark_noise():
    with warnings.catch_warnings():
        for category, message in _NOISY_WARNINGS:
            warnings.filterwarnings('ignore', message=message, category=category)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield


def _write_summary(path, summary):
    path = Path(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')


def _build_model(config, device, observer=None):
    model_path = loc.fetch(config=config, observer=observer)
    return tile_batched.Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=LOC_THRESH,
        device=device,
        observer=observer,
    )


def _run_variant(config, device, images):
    """Return (mean_ms_per_image, {image_path: detections}, summary)."""
    events = []
    observer = list_observer(events)
    model = _build_model(config, device, observer=observer)
    timings = []
    warmup_ns = 0
    timed_events = []
    detections = {}
    for idx, img in enumerate(images):
        event_start = len(events)
        t0_ns = time.perf_counter_ns()
        _wic, dets = scoutbot.pipeline(
            str(img),
            config=config,
            batched_detection_model=model,
            backend_device=device,
            loc_thresh=LOC_THRESH,
            observer=observer,
        )
        elapsed_ns = time.perf_counter_ns() - t0_ns
        elapsed_ms = elapsed_ns / 1_000_000.0
        detections[str(img)] = dets
        image_events = events[event_start:]
        if idx > 0:  # discard the first run (warm-up: lazy load, caches, allocator)
            timings.append(elapsed_ms)
            timed_events.extend(image_events)
        else:
            warmup_ns = elapsed_ns
    mean_ms = sum(timings) / len(timings) if timings else float('nan')
    total_wall_ns = sum(int(round(ms * 1_000_000)) for ms in timings)
    summary = aggregate_events(timed_events, total_wall_ns=total_wall_ns)
    # fetch() (which carries compute_units_requested) fires before the timing loop
    # and is never in timed_events; pull it from the full event stream.
    if summary['compute_units_requested'] is None:
        _init_summary = aggregate_events(events, total_wall_ns=0)
        summary['compute_units_requested'] = _init_summary['compute_units_requested']
    summary['warmup_images'] = 1 if len(images) > 0 else 0
    summary['timed_images'] = max(len(images) - 1, 0)
    summary['warmup_wall_ns'] = warmup_ns
    summary['mean_ms_per_image'] = mean_ms
    summary['wall_ms'] = sum(timings)
    return mean_ms, detections, summary


def _center(d):
    return (d['x'] + d['w'] / 2.0, d['y'] + d['h'] / 2.0)


# A match beyond this center distance (px) is not the same detection -- it's the
# greedy matcher pairing a leftover box when the true partner crossed the FP16
# confidence threshold and dropped out. Gate so worst_px reflects real matches and
# the difference shows up honestly as an unmatched count. FP16 nudges boxes <~2px,
# so any sane gate cleanly separates true matches from artifacts.
MATCH_GATE_PX = 16.0

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
        extra = f' (+{len(stubs) - 3} more)' if len(stubs) > 3 else ''
        raise SystemExit(
            f'{len(stubs)} image(s) are Git LFS pointer stubs ({names}{extra}).\n'
            f'Pull real image data:\n'
            f'  brew install git-lfs   # once per machine\n'
            f'  git lfs install        # once per clone\n'
            f'  git lfs pull'
        )
    if bad:
        names = ', '.join(p.name for p in bad[:3])
        extra = f' (+{len(bad) - 3} more)' if len(bad) > 3 else ''
        raise SystemExit(
            f'{len(bad)} file(s) are not valid JPEG/PNG images: {names}{extra}'
        )


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


def benchmark_family(label, pt_config, coreml_config, images, include_coreml=True):
    print(f'\n=== Family: {label}  ({pt_config} vs {coreml_config}) ===')
    print(f'Images: {len(images)} (first is warm-up, excluded from timing)\n')

    results = {}
    detections_by_variant = {}
    summaries = {}
    variants = (
        VARIANTS
        if include_coreml
        else [variant for variant in VARIANTS if variant[0] != 'coreml']
    )
    for vlabel, kind, device in variants:
        config = pt_config if kind == 'pt' else coreml_config
        try:
            with _suppress_benchmark_noise():
                mean_ms, dets, summary = _run_variant(config, device, images)
        except Exception as exc:  # noqa: BLE001 - report and continue (e.g. no MPS)
            print(f'  {vlabel:<8} SKIPPED: {type(exc).__name__}: {exc}')
            continue
        results[vlabel] = mean_ms
        detections_by_variant[vlabel] = dets
        summaries[vlabel] = summary

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

    print('\n  phase totals (timed runs):')
    for vlabel in results:
        summary = summaries.get(vlabel, {})
        phase_totals = summary.get('phase_totals_ns', {})
        phase_bits = ', '.join(
            f'{name}={ns / 1_000_000:.1f}ms' for name, ns in phase_totals.items()
        )
        req = summary.get('compute_units_requested')
        actual = summary.get('compute_units_actual')
        if req is None:
            suffix = ''
        else:
            suffix = f' requested={req} actual={actual if actual is not None else "null"}'
        print(
            f'  {vlabel:<8} tiles={summary.get("tile_count", 0)} '
            f'invocations={summary.get("model_invocations", 0)}'
            f'{suffix}'
        )
        if phase_bits:
            print(f'    {phase_bits}')

    return {
        'family': label,
        'timings_ms': results,
        'variants': summaries,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', default='examples', help='dir of images to benchmark')
    ap.add_argument(
        '--family',
        choices=list(FAMILIES) + ['all'],
        default='all',
        help='which detector family to benchmark',
    )
    ap.add_argument(
        '--summary-json',
        default=None,
        help='optional path for a machine-readable JSON summary',
    )
    args = ap.parse_args(argv)

    image_dir = Path(args.images)
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    if not images:
        raise SystemExit(f'No images found in {image_dir}')
    _validate_images(images)

    families = FAMILIES if args.family == 'all' else {args.family: FAMILIES[args.family]}
    summaries = {}
    coremltools_available = _coremltools_available()
    if not coremltools_available:
        print('coreml variant skipped: coremltools not installed')
    for label, (pt_config, coreml_config) in families.items():
        summaries[label] = benchmark_family(
            label,
            pt_config,
            coreml_config,
            images,
            include_coreml=coremltools_available,
        )

    if args.summary_json is not None:
        _write_summary(args.summary_json, summaries)

    return summaries


if __name__ == '__main__':
    main()
