"""Multiprocess batch orchestration for scoutbot.batch_resumable(processes=N).

Split into a GPU-less orchestrator/resume layer (this file's top half) and a
forkserver-based inference submitter (bottom half). Kept private.
"""

import json
import logging
import multiprocessing
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from scoutbot._observability import build_batch_summary, emit

log = logging.getLogger(__name__)


# Characterized on darwin/forkserver (Task 1, DL-5): a hard worker death (e.g.
# os._exit) is NOT surfaced by Pool.imap_unordered -- the main process blocks
# forever waiting for the dead worker's result (HANG/SILENT finding). The
# bounded-wait fallback below converts that silent hang into a raised error
# after RESULT_TIMEOUT_S so a lost worker is treated as fail-fast, matching
# run_batch's write-integrity contract.
def _parse_result_timeout():
    raw = os.getenv('SCOUTBOT_MP_RESULT_TIMEOUT', '600')
    try:
        return float(raw)
    except ValueError:
        log.warning('Invalid SCOUTBOT_MP_RESULT_TIMEOUT=%r; using default 600s', raw)
        return 600.0


RESULT_TIMEOUT_S = _parse_result_timeout()


def _effective_processes(processes):
    """Clamp the requested worker count to what the platform can actually run.

    Multiprocess batch mode uses the ``forkserver`` start method (Linux/CUDA-
    first). Spawn-only platforms (e.g. Windows) have no forkserver context, so
    ``get_context('forkserver')`` would raise mid-run. Fall back to sequential
    and warn instead of crashing -- callers that auto-enable ``--processes``
    (e.g. Habitat Hub) keep working, just single-process.
    """
    if processes > 1 and 'forkserver' not in multiprocessing.get_all_start_methods():
        log.warning(
            "--processes=%d requested but the 'forkserver' start method is "
            'unavailable on this platform; running sequentially. Multiprocess '
            'batch mode is Linux/CUDA-first.',
            processes,
        )
        return 1
    return processes


@dataclass(frozen=True)
class BatchWork:
    """One resume-filtering result, kept coherent as a single value."""

    remaining: list  # list[str], already resume-filtered
    skipped_count: int  # len(filepaths) - len(remaining)
    total_count: int  # len(filepaths)


def _load_processed(output_path):
    """Return the set of filepaths already successfully processed.

    Reads the resumable JSONL at ``output_path``; a record counts as processed
    only if it has a ``filepath`` and no ``error`` key. Blank and malformed
    lines are ignored. A missing file yields an empty set.
    """
    output_path = Path(output_path)
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'filepath' in record and 'error' not in record:
                processed.add(record['filepath'])
    return processed


def plan_work(filepaths, output_path):
    """Load prior progress and return the BatchWork for this run."""
    processed = _load_processed(output_path)
    remaining = [fp for fp in filepaths if fp not in processed]
    return BatchWork(
        remaining=remaining,
        skipped_count=len(filepaths) - len(remaining),
        total_count=len(filepaths),
    )


def run_batch(
    work,
    output_path,
    submit,
    *,
    summary_meta,
    provenance=None,
    on_progress=None,
    observer=None,
):
    """Write-integrity orchestrator: one writer, completion order, fail-fast.

    ``submit(work.remaining)`` yields ``(filepath, record_or_error, events)`` in
    completion order. Success records (those without an ``error`` key) get the
    ``provenance`` block attached. Only completed, flushed results are ever
    written, so a mid-stream exception (a lost worker) leaves a valid,
    resumable file.
    """
    output_path = Path(output_path)
    started_ns = perf_counter_ns()
    processed_count = 0
    error_count = 0
    all_events = []

    try:
        with open(output_path, 'a') as f:
            for filepath, record, events in submit(work.remaining):
                all_events.extend(events)
                is_error = 'error' in record
                if not is_error and provenance is not None:
                    record['model'] = provenance
                f.write(json.dumps(record) + '\n')
                f.flush()
                if is_error:
                    error_count += 1
                    log.error(f'Failed to process {filepath}: {record.get("error")}')
                else:
                    processed_count += 1
                if on_progress is not None:
                    on_progress(filepath, record)
    except Exception:
        log.error(
            f'Batch aborted after a lost worker: {processed_count} done, '
            f'{len(work.remaining) - processed_count - error_count} remaining '
            f'- re-run with the same output to resume'
        )
        raise

    wall_ns = perf_counter_ns() - started_ns
    summary_event = build_batch_summary(
        all_events,
        wall_ns=wall_ns,
        processes=summary_meta['processes'],
        mode=summary_meta['mode'],
        start_method=summary_meta['start_method'],
        processed=processed_count,
        skipped=work.skipped_count,
        errors=error_count,
        total=work.total_count,
        default_pid=summary_meta['default_pid'],
        base=summary_meta['base'],
    )
    emit(observer, summary_event)
    return {
        'processed': processed_count,
        'skipped': work.skipped_count,
        'errors': error_count,
        'total': work.total_count,
        'summary': summary_event,
    }


# --- forkserver submitter + worker -----------------------------------------
#
# Each worker process loads exactly one model (in _mp_init) from the
# parent-resolved model_path, then processes images one at a time via
# _worker_process_image. Per-image failures become error records (retried on
# resume); only a hard process death is allowed to reach the orchestrator as
# an exception, which run_batch treats as fail-fast.

_WORKER: dict = {}
_WORKER_EVENTS: list = []  # per-worker; model perform_inference + pipeline events land here


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _mp_init(model_path, effective_run_config, config_name, backend_device):
    # Runs once per worker. The model's own observer is fixed at construction,
    # so wire it to this worker's event list; the per-call observer below reuses
    # the same list. model_path was resolved once by the parent (DL-2).
    from scoutbot import tile_batched
    from scoutbot._observability import list_observer

    observer = list_observer(_WORKER_EVENTS)
    model = tile_batched.Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=effective_run_config.get('confidence_threshold', 0.45),
        device=backend_device,
        observer=observer,
    )
    _WORKER.clear()
    _WORKER.update(
        model=model,
        run_config=effective_run_config,
        config_name=config_name,
        device=backend_device,
        observer=observer,
        pid=os.getpid(),
    )


def _worker_process_image(filepath):
    import scoutbot

    rc = _WORKER['run_config']
    _WORKER_EVENTS.clear()  # bound memory: only this image's events accumulate
    try:
        wic_, detects = scoutbot.pipeline(
            filepath,
            _WORKER['config_name'],
            batched_detection_model=_WORKER['model'],
            backend_device=_WORKER['device'],
            loc_thresh=rc.get('confidence_threshold', 0.45),
            slice_height=rc.get('slice_height', 512),
            slice_width=rc.get('slice_width', 512),
            overlap_height_ratio=rc.get('overlap_height_ratio', 0.25),
            overlap_width_ratio=rc.get('overlap_width_ratio', 0.25),
            perform_standard_pred=rc.get('perform_standard_pred', False),
            postprocess_class_agnostic=rc.get('postprocess_class_agnostic', True),
            observer=_WORKER['observer'],
        )
        record = {
            'filepath': filepath,
            'wic': wic_,
            'detections': detects,
            'timestamp': _utcnow_iso(),
            'config': _WORKER['config_name'],
        }
    except Exception as exc:  # per-image failure -> error record (retried on resume)
        record = {
            'filepath': filepath,
            'error': str(exc),
            'timestamp': _utcnow_iso(),
            'config': _WORKER['config_name'],
        }
    events = list(_WORKER_EVENTS)
    for event in events:
        event['worker_pid'] = _WORKER['pid']
    return filepath, record, events


def make_mp_submit(
    *, model_path, effective_run_config, config_name, backend_device, processes
):
    """Return a completion-order ``submit`` backed by a forkserver pool.

    A local forkserver context is used (never set_start_method, which would
    mutate the caller's global default). The heavy third-party imports are
    preloaded once in the server so forked workers inherit them instead of
    re-importing them per worker.

    Task 1's characterization (darwin, forkserver) found that a hard worker
    death is not surfaced by plain ``imap_unordered`` -- the main process
    hangs forever waiting on the dead worker's result. ``submit`` therefore
    polls with a bounded per-result timeout and converts a stalled result
    into a raised error so run_batch's fail-fast contract still holds.
    """

    def submit(remaining):
        ctx = multiprocessing.get_context('forkserver')
        # Preload ONLY the heavy third-party imports whose per-worker cost we are
        # eliminating (this is the one place we touch preload; it configures this
        # process's forkserver). Keep the list minimal: 'scoutbot' is deliberately
        # NOT preloaded - it is imported lazily in _mp_init/_worker and is cheap
        # once torch is already loaded. Importing torch here does not initialize
        # CUDA (that stays lazy, per worker, after fork).
        ctx.set_forkserver_preload(['torch', 'ultralytics'])
        with ctx.Pool(
            processes=processes,
            initializer=_mp_init,
            initargs=(model_path, effective_run_config, config_name, backend_device),
        ) as pool:
            it = pool.imap_unordered(_worker_process_image, remaining, chunksize=1)
            while True:
                try:
                    yield it.next(timeout=RESULT_TIMEOUT_S)
                except StopIteration:
                    break
                except multiprocessing.TimeoutError as exc:
                    raise RuntimeError(
                        'worker exceeded result deadline (treated as dead)'
                    ) from exc

    return submit
