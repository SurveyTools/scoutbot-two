"""Multiprocess batch orchestration for scoutbot.batch_resumable(processes=N).

Split into a GPU-less orchestrator/resume layer (this file's top half) and a
forkserver-based inference submitter (added in a later task). Kept private.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

from scoutbot._observability import build_batch_summary, emit

log = logging.getLogger(__name__)


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
