"""Multiprocess batch orchestration for scoutbot.batch_resumable(processes=N).

Split into a GPU-less orchestrator/resume layer (this file's top half) and a
forkserver-based inference submitter (added in a later task). Kept private.
"""

import json
from dataclasses import dataclass
from pathlib import Path


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
