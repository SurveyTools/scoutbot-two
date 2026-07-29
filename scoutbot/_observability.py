"""Private observability helpers for ScoutBot runtime hooks."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

log = logging.getLogger(__name__)

ObservabilityEvent = dict[str, Any]

OUTER_WRAPPER_NAMES = frozenset({'batch', 'batch_resumable'})


class Observer(Protocol):
    def emit(self, event: Mapping[str, Any], /) -> None: ...


class ListObserver:
    """Collect emitted events into a list."""

    def __init__(self, events: list[ObservabilityEvent]):
        self.events = events

    def emit(self, event: Mapping[str, Any], /) -> None:
        self.events.append(dict(event))


def noop_observer(event: Mapping[str, Any], /) -> None:
    """Default observer that discards events."""


def list_observer(events: list[ObservabilityEvent]) -> Observer:
    """Collect emitted events into the provided list."""
    return ListObserver(events)


def emit(observer: Observer | None, event: Mapping[str, Any]) -> None:
    """Safely send an event to an observer without affecting runtime behavior."""
    if observer is None:
        return

    try:
        if hasattr(observer, 'emit'):
            observer.emit(dict(event))
        else:
            observer(dict(event))  # type: ignore[operator]
    except Exception:  # pragma: no cover - observer failures are intentionally swallowed
        log.debug('Observability observer failed', exc_info=True)


@contextmanager
def span(observer: Observer | None, name: str, **fields: Any):
    """Emit a span event with perf-counter timing on exit."""
    started_ns = time.perf_counter_ns()
    try:
        yield started_ns
    except Exception as exc:
        ended_ns = time.perf_counter_ns()
        emit(
            observer,
            {
                'name': name,
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'status': 'error',
                'error_type': type(exc).__name__,
                **fields,
            },
        )
        raise
    else:
        ended_ns = time.perf_counter_ns()
        emit(
            observer,
            {
                'name': name,
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'status': 'ok',
                **fields,
            },
        )


def dumps_jsonl(events: Iterable[Mapping[str, Any]]) -> str:
    """Serialize an iterable of event mappings to JSON Lines."""
    return ''.join(json.dumps(dict(event), sort_keys=True) + '\n' for event in events)


def write_jsonl(path: str | Path, events: Iterable[Mapping[str, Any]]) -> None:
    """Write event mappings to a JSON Lines file."""
    Path(path).write_text(dumps_jsonl(events))


def aggregate_phase_ns(events: list[dict[str, Any]]) -> dict[str, int]:
    """Sum event durations by phase and derive the CPU pre/post bucket.

    Named child phases (perform_inference, load_model, ...) are summed
    directly. Only the ``pipeline`` wrapper defines the per-image span, so
    ``sahi+post = Sum(pipeline) - Sum(child phases)`` captures the pure-Python
    CPU work (SAHI slicing, NMS merge, serialization). Outer wrappers
    (``batch``/``batch_resumable``) are excluded from both the span and the
    child totals so a production stream cannot double-count the outer batch.
    """
    totals: dict[str, int] = defaultdict(int)
    pipeline_ns = 0
    for event in events:
        name = event.get('name')
        duration_ns = event.get('duration_ns')
        if not isinstance(name, str):
            continue
        if not isinstance(duration_ns, (int, float)):
            continue
        if name == 'pipeline':
            pipeline_ns += int(duration_ns)
        elif name in OUTER_WRAPPER_NAMES:
            continue
        else:
            totals[name] += int(duration_ns)
    if pipeline_ns:
        totals['sahi+post'] = max(0, pipeline_ns - sum(totals.values()))
    return dict(sorted(totals.items()))


def _per_worker_counts(events: list[dict[str, Any]], default_pid: int) -> dict[Any, int]:
    """Count ``pipeline`` events per worker PID, defaulting missing PIDs."""
    counts: dict[Any, int] = defaultdict(int)
    for event in events:
        if event.get('name') == 'pipeline':
            counts[event.get('worker_pid') or default_pid] += 1
    return dict(counts)


def build_batch_summary(
    events: list[dict[str, Any]],
    *,
    wall_ns: int,
    processes: int,
    mode: str,
    start_method: str | None,
    processed: int,
    skipped: int,
    errors: int,
    total: int,
    default_pid: int,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the enriched, machine-readable batch summary dict."""
    inference_ns = sum(
        int(e['duration_ns'])
        for e in events
        if e.get('name') == 'perform_inference'
        and isinstance(e.get('duration_ns'), (int, float))
    )
    wall_s = wall_ns / 1e9 if wall_ns else 0.0
    return {
        **base,
        'status': 'ok',
        'mode': mode,
        'processes': processes,
        'start_method': start_method,
        'wall_ns': wall_ns,
        'images_processed': processed,
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'total': total,
        'throughput_img_per_s': (processed / wall_s) if wall_s else 0.0,
        'per_worker_counts': _per_worker_counts(events, default_pid),
        'phase_totals_ns': aggregate_phase_ns(events),
        'inference_duty_per_worker': (
            inference_ns / (wall_ns * processes) if wall_ns and processes else 0.0
        ),
    }
