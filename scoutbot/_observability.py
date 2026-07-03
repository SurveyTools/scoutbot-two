"""Private observability helpers for ScoutBot runtime hooks."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

log = logging.getLogger(__name__)

ObservabilityEvent = dict[str, Any]


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
            observer(dict(event))
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
