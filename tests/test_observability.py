# -*- coding: utf-8 -*-
"""Tests for private observability helpers."""

from scoutbot._observability import emit, list_observer, span


def test_emit_copies_event_payload():
    events = []
    observer = list_observer(events)

    payload = {'name': 'pipeline', 'status': 'ok', 'started_ns': 1}
    emit(observer, payload)
    payload['name'] = 'mutated'

    assert len(events) == 1
    assert events[0]['name'] == 'pipeline'
    assert events[0]['status'] == 'ok'


def test_span_emits_required_timing_keys():
    events = []
    observer = list_observer(events)

    with span(observer, 'pipeline', config='v3', backend='pt', device='cpu'):
        pass

    assert len(events) == 1
    event = events[0]
    for key in ('name', 'started_ns', 'ended_ns', 'duration_ns', 'status', 'config', 'backend', 'device'):
        assert key in event
    assert event['name'] == 'pipeline'
    assert event['status'] == 'ok'
    assert event['config'] == 'v3'
