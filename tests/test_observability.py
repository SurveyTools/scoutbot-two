# -*- coding: utf-8 -*-
"""Tests for private observability helpers."""

from scoutbot._observability import aggregate_phase_ns, emit, list_observer, span


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
    for key in (
        'name',
        'started_ns',
        'ended_ns',
        'duration_ns',
        'status',
        'config',
        'backend',
        'device',
    ):
        assert key in event
    assert event['name'] == 'pipeline'
    assert event['status'] == 'ok'
    assert event['config'] == 'v3'


def test_aggregate_phase_ns_sums_children_and_derives_sahi_post():
    events = [
        {'name': 'perform_inference', 'duration_ns': 600},
        {'name': 'perform_inference', 'duration_ns': 400},
        {'name': 'pipeline', 'duration_ns': 2000},
    ]
    # sahi+post = pipeline(2000) - children(1000) = 1000
    assert aggregate_phase_ns(events) == {'perform_inference': 1000, 'sahi+post': 1000}


def test_aggregate_phase_ns_ignores_outer_batch_wrapper():
    events = [
        {'name': 'perform_inference', 'duration_ns': 1000},
        {'name': 'pipeline', 'duration_ns': 2000},
        {'name': 'batch_resumable', 'duration_ns': 999999},  # must NOT affect sahi+post
    ]
    assert aggregate_phase_ns(events) == {'perform_inference': 1000, 'sahi+post': 1000}


def test_aggregate_phase_ns_negative_guard():
    events = [
        {'name': 'perform_inference', 'duration_ns': 5000},
        {'name': 'pipeline', 'duration_ns': 2000},
    ]
    assert aggregate_phase_ns(events) == {'perform_inference': 5000, 'sahi+post': 0}
