# -*- coding: utf-8 -*-
"""Tests for private observability helpers."""

import pytest

from scoutbot._observability import (
    aggregate_phase_ns,
    build_batch_summary,
    emit,
    list_observer,
    span,
)


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


def test_build_batch_summary_core_fields():
    events = [
        {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 11},
        {'name': 'perform_inference', 'duration_ns': 1_000_000_000, 'worker_pid': 11},
        {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 12},
        {'name': 'perform_inference', 'duration_ns': 1_000_000_000, 'worker_pid': 12},
    ]
    s = build_batch_summary(
        events,
        wall_ns=2_000_000_000,
        processes=2,
        mode='multiprocess',
        start_method='forkserver',
        processed=2,
        skipped=0,
        errors=0,
        total=2,
        default_pid=999,
        base={'name': 'batch_resumable', 'config': 'v3'},
    )
    assert s['mode'] == 'multiprocess'
    assert s['processes'] == 2
    assert s['start_method'] == 'forkserver'
    assert s['processed'] == 2 and s['total'] == 2
    assert s['per_worker_counts'] == {11: 1, 12: 1}
    assert s['phase_totals_ns'] == {
        'perform_inference': 2_000_000_000,
        'sahi+post': 2_000_000_000,
    }
    # inference_duty_per_worker = inference(2e9) / (wall 2e9 * procs 2) = 0.5
    assert s['inference_duty_per_worker'] == pytest.approx(0.5)
    assert s['config'] == 'v3'


def test_build_batch_summary_defaults_missing_worker_pid():
    events = [
        {'name': 'pipeline', 'duration_ns': 1_000_000_000}
    ]  # no worker_pid (sequential)
    s = build_batch_summary(
        events,
        wall_ns=1_000_000_000,
        processes=1,
        mode='sequential',
        start_method=None,
        processed=1,
        skipped=0,
        errors=0,
        total=1,
        default_pid=555,
        base={'name': 'batch_resumable'},
    )
    assert s['per_worker_counts'] == {555: 1}
    assert s['start_method'] is None
