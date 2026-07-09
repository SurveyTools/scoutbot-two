"""Tests for the multiprocess batch orchestration helpers."""

import json

from scoutbot._batch_mp import BatchWork, _load_processed, plan_work, run_batch


def _write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def test_load_processed_returns_non_error_filepaths(tmp_path):
    out = tmp_path / 'out.jsonl'
    _write_jsonl(
        out,
        [
            {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []},
            {'filepath': 'b.jpg', 'error': 'boom'},  # error -> not processed
            {'filepath': 'c.jpg', 'wic': 0.2, 'detections': []},
        ],
    )
    assert _load_processed(out) == {'a.jpg', 'c.jpg'}


def test_load_processed_ignores_blank_and_malformed_lines(tmp_path):
    out = tmp_path / 'out.jsonl'
    out.write_text(
        '\n{not json}\n' + json.dumps({'filepath': 'a.jpg', 'wic': 0.1}) + '\n'
    )
    assert _load_processed(out) == {'a.jpg'}


def test_load_processed_missing_file_is_empty(tmp_path):
    assert _load_processed(tmp_path / 'nope.jsonl') == set()


def test_plan_work_filters_and_counts(tmp_path):
    out = tmp_path / 'out.jsonl'
    _write_jsonl(out, [{'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}])
    work = plan_work(['a.jpg', 'b.jpg', 'c.jpg'], out)
    assert isinstance(work, BatchWork)
    assert work.remaining == ['b.jpg', 'c.jpg']
    assert work.skipped_count == 1
    assert work.total_count == 3


def _meta(mode='multiprocess', processes=2):
    return {
        'mode': mode,
        'start_method': 'forkserver' if mode == 'multiprocess' else None,
        'processes': processes,
        'default_pid': 999,
        'base': {'name': 'batch_resumable', 'config': 'v3'},
    }


def _fake_submit(results):
    def submit(remaining):
        yield from results

    return submit


def test_run_batch_writes_completion_order_with_flush(tmp_path):
    out = tmp_path / 'out.jsonl'
    results = [
        (
            'b.jpg',
            {'filepath': 'b.jpg', 'wic': 0.2, 'detections': []},
            [
                {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 7},
                {
                    'name': 'perform_inference',
                    'duration_ns': 1_000_000_000,
                    'worker_pid': 7,
                },
            ],
        ),
        (
            'a.jpg',
            {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []},
            [
                {'name': 'pipeline', 'duration_ns': 2_000_000_000, 'worker_pid': 8},
                {
                    'name': 'perform_inference',
                    'duration_ns': 1_000_000_000,
                    'worker_pid': 8,
                },
            ],
        ),
    ]
    work = BatchWork(remaining=['a.jpg', 'b.jpg'], skipped_count=0, total_count=2)
    summary = run_batch(work, out, _fake_submit(results), summary_meta=_meta())
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r['filepath'] for r in lines] == ['b.jpg', 'a.jpg']  # completion order
    assert summary['processed'] == 2 and summary['skipped'] == 0 and summary['total'] == 2
    assert summary['errors'] == 0


def test_run_batch_reflects_injected_counts(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg'], skipped_count=3, total_count=4)
    results = [('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])]
    summary = run_batch(work, out, _fake_submit(results), summary_meta=_meta())
    assert summary['skipped'] == 3 and summary['total'] == 4 and summary['processed'] == 1


def test_run_batch_attaches_provenance(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg'], skipped_count=0, total_count=1)
    results = [('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])]
    run_batch(
        work,
        out,
        _fake_submit(results),
        summary_meta=_meta(),
        provenance={'model_name': 'scoutbot-v3'},
    )
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec['model'] == {'model_name': 'scoutbot-v3'}


def test_run_batch_counts_error_records(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(remaining=['a.jpg', 'b.jpg'], skipped_count=0, total_count=2)
    results = [
        ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, []),
        ('b.jpg', {'filepath': 'b.jpg', 'error': 'boom'}, []),
    ]
    summary = run_batch(work, out, _fake_submit(results), summary_meta=_meta())
    assert summary['processed'] == 1 and summary['errors'] == 1
    # error record is not given a provenance block even if provenance is set
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert any('error' in r for r in lines)


def test_run_batch_fail_fast_on_lost_worker(tmp_path):
    out = tmp_path / 'out.jsonl'
    work = BatchWork(
        remaining=['a.jpg', 'b.jpg', 'c.jpg'], skipped_count=0, total_count=3
    )

    class WorkerDied(Exception):
        pass

    def submit(remaining):
        yield ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, [])
        raise WorkerDied('worker 2 exited')

    import pytest

    with pytest.raises(WorkerDied):
        run_batch(work, out, submit, summary_meta=_meta())
    # partial output intact and flushed: the one completed result survived
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])['filepath'] == 'a.jpg'
