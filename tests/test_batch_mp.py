"""Tests for the multiprocess batch orchestration helpers."""

import json
import multiprocessing
import sys
from unittest.mock import MagicMock

import pytest

from scoutbot._batch_mp import (
    BatchWork,
    _load_processed,
    _parse_result_timeout,
    make_mp_submit,
    plan_work,
    run_batch,
)


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


def test_parse_result_timeout_falls_back_on_invalid_value(monkeypatch):
    monkeypatch.setenv('SCOUTBOT_MP_RESULT_TIMEOUT', 'not-a-number')
    assert _parse_result_timeout() == 600.0


def test_parse_result_timeout_accepts_valid_value(monkeypatch):
    monkeypatch.setenv('SCOUTBOT_MP_RESULT_TIMEOUT', '42')
    assert _parse_result_timeout() == 42.0


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
    summary = run_batch(
        work,
        out,
        _fake_submit(results),
        summary_meta=_meta(),
        provenance={'model_name': 'scoutbot-v3'},
    )
    assert summary['processed'] == 1 and summary['errors'] == 1
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert any('error' in r for r in lines)
    success_record = next(r for r in lines if 'error' not in r)
    error_record = next(r for r in lines if 'error' in r)
    # success record gets the provenance block ...
    assert success_record['model'] == {'model_name': 'scoutbot-v3'}
    # ... but an error record is NOT given a provenance block, even when set
    assert 'model' not in error_record


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

    with pytest.raises(WorkerDied):
        run_batch(work, out, submit, summary_meta=_meta())
    # partial output intact and flushed: the one completed result survived
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])['filepath'] == 'a.jpg'


class _FakeAsyncResultIter:
    """Stand-in for ``multiprocessing.pool.IMapUnorderedIterator``.

    ``submit`` only ever calls ``.next(timeout=...)`` on the real iterator, so
    that is the only method this fake needs. Items are yielded in order;
    an item that is an ``Exception`` instance is raised instead of returned,
    which lets a test script a mid-stream ``TimeoutError``.
    """

    def __init__(self, items):
        self._items = list(items)
        self._idx = 0

    def next(self, timeout=None):
        if self._idx >= len(self._items):
            raise StopIteration
        item = self._items[self._idx]
        self._idx += 1
        if isinstance(item, Exception):
            raise item
        return item


def _fake_ctx(fake_iter):
    """A MagicMock standing in for multiprocessing.get_context('forkserver').

    Satisfies ``ctx.set_forkserver_preload(...)``, the ``with ctx.Pool(...) as
    pool:`` context-manager protocol, and ``pool.imap_unordered(...)``
    returning the caller-supplied fake iterator -- no real pool/process is
    ever created.
    """
    pool = MagicMock()
    pool.imap_unordered.return_value = fake_iter
    pool_cm = MagicMock()
    pool_cm.__enter__.return_value = pool
    pool_cm.__exit__.return_value = False
    ctx = MagicMock()
    ctx.Pool.return_value = pool_cm
    return ctx


def _make_submit(monkeypatch, fake_iter):
    ctx = _fake_ctx(fake_iter)
    monkeypatch.setattr(
        'scoutbot._batch_mp.multiprocessing.get_context', lambda name: ctx
    )
    return make_mp_submit(
        model_path='/fake',
        effective_run_config={},
        config_name='v3',
        backend_device='cpu',
        processes=2,
    )


def test_mp_submit_yields_results_in_completion_order(monkeypatch):
    results = [
        ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, []),
        ('b.jpg', {'filepath': 'b.jpg', 'wic': 0.2, 'detections': []}, []),
    ]
    submit = _make_submit(monkeypatch, _FakeAsyncResultIter(results))
    assert list(submit(['a.jpg', 'b.jpg'])) == results


def test_mp_submit_converts_result_timeout_to_fail_fast_runtime_error(monkeypatch):
    # One good result, then a stalled/dead worker signaled by TimeoutError.
    items = [
        ('a.jpg', {'filepath': 'a.jpg', 'wic': 0.1, 'detections': []}, []),
        multiprocessing.TimeoutError(),
    ]
    submit = _make_submit(monkeypatch, _FakeAsyncResultIter(items))
    with pytest.raises(RuntimeError, match='deadline'):
        list(submit(['a.jpg', 'b.jpg']))


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux', reason='forkserver path is Linux/CUDA-first')
def test_mp_submit_end_to_end_over_examples(sample_image, tmp_path):
    from scoutbot import model_store

    out = tmp_path / 'out.jsonl'
    model_path = model_store.fetch(config='v3')
    effective = {
        'confidence_threshold': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    }
    submit = make_mp_submit(
        model_path=model_path,
        effective_run_config=effective,
        config_name='v3',
        backend_device='cpu',
        processes=2,
    )
    work = plan_work([sample_image, sample_image], out)
    meta = {
        'mode': 'multiprocess',
        'start_method': 'forkserver',
        'processes': 2,
        'default_pid': 0,
        'base': {'name': 'batch_resumable', 'config': 'v3'},
    }
    summary = run_batch(work, out, submit, summary_meta=meta)
    assert summary['processed'] >= 1
    assert out.read_text().strip() != ''
