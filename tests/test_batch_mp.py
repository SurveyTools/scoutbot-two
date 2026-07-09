"""Tests for the multiprocess batch orchestration helpers."""

import json

from scoutbot._batch_mp import BatchWork, _load_processed, plan_work


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
