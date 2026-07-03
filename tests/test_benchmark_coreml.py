# -*- coding: utf-8 -*-
"""Tests for benchmarks/benchmark_coreml.py helpers (no model/network needed)."""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_benchmark():
    """Import _validate_images from the benchmark script without executing main()."""
    bench = Path(__file__).parent.parent / 'benchmarks' / 'benchmark_coreml.py'
    spec = importlib.util.spec_from_file_location('benchmark_coreml', bench)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['benchmark_coreml'] = mod
    spec.loader.exec_module(mod)
    return mod


class TestValidateImages:
    """Unit tests for _validate_images() in the benchmark script."""

    @pytest.fixture(autouse=True)
    def benchmark_mod(self):
        self.mod = _load_benchmark()

    def test_raises_on_lfs_stub(self, tmp_path):
        """Git LFS pointer stubs must produce a clear SystemExit with git lfs pull hint."""
        stub = tmp_path / 'lfs.jpg'
        stub.write_bytes(
            b'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3391180\n'
        )

        with pytest.raises(SystemExit, match='Git LFS pointer'):
            self.mod._validate_images([stub])

    def test_raises_on_non_image_file(self, tmp_path):
        """Files with a .jpg extension that are not images must raise SystemExit."""
        bad = tmp_path / 'notanimage.jpg'
        bad.write_bytes(b'This is just a text file, not an image at all.\n')

        with pytest.raises(SystemExit, match='not valid JPEG/PNG'):
            self.mod._validate_images([bad])

    def test_passes_for_valid_jpeg(self, tmp_path):
        """Files starting with JPEG magic bytes must pass silently."""
        jpg = tmp_path / 'valid.jpg'
        # Minimal JPEG header: SOI marker (FF D8) + APP0 marker (FF E0)
        jpg.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b'\x00' * 64)

        self.mod._validate_images([jpg])  # must not raise

    def test_passes_for_valid_png(self, tmp_path):
        """Files starting with PNG magic bytes must pass silently."""
        png = tmp_path / 'valid.png'
        png.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 64)

        self.mod._validate_images([png])  # must not raise

    def test_lfs_error_mentions_git_commands(self, tmp_path):
        """The LFS error message must include the remediation commands."""
        stub = tmp_path / 'lfs.jpg'
        stub.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')

        with pytest.raises(SystemExit) as exc_info:
            self.mod._validate_images([stub])

        message = str(exc_info.value)
        assert 'git lfs pull' in message

    def test_reports_multiple_stubs(self, tmp_path):
        """When multiple stubs are found the count must appear in the error message."""
        for i in range(3):
            p = tmp_path / f'stub{i}.jpg'
            p.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')

        with pytest.raises(SystemExit) as exc_info:
            self.mod._validate_images(list(tmp_path.glob('*.jpg')))

        message = str(exc_info.value)
        assert '3' in message


def test_aggregate_events_sums_phases_and_unattributed():
    bench = _load_benchmark()

    summary = bench.aggregate_events(
        [
            {'name': 'slice_generation', 'duration_ns': 10},
            {'name': 'per_tile_prediction', 'duration_ns': 20},
            {'name': 'per_tile_prediction', 'duration_ns': 5},
        ],
        total_wall_ns=40,
    )

    assert summary['phase_totals_ns']['slice_generation'] == 10
    assert summary['phase_totals_ns']['per_tile_prediction'] == 25
    assert summary['unattributed_ns'] == 5


def test_parse_jsonl_event_stream_round_trips(tmp_path):
    bench = _load_benchmark()
    path = tmp_path / 'events.jsonl'
    path.write_text(
        '{"name": "pipeline", "duration_ns": 10}\n'
        '{"name": "fetch", "duration_ns": 20}\n'
    )

    events = bench.read_event_stream(path)

    assert events == [
        {'name': 'pipeline', 'duration_ns': 10},
        {'name': 'fetch', 'duration_ns': 20},
    ]
