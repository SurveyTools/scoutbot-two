# -*- coding: utf-8 -*-
"""Tests for benchmarks/benchmark_coreml.py helpers (no model/network needed)."""

import importlib.util
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

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
        '{"name": "pipeline", "duration_ns": 10}\n{"name": "fetch", "duration_ns": 20}\n'
    )

    events = bench.read_event_stream(path)

    assert events == [
        {'name': 'pipeline', 'duration_ns': 10},
        {'name': 'fetch', 'duration_ns': 20},
    ]


def test_missing_coremltools_is_skipped_once_at_startup(tmp_path, capsys):
    bench = _load_benchmark()

    image = tmp_path / 'image.jpg'
    image.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b'\x00' * 64)

    run_calls = []

    def _run_variant(config, device, images):
        run_calls.append((config, device))
        if config.endswith('coreml'):
            raise ModuleNotFoundError("No module named 'coremltools'")
        return (
            1.0,
            {str(image): []},
            {
                'phase_totals_ns': {},
                'phase_counts': {},
                'unattributed_ns': 0,
                'model_invocations': 0,
                'tile_count': 0,
                'compute_units_requested': None,
                'compute_units_actual': None,
            },
        )

    with (
        patch.object(bench, '_run_variant', side_effect=_run_variant),
        patch('importlib.util.find_spec', return_value=None),
    ):
        bench.main(['--images', str(tmp_path)])

    captured = capsys.readouterr()
    assert run_calls == [
        ('v3', 'cpu'),
        ('v3', 'mps'),
        ('v3-cls', 'cpu'),
        ('v3-cls', 'mps'),
    ]
    assert captured.out.count('coremltools not installed') == 1
    assert 'ModuleNotFoundError' not in captured.out
    assert '  coreml ' not in captured.out


def test_aggregate_events_collects_compute_units_from_events():
    """aggregate_events() must propagate compute_units_* fields from the event stream
    into the summary dict.
    """
    bench = _load_benchmark()

    events = [
        {
            'name': 'fetch',
            'duration_ns': 1_000_000,
            'compute_units_requested': 'cpu_and_ne',
        },
        {
            'name': 'perform_inference',
            'duration_ns': 5_000_000,
            'tile_count': 4,
            'compute_units_actual': 'cpu_and_gpu',
        },
    ]

    summary = bench.aggregate_events(events, total_wall_ns=10_000_000)

    assert summary['compute_units_requested'] == 'cpu_and_ne'
    assert summary['compute_units_actual'] == 'cpu_and_gpu'


def test_benchmark_family_renders_null_for_none_compute_units_actual(tmp_path, capsys):
    """benchmark_family() must print `actual=null` when compute_units_actual is None."""
    bench = _load_benchmark()
    image = tmp_path / 'image.jpg'
    image.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b'\x00' * 64)

    def _run_variant_with_none_actual(config, device, images):
        return (
            10.0,
            {str(image): []},
            {
                'phase_totals_ns': {},
                'phase_counts': {},
                'unattributed_ns': 0,
                'model_invocations': 2,
                'tile_count': 8,
                'compute_units_requested': 'cpu_and_ne',
                'compute_units_actual': None,
            },
        )

    with patch.object(bench, '_run_variant', side_effect=_run_variant_with_none_actual):
        bench.benchmark_family('onecls', 'v3', 'v3-coreml', [image, image])

    captured = capsys.readouterr()
    assert 'actual=null' in captured.out, (
        f'Expected "actual=null" in benchmark output; got:\n{captured.out}'
    )


def test_benchmark_family_suppresses_known_third_party_noise(tmp_path, capsys):
    bench = _load_benchmark()
    image = tmp_path / 'image.jpg'
    image.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b'\x00' * 64)

    def _noisy_run_variant(config, device, images):
        warnings.warn(
            "'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'",
            UserWarning,
        )
        warnings.warn(
            '[coremltools] Torch version X.Y.Z has not been tested with coremltools',
            UserWarning,
        )
        warnings.warn(
            'SyntaxWarning: invalid escape sequence "\\_"',
            SyntaxWarning,
        )
        print('Loading example.mlpackage for CoreML inference...')
        return (
            12.3,
            {str(image): []},
            {
                'phase_totals_ns': {},
                'phase_counts': {},
                'unattributed_ns': 0,
                'model_invocations': 0,
                'tile_count': 0,
                'compute_units_requested': None,
                'compute_units_actual': None,
            },
        )

    with patch.object(bench, '_run_variant', side_effect=_noisy_run_variant):
        bench.benchmark_family('cls', 'v3-cls', 'v3-cls-coreml', [image])

    captured = capsys.readouterr()
    assert 'Loading example.mlpackage for CoreML inference...' not in captured.out
    assert "'has_mps' is deprecated" not in captured.out
    assert 'has not been tested with coremltools' not in captured.out
    assert 'invalid escape sequence' not in captured.out
    assert '=== Family: cls  (v3-cls vs v3-cls-coreml) ===' in captured.out
    assert 'Images: 1 (first is warm-up, excluded from timing)' in captured.out
