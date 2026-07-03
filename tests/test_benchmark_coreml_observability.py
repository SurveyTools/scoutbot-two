"""Tests for benchmark CLI observability behavior."""

from unittest.mock import patch


def test_missing_coremltools_is_reported_once(tmp_path, capsys):
    import benchmark_coreml

    image = tmp_path / 'image.jpg'
    image.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b'\x00' * 64)

    def _run_variant(config, device, images):
        if config.endswith('coreml'):
            raise RuntimeError('coremltools missing')
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

    with patch.object(benchmark_coreml, '_run_variant', side_effect=_run_variant):
        benchmark_coreml.main(['--family', 'onecls', '--images', str(tmp_path)])

    captured = capsys.readouterr()
    assert captured.out.count('SKIPPED') == 1
    assert 'Traceback' not in captured.out
