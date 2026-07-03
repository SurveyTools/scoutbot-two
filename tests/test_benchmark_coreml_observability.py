"""Tests for benchmark CLI observability behavior."""

from unittest.mock import patch


def test_benchmark_family_renders_null_for_none_compute_units_requested(tmp_path, capsys):
    """When both fields are None (PT variant), no compute-unit suffix must appear in
    the output.

    Per the plan spec: when req is None the suffix is empty — nothing is printed.
    Fails today because the rendering code uses `req or "unknown"` which maps
    None → "unknown" and still emits a `requested=` suffix. Task 3 must change this
    so that the suffix is omitted entirely when req is None.
    """
    import benchmark_coreml

    image = tmp_path / 'image.jpg'
    image.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b'\x00' * 64)

    def _run_variant(config, device, images):
        return (
            8.0,
            {str(image): []},
            {
                'phase_totals_ns': {},
                'phase_counts': {},
                'unattributed_ns': 0,
                'model_invocations': 1,
                'tile_count': 4,
                # Both fields None — simulates a PT run before Task 2 emits them.
                'compute_units_requested': None,
                'compute_units_actual': None,
            },
        )

    with patch.object(benchmark_coreml, '_run_variant', side_effect=_run_variant):
        benchmark_coreml.benchmark_family('onecls', 'v3', 'v3-coreml', [image, image])

    captured = capsys.readouterr()
    # Fails today: the code prints `requested=unknown` (req is None → "unknown"),
    # so `requested=` appears in the output. Task 3 must eliminate the suffix entirely
    # when req is None, making this assertion pass.
    assert 'requested=' not in captured.out, (
        f'Expected no "requested=" suffix in benchmark output for PT variant; got:\n{captured.out}'
    )


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
