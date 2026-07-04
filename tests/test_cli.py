# -*- coding: utf-8 -*-
"""Tests for ScoutBot CLI."""

import json
from unittest.mock import patch

from click.testing import CliRunner


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self):
        """Test that CLI shows help."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'ScoutBot CLI' in result.output

    def test_cli_pipeline_help(self):
        """Test pipeline command help."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['pipeline', '--help'])

        assert result.exit_code == 0
        assert '--config' in result.output
        assert '--device' in result.output
        assert '--loc-thresh' in result.output

    def test_cli_batch_help(self):
        """Test batch command help."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['batch', '--help'])

        assert result.exit_code == 0
        assert '--config' in result.output

    def test_cli_batch_resumable_help(self):
        """Test batch-resumable command help."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['batch-resumable', '--help'])

        assert result.exit_code == 0
        assert '--output' in result.output
        assert '--config' in result.output
        assert '--no-progress' in result.output

    def test_cli_fetch_help(self):
        """Test fetch command help."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['fetch', '--help'])

        assert result.exit_code == 0
        assert '--config' in result.output

    def test_cli_get_classes_help(self):
        """Test get_classes command help."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['get_classes', '--help'])

        assert result.exit_code == 0
        assert '--config' in result.output
        assert '--output' in result.output

    def test_cli_pipeline_file_not_found(self):
        """Test pipeline with non-existent file."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['pipeline', '/nonexistent/file.jpg'])

        # Should fail gracefully
        assert result.exit_code != 0

    def test_cli_config_choices(self):
        """Invalid config names are rejected; valid set includes v3-coreml variants."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        runner = CliRunner()

        # Invalid config should fail
        result = runner.invoke(cli, ['pipeline', '--config', 'mvp', 'test.jpg'])
        assert result.exit_code != 0

        result = runner.invoke(cli, ['pipeline', '--config', 'phase1', 'test.jpg'])
        assert result.exit_code != 0

    def test_pipeline_help_includes_coreml_configs(self):
        """pipeline --help must list v3-coreml and v3-cls-coreml as valid --config choices."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['pipeline', '--help'])

        assert result.exit_code == 0
        assert 'v3-coreml' in result.output
        assert 'v3-cls-coreml' in result.output

    def test_export_coreml_help_lists_coreml_configs(self):
        """export-coreml --help must enumerate the valid coreml config choices."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['export-coreml', '--help'])

        assert result.exit_code == 0
        assert 'v3-coreml' in result.output
        assert 'v3-cls-coreml' in result.output

    def test_export_coreml_rejects_non_coreml_config_with_click_error(self):
        """Passing a non-CoreML config must produce a Click usage error, not an AssertionError."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['export-coreml', '--config', 'v3'])

        assert result.exit_code != 0
        assert 'AssertionError' not in result.output, (
            'the error must be caught by Click before reaching loc.export_coreml'
        )
        assert 'invalid' in result.output.lower()


class TestCLIExecutionBodies:
    """Tests for CLI command execution — all scoutbot functions mocked."""

    def test_pipeline_runs_and_prints_json(self, tmp_path):
        """Covers lines 109-136 and the validator return path (line 23)."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img = tmp_path / 'test.jpg'
        img.write_bytes(b'fake')

        with patch(
            'scoutbot.pipeline',
            return_value=(
                0.85,
                [{'l': 'animal', 'c': 0.85, 'x': 0, 'y': 0, 'w': 5, 'h': 5}],
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ['pipeline', '--device', 'cpu', str(img)])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert str(img) in output
        assert 'wic' in output[str(img)]
        assert output[str(img)]['wic'] == 0.85

    def test_pipeline_writes_output_file(self, tmp_path):
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img = tmp_path / 'test.jpg'
        img.write_bytes(b'fake')
        out = tmp_path / 'result.json'

        with patch('scoutbot.pipeline', return_value=(0.0, [])):
            runner = CliRunner()
            result = runner.invoke(
                cli, ['pipeline', '--device', 'cpu', '--output', str(out), str(img)]
            )

        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert str(img) in data

    def test_batch_runs_and_prints_json(self, tmp_path):
        """Covers lines 179-210."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img1 = tmp_path / 'a.jpg'
        img2 = tmp_path / 'b.jpg'
        img1.write_bytes(b'fake')
        img2.write_bytes(b'fake')

        with patch(
            'scoutbot.batch',
            return_value=(
                [0.85, 0.0],
                [[{'l': 'animal', 'c': 0.85, 'x': 0, 'y': 0, 'w': 5, 'h': 5}], []],
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ['batch', '--device', 'cpu', '--no-progress', str(img1), str(img2)],
            )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert str(img1) in data
        assert str(img2) in data

    def test_batch_resumable_runs_and_prints_summary(self, tmp_path):
        """Covers lines 253-278."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img1 = str(tmp_path / 'a.jpg')
        img2 = str(tmp_path / 'b.jpg')
        out = tmp_path / 'output.jsonl'

        with patch(
            'scoutbot.batch_resumable',
            return_value={'processed': 2, 'skipped': 0, 'errors': 0, 'total': 2},
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ['batch-resumable', '--output', str(out), '--no-progress', img1, img2],
            )

        assert result.exit_code == 0
        assert 'Processed: 2' in result.output
        assert 'Total: 2' in result.output

    def test_fetch_command_calls_scoutbot_fetch(self):
        """Covers line 287."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch('scoutbot.fetch') as mock_fetch:
            runner = CliRunner()
            result = runner.invoke(cli, ['fetch', '--config', 'v3'])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(config='v3')

    def test_get_classes_command_prints_json(self):
        """Covers lines 337-343."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch('scoutbot.get_classes', return_value=['animal', 'vehicle']):
            runner = CliRunner()
            result = runner.invoke(cli, ['get_classes', '--config', 'v3'])

        assert result.exit_code == 0
        classes = json.loads(result.output)
        assert classes == ['animal', 'vehicle']

    def test_get_classes_command_writes_output_file(self, tmp_path):
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        out = tmp_path / 'classes.json'

        with patch('scoutbot.get_classes', return_value=['animal']):
            runner = CliRunner()
            result = runner.invoke(
                cli, ['get_classes', '--config', 'v3', '--output', str(out)]
            )

        assert result.exit_code == 0
        assert out.exists()
        assert json.loads(out.read_text()) == ['animal']

    def test_export_coreml_command_with_explicit_config(self):
        """Covers the `if config:` branch of export-coreml (lines 308-310, 318-319)."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch(
            'scoutbot.loc.export_coreml', return_value='/fake/model.mlpackage'
        ) as mock_export:
            runner = CliRunner()
            result = runner.invoke(cli, ['export-coreml', '--config', 'v3-coreml'])

        assert result.exit_code == 0
        mock_export.assert_called_once_with('v3-coreml', force=False)

    def test_export_coreml_command_without_config_exports_all(self):
        """Covers the `else` branch (lines 311-317) — iterates all coreml configs."""
        from click.testing import CliRunner

        from scoutbot import loc
        from scoutbot.scoutbot import cli

        with patch(
            'scoutbot.loc.export_coreml', return_value='/fake/model.mlpackage'
        ) as mock_export:
            runner = CliRunner()
            result = runner.invoke(cli, ['export-coreml'])

        assert result.exit_code == 0
        # One call per coreml config (v3-coreml, v3-cls-coreml)
        expected_configs = [
            name
            for name, cfg in loc.CONFIGS.items()
            if name is not None
            and isinstance(cfg, dict)
            and cfg.get('backend') == 'coreml'
        ]
        assert mock_export.call_count == len(expected_configs)
        called_configs = [call.args[0] for call in mock_export.call_args_list]
        assert sorted(called_configs) == sorted(expected_configs)

    def test_example_command_calls_scoutbot_example(self):
        """Covers line 327."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch('scoutbot.example') as mock_example:
            runner = CliRunner()
            result = runner.invoke(cli, ['example'])

        assert result.exit_code == 0
        mock_example.assert_called_once()


# --- register command tests ---

import os
import pytest
from pathlib import Path


@pytest.fixture
def isolated_registry_cli(tmp_path, monkeypatch):
    monkeypatch.setenv('RW_REGISTRY_PATH', str(tmp_path / 'registry.json'))
    return tmp_path


def test_register_adds_model_to_registry(isolated_registry_cli, tmp_path):
    from scoutbot.scoutbot import cli
    from scoutbot import registry

    model_file = tmp_path / 'custom.pt'
    model_file.write_bytes(b'fake model')

    runner = CliRunner()
    result = runner.invoke(cli, [
        'register',
        '--name', 'my-model',
        '--path', str(model_file),
        '--arch', 'yolov8',
        '--description', 'Test model',
        '--threshold', '0.60',
    ])

    assert result.exit_code == 0, result.output
    entry = registry.lookup('my-model')
    assert entry is not None
    assert entry.framework == 'yolov8'
    assert entry.description == 'Test model'
    assert entry.default_run_config['confidence_threshold'] == 0.60
    assert entry.source.type == 'local'


def test_register_fails_if_path_missing(isolated_registry_cli):
    from scoutbot.scoutbot import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        'register',
        '--name', 'x', '--path', '/does/not/exist.pt', '--arch', 'yolov8',
    ])
    assert result.exit_code != 0


# --- models command tests ---


def test_models_list_empty(isolated_registry_cli):
    from scoutbot.scoutbot import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['models'])
    assert result.exit_code == 0
    assert 'No models' in result.output


def test_models_list_shows_registered(isolated_registry_cli, tmp_path):
    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    from scoutbot.scoutbot import cli

    model_file = tmp_path / 'm.pt'
    model_file.write_bytes(b'x')
    registry.register(ModelEntry(
        name='test-yolo', path=str(model_file), md5='abc123',
        framework='yolov8', registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'), default_run_config={},
    ))

    runner = CliRunner()
    result = runner.invoke(cli, ['models'])
    assert result.exit_code == 0
    assert 'test-yolo' in result.output
    assert 'yolov8' in result.output


def test_models_info_prints_json(isolated_registry_cli, tmp_path):
    import json

    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    from scoutbot.scoutbot import cli

    model_file = tmp_path / 'm.pt'
    model_file.write_bytes(b'x')
    registry.register(ModelEntry(
        name='info-model', path=str(model_file), md5='deadbeef',
        framework='herdnet', registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual', note='test'),
        default_run_config={'patch_size': 256},
    ))

    runner = CliRunner()
    result = runner.invoke(cli, ['models', 'info', 'info-model'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['name'] == 'info-model'
    assert data['framework'] == 'herdnet'
    assert data['default_run_config'] == {'patch_size': 256}


def test_models_info_unknown_exits_nonzero(isolated_registry_cli):
    from scoutbot.scoutbot import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['models', 'info', 'nonexistent'])
    assert result.exit_code != 0


# --- --model flag tests ---


def test_pipeline_accepts_model_flag(isolated_registry_cli, tmp_path):
    """--model flag routes pipeline through the registry entry."""
    from unittest.mock import patch

    import scoutbot
    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    from scoutbot.scoutbot import cli

    model_file = tmp_path / 'custom.pt'
    model_file.write_bytes(b'fake')
    img_file = tmp_path / 'img.jpg'
    img_file.write_bytes(b'fake image')

    registry.register(ModelEntry(
        name='my-custom',
        path=str(model_file),
        md5='abc',
        framework='yolov8',
        registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'),
        default_run_config={
            'slice_height': 512, 'slice_width': 512,
            'overlap_height_ratio': 0.25, 'overlap_width_ratio': 0.25,
            'perform_standard_pred': False, 'postprocess_class_agnostic': True,
            'confidence_threshold': 0.45,
        },
    ))

    runner = CliRunner()
    with patch.object(scoutbot, 'pipeline', return_value=(0.0, [])):
        result = runner.invoke(cli, ['pipeline', str(img_file), '--model', 'my-custom'])
    assert result.exit_code == 0, result.output


def test_pipeline_model_and_config_are_exclusive(isolated_registry_cli, tmp_path):
    from scoutbot.scoutbot import cli

    img_file = tmp_path / 'img.jpg'
    img_file.write_bytes(b'x')
    runner = CliRunner()
    result = runner.invoke(cli, ['pipeline', str(img_file), '--model', 'x', '--config', 'v3'])
    assert result.exit_code != 0
    assert 'mutually exclusive' in result.output.lower()


def test_batch_accepts_model_flag(isolated_registry_cli, tmp_path):
    """--model flag routes batch command through the registry entry."""
    from unittest.mock import patch

    import scoutbot
    from scoutbot import registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource
    from scoutbot.scoutbot import cli

    model_file = tmp_path / 'custom.pt'
    model_file.write_bytes(b'fake')
    img_file = tmp_path / 'img.jpg'
    img_file.write_bytes(b'fake image')

    registry.register(ModelEntry(
        name='my-custom',
        path=str(model_file),
        md5='abc',
        framework='yolov8',
        registered='2026-05-09T00:00:00Z',
        source=RegistrySource(type='manual'),
        default_run_config={
            'slice_height': 512, 'slice_width': 512,
            'overlap_height_ratio': 0.25, 'overlap_width_ratio': 0.25,
            'perform_standard_pred': False, 'postprocess_class_agnostic': True,
            'confidence_threshold': 0.45,
        },
    ))

    runner = CliRunner()
    with (
        patch.object(scoutbot, 'pipeline', return_value=(0.0, [])),
        patch('scoutbot.tile_batched.Yolov8DetectionModel'),
    ):
        result = runner.invoke(cli, ['batch', str(img_file), '--model', 'my-custom'])
    assert result.exit_code == 0, result.output
