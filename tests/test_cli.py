# -*- coding: utf-8 -*-
"""Tests for ScoutBot CLI."""
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
