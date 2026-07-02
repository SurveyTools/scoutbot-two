# -*- coding: utf-8 -*-
"""Tests for scoutbot.loc module."""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestLocConfig:
    """Tests for LOC configuration."""

    def test_configs_exist(self):
        """Test that v3, v3-cls, v3-coreml, and v3-cls-coreml configs exist."""
        from scoutbot import loc

        assert 'v3' in loc.CONFIGS
        assert 'v3-cls' in loc.CONFIGS
        assert 'v3-coreml' in loc.CONFIGS
        assert 'v3-cls-coreml' in loc.CONFIGS

    def test_coreml_config_schema(self):
        """CoreML configs must have backend/source/name keys; not url/md5_url."""
        from scoutbot import loc

        for name in ['v3-coreml', 'v3-cls-coreml']:
            cfg = loc.CONFIGS[name]

            # CoreML-specific required keys
            assert cfg.get('backend') == 'coreml', f'{name} must have backend=coreml'
            assert 'source' in cfg, f'{name} must have a source key'
            assert cfg['source'] in loc.CONFIGS, (
                f"{name} source '{cfg['source']}' must itself be a known config"
            )
            assert 'name' in cfg, f'{name} must have a name key'
            assert cfg['name'].endswith('.mlpackage'), (
                f'{name} bundle name must end in .mlpackage, got: {cfg["name"]}'
            )

            # Shared inference keys must be present
            for key in (
                'thresh',
                'slice_height',
                'slice_width',
                'overlap_height_ratio',
                'overlap_width_ratio',
                'perform_standard_pred',
                'postprocess_class_agnostic',
            ):
                assert key in cfg, f'{name} missing shared key: {key}'

            # CoreML configs must NOT have download keys (no remote bundle)
            assert 'url' not in cfg, f'{name} must not have a url key'
            assert 'md5_url' not in cfg, f'{name} must not have a md5_url key'

    def test_config_has_required_keys(self):
        """Test that configs have all required keys."""
        from scoutbot import loc

        required_keys = [
            'name',
            'url',
            'md5_url',
            'thresh',
            'slice_height',
            'slice_width',
            'overlap_height_ratio',
            'overlap_width_ratio',
            'perform_standard_pred',
            'postprocess_class_agnostic',
        ]

        for config_name in ['v3', 'v3-cls']:
            config = loc.CONFIGS[config_name]
            for key in required_keys:
                assert key in config, f'Config {config_name} missing key: {key}'

    def test_default_config_is_v3(self):
        """Test that default config points to v3."""
        from scoutbot import loc

        # None should map to default
        assert loc.CONFIGS[None] is not None

    def test_config_values_valid(self):
        """Test that config values are valid."""
        from scoutbot import loc

        for config_name in ['v3', 'v3-cls']:
            config = loc.CONFIGS[config_name]

            # Thresholds should be between 0 and 1
            assert 0.0 <= config['thresh'] <= 1.0

            # Slice dimensions should be positive
            assert config['slice_height'] > 0
            assert config['slice_width'] > 0

            # Overlap ratios should be between 0 and 1
            assert 0.0 <= config['overlap_height_ratio'] <= 1.0
            assert 0.0 <= config['overlap_width_ratio'] <= 1.0


@pytest.mark.slow
class TestLocFetch:
    """Tests for model fetching."""

    def test_fetch_v3_returns_path(self):
        """Test that fetch returns a valid path for v3."""
        from scoutbot import loc

        model_path = loc.fetch(config='v3')

        assert model_path is not None
        assert isinstance(model_path, str)
        # Path should exist after fetch
        assert Path(model_path).exists()

    def test_fetch_v3_cls_returns_path(self):
        """Test that fetch returns a valid path for v3-cls."""
        from scoutbot import loc

        model_path = loc.fetch(config='v3-cls')

        assert model_path is not None
        assert isinstance(model_path, str)
        assert Path(model_path).exists()

    def test_fetch_default_config(self):
        """Test fetch with default (None) config."""
        from scoutbot import loc

        model_path = loc.fetch(config=None)

        assert model_path is not None
        assert Path(model_path).exists()


class TestExportCoreml:
    """Unit tests for export_coreml (no network, no model load needed)."""

    def test_export_uses_shutil_move_not_rename(self, tmp_path):
        """export_coreml must call shutil.move (not Path.rename) to handle cross-device moves."""
        from scoutbot import loc

        # Simulate Ultralytics writing the bundle to an unexpected location
        other = tmp_path / 'ultralytics_out' / 'yolov8-onecls.kaza.mlpackage'
        other.parent.mkdir()
        other.mkdir()  # .mlpackage is a directory bundle

        move_calls = []

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('scoutbot.loc.fetch', return_value=str(tmp_path / 'fake.pt')),
            patch('ultralytics.YOLO') as MockYOLO,
            patch(
                'shutil.move',
                side_effect=lambda s, d: move_calls.append((str(s), str(d))),
            ),
        ):
            MockYOLO.return_value.export.return_value = str(other)
            loc.export_coreml('v3-coreml', force=True)

        assert move_calls, (
            'shutil.move must be called to relocate the bundle; '
            'the current code uses Path.rename which fails across filesystems (EXDEV)'
        )
        src, dst = move_calls[0]
        assert dst == str(tmp_path / 'yolov8-onecls.kaza.mlpackage')
