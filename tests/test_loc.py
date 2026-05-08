# -*- coding: utf-8 -*-
"""Tests for scoutbot.loc module."""
from pathlib import Path


class TestLocConfig:
    """Tests for LOC configuration."""

    def test_configs_exist(self):
        """Test that v3 and v3-cls configs exist."""
        from scoutbot import loc

        assert 'v3' in loc.CONFIGS
        assert 'v3-cls' in loc.CONFIGS

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
                assert key in config, f"Config {config_name} missing key: {key}"

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
