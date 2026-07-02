# -*- coding: utf-8 -*-
"""Tests for scoutbot.loc module."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestLocMd5:
    """Unit tests for scoutbot.loc._md5()."""

    def test_md5_returns_correct_hex_digest(self, tmp_path):
        from scoutbot.loc import _md5

        content = b'hello world'
        f = tmp_path / 'test.bin'
        f.write_bytes(content)

        result = _md5(f)

        assert result == hashlib.md5(content).hexdigest()
        assert isinstance(result, str)
        assert len(result) == 32

    def test_md5_handles_large_file_in_chunks(self, tmp_path):
        from scoutbot.loc import _md5

        # 20KB — forces multiple 8192-byte chunks
        content = b'x' * 20_000
        f = tmp_path / 'large.bin'
        f.write_bytes(content)

        result = _md5(f)
        assert result == hashlib.md5(content).hexdigest()


class TestLocFetchUnit:
    """Unit tests for scoutbot.loc.fetch() — no network, no real model."""

    def test_fetch_coreml_returns_path_when_bundle_exists(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        bundle = tmp_path / cfg['name']
        bundle.mkdir()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('scoutbot.loc.export_coreml') as mock_export,
        ):
            result = loc.fetch('v3-coreml')

        mock_export.assert_not_called()
        assert result == str(bundle)

    def test_fetch_coreml_triggers_export_when_bundle_missing(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        bundle = tmp_path / cfg['name']

        def _create_bundle(config, **kwargs):
            bundle.mkdir(exist_ok=True)

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('scoutbot.loc.export_coreml', side_effect=_create_bundle) as mock_export,
        ):
            result = loc.fetch('v3-coreml')

        mock_export.assert_called_once_with('v3-coreml')
        assert result == str(bundle)

    def test_fetch_pt_returns_cached_path_when_md5_matches(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3']
        local = tmp_path / cfg['name']
        local.write_bytes(b'fake model bytes')
        remote_md5 = hashlib.md5(b'fake model bytes').hexdigest()

        md5_response = MagicMock()
        md5_response.text = remote_md5 + '  ' + cfg['name']
        md5_response.raise_for_status = MagicMock()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('requests.get', return_value=md5_response),
            patch('scoutbot.loc._md5', return_value=remote_md5),
        ):
            result = loc.fetch('v3')

        assert result == str(local)

    def test_fetch_pt_downloads_when_local_missing(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3']
        local = tmp_path / cfg['name']
        remote_md5 = 'abc123deadbeef00' * 2  # 32-char fake md5

        fake_content = b'model weights'
        remote_md5 = hashlib.md5(fake_content).hexdigest()

        md5_response = MagicMock()
        md5_response.text = remote_md5 + '  ' + cfg['name']
        md5_response.raise_for_status = MagicMock()

        download_response = MagicMock()
        download_response.raise_for_status = MagicMock()
        download_response.headers = {'content-length': str(len(fake_content))}
        download_response.iter_content.return_value = [fake_content]
        download_response.__enter__ = MagicMock(return_value=download_response)
        download_response.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def _get(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return md5_response
            return download_response

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('requests.get', side_effect=_get),
        ):
            result = loc.fetch('v3')

        assert result == str(local)
        assert local.exists()
        assert local.read_bytes() == fake_content

    def test_fetch_none_config_uses_default(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS[loc.DEFAULT_CONFIG]
        local = tmp_path / cfg['name']
        local.write_bytes(b'bytes')
        remote_md5 = hashlib.md5(b'bytes').hexdigest()

        md5_response = MagicMock()
        md5_response.text = remote_md5 + '  ' + cfg['name']
        md5_response.raise_for_status = MagicMock()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('requests.get', return_value=md5_response),
            patch('scoutbot.loc._md5', return_value=remote_md5),
        ):
            result = loc.fetch(None)

        assert result is not None
        assert isinstance(result, str)


class TestExportCoremlBranches:
    """Unit tests for export_coreml() branches not covered by TestExportCoreml."""

    def test_early_return_when_target_exists_and_no_force(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        target = tmp_path / cfg['name']
        target.mkdir()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('ultralytics.YOLO') as MockYOLO,
        ):
            result = loc.export_coreml('v3-coreml', force=False)

        MockYOLO.assert_not_called()
        assert result == str(target)

    def test_rmtree_called_when_target_exists_and_exported_differs(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        target = tmp_path / cfg['name']
        target.mkdir()  # target already exists

        # exported lands somewhere else (different from target)
        other_dir = tmp_path / 'tmp_export'
        other_dir.mkdir()
        exported_path = other_dir / cfg['name']
        exported_path.mkdir()

        rmtree_calls = []

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('ultralytics.YOLO') as MockYOLO,
            patch('shutil.rmtree', side_effect=lambda p, **kw: rmtree_calls.append(str(p))),
            patch('shutil.move'),
        ):
            MockYOLO.return_value.export.return_value = str(exported_path)
            loc.export_coreml('v3-coreml', force=True)

        assert any(str(target) in c for c in rmtree_calls), (
            'shutil.rmtree must remove the existing target bundle '
            'before shutil.move places the newly-exported one there'
        )
