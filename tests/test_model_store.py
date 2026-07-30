from importlib import import_module
from pathlib import Path

from scoutbot import loc


def test_model_store_owns_catalog():
    model_store = import_module('scoutbot.model_store')

    assert model_store.DEFAULT_CONFIG == 'v3'
    assert 'v3' in model_store.CONFIGS
    assert 'v3-cls' in model_store.CONFIGS
    assert 'v3-coreml' in model_store.CONFIGS
    assert 'v3-cls-coreml' in model_store.CONFIGS
    assert model_store.CONFIGS['v3-coreml']['backend'] == 'coreml'
    assert model_store.CONFIGS['v3-coreml']['source'] == 'v3'
    assert model_store.CONFIGS[None] is model_store.CONFIGS[model_store.DEFAULT_CONFIG]


def test_loc_reexports_model_store_api():
    model_store = import_module('scoutbot.model_store')

    assert loc.DEFAULT_CONFIG == model_store.DEFAULT_CONFIG
    assert loc.CONFIGS is model_store.CONFIGS
    assert loc.fetch is model_store.fetch
    assert loc.export_coreml is model_store.export_coreml
    assert loc._md5 is model_store._md5


# --- registry-integrated fetch tests ---

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv('OSW_REGISTRY_PATH', str(tmp_path / 'registry.json'))
    monkeypatch.setenv('OSW_MODELS_DIR', str(tmp_path / 'models'))
    return tmp_path


def test_fetch_uses_registry_when_model_cached(isolated_registry):
    """fetch() does not hit the network if the model is in the registry with matching MD5."""
    from scoutbot import model_store, registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource

    # Create a fake model file
    model_file = isolated_registry / 'models' / 'yolov8-onecls.kaza.pt'
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b'fake model content')
    md5 = model_store._md5(model_file)

    registry.register(
        ModelEntry(
            name='scoutbot-v3',
            path=str(model_file),
            md5=md5,
            framework='yolov8',
            registered='2026-05-09T00:00:00Z',
            source=RegistrySource(type='fetched', url='https://example.com/m.pt'),
            default_run_config={},
        )
    )

    with (
        patch('requests.get') as mock_get,
        patch.dict(model_store.CONFIGS['v3'], {'md5': md5}),
    ):
        result = model_store.fetch(config='v3')
        mock_get.assert_not_called()

    assert result == str(model_file)


def test_fetch_fails_clearly_when_not_registered_and_no_network(isolated_registry):
    """fetch() raises a clear error if model not in registry and network unavailable."""
    import requests

    from scoutbot import model_store

    with patch('requests.get', side_effect=requests.ConnectionError('no network')):
        with pytest.raises(Exception):
            model_store.fetch(config='v3')


def test_fetch_registers_model_after_download(isolated_registry):
    """After a successful download, fetch() writes a ModelEntry to the registry."""
    from scoutbot import model_store, registry

    fake_content = b'fake model data'

    import hashlib

    h = hashlib.md5(usedforsecurity=False)
    h.update(fake_content)
    expected_md5 = h.hexdigest()

    mock_md5_response = MagicMock()
    mock_md5_response.text = expected_md5
    mock_md5_response.raise_for_status = MagicMock()

    mock_model_response = MagicMock()
    mock_model_response.raise_for_status = MagicMock()
    mock_model_response.headers = {'content-length': str(len(fake_content))}
    mock_model_response.iter_content = MagicMock(return_value=[fake_content])
    mock_model_response.__enter__ = MagicMock(return_value=mock_model_response)
    mock_model_response.__exit__ = MagicMock(return_value=False)

    def mock_get(url, **kwargs):
        if 'md5' in url:
            return mock_md5_response
        return mock_model_response

    with (
        patch('requests.get', side_effect=mock_get),
        patch('scoutbot.model_store.CACHE_DIR', isolated_registry),
        patch.dict(model_store.CONFIGS['v3'], {'md5': expected_md5}),
    ):
        model_store.fetch(config='v3')

    entry = registry.lookup('scoutbot-v3')
    assert entry is not None
    assert entry.framework == 'yolov8'
    assert entry.source.type == 'fetched'
    assert 'confidence_threshold' in entry.default_run_config


def test_export_coreml_uses_valid_fp16_quantize(tmp_path, monkeypatch):
    """Guard the Core ML export precision argument.

    ultralytics >=8.3 rejects the boolean ``quantize=True`` (valid values are
    ``'fp16'``/``'int8'``/``8``/``16``/...). That broken value shipped undetected
    because the CLI tests mock ``loc.export_coreml`` and there is no macOS CI to
    run a real export. This exercises the real ``export_coreml`` with ``YOLO`` and
    the network fetch mocked, and asserts the export uses ``quantize='fp16'``.
    """
    from scoutbot import loc, model_store

    monkeypatch.setattr(loc, 'CACHE_DIR', tmp_path)
    target = tmp_path / str(loc.CONFIGS['v3-coreml']['name'])

    def fake_export(**kwargs):
        target.mkdir(parents=True, exist_ok=True)  # a .mlpackage is a directory bundle
        return str(target)

    yolo = MagicMock()
    yolo.return_value.export.side_effect = fake_export

    with (
        patch('ultralytics.YOLO', yolo),
        patch.object(loc, 'fetch', return_value=str(tmp_path / 'src.pt')),
    ):
        result = model_store.export_coreml('v3-coreml', force=True)

    assert result == str(target)
    assert yolo.return_value.export.call_args.kwargs['quantize'] == 'fp16'


# --- pinned-MD5 cache invalidation ---


def test_pinned_md5_rejects_stale_cache(isolated_registry):
    """A registered cache whose hash != CONFIGS['md5'] is re-downloaded, not served."""
    from scoutbot import model_store, registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource

    # Stale cached file registered under the OLD hash.
    stale = isolated_registry / 'models' / 'yolov8-onecls.kaza.pt'
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b'old dill-embedded bytes')
    old_md5 = model_store._md5(stale)
    registry.register(
        ModelEntry(
            name='scoutbot-v3',
            path=str(stale),
            md5=old_md5,
            framework='yolov8',
            registered='2026-05-09T00:00:00Z',
            source=RegistrySource(type='fetched', url='https://example.com/old.pt'),
            default_run_config={},
        )
    )

    new_content = b'new dill-free bytes'
    import hashlib

    new_md5 = hashlib.md5(new_content, usedforsecurity=False).hexdigest()

    md5_resp = MagicMock()
    md5_resp.text = new_md5
    md5_resp.raise_for_status = MagicMock()
    model_resp = MagicMock()
    model_resp.raise_for_status = MagicMock()
    model_resp.headers = {'content-length': str(len(new_content))}
    model_resp.iter_content = MagicMock(return_value=[new_content])
    model_resp.__enter__ = MagicMock(return_value=model_resp)
    model_resp.__exit__ = MagicMock(return_value=False)

    def fake_get(url, **kwargs):
        return md5_resp if url.endswith('.md5') else model_resp

    with (
        patch('requests.get', side_effect=fake_get),
        patch.object(model_store, 'CACHE_DIR', isolated_registry / 'models'),
        patch.dict(model_store.CONFIGS['v3'], {'md5': new_md5}),
    ):
        result = model_store.fetch(config='v3')

    assert (
        model_store._md5(Path(result)) == new_md5
    )  # got the NEW bytes, not the stale cache


def test_pinned_md5_serves_matching_cache(isolated_registry):
    """When the registered hash equals CONFIGS['md5'], no network call is made."""
    from scoutbot import model_store, registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource

    cached = isolated_registry / 'models' / 'yolov8-onecls.kaza.20260729.pt'
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'dill-free bytes')
    good_md5 = model_store._md5(cached)
    registry.register(
        ModelEntry(
            name='scoutbot-v3',
            path=str(cached),
            md5=good_md5,
            framework='yolov8',
            registered='2026-07-29T00:00:00Z',
            source=RegistrySource(type='fetched', url='https://example.com/new.pt'),
            default_run_config={},
        )
    )

    with (
        patch('requests.get') as mock_get,
        patch.dict(model_store.CONFIGS['v3'], {'md5': good_md5}),
    ):
        result = model_store.fetch(config='v3')
        mock_get.assert_not_called()

    assert result == str(cached)


def test_pinned_md5_raises_on_sidecar_mismatch(isolated_registry):
    """If the hosted sidecar hash != the pinned md5, fetch raises rather than trust it."""
    from scoutbot import model_store

    served = b'served bytes'
    import hashlib

    served_md5 = hashlib.md5(served, usedforsecurity=False).hexdigest()

    md5_resp = MagicMock()
    md5_resp.text = served_md5
    md5_resp.raise_for_status = MagicMock()
    model_resp = MagicMock()
    model_resp.raise_for_status = MagicMock()
    model_resp.headers = {'content-length': str(len(served))}
    model_resp.iter_content = MagicMock(return_value=[served])
    model_resp.__enter__ = MagicMock(return_value=model_resp)
    model_resp.__exit__ = MagicMock(return_value=False)

    def fake_get(url, **kwargs):
        return md5_resp if url.endswith('.md5') else model_resp

    with (
        patch('requests.get', side_effect=fake_get),
        patch.object(model_store, 'CACHE_DIR', isolated_registry / 'models'),
        patch.dict(model_store.CONFIGS['v3'], {'md5': 'deadbeef' * 4}),  # != served_md5
    ):
        with pytest.raises(AssertionError, match='[Pp]inned'):
            model_store.fetch(config='v3')


def test_pt_configs_are_gcs_first_and_pinned():
    from scoutbot import model_store

    # Exact hashes recorded in Task 2 Step 5 (verified against the GCS upload).
    expected = {
        'v3': ('yolov8-onecls.kaza.20260729.pt', 'c4954cd3957b286913c886c4219cd6c5'),
        'v3-cls': ('yolov8-cls.kaza.20260729.pt', '6a1565be5460d88a0e2eea2a74e66863'),
    }
    for key, (fname, expected_md5) in expected.items():
        cfg = model_store.CONFIGS[key]
        assert cfg['name'] == fname
        providers = [s['provider'] for s in cfg['sources']]
        assert providers == ['gcs', 'azure']  # GCS primary, Azure fallback
        assert cfg['md5'] == expected_md5  # exact pinned MD5 (not just shape)
        for s in cfg['sources']:
            assert s['url'].endswith(fname)
            assert s['md5_url'].endswith(fname + '.md5')


def test_pt_configs_use_osw_bucket():
    from scoutbot import model_store

    for key in ('v3', 'v3-cls'):
        gcs = model_store.CONFIGS[key]['sources'][0]
        assert gcs['provider'] == 'gcs'
        assert gcs['url'].startswith(
            'https://storage.googleapis.com/osw-public-models/scoutbot/'
        )
        assert gcs['md5_url'].startswith(
            'https://storage.googleapis.com/osw-public-models/scoutbot/'
        )
        assert 'rebel-pipeline-public-models' not in gcs['url']
