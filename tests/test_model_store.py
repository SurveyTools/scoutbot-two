from importlib import import_module

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
