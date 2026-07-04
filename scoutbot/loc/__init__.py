# -*- coding: utf-8 -*-
"""
Compatibility wrapper for ScoutBot model-store configuration and fetching.
"""

from scoutbot.model_store import CACHE_DIR
from scoutbot.model_store import CONFIGS
from scoutbot.model_store import DEFAULT_CONFIG
from scoutbot.model_store import _md5
from scoutbot.model_store import export_coreml
from scoutbot.model_store import fetch

__all__ = [
    'CACHE_DIR',
    'CONFIGS',
    'DEFAULT_CONFIG',
    '_md5',
    'export_coreml',
    'fetch',
]
