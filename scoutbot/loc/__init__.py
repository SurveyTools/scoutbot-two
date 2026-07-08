# -*- coding: utf-8 -*-
"""
Compatibility wrapper for ScoutBot model-store configuration and fetching.
"""

from scoutbot.model_store import (
    CACHE_DIR,
    CONFIGS,
    DEFAULT_CONFIG,
    _md5,
    export_coreml,
    fetch,
)

__all__ = [
    'CACHE_DIR',
    'CONFIGS',
    'DEFAULT_CONFIG',
    '_md5',
    'export_coreml',
    'fetch',
]
