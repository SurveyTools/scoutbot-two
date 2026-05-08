# -*- coding: utf-8 -*-
"""Shared pytest fixtures for ScoutBot tests."""
import hashlib
from pathlib import Path

import pytest
import requests


@pytest.fixture
def sample_image():
    """Download and cache sample image for testing."""
    TEST_IMAGE_URL = 'https://wildbookiarepository.azureedge.net/data/scout.example.jpg'
    TEST_IMAGE_HASH = '786a940b062a90961f409539292f09144c3dbdbc6b6faa64c3e764d63d55c988'

    cache_dir = Path.home() / '.cache' / 'scoutbot'
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_path = cache_dir / 'scout.example.jpg'

    def _sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    if not img_path.exists() or _sha256(img_path) != TEST_IMAGE_HASH:
        with requests.get(TEST_IMAGE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(img_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    return str(img_path)


@pytest.fixture
def temp_output(tmp_path):
    """Provide a temporary output file path."""
    return tmp_path / 'output.jsonl'


@pytest.fixture
def temp_json_output(tmp_path):
    """Provide a temporary JSON output file path."""
    return tmp_path / 'output.json'
