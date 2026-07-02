# -*- coding: utf-8 -*-
"""Shared pytest fixtures for ScoutBot tests."""
import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope='session')
def shutdown_logging():
    yield
    logging.shutdown()


@pytest.fixture
def sample_image():
    """Return path to a bundled test image (from examples/)."""
    repo_root = Path(__file__).resolve().parent.parent
    img_path = repo_root / 'examples' / '0d4e4df2-7b69-91b1-1985-c8421f2f3253.jpg'
    assert img_path.exists(), f'Test image not found: {img_path}'
    return str(img_path)


@pytest.fixture
def temp_output(tmp_path):
    """Provide a temporary output file path."""
    return tmp_path / 'output.jsonl'


@pytest.fixture
def temp_json_output(tmp_path):
    """Provide a temporary JSON output file path."""
    return tmp_path / 'output.json'
