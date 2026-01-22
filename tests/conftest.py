# -*- coding: utf-8 -*-
"""Shared pytest fixtures for ScoutBot tests."""
import pytest
from pathlib import Path


@pytest.fixture
def sample_image():
    """Download and cache sample image for testing."""
    import pooch

    return pooch.retrieve(
        url='https://wildbookiarepository.azureedge.net/data/scout.example.jpg',
        known_hash='786a940b062a90961f409539292f09144c3dbdbc6b6faa64c3e764d63d55c988',
    )


@pytest.fixture
def temp_output(tmp_path):
    """Provide a temporary output file path."""
    return tmp_path / 'output.jsonl'


@pytest.fixture
def temp_json_output(tmp_path):
    """Provide a temporary JSON output file path."""
    return tmp_path / 'output.json'
