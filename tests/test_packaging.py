"""Packaging metadata guards for the PyPI page and the built wheel.

These read the *installed* metadata (``uv sync`` rebuilds it from pyproject.toml), so they check
what PyPI will actually show rather than re-parsing the TOML.
"""

from importlib.metadata import metadata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

md = metadata('scoutbot')


def test_summary_describes_scoutbot_two():
    assert md.get('Summary') == (
        'YOLOv8-based machine learning pipeline for detecting and localising wildlife in aerial '
        'survey imagery.'
    )


def test_license_is_spdx_expression():
    assert md.get('License-Expression') == 'Apache-2.0'


def test_python_classifiers_match_supported_range():
    classifiers = md.get_all('Classifier') or []
    for v in ('3.10', '3.11', '3.12', '3.13'):
        assert f'Programming Language :: Python :: {v}' in classifiers
    assert 'Programming Language :: Python :: 3.14' not in classifiers


def test_project_urls_point_at_scoutbot_two():
    urls = dict(u.split(', ', 1) for u in md.get_all('Project-URL') or [])
    assert urls == {
        'Repository': 'https://github.com/SurveyTools/scoutbot-two',
        'Documentation': 'https://surveytools.github.io/scoutbot-two/',
        'Changelog': 'https://surveytools.github.io/scoutbot-two/changelog.html',
        'Issues': 'https://github.com/SurveyTools/scoutbot-two/issues',
    }


def test_setuptools_packages_cover_source_tree():
    # [tool.setuptools] packages is a hand-maintained list, so a new subpackage would otherwise
    # build fine and silently ship missing from the wheel.
    tomllib = pytest.importorskip('tomllib')  # stdlib from 3.11; skipped on 3.10
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as f:
        pyproject = tomllib.load(f)
    listed = set(pyproject['tool']['setuptools']['packages'])
    discovered = {
        '.'.join(init.parent.relative_to(REPO_ROOT).parts)
        for init in (REPO_ROOT / 'scoutbot').rglob('__init__.py')
    }
    assert listed == discovered
