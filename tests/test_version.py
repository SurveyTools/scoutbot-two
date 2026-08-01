from importlib.metadata import version as package_version

import scoutbot


def test_version_matches_package_metadata():
    # pyproject.toml is the single source of truth; __version__ derives from installed metadata,
    # so it must never drift from the packaged version (regression against the old hardcoded string).
    assert scoutbot.__version__ == package_version('scoutbot')


def test_version_aliases_agree():
    assert scoutbot.VERSION == scoutbot.__version__
    assert scoutbot.version == scoutbot.__version__
