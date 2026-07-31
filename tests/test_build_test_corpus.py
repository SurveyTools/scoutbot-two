import build_test_corpus as btc
import pytest


def test_resolve_local_full_suffix_ignores_basename_collisions(tmp_path):
    # Two sibling dirs hold the SAME basename; full-suffix resolution must land on the right file.
    src_root = '/home/brandon/matetsi_images'
    left = tmp_path / 'day1' / 'flight1' / 'left'
    right = tmp_path / 'day1' / 'flight1' / 'right'
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    (left / 'BCK_0001.JPG').write_bytes(b'L')
    (right / 'BCK_0001.JPG').write_bytes(b'R')

    got = btc.resolve_local(
        f'{src_root}/day1/flight1/right/BCK_0001.JPG', src_root, tmp_path
    )
    assert got == right / 'BCK_0001.JPG'
    assert got.read_bytes() == b'R'


def test_resolve_local_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        btc.resolve_local(
            '/home/brandon/matetsi_images/day1/nope.JPG',
            '/home/brandon/matetsi_images',
            tmp_path,
        )


def test_resolve_local_foreign_prefix_raises(tmp_path):
    with pytest.raises(ValueError):
        btc.resolve_local('/mnt/e/other/x.JPG', '/home/brandon/matetsi_images', tmp_path)


def test_resolve_local_rejects_traversal(tmp_path):
    # Maintainer tool consuming path strings from a JSONL — refuse '..' escaping the root.
    with pytest.raises(ValueError, match='traversal|escapes'):
        btc.resolve_local(
            '/home/brandon/matetsi_images/../../etc/passwd',
            '/home/brandon/matetsi_images',
            tmp_path,
        )
