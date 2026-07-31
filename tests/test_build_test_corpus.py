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


def _rec(fp, n):
    return {'filepath': fp, 'detections': [{} for _ in range(n)]}


def test_select_tiers_buckets_and_counts():
    records = (
        [_rec(f'/r/empty{i}.jpg', 0) for i in range(6)]
        + [_rec(f'/r/typ{i}.jpg', c) for i, c in enumerate([3, 4, 5, 6, 4, 5])]
        + [_rec(f'/r/den{i}.jpg', c) for i, c in enumerate([9, 10, 11, 12, 13, 10])]
        + [_rec(f'/r/ext{i}.jpg', c) for i, c in enumerate([20, 40, 80, 175, 30, 14])]
        # counts 1,2,7,8 fall in no tier and must be ignored:
        + [_rec('/r/gap1.jpg', 1), _rec('/r/gap7.jpg', 7), _rec('/r/gap8.jpg', 8)]
    )
    picked = btc.select_tiers(records, per_tier=3)
    assert set(picked) == set(btc.TIERS)
    for tier in btc.TIERS:
        assert len(picked[tier]) == 3
    assert all(r['_count'] == 0 for r in picked['empty'])
    assert all(3 <= r['_count'] <= 6 for r in picked['typical'])
    assert all(9 <= r['_count'] <= 13 for r in picked['dense'])
    assert all(r['_count'] >= 14 for r in picked['extreme'])
    # extreme is top-N desc → always includes the global max (175)
    assert max(r['_count'] for r in picked['extreme']) == 175


def test_select_tiers_is_deterministic():
    records = [_rec(f'/r/den{i}.jpg', c) for i, c in enumerate([9, 10, 11, 12, 13])] + [
        _rec(f'/r/e{i}.jpg', c) for i, c in enumerate([0, 0, 0, 4, 5, 6, 20, 40, 60])
    ]
    a = btc.select_tiers(records, per_tier=2)
    b = btc.select_tiers(records, per_tier=2)
    assert [r['filepath'] for r in a['dense']] == [r['filepath'] for r in b['dense']]


def test_select_tiers_short_bucket_raises():
    records = (
        [_rec('/r/e1.jpg', 0)]
        + [_rec(f'/r/t{i}.jpg', 4) for i in range(5)]
        + [_rec(f'/r/d{i}.jpg', 10) for i in range(5)]
        + [_rec(f'/r/x{i}.jpg', 50) for i in range(5)]
    )
    with pytest.raises(ValueError, match='empty'):
        btc.select_tiers(records, per_tier=5)  # only 1 empty available
