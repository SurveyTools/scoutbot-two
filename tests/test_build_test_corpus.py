from pathlib import Path

import build_test_corpus as btc
import piexif
import pytest
from PIL import Image

import scoutbot
from scoutbot.model_store import CONFIGS


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


def _jpeg_with_gps(path, size=(64, 48)):
    Image.new('RGB', size, (10, 20, 30)).save(path, 'JPEG')
    exif = {
        '0th': {piexif.ImageIFD.Make: b'TestCam'},
        'GPS': {piexif.GPSIFD.GPSLatitudeRef: b'S'},
        'Exif': {},
        '1st': {},
        'thumbnail': None,
    }
    piexif.insert(piexif.dump(exif), str(path))


def test_strip_and_verify_removes_gps_and_keeps_dimensions(tmp_path):
    src = tmp_path / 'src.jpg'
    dst = tmp_path / 'dst.jpg'
    _jpeg_with_gps(src, size=(64, 48))
    # sanity: GPS present before strip
    assert piexif.load(str(src))['GPS']

    btc.strip_and_verify(src, dst)

    after = piexif.load(str(dst))
    assert not after['GPS'] and not after['0th']
    with Image.open(dst) as im:
        assert im.size == (64, 48)


def test_strip_and_verify_on_clean_jpeg_is_ok(tmp_path):
    src = tmp_path / 'clean.jpg'
    dst = tmp_path / 'clean_out.jpg'
    Image.new('RGB', (32, 32), (1, 2, 3)).save(src, 'JPEG')  # no EXIF
    btc.strip_and_verify(src, dst)  # must not raise
    assert dst.is_file()


def test_run_config_for_maps_thresh_to_loc_thresh():
    rc = btc.run_config_for('v3')
    assert rc['backend_device'] == 'cpu'
    assert rc['loc_thresh'] == CONFIGS['v3']['thresh']
    assert rc['slice_height'] == CONFIGS['v3']['slice_height']
    assert rc['postprocess_class_agnostic'] == CONFIGS['v3']['postprocess_class_agnostic']


def test_collect_runtime_has_required_keys(tmp_path):
    lock = tmp_path / 'uv.lock'
    lock.write_bytes(b'lockdata')
    rt = btc.collect_runtime(lock)
    for k in (
        'scoutbot_version',
        'scoutbot_commit',
        'python',
        'platform',
        'platform_system',
        'platform_machine',
        'torch',
        'numpy',
        'lockfile_sha256',
    ):
        assert k in rt
    assert rt['scoutbot_version'] == scoutbot.__version__
    assert len(rt['lockfile_sha256']) == 64  # sha256 hex


def test_build_manifest_shape():
    images = [
        {
            'name': 'dense_01.jpg',
            'tier': 'dense',
            'md5': 'a' * 32,
            'source_density': 11,
            'baseline_detection_count': 11,
        },
    ]
    runtime = {'scoutbot_version': '0.5.0'}
    m = btc.build_manifest('v3', 5, images, runtime, model_md5='b' * 32)
    assert m['schema'] == 'scoutbot:test-corpus:1.0'
    assert m['model'] == {'config': 'v3', 'name': CONFIGS['v3']['name'], 'md5': 'b' * 32}
    assert m['run_config']['loc_thresh'] == CONFIGS['v3']['thresh']
    assert m['runtime'] is runtime
    assert m['images'][0]['baseline_detection_count'] == 11
    assert m['per_tier'] == 5


def test_assert_model_present_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(btc, 'get_models_dir', lambda: tmp_path)  # empty dir
    with pytest.raises(SystemExit, match='fetch'):
        btc.assert_model_present('v3')


def test_source_segments_collects_path_parts_and_basenames():
    records = [
        {'filepath': '/home/brandon/matetsi_images/141025_day1/flight1/left/BCK_0001.JPG'}
    ]
    segs = btc.source_segments(
        records, '/home/brandon/matetsi_images', Path('/vol/matetsi/images')
    )
    assert '/home/brandon/matetsi_images' in segs
    assert '/vol/matetsi/images' in segs
    assert '141025_day1' in segs
    assert 'left' in segs
    assert 'BCK_0001.JPG' in segs
    assert 'BCK_0001' in segs  # stem too


def test_source_segments_excludes_tier_labels():
    # A source dir that happens to equal a tier name must NOT become a forbidden token — tier
    # labels are legitimate emitted vocabulary.
    records = [{'filepath': '/home/brandon/matetsi_images/dense/left/CAM_0001.JPG'}]
    segs = btc.source_segments(records, '/home/brandon/matetsi_images', Path('/vol/x'))
    assert 'dense' not in segs
    assert 'CAM_0001' in segs


def test_assert_no_source_identifiers_flags_leak(tmp_path):
    (tmp_path / 'manifest.json').write_text('{"note": "from 141025_day1/left"}')
    with pytest.raises(AssertionError, match='141025_day1'):
        btc.assert_no_source_identifiers(tmp_path, {'141025_day1'})


def test_assert_no_source_identifiers_allows_readme_paths(tmp_path):
    # README legitimately carries commands, env vars, and corpus-relative paths — '/' is fine.
    (tmp_path / 'README.md').write_text(
        'Set `SCOUTBOT_TEST_CORPUS=/path/to/corpus` and run `uv run pytest tests/`.\n'
    )
    btc.assert_no_source_identifiers(
        tmp_path, {'141025_day1', 'BCK_0001'}
    )  # must not raise
