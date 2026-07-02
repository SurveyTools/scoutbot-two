# -*- coding: utf-8 -*-
"""Tests for scoutbot.utils module."""

import hashlib


class TestCheckFileIntegrity:
    """Tests for check_file_integrity() — no model or network needed."""

    def test_returns_false_when_file_does_not_exist(self):
        from scoutbot.utils import check_file_integrity

        result = check_file_integrity('/nonexistent/path/to/file.bin', 'anyhash')
        assert result is False

    def test_returns_true_when_expected_hash_is_none(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'file.bin'
        f.write_bytes(b'some content')

        result = check_file_integrity(str(f), None)
        assert result is True

    def test_returns_true_when_sha256_matches(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        content = b'wildlife detection payload'
        f = tmp_path / 'model.pt'
        f.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = check_file_integrity(str(f), expected)
        assert result is True

    def test_returns_false_when_sha256_does_not_match(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'model.pt'
        f.write_bytes(b'actual content')

        result = check_file_integrity(str(f), 'wrong_hash_value')
        assert result is False

    def test_handles_large_file_in_blocks(self, tmp_path):
        """Exercises the iter(lambda: f.read(4096), b'') chunk loop."""
        from scoutbot.utils import check_file_integrity

        # 16KB forces multiple 4096-byte reads
        content = b'z' * 16_384
        f = tmp_path / 'large.bin'
        f.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert check_file_integrity(str(f), expected) is True

    def test_returns_false_for_empty_file_with_wrong_hash(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'empty.bin'
        f.write_bytes(b'')

        result = check_file_integrity(str(f), 'not_the_sha256_of_empty')
        assert result is False

    def test_returns_true_for_empty_file_with_correct_hash(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'empty.bin'
        f.write_bytes(b'')

        expected = hashlib.sha256(b'').hexdigest()
        assert check_file_integrity(str(f), expected) is True
