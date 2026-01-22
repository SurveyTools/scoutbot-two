# -*- coding: utf-8 -*-
"""Tests for scoutbot.batch() and scoutbot.batch_resumable() functions."""
import json
import pytest
import shutil


class TestBatch:
    """Tests for batch processing."""

    def test_batch_returns_lists(self, sample_image):
        """Test that batch returns two lists."""
        import scoutbot

        wic_list, detects_list = scoutbot.batch(
            [sample_image],
            config='v3',
            show_progress=False,
        )

        assert isinstance(wic_list, list)
        assert isinstance(detects_list, list)
        assert len(wic_list) == 1
        assert len(detects_list) == 1

    def test_batch_multiple_images(self, sample_image, tmp_path):
        """Test batch with multiple images."""
        import scoutbot

        # Create a copy of the sample image
        img2 = tmp_path / 'img2.jpg'
        shutil.copy(sample_image, img2)

        wic_list, detects_list = scoutbot.batch(
            [sample_image, str(img2)],
            config='v3',
            show_progress=False,
        )

        assert len(wic_list) == 2
        assert len(detects_list) == 2

    def test_batch_progress_disabled(self, sample_image):
        """Test that progress bar can be disabled."""
        import scoutbot

        # Should not raise an error
        wic_list, detects_list = scoutbot.batch(
            [sample_image],
            config='v3',
            show_progress=False,
        )

        assert len(wic_list) == 1


class TestBatchResumable:
    """Tests for resumable batch processing."""

    def test_batch_resumable_creates_jsonl(self, sample_image, temp_output):
        """Test that batch_resumable creates a JSONL file."""
        import scoutbot

        scoutbot.batch_resumable(
            [sample_image],
            temp_output,
            config='v3',
            show_progress=False,
        )

        assert temp_output.exists()

        # Read and verify content
        with open(temp_output, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 1

        record = json.loads(lines[0])
        assert 'filepath' in record
        assert 'wic' in record
        assert 'detections' in record
        assert 'timestamp' in record
        assert 'config' in record

    def test_batch_resumable_returns_summary(self, sample_image, temp_output):
        """Test that batch_resumable returns a summary dict."""
        import scoutbot

        summary = scoutbot.batch_resumable(
            [sample_image],
            temp_output,
            config='v3',
            show_progress=False,
        )

        assert isinstance(summary, dict)
        assert 'processed' in summary
        assert 'skipped' in summary
        assert 'errors' in summary
        assert 'total' in summary

        assert summary['processed'] == 1
        assert summary['skipped'] == 0
        assert summary['errors'] == 0
        assert summary['total'] == 1

    def test_batch_resumable_skips_processed(self, sample_image, temp_output):
        """Test that resume skips already-processed files."""
        import scoutbot

        # Process once
        scoutbot.batch_resumable(
            [sample_image],
            temp_output,
            config='v3',
            show_progress=False,
        )

        # Count lines
        with open(temp_output, 'r') as f:
            lines_before = len(f.readlines())

        # Process again (should skip)
        summary = scoutbot.batch_resumable(
            [sample_image],
            temp_output,
            config='v3',
            show_progress=False,
        )

        # Count lines again
        with open(temp_output, 'r') as f:
            lines_after = len(f.readlines())

        # Should not have added new lines
        assert lines_after == lines_before
        assert summary['processed'] == 0
        assert summary['skipped'] == 1

    def test_batch_resumable_continues_partial(self, sample_image, temp_output, tmp_path):
        """Test that processing can resume after partial completion."""
        import scoutbot

        # Create a second test image
        img2 = tmp_path / 'img2.jpg'
        shutil.copy(sample_image, img2)

        # Process first image only
        scoutbot.batch_resumable(
            [sample_image],
            temp_output,
            config='v3',
            show_progress=False,
        )

        # Now add second image to batch (simulating resume)
        summary = scoutbot.batch_resumable(
            [sample_image, str(img2)],
            temp_output,
            config='v3',
            show_progress=False,
        )

        # Should have processed only the new image
        assert summary['processed'] == 1
        assert summary['skipped'] == 1

        # Should have 2 results total
        with open(temp_output, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_batch_resumable_all_processed_returns_early(self, sample_image, temp_output):
        """Test that all-processed state returns immediately."""
        import scoutbot

        # Process once
        scoutbot.batch_resumable(
            [sample_image],
            temp_output,
            config='v3',
            show_progress=False,
        )

        # Process again
        summary = scoutbot.batch_resumable(
            [sample_image],
            temp_output,
            config='v3',
            show_progress=False,
        )

        assert summary['processed'] == 0
        assert summary['skipped'] == 1
        assert summary['total'] == 1
