# -*- coding: utf-8 -*-
"""Tests for scoutbot.batch() and scoutbot.batch_resumable() functions."""
import json
import shutil
from unittest.mock import patch


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


class TestBatchResumableProgressBar:
    """Unit tests for batch_resumable progress bar behaviour (no model/network needed)."""

    def _fake_tqdm(self, captured):
        """Return a tqdm stand-in that records the disable kwarg."""
        def factory(iterable, **kwargs):
            captured.update(kwargs)

            class FakePbar:
                def __iter__(self):
                    return iter(iterable)

                def set_postfix(self, **kw):
                    pass

            return FakePbar()

        return factory

    def test_progress_shown_when_show_progress_true(self, tmp_path):
        """show_progress=True must not be suppressed by the QUIET module constant."""
        import scoutbot

        assert scoutbot.QUIET is True, (
            'QUIET must be True (VERBOSE env var unset) for this test to prove the regression'
        )

        captured = {}

        with patch('tqdm.tqdm', self._fake_tqdm(captured)), \
             patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', return_value=(0.5, [])):
            scoutbot.batch_resumable(
                filepaths=['fake.jpg'],
                output_path=str(tmp_path / 'out.jsonl'),
                show_progress=True,
            )

        assert captured.get('disable') is False, (
            'progress bar must be enabled when show_progress=True; '
            'check that QUIET is not short-circuiting the disable flag in batch_resumable'
        )

    def test_progress_hidden_when_show_progress_false(self, tmp_path):
        """show_progress=False must suppress the progress bar regardless of QUIET."""
        import scoutbot

        captured = {}

        with patch('tqdm.tqdm', self._fake_tqdm(captured)), \
             patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', return_value=(0.5, [])):
            scoutbot.batch_resumable(
                filepaths=['fake.jpg'],
                output_path=str(tmp_path / 'out.jsonl'),
                show_progress=False,
            )

        assert captured.get('disable') is True


class TestBatchResumableErrorResume:
    """Unit tests for resume-on-error behaviour (no model/network needed)."""

    def test_errored_files_retried_on_resume(self, tmp_path):
        """A file that errored previously must be retried, not silently skipped."""
        import scoutbot

        output = tmp_path / 'out.jsonl'

        # Seed JSONL: one success record, one error record
        with open(output, 'w') as f:
            f.write(json.dumps({
                'filepath': 'good.jpg',
                'wic': 0.5,
                'detections': [],
                'config': 'v3',
            }) + '\n')
            f.write(json.dumps({
                'filepath': 'bad.jpg',
                'error': 'some transient error',
                'config': 'v3',
            }) + '\n')

        processed = []

        def mock_pipeline(filepath, *args, **kwargs):
            processed.append(filepath)
            return 0.0, []

        with patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', side_effect=mock_pipeline):
            summary = scoutbot.batch_resumable(
                filepaths=['good.jpg', 'bad.jpg'],
                output_path=str(output),
                show_progress=False,
            )

        assert 'good.jpg' not in processed, 'successfully completed file must be skipped'
        assert 'bad.jpg' in processed, 'previously errored file must be retried'
        assert summary['skipped'] == 1
        assert summary['processed'] == 1

    def test_successful_files_still_skipped_on_resume(self, tmp_path):
        """Files completed successfully must continue to be skipped on resume."""
        import scoutbot

        output = tmp_path / 'out.jsonl'

        with open(output, 'w') as f:
            f.write(json.dumps({
                'filepath': 'done.jpg',
                'wic': 0.9,
                'detections': [],
                'config': 'v3',
            }) + '\n')

        processed = []

        def mock_pipeline(filepath, *args, **kwargs):
            processed.append(filepath)
            return 0.0, []

        with patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', side_effect=mock_pipeline):
            summary = scoutbot.batch_resumable(
                filepaths=['done.jpg'],
                output_path=str(output),
                show_progress=False,
            )

        assert 'done.jpg' not in processed
        assert summary['skipped'] == 1
        assert summary['processed'] == 0
