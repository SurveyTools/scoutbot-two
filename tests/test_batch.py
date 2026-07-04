# -*- coding: utf-8 -*-
"""Tests for scoutbot.batch() and scoutbot.batch_resumable() functions."""

import json
import shutil
from unittest.mock import MagicMock, patch

import pytest


def _empty_sahi_result():
    class _Result:
        object_prediction_list = []

    return _Result()


@pytest.mark.slow
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

    def test_batch_forwards_observer_and_emits_event(self, sample_image):
        import scoutbot
        from scoutbot._observability import list_observer

        observer_events = []
        observer = list_observer(observer_events)
        mock_model = MagicMock()

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch,
            patch(
                'scoutbot.tile_batched.Yolov8DetectionModel',
                return_value=mock_model,
            ),
            patch('scoutbot.pipeline', return_value=(0.5, [])) as mock_pipeline,
        ):
            scoutbot.batch(
                [sample_image], config='v3', show_progress=False, observer=observer
            )

        mock_fetch.assert_called_once_with(config='v3', observer=observer)
        mock_pipeline.assert_called_once()
        assert mock_pipeline.call_args.kwargs['observer'] == observer

        batch_events = [event for event in observer_events if event['name'] == 'batch']
        assert batch_events, 'batch() must emit an observability event'
        event = batch_events[0]
        for key in (
            'name',
            'started_ns',
            'ended_ns',
            'duration_ns',
            'config',
            'backend',
            'device',
        ):
            assert key in event


@pytest.mark.slow
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

    def test_batch_resumable_forwards_observer_and_emits_event(
        self, sample_image, temp_output
    ):
        import scoutbot
        from scoutbot._observability import list_observer

        observer_events = []
        observer = list_observer(observer_events)

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch,
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.5, [])) as mock_pipeline,
        ):
            scoutbot.batch_resumable(
                [sample_image],
                temp_output,
                config='v3',
                show_progress=False,
                observer=observer,
            )

        mock_fetch.assert_called_once_with(config='v3', observer=observer)
        assert mock_pipeline.call_args.kwargs['observer'] == observer

        events = [
            event for event in observer_events if event['name'] == 'batch_resumable'
        ]
        assert events, 'batch_resumable() must emit an observability event'
        event = events[0]
        for key in (
            'name',
            'started_ns',
            'ended_ns',
            'duration_ns',
            'config',
            'backend',
            'device',
        ):
            assert key in event


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

        with (
            patch('tqdm.tqdm', self._fake_tqdm(captured)),
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.5, [])),
        ):
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

        with (
            patch('tqdm.tqdm', self._fake_tqdm(captured)),
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.5, [])),
        ):
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
            f.write(
                json.dumps(
                    {
                        'filepath': 'good.jpg',
                        'wic': 0.5,
                        'detections': [],
                        'config': 'v3',
                    }
                )
                + '\n'
            )
            f.write(
                json.dumps(
                    {
                        'filepath': 'bad.jpg',
                        'error': 'some transient error',
                        'config': 'v3',
                    }
                )
                + '\n'
            )

        processed = []

        def mock_pipeline(filepath, *args, **kwargs):
            processed.append(filepath)
            return 0.0, []

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=mock_pipeline),
        ):
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
            f.write(
                json.dumps(
                    {
                        'filepath': 'done.jpg',
                        'wic': 0.9,
                        'detections': [],
                        'config': 'v3',
                    }
                )
                + '\n'
            )

        processed = []

        def mock_pipeline(filepath, *args, **kwargs):
            processed.append(filepath)
            return 0.0, []

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=mock_pipeline),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=['done.jpg'],
                output_path=str(output),
                show_progress=False,
            )

        assert 'done.jpg' not in processed
        assert summary['skipped'] == 1
        assert summary['processed'] == 0


class TestApiEnvVarConfig:
    """LOC_CONFIG/CONFIG env vars must be respected by public API functions."""

    def test_fetch_uses_loc_default_config_when_no_config_given(self):
        """scoutbot.fetch() with no args must delegate to loc.DEFAULT_CONFIG."""
        import scoutbot
        from scoutbot import loc

        with (
            patch.object(loc, 'DEFAULT_CONFIG', 'v3-cls'),
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch,
        ):
            scoutbot.fetch()

        mock_fetch.assert_called_with(config='v3-cls')

    def test_get_classes_uses_loc_default_config_when_no_config_given(self):
        """scoutbot.get_classes() with no args must delegate to loc.DEFAULT_CONFIG."""
        from unittest.mock import MagicMock

        import scoutbot
        from scoutbot import loc

        mock_model = MagicMock()
        mock_model.category_names = ['animal']

        with (
            patch.object(loc, 'DEFAULT_CONFIG', 'v3-cls'),
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel', return_value=mock_model),
        ):
            result = scoutbot.get_classes()

        assert result == ['animal']


class TestBatchUnit:
    """Unit tests for scoutbot.batch() — no model or network."""

    def test_batch_returns_wic_and_detects_lists(self, tmp_path):
        import scoutbot

        fake_imgs = [str(tmp_path / f'{i}.jpg') for i in range(3)]
        for p in fake_imgs:
            open(p, 'wb').close()

        fake_wic = 0.75
        fake_detects = [{'l': 'animal', 'c': 0.75, 'x': 0, 'y': 0, 'w': 10, 'h': 10}]

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(fake_wic, fake_detects)),
        ):
            wic_list, detects_list = scoutbot.batch(
                fake_imgs, config='v3', show_progress=False
            )

        assert len(wic_list) == 3
        assert len(detects_list) == 3
        assert all(w == fake_wic for w in wic_list)

    def test_batch_none_config_uses_default(self, tmp_path):
        import scoutbot

        fake_img = str(tmp_path / 'img.jpg')
        open(fake_img, 'wb').close()

        with (
            patch('scoutbot._resolve_model', return_value=('/fake/model.pt', {}, {})) as mock_resolve,
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.0, [])),
            patch('scoutbot.loc.DEFAULT_CONFIG', 'v3'),
        ):
            scoutbot.batch([fake_img], config=None, show_progress=False)

        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs.get('config') == 'v3'

    def test_batch_emits_observer_events(self, tmp_path):
        import scoutbot

        fake_imgs = [str(tmp_path / f'{i}.jpg') for i in range(2)]
        for p in fake_imgs:
            open(p, 'wb').close()

        events = []

        class Collector:
            def emit(self, event):
                events.append(event)

        def _fake_pipeline(filepath, *args, observer=None, **kwargs):
            if observer is not None:
                observer.emit(
                    {
                        'name': 'pipeline',
                        'started_ns': 1,
                        'ended_ns': 2,
                        'duration_ns': 1,
                        'config': kwargs.get('config', 'v3'),
                        'backend': 'pt',
                        'device': kwargs.get('backend_device', 'cpu'),
                    }
                )
            return 0.5, []

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=_fake_pipeline),
        ):
            scoutbot.batch(
                fake_imgs, config='v3', show_progress=False, observer=Collector()
            )

        names = [event['name'] for event in events]
        assert 'batch' in names
        assert names.count('pipeline') == 2
        assert all('started_ns' in event for event in events)


class TestBatchResumableUnit:
    """Unit tests for scoutbot.batch_resumable() — no model or network."""

    def test_all_already_processed_returns_early(self, tmp_path):
        import scoutbot

        output = tmp_path / 'output.jsonl'
        filepaths = [str(tmp_path / 'a.jpg'), str(tmp_path / 'b.jpg')]

        # Write records for both filepaths as already done
        with open(output, 'w') as f:
            for fp in filepaths:
                f.write(json.dumps({'filepath': fp, 'wic': 0.5, 'detections': []}) + '\n')

        with patch('scoutbot.loc.fetch') as mock_fetch:
            summary = scoutbot.batch_resumable(
                filepaths=filepaths,
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        mock_fetch.assert_not_called()
        assert summary['processed'] == 0
        assert summary['skipped'] == 2
        assert summary['total'] == 2

    def test_jsonl_parser_skips_blank_lines_and_bad_json(self, tmp_path):
        """Lines 316 (blank) and 321-322 (JSONDecodeError) of batch_resumable."""
        import scoutbot

        output = tmp_path / 'output.jsonl'
        fake_img = str(tmp_path / 'img.jpg')
        open(fake_img, 'wb').close()

        with open(output, 'w') as f:
            f.write('\n')  # blank line → hits line 316
            f.write('not valid json\n')  # bad JSON → hits lines 321-322
            f.write(json.dumps({'bad': 'record'}) + '\n')  # valid JSON, no filepath key

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.0, [])),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=[fake_img],
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        # img.jpg was not in processed set (no valid completed record for it)
        assert summary['processed'] == 1
        assert summary['errors'] == 0

    def test_batch_resumable_runs_and_writes_jsonl(self, tmp_path):
        import scoutbot

        output = tmp_path / 'output.jsonl'
        fake_imgs = [str(tmp_path / f'{i}.jpg') for i in range(2)]
        for p in fake_imgs:
            open(p, 'wb').close()

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch(
                'scoutbot.pipeline',
                return_value=(
                    0.85,
                    [{'l': 'animal', 'c': 0.85, 'x': 0, 'y': 0, 'w': 5, 'h': 5}],
                ),
            ),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=fake_imgs,
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        assert summary['processed'] == 2
        assert summary['errors'] == 0
        assert summary['skipped'] == 0
        assert summary['total'] == 2
        assert output.exists()
        lines = [
            json.loads(line) for line in output.read_text().splitlines() if line.strip()
        ]
        assert len(lines) == 2
        for rec in lines:
            assert 'filepath' in rec
            assert 'wic' in rec
            assert 'detections' in rec
            assert 'config' in rec

    def test_batch_resumable_resumes_from_partial_output(self, tmp_path):
        import scoutbot

        output = tmp_path / 'output.jsonl'
        img_a = str(tmp_path / 'a.jpg')
        img_b = str(tmp_path / 'b.jpg')
        open(img_a, 'wb').close()
        open(img_b, 'wb').close()

        # Pre-populate with img_a already done
        with open(output, 'w') as f:
            f.write(json.dumps({'filepath': img_a, 'wic': 0.5, 'detections': []}) + '\n')

        pipeline_calls = []

        def _fake_pipeline(fp, *args, **kwargs):
            pipeline_calls.append(fp)
            return (0.9, [])

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=_fake_pipeline),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=[img_a, img_b],
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        # Only img_b should have been processed
        assert pipeline_calls == [img_b]
        assert summary['processed'] == 1
        assert summary['skipped'] == 1

    def test_batch_resumable_writes_error_record_on_pipeline_failure(self, tmp_path):
        """Covers lines 392-402: except Exception path writes an error record."""
        import scoutbot

        output = tmp_path / 'output.jsonl'
        fake_img = str(tmp_path / 'img.jpg')
        open(fake_img, 'wb').close()

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=RuntimeError('inference exploded')),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=[fake_img],
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        assert summary['errors'] == 1
        assert summary['processed'] == 0
        assert output.exists()
        lines = [
            json.loads(line) for line in output.read_text().splitlines() if line.strip()
        ]
        assert len(lines) == 1
        assert lines[0]['filepath'] == fake_img
        assert 'error' in lines[0]
        assert 'inference exploded' in lines[0]['error']

    def test_batch_resumable_emits_observer_events(self, tmp_path):
        import scoutbot

        output = tmp_path / 'output.jsonl'
        fake_img = str(tmp_path / 'img.jpg')
        open(fake_img, 'wb').close()

        events = []

        class Collector:
            def emit(self, event):
                events.append(event)

        def _fake_pipeline(filepath, *args, observer=None, **kwargs):
            if observer is not None:
                observer.emit(
                    {
                        'name': 'pipeline',
                        'started_ns': 1,
                        'ended_ns': 2,
                        'duration_ns': 1,
                        'config': kwargs.get('config', 'v3'),
                        'backend': 'pt',
                        'device': kwargs.get('backend_device', 'cpu'),
                    }
                )
            return 0.5, []

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=_fake_pipeline),
        ):
            scoutbot.batch_resumable(
                filepaths=[fake_img],
                output_path=str(output),
                config='v3',
                show_progress=False,
                observer=Collector(),
            )

        names = [event['name'] for event in events]
        assert 'batch_resumable' in names
        assert names.count('pipeline') == 1
        assert all('started_ns' in event for event in events)


@pytest.mark.slow
class TestExampleSlow:
    """Integration test for scoutbot.example() — downloads from network."""

    def test_example_runs_without_error(self):
        import scoutbot

        fake_response = MagicMock()
        fake_response.iter_content.return_value = [b'fake-image-bytes']
        fake_response.raise_for_status = MagicMock()
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)

        with (
            patch('requests.get', return_value=fake_response),
            patch('scoutbot.pipeline', return_value=(0.5, [])),
        ):
            scoutbot.example()
