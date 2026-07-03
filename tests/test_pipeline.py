# -*- coding: utf-8 -*-
"""Tests for scoutbot.pipeline() function."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.slow
class TestPipeline:
    """Tests for single image processing."""

    def test_pipeline_v3_returns_tuple(self, sample_image):
        """Test that pipeline returns (wic, detections) tuple."""
        import scoutbot

        result = scoutbot.pipeline(sample_image, config='v3')

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pipeline_v3_wic_score(self, sample_image):
        """Test that WIC score is a valid float between 0 and 1."""
        import scoutbot

        wic, _ = scoutbot.pipeline(sample_image, config='v3')

        assert isinstance(wic, float)
        assert 0.0 <= wic <= 1.0

    def test_pipeline_v3_detections_format(self, sample_image):
        """Test that detections have correct format."""
        import scoutbot

        _, detects = scoutbot.pipeline(sample_image, config='v3')

        assert isinstance(detects, list)

        # If there are detections, verify their structure
        for det in detects:
            assert isinstance(det, dict)
            assert 'l' in det  # label
            assert 'c' in det  # confidence
            assert 'x' in det  # x coordinate
            assert 'y' in det  # y coordinate
            assert 'w' in det  # width
            assert 'h' in det  # height

            # Validate types
            assert isinstance(det['l'], str)
            assert isinstance(det['c'], (int, float))
            assert 0.0 <= det['c'] <= 1.0

    def test_pipeline_v3_cls(self, sample_image):
        """Test pipeline with v3-cls config."""
        import scoutbot

        wic, detects = scoutbot.pipeline(sample_image, config='v3-cls')

        assert isinstance(wic, float)
        assert isinstance(detects, list)

    def test_pipeline_default_config(self, sample_image):
        """Test that default config is v3."""
        import scoutbot

        # Should not raise an error with None config
        wic, detects = scoutbot.pipeline(sample_image, config=None)

        assert isinstance(wic, float)
        assert isinstance(detects, list)

    def test_pipeline_cpu_device(self, sample_image):
        """Test pipeline on CPU device."""
        import scoutbot

        wic, detects = scoutbot.pipeline(
            sample_image,
            config='v3',
            backend_device='cpu',
        )

        assert isinstance(wic, float)
        assert isinstance(detects, list)

    def test_pipeline_custom_threshold(self, sample_image):
        """Test pipeline with custom detection threshold."""
        import scoutbot

        # High threshold should result in fewer detections
        _, detects_high = scoutbot.pipeline(
            sample_image,
            config='v3',
            loc_thresh=0.9,
        )

        # Low threshold should result in more detections
        _, detects_low = scoutbot.pipeline(
            sample_image,
            config='v3',
            loc_thresh=0.1,
        )

        # Low threshold should have >= detections than high threshold
        assert len(detects_low) >= len(detects_high)


def _fake_sahi_result(predictions):
    """Build a fake SlicedPredictionResult with the given COCO-format predictions."""
    mock_result = MagicMock()
    mock_preds = []
    for score, category, bbox in predictions:
        pred = MagicMock()
        pred.to_coco_prediction.return_value.json = {
            'score': score,
            'category_name': category,
            'bbox': bbox,
        }
        mock_preds.append(pred)
    mock_result.object_prediction_list = mock_preds
    return mock_result


def _empty_sahi_result():
    return _fake_sahi_result([])


class TestPipelineUnit:
    """Unit tests for scoutbot.pipeline() using mocked SAHI — no model or network."""

    def test_pipeline_returns_wic_and_detects(self, tmp_path):
        import scoutbot

        fake_img = tmp_path / 'test.jpg'
        fake_img.write_bytes(b'fake')

        fake_result = _fake_sahi_result(
            [
                (0.85, 'animal', [10.0, 20.0, 50.0, 30.0]),
                (0.60, 'vehicle', [100.0, 50.0, 20.0, 15.0]),
            ]
        )

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            wic, detects = scoutbot.pipeline(str(fake_img), config='v3')

        assert wic == 0.85
        assert len(detects) == 2
        assert detects[0]['l'] == 'animal'
        assert detects[0]['c'] == 0.85
        assert detects[0]['x'] == 10.0
        assert detects[0]['y'] == 20.0
        assert detects[0]['w'] == 50.0
        assert detects[0]['h'] == 30.0

    def test_pipeline_returns_zero_wic_when_no_detections(self, tmp_path):
        import scoutbot

        fake_img = tmp_path / 'empty.jpg'
        fake_img.write_bytes(b'fake')

        fake_result = _fake_sahi_result([])  # no detections

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            wic, detects = scoutbot.pipeline(str(fake_img), config='v3')

        assert wic == 0
        assert detects == []

    def test_pipeline_reuses_provided_model(self, tmp_path):
        """When batched_detection_model is given, loc.fetch must not be called."""
        import scoutbot

        fake_img = tmp_path / 'test.jpg'
        fake_img.write_bytes(b'fake')

        mock_model = MagicMock()
        fake_result = _fake_sahi_result([])

        with (
            patch('scoutbot.loc.fetch') as mock_fetch,
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            scoutbot.pipeline(
                str(fake_img), config='v3', batched_detection_model=mock_model
            )

        mock_fetch.assert_not_called()

    def test_pipeline_none_config_defaults_to_v3(self, tmp_path):
        import scoutbot

        fake_img = tmp_path / 'test.jpg'
        fake_img.write_bytes(b'fake')
        fake_result = _fake_sahi_result([])

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch,
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
            patch('scoutbot.loc.DEFAULT_CONFIG', 'v3'),
        ):
            scoutbot.pipeline(str(fake_img), config=None)

        mock_fetch.assert_called_once_with(config='v3')

    def test_pipeline_emits_observability_event_and_forwards_observer(self, tmp_path):
        import scoutbot
        from scoutbot._observability import list_observer

        fake_img = tmp_path / 'observed.jpg'
        fake_img.write_bytes(b'fake')
        fake_result = _fake_sahi_result([(0.5, 'animal', [1.0, 2.0, 3.0, 4.0])])
        observer_events = []
        observer = list_observer(observer_events)

        mock_model = MagicMock()
        mock_model.model_path = '/fake/model.pt'

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch,
            patch(
                'scoutbot.tile_batched.Yolov8DetectionModel', return_value=mock_model
            ) as MockModel,
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            scoutbot.pipeline(str(fake_img), config='v3', observer=observer)

        mock_fetch.assert_called_once_with(config='v3', observer=observer)
        MockModel.assert_called_once_with(
            model_path='/fake/model.pt',
            confidence_threshold=0.45,
            device='cuda:0',
            observer=observer,
        )

        pipeline_events = [
            event for event in observer_events if event['name'] == 'pipeline'
        ]
        assert pipeline_events, 'pipeline() must emit an observability event'
        event = pipeline_events[0]
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
        assert event['filepath'] == str(fake_img)
        assert event['config'] == 'v3'

    def test_pipeline_emits_observer_events(self, tmp_path):
        import scoutbot

        fake_img = tmp_path / 'observer.jpg'
        fake_img.write_bytes(b'fake')
        events = []

        class Collector:
            def emit(self, event):
                events.append(event)

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch(
                'sahi.predict.get_sliced_prediction', return_value=_empty_sahi_result()
            ),
        ):
            scoutbot.pipeline(
                str(fake_img),
                config='v3',
                observer=Collector(),
                batched_detection_model=object(),
            )

        assert events
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
