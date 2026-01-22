# -*- coding: utf-8 -*-
"""Tests for scoutbot.pipeline() function."""
import pytest


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
