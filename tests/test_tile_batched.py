# -*- coding: utf-8 -*-
"""Tests for scoutbot.tile_batched.main (no model/network needed)."""
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestLoadModelPlatformGuard:
    """Unit tests for Yolov8DetectionModel.load_model() platform checks."""

    def test_mlpackage_raises_on_non_darwin(self):
        """Loading a .mlpackage on non-macOS must raise RuntimeError, not ModuleNotFoundError."""
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance.model_path = '/fake/model.mlpackage'

        with patch.object(sys, 'platform', 'linux'):
            with pytest.raises(RuntimeError, match='macOS'):
                Yolov8DetectionModel.load_model(instance)

    def test_mlpackage_allowed_on_darwin(self):
        """Loading a .mlpackage on macOS must not raise a platform error."""
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance.model_path = '/fake/model.mlpackage'

        with patch.object(sys, 'platform', 'darwin'), \
             patch('scoutbot.tile_batched.main._coreml_input_size', return_value=512), \
             patch('ultralytics.YOLO'):
            Yolov8DetectionModel.load_model(instance)  # must not raise


class TestLoadModelPtTaskDetect:
    """Verify PT load_model passes task='detect' to prevent cls misidentification."""

    def test_pt_load_model_passes_task_detect(self):
        """PT load_model must call YOLO(path, task='detect') to prevent boxes=None."""
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance.model_path = '/fake/yolov8-cls.kaza.pt'
        instance.device = 'cpu'

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.to.return_value = mock_yolo_instance

        with patch('ultralytics.YOLO', return_value=mock_yolo_instance) as MockYOLO:
            Yolov8DetectionModel.load_model(instance)

        MockYOLO.assert_called_once_with('/fake/yolov8-cls.kaza.pt', task='detect')
        instance.set_model.assert_called_once_with(mock_yolo_instance)


class TestPerformInferenceNoneBoxes:
    """Verify perform_inference handles result.boxes=None without crashing."""

    def test_none_boxes_produces_empty_prediction(self):
        """result.boxes=None (from a misidentified model) must not raise AttributeError."""
        import numpy as np
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance._coreml = False
        instance.batch_size = 128
        instance.device = 'cpu'
        instance.confidence_threshold = 0.45
        instance.model = MagicMock()

        result_with_none_boxes = MagicMock()
        result_with_none_boxes.boxes = None
        instance.model.predict.return_value = [result_with_none_boxes]

        Yolov8DetectionModel.perform_inference(instance, [np.zeros((512, 512, 3), dtype=np.uint8)])

        assert instance._original_predictions is not None
        assert len(instance._original_predictions) == 1
