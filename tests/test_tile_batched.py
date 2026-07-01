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
