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

    def test_mlpackage_allowed_on_darwin(self, ultralytics_stub):
        """Loading a .mlpackage on macOS must not raise a platform error."""
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance.model_path = '/fake/model.mlpackage'

        with (
            patch.object(sys, 'platform', 'darwin'),
            patch('scoutbot.tile_batched.main._coreml_input_size', return_value=512),
            patch('ultralytics.YOLO'),
        ):
            Yolov8DetectionModel.load_model(instance)  # must not raise


class TestLoadModelPtTaskDetect:
    """Verify PT load_model passes task='detect' to prevent cls misidentification."""

    def test_pt_load_model_passes_task_detect(self, ultralytics_stub):
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

        Yolov8DetectionModel.perform_inference(
            instance, [np.zeros((512, 512, 3), dtype=np.uint8)]
        )

        assert instance._original_predictions is not None
        assert len(instance._original_predictions) == 1


class TestPerformInferenceConfKwarg:
    """Verify model.predict() receives conf= to avoid below-threshold inference work."""

    def test_predict_receives_conf_kwarg(self):
        """model.predict() must receive conf=confidence_threshold."""
        import numpy as np
        import torch

        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance._coreml = False
        instance.batch_size = 128
        instance.device = 'cpu'
        instance.confidence_threshold = 0.45
        instance.model = MagicMock()

        fake_boxes = MagicMock()
        fake_boxes.data = torch.zeros((0, 6))
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        instance.model.predict.return_value = [fake_result]

        Yolov8DetectionModel.perform_inference(
            instance, [np.zeros((512, 512, 3), dtype=np.uint8)]
        )

        call_kwargs = instance.model.predict.call_args.kwargs
        assert 'conf' in call_kwargs, (
            'model.predict() must receive conf= to skip below-threshold inference; '
            'without it, Ultralytics defaults to conf=0.25 and wastes ANE compute'
        )
        assert call_kwargs['conf'] == 0.45

    def test_predict_receives_conf_kwarg_for_coreml(self):
        """CoreML path must also receive conf=confidence_threshold alongside imgsz."""
        import numpy as np
        import torch

        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance._coreml = True
        instance._coreml_imgsz = 512
        instance.batch_size = 128
        instance.device = 'cpu'
        instance.confidence_threshold = 0.45
        instance.model = MagicMock()

        fake_boxes = MagicMock()
        fake_boxes.data = torch.zeros((0, 6))
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        instance.model.predict.return_value = [fake_result]

        Yolov8DetectionModel.perform_inference(
            instance, [np.zeros((512, 512, 3), dtype=np.uint8)]
        )

        call_kwargs = instance.model.predict.call_args.kwargs
        assert call_kwargs.get('conf') == 0.45
        assert call_kwargs.get('imgsz') == 512


class TestCoremlInputSize:
    """Unit tests for _coreml_input_size() — no coremltools installation needed."""

    def test_reads_width_from_model_spec(self):
        # Import outside the patch context so torch and friends stay in sys.modules
        # after the context exits.  The mock only needs to be active when the function
        # is *called* (that's when `import coremltools as ct` executes lazily).
        from scoutbot.tile_batched.main import _coreml_input_size

        mock_ct = MagicMock()
        mock_spec = MagicMock()
        mock_spec.description.input[0].type.imageType.width = 512
        mock_ct.models.MLModel.return_value.get_spec.return_value = mock_spec

        with patch.dict(sys.modules, {'coremltools': mock_ct}):
            result = _coreml_input_size('/fake/model.mlpackage')

        assert result == 512
        mock_ct.models.MLModel.assert_called_once_with('/fake/model.mlpackage')

    def test_returns_int(self):
        from scoutbot.tile_batched.main import _coreml_input_size

        mock_ct = MagicMock()
        mock_spec = MagicMock()
        mock_spec.description.input[0].type.imageType.width = '640'  # str input
        mock_ct.models.MLModel.return_value.get_spec.return_value = mock_spec

        with patch.dict(sys.modules, {'coremltools': mock_ct}):
            result = _coreml_input_size('/fake/model.mlpackage')

        assert isinstance(result, int)
        assert result == 640


class TestYolov8DetectionModelInit:
    """Tests for Yolov8DetectionModel.__init__() — covers lines 23-25."""

    def test_default_batch_size_and_coreml_flag(self):
        from scoutbot.tile_batched.main import DETECTOR_BATCH_SIZE, Yolov8DetectionModel

        with patch.object(Yolov8DetectionModel, 'load_model'):
            model = Yolov8DetectionModel(
                model_path='/fake/model.pt',
                confidence_threshold=0.45,
                device='cpu',
            )

        assert model.batch_size == DETECTOR_BATCH_SIZE
        assert model._coreml is False

    def test_custom_batch_size(self):
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        with patch.object(Yolov8DetectionModel, 'load_model'):
            model = Yolov8DetectionModel(
                model_path='/fake/model.pt',
                confidence_threshold=0.45,
                device='cpu',
                batch_size=32,
            )

        assert model.batch_size == 32


class TestPerformInferenceModelNone:
    """Covers line 59: raise ValueError when model is None."""

    def test_raises_when_model_not_loaded(self):
        import numpy as np

        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance.model = None

        with pytest.raises(ValueError, match='not loaded'):
            Yolov8DetectionModel.perform_inference(
                instance, [np.zeros((512, 512, 3), dtype=np.uint8)]
            )


class TestPerformInferenceNdarrayInput:
    """Covers line 68: the ndarray-to-list wrapper (`images = [images]`)."""

    def test_bare_ndarray_is_wrapped_in_list(self):
        """SAHI passes a single np.ndarray; the wrapper must convert it to [array]."""
        import numpy as np
        import torch

        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance._coreml = False
        instance.batch_size = 128
        instance.device = 'cpu'
        instance.confidence_threshold = 0.45
        instance.model = MagicMock()

        fake_boxes = MagicMock()
        fake_boxes.data = torch.zeros((0, 6))
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        instance.model.predict.return_value = [fake_result]

        # Pass a bare ndarray (not a list) — triggers line 68
        Yolov8DetectionModel.perform_inference(
            instance, np.zeros((512, 512, 3), dtype=np.uint8)
        )

        # model.predict should have been called with a 1-element source list
        call_kwargs = instance.model.predict.call_args
        source = call_kwargs.kwargs.get('source')
        if source is None:
            source = call_kwargs.args[0]
        assert isinstance(source, list)
        assert len(source) == 1


class TestLoadModelStoresComputeUnitsActual:
    """Tests that load_model() calls _resolve_compute_units_actual and stores the result."""

    def test_load_model_stores_compute_units_actual_for_mlpackage(self, ultralytics_stub):
        """load_model() for a .mlpackage must call _resolve_compute_units_actual and
        store the result in self._compute_units_actual.
        """
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance.model_path = '/fake/model.mlpackage'

        with (
            patch.object(sys, 'platform', 'darwin'),
            patch('scoutbot.tile_batched.main._coreml_input_size', return_value=512),
            patch('ultralytics.YOLO'),
            patch(
                'scoutbot.tile_batched.main._resolve_compute_units_actual',
                return_value='cpu_and_gpu',
                create=True,
            ),
        ):
            Yolov8DetectionModel.load_model(instance)

        assert instance._compute_units_actual == 'cpu_and_gpu'


class TestPerformInferenceComputeUnitsActual:
    """Tests that perform_inference() emits compute_units_actual in its event."""

    def test_perform_inference_coreml_emits_compute_units_actual(self, ultralytics_stub):
        """perform_inference() for a CoreML model must include compute_units_actual
        in the emitted observability event.
        """
        import numpy as np
        import torch

        from scoutbot._observability import list_observer
        from scoutbot.tile_batched.main import Yolov8DetectionModel

        instance = MagicMock(spec=Yolov8DetectionModel)
        instance._coreml = True
        instance._coreml_imgsz = 512
        instance.batch_size = 128
        instance.device = 'cpu'
        instance.confidence_threshold = 0.45
        instance.model = MagicMock()
        # Set the attribute directly; the real code will read it and include it in the event.
        instance._compute_units_actual = 'cpu_and_gpu'

        events = []
        observer = list_observer(events)
        instance.observer = observer

        fake_boxes = MagicMock()
        fake_boxes.data = torch.zeros((0, 6))
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        instance.model.predict.return_value = [fake_result]

        Yolov8DetectionModel.perform_inference(
            instance, [np.zeros((512, 512, 3), dtype=np.uint8)]
        )

        infer_event = next(e for e in events if e['name'] == 'perform_inference')
        assert infer_event['compute_units_actual'] == 'cpu_and_gpu'
