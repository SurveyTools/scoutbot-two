import os
import sys
from typing import Optional

import numpy as np
import torch
from sahi.models.ultralytics import UltralyticsDetectionModel as _Base

DETECTOR_BATCH_SIZE = int(os.getenv('DETECTOR_BATCH_SIZE', 128))


def _coreml_input_size(model_path) -> int:
    """Read the fixed square input size (pixels) baked into a Core ML .mlpackage."""
    import coremltools as ct

    spec = ct.models.MLModel(str(model_path)).get_spec()
    image_type = spec.description.input[0].type.imageType
    return int(image_type.width)


class Yolov8DetectionModel(_Base):

    def __init__(self, *args, batch_size: Optional[int] = None, **kwargs):
        self.batch_size = batch_size or DETECTOR_BATCH_SIZE
        self._coreml = False
        super().__init__(*args, **kwargs)

    def load_model(self):
        # Core ML (.mlpackage) bundles need special handling that the base
        # UltralyticsDetectionModel doesn't provide:
        #   - task='detect': without it, a '-cls' filename is mis-guessed as a
        #     classifier and result.boxes comes back None.
        #   - no model.to(device): Ultralytics routes Core ML through its own
        #     runtime, so a torch device (esp. the cuda:0 default) is invalid here.
        if str(self.model_path).endswith('.mlpackage'):
            if sys.platform != 'darwin':
                raise RuntimeError(
                    f'Core ML .mlpackage models require macOS (darwin); '
                    f'current platform is {sys.platform!r}. '
                    f'Use a .pt config instead (e.g. config="v3").'
                )
            from ultralytics import YOLO

            self._coreml = True
            # The export has a FIXED input size (static shapes are an ANE
            # requirement). Ultralytics' predict() otherwise defaults to imgsz=640
            # and Core ML rejects the mismatch, so read the real size off the model.
            self._coreml_imgsz = _coreml_input_size(self.model_path)
            self.set_model(YOLO(self.model_path, task='detect'))
            return

        from ultralytics import YOLO

        model = YOLO(self.model_path, task='detect')
        model.to(self.device)
        self.set_model(model)

    def perform_inference(self, images, batch_size: Optional[int] = None):
        if self.model is None:
            raise ValueError('Model is not loaded, call .load_model() first')

        # Core ML inference is dispatched by Ultralytics regardless of the nominal
        # torch device; force 'cpu' so the cuda:0 default is never passed through.
        device = 'cpu' if self._coreml else self.device

        # SAHI calls with a single numpy array (one slice at a time);
        # external callers may pass a list for true batch inference.
        if isinstance(images, np.ndarray):
            images = [images]

        batch_size = batch_size or self.batch_size
        all_preds = []

        predict_kwargs = {'conf': self.confidence_threshold}
        if self._coreml:
            predict_kwargs['imgsz'] = self._coreml_imgsz

        for i in range(0, len(images), batch_size):
            preds = self.model.predict(
                source=images[i : i + batch_size],
                verbose=False,
                device=device,
                **predict_kwargs,
            )
            all_preds.extend(preds)

        self._original_predictions = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold]
            if result.boxes is not None
            else torch.zeros((0, 6))
            for result in all_preds
        ]
