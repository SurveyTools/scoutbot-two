import os
from typing import List, Optional

import numpy as np
from sahi.models.ultralytics import UltralyticsDetectionModel as _Base

DETECTOR_BATCH_SIZE = int(os.getenv('DETECTOR_BATCH_SIZE', 128))


class Yolov8DetectionModel(_Base):

    def __init__(self, *args, batch_size: Optional[int] = None, **kwargs):
        self.batch_size = batch_size or DETECTOR_BATCH_SIZE
        super().__init__(*args, **kwargs)

    def perform_inference(self, images, batch_size: Optional[int] = None):
        if self.model is None:
            raise ValueError('Model is not loaded, call .load_model() first')

        # SAHI calls with a single numpy array (one slice at a time);
        # external callers may pass a list for true batch inference.
        if isinstance(images, np.ndarray):
            images = [images]

        batch_size = batch_size or self.batch_size
        all_preds = []

        for i in range(0, len(images), batch_size):
            preds = self.model.predict(
                source=images[i : i + batch_size],
                verbose=False,
                device=self.device,
            )
            all_preds.extend(preds)

        self._original_predictions = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold]
            for result in all_preds
        ]
