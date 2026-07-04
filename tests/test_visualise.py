import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import visualise

SAMPLE_RESULT = {
    'wic': 0.9,
    'loc': [
        {'l': 'elephant', 'c': 0.85, 'x': 10, 'y': 10, 'w': 50, 'h': 80},
        {'l': 'giraffe', 'c': 0.45, 'x': 100, 'y': 100, 'w': 30, 'h': 60},
    ],
}


def _write_json(tmp_path, data):
    p = tmp_path / 'detections.json'
    p.write_text(json.dumps(data))
    return p


def _write_image(tmp_path, h=200, w=400):
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    p = tmp_path / 'img.jpg'
    cv2.imwrite(str(p), img)
    return p


class TestLoadDetections:
    def test_single_key_returns_result(self, tmp_path):
        p = _write_json(tmp_path, {'/img.jpg': SAMPLE_RESULT})
        result = visualise.load_detections(p)
        assert result['wic'] == 0.9
        assert len(result['loc']) == 2

    def test_empty_json_raises(self, tmp_path):
        p = _write_json(tmp_path, {})
        with pytest.raises(ValueError, match='Empty JSON'):
            visualise.load_detections(p)

    def test_multi_key_without_image_key_raises(self, tmp_path):
        p = _write_json(tmp_path, {'/a.jpg': SAMPLE_RESULT, '/b.jpg': SAMPLE_RESULT})
        with pytest.raises(ValueError, match='2 images'):
            visualise.load_detections(p)


class TestDrawLabel:
    def test_label_above_when_room(self):
        img = np.ones((200, 400, 3), dtype=np.uint8) * 200
        visualise._draw_label(img, 'elephant: 0.85', x=10, y=100, box_h=50)
        # Label drawn above y=100 — some pixels in top half changed
        assert not np.all(img[:100, :] == 200)

    def test_label_below_when_near_top_edge(self):
        img = np.ones((200, 400, 3), dtype=np.uint8) * 200
        visualise._draw_label(img, 'elephant: 0.85', x=10, y=2, box_h=50)
        # Not enough room above y=2 — pixels above row 2 must be untouched
        assert np.all(img[:2, :] == 200)
        # Label must have appeared below the box (rows 52-70)
        assert not np.all(img[52:70, 10:200] == 200)


class TestVisualizeDetections:
    def test_saves_output_file(self, tmp_path):
        img_path = _write_image(tmp_path)
        json_path = _write_json(tmp_path, {str(img_path): SAMPLE_RESULT})
        out_path = tmp_path / 'out.jpg'
        with patch.object(visualise, '_open_for_display'):
            visualise.visualize_detections(
                img_path, json_path, output_path=out_path, display=False
            )
        assert out_path.exists()

    def test_threshold_filters_detections(self, tmp_path, capsys):
        img_path = _write_image(tmp_path)
        json_path = _write_json(tmp_path, {str(img_path): SAMPLE_RESULT})
        out_path = tmp_path / 'out.jpg'
        with patch.object(visualise, '_open_for_display'):
            visualise.visualize_detections(
                img_path, json_path, threshold=0.5, output_path=out_path, display=False
            )
        assert '1 above threshold' in capsys.readouterr().out

    def test_display_calls_open(self, tmp_path):
        img_path = _write_image(tmp_path)
        json_path = _write_json(tmp_path, {str(img_path): SAMPLE_RESULT})
        out_path = tmp_path / 'out.jpg'
        with patch.object(visualise, '_open_for_display') as mock_open:
            visualise.visualize_detections(
                img_path, json_path, output_path=out_path, display=True
            )
        mock_open.assert_called_once_with(out_path)

    def test_no_display_skips_open(self, tmp_path):
        img_path = _write_image(tmp_path)
        json_path = _write_json(tmp_path, {str(img_path): SAMPLE_RESULT})
        out_path = tmp_path / 'out.jpg'
        with patch.object(visualise, '_open_for_display') as mock_open:
            visualise.visualize_detections(
                img_path, json_path, output_path=out_path, display=False
            )
        mock_open.assert_not_called()
