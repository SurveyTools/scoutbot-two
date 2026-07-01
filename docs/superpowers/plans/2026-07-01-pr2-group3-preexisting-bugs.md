# PR #2 Group 3 — Pre-existing Bug Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four pre-existing bugs exposed or adjacent to the CoreML backend added in PR #2.

**Architecture:** All four fixes are surgical changes to `scoutbot/tile_batched/main.py` (Tasks 1–3) and `scoutbot/__init__.py` (Task 4). Each has its own offline unit test using mocks — no model downloads, no network. Tasks 1–3 are independent of each other. Task 4 is independent of 1–3 but touches a different file.

**Tech Stack:** Python, pytest, `unittest.mock` (`patch`, `MagicMock`), `sys` stdlib.

## Global Constraints

- Python ≥ 3.8, < 3.13
- Formatter: Brunette. Linting: flake8. Import order: isort. Run `uv run pre-commit run --all-files` before every commit (or `uv run flake8 <files>` if pre-commit env is broken)
- New tests must be offline: no network, no reads/writes to `~/.cache/scoutbot/`
- Test runner: `uv run pytest` from repo root
- Do NOT run `git add` or `git commit` — user handles all commits

---

## File Map

| Action | Path | What changes |
|--------|------|-------------|
| Modify | `scoutbot/tile_batched/main.py:1` | Tasks 1–3 — add `import sys`; add `import torch` |
| Modify | `scoutbot/tile_batched/main.py:26-44` | Task 1 — `sys.platform` guard before `_coreml_input_size()` |
| Modify | `scoutbot/tile_batched/main.py:44` | Task 2 — replace `super().load_model()` with explicit `task='detect'` |
| Modify | `scoutbot/tile_batched/main.py:62-78` | Task 2+3 — `conf=` in predict_kwargs; `result.boxes is not None` guard |
| Modify | `tests/test_tile_batched.py` | Tasks 1–3 — create new test file |
| Modify | `scoutbot/__init__.py:50,66,163,250,403` | Task 4 — change five `config='v3'` defaults to `config=None` + `loc.DEFAULT_CONFIG` |
| Modify | `tests/test_batch.py` | Task 4 — new `TestApiEnvVarConfig` class |

---

## Task 1: Guard CoreML load against non-macOS platforms

**Files:**
- Modify: `scoutbot/tile_batched/main.py`
- Create: `tests/test_tile_batched.py`

**Background:** `load_model()` detects a `.mlpackage` path and calls `_coreml_input_size()`, which does `import coremltools as ct`. `coremltools` is only installed on macOS (the `[coreml]` extra has `sys_platform == 'darwin'`). On Linux CI, this silently raises `ModuleNotFoundError` instead of a clear platform error. Fix: check `sys.platform` before calling `_coreml_input_size()`.

- [x] **Step 1: Read `scoutbot/tile_batched/main.py`**

Read the full file to confirm the exact current imports and `load_model()` body.

- [x] **Step 2: Create `tests/test_tile_batched.py` with a failing test**

```python
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
```

- [x] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_tile_batched.py::TestLoadModelPlatformGuard -v
```

Expected: both **FAIL** — `test_mlpackage_raises_on_non_darwin` fails because no `RuntimeError` is raised (currently raises `ModuleNotFoundError` or nothing); `test_mlpackage_allowed_on_darwin` likely passes already (it may pass because the mock covers `_coreml_input_size`). Focus on the first test failing.

- [x] **Step 4: Add `import sys` to `main.py` top-level imports**

Current imports at top of `scoutbot/tile_batched/main.py`:
```python
import os
from typing import Optional
```

Change to:
```python
import os
import sys
from typing import Optional
```

- [x] **Step 5: Add the platform guard to `load_model()`**

Find this block in `load_model()`:
```python
        if str(self.model_path).endswith('.mlpackage'):
            from ultralytics import YOLO

            self._coreml = True
            # The export has a FIXED input size ...
            self._coreml_imgsz = _coreml_input_size(self.model_path)
            self.set_model(YOLO(self.model_path, task='detect'))
            return
```

Replace with:
```python
        if str(self.model_path).endswith('.mlpackage'):
            if sys.platform != 'darwin':
                raise RuntimeError(
                    f'Core ML .mlpackage models require macOS (darwin); '
                    f'current platform is {sys.platform!r}. '
                    f'Use a .pt config instead (e.g. config="v3").'
                )
            from ultralytics import YOLO

            self._coreml = True
            # The export has a FIXED input size ...
            self._coreml_imgsz = _coreml_input_size(self.model_path)
            self.set_model(YOLO(self.model_path, task='detect'))
            return
```

- [x] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_tile_batched.py::TestLoadModelPlatformGuard -v
```

Expected: both **PASS**.

- [x] **Step 7: Run flake8 on changed files**

```bash
uv run flake8 scoutbot/tile_batched/main.py tests/test_tile_batched.py
```

Expected: no output (clean).

---

## Task 2: Pass `task='detect'` for PT models; guard `result.boxes=None`

**Files:**
- Modify: `scoutbot/tile_batched/main.py`
- Modify: `tests/test_tile_batched.py`

**Background (two related bugs):**
1. `super().load_model()` (the PT path) calls `YOLO(path)` without `task='detect'`. A `v3-cls.pt` model may be mis-identified by Ultralytics as a classifier from its filename, returning `result.boxes=None` and crashing `perform_inference`. The CoreML path already passes `task='detect'`; the PT path should too.
2. As belt-and-suspenders, the `_original_predictions` comprehension should also guard against `result.boxes is None` so that a None result produces an empty tensor rather than an `AttributeError`.

These two changes address the same bug from two angles: fix the root cause in `load_model` and make `perform_inference` robust to the edge case.

- [x] **Step 1: Write failing tests**

Append these two classes to `tests/test_tile_batched.py`:

```python
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

        result_with_none_boxes = MagicMock()
        result_with_none_boxes.boxes = None
        instance.model.predict.return_value = [result_with_none_boxes]

        Yolov8DetectionModel.perform_inference(instance, [np.zeros((512, 512, 3), dtype=np.uint8)])

        assert instance._original_predictions is not None
        assert len(instance._original_predictions) == 1
```

- [x] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tile_batched.py::TestLoadModelPtTaskDetect tests/test_tile_batched.py::TestPerformInferenceNoneBoxes -v
```

Expected:
- `test_pt_load_model_passes_task_detect` — **FAILS**: `MockYOLO` is called without `task=` kwarg (current code calls `super().load_model()`)
- `test_none_boxes_produces_empty_prediction` — **FAILS**: `AttributeError: 'NoneType' object has no attribute 'data'`

- [x] **Step 3: Add `import torch` to `main.py` top-level imports**

`torch` is a direct transitive dependency (via ultralytics) — add it explicitly:

```python
import os
import sys
from typing import Optional

import numpy as np
import torch
from sahi.models.ultralytics import UltralyticsDetectionModel as _Base
```

- [x] **Step 4: Replace `super().load_model()` for the PT path**

Find the end of `load_model()`:
```python
        super().load_model()
```

Replace with:
```python
        from ultralytics import YOLO

        model = YOLO(self.model_path, task='detect')
        model.to(self.device)
        self.set_model(model)
```

- [x] **Step 5: Add `result.boxes is not None` guard in `perform_inference()`**

Find:
```python
        self._original_predictions = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold]
            for result in all_preds
        ]
```

Replace with:
```python
        self._original_predictions = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold]
            if result.boxes is not None
            else torch.zeros((0, 6))
            for result in all_preds
        ]
```

- [x] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_tile_batched.py::TestLoadModelPtTaskDetect tests/test_tile_batched.py::TestPerformInferenceNoneBoxes -v
```

Expected: both **PASS**.

- [x] **Step 7: Run all tile_batched tests and flake8**

```bash
uv run pytest tests/test_tile_batched.py -v
uv run flake8 scoutbot/tile_batched/main.py tests/test_tile_batched.py
```

Expected: all pass, flake8 clean.

---

## Task 3: Pass `conf=` to `model.predict()` to avoid below-threshold wasted inference

**Files:**
- Modify: `scoutbot/tile_batched/main.py`
- Modify: `tests/test_tile_batched.py`

**Background:** `perform_inference()` builds `predict_kwargs` but never passes `conf=self.confidence_threshold` to `model.predict()`. Ultralytics defaults to `conf=0.25` internally, so predictions in [0.25, 0.45) are computed on the ANE/GPU and then discarded by the comprehension at line 75. Passing `conf=` explicitly prevents this wasted work on every tile.

- [x] **Step 1: Write failing test**

Append to `tests/test_tile_batched.py`:

```python
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

        fake_boxes = MagicMock()
        fake_boxes.data = torch.zeros((0, 6))
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        instance.model.predict.return_value = [fake_result]

        Yolov8DetectionModel.perform_inference(instance, [np.zeros((512, 512, 3), dtype=np.uint8)])

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

        fake_boxes = MagicMock()
        fake_boxes.data = torch.zeros((0, 6))
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        instance.model.predict.return_value = [fake_result]

        Yolov8DetectionModel.perform_inference(instance, [np.zeros((512, 512, 3), dtype=np.uint8)])

        call_kwargs = instance.model.predict.call_args.kwargs
        assert call_kwargs.get('conf') == 0.45
        assert call_kwargs.get('imgsz') == 512
```

- [x] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tile_batched.py::TestPerformInferenceConfKwarg -v
```

Expected: both **FAIL** — `conf` key not present in `call_kwargs`.

- [x] **Step 3: Apply the fix**

In `scoutbot/tile_batched/main.py`, find:
```python
        predict_kwargs = {}
        if self._coreml:
            predict_kwargs['imgsz'] = self._coreml_imgsz
```

Replace with:
```python
        predict_kwargs = {'conf': self.confidence_threshold}
        if self._coreml:
            predict_kwargs['imgsz'] = self._coreml_imgsz
```

- [x] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_tile_batched.py::TestPerformInferenceConfKwarg -v
```

Expected: both **PASS**.

- [x] **Step 5: Run full tile_batched test file and flake8**

```bash
uv run pytest tests/test_tile_batched.py -v
uv run flake8 scoutbot/tile_batched/main.py tests/test_tile_batched.py
```

Expected: all pass, flake8 clean.

---

## Task 4: Wire `LOC_CONFIG`/`CONFIG` env vars to public API

**Files:**
- Modify: `scoutbot/__init__.py` — five functions
- Modify: `tests/test_batch.py` — new test class

**Background:** `fetch()`, `pipeline()`, `batch()`, `batch_resumable()`, and `get_classes()` all have `config='v3'` as their default, so `loc.DEFAULT_CONFIG` (which reads the `LOC_CONFIG` / `CONFIG` env vars) is never consulted when these functions are called without an explicit config. Setting `LOC_CONFIG=v3-coreml` and calling `scoutbot.pipeline(image)` silently uses PyTorch. Fix: change the default to `config=None` and replace each `if config is None: config = 'v3'` guard with `if config is None: config = loc.DEFAULT_CONFIG`. Behavior is unchanged when no env var is set (`DEFAULT_CONFIG` falls back to `'v3'`).

- [x] **Step 1: Write failing tests**

Append a new class to `tests/test_batch.py`:

```python
class TestApiEnvVarConfig:
    """LOC_CONFIG/CONFIG env vars must be respected by public API functions."""

    def test_fetch_uses_loc_default_config_when_no_config_given(self):
        """scoutbot.fetch() with no args must delegate to loc.DEFAULT_CONFIG."""
        from unittest.mock import patch
        import scoutbot
        from scoutbot import loc

        with patch.object(loc, 'DEFAULT_CONFIG', 'v3-cls'), \
             patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch:
            scoutbot.fetch()

        mock_fetch.assert_called_with(config='v3-cls')

    def test_get_classes_uses_loc_default_config_when_no_config_given(self):
        """scoutbot.get_classes() with no args must delegate to loc.DEFAULT_CONFIG."""
        from unittest.mock import patch, MagicMock
        import scoutbot
        from scoutbot import loc

        mock_model = MagicMock()
        mock_model.category_names = ['animal']

        with patch.object(loc, 'DEFAULT_CONFIG', 'v3-cls'), \
             patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel', return_value=mock_model):
            result = scoutbot.get_classes()

        assert result == ['animal']
```

- [x] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_batch.py::TestApiEnvVarConfig -v
```

Expected: both **FAIL** — `mock_fetch` is called with `config='v3'` (the hardcoded default), not `'v3-cls'`.

- [x] **Step 3: Apply the fix to `scoutbot/__init__.py`**

Five changes — each is the same two-part pattern:
1. Change `config='v3'` in the function signature to `config=None`
2. Change `if config is None: config = 'v3'` guard to `if config is None: config = loc.DEFAULT_CONFIG`

**`fetch()` (line 50):**
```python
# Before:
def fetch(config='v3'):
    ...
    if config is None:
        config = 'v3'
    return loc.fetch(config=config)
```
```python
# After:
def fetch(config=None):
    ...
    if config is None:
        config = loc.DEFAULT_CONFIG
    return loc.fetch(config=config)
```

**`pipeline()` (line 66):**
```python
# Before:
def pipeline(
        filepath,
        config='v3',
        ...
):
    ...
    if config is None:
        config = 'v3'
```
```python
# After:
def pipeline(
        filepath,
        config=None,
        ...
):
    ...
    if config is None:
        config = loc.DEFAULT_CONFIG
```

**`batch()` (line 163):**
```python
# Before:
def batch(
        filepaths,
        config='v3',
        ...
):
    ...
    if config is None:
        config = 'v3'
```
```python
# After:
def batch(
        filepaths,
        config=None,
        ...
):
    ...
    if config is None:
        config = loc.DEFAULT_CONFIG
```

**`batch_resumable()` (line 250):**
```python
# Before:
def batch_resumable(
        filepaths,
        output_path,
        config='v3',
        ...
):
    ...
    if config is None:
        config = 'v3'
```
```python
# After:
def batch_resumable(
        filepaths,
        output_path,
        config=None,
        ...
):
    ...
    if config is None:
        config = loc.DEFAULT_CONFIG
```

**`get_classes()` (line 403):**
```python
# Before:
def get_classes(config='v3'):
    ...
    if config is None:
        config = 'v3'
```
```python
# After:
def get_classes(config=None):
    ...
    if config is None:
        config = loc.DEFAULT_CONFIG
```

- [x] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_batch.py::TestApiEnvVarConfig -v
```

Expected: both **PASS**.

- [x] **Step 5: Run the full test suite**

```bash
uv run pytest -v 2>&1 | tail -20
```

Expected: all existing tests still pass. The `config=None` default is backward-compatible — `loc.DEFAULT_CONFIG` falls through to `'v3'` when no env var is set.

- [x] **Step 6: Run flake8 on changed files**

```bash
uv run flake8 scoutbot/__init__.py tests/test_batch.py
```

Expected: no output (clean).

---

## Self-Review

**Spec coverage:**
- 3a (non-macOS CoreML crash) ✓ Task 1
- 3b (env vars ignored by public API) ✓ Task 4
- 3c (conf= not passed to predict) ✓ Task 3
- 3d (result.boxes=None crash) ✓ Task 2 (root cause: task='detect'; guard: None check)

**Placeholder scan:** No TBD, no "similar to Task N", all code is complete and runnable.

**Type/name consistency:**
- `Yolov8DetectionModel.load_model`, `Yolov8DetectionModel.perform_inference` — match current `main.py`
- `loc.DEFAULT_CONFIG` — matches `scoutbot/loc/__init__.py:17`
- `torch.zeros((0, 6))` — correct shape for SAHI's prediction processing (N rows × 6 cols: x1y1x2y2 + conf + cls)
- `predict_kwargs = {'conf': self.confidence_threshold}` — replaces `predict_kwargs = {}` in Task 3

**Ordering notes:**
- Tasks 1, 2, 3 all modify `scoutbot/tile_batched/main.py` and `tests/test_tile_batched.py`. Commit each before starting the next to avoid intra-session conflicts.
- Task 4 touches only `scoutbot/__init__.py` and `tests/test_batch.py` — independent of 1–3.
