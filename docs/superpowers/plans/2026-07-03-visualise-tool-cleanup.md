# Visualise Tool Cleanup and Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `visualise.py` from the repo root to `tools/visualise.py`, fix every
identified correctness and design issue, and wire the tool into all quality gates.

**Architecture:** The script is a standalone developer utility — not part of the
`scoutbot` package. It lives in a new `tools/` directory alongside future one-off
scripts. Quality tools (ruff, mypy, bandit) are extended to cover `tools/`. The
display path is rewritten to use `webbrowser` instead of `cv2.imshow`, removing the
GUI dependency that conflicts with the headless OpenCV build the project ships. Two new
flags (`--no-display`, `--image-key`) cover the save-only and multi-image use cases.

**Tech Stack:** Python 3.10+, cv2 (headless), webbrowser (stdlib), argparse, pytest,
uv, GitHub Actions.

## Global Constraints

- Use `uv run` for all Python/tool invocations; never call `python` or `pytest` directly
- Ruff quote style: single quotes; line length: 90
- No comments unless the WHY is non-obvious
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Pause after each task for review before moving on
- `x`, `y` in ScoutBot detection dicts are the **top-left corner** of the bounding box
  (confirmed in `docs/usage.rst`); do not change the box-drawing coordinate math

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Create | `tools/visualise.py` | Cleaned-up, fully rewritten version of the script |
| Delete | `visualise.py` | Old root-level script removed |
| Modify | `pyproject.toml` | Add `tools` to `pythonpath` |
| Modify | `.github/workflows/quality.yml` | Task 1: add `tools/` to bandit; Task 2: add `tools/visualise.py` to mypy |
| Create | `tests/test_visualise.py` | Full test suite for the new script |
| Modify | `docs/usage.rst` | Add a "Visualisation" section pointing to the tool |
| Modify | `docs/development.rst` | Note `tools/` in quality-check coverage |
| Modify | `CONTRIBUTING.md` | Update bandit and mypy commands to include `tools/` |
| Modify | `CHANGELOG.rst` | Record the cleanup and move |

---

## Task 1: Move to `tools/` and wire quality gates

**Files:**
- Create: `tools/visualise.py` (copy of current `visualise.py`, unchanged code)
- Delete: `visualise.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/quality.yml`

**Background:** `tools/` is a new sibling directory for developer utilities that are
not part of the `scoutbot` package and not benchmarks. The copy in Task 1 is
intentionally unchanged — the rewrite happens in Task 2. Getting the file in its new
home first lets quality tools catch issues before code changes are made.

- [ ] **Step 1: Create the `tools/` directory and copy the script**

```bash
mkdir -p tools
cp visualise.py tools/visualise.py
```

- [ ] **Step 2: Add `tools` to pytest pythonpath in `pyproject.toml`**

Change:
```toml
pythonpath = ["benchmarks", "tests"]
```
To:
```toml
pythonpath = ["benchmarks", "tests", "tools"]
```

- [ ] **Step 3: Add `tools/` to bandit in `quality.yml`**

Change only the Bandit step (mypy is extended in Task 2, after the typed rewrite):
```yaml
      - name: Bandit security scan
        run: uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
```

- [ ] **Step 4: Delete the root-level script**

```bash
rm visualise.py
```

- [ ] **Step 5: Run quality checks and confirm the gate is green**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
uv audit --locked --preview-features audit-command
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
uv run pytest -m "not slow" -q
```

Mypy runs only against `scoutbot` here — `tools/visualise.py` is added to mypy in Task 2
after the typed rewrite so the gate stays fully green at this checkpoint.

**Exit criteria:**
- `tools/visualise.py` exists; `visualise.py` at repo root is gone
- All quality checks pass with no errors or recorded exceptions
- pytest fast suite still reports 103 passed

---

## Task 2: Rewrite — cleanup, correctness fixes, and tests

**Files:**
- Modify: `tools/visualise.py`
- Create: `tests/test_visualise.py`

**Interfaces:**
- Produces: `load_detections(json_path, image_key=None) -> dict`
- Produces: `_draw_label(img, text, x, y, box_h) -> None`
- Produces: `_box_color(label) -> tuple[int, int, int]`
- Produces: `visualize_detections(image_path, json_path, threshold, output_path, display, image_key) -> None`
- Produces: `_open_for_display(path) -> None`
- Produces: `main() -> None`

**Background:** This task is a full rewrite of the script body. Issues addressed:

1. Remove `numpy` import — `np.around(x)` becomes `round(x)` (JSON coords are
   already Python floats; `numpy` is not needed)
2. Fix shebang from `#!/usr/bin/env python` to `#!/usr/bin/env python3`
3. Remove all comments that restate what the code does; keep only non-obvious WHY comments
4. Replace single-colour constant with a palette-based `_box_color(label)` helper so
   multi-class detections are visually distinguishable
5. Extract `_draw_label` helper; fix the label clipping bug — when a detection box is
   near the top edge, the label background renders with a negative y-coordinate and
   is invisible. Fix: place the label *below* the box when `y < th + baseline + pad`
6. Add type annotations throughout
7. Replace the bare `except Exception` in `main()` with `except (ValueError, OSError)`
8. Extract `_open_for_display(path)` as a named helper (needed for mocking in tests
   and for the display fix in Task 3 — leave it as a stub returning `None` in this
   task; Task 3 fills in the real implementation)

The **display block** (`cv2.namedWindow`, `cv2.imshow`, `cv2.waitKey`,
`cv2.destroyAllWindows`) is **removed in this task**. Replace it with a call to
`_open_for_display(output_path)` so Task 3 can wire it up. Add `display: bool = True`
parameter to `visualize_detections` so `--no-display` (Task 3) can suppress it.

**Multi-image JSON** is handled in Task 3. In this task, `load_detections` still reads
the first key — but raise `ValueError` with a clear message if `len(data) > 1` and no
`image_key` was provided.

- [ ] **Step 1: Write failing tests for `load_detections`**

Create `tests/test_visualise.py`:

```python
import json
import cv2
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_visualise.py -v
```

Expected: `ImportError` or `AttributeError` — `load_detections` signature doesn't match yet.

- [ ] **Step 3: Write failing tests for `_draw_label` label clipping**

Add to `tests/test_visualise.py`:

```python
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
```

- [ ] **Step 4: Write failing tests for `visualize_detections`**

Add to `tests/test_visualise.py`:

```python
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
```

- [ ] **Step 5: Run tests to confirm they all fail**

```bash
uv run pytest tests/test_visualise.py -v
```

Expected: failures across all test classes.

- [ ] **Step 6: Rewrite `tools/visualise.py`**

Replace the entire file with:

```python
#!/usr/bin/env python3
"""
Standalone visualisation tool for ScoutBot detection results.

Usage:
    uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json [OPTIONS]

Renders bounding boxes and confidence scores onto the source image.
Supports single-image and multi-image JSON output from the ScoutBot CLI.
"""

import argparse
import atexit
import json
import sys
import tempfile
import zlib
from pathlib import Path

import cv2

# Fixed BGR palette for per-class box colours. zlib.crc32 gives a stable hash
# across runs (unlike hash(), which is salted per-process since Python 3.3).
_PALETTE = [
    (255, 100, 0),
    (0, 200, 60),
    (0, 60, 255),
    (0, 200, 255),
    (200, 0, 200),
    (0, 200, 200),
]


def _box_color(label: str) -> tuple[int, int, int]:
    return _PALETTE[zlib.crc32(label.encode()) % len(_PALETTE)]


def _draw_label(img, text: str, x: int, y: int, box_h: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4

    if y < th + baseline + pad:
        bg_y1, bg_y2 = y + box_h, y + box_h + th + baseline + pad
        text_y = y + box_h + th + 2
    else:
        bg_y1, bg_y2 = y - th - baseline - pad, y
        text_y = y - baseline - 2

    cv2.rectangle(img, (x, bg_y1), (x + tw + pad, bg_y2), (255, 255, 255), -1)
    cv2.putText(img, text, (x + 2, text_y), font, scale, (0, 0, 0), thickness)


def _open_for_display(path: Path) -> None:
    pass  # wired up in Task 3


def load_detections(json_path: Path, image_key: str | None = None) -> dict:
    data = json.loads(json_path.read_text())
    if not data:
        raise ValueError(f'Empty JSON file: {json_path}')

    if len(data) > 1 and image_key is None:
        keys = '\n  '.join(data)
        raise ValueError(
            f'JSON contains {len(data)} images. Specify one with --image-key:\n  {keys}'
        )

    if image_key is not None:
        if image_key not in data:
            available = ', '.join(repr(k) for k in data)
            raise ValueError(
                f'Image key {image_key!r} not found. Available: {available}'
            )
        return data[image_key]

    return next(iter(data.values()))


def visualize_detections(
    image_path: Path,
    json_path: Path,
    threshold: float = 0.0,
    output_path: Path | None = None,
    display: bool = True,
    image_key: str | None = None,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f'Could not load image: {image_path}')

    result = load_detections(json_path, image_key=image_key)
    detections = result.get('loc', [])
    wic_score = result.get('wic', 0.0)

    print(f'Image: {image_path}')
    print(f'WIC score: {wic_score:.4f}')

    visible = [d for d in detections if d['c'] >= threshold]
    print(f'Detections: {len(visible)} above threshold {threshold} (total {len(detections)})')

    for det in visible:
        label, conf = det['l'], det['c']
        x, y, w, h = round(det['x']), round(det['y']), round(det['w']), round(det['h'])
        cv2.rectangle(img, (x, y), (x + w, y + h), _box_color(label), 2)
        _draw_label(img, f'{label}: {conf:.3f}', x, y, h)

    if output_path is not None:
        cv2.imwrite(str(output_path), img)
        print(f'Saved: {output_path}')
        if display:
            _open_for_display(output_path)
    elif display:
        # atexit covers normal exit and unhandled exceptions but not SIGKILL or
        # interpreter crashes — temp files can survive those. Acceptable for a
        # developer utility; the OS reclaims /tmp on reboot regardless.
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp = Path(f.name)
        atexit.register(tmp.unlink, missing_ok=True)
        cv2.imwrite(str(tmp), img)
        _open_for_display(tmp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render ScoutBot detections as bounding boxes on the source image.'
    )
    parser.add_argument('image', help='Source image file')
    parser.add_argument('json', help='ScoutBot JSON output file')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.0,
        help='Minimum confidence to display (default: 0.0)',
    )
    parser.add_argument('-o', '--output', help='Save annotated image to this path')
    parser.add_argument(
        '-k', '--image-key',
        help='Image key to use when JSON contains multiple images',
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Save output without opening a viewer (requires --output)',
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    json_path = Path(args.json)

    if not image_path.exists():
        print(f'Error: image not found: {image_path}', file=sys.stderr)
        sys.exit(1)
    if not json_path.exists():
        print(f'Error: JSON not found: {json_path}', file=sys.stderr)
        sys.exit(1)
    if args.no_display and args.output is None:
        print('Error: --no-display requires --output', file=sys.stderr)
        sys.exit(1)

    try:
        visualize_detections(
            image_path,
            json_path,
            threshold=args.threshold,
            output_path=Path(args.output) if args.output else None,
            display=not args.no_display,
            image_key=args.image_key,
        )
    except (ValueError, OSError) as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
uv run pytest tests/test_visualise.py -v
```

Expected: all tests in `TestLoadDetections`, `TestDrawLabel`, and
`TestVisualizeDetections` pass.

- [ ] **Step 8: Add `tools/visualise.py` to mypy in `quality.yml`**

Now that the rewrite is complete and fully typed, extend the mypy step:
```yaml
      - name: Mypy
        run: uv run mypy scoutbot tools/visualise.py
```

- [ ] **Step 9: Run the full quality suite**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
uv run pytest -m "not slow" -q
```

Expected: all pass; 103 + new tests passed.

**Exit criteria:**
- All new tests pass
- Full quality suite passes
- `tools/visualise.py` has no `numpy` import, no `cv2.imshow`, correct label clipping

---

## Task 3: Wire `_open_for_display` and add CLI tests

**Files:**
- Modify: `tools/visualise.py` (fill in `_open_for_display`)
- Modify: `tests/test_visualise.py` (add CLI and display tests)

**Background:** Task 2 left `_open_for_display` as a no-op stub. This task implements
it using `webbrowser` (stdlib, cross-platform, no new dependencies, no subprocess, no
Bandit issues). Also adds CLI-level tests covering the new `--no-display` and
`--image-key` flags and the existing error-exit paths.

**Interfaces:**
- Consumes: `_open_for_display(path: Path) -> None` stub from Task 2
- `webbrowser.open(path.as_uri())` — `path.as_uri()` returns a `file:///` URI handed to
  the OS default handler. On macOS this is typically Preview; on Linux it is whatever
  `xdg-open` maps to; in a headless environment it may silently no-op. It is not
  guaranteed to open a native image viewer on every platform.

- [ ] **Step 1: Write failing CLI tests**

Add to `tests/test_visualise.py`:

```python
class TestCLI:
    def test_missing_image_exits_1(self, tmp_path):
        json_path = _write_json(tmp_path, {})
        with patch('sys.argv', ['visualise.py', '/nonexistent.jpg', str(json_path)]):
            with pytest.raises(SystemExit) as exc:
                visualise.main()
        assert exc.value.code == 1

    def test_missing_json_exits_1(self, tmp_path):
        img_path = _write_image(tmp_path)
        with patch('sys.argv', ['visualise.py', str(img_path), '/nonexistent.json']):
            with pytest.raises(SystemExit) as exc:
                visualise.main()
        assert exc.value.code == 1

    def test_no_display_without_output_exits_1(self, tmp_path):
        img_path = _write_image(tmp_path)
        json_path = _write_json(tmp_path, {str(img_path): SAMPLE_RESULT})
        with patch('sys.argv', [
            'visualise.py', str(img_path), str(json_path), '--no-display'
        ]):
            with pytest.raises(SystemExit) as exc:
                visualise.main()
        assert exc.value.code == 1

    def test_image_key_flag_selects_entry(self, tmp_path):
        img_path = _write_image(tmp_path)
        other = {'wic': 0.1, 'loc': []}
        json_path = _write_json(tmp_path, {
            str(img_path): SAMPLE_RESULT,
            '/other.jpg': other,
        })
        out_path = tmp_path / 'out.jpg'
        with patch.object(visualise, '_open_for_display'):
            with patch('sys.argv', [
                'visualise.py', str(img_path), str(json_path),
                '--image-key', str(img_path),
                '--output', str(out_path),
                '--no-display',
            ]):
                visualise.main()
        assert out_path.exists()

    def test_display_without_output_uses_temp_file(self, tmp_path):
        img_path = _write_image(tmp_path)
        json_path = _write_json(tmp_path, {str(img_path): SAMPLE_RESULT})
        with patch.object(visualise, '_open_for_display') as mock_open:
            with patch('sys.argv', ['visualise.py', str(img_path), str(json_path)]):
                visualise.main()
        mock_open.assert_called_once()
        called_path = mock_open.call_args[0][0]
        assert called_path.suffix == '.jpg'
```

- [ ] **Step 2: Run to confirm tests fail (stub returns None, open not called yet)**

```bash
uv run pytest tests/test_visualise.py::TestCLI -v
```

Expected: `test_display_without_output_uses_temp_file` fails because `_open_for_display`
is a no-op stub; the others may pass or fail depending on mock state.

- [ ] **Step 3: Implement `_open_for_display`**

In `tools/visualise.py`, replace:

```python
def _open_for_display(path: Path) -> None:
    pass  # wired up in Task 3
```

With:

```python
def _open_for_display(path: Path) -> None:
    import webbrowser
    # Hands a file:// URI to the OS default handler. On macOS this is typically
    # Preview; on Linux it delegates to xdg-open. Headless environments may no-op.
    webbrowser.open(path.as_uri())
```

- [ ] **Step 4: Run all tests to confirm they pass**

```bash
uv run pytest tests/test_visualise.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full quality suite**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
uv run pytest -m "not slow" -q
```

Expected: all pass.

**Exit criteria:**
- `_open_for_display` uses `webbrowser.open`; no `cv2.imshow` anywhere in the file
- All CLI tests pass, including `--no-display`, `--image-key`, and the temp-file display path
- Bandit reports no issues for `tools/visualise.py`

---

## Task 4: Docs and consistency pass

**Files:**
- Modify: `docs/usage.rst`
- Modify: `docs/development.rst`
- Modify: `CHANGELOG.rst`

**Background:** `docs/usage.rst` should tell users and integrators that a visualisation
tool exists. `docs/development.rst` should note that `tools/` is now covered by all
quality checks. The changelog records the move and cleanup.

- [ ] **Step 1: Add a "Visualisation" section to `docs/usage.rst`**

Add after the "Review notes" section:

```rst
Visualisation
-------------

A standalone visualisation tool is provided in ``tools/visualise.py``. It renders
bounding boxes and confidence scores from ScoutBot JSON output onto the source image.
By default it hands the result to the OS default handler for the file URI (Preview on
macOS, ``xdg-open`` on Linux); use ``--output`` with ``--no-display`` to save without
opening anything.

.. code-block:: bash

   uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json
   uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json --threshold 0.5
   uv run python tools/visualise.py IMAGE.jpg DETECTIONS.json --output annotated.jpg --no-display

When the JSON file contains multiple image entries (e.g. from ``scoutbot batch``
output), specify which image to display with ``--image-key``:

.. code-block:: bash

   uv run python tools/visualise.py IMAGE.jpg batch_results.json --image-key /path/to/image.jpg

Pass ``--help`` for the full option list.
```

- [ ] **Step 2: Update quality-checks section in `docs/development.rst`**

Extend the Bandit paragraph to mention `tools/`:

Find this line:
```
Source-code security scanning checks the ``scoutbot/`` and ``benchmarks/`` trees for common
Python security mistakes (``tests/`` is excluded — pytest assertions are expected there):
```

Replace with:
```
Source-code security scanning checks the ``scoutbot/``, ``benchmarks/``, and ``tools/``
trees for common Python security mistakes (``tests/`` is excluded — pytest assertions are
expected there):
```

Also update the `uv run bandit` command shown:
```
   uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
```

And update the `quality.yml` CI description to mention `tools/`:
```
   runs Ruff format check, Ruff lint, ``uv run mypy scoutbot tools/visualise.py``, a
   locked-graph dependency audit (``uv audit --locked``), and a Bandit source-code security
   scan (``scoutbot/``, ``benchmarks/``, ``tools/``).
```

- [ ] **Step 3: Update quality commands in `CONTRIBUTING.md`**

Change the bandit line in the quality checklist:
```bash
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/
```
To:
```bash
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
```

Change the mypy line:
```bash
uv run mypy scoutbot
```
To:
```bash
uv run mypy scoutbot tools/visualise.py
```

In the "Type checking" section body, update the command shown there too:
```bash
uv run mypy scoutbot tools/visualise.py
```

- [ ] **Step 5: Add changelog entry**

Add to the Unreleased section of `CHANGELOG.rst`:

```rst
* Moved ``visualise.py`` from the repo root to ``tools/visualise.py`` and included it
  in all quality gates (ruff, mypy, bandit). Rewrote the script: replaced
  ``cv2.imshow`` (incompatible with the headless OpenCV build) with ``webbrowser``
  for display; added ``--no-display`` and ``--image-key`` flags; fixed label rendering
  near the top edge of the image; removed the ``numpy`` dependency; added per-class
  box colours; added a test suite in ``tests/test_visualise.py``.
```

- [ ] **Step 6: Run the fast test suite and all quality checks one final time**

```bash
uv run pytest -m "not slow" -q
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv audit --locked --preview-features audit-command
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
```

Expected: all pass.

**Exit criteria:**
- `docs/usage.rst` tells readers about the visualisation tool and its flags
- `docs/development.rst` names `tools/` in the bandit coverage list and CI description
- `CHANGELOG.rst` has an unreleased entry covering the move, rewrite, and new flags
- All quality checks pass
