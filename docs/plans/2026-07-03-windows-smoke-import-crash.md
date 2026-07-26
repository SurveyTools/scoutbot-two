# Windows Smoke Import Crash Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `import scoutbot` stop pulling `numpy` in at module import time so the Windows smoke
test can collect the suite reliably.

**Architecture:** The crash is happening during test collection because `scoutbot.tile_batched.main`
imports `numpy` at module scope. The fix should move that dependency to the point of use inside
`perform_inference()` so the package can be imported on Windows without touching the fragile native
initializer until inference actually runs. A regression test should prove that `import scoutbot`
succeeds even when a stub `numpy` module is placed ahead of the real one on `PYTHONPATH`.

**Tech Stack:** Python stdlib, pytest, uv.

## Global Constraints

- Use `uv run` for Python/tool commands
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Pause after each task for review before moving on

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `scoutbot/tile_batched/main.py` | Move the `numpy` import out of module scope and into the inference path that actually uses it |
| Add | `tests/test_imports.py` | Add a regression test that proves `import scoutbot` does not require module-level `numpy` import |

---

## Task 1: Add the import-regression test first

**Files:**
- Create: `tests/test_imports.py`

**Background:** The failure happens during import collection, so the regression test needs to
simulate a broken `numpy` import before we change the production code. The cleanest way is a fresh
subprocess with a temporary `numpy.py` placed on `PYTHONPATH` ahead of the real package.

- [ ] Write a test that creates a temporary directory containing a `numpy.py` stub which raises `RuntimeError` if imported
- [ ] Run a subprocess with that directory prepended to `PYTHONPATH` and execute `import scoutbot`
- [ ] Assert that the subprocess exits successfully and prints `ok`

**Test code:**

```python
import os
import subprocess
import sys


def test_import_scoutbot_does_not_require_numpy_at_module_import(tmp_path):
    stub_dir = tmp_path / 'stubs'
    stub_dir.mkdir()
    (stub_dir / 'numpy.py').write_text(
        "raise RuntimeError('numpy should not be imported during scoutbot import')\n"
    )

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(stub_dir), env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, '-c', 'import scoutbot; print("ok")'],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'ok'
```

**Exit criteria:**
- The test fails on the current codebase before the fix
- The test describes the exact import-time behavior we want to protect

---

## Task 2: Move `numpy` to the point of use

**Files:**
- Modify: `scoutbot/tile_batched/main.py`

**Background:** `numpy` is only used inside `perform_inference()` to normalize a single array into
a list. Keeping it at module scope forces Windows to initialize `numpy` during test collection.
Import it locally so package import stays lightweight.

- [ ] Remove the top-level `import numpy as np`
- [ ] Add `import numpy as np` inside `perform_inference()` just before the `isinstance(images, np.ndarray)` check
- [ ] Leave the rest of the inference logic unchanged so runtime behavior stays the same

**Target code:**

```python
def perform_inference(self, images, batch_size: Optional[int] = None):
    import numpy as np

    if self.model is None:
        raise ValueError('Model is not loaded, call .load_model() first')

    # Core ML inference is dispatched by Ultralytics regardless of the nominal
    # torch device; force 'cpu' so the cuda:0 default is never passed through.
    device = 'cpu' if self._coreml else self.device
    if isinstance(images, np.ndarray):
        images = [images]
```

**Verification:**
- Run the new import regression test and confirm it passes
- Run `uv run pytest tests/test_tile_batched.py -q` and confirm the existing tile-batched unit tests still pass

---

## Task 3: Verify the Windows smoke path

**Files:**
- Re-read: `scoutbot/tile_batched/main.py`, `tests/test_imports.py`

**Background:** The Windows smoke job is only useful if the package import is stable during
collection. The final check should use the same fast, non-slow test gate that CI runs on Windows.

- [ ] Re-read the touched files and confirm the import regression is testing the same failure mode reported by Windows CI
- [ ] Run `uv run pytest -m "not slow" -q`
- [ ] Confirm the suite collects and runs without crashing at import time
- [ ] Pause for review with a concise summary of the import-time fix and its CI impact
