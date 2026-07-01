# PR #2 Group 1 Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four correctness regressions introduced in PR #2 (Core ML localizer backend) before merging.

**Architecture:** All four fixes are surgical — one-line or few-line changes to existing functions, each backed by a new unit test that mocks infrastructure so no network or model download is needed. Tasks are independent; each produces its own passing test and commit.

**Tech Stack:** Python, pytest (`tmp_path` fixture, `monkeypatch`), `unittest.mock` (`patch`, `MagicMock`), Click `CliRunner`.

## Global Constraints

- Python ≥ 3.8, < 3.13
- Formatter: Brunette. Linting: flake8. Import order: isort. All enforced via pre-commit — run `pre-commit run --all-files` before every commit
- New tests must be offline: no network calls, no reads/writes to `~/.cache/scoutbot/`. Mock `scoutbot.loc.fetch`, `scoutbot.tile_batched.Yolov8DetectionModel`, and `scoutbot.pipeline` wherever the full pipeline is not under test
- Test runner: `pytest` from repo root
- `import shutil` belongs with the stdlib imports, alphabetically between `import os` and `from pathlib import Path` — isort will enforce this
- Tasks 1 and 2 both modify `scoutbot/__init__.py` and `tests/test_batch.py`. **Complete each task including its commit before starting the next** to avoid intra-session conflicts on those files

---

## File Map

| Action | Path | What changes |
|--------|------|-------------|
| Modify | `scoutbot/__init__.py:346` | Task 1 — remove `or QUIET` from tqdm `disable=` in `batch_resumable` |
| Modify | `scoutbot/__init__.py:313` | Task 2 — add `and 'error' not in record` to resume reader |
| Modify | `scoutbot/loc/__init__.py:6, 177-180` | Task 3 — add `import shutil` at top; replace deferred import + `.rename()` with `shutil.move()` |
| Modify | `scoutbot/scoutbot.py:289-299` | Task 4 — add `click.Choice` to `export-coreml --config`; remove redundant local import |
| Modify | `tests/test_batch.py` | Tasks 1+2 — new test classes |
| Modify | `tests/test_loc.py` | Task 3 — new test class |
| Modify | `tests/test_cli.py` | Task 4 — new test methods on `TestCLI` |

---

## Task 1: Fix progress bar silently disabled in `batch_resumable`

**Files:**
- Modify: `scoutbot/__init__.py:346`
- Test: `tests/test_batch.py`

**Background:** `QUIET = not utils.VERBOSE` evaluates to `True` in any standard environment (no `VERBOSE` env var). Line 346 reads `disable=not show_progress or QUIET`, which short-circuits to `True` unconditionally — the progress bar never renders, even with `show_progress=True`. The sibling `batch()` at line 220 correctly uses only `disable=not show_progress`.

- [ ] **Step 1: Write the failing tests**

Append after the existing `TestBatchResumable` class in `tests/test_batch.py`:

```python
from unittest.mock import patch


class TestBatchResumableProgressBar:
    """Unit tests for batch_resumable progress bar behaviour (no model/network needed)."""

    def _fake_tqdm(self, captured):
        """Return a tqdm stand-in that records the disable kwarg."""
        def factory(iterable, **kwargs):
            captured.update(kwargs)

            class FakePbar:
                def __iter__(self):
                    return iter(iterable)

                def set_postfix(self, **kw):
                    pass

            return FakePbar()

        return factory

    def test_progress_shown_when_show_progress_true(self, tmp_path):
        """show_progress=True must not be suppressed by the QUIET module constant."""
        import scoutbot

        assert scoutbot.QUIET is True, (
            'QUIET must be True (VERBOSE env var unset) for this test to prove the regression'
        )

        captured = {}

        with patch('tqdm.tqdm', self._fake_tqdm(captured)), \
             patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', return_value=(0.5, [])):
            scoutbot.batch_resumable(
                filepaths=['fake.jpg'],
                output_path=str(tmp_path / 'out.jsonl'),
                show_progress=True,
            )

        assert captured.get('disable') is False, (
            'progress bar must be enabled when show_progress=True; '
            'check that QUIET is not short-circuiting the disable flag in batch_resumable'
        )

    def test_progress_hidden_when_show_progress_false(self, tmp_path):
        """show_progress=False must suppress the progress bar regardless of QUIET."""
        import scoutbot

        captured = {}

        with patch('tqdm.tqdm', self._fake_tqdm(captured)), \
             patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', return_value=(0.5, [])):
            scoutbot.batch_resumable(
                filepaths=['fake.jpg'],
                output_path=str(tmp_path / 'out.jsonl'),
                show_progress=False,
            )

        assert captured.get('disable') is True
```

- [ ] **Step 2: Run tests to verify they fail correctly**

```bash
pytest tests/test_batch.py::TestBatchResumableProgressBar -v
```

Expected:
- `test_progress_shown_when_show_progress_true` — **FAILS** with `AssertionError: progress bar must be enabled when show_progress=True`
- `test_progress_hidden_when_show_progress_false` — PASSES (QUIET=True makes `disable=True`, which is also correct for `show_progress=False`)

- [ ] **Step 3: Apply the one-word fix**

In `scoutbot/__init__.py` line 346:

```python
# Before:
            disable=not show_progress or QUIET,
```

```python
# After:
            disable=not show_progress,
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_batch.py::TestBatchResumableProgressBar -v
```

Expected: both PASS.

- [ ] **Step 5: Run full suite and linter**

```bash
pytest
pre-commit run --all-files
```

Expected: all tests pass, linter clean.

- [ ] **Step 6: Commit**

```bash
git add scoutbot/__init__.py tests/test_batch.py
git commit -m "fix: restore progress bar in batch_resumable (QUIET was short-circuiting show_progress)"
```

---

## Task 2: Fix errored files silently skipped on resume

**Files:**
- Modify: `scoutbot/__init__.py:313`
- Test: `tests/test_batch.py`

**Background:** When a file errors during `batch_resumable`, an error record `{'filepath': ..., 'error': ...}` is appended to the JSONL. The resume reader at lines 311-315 adds any record containing `'filepath'` to `processed` — including error records — so failed images are permanently skipped on subsequent runs. Fix: only count records that have `'filepath'` AND do NOT have `'error'`.

- [ ] **Step 1: Write the failing tests**

Append after `TestBatchResumableProgressBar` in `tests/test_batch.py`:

```python
class TestBatchResumableErrorResume:
    """Unit tests for resume-on-error behaviour (no model/network needed)."""

    def test_errored_files_retried_on_resume(self, tmp_path):
        """A file that errored previously must be retried, not silently skipped."""
        import scoutbot

        output = tmp_path / 'out.jsonl'

        # Seed JSONL: one success record, one error record
        with open(output, 'w') as f:
            f.write(json.dumps({
                'filepath': 'good.jpg',
                'wic': 0.5,
                'detections': [],
                'config': 'v3',
            }) + '\n')
            f.write(json.dumps({
                'filepath': 'bad.jpg',
                'error': 'some transient error',
                'config': 'v3',
            }) + '\n')

        processed = []

        def mock_pipeline(filepath, *args, **kwargs):
            processed.append(filepath)
            return 0.0, []

        with patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', side_effect=mock_pipeline):
            summary = scoutbot.batch_resumable(
                filepaths=['good.jpg', 'bad.jpg'],
                output_path=str(output),
                show_progress=False,
            )

        assert 'good.jpg' not in processed, 'successfully completed file must be skipped'
        assert 'bad.jpg' in processed, 'previously errored file must be retried'
        assert summary['skipped'] == 1
        assert summary['processed'] == 1

    def test_successful_files_still_skipped_on_resume(self, tmp_path):
        """Files completed successfully must continue to be skipped on resume."""
        import scoutbot

        output = tmp_path / 'out.jsonl'

        with open(output, 'w') as f:
            f.write(json.dumps({
                'filepath': 'done.jpg',
                'wic': 0.9,
                'detections': [],
                'config': 'v3',
            }) + '\n')

        processed = []

        def mock_pipeline(filepath, *args, **kwargs):
            processed.append(filepath)
            return 0.0, []

        with patch('scoutbot.loc.fetch', return_value='/fake/model.pt'), \
             patch('scoutbot.tile_batched.Yolov8DetectionModel'), \
             patch('scoutbot.pipeline', side_effect=mock_pipeline):
            summary = scoutbot.batch_resumable(
                filepaths=['done.jpg'],
                output_path=str(output),
                show_progress=False,
            )

        assert 'done.jpg' not in processed
        assert summary['skipped'] == 1
        assert summary['processed'] == 0
```

- [ ] **Step 2: Run tests to verify they fail correctly**

```bash
pytest tests/test_batch.py::TestBatchResumableErrorResume -v
```

Expected:
- `test_errored_files_retried_on_resume` — **FAILS**: `bad.jpg` is currently treated as done
- `test_successful_files_still_skipped_on_resume` — PASSES

- [ ] **Step 3: Apply the fix**

In `scoutbot/__init__.py`, line 313:

```python
# Before:
                    if 'filepath' in record:
                        processed.add(record['filepath'])
```

```python
# After:
                    if 'filepath' in record and 'error' not in record:
                        processed.add(record['filepath'])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_batch.py::TestBatchResumableErrorResume -v
```

Expected: both PASS.

- [ ] **Step 5: Run full suite and linter**

```bash
pytest
pre-commit run --all-files
```

- [ ] **Step 6: Commit**

```bash
git add scoutbot/__init__.py tests/test_batch.py
git commit -m "fix: exclude error records from processed set on batch_resumable resume"
```

---

## Task 3: Replace `Path.rename()` with `shutil.move()` in `export_coreml`

**Files:**
- Modify: `scoutbot/loc/__init__.py` — add `import shutil` to top-level imports; replace deferred import + `.rename()` with `shutil.move()`
- Test: `tests/test_loc.py`

**Background:** `export_coreml` calls `exported.rename(target)` to place the `.mlpackage` bundle in `CACHE_DIR`. `Path.rename()` wraps `os.rename()`, which raises `OSError: [Errno 18] Invalid cross-device link` when source and destination are on different filesystems. Ultralytics has changed its export output directory across versions; the comment in the code acknowledges this assumption. `shutil.move()` handles cross-device moves via copy-then-delete. This fix also moves the `import shutil` from its deferred position inside an `if` block to the file's top-level stdlib imports, per isort conventions.

- [ ] **Step 1: Write the failing test**

Append after the existing classes in `tests/test_loc.py`:

```python
from unittest.mock import patch


class TestExportCoreml:
    """Unit tests for export_coreml (no network, no model load needed)."""

    def test_export_uses_shutil_move_not_rename(self, tmp_path):
        """export_coreml must call shutil.move (not Path.rename) to handle cross-device moves."""
        from scoutbot import loc

        # Simulate Ultralytics writing the bundle to an unexpected location
        other = tmp_path / 'ultralytics_out' / 'yolov8-onecls.kaza.mlpackage'
        other.parent.mkdir()
        other.mkdir()  # .mlpackage is a directory bundle

        move_calls = []

        with patch.object(loc, 'CACHE_DIR', tmp_path), \
             patch('scoutbot.loc.fetch', return_value=str(tmp_path / 'fake.pt')), \
             patch('ultralytics.YOLO') as MockYOLO, \
             patch('shutil.move', side_effect=lambda s, d: move_calls.append((str(s), str(d)))):
            MockYOLO.return_value.export.return_value = str(other)
            loc.export_coreml('v3-coreml', force=True)

        assert move_calls, (
            'shutil.move must be called to relocate the bundle; '
            'the current code uses Path.rename which fails across filesystems (EXDEV)'
        )
        src, dst = move_calls[0]
        assert dst == str(tmp_path / 'yolov8-onecls.kaza.mlpackage')
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_loc.py::TestExportCoreml::test_export_uses_shutil_move_not_rename -v
```

Expected: **FAILS** — `move_calls` is empty because the current code calls `Path.rename`, not `shutil.move`.

- [ ] **Step 3: Add `import shutil` to top-level imports**

In `scoutbot/loc/__init__.py`, the current top-level imports are:

```python
import hashlib
import os
from pathlib import Path
```

Change to:

```python
import hashlib
import os
import shutil
from pathlib import Path
```

- [ ] **Step 4: Replace the deferred import and `.rename()` call**

Find this block in `export_coreml` (around line 175):

```python
        if exported.resolve() != target.resolve():
            if target.exists():
                import shutil

                shutil.rmtree(target)
            exported.rename(target)
```

Replace with:

```python
        if exported.resolve() != target.resolve():
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(exported), str(target))
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_loc.py::TestExportCoreml::test_export_uses_shutil_move_not_rename -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite and linter**

```bash
pytest
pre-commit run --all-files
```

- [ ] **Step 7: Commit**

```bash
git add scoutbot/loc/__init__.py tests/test_loc.py
git commit -m "fix: use shutil.move instead of Path.rename in export_coreml (cross-device safety)"
```

---

## Task 4: Add `click.Choice` validation to `export-coreml --config`

**Files:**
- Modify: `scoutbot/scoutbot.py:289-299`
- Test: `tests/test_cli.py`

**Background:** The `--config` option on `export-coreml` is declared as a plain `str` with no `type=click.Choice(...)`. Passing a non-CoreML config (e.g. `--config v3`) causes Click to pass the value through to `loc.export_coreml`, which raises a bare `AssertionError` with a Python traceback. Every other command uses `click.Choice` derived from `loc.CONFIGS`. This fix derives choices the same way, filtered to configs where `backend == 'coreml'`. It also removes a redundant `from scoutbot import loc` local import inside the command function — `loc` is already imported at module level on line 15.

- [ ] **Step 1: Write the failing tests**

Add inside the existing `TestCLI` class in `tests/test_cli.py`:

```python
    def test_export_coreml_help_lists_coreml_configs(self):
        """export-coreml --help must enumerate the valid coreml config choices."""
        from scoutbot.scoutbot import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ['export-coreml', '--help'])

        assert result.exit_code == 0
        assert 'v3-coreml' in result.output
        assert 'v3-cls-coreml' in result.output

    def test_export_coreml_rejects_non_coreml_config_with_click_error(self):
        """Passing a non-CoreML config must produce a Click usage error, not an AssertionError."""
        from scoutbot.scoutbot import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ['export-coreml', '--config', 'v3'])

        assert result.exit_code != 0
        assert 'AssertionError' not in result.output, (
            'the error must be caught by Click before reaching loc.export_coreml'
        )
        # Click emits "Invalid value for '--config'" for bad Choice values
        assert 'invalid' in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py::TestCLI::test_export_coreml_help_lists_coreml_configs tests/test_cli.py::TestCLI::test_export_coreml_rejects_non_coreml_config_with_click_error -v
```

Expected:
- `test_export_coreml_help_lists_coreml_configs` — **FAILS**: `--help` shows `--config TEXT`, no choices listed
- `test_export_coreml_rejects_non_coreml_config_with_click_error` — **FAILS**: output contains `AssertionError` traceback

- [ ] **Step 3: Apply the fix**

In `scoutbot/scoutbot.py`, replace the `export_coreml` command's decorator and first line of the function body:

```python
# Before:
@click.command('export-coreml')
@click.option(
    '--config',
    default=None,
    help='Core ML config to export (default: all coreml configs).',
)
@click.option('--force', is_flag=True, help='Re-export even if the bundle exists.')
def export_coreml(config, force):
    """
    Export Core ML (.mlpackage) models from their source .pt models.
    """
    from scoutbot import loc

    if config:
```

```python
# After:
@click.command('export-coreml')
@click.option(
    '--config',
    default=None,
    type=click.Choice(
        [k for k, v in loc.CONFIGS.items()
         if k is not None and isinstance(v, dict) and v.get('backend') == 'coreml'],
    ),
    help='Core ML config to export (default: all coreml configs).',
)
@click.option('--force', is_flag=True, help='Re-export even if the bundle exists.')
def export_coreml(config, force):
    """
    Export Core ML (.mlpackage) models from their source .pt models.
    """
    if config:
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli.py::TestCLI::test_export_coreml_help_lists_coreml_configs tests/test_cli.py::TestCLI::test_export_coreml_rejects_non_coreml_config_with_click_error -v
```

Expected: both PASS.

- [ ] **Step 5: Run full suite and linter**

```bash
pytest
pre-commit run --all-files
```

- [ ] **Step 6: Commit**

```bash
git add scoutbot/scoutbot.py tests/test_cli.py
git commit -m "fix: add click.Choice validation to export-coreml --config; remove redundant local import"
```

---

## Self-Review

**Spec coverage:**
- 1a (progress bar regression) ✓ Task 1
- 1b (error records skipped on resume) ✓ Task 2
- 1c (cross-device rename) ✓ Task 3
- 1d (export-coreml CLI validation) ✓ Task 4

**Placeholder scan:** No TBD, no "similar to Task N", no steps without code. All test code is complete and runnable.

**Type/name consistency:**
- `scoutbot.batch_resumable`, `scoutbot.loc.fetch`, `scoutbot.tile_batched.Yolov8DetectionModel`, `scoutbot.pipeline` — all match the current signatures in `scoutbot/__init__.py`
- `loc.CONFIGS`, `loc.export_coreml` — match `scoutbot/loc/__init__.py`
- `loc.CACHE_DIR` — patched via `patch.object(loc, 'CACHE_DIR', tmp_path)` in Task 3

**Ordering note:** Tasks 1 and 2 both modify `scoutbot/__init__.py` and `tests/test_batch.py`. Commit Task 1 before starting Task 2.
