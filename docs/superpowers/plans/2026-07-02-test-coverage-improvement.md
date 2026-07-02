# Test Coverage Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise ScoutBot's fast-suite test coverage from 58% (223/385 stmts) to 85%+ by adding unit tests that use mocks instead of network calls or model loading.

**Architecture:** Each task adds tests to an existing or new test file. All tests use `unittest.mock` to avoid network/model dependencies. Slow tests that require real models or network access are marked `@pytest.mark.slow` and excluded from the fast CI suite. No production code changes are required.

**Tech Stack:** pytest, `unittest.mock` (`patch`, `MagicMock`), `click.testing.CliRunner`, `tmp_path` fixture.

**Prioritization approach:** Tasks are ordered and scoped for *meaningful branch coverage*, not raw line count. A branch is meaningful if skipping it could mask a real defect — exception handlers, early-return guards, "export all" vs "export one" paths, error-record writes. Branches that exist only to satisfy a metric (e.g. `if __name__ == '__main__':`) are explicitly excluded. When a choice exists between covering more lines cheaply or covering fewer lines that represent actual robustness paths, prefer the robustness paths.

## Global Constraints

- Always run pytest as: `uv run pytest -m "not slow" --cov=scoutbot --cov-report=term-missing`
- Never run `python`, `pytest`, or `ruff` without the `uv run` prefix
- Do NOT run `pre-commit run` — the pre-commit environment has a Poetry/venv conflict
- Do NOT run `git add` or `git commit` — the user handles all commits
- Do not add `@pytest.mark.slow` unless the test actually hits network or loads a real model file from disk
- Test files follow the pattern: one class per behavior cluster, docstrings only on classes
- Use `patch` and `MagicMock` from `unittest.mock` throughout; no `pytest-mock`
- Quote style: single quotes (Ruff enforces `quote-style = "single"`)
- Run only the specific test file after each step to keep iteration fast

---

## File Map

| File | Role |
|------|------|
| `tests/test_loc.py` | Task 1 — add `TestLocMd5`, `TestLocFetchUnit`, `TestExportCoremlBranches` classes |
| `tests/test_pipeline.py` | Task 2 — add `TestPipelineUnit` class (mocked SAHI) |
| `tests/test_batch.py` | Task 2 — add `TestBatchUnit`, `TestBatchResumableUnit` classes |
| `tests/test_tile_batched.py` | Task 4 — add `TestCoremlInputSize`, `TestPerformInferenceModelNone` classes |
| `tests/test_cli.py` | Task 3 — add `TestCLIExecutionBodies` class |
| `tests/test_utils.py` | Task 5 — create new file with `TestCheckFileIntegrity` |

---

## Task 1: `loc` unit tests — `_md5()`, `fetch()` branches, `export_coreml()` branches

**Files:**
- Modify: `tests/test_loc.py`

**Covers:** `scoutbot/loc/__init__.py` lines 79-83 (`_md5`), 99-138 (`fetch()` body), 161-163 (early return), 178 (rmtree before move)

**Interfaces:**
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Write the failing tests for `_md5()` and `fetch()` (unit, no network)**

Add to the bottom of `tests/test_loc.py`:

```python
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestLocMd5:
    """Unit tests for scoutbot.loc._md5()."""

    def test_md5_returns_correct_hex_digest(self, tmp_path):
        from scoutbot.loc import _md5

        content = b'hello world'
        f = tmp_path / 'test.bin'
        f.write_bytes(content)

        result = _md5(f)

        assert result == hashlib.md5(content).hexdigest()
        assert isinstance(result, str)
        assert len(result) == 32

    def test_md5_handles_large_file_in_chunks(self, tmp_path):
        from scoutbot.loc import _md5

        # 20KB — forces multiple 8192-byte chunks
        content = b'x' * 20_000
        f = tmp_path / 'large.bin'
        f.write_bytes(content)

        result = _md5(f)
        assert result == hashlib.md5(content).hexdigest()


class TestLocFetchUnit:
    """Unit tests for scoutbot.loc.fetch() — no network, no real model."""

    def test_fetch_coreml_returns_path_when_bundle_exists(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        bundle = tmp_path / cfg['name']
        bundle.mkdir()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('scoutbot.loc.export_coreml') as mock_export,
        ):
            result = loc.fetch('v3-coreml')

        mock_export.assert_not_called()
        assert result == str(bundle)

    def test_fetch_coreml_triggers_export_when_bundle_missing(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        bundle = tmp_path / cfg['name']

        def _create_bundle(config, **kwargs):
            bundle.mkdir(exist_ok=True)

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('scoutbot.loc.export_coreml', side_effect=_create_bundle) as mock_export,
        ):
            result = loc.fetch('v3-coreml')

        mock_export.assert_called_once_with('v3-coreml')
        assert result == str(bundle)

    def test_fetch_pt_returns_cached_path_when_md5_matches(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3']
        local = tmp_path / cfg['name']
        local.write_bytes(b'fake model bytes')
        remote_md5 = hashlib.md5(b'fake model bytes').hexdigest()

        md5_response = MagicMock()
        md5_response.text = remote_md5 + '  ' + cfg['name']
        md5_response.raise_for_status = MagicMock()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('requests.get', return_value=md5_response),
            patch('scoutbot.loc._md5', return_value=remote_md5),
        ):
            result = loc.fetch('v3')

        assert result == str(local)

    def test_fetch_pt_downloads_when_local_missing(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3']
        local = tmp_path / cfg['name']
        remote_md5 = 'abc123deadbeef00' * 2  # 32-char fake md5

        fake_content = b'model weights'
        remote_md5 = hashlib.md5(fake_content).hexdigest()

        md5_response = MagicMock()
        md5_response.text = remote_md5 + '  ' + cfg['name']
        md5_response.raise_for_status = MagicMock()

        download_response = MagicMock()
        download_response.raise_for_status = MagicMock()
        download_response.headers = {'content-length': str(len(fake_content))}
        download_response.iter_content.return_value = [fake_content]
        download_response.__enter__ = MagicMock(return_value=download_response)
        download_response.__exit__ = MagicMock(return_value=False)

        call_count = [0]

        def _get(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return md5_response
            return download_response

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('requests.get', side_effect=_get),
        ):
            result = loc.fetch('v3')

        assert result == str(local)
        assert local.exists()
        assert local.read_bytes() == fake_content

    def test_fetch_none_config_uses_default(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS[loc.DEFAULT_CONFIG]
        local = tmp_path / cfg['name']
        local.write_bytes(b'bytes')
        remote_md5 = hashlib.md5(b'bytes').hexdigest()

        md5_response = MagicMock()
        md5_response.text = remote_md5 + '  ' + cfg['name']
        md5_response.raise_for_status = MagicMock()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('requests.get', return_value=md5_response),
            patch('scoutbot.loc._md5', return_value=remote_md5),
        ):
            result = loc.fetch(None)

        assert result is not None
        assert isinstance(result, str)


class TestExportCoremlBranches:
    """Unit tests for export_coreml() branches not covered by TestExportCoreml."""

    def test_early_return_when_target_exists_and_no_force(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        target = tmp_path / cfg['name']
        target.mkdir()

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('ultralytics.YOLO') as MockYOLO,
        ):
            result = loc.export_coreml('v3-coreml', force=False)

        MockYOLO.assert_not_called()
        assert result == str(target)

    def test_rmtree_called_when_target_exists_and_exported_differs(self, tmp_path):
        from scoutbot import loc

        cfg = loc.CONFIGS['v3-coreml']
        target = tmp_path / cfg['name']
        target.mkdir()  # target already exists

        # exported lands somewhere else (different from target)
        other_dir = tmp_path / 'tmp_export'
        other_dir.mkdir()
        exported_path = other_dir / cfg['name']
        exported_path.mkdir()

        rmtree_calls = []

        with (
            patch.object(loc, 'CACHE_DIR', tmp_path),
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('ultralytics.YOLO') as MockYOLO,
            patch('shutil.rmtree', side_effect=lambda p, **kw: rmtree_calls.append(str(p))),
            patch('shutil.move'),
        ):
            MockYOLO.return_value.export.return_value = str(exported_path)
            loc.export_coreml('v3-coreml', force=True)

        assert any(str(target) in c for c in rmtree_calls), (
            'shutil.rmtree must remove the existing target bundle '
            'before shutil.move places the newly-exported one there'
        )
```

- [ ] **Step 2: Run to verify tests fail (no production changes needed — these are all pure test files)**

```bash
uv run pytest tests/test_loc.py::TestLocMd5 tests/test_loc.py::TestLocFetchUnit tests/test_loc.py::TestExportCoremlBranches -v
```

Expected: tests run and pass (the production code already exists; we're adding tests that call it via mocks). If any fail, the mock setup is wrong — fix the mock, not the source.

- [ ] **Step 3: Verify coverage improvement on loc module**

```bash
uv run pytest tests/test_loc.py -m "not slow" --cov=scoutbot/loc --cov-report=term-missing -v
```

Expected: `scoutbot/loc/__init__.py` coverage rises from 45% to 80%+. Lines 79-83, 99-138, 161-163, and 178 should appear covered. If any line is still missing, add a test case that exercises the missing branch.

- [ ] **Step 4: Commit**

```
git add tests/test_loc.py
git commit -m "test: add unit tests for loc._md5, fetch branches, and export_coreml branches"
```

---

## Task 2: `scoutbot/__init__.py` unit tests — `pipeline()`, `batch()`, `batch_resumable()`

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_batch.py`

**Covers:** `scoutbot/__init__.py` lines 114-163 (`pipeline()`), 202-253 (`batch()`), 316 (blank line skip), 321-322 (JSONDecodeError), 326-333 (all-processed early return), 392-402 (`except Exception` error record path), 438-468 (`example()`, marked slow)

**Interfaces:**
- Consumes: nothing from earlier tasks

The key mock pattern for `pipeline()` is to patch `sahi.predict.get_sliced_prediction` so it returns a fake `SlicedPredictionResult` with an `object_prediction_list`. Each item in that list has a `.to_coco_prediction(image_id=None).json` that returns a COCO-format dict:

```python
# Format expected by pipeline():
{
    'score': 0.85,
    'category_name': 'animal',
    'bbox': [10.0, 20.0, 50.0, 30.0],  # x, y, w, h
}
```

- [ ] **Step 1: Write failing tests — add `TestPipelineUnit` to `tests/test_pipeline.py`**

Add below the existing `TestPipeline` class in `tests/test_pipeline.py`:

```python
from unittest.mock import MagicMock, patch


def _fake_sahi_result(predictions):
    """Build a fake SlicedPredictionResult with the given COCO-format predictions."""
    mock_result = MagicMock()
    mock_preds = []
    for score, category, bbox in predictions:
        pred = MagicMock()
        pred.to_coco_prediction.return_value.json = {
            'score': score,
            'category_name': category,
            'bbox': bbox,
        }
        mock_preds.append(pred)
    mock_result.object_prediction_list = mock_preds
    return mock_result


class TestPipelineUnit:
    """Unit tests for scoutbot.pipeline() using mocked SAHI — no model or network."""

    def test_pipeline_returns_wic_and_detects(self, tmp_path):
        import scoutbot

        fake_img = tmp_path / 'test.jpg'
        fake_img.write_bytes(b'fake')

        fake_result = _fake_sahi_result([
            (0.85, 'animal', [10.0, 20.0, 50.0, 30.0]),
            (0.60, 'vehicle', [100.0, 50.0, 20.0, 15.0]),
        ])

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            wic, detects = scoutbot.pipeline(str(fake_img), config='v3')

        assert wic == 0.85
        assert len(detects) == 2
        assert detects[0]['l'] == 'animal'
        assert detects[0]['c'] == 0.85
        assert detects[0]['x'] == 10.0
        assert detects[0]['y'] == 20.0
        assert detects[0]['w'] == 50.0
        assert detects[0]['h'] == 30.0

    def test_pipeline_returns_zero_wic_when_no_detections(self, tmp_path):
        import scoutbot

        fake_img = tmp_path / 'empty.jpg'
        fake_img.write_bytes(b'fake')

        fake_result = _fake_sahi_result([])  # no detections

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            wic, detects = scoutbot.pipeline(str(fake_img), config='v3')

        assert wic == 0
        assert detects == []

    def test_pipeline_reuses_provided_model(self, tmp_path):
        """When batched_detection_model is given, loc.fetch must not be called."""
        import scoutbot

        fake_img = tmp_path / 'test.jpg'
        fake_img.write_bytes(b'fake')

        mock_model = MagicMock()
        fake_result = _fake_sahi_result([])

        with (
            patch('scoutbot.loc.fetch') as mock_fetch,
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            scoutbot.pipeline(
                str(fake_img), config='v3', batched_detection_model=mock_model
            )

        mock_fetch.assert_not_called()

    def test_pipeline_none_config_defaults_to_v3(self, tmp_path):
        import scoutbot

        fake_img = tmp_path / 'test.jpg'
        fake_img.write_bytes(b'fake')
        fake_result = _fake_sahi_result([])

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch,
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('sahi.predict.get_sliced_prediction', return_value=fake_result),
        ):
            scoutbot.pipeline(str(fake_img), config=None)

        mock_fetch.assert_called_once_with(config='v3')
```

- [ ] **Step 2: Run and verify pipeline tests pass**

```bash
uv run pytest tests/test_pipeline.py::TestPipelineUnit -v
```

Expected: 4 passed.

- [ ] **Step 3: Write failing tests — add `TestBatchUnit` and `TestBatchResumableUnit` to `tests/test_batch.py`**

Add below the existing `TestBatchResumable` class in `tests/test_batch.py`:

```python
import json
from unittest.mock import MagicMock, patch


class TestBatchUnit:
    """Unit tests for scoutbot.batch() — no model or network."""

    def test_batch_returns_wic_and_detects_lists(self, tmp_path):
        import scoutbot

        fake_imgs = [str(tmp_path / f'{i}.jpg') for i in range(3)]
        for p in fake_imgs:
            open(p, 'wb').close()

        fake_wic = 0.75
        fake_detects = [{'l': 'animal', 'c': 0.75, 'x': 0, 'y': 0, 'w': 10, 'h': 10}]

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(fake_wic, fake_detects)),
        ):
            wic_list, detects_list = scoutbot.batch(
                fake_imgs, config='v3', show_progress=False
            )

        assert len(wic_list) == 3
        assert len(detects_list) == 3
        assert all(w == fake_wic for w in wic_list)

    def test_batch_none_config_uses_default(self, tmp_path):
        import scoutbot

        fake_img = str(tmp_path / 'img.jpg')
        open(fake_img, 'wb').close()

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt') as mock_fetch,
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.0, [])),
        ):
            scoutbot.batch([fake_img], config=None, show_progress=False)

        mock_fetch.assert_called_once_with(config='v3')


class TestBatchResumableUnit:
    """Unit tests for scoutbot.batch_resumable() — no model or network."""

    def test_all_already_processed_returns_early(self, tmp_path):
        import scoutbot

        output = tmp_path / 'output.jsonl'
        filepaths = [str(tmp_path / 'a.jpg'), str(tmp_path / 'b.jpg')]

        # Write records for both filepaths as already done
        with open(output, 'w') as f:
            for fp in filepaths:
                f.write(json.dumps({'filepath': fp, 'wic': 0.5, 'detections': []}) + '\n')

        with patch('scoutbot.loc.fetch') as mock_fetch:
            summary = scoutbot.batch_resumable(
                filepaths=filepaths,
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        mock_fetch.assert_not_called()
        assert summary['processed'] == 0
        assert summary['skipped'] == 2
        assert summary['total'] == 2

    def test_jsonl_parser_skips_blank_lines_and_bad_json(self, tmp_path):
        """Lines 316 (blank) and 321-322 (JSONDecodeError) of batch_resumable."""
        import scoutbot

        output = tmp_path / 'output.jsonl'
        fake_img = str(tmp_path / 'img.jpg')
        open(fake_img, 'wb').close()

        with open(output, 'w') as f:
            f.write('\n')                          # blank line → hits line 316
            f.write('not valid json\n')            # bad JSON → hits lines 321-322
            f.write(json.dumps({'bad': 'record'}) + '\n')  # valid JSON, no filepath key

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', return_value=(0.0, [])),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=[fake_img],
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        # img.jpg was not in processed set (no valid completed record for it)
        assert summary['processed'] == 1
        assert summary['errors'] == 0

    def test_batch_resumable_runs_and_writes_jsonl(self, tmp_path):
        import scoutbot

        output = tmp_path / 'output.jsonl'
        fake_imgs = [str(tmp_path / f'{i}.jpg') for i in range(2)]
        for p in fake_imgs:
            open(p, 'wb').close()

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch(
                'scoutbot.pipeline',
                return_value=(0.85, [{'l': 'animal', 'c': 0.85, 'x': 0, 'y': 0, 'w': 5, 'h': 5}]),
            ),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=fake_imgs,
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        assert summary['processed'] == 2
        assert summary['errors'] == 0
        assert summary['skipped'] == 0
        assert summary['total'] == 2
        assert output.exists()
        lines = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        for rec in lines:
            assert 'filepath' in rec
            assert 'wic' in rec
            assert 'detections' in rec
            assert 'config' in rec

    def test_batch_resumable_resumes_from_partial_output(self, tmp_path):
        import scoutbot

        output = tmp_path / 'output.jsonl'
        img_a = str(tmp_path / 'a.jpg')
        img_b = str(tmp_path / 'b.jpg')
        open(img_a, 'wb').close()
        open(img_b, 'wb').close()

        # Pre-populate with img_a already done
        with open(output, 'w') as f:
            f.write(json.dumps({'filepath': img_a, 'wic': 0.5, 'detections': []}) + '\n')

        pipeline_calls = []

        def _fake_pipeline(fp, *args, **kwargs):
            pipeline_calls.append(fp)
            return (0.9, [])

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=_fake_pipeline),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=[img_a, img_b],
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        # Only img_b should have been processed
        assert pipeline_calls == [img_b]
        assert summary['processed'] == 1
        assert summary['skipped'] == 1

    def test_batch_resumable_writes_error_record_on_pipeline_failure(self, tmp_path):
        """Covers lines 392-402: except Exception path writes an error record."""
        import scoutbot

        output = tmp_path / 'output.jsonl'
        fake_img = str(tmp_path / 'img.jpg')
        open(fake_img, 'wb').close()

        with (
            patch('scoutbot.loc.fetch', return_value='/fake/model.pt'),
            patch('scoutbot.tile_batched.Yolov8DetectionModel'),
            patch('scoutbot.pipeline', side_effect=RuntimeError('inference exploded')),
        ):
            summary = scoutbot.batch_resumable(
                filepaths=[fake_img],
                output_path=str(output),
                config='v3',
                show_progress=False,
            )

        assert summary['errors'] == 1
        assert summary['processed'] == 0
        assert output.exists()
        lines = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]['filepath'] == fake_img
        assert 'error' in lines[0]
        assert 'inference exploded' in lines[0]['error']


@pytest.mark.slow
class TestExampleSlow:
    """Integration test for scoutbot.example() — downloads from network."""

    def test_example_runs_without_error(self):
        import scoutbot

        # Downloads example image and runs the full pipeline
        # Requires network access and model download
        scoutbot.example()
```

- [ ] **Step 4: Run and verify batch tests pass**

```bash
uv run pytest tests/test_batch.py::TestBatchUnit tests/test_batch.py::TestBatchResumableUnit -v
```

Expected: all unit tests pass. The `TestExampleSlow` class is excluded because it's marked slow.

- [ ] **Step 5: Check coverage on `scoutbot/__init__.py`**

```bash
uv run pytest tests/test_pipeline.py tests/test_batch.py -m "not slow" --cov=scoutbot/__init__.py --cov-report=term-missing -v
```

Expected: `scoutbot/__init__.py` rises from 52% to 85%+. Lines 114-163 and 202-253 should be covered. Lines 316, 321-322, 326-333, and 392-402 should be covered. Lines 438-468 remain uncovered (slow test excluded from fast suite).

- [ ] **Step 6: Commit**

```
git add tests/test_pipeline.py tests/test_batch.py
git commit -m "test: add unit tests for pipeline, batch, and batch_resumable with mocked SAHI"
```

---

## Task 3: CLI execution tests

**Files:**
- Modify: `tests/test_cli.py`

**Covers:** `scoutbot/scoutbot.py` lines 23 (validator return path), 109-136 (pipeline body), 179-210 (batch body), 253-278 (batch-resumable body), 287 (fetch body), 308-319 (export-coreml body), 327 (example body), 337-343 (get_classes body)

**Interfaces:**
- Consumes: nothing from earlier tasks

The CLI's `pipeline` command uses `pipeline_filepath_validator` which calls `exists(value)`. Line 23 (`return value`) is hit when the file DOES exist. The validator check happens before `scoutbot.pipeline` is called, so we need an actual file in `tmp_path` to cover line 23.

- [ ] **Step 1: Write the failing tests — add `TestCLIExecutionBodies` to `tests/test_cli.py`**

Add below the existing `TestCLI` class in `tests/test_cli.py`:

```python
import json
from unittest.mock import patch


class TestCLIExecutionBodies:
    """Tests for CLI command execution — all scoutbot functions mocked."""

    def test_pipeline_runs_and_prints_json(self, tmp_path):
        """Covers lines 109-136 and the validator return path (line 23)."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img = tmp_path / 'test.jpg'
        img.write_bytes(b'fake')

        with patch('scoutbot.pipeline', return_value=(0.85, [{'l': 'animal', 'c': 0.85, 'x': 0, 'y': 0, 'w': 5, 'h': 5}])):
            runner = CliRunner()
            result = runner.invoke(cli, ['pipeline', '--device', 'cpu', str(img)])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert str(img) in output
        assert 'wic' in output[str(img)]
        assert output[str(img)]['wic'] == 0.85

    def test_pipeline_writes_output_file(self, tmp_path):
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img = tmp_path / 'test.jpg'
        img.write_bytes(b'fake')
        out = tmp_path / 'result.json'

        with patch('scoutbot.pipeline', return_value=(0.0, [])):
            runner = CliRunner()
            result = runner.invoke(cli, ['pipeline', '--device', 'cpu', '--output', str(out), str(img)])

        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert str(img) in data

    def test_batch_runs_and_prints_json(self, tmp_path):
        """Covers lines 179-210."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img1 = tmp_path / 'a.jpg'
        img2 = tmp_path / 'b.jpg'
        img1.write_bytes(b'fake')
        img2.write_bytes(b'fake')

        with patch(
            'scoutbot.batch',
            return_value=([0.85, 0.0], [[{'l': 'animal', 'c': 0.85, 'x': 0, 'y': 0, 'w': 5, 'h': 5}], []]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ['batch', '--device', 'cpu', '--no-progress', str(img1), str(img2)],
            )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert str(img1) in data
        assert str(img2) in data

    def test_batch_resumable_runs_and_prints_summary(self, tmp_path):
        """Covers lines 253-278."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        img1 = str(tmp_path / 'a.jpg')
        img2 = str(tmp_path / 'b.jpg')
        out = tmp_path / 'output.jsonl'

        with patch(
            'scoutbot.batch_resumable',
            return_value={'processed': 2, 'skipped': 0, 'errors': 0, 'total': 2},
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ['batch-resumable', '--output', str(out), '--no-progress', img1, img2],
            )

        assert result.exit_code == 0
        assert 'Processed: 2' in result.output
        assert 'Total: 2' in result.output

    def test_fetch_command_calls_scoutbot_fetch(self):
        """Covers line 287."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch('scoutbot.fetch') as mock_fetch:
            runner = CliRunner()
            result = runner.invoke(cli, ['fetch', '--config', 'v3'])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(config='v3')

    def test_get_classes_command_prints_json(self):
        """Covers lines 337-343."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch('scoutbot.get_classes', return_value=['animal', 'vehicle']):
            runner = CliRunner()
            result = runner.invoke(cli, ['get_classes', '--config', 'v3'])

        assert result.exit_code == 0
        classes = json.loads(result.output)
        assert classes == ['animal', 'vehicle']

    def test_get_classes_command_writes_output_file(self, tmp_path):
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        out = tmp_path / 'classes.json'

        with patch('scoutbot.get_classes', return_value=['animal']):
            runner = CliRunner()
            result = runner.invoke(cli, ['get_classes', '--config', 'v3', '--output', str(out)])

        assert result.exit_code == 0
        assert out.exists()
        assert json.loads(out.read_text()) == ['animal']

    def test_export_coreml_command_with_explicit_config(self):
        """Covers the `if config:` branch of export-coreml (lines 308-310, 318-319)."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch('scoutbot.loc.export_coreml', return_value='/fake/model.mlpackage') as mock_export:
            runner = CliRunner()
            result = runner.invoke(cli, ['export-coreml', '--config', 'v3-coreml'])

        assert result.exit_code == 0
        mock_export.assert_called_once_with('v3-coreml', force=False)

    def test_export_coreml_command_without_config_exports_all(self):
        """Covers the `else` branch (lines 311-317) — iterates all coreml configs."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli
        from scoutbot import loc

        with patch('scoutbot.loc.export_coreml', return_value='/fake/model.mlpackage') as mock_export:
            runner = CliRunner()
            result = runner.invoke(cli, ['export-coreml'])

        assert result.exit_code == 0
        # One call per coreml config (v3-coreml, v3-cls-coreml)
        expected_configs = [
            name
            for name, cfg in loc.CONFIGS.items()
            if name is not None and isinstance(cfg, dict) and cfg.get('backend') == 'coreml'
        ]
        assert mock_export.call_count == len(expected_configs)
        called_configs = [call.args[0] for call in mock_export.call_args_list]
        assert sorted(called_configs) == sorted(expected_configs)

    def test_example_command_calls_scoutbot_example(self):
        """Covers line 327."""
        from click.testing import CliRunner

        from scoutbot.scoutbot import cli

        with patch('scoutbot.example') as mock_example:
            runner = CliRunner()
            result = runner.invoke(cli, ['example'])

        assert result.exit_code == 0
        mock_example.assert_called_once()
```

- [ ] **Step 2: Run to verify tests pass**

```bash
uv run pytest tests/test_cli.py::TestCLIExecutionBodies -v
```

Expected: all 10 tests pass.

- [ ] **Step 3: Check coverage on `scoutbot/scoutbot.py`**

```bash
uv run pytest tests/test_cli.py -m "not slow" --cov=scoutbot/scoutbot --cov-report=term-missing -v
```

Expected: `scoutbot/scoutbot.py` rises from 58% to 90%+. Lines 23, 109-136, 179-210, 253-278, 287, 308-319, 327, and 337-343 should now be covered. Line 364 (`if __name__ == '__main__':`) is expected to remain uncovered — this is normal and does not require a fix.

- [ ] **Step 4: Commit**

```
git add tests/test_cli.py
git commit -m "test: add CLI execution body tests with mocked scoutbot functions"
```

---

## Task 4: `tile_batched` unit tests — `_coreml_input_size()` and model-is-None guard

**Files:**
- Modify: `tests/test_tile_batched.py`

**Covers:** `scoutbot/tile_batched/main.py` lines 14-18 (`_coreml_input_size` body), 23-25 (`__init__` body), 59 (`model is None` raise), 68 (ndarray-to-list wrapper in `perform_inference`)

**Interfaces:**
- Consumes: nothing from earlier tasks

Notes:
- `_coreml_input_size` does `import coremltools as ct` inside the function. Patch `sys.modules['coremltools']` with a `MagicMock()` so coremltools doesn't need to be installed.
- Lines 23-25 are `Yolov8DetectionModel.__init__`. The SAHI base class calls `self.load_model()` during `__init__`. Patch `load_model` at the class level to prevent any model loading.
- Line 59 is `raise ValueError(...)` when `self.model is None`. Existing tests always set `instance.model = MagicMock()` — this test sets it to `None`.
- Line 68 is `images = [images]` — the ndarray-to-list wrapper that fires when `isinstance(images, np.ndarray)` is True. All existing tests pass a list `[np.zeros(...)]` so the `isinstance` branch is never taken. Add a test that passes a bare `np.ndarray` (not wrapped in a list) to cover this branch. Note: the `imgsz` assignment is on line 75, not 68 — the existing coreml conf-kwarg test already covers that line.

- [ ] **Step 1: Write failing tests — add classes to `tests/test_tile_batched.py`**

Add below the existing classes in `tests/test_tile_batched.py`:

```python
import sys
from unittest.mock import MagicMock, patch


class TestCoremlInputSize:
    """Unit tests for _coreml_input_size() — no coremltools installation needed."""

    def test_reads_width_from_model_spec(self):
        mock_ct = MagicMock()
        mock_spec = MagicMock()
        mock_spec.description.input[0].type.imageType.width = 512
        mock_ct.models.MLModel.return_value.get_spec.return_value = mock_spec

        with patch.dict(sys.modules, {'coremltools': mock_ct}):
            from scoutbot.tile_batched.main import _coreml_input_size
            result = _coreml_input_size('/fake/model.mlpackage')

        assert result == 512
        mock_ct.models.MLModel.assert_called_once_with('/fake/model.mlpackage')

    def test_returns_int(self):
        mock_ct = MagicMock()
        mock_spec = MagicMock()
        mock_spec.description.input[0].type.imageType.width = '640'  # str input
        mock_ct.models.MLModel.return_value.get_spec.return_value = mock_spec

        with patch.dict(sys.modules, {'coremltools': mock_ct}):
            from scoutbot.tile_batched.main import _coreml_input_size
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
        import pytest

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
        source = call_kwargs.kwargs.get('source') or call_kwargs.args[0]
        assert isinstance(source, list)
        assert len(source) == 1
```

- [ ] **Step 2: Run to verify tests pass**

```bash
uv run pytest tests/test_tile_batched.py::TestCoremlInputSize tests/test_tile_batched.py::TestYolov8DetectionModelInit tests/test_tile_batched.py::TestPerformInferenceModelNone tests/test_tile_batched.py::TestPerformInferenceNdarrayInput -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Check coverage on `tile_batched/main.py`**

```bash
uv run pytest tests/test_tile_batched.py -m "not slow" --cov=scoutbot/tile_batched --cov-report=term-missing -v
```

Expected: `scoutbot/tile_batched/main.py` rises from 80% to 95%+. Lines 14-18, 23-25, 59, and 68 should all be covered.

- [ ] **Step 4: Commit**

```
git add tests/test_tile_batched.py
git commit -m "test: add unit tests for _coreml_input_size, Yolov8DetectionModel.__init__, and model-is-None guard"
```

---

## Task 5: `utils` unit tests — `check_file_integrity()`

**Files:**
- Create: `tests/test_utils.py`

**Covers:** `scoutbot/utils.py` lines 21-33 (`check_file_integrity()` body)

**Interfaces:**
- Consumes: nothing from earlier tasks

Note: `init_logging()` is called at module import time in `scoutbot/__init__.py` (line 41). The `ResourceWarning: unclosed file <scoutbot.log>` comes from the `TimedRotatingFileHandler` that `init_logging()` creates. The existing `shutdown_logging` fixture in `conftest.py` already calls `logging.shutdown()` at session teardown, which is the correct fix. No further action is needed for the warning.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_utils.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for scoutbot.utils module."""

import hashlib


class TestCheckFileIntegrity:
    """Tests for check_file_integrity() — no model or network needed."""

    def test_returns_false_when_file_does_not_exist(self):
        from scoutbot.utils import check_file_integrity

        result = check_file_integrity('/nonexistent/path/to/file.bin', 'anyhash')
        assert result is False

    def test_returns_true_when_expected_hash_is_none(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'file.bin'
        f.write_bytes(b'some content')

        result = check_file_integrity(str(f), None)
        assert result is True

    def test_returns_true_when_sha256_matches(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        content = b'wildlife detection payload'
        f = tmp_path / 'model.pt'
        f.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        result = check_file_integrity(str(f), expected)
        assert result is True

    def test_returns_false_when_sha256_does_not_match(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'model.pt'
        f.write_bytes(b'actual content')

        result = check_file_integrity(str(f), 'wrong_hash_value')
        assert result is False

    def test_handles_large_file_in_blocks(self, tmp_path):
        """Exercises the iter(lambda: f.read(4096), b'') chunk loop."""
        from scoutbot.utils import check_file_integrity

        # 16KB forces multiple 4096-byte reads
        content = b'z' * 16_384
        f = tmp_path / 'large.bin'
        f.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert check_file_integrity(str(f), expected) is True

    def test_returns_false_for_empty_file_with_wrong_hash(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'empty.bin'
        f.write_bytes(b'')

        result = check_file_integrity(str(f), 'not_the_sha256_of_empty')
        assert result is False

    def test_returns_true_for_empty_file_with_correct_hash(self, tmp_path):
        from scoutbot.utils import check_file_integrity

        f = tmp_path / 'empty.bin'
        f.write_bytes(b'')

        expected = hashlib.sha256(b'').hexdigest()
        assert check_file_integrity(str(f), expected) is True
```

- [ ] **Step 2: Run to verify tests pass**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: 7 tests pass.

- [ ] **Step 3: Check coverage on `utils.py`**

```bash
uv run pytest tests/test_utils.py --cov=scoutbot/utils --cov-report=term-missing -v
```

Expected: `scoutbot/utils.py` rises from 74% to 95%+. Lines 21-33 should now be covered.

- [ ] **Step 4: Run full fast suite to confirm total coverage**

```bash
uv run pytest -m "not slow" --cov=scoutbot --cov-report=term-missing --random-order-seed=1
```

Expected total: 85%+ overall. The remaining uncovered lines should be:
- `scoutbot/__init__.py` lines 438-468 (`example()`) — network/model; slow test
- `scoutbot/scoutbot.py` line 364 (`if __name__ == '__main__':`) — not coverable in pytest

- [ ] **Step 5: Commit**

```
git add tests/test_utils.py
git commit -m "test: add unit tests for check_file_integrity covering all branches"
```

---

## Documentation Pass

- [ ] Update `docs/roadmap.rst` to add a new "Test coverage" section (status: `in-progress` → `done` after merge) reflecting the target coverage level achieved.
- [ ] Update `CHANGELOG.rst` `Unreleased` section with: "Raised fast-suite coverage from 58% to 85%+ by adding unit tests with mocked SAHI, requests, and coremltools dependencies."
- [ ] Update `docs/development.rst` Testing section to document the testing philosophy for contributors: new tests must cover meaningful branches (exception handlers, guard clauses, behavioral forks), not just add lines to a coverage metric. Note the mock pattern used throughout: `unittest.mock.patch` for network/model dependencies, `tmp_path` for filesystem, `CliRunner` for CLI. Reference the "no `@pytest.mark.slow` unless the test hits network or loads a real model file" rule.
- [ ] Update `docs/roadmap.rst` Quality workflow — v2 scope section to note that coverage enforcement (`--cov-fail-under`) should be gated on meaningful branch coverage, not raw line count, when it is added in v2.

---

## V2 Scope (deferred)

- Add `example()` slow-test to `@pytest.mark.slow` suite (already stubbed in Task 2 as `TestExampleSlow`)
- Branch coverage (`--cov-branch`) once line coverage stabilizes
- Coverage enforcement (`--cov-fail-under=85`) in `testing.yml`
- Expand test matrix for `batch()` with `show_progress=True` path to cover tqdm postfix lines 246-251
