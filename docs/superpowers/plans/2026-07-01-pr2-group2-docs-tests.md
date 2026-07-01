# PR #2 Group 2 — Tests, Docs, and Benchmark Guard

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the test and documentation gaps for the Core ML localizer backend before merging PR #2.

**Architecture:** Five independent tasks. Tasks 1–3 produce commits. Task 4 is blocked on hardware confirmation and must be done manually when that information is available. Task 5 is a PR description edit with no code commit. No new modules; all additions go into existing files.

**Tech Stack:** Python/pytest for tests, reStructuredText for `docs/coreml.rst`, Markdown for `README.md`, `gh` CLI for the PR body update.

**Prerequisite:** Complete the Group 1 plan (`2026-07-01-pr2-group1-fixes.md`) before executing this plan. Group 1 adds `export-coreml --help` tests and the Click validation that some assertions here depend on.

## Global Constraints

- Python ≥ 3.8, < 3.13
- Formatter: Brunette. Linting: flake8. Import order: isort. Run `pre-commit run --all-files` before every commit
- New tests must be offline: no network calls, no reads/writes to `~/.cache/scoutbot/`
- Test runner: `pytest` from repo root

---

## File Map

| Action | Path | What changes |
|--------|------|-------------|
| Modify | `tests/test_loc.py` | Task 1 — expand `test_configs_exist`; add `test_coreml_config_schema` |
| Modify | `tests/test_cli.py` | Task 1 — add `test_pipeline_help_includes_coreml_configs`; fix `test_cli_config_choices` docstring |
| Create | `tests/test_benchmark_coreml.py` | Task 2 — unit tests for `_validate_images` |
| Modify | `benchmarks/benchmark_coreml.py` | Task 2 — add `_LFS_PREFIX`, `_VALID_SIGS`, `_validate_images()`; call in `main()` |
| Modify | `docs/coreml.rst` | Task 3 — add Git LFS note before benchmark block; add Torch version caveat to Parity section |
| Modify | `README.md` | Task 3 — add Core ML section; update CLI Options table |
| Modify | `docs/coreml.rst` | Task 4 (BLOCKED) — expand benchmark table with M4 numbers, labelled by hardware |
| — | PR #2 body | Task 5 — update test plan to use `uv sync` via `gh pr edit` |

---

## Task 1: Add missing offline tests for CoreML configs

**Files:**
- Modify: `tests/test_loc.py`
- Modify: `tests/test_cli.py`

**Background:** `test_configs_exist` only asserts the two legacy PT configs. `test_config_has_required_keys` hardcodes `url`/`md5_url` as required — CoreML configs intentionally omit those and have `backend`/`source` instead, but no test validates that schema. In `test_cli.py`, `test_cli_config_choices` has a stale docstring ("only accepts v3 or v3-cls") and no assertion that CoreML configs appear as valid choices in the CLI help.

These tests cover already-implemented behaviour, so they will **PASS on first run** — that is expected and correct. They serve as regression guards against future accidental removal or schema changes.

- [ ] **Step 1: Expand `test_configs_exist` in `tests/test_loc.py`**

Find the existing `test_configs_exist` method inside `TestLocConfig` and add two lines:

```python
    def test_configs_exist(self):
        """Test that v3, v3-cls, v3-coreml, and v3-cls-coreml configs exist."""
        from scoutbot import loc

        assert 'v3' in loc.CONFIGS
        assert 'v3-cls' in loc.CONFIGS
        assert 'v3-coreml' in loc.CONFIGS        # ← add
        assert 'v3-cls-coreml' in loc.CONFIGS    # ← add
```

- [ ] **Step 2: Add `test_coreml_config_schema` to `TestLocConfig`**

Append inside the `TestLocConfig` class immediately after `test_configs_exist`:

```python
    def test_coreml_config_schema(self):
        """CoreML configs must have backend/source/name keys; not url/md5_url."""
        from scoutbot import loc

        for name in ['v3-coreml', 'v3-cls-coreml']:
            cfg = loc.CONFIGS[name]

            # CoreML-specific required keys
            assert cfg.get('backend') == 'coreml', f'{name} must have backend=coreml'
            assert 'source' in cfg, f'{name} must have a source key'
            assert cfg['source'] in loc.CONFIGS, (
                f"{name} source '{cfg['source']}' must itself be a known config"
            )
            assert 'name' in cfg, f'{name} must have a name key'
            assert cfg['name'].endswith('.mlpackage'), (
                f"{name} bundle name must end in .mlpackage, got: {cfg['name']}"
            )

            # Shared inference keys must be present
            for key in ('thresh', 'slice_height', 'slice_width',
                        'overlap_height_ratio', 'overlap_width_ratio',
                        'perform_standard_pred', 'postprocess_class_agnostic'):
                assert key in cfg, f'{name} missing shared key: {key}'

            # CoreML configs must NOT have download keys (no remote bundle)
            assert 'url' not in cfg, f'{name} must not have a url key'
            assert 'md5_url' not in cfg, f'{name} must not have a md5_url key'
```

- [ ] **Step 3: Run the two loc tests to confirm they pass**

```bash
pytest tests/test_loc.py::TestLocConfig::test_configs_exist tests/test_loc.py::TestLocConfig::test_coreml_config_schema -v
```

Expected: both **PASS** (these document existing behaviour).

- [ ] **Step 4: Fix `test_cli_config_choices` docstring and add a coreml assertion**

In `tests/test_cli.py`, inside `TestCLI`, update `test_cli_config_choices` and add a new test after it:

```python
    def test_cli_config_choices(self):
        """Invalid config names are rejected; valid set includes v3-coreml variants."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()

        # Invalid config should fail
        result = runner.invoke(cli, ['pipeline', '--config', 'mvp', 'test.jpg'])
        assert result.exit_code != 0

        result = runner.invoke(cli, ['pipeline', '--config', 'phase1', 'test.jpg'])
        assert result.exit_code != 0

    def test_pipeline_help_includes_coreml_configs(self):
        """pipeline --help must list v3-coreml and v3-cls-coreml as valid --config choices."""
        from scoutbot.scoutbot import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['pipeline', '--help'])

        assert result.exit_code == 0
        assert 'v3-coreml' in result.output
        assert 'v3-cls-coreml' in result.output
```

- [ ] **Step 5: Run the CLI tests to confirm they pass**

```bash
pytest tests/test_cli.py::TestCLI::test_cli_config_choices tests/test_cli.py::TestCLI::test_pipeline_help_includes_coreml_configs -v
```

Expected: both **PASS**.

- [ ] **Step 6: Run full suite and linter**

```bash
pytest
pre-commit run --all-files
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_loc.py tests/test_cli.py
git commit -m "test: add offline CoreML config schema and CLI choice regression tests"
```

---

## Task 2: Add image validation guard to benchmark script

**Files:**
- Create: `tests/test_benchmark_coreml.py`
- Modify: `benchmarks/benchmark_coreml.py`

**Background:** When `examples/` images are Git LFS pointer stubs (not yet pulled), the benchmark discovers them (they match `*.jpg`), loads models, then crashes deep inside SAHI with a confusing PIL/skimage error. The guard reads the first few bytes of each image before any model is loaded and raises a clear `SystemExit` with remediation instructions. It also catches non-image files with `.jpg` extension. No PIL or external library needed — just magic bytes.

- [ ] **Step 1: Write failing tests**

Create `tests/test_benchmark_coreml.py`:

```python
# -*- coding: utf-8 -*-
"""Tests for benchmarks/benchmark_coreml.py helpers (no model/network needed)."""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_benchmark():
    """Import _validate_images from the benchmark script without executing main()."""
    bench = Path(__file__).parent.parent / 'benchmarks' / 'benchmark_coreml.py'
    spec = importlib.util.spec_from_file_location('benchmark_coreml', bench)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['benchmark_coreml'] = mod
    spec.loader.exec_module(mod)
    return mod


class TestValidateImages:
    """Unit tests for _validate_images() in the benchmark script."""

    @pytest.fixture(autouse=True)
    def benchmark_mod(self):
        self.mod = _load_benchmark()

    def test_raises_on_lfs_stub(self, tmp_path):
        """Git LFS pointer stubs must produce a clear SystemExit with git lfs pull hint."""
        stub = tmp_path / 'lfs.jpg'
        stub.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3391180\n')

        with pytest.raises(SystemExit, match='Git LFS pointer'):
            self.mod._validate_images([stub])

    def test_raises_on_non_image_file(self, tmp_path):
        """Files with a .jpg extension that are not images must raise SystemExit."""
        bad = tmp_path / 'notanimage.jpg'
        bad.write_bytes(b'This is just a text file, not an image at all.\n')

        with pytest.raises(SystemExit, match='not valid JPEG/PNG'):
            self.mod._validate_images([bad])

    def test_passes_for_valid_jpeg(self, tmp_path):
        """Files starting with JPEG magic bytes must pass silently."""
        jpg = tmp_path / 'valid.jpg'
        # Minimal JPEG header: SOI marker (FF D8) + APP0 marker (FF E0)
        jpg.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b'\x00' * 64)

        self.mod._validate_images([jpg])  # must not raise

    def test_passes_for_valid_png(self, tmp_path):
        """Files starting with PNG magic bytes must pass silently."""
        png = tmp_path / 'valid.png'
        png.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 64)

        self.mod._validate_images([png])  # must not raise

    def test_lfs_error_mentions_git_commands(self, tmp_path):
        """The LFS error message must include the remediation commands."""
        stub = tmp_path / 'lfs.jpg'
        stub.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')

        with pytest.raises(SystemExit) as exc_info:
            self.mod._validate_images([stub])

        message = str(exc_info.value)
        assert 'git lfs pull' in message

    def test_reports_multiple_stubs(self, tmp_path):
        """When multiple stubs are found the count must appear in the error message."""
        for i in range(3):
            p = tmp_path / f'stub{i}.jpg'
            p.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')

        with pytest.raises(SystemExit) as exc_info:
            self.mod._validate_images(list(tmp_path.glob('*.jpg')))

        message = str(exc_info.value)
        assert '3' in message
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_benchmark_coreml.py -v
```

Expected: all tests **FAIL** with `AttributeError: module 'benchmark_coreml' has no attribute '_validate_images'`.

- [ ] **Step 3: Add the constants and function to `benchmarks/benchmark_coreml.py`**

After the existing module-level constants (after `MATCH_GATE_PX = 16.0`, before `def _parity`), add:

```python
_LFS_PREFIX = b'version https://git-lfs.github.com'
_VALID_SIGS = (b'\xff\xd8\xff', b'\x89PNG')  # JPEG and PNG magic bytes


def _validate_images(images):
    """Fail fast with a clear message if images are Git LFS stubs or wrong type."""
    stubs, bad = [], []
    for p in images:
        header = p.read_bytes()[: max(len(_LFS_PREFIX), 8)]
        if header[: len(_LFS_PREFIX)] == _LFS_PREFIX:
            stubs.append(p)
        elif not any(header[: len(sig)] == sig for sig in _VALID_SIGS):
            bad.append(p)
    if stubs:
        names = ', '.join(p.name for p in stubs[:3])
        extra = f' (+{len(stubs) - 3} more)' if len(stubs) > 3 else ''
        raise SystemExit(
            f'{len(stubs)} image(s) are Git LFS pointer stubs ({names}{extra}).\n'
            f'Pull real image data:\n'
            f'  brew install git-lfs   # once per machine\n'
            f'  git lfs install        # once per clone\n'
            f'  git lfs pull'
        )
    if bad:
        names = ', '.join(p.name for p in bad[:3])
        extra = f' (+{len(bad) - 3} more)' if len(bad) > 3 else ''
        raise SystemExit(
            f'{len(bad)} file(s) are not valid JPEG/PNG images: {names}{extra}'
        )
```

- [ ] **Step 4: Call `_validate_images` in `main()`**

Find the `if not images:` guard in `main()` and add one line after it:

```python
    if not images:
        raise SystemExit(f'No images found in {image_dir}')
    _validate_images(images)    # ← add this line
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_benchmark_coreml.py -v
```

Expected: all 6 tests **PASS**.

- [ ] **Step 6: Run full suite and linter**

```bash
pytest
pre-commit run --all-files
```

- [ ] **Step 7: Commit**

```bash
git add benchmarks/benchmark_coreml.py tests/test_benchmark_coreml.py
git commit -m "feat: add image validation guard to benchmark (clear error for LFS stubs and non-images)"
```

---

## Task 3: Documentation updates

**Files:**
- Modify: `docs/coreml.rst`
- Modify: `README.md`

**No tests for this task.** Run the linter and inspect the rendered output.

**Changes:**
- `docs/coreml.rst`: add Git LFS prerequisite note before the benchmark block; add Torch/coremltools version caveat to the Parity section
- `README.md`: add Core ML section; update CLI Options table to include CoreML configs

### 3a — `docs/coreml.rst`: Git LFS note

- [ ] **Step 1: Add the LFS note before the benchmark usage block**

Find line 91 (`uv run python benchmarks/benchmark_coreml.py`) and insert this block **before** it (after the sentence ending `PyTorch-CPU reference::`):

```rst
.. note::

   ``examples/`` images are stored via Git LFS. Before running the benchmark::

       brew install git-lfs   # once per machine
       git lfs install        # once per clone
       git lfs pull           # downloads the actual images (~23 MB)

   Or point ``--images`` at your own directory of aerial survey JPEGs.

```

The section should then read:

```rst
``benchmarks/benchmark_coreml.py`` runs the full SAHI-tiled pipeline over the images in
``examples/`` for each variant and reports wall-clock per image plus detection parity against the
PyTorch-CPU reference::

.. note::

   ``examples/`` images are stored via Git LFS. Before running the benchmark::

       brew install git-lfs   # once per machine
       git lfs install        # once per clone
       git lfs pull           # downloads the actual images (~23 MB)

   Or point ``--images`` at your own directory of aerial survey JPEGs.

    uv run python benchmarks/benchmark_coreml.py
    uv run python benchmarks/benchmark_coreml.py --family cls
```

### 3b — `docs/coreml.rst`: Torch version caveat

- [ ] **Step 2: Add Torch/coremltools version caveat to Parity and caveats section**

The "Parity and caveats" section ends at line 150. Append a new bullet after the last existing bullet (`Re-export after changing the slice size`):

```rst
* **Torch / coremltools version pairing.** The export and inference paths have been
  validated with Torch 2.7.0 (the most recent version tested by coremltools at writing).
  Torch 2.11.0 produces a runtime warning but inference results are correct in practice.
  If you see unexpected export failures or accuracy regressions, check whether a newer
  ``coremltools`` release supports your installed Torch version before filing a bug.
```

### 3c — `README.md`: Core ML section

- [ ] **Step 3: Add a Core ML subsection under `## How to Run`**

Find the `### CLI Options` table (around line 103). Insert a new subsection **before** it:

```markdown
### CLI — Core ML (Apple Silicon, macOS)

On M-series Macs, install the `coreml` extra and use the `v3-coreml` or `v3-cls-coreml` config to route inference through Apple's Neural Engine (~7–13× faster than CPU):

```bash
uv sync --extra coreml

scoutbot pipeline image.jpg --config v3-coreml
scoutbot batch *.jpg --config v3-coreml --no-progress
```

The `.mlpackage` bundle is exported on first use (~10–30 s) and cached at `~/.cache/scoutbot/`. See [docs/coreml.rst](docs/coreml.rst) for benchmarks and full details.

```

### 3d — `README.md`: update CLI Options table

- [ ] **Step 4: Update the `--config` row in the CLI Options table**

Find the table row:

```markdown
| `--config` | `v3`, `v3-cls` | `v3` |
```

Replace with:

```markdown
| `--config` | `v3`, `v3-cls`, `v3-coreml`, `v3-cls-coreml` | `v3` |
```

- [ ] **Step 5: Run linter**

```bash
pre-commit run --all-files
```

- [ ] **Step 6: Commit**

```bash
git add docs/coreml.rst README.md
git commit -m "docs: add LFS prerequisite, Torch version caveat, Core ML README section"
```

---

## Task 4: Expand benchmark table with M4 results (BLOCKED)

**Files:**
- Modify: `docs/coreml.rst:100–131`

**Blocked on:** Kirk to confirm the M2 chip variant and RAM for the existing numbers in the table. Current table has no hardware label.

**When unblocked — what to do:**

Replace the existing `.. list-table::` block (lines 100–130) with two labelled sub-tables, one per hardware configuration:

```rst
Representative results (7 images, warm-up excluded):

**Apple M2 / [RAM TBD]**  *(Torch [version TBD], ultralytics 8.2.103)*

.. list-table::
   :header-rows: 1

   * - Family
     - Variant
     - ms / image
     - Speedup vs CPU
   * - onecls
     - pt-cpu
     - 29443
     - 1.00×
   * - onecls
     - pt-mps
     - 5049
     - 5.83×
   * - onecls
     - **coreml**
     - **2705**
     - **10.88×**
   * - cls
     - pt-cpu
     - 17112
     - 1.00×
   * - cls
     - pt-mps
     - 4834
     - 3.54×
   * - cls
     - **coreml**
     - **2696**
     - **6.35×**

**Apple M4 / 48 GB**  *(Torch 2.11.0, ultralytics 8.2.103)*

.. list-table::
   :header-rows: 1

   * - Family
     - Variant
     - ms / image
     - Speedup vs CPU
   * - onecls
     - pt-cpu
     - 27178
     - 1.00×
   * - onecls
     - pt-mps
     - 3204
     - 8.48×
   * - onecls
     - **coreml**
     - **2114**
     - **12.86×**
   * - cls
     - pt-cpu
     - 14368
     - 1.00×
   * - cls
     - pt-mps
     - 3201
     - 4.49×
   * - cls
     - **coreml**
     - **2049**
     - **7.01×**
```

Also update the parity notes to cover both runs:

```rst
Parity (M4 run): onecls worst 0.52 px / 0.0074 conf / 5 unmatched; cls worst 0.85 px / 0.0200 conf / 0 unmatched.
```

And add a note about M4 parallelisation potential:

```rst
.. note::

   The M4's larger unified memory and additional ANE cores may support batched or concurrent
   tile inference not captured here. The current pipeline runs one 512×512 tile at a time.
   A parallel CoreML path is a planned future investigation.
```

Commit when ready:

```bash
git add docs/coreml.rst
git commit -m "docs: add M4 benchmark results alongside M2 baseline; label by hardware"
```

---

## Task 5: Update PR body (no code commit)

**File:** PR #2 description on GitHub

- [ ] **Step 1: Fetch the current PR body for reference**

```bash
gh pr view 2 --json body -q .body
```

- [ ] **Step 2: Edit the PR with an updated test plan**

Replace the test plan checklist in the PR body. The key change is step 1 (`pip install` → `uv sync`) and adding a git lfs step:

```bash
gh pr edit 2 --body "$(cat <<'EOF'
## Summary

- Adds \`v3-coreml\` and \`v3-cls-coreml\` configs in \`loc/__init__.py\` that export \`.mlpackage\` bundles on demand from the existing \`.pt\` models (FP16, static 512×512 input, NMS disabled so SAHI handles it)
- Extends \`tile_batched/main.py\` to detect \`.mlpackage\` paths, read the baked-in input size via \`coremltools\`, and force \`device='cpu'\` so Ultralytics/CoreML route inference to the Apple Neural Engine rather than a CUDA device
- CLI \`--config\` choices are now derived from \`loc.CONFIGS\` dynamically, so new configs appear automatically
- Adds \`scoutbot export-coreml\` CLI command and \`--no-progress\` flag to \`scoutbot batch\`
- Adds \`s/img\` timing to the batch progress bar
- Adds \`coremltools\` as an optional macOS extra (\`uv sync --extra coreml\`) and \`scikit-image\` as a runtime dep
- Includes a benchmark script and docs page for the CoreML backend

## Test plan

- [ ] \`uv sync --extra coreml\` on macOS with M-series chip
- [ ] \`git lfs pull\` to download example images for benchmarking
- [ ] \`scoutbot export-coreml --config v3-coreml\` — exports bundle to \`~/.cache/scoutbot/\`
- [ ] \`scoutbot pipeline <image> --config v3-coreml\` — runs inference, detections match \`v3\` output within tolerance
- [ ] \`scoutbot batch <images> --config v3-coreml --no-progress\` — processes without progress bar, shows \`s/img\` when enabled
- [ ] \`uv run python benchmarks/benchmark_coreml.py\` — all variants complete, CoreML fastest
- [ ] \`pytest\` — existing test suite passes unchanged
- [ ] \`pre-commit run --all-files\` — linting clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- 2a (offline config tests) ✓ Task 1
- 2b (PR body uv) ✓ Task 5
- 2c (LFS guard code + doc note) ✓ Tasks 2 + 3
- 2e (Torch version caveat) ✓ Task 3
- 2f (dual-hardware table) ✓ Task 4 (blocked, content ready)
- 2g (README Core ML section) ✓ Task 3

**Placeholder scan:** No TBD except in Task 4 (blocked by design — M2 spec must come from Kirk). All other steps have complete code and exact commands.

**Type/name consistency:** `_validate_images`, `_LFS_PREFIX`, `_VALID_SIGS` used consistently between the implementation (Task 2 Step 3) and the tests (Task 2 Step 1).

**Ordering notes:**
- Task 4 is explicitly blocked and must not be executed until M2 chip/RAM is confirmed.
- Tasks 1, 2, 3, 5 are fully independent of each other and can run in any order after Group 1 is complete.
- Task 5 (`gh pr edit`) should be done last so the test plan reflects everything merged in Groups 1 + 2.
