# Model Store Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the default model catalog and fetch/export plumbing out of `scoutbot/loc/__init__.py`
into a dedicated model-store module, while keeping `scoutbot.loc` as a compatibility wrapper.

**Architecture:** `scoutbot.model_store` becomes the source of truth for `DEFAULT_CONFIG`,
`CONFIGS`, `fetch()`, and `export_coreml()`. `scoutbot.loc` stops owning policy and becomes a thin
re-export layer so existing callers keep working during the registry/model-store transition. The
docs and roadmap should point at the new module so future model work does not keep landing in the
inference path.

**Dependency on the registry plan:** This refactor must land before
`docs/superpowers/plans/2026-05-09-scoutbot-registry.md` is executed. That registry plan currently
targets `scoutbot/loc/__init__.py` for the offline-first `fetch()` rewrite; after this refactor,
that task must be updated to target `scoutbot/model_store.py` instead. The compatibility wrapper in
`scoutbot.loc` is intentionally thin and should not become the new implementation target.

**Tech Stack:** Python stdlib, `requests`, `tqdm`, `uv`, pytest, Sphinx docs.

## Global Constraints

- Use `uv run` for all Python and test commands
- Do NOT run `pre-commit run`
- Do NOT run `git add` or `git commit`; the user handles staging and commits
- Keep the change narrow: this plan is a module-boundary refactor, not a registry redesign

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Create | `scoutbot/model_store.py` | Own the default catalog, cache path, MD5 helper, `fetch()`, and `export_coreml()` |
| Modify | `scoutbot/loc/__init__.py` | Turn into a compatibility wrapper that re-exports the model-store API |
| Create | `tests/test_model_store.py` | Assert the new module owns the catalog and that `loc` re-exports it |
| Modify | `tests/test_loc.py` | Keep existing fetch/export tests aligned with the wrapper surface and private helper re-export |
| Modify | `docs/development.rst` | Point model-extension guidance at `scoutbot.model_store` instead of `scoutbot/loc/__init__.py` |
| Modify | `docs/security.rst` | Update the model-download policy wording to cite `scoutbot.model_store.CONFIGS` |
| Modify | `docs/coreml.rst` | Update the Core ML config-location wording to cite the new module |
| Modify | `docs/models.rst` | Update the model-catalog references and hosted-model notes |
| Modify | `docs/roadmap.rst` | Record the module move as part of the ongoing model-store / registry cleanup |

---

## Task 1: Lock the boundary with tests

**Files:**
- Create: `tests/test_model_store.py`
- Modify: `tests/test_loc.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_store.py
from scoutbot import loc
from scoutbot import model_store


def test_model_store_owns_catalog():
    assert model_store.DEFAULT_CONFIG == 'v3'
    assert 'v3' in model_store.CONFIGS
    assert 'v3-cls' in model_store.CONFIGS
    assert 'v3-coreml' in model_store.CONFIGS
    assert 'v3-cls-coreml' in model_store.CONFIGS
    assert model_store.CONFIGS['v3-coreml']['backend'] == 'coreml'
    assert model_store.CONFIGS['v3-coreml']['source'] == 'v3'
    assert model_store.CONFIGS[None] is model_store.CONFIGS[model_store.DEFAULT_CONFIG]


def test_loc_reexports_model_store_api():
    assert loc.DEFAULT_CONFIG == model_store.DEFAULT_CONFIG
    assert loc.CONFIGS is model_store.CONFIGS
    assert loc.fetch is model_store.fetch
    assert loc.export_coreml is model_store.export_coreml
    assert loc._md5 is model_store._md5
```

- [ ] **Step 2: Run the new test file**

Run: `uv run pytest tests/test_model_store.py -v`

Expected: `ModuleNotFoundError: No module named 'scoutbot.model_store'`

- [ ] **Step 3: Verify the existing loc tests still describe the compatibility surface**

```python
# tests/test_loc.py
class TestLocConfig:
    def test_configs_exist(self):
        from scoutbot import loc
        from scoutbot import model_store

        assert loc.CONFIGS is model_store.CONFIGS
        assert loc.DEFAULT_CONFIG == model_store.DEFAULT_CONFIG
        assert 'v3' in loc.CONFIGS
        assert 'v3-cls' in loc.CONFIGS
        assert 'v3-coreml' in loc.CONFIGS
        assert 'v3-cls-coreml' in loc.CONFIGS
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run pytest tests/test_model_store.py tests/test_loc.py -v -k "model_store or configs_exist"`

Expected: both tests fail for the same reason until the new module exists.

---

## Task 2: Move the model store out of `loc`

**Files:**
- Create: `scoutbot/model_store.py`
- Modify: `scoutbot/loc/__init__.py`

- [ ] **Step 1: Create the new model-store module**

```python
# scoutbot/model_store.py
"""
Model catalog, cache path, and fetch/export plumbing for ScoutBot model artifacts.
"""

import hashlib
import os
import shutil
from pathlib import Path
from time import perf_counter_ns

import requests
from tqdm import tqdm

from scoutbot import log
from scoutbot._observability import emit

CACHE_DIR = Path.home() / '.cache' / 'scoutbot'
DEFAULT_CONFIG = os.getenv('LOC_CONFIG', os.getenv('CONFIG', 'v3')).strip().lower()

CONFIGS = {
    'v3': {
        'name': 'yolov8-onecls.kaza.pt',
        'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt',
        'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5',
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    'v3-cls': {
        'name': 'yolov8-cls.kaza.pt',
        'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt',
        'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5',
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    'v3-coreml': {
        'name': 'yolov8-onecls.kaza.mlpackage',
        'backend': 'coreml',
        'source': 'v3',
        'compute_units_requested': 'all',
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    'v3-cls-coreml': {
        'name': 'yolov8-cls.kaza.mlpackage',
        'backend': 'coreml',
        'source': 'v3-cls',
        'compute_units_requested': 'all',
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
}
CONFIGS[None] = CONFIGS[DEFAULT_CONFIG]  # type: ignore[index]


def _md5(filepath: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
```

Move the current `fetch()` and `export_coreml()` bodies from `scoutbot/loc/__init__.py` into this
module unchanged. The only behavioral change in this task is the import boundary: the catalog and
fetch/export logic now live in `scoutbot.model_store`.

Add a short module comment noting that `scoutbot/__init__.py` must continue to define `log` before
importing `loc` / `model_store`, because this module imports `log` from the package namespace.

- [ ] **Step 2: Replace `scoutbot.loc` with a compatibility wrapper**

```python
# scoutbot/loc/__init__.py
"""
Compatibility wrapper for ScoutBot model-store configuration and fetching.
"""

from scoutbot.model_store import CACHE_DIR
from scoutbot.model_store import CONFIGS
from scoutbot.model_store import DEFAULT_CONFIG
from scoutbot.model_store import _md5
from scoutbot.model_store import export_coreml
from scoutbot.model_store import fetch

__all__ = [
    'CACHE_DIR',
    'CONFIGS',
    'DEFAULT_CONFIG',
    '_md5',
    'export_coreml',
    'fetch',
]
```

- [ ] **Step 3: Run the focused tests**

Run: `uv run pytest tests/test_model_store.py tests/test_loc.py -v`

Expected: the tests pass once the module move is complete and the wrapper re-exports the same API.

---

## Task 3: Update the docs and roadmap

**Files:**
- Modify: `docs/development.rst`
- Modify: `docs/security.rst`
- Modify: `docs/coreml.rst`
- Modify: `docs/models.rst`
- Modify: `docs/roadmap.rst`

- [ ] **Step 1: Update the model-extension guidance in `docs/development.rst`**

```rst
The current public model-extension API is not mature. Model configs live in
``scoutbot.model_store`` today, and future registry work is tracked in
``docs/superpowers/plans/2026-05-09-scoutbot-registry.md`` and :doc:`roadmap`.
```

Replace the existing paragraph under ``Model contributors and integrators`` that currently says
model configs live in ``scoutbot/loc/__init__.py``.

- [ ] **Step 2: Update the security page to cite the new source of truth**

```rst
Hosted model artifacts are fetched from the URLs configured in
``scoutbot.model_store.CONFIGS``. Current PyTorch-backed models are hosted on Azure Blob Storage:
```

Replace the first paragraph under ``Model download policy`` in `docs/security.rst`.

- [ ] **Step 3: Update the model docs and Core ML docs to cite `scoutbot.model_store`**

```rst
Two Core ML configs are registered in ``scoutbot.model_store`` alongside ``v3`` / ``v3-cls``:
```

Replace the opening paragraph under ``Configs`` in `docs/coreml.rst`.

```rst
The current models were trained on KAZA aerial survey data. New model integration is planned but
not yet a mature public extension API. Future registry work is tracked in
``docs/superpowers/plans/2026-05-09-scoutbot-registry.md`` and in :doc:`roadmap`.
```

Replace the closing paragraph under ``Model provenance and future integrations`` in `docs/models.rst`.

- [ ] **Step 4: Update the roadmap note for the model-store move**

```rst
Verify MD5s, update ``scoutbot/model_store.py`` ``CONFIGS`` with the new hashes and URLs,
and re-upload the artifacts to Azure.
```

Replace the sentence in ``Model re-export / dill removal`` that currently says to update
``scoutbot/loc/__init__.py`` ``CONFIGS``.

```rst
``scoutbot/model_store.py`` and new model work should include artifact provenance, integrity
verification, config schema tests, CLI choice tests, and output compatibility checks.
```

Replace the paragraph under ``Model registry / model integration`` that currently mentions
``scoutbot/loc/__init__.py`` as the place where model configs remain internal implementation
details.

- [ ] **Step 5: Re-read the touched docs for consistency**

Confirm that all references now point at `scoutbot.model_store` for the catalog and that `scoutbot.loc`
is described only as the compatibility wrapper.

---

## Task 4: Final verification

**Files:**
- Re-read: `scoutbot/model_store.py`, `scoutbot/loc/__init__.py`, `tests/test_model_store.py`,
  `tests/test_loc.py`, `docs/development.rst`, `docs/security.rst`, `docs/coreml.rst`,
  `docs/models.rst`, `docs/roadmap.rst`

- [ ] Run the targeted test slice: `uv run pytest tests/test_model_store.py tests/test_loc.py -q`
- [ ] Run the fast suite: `uv run pytest -m "not slow" -q`
- [ ] Run the repo quality gates: `uv run ruff format --check .`, `uv run ruff check .`,
      `uv run mypy scoutbot tools/visualise.py`, and
      `uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/`
- [ ] Pause for review with a short summary of the new model-store boundary and the compatibility
      wrapper
