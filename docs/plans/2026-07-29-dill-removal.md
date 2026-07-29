# Dill Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the runtime `dill` dependency by re-serializing the two bundled KAZA `.pt` models with the current stack, pinning their MD5s in code, and re-hosting them GCS-first.

**Architecture:** `dill` is a runtime dependency only because the two KAZA `.pt` files were serialized by an old Ultralytics that embedded `dill` pickle references. Re-serialize each with `YOLO(old).save(new)` (identical weights, dill-free pickle), re-host under a neutral build-date filename, pin the new MD5 in `CONFIGS`, teach `fetch()` to treat that pin as authoritative (auto-invalidating stale caches), then drop `dill`. Weights are unchanged, so inference output is unchanged.

**Tech Stack:** Python, `uv`, Ultralytics YOLOv8, SAHI, PyTorch, pytest.

## Global Constraints

- Runtime: **always** `uv run` — never bare `python`/`pytest`/`ruff`/`mypy`.
- **No git in this plan.** Do not run `git add`/`commit`/`push`. Each task ends at a review checkpoint with a suggested commit message; the user stages and commits.
- **Pause after each task** for human review before starting the next (repo rule; overrides any skill's continuous-execution default).
- Test-first: write the failing test and run it red before implementing.
- Uploading artifacts to GCS/Azure is an **outward-facing action performed by the user**, not the agent. The agent produces and verifies artifacts locally and hands off an upload checklist (see the Upload Hand-off gate).
- Artifact naming: neutral build-date tag `20260729` in `YYYYMMDD` form on the filename (the re-serialization/build date, identical on GCS and Azure): `yolov8-onecls.kaza.20260729.pt`, `yolov8-cls.kaza.20260729.pt`.
- Source order after this change: **GCS primary, Azure fallback** (swapped from today's Azure-first).
- Scope is the two PyTorch configs (`v3`, `v3-cls`). Core ML configs (`v3-coreml`, `v3-cls-coreml`) are out of scope: `.mlpackage` bundles never needed `dill` and regenerate on demand from the source `.pt`; they carry no `md5` and their behavior must not change.
- Parity comparison is **CPU-only** (device `cpu`) for exact equality; any MPS/CUDA comparison uses a strict tolerance instead.

---

## File Map

- `scoutbot/model_store.py` — `CONFIGS` catalog + `fetch()`. Add an `md5` field per `.pt` config; make the registry short-circuit and post-download path treat the pinned `md5` as authoritative. Reorder `sources` to GCS-first and repoint to the date-tagged filenames.
- `tests/test_model_store.py` — unit tests for the pinned-MD5 invalidation logic and CONFIGS shape.
- `pyproject.toml` — remove the `dill>=0.4.0` dependency and its comment.
- `uv.lock` — regenerated after the dependency removal.
- Docs: `docs/models.rst`, `docs/configuration.rst`, `docs/security.rst`, `docs/installation.rst`, `docs/roadmap.rst`, `CHANGELOG.rst`.
- **Not committed:** the re-export is a one-shot procedure run from a `uv run python` heredoc; produced artifacts live in a build directory (`~/scoutbot-model-build/`) and are handed to the user, not added to the repo.

---

## Task 1: Pinned-MD5 cache invalidation in `fetch()`

Add support for an authoritative `md5` pin per config, independent of the actual artifacts (tests inject a synthetic pin). This is pure logic + unit tests and can be fully reviewed before any re-export.

**Files:**
- Modify: `scoutbot/model_store.py` (registry short-circuit at lines 206–229; post-download block after line 252; registration description at line 302)
- Test: `tests/test_model_store.py`

**Interfaces:**
- Consumes: existing `CONFIGS` dict, `_md5(path) -> str`, `fetch(config, observer)`, `_download_from_source(source, local_path) -> tuple[str, bool]`, the `scoutbot.registry` lookup/register API, and `RW_REGISTRY_PATH` / `RW_MODELS_DIR` env vars used by existing tests.
- Produces: `fetch()` honoring `cfg.get('md5')` — a config with a pinned `md5` re-downloads whenever the cached/registered hash differs, and raises if the hosted sidecar hash ≠ the pin. Configs without `md5` (Core ML) are unchanged.

- [ ] **Step 1: Write the failing tests**

First add the `Path` import to the top of `tests/test_model_store.py` (the file does not currently import it, and the first test below uses `Path(result)`):

```python
from pathlib import Path
```

Then add these tests to `tests/test_model_store.py`:

```python
def test_pinned_md5_rejects_stale_cache(isolated_registry):
    """A registered cache whose hash != CONFIGS['md5'] is re-downloaded, not served."""
    from scoutbot import model_store, registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource

    # Stale cached file registered under the OLD hash.
    stale = isolated_registry / 'models' / 'yolov8-onecls.kaza.pt'
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b'old dill-embedded bytes')
    old_md5 = model_store._md5(stale)
    registry.register(
        ModelEntry(
            name='scoutbot-v3', path=str(stale), md5=old_md5, framework='yolov8',
            registered='2026-05-09T00:00:00Z',
            source=RegistrySource(type='fetched', url='https://example.com/old.pt'),
            default_run_config={},
        )
    )

    new_content = b'new dill-free bytes'
    import hashlib
    h = hashlib.md5(usedforsecurity=False); h.update(new_content)
    new_md5 = h.hexdigest()

    md5_resp = MagicMock(); md5_resp.text = new_md5; md5_resp.raise_for_status = MagicMock()
    model_resp = MagicMock()
    model_resp.raise_for_status = MagicMock()
    model_resp.headers = {'content-length': str(len(new_content))}
    model_resp.iter_content = MagicMock(return_value=[new_content])
    model_resp.__enter__ = MagicMock(return_value=model_resp)
    model_resp.__exit__ = MagicMock(return_value=False)

    def fake_get(url, **kwargs):
        return md5_resp if url.endswith('.md5') else model_resp

    with (
        patch('requests.get', side_effect=fake_get),
        patch.object(model_store, 'CACHE_DIR', isolated_registry / 'models'),
        patch.dict(model_store.CONFIGS['v3'], {'md5': new_md5}),
    ):
        result = model_store.fetch(config='v3')

    assert model_store._md5(Path(result)) == new_md5  # got the NEW bytes, not the stale cache


def test_pinned_md5_serves_matching_cache(isolated_registry):
    """When the registered hash equals CONFIGS['md5'], no network call is made."""
    from scoutbot import model_store, registry
    from scoutbot.registry.schema import ModelEntry, RegistrySource

    cached = isolated_registry / 'models' / 'yolov8-onecls.kaza.20260729.pt'
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'dill-free bytes')
    good_md5 = model_store._md5(cached)
    registry.register(
        ModelEntry(
            name='scoutbot-v3', path=str(cached), md5=good_md5, framework='yolov8',
            registered='2026-07-29T00:00:00Z',
            source=RegistrySource(type='fetched', url='https://example.com/new.pt'),
            default_run_config={},
        )
    )

    with (
        patch('requests.get') as mock_get,
        patch.dict(model_store.CONFIGS['v3'], {'md5': good_md5}),
    ):
        result = model_store.fetch(config='v3')
        mock_get.assert_not_called()

    assert result == str(cached)


def test_pinned_md5_raises_on_sidecar_mismatch(isolated_registry):
    """If the hosted sidecar hash != the pinned md5, fetch raises rather than trust it."""
    from scoutbot import model_store

    served = b'served bytes'
    import hashlib
    h = hashlib.md5(usedforsecurity=False); h.update(served)
    served_md5 = h.hexdigest()

    md5_resp = MagicMock(); md5_resp.text = served_md5; md5_resp.raise_for_status = MagicMock()
    model_resp = MagicMock()
    model_resp.raise_for_status = MagicMock()
    model_resp.headers = {'content-length': str(len(served))}
    model_resp.iter_content = MagicMock(return_value=[served])
    model_resp.__enter__ = MagicMock(return_value=model_resp)
    model_resp.__exit__ = MagicMock(return_value=False)

    def fake_get(url, **kwargs):
        return md5_resp if url.endswith('.md5') else model_resp

    with (
        patch('requests.get', side_effect=fake_get),
        patch.object(model_store, 'CACHE_DIR', isolated_registry / 'models'),
        patch.dict(model_store.CONFIGS['v3'], {'md5': 'deadbeef' * 4}),  # != served_md5
    ):
        with pytest.raises(AssertionError, match='[Pp]inned'):
            model_store.fetch(config='v3')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_model_store.py -k "pinned_md5" -v`
Expected: FAIL — the stale-cache test returns the stale path (no re-download), and the mismatch test does not raise, because `fetch()` does not yet read `cfg['md5']`.

- [ ] **Step 3: Make the registry short-circuit honor the pin**

In `scoutbot/model_store.py`, replace the registry short-circuit block (currently lines ~206–211):

```python
    from scoutbot import registry as _registry

    entry = _registry.lookup(registry_name)
    if entry is not None:
        cached_path = Path(entry.path)
        if cached_path.exists() and _md5(cached_path) == entry.md5:
```

with:

```python
    from scoutbot import registry as _registry

    # A pinned md5 in CONFIGS is authoritative: a cache registered under a
    # different hash (e.g. a pre-migration artifact) must be re-fetched, not
    # served, even though it still matches its own registry entry.
    pinned_md5 = cfg.get('md5')
    entry = _registry.lookup(registry_name)
    if entry is not None:
        cached_path = Path(entry.path)
        if (
            cached_path.exists()
            and _md5(cached_path) == entry.md5
            and (pinned_md5 is None or entry.md5 == pinned_md5)
        ):
```

- [ ] **Step 4: Assert the hosted sidecar matches the pin after download**

In `scoutbot/model_store.py`, immediately after the `if used_source is None:` block that raises `RuntimeError` (currently ends at line ~252), insert:

```python
    # Defense in depth: the hosted sidecar hash must equal the pinned md5, so a
    # bad or partial re-upload fails loudly here instead of silently serving
    # mismatched bytes. _download_from_source already verified the downloaded
    # file against remote_md5; this ties remote_md5 to the code-pinned value.
    if pinned_md5 is not None:
        assert remote_md5 == pinned_md5, (
            f'Pinned MD5 mismatch for {cfg["name"]}: hosted sidecar {remote_md5} '
            f'!= pinned {pinned_md5}'
        )
```

- [ ] **Step 5: Align the registry description casing**

While in `fetch()`, change the model-registration description (`scoutbot/model_store.py:302`) from:
```python
            description=f'ScoutBot {config} model (KAZA)',
```
to:
```python
            description=f'Scoutbot {config} model (KAZA)',
```
`scoutbot/model_store.py` is already in scope, so this one-word casing fix rides along and keeps the real registration consistent with the Task 3 priming description. Broader `ScoutBot` capitalization in docstrings/CLAUDE.md stays out of scope — those files are not touched by this plan. No test asserts this string, so no test change is needed.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_model_store.py -k "pinned_md5" -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Run the full fast model-store suite + lint/type**

Run:
```bash
uv run pytest tests/test_model_store.py -v
uv run ruff check scoutbot/model_store.py tests/test_model_store.py
uv run mypy scoutbot
```
Expected: all pass; existing tests still green (the `pinned_md5 is None` branch preserves current behavior for configs without a pin).

- [ ] **Step 8: Review checkpoint**

Pause for human review. Suggested commit message:
```
feat: honor pinned model md5 in fetch() cache invalidation
```

---

## Task 2: Re-export the models and produce upload artifacts

One-shot maintainer procedure (run in this repo's env, which still has `dill`). Produces the two date-tagged `.pt` files, their `.md5` sidecars, an `INDEX.txt`, and the pinned hashes — verified dill-free and inference-identical before anything leaves the machine.

**Files:**
- Produce (not committed): `~/scoutbot-model-build/yolov8-onecls.kaza.20260729.pt` (+ `.md5`), `~/scoutbot-model-build/yolov8-cls.kaza.20260729.pt` (+ `.md5`), `~/scoutbot-model-build/INDEX.txt`
- Reads: current `.pt` via `scoutbot fetch`; `examples/*.jpg`

**Interfaces:**
- Consumes: `scoutbot.pipeline(filepath, batched_detection_model=..., backend_device='cpu')` and `scoutbot.tile_batched.main.Yolov8DetectionModel(model_path=..., device='cpu', confidence_threshold=..., load_at_init=True)`.
- Produces: the two date-tagged artifacts and their MD5 hex strings — consumed by Task 3 (`CONFIGS['md5']`) and the Upload Hand-off.

- [ ] **Step 1: Fetch the current (old) models into the cache**

Run:
```bash
uv run scoutbot fetch --config v3
uv run scoutbot fetch --config v3-cls
```
Expected: both resolve to `~/.cache/scoutbot/yolov8-onecls.kaza.pt` and `~/.cache/scoutbot/yolov8-cls.kaza.pt`.

- [ ] **Step 2: Re-serialize each model dill-free with `model.save()`**

Run (adjust `$HOME` cache path if `SCOUTBOT`/cache env differs):
```bash
mkdir -p ~/scoutbot-model-build
uv run python - <<'PY'
from pathlib import Path
from ultralytics import YOLO

CACHE = Path.home() / '.cache' / 'scoutbot'
OUT = Path.home() / 'scoutbot-model-build'
pairs = [
    (CACHE / 'yolov8-onecls.kaza.pt', OUT / 'yolov8-onecls.kaza.20260729.pt'),
    (CACHE / 'yolov8-cls.kaza.pt',    OUT / 'yolov8-cls.kaza.20260729.pt'),
]
for old, new in pairs:
    YOLO(str(old), task='detect').save(str(new))
    print('re-saved', new)
PY
```
Expected: two `re-saved …` lines; both new files exist.

- [ ] **Step 3: Verify each new file loads with `dill` UNINSTALLED (subprocess)**

Run:
```bash
uv run python - <<'PY'
import sys
sys.modules['dill'] = None  # make `import dill` raise ImportError, simulating absence
from pathlib import Path
from ultralytics import YOLO
OUT = Path.home() / 'scoutbot-model-build'
for name in ('yolov8-onecls.kaza.20260729.pt', 'yolov8-cls.kaza.20260729.pt'):
    YOLO(str(OUT / name), task='detect')  # must NOT raise
    print('loaded without dill:', name)
PY
```
Expected: two `loaded without dill: …` lines and exit 0. A `ModuleNotFoundError: dill` means the re-serialization did not drop the reference — stop and investigate before proceeding.

- [ ] **Step 4: Verify CPU inference parity (old vs new) on the examples corpus**

Run:
```bash
uv run python - <<'PY'
import glob
from pathlib import Path
from scoutbot import pipeline
from scoutbot.tile_batched.main import Yolov8DetectionModel

CACHE = Path.home() / '.cache' / 'scoutbot'
OUT = Path.home() / 'scoutbot-model-build'

def build(path):
    return Yolov8DetectionModel(
        model_path=str(path), device='cpu', confidence_threshold=0.45, load_at_init=True
    )

configs = [
    (CACHE / 'yolov8-onecls.kaza.pt', OUT / 'yolov8-onecls.kaza.20260729.pt'),
    (CACHE / 'yolov8-cls.kaza.pt',    OUT / 'yolov8-cls.kaza.20260729.pt'),
]
images = sorted(glob.glob('examples/*.jpg'))
for old_pt, new_pt in configs:
    old_m, new_m = build(old_pt), build(new_pt)
    for img in images:
        got_old = pipeline(img, batched_detection_model=old_m, backend_device='cpu')
        got_new = pipeline(img, batched_detection_model=new_m, backend_device='cpu')
        assert got_old == got_new, f'PARITY MISMATCH {new_pt.name} on {img}'
    print('parity OK:', new_pt.name)
PY
```
Expected: two `parity OK: …` lines. Any `PARITY MISMATCH` aborts the migration (weights should be identical; a mismatch means `model.save()` altered something — stop and investigate).

- [ ] **Step 5: Write the MD5 sidecars and generate `INDEX.txt` from the computed hashes**

One script computes each MD5, writes the sidecar, and builds `INDEX.txt` from those same values (no hand-edited placeholders):

```bash
uv run python - <<'PY'
from pathlib import Path
from scoutbot import model_store   # reuse the repo's chunked _md5 (low memory, one source of truth)

OUT = Path.home() / 'scoutbot-model-build'
MODELS = [
    ('yolov8-onecls.kaza.20260729.pt', 'v3 single-class KAZA detector'),
    ('yolov8-cls.kaza.20260729.pt',    'v3-cls multi-class KAZA detector'),
]

rows = []
for name, desc in MODELS:
    digest = model_store._md5(OUT / name)
    (OUT / (name + '.md5')).write_text(digest + '\n')
    rows.append((name, digest, desc))
    print(f'{name}  md5={digest}')

lines = ['scoutbot KAZA detection models', '==============================', '']
lines += [f'{name}   md5={digest}   {desc}' for name, digest, desc in rows]
lines += [
    '',
    'Re-serialized 2026-07-29 from the original v3 KAZA weights with the current',
    'Ultralytics/torch stack. Weights are identical to the earlier yolov8-*.kaza.pt',
    'files; only the on-disk serialization changed (the older files embedded a dill',
    'dependency, which these do not).',
    '',
]
(OUT / 'INDEX.txt').write_text('\n'.join(lines))
print('wrote INDEX.txt with computed hashes')
PY
```
Expected: two `…  md5=…` lines and `wrote INDEX.txt with computed hashes`. **Record both hashes** — Task 3 pins them in `CONFIGS` and asserts them exactly.

- [ ] **Step 6: Review checkpoint**

Pause for human review of the produced artifacts and the recorded hashes. No repo changes in this task — nothing to commit. Carry the two MD5 hashes into Task 3.

---

## Task 3: Wire `CONFIGS` and validate locally before upload

Point `CONFIGS` at the new artifacts (GCS-first, date-tagged names, pinned `md5`), then prove dill-removal + parity locally by priming the cache with the Task 2 files and running the slow suite with `dill` blocked — all before anything is hosted.

**Files:**
- Modify: `scoutbot/model_store.py` (`CONFIGS['v3']` lines 27–48, `CONFIGS['v3-cls']` lines 49–70)
- Test: `tests/test_model_store.py`

**Interfaces:**
- Consumes: the two MD5 hashes from Task 2; the pinned-md5 `fetch()` logic from Task 1.
- Produces: `CONFIGS['v3']` and `CONFIGS['v3-cls']` with `name` = date-tagged filename, `sources` GCS-first then Azure, `url`/`md5_url` at the new filenames, and an `md5` key holding the pinned hash.

- [ ] **Step 1: Write the failing CONFIGS-shape test**

Add to `tests/test_model_store.py`:

```python
def test_pt_configs_are_gcs_first_and_pinned():
    from scoutbot import model_store

    # Exact hashes recorded in Task 2 Step 5 — substitute the real values.
    EXPECTED = {
        'v3': ('yolov8-onecls.kaza.20260729.pt', '<onecls-md5>'),
        'v3-cls': ('yolov8-cls.kaza.20260729.pt', '<cls-md5>'),
    }
    for key, (fname, expected_md5) in EXPECTED.items():
        cfg = model_store.CONFIGS[key]
        assert cfg['name'] == fname
        providers = [s['provider'] for s in cfg['sources']]
        assert providers == ['gcs', 'azure']            # GCS primary, Azure fallback
        assert cfg['md5'] == expected_md5                 # exact pinned MD5 (not just shape)
        for s in cfg['sources']:
            assert s['url'].endswith(fname)
            assert s['md5_url'].endswith(fname + '.md5')
```

The `<onecls-md5>` / `<cls-md5>` tokens are the two hashes recorded in Task 2 Step 5; substitute the real 32-char values here and in the `CONFIGS` edits below.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_model_store.py::test_pt_configs_are_gcs_first_and_pinned -v`
Expected: FAIL — current `name` is `yolov8-onecls.kaza.pt`, providers are `['azure', 'gcs']`, and there is no `md5` key.

- [ ] **Step 3: Update `CONFIGS['v3']`**

In `scoutbot/model_store.py`, replace the `v3` entry's `name` + `sources` (lines 28–40) with (substitute the recorded `<onecls-md5>`):

```python
        'name': 'yolov8-onecls.kaza.20260729.pt',
        'md5': '<onecls-md5>',
        'sources': [
            {
                'provider': 'gcs',
                'url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.20260729.pt',
                'md5_url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.20260729.pt.md5',
            },
            {
                'provider': 'azure',
                'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.20260729.pt',
                'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.20260729.pt.md5',
            },
        ],
```

- [ ] **Step 4: Update `CONFIGS['v3-cls']`**

Replace the `v3-cls` entry's `name` + `sources` (lines 50–62) with (substitute `<cls-md5>`):

```python
        'name': 'yolov8-cls.kaza.20260729.pt',
        'md5': '<cls-md5>',
        'sources': [
            {
                'provider': 'gcs',
                'url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.20260729.pt',
                'md5_url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.20260729.pt.md5',
            },
            {
                'provider': 'azure',
                'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.20260729.pt',
                'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.20260729.pt.md5',
            },
        ],
```

- [ ] **Step 5: Update the two existing registry tests for the pin**

Adding a real `md5` to `CONFIGS['v3']` makes the pin authoritative, so the two existing registry tests (which register/serve under a synthetic hash) must set the pin to match or they will fall through to the network. In `tests/test_model_store.py`:

In `test_fetch_uses_registry_when_model_cached`, change:
```python
    with patch('requests.get') as mock_get:
        result = model_store.fetch(config='v3')
        mock_get.assert_not_called()
```
to:
```python
    with (
        patch('requests.get') as mock_get,
        patch.dict(model_store.CONFIGS['v3'], {'md5': md5}),
    ):
        result = model_store.fetch(config='v3')
        mock_get.assert_not_called()
```

In `test_fetch_registers_model_after_download`, change:
```python
    with (
        patch('requests.get', side_effect=mock_get),
        patch('scoutbot.model_store.CACHE_DIR', isolated_registry),
    ):
        model_store.fetch(config='v3')
```
to:
```python
    with (
        patch('requests.get', side_effect=mock_get),
        patch('scoutbot.model_store.CACHE_DIR', isolated_registry),
        patch.dict(model_store.CONFIGS['v3'], {'md5': expected_md5}),
    ):
        model_store.fetch(config='v3')
```

- [ ] **Step 6: Run the CONFIGS test + fast suite**

Run:
```bash
uv run pytest tests/test_model_store.py -v
uv run ruff check scoutbot/model_store.py
uv run mypy scoutbot
```
Expected: all pass, including `test_pt_configs_are_gcs_first_and_pinned` and the two updated registry tests.

- [ ] **Step 7: Prime the cache with the Task 2 artifacts and register them at the pinned hash**

The hosted files do not exist yet, so seed the cache + registry directly from the local build so the slow suite runs offline against the new files:

```bash
uv run python - <<'PY'
import shutil
from datetime import datetime, timezone
from pathlib import Path
from scoutbot import model_store, registry
from scoutbot.registry.schema import ModelEntry, RegistrySource

CACHE = model_store.CACHE_DIR
OUT = Path.home() / 'scoutbot-model-build'
CACHE.mkdir(parents=True, exist_ok=True)
for key in ('v3', 'v3-cls'):
    cfg = model_store.CONFIGS[key]
    src = OUT / cfg['name']
    dst = CACHE / cfg['name']
    shutil.copyfile(src, dst)
    assert model_store._md5(dst) == cfg['md5'], f'seed hash mismatch for {key}'
    registry.register(ModelEntry(
        name=f'scoutbot-{key}', path=str(dst), md5=cfg['md5'], framework='yolov8',
        registered=datetime.now(timezone.utc).isoformat(),
        source=RegistrySource(type='fetched', url=cfg['sources'][0]['url'], provider='gcs'),
        default_run_config={}, description=f'Scoutbot {key} model (KAZA)',
    ))
    print('primed', key, '->', dst)
PY
```
Expected: `primed v3 -> …` and `primed v3-cls -> …`; a hash mismatch here means the CONFIGS `md5` does not match the built file — fix before proceeding.

- [ ] **Step 8: Run the slow suite with `dill` uninstalled (acceptance gate)**

Temporarily remove `dill` from the environment and run the slow model tests against the primed cache. **`dill` is still in `pyproject.toml`/`uv.lock` at this point**, so `uv run` would auto-sync and silently reinstall it — defeating the gate. Use `--no-sync` to run against the just-modified environment:

```bash
uv pip uninstall dill
uv run --no-sync pytest -m slow -v   # --no-sync: skip uv's automatic sync, which would reinstall dill from uv.lock
```
Expected: the slow suite passes (real inference on the re-serialized models, loaded without `dill`). If any model-load test errors with a `dill` import failure, the re-export did not fully drop the reference — stop.

**Restore the env before pausing or reporting — pass or fail.** This step removes `dill` from the working venv; always run the restore below (even when the slow suite fails, or you stop to investigate) so later tasks and interactive work are not left with a broken environment:
```bash
uv sync --extra test
```

- [ ] **Step 9: Review checkpoint**

Pause for human review. Suggested commit message:
```
feat: re-host KAZA models dill-free, gcs-first, md5-pinned
```

---

## Upload Hand-off (gate — user action)

The agent hands the user the Task 2 build directory and this checklist; **the user uploads**.

Upload these exact bytes (do not let any tool re-compress or rename):

- GCS bucket `rebel-pipeline-public-models`, prefix `scoutbot/` — **DONE (uploaded and verified 2026-07-29)**:
  - `yolov8-onecls.kaza.20260729.pt` + `yolov8-onecls.kaza.20260729.pt.md5`
  - `yolov8-cls.kaza.20260729.pt` + `yolov8-cls.kaza.20260729.pt.md5`
  - `INDEX.txt`
- Azure container `localizer` (`scoutmodels`) — **DEFERRED (Azure storage unavailable, no ETA; do NOT block the plan on this — tracked in `docs/roadmap.rst`)**:
  - the same four model/sidecar files (same names) + `INDEX.txt`

Old `yolov8-*.kaza.pt` objects are left in place for rollback.

**Azure deferral.** GCS is the primary source and is fully validated (Task 4 Step 1). Azure is the fallback only. Until the four dated files + `INDEX.txt` are uploaded to `localizer`, the Azure fallback URLs in `CONFIGS` point at objects that do not exist yet, so failover is temporarily non-functional — reduced redundancy, not broken functionality while GCS is healthy. Uploading to Azure and validating failover (Task 4 Step 2) are deferred to the roadmap item *Dill-free model Azure fallback upload + failover validation*.

---

## Task 4: Post-upload real-network validation

Confirm the hosted artifacts resolve GCS-first and match the pinned hashes — from a **fully isolated** cache and registry so a real network download is proven and the Task 3 seed cannot short-circuit it. Azure failover (Step 2) is **deferred** — see the note on Step 2.

**Files:** none modified (validation only).

**Interfaces:**
- Consumes: the live GCS objects and `CONFIGS` from Task 3. (Azure objects are not yet uploaded — see Step 2 deferral.)
- Isolation note: `CACHE_DIR` is a module global (`scoutbot/model_store.py:22`, not env-driven) and the registry default is `~/.rebel-workflow/model-registry.json` (`scoutbot/registry/paths.py`) — **not** under `~/.cache/scoutbot`. Deleting `~/.cache/scoutbot/registry.json` would miss the real registry and false-pass. Isolation therefore overrides `CACHE_DIR` in-process AND points `RW_REGISTRY_PATH`/`RW_MODELS_DIR` at temp paths.

- [ ] **Step 1: Clean-room network fetch of BOTH configs (GCS primary)**

Run:
```bash
uv run python - <<'PY'
import os, tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ['RW_REGISTRY_PATH'] = str(tmp / 'registry.json')
os.environ['RW_MODELS_DIR'] = str(tmp / 'models')

from scoutbot import model_store
model_store.CACHE_DIR = tmp / 'cache'   # isolate the model cache (module global; not env-driven)

for key in ('v3', 'v3-cls'):
    events = []
    path = model_store.fetch(config=key, observer=lambda e: events.append(e))
    evt = [e for e in events if e.get('name') == 'fetch'][-1]
    assert evt['cache_hit'] is False, f'{key}: expected a real download, got a cache hit'
    assert evt.get('provider') == 'gcs', f'{key}: expected gcs, got {evt.get("provider")}'
    assert model_store._md5(Path(path)) == model_store.CONFIGS[key]['md5'], f'{key}: pinned hash mismatch'
    print(f'{key}: downloaded from gcs, pinned hash verified -> {path}')
PY
```
Expected: two lines confirming each config downloaded from **gcs** with the pinned hash verified. A hash assertion failure means the uploaded bytes or sidecar differ from the pin — fix the upload.

- [ ] **Step 2: Confirm Azure failover for BOTH configs — DEFERRED**

> **DEFERRED (2026-07-29): Azure storage is unavailable with no ETA.** Per the decision not to block the plan mid-flight, this step is deferred to the roadmap item *Dill-free model Azure fallback upload + failover validation*. It requires (a) the four dated files + `INDEX.txt` uploaded to the Azure `localizer` container, then (b) running the check below. Until then the Azure fallback in `CONFIGS` points at not-yet-existing objects. **Do not run this step or block Task 5 on it.** When Azure returns, upload the artifacts and run the check below to close the roadmap item.

Run (only after Azure is populated):
```bash
uv run python - <<'PY'
import os, tempfile
from pathlib import Path
from unittest.mock import patch
import requests

tmp = Path(tempfile.mkdtemp())
os.environ['RW_REGISTRY_PATH'] = str(tmp / 'registry.json')
os.environ['RW_MODELS_DIR'] = str(tmp / 'models')

from scoutbot import model_store
model_store.CACHE_DIR = tmp / 'cache'

real_get = requests.get
def flaky(url, **kw):
    if 'storage.googleapis.com' in url:
        raise requests.ConnectionError('simulated GCS outage')
    return real_get(url, **kw)

with patch('requests.get', side_effect=flaky):
    for key in ('v3', 'v3-cls'):
        events = []
        path = model_store.fetch(config=key, observer=lambda e: events.append(e))
        evt = [e for e in events if e.get('name') == 'fetch'][-1]
        assert evt.get('provider') == 'azure', f'{key}: expected azure failover, got {evt.get("provider")}'
        assert model_store._md5(Path(path)) == model_store.CONFIGS[key]['md5'], f'{key}: pinned hash mismatch'
        print(f'{key}: failed over to azure, pinned hash verified -> {path}')
PY
```
Expected: two lines confirming each config failed over to **azure** with the pinned hash verified (a GCS-unavailable warning is logged first).

- [ ] **Step 3: Review checkpoint**

Pause for human review. No repo changes — validation only. Task 4 is complete for the purpose of proceeding to Task 5 once **Step 1 (GCS-primary) passes**; Step 2 (Azure failover) stays open on the roadmap and does **not** gate Task 5.

---

## Task 5: Remove the `dill` dependency

Drop `dill` from the project, re-lock, and confirm the whole suite + audit pass without it.

**Files:**
- Modify: `pyproject.toml` (lines 18–21: the comment + `"dill>=0.4.0",`)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Consumes: the validated hosted artifacts (Task 4).

- [ ] **Step 1: Remove the dependency and its comment**

In `pyproject.toml`, delete these four lines from `[project.dependencies]`:
```toml
    # Required to load KAZA .pt model files serialized with an older Ultralytics
    # version that embedded dill pickle references. Remove once models are re-exported
    # without dill serialization; see docs/roadmap.rst "Model re-export / dill removal".
    "dill>=0.4.0",
```

- [ ] **Step 2: Re-lock and sync**

Run:
```bash
uv lock
uv sync --extra test
```
Expected: `uv.lock` updates; `dill` no longer appears as a project dependency.

- [ ] **Step 3: Confirm `dill` is gone from the synced environment**

Run a portable check (no shell-pipe parsing):
```bash
uv run python -c "import importlib.util; assert importlib.util.find_spec('dill') is None, 'dill still installed'; print('dill absent')"
```
Expected: `dill absent` (nothing else in the graph pulls it in). A non-zero exit means something still depends on `dill` — investigate before proceeding.

- [ ] **Step 4: Full gate without `dill`**

Run the repo's complete CI quality suite:
```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot tools/visualise.py
uv audit --locked --preview-features audit-command
uv run bandit -c pyproject.toml -r scoutbot/ benchmarks/ tools/
uv run pytest -m "not slow"
uv run pytest -m slow
```
Expected: all pass. (Run the `slow` suite on a machine that can do real inference; models load without `dill`.)

- [ ] **Step 5: Review checkpoint**

Pause for human review. Suggested commit message:
```
build: drop runtime dill dependency
```

---

## Task 6: Documentation and roadmap pass

Make user-facing and contributor docs reflect the re-hosted models, the GCS-primary swap, the in-code MD5 pin, and the removed dependency.

**Files:**
- Modify: `docs/models.rst`, `docs/configuration.rst`, `docs/security.rst`, `docs/installation.rst`, `docs/roadmap.rst`, `CHANGELOG.rst`

- [ ] **Step 1: `docs/models.rst`**

Update the model artifact list to the date-tagged filenames (`yolov8-onecls.kaza.20260729.pt`, `yolov8-cls.kaza.20260729.pt`), record the pinned MD5s, state the provenance ("same v3 weights, re-serialized 2026-07-29 to remove the `dill` dependency"), and describe the source order as **GCS primary, Azure fallback**.

Also add a short note that the Scoutbot model collection includes other artifacts — notably `.onnx` exports — which are **not** part of this re-serialization: they are not loaded through the `dill`-affected PyTorch pickle path, keep their existing names/URLs, and are unaffected by this change.

- [ ] **Step 2: `docs/configuration.rst`**

Document the download source order (GCS primary, Azure fallback) as user-facing behavior in the model-download/cache section.

- [ ] **Step 3: `docs/security.rst`**

Note that model integrity is now pinned in code (the `md5` field in `CONFIGS`), a strengthening over the previous sidecar-only trust, and that the trusted source order is GCS-first then Azure. Keep the SHA-256 / signed-manifest direction as future work.

- [ ] **Step 4: `docs/installation.rst`**

Add one line that upgrading clients automatically re-download the re-serialized models on next use (the old cached file is left in place; the pinned hash triggers the refresh). Only if it reads as warranted; otherwise record here that no change was needed.

- [ ] **Step 5: `CHANGELOG.rst`**

Add an Unreleased bullet: `dill` runtime dependency removed; the two KAZA models re-serialized (identical weights) and re-hosted GCS-first under date-tagged filenames; model MD5 now pinned in `CONFIGS`.

- [ ] **Step 6: `docs/roadmap.rst`**

Move *Model re-export / dill removal* (the `~~~~` subsection under "Dependency and maintenance policy") to the **Completed work archive** as a bullet, and note the source-order swap. Remove the now-satisfied 5-step procedure.

- [ ] **Step 7: Build the docs**

Run:
```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```
Expected: **clean build with zero warnings.** Because this task edits the docs, treat any warning as a defect and fix it — including the `roadmap.rst` "underline too short" warning if it appears in the sections this task touches (extend the underline to match the title length). The only warning that may be left standing is one explicitly confirmed as pre-existing and outside this task's edits *before* running the build; otherwise the build must be warning-free.

- [ ] **Step 8: Review checkpoint**

Pause for human review. Suggested commit message:
```
docs: close dill removal (re-hosted models, gcs-first, md5 pin)
```

---

## Final Handoff Checklist

- [ ] `fetch()` honors a pinned `CONFIGS['md5']` (stale caches re-download; sidecar-mismatch raises); Core ML configs unchanged.
- [ ] Both `.pt` models re-serialized, verified dill-free, and CPU-parity-identical to the originals.
- [ ] `CONFIGS` for `v3`/`v3-cls` is GCS-first, date-tagged, and pinned.
- [ ] Artifacts uploaded by the user to GCS + Azure (models, sidecars, `INDEX.txt`).
- [ ] Clean-cache network fetch resolves GCS-first with Azure failover and matching pinned hashes.
- [ ] `dill` removed from `pyproject.toml`; `uv.lock` regenerated; not present in the graph.
- [ ] Full fast + slow suites and `uv audit` pass without `dill`.
- [ ] Docs and roadmap reflect the final state.
- [ ] User reviewed each task's output before staging/committing.
