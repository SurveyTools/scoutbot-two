# Multi-Source Model Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `fetch()` in `model_store.py` so each model config lists multiple download sources (Azure, GCS, …); sources are tried in order and the first success wins, making downloads resilient to any single provider outage.

**Architecture:** Each non-Core ML entry in `CONFIGS` gains a `sources` list — ordered dicts with `provider`, `url`, and `md5_url` — replacing the flat `url`/`md5_url` keys. A private helper `_download_from_source()` handles one source attempt (MD5 check → optional download → local MD5 verify); `fetch()` iterates the list, catches `requests.RequestException` and `AssertionError`, logs a warning, and moves to the next source. `RegistrySource` gains an optional `provider` field so provenance records which backend actually delivered the file.

**Tech Stack:** Python 3.10+, `requests`, `scoutbot/model_store.py`, `scoutbot/registry/schema.py`, `tests/test_loc.py`

## Global Constraints

- Always use `uv run` — never bare `python`, `pytest`, `ruff`, or `mypy`
- Never run `git add`, `git commit`, or any staging/commit commands — user handles all commits
- Runtime rule: `uv run pytest -m "not slow"` for fast tests; `uv run pytest -m "slow"` for model-download tests
- Ruff for formatting/linting (`uv run ruff format .` then `uv run ruff check .`); mypy scoped to `scoutbot/` (`uv run mypy scoutbot`)
- Patch target: `scoutbot.model_store.*` (not `scoutbot.loc.*`) — `fetch()` uses its own module namespace
- No new dependencies — `requests` is already in the dependency list
- GCS public base URL: `https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/`
- Azure public base URL: `https://scoutmodels.blob.core.windows.net/localizer/`
- chunk_size for `iter_content`: change from 8192 → 65536 while touching the download loop

---

## File Map

| File | Change |
|---|---|
| `scoutbot/model_store.py` | Replace `url`/`md5_url` flat keys with `sources` list in `CONFIGS`; add `_download_from_source()` helper; refactor `fetch()` fallback loop |
| `scoutbot/registry/schema.py` | Add `provider: Optional[str] = None` to `RegistrySource` |
| `tests/test_loc.py` | Update `test_config_has_required_keys` (checks `url`/`md5_url` → check `sources`); add 3 new fallback tests |
| `docs/configuration.rst` | Document fallback source order and GCS as secondary source |
| `docs/models.rst` | Update "Hosted model files" section to list GCS URLs alongside Azure; update "Integrity verification" flow to describe multi-source fallback |
| `docs/security.rst` | Update "Model download policy" and "Integrity verification" sections to reflect two-provider fallback |
| `CHANGELOG.rst` | Add entry for multi-source fetch |

---

## Task 1: CONFIGS restructure + `_download_from_source` helper + `fetch()` fallback + schema + tests

**Files:**
- Modify: `scoutbot/model_store.py`
- Modify: `scoutbot/registry/schema.py`
- Modify: `tests/test_loc.py`

**Interfaces:**
- Produces: `_download_from_source(source: dict, local_path: Path, cfg_name: str) -> tuple[str, bool]`
  — returns `(remote_md5, cache_hit)` on success; raises `requests.RequestException` or `AssertionError` on failure
- Produces: `CONFIGS['v3']['sources']` — list of `{'provider': str, 'url': str, 'md5_url': str}`
- Produces: `RegistrySource.provider: Optional[str]`

- [ ] **Step 1: Write the failing tests**

Add these three tests to the `TestLocModelStore` class (or the relevant non-`@pytest.mark.slow` class) in `tests/test_loc.py`. Also update `test_config_has_required_keys`. All four should fail before the implementation.

```python
# Replace the existing test_config_has_required_keys body:
def test_config_has_required_keys(self):
    """Test that configs have all required keys."""
    from scoutbot import loc

    required_keys = [
        'name',
        'sources',
        'thresh',
        'slice_height',
        'slice_width',
        'overlap_height_ratio',
        'overlap_width_ratio',
        'perform_standard_pred',
        'postprocess_class_agnostic',
    ]

    for config_name in ['v3', 'v3-cls']:
        config = loc.CONFIGS[config_name]
        for key in required_keys:
            assert key in config, f'Config {config_name} missing key: {key}'

    # Each source entry must have provider, url, md5_url
    for config_name in ['v3', 'v3-cls']:
        for source in loc.CONFIGS[config_name]['sources']:
            assert 'provider' in source
            assert 'url' in source
            assert 'md5_url' in source
```

```python
# Add to the fast (non-slow) test class — place near the other fetch unit tests:

def test_fetch_falls_back_to_second_source_on_network_error(self, tmp_path):
    """If the first source raises a network error, fetch() tries the next source."""
    import requests as _requests
    from scoutbot import loc

    cfg = loc.CONFIGS['v3']
    local = tmp_path / cfg['name']
    fake_content = b'model weights fallback'
    remote_md5 = hashlib.md5(fake_content).hexdigest()

    md5_ok = MagicMock()
    md5_ok.text = remote_md5
    md5_ok.raise_for_status = MagicMock()

    download_ok = MagicMock()
    download_ok.raise_for_status = MagicMock()
    download_ok.headers = {'content-length': str(len(fake_content))}
    download_ok.iter_content.return_value = [fake_content]
    download_ok.__enter__ = MagicMock(return_value=download_ok)
    download_ok.__exit__ = MagicMock(return_value=False)

    call_count = [0]

    def _get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First source md5_url → network failure
            raise _requests.ConnectionError('Azure down')
        if call_count[0] == 2:
            # Second source md5_url → success
            return md5_ok
        # Second source download → success
        return download_ok

    with (
        patch.dict('os.environ', {'RW_REGISTRY_PATH': str(tmp_path / 'rw-registry.json')}),
        patch('scoutbot.model_store.CACHE_DIR', tmp_path),
        patch('requests.get', side_effect=_get),
    ):
        result = loc.fetch('v3')

    assert result == str(local)
    assert local.read_bytes() == fake_content


def test_fetch_raises_when_all_sources_fail(self, tmp_path):
    """RuntimeError is raised if every source raises a network error."""
    import requests as _requests
    from scoutbot import loc

    with (
        patch.dict('os.environ', {'RW_REGISTRY_PATH': str(tmp_path / 'rw-registry.json')}),
        patch('scoutbot.model_store.CACHE_DIR', tmp_path),
        patch('requests.get', side_effect=_requests.ConnectionError('all down')),
    ):
        with pytest.raises(RuntimeError, match='All sources failed'):
            loc.fetch('v3')


def test_fetch_records_provider_in_registry(self, tmp_path):
    """Registry entry for a freshly downloaded model includes the source provider."""
    import json
    from scoutbot import loc

    cfg = loc.CONFIGS['v3']
    local = tmp_path / cfg['name']
    fake_content = b'model data'
    remote_md5 = hashlib.md5(fake_content).hexdigest()

    md5_ok = MagicMock()
    md5_ok.text = remote_md5
    md5_ok.raise_for_status = MagicMock()

    download_ok = MagicMock()
    download_ok.raise_for_status = MagicMock()
    download_ok.headers = {'content-length': str(len(fake_content))}
    download_ok.iter_content.return_value = [fake_content]
    download_ok.__enter__ = MagicMock(return_value=download_ok)
    download_ok.__exit__ = MagicMock(return_value=False)

    call_count = [0]

    def _get(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return md5_ok
        return download_ok

    registry_path = tmp_path / 'rw-registry.json'

    with (
        patch.dict('os.environ', {'RW_REGISTRY_PATH': str(registry_path)}),
        patch('scoutbot.model_store.CACHE_DIR', tmp_path),
        patch('requests.get', side_effect=_get),
    ):
        loc.fetch('v3')

    registry = json.loads(registry_path.read_text())
    entry = next(e for e in registry['models'] if e['name'] == 'scoutbot-v3')
    assert entry['source']['provider'] == 'azure'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_loc.py::TestLocConfig::test_config_has_required_keys \
               tests/test_loc.py -k "test_fetch_falls_back or test_fetch_raises_when_all or test_fetch_records_provider" \
               -v -m "not slow"
```

Expected: 4 failures — `KeyError: 'sources'`, `AttributeError`, etc.

- [ ] **Step 3: Add `provider` to `RegistrySource` in `scoutbot/registry/schema.py`**

```python
@dataclass
class RegistrySource:
    type: str
    url: Optional[str] = None
    fetched_at: Optional[str] = None
    original_path: Optional[str] = None
    copied_from: Optional[str] = None
    note: Optional[str] = None
    provider: Optional[str] = None   # ← add this line

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> RegistrySource:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
```

- [ ] **Step 4: Replace `url`/`md5_url` with `sources` in `CONFIGS` and add `_download_from_source` helper in `scoutbot/model_store.py`**

Replace the `CONFIGS` dict (keep the coreml entries unchanged — they never had `url`/`md5_url`):

```python
CONFIGS = {
    'v3': {
        'name': 'yolov8-onecls.kaza.pt',
        'sources': [
            {
                'provider': 'azure',
                'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt',
                'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5',
            },
            {
                'provider': 'gcs',
                'url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt',
                'md5_url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt.md5',
            },
        ],
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
        'sources': [
            {
                'provider': 'azure',
                'url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt',
                'md5_url': 'https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5',
            },
            {
                'provider': 'gcs',
                'url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt',
                'md5_url': 'https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt.md5',
            },
        ],
        'thresh': 0.45,
        'slice_height': 512,
        'slice_width': 512,
        'overlap_height_ratio': 0.25,
        'overlap_width_ratio': 0.25,
        'perform_standard_pred': False,
        'postprocess_class_agnostic': True,
    },
    # Core ML entries unchanged — no url/md5_url to begin with
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
```

Add the helper function directly after `_md5()` and before `fetch()`:

```python
def _download_from_source(source: dict, local_path: Path) -> tuple[str, bool]:
    """
    Attempt to resolve the model from one source entry.

    Fetches the remote MD5, skips the download if the local file already
    matches, otherwise streams the download and verifies locally.

    Returns:
        (remote_md5, cache_hit) on success.

    Raises:
        requests.RequestException: on any network failure.
        AssertionError: if the downloaded file fails MD5 verification.
    """
    md5_resp = requests.get(source['md5_url'], timeout=30)
    md5_resp.raise_for_status()
    remote_md5 = md5_resp.text.strip().split()[0]

    if local_path.exists() and _md5(local_path) == remote_md5:
        return remote_md5, True

    with requests.get(source['url'], stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with (
            open(local_path, 'wb') as f,
            tqdm(total=total, unit='B', unit_scale=True, desc=local_path.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

    assert _md5(local_path) == remote_md5, (
        f'MD5 mismatch after download from {source["provider"]}: {local_path.name}'
    )
    return remote_md5, False
```

- [ ] **Step 5: Refactor the non-coreml download section of `fetch()` to use the fallback loop**

The section currently starting at `response = requests.get(str(cfg['md5_url']), ...)` through the `_registry.register(...)` call should be replaced with:

```python
    sources = cfg['sources']
    last_exc: Exception | None = None
    used_source: dict | None = None
    remote_md5 = ''

    for source in sources:
        try:
            remote_md5, cache_hit = _download_from_source(source, local_path)
            used_source = source
            break
        except (requests.RequestException, AssertionError) as exc:
            log.warning(
                f'Source {source["provider"]} unavailable for {cfg["name"]}: {exc!r}'
            )
            last_exc = exc
            # Remove any partial download so the next source starts clean
            if local_path.exists():
                local_path.unlink()

    if used_source is None:
        raise RuntimeError(
            f'All sources failed for {cfg["name"]}'
        ) from last_exc

    if cache_hit:
        log.debug(f'Model up to date: {local_path}')
        ended_ns = perf_counter_ns()
        emit(
            observer,
            {
                'name': 'fetch',
                'status': 'ok',
                'started_ns': started_ns,
                'ended_ns': ended_ns,
                'duration_ns': ended_ns - started_ns,
                'config': config,
                'backend': 'pt',
                'device': 'cpu',
                'cache_hit': True,
                'model_path': str(local_path),
                'provider': used_source['provider'],
            },
        )
        return str(local_path)

    log.info(f'Saved: {local_path} (from {used_source["provider"]})')
    ended_ns = perf_counter_ns()
    emit(
        observer,
        {
            'name': 'fetch',
            'status': 'ok',
            'started_ns': started_ns,
            'ended_ns': ended_ns,
            'duration_ns': ended_ns - started_ns,
            'config': config,
            'backend': 'pt',
            'device': 'cpu',
            'cache_hit': False,
            'model_path': str(local_path),
            'provider': used_source['provider'],
        },
    )

    from datetime import datetime, timezone

    from scoutbot.registry.schema import ModelEntry, RegistrySource

    _registry.register(
        ModelEntry(
            name=registry_name,
            path=str(local_path),
            md5=remote_md5,
            framework='yolov8',
            registered=datetime.now(timezone.utc).isoformat(),
            source=RegistrySource(
                type='fetched',
                url=str(used_source['url']),
                provider=used_source['provider'],
                fetched_at=datetime.now(timezone.utc).isoformat(),
            ),
            default_run_config={
                'slice_height': cfg['slice_height'],
                'slice_width': cfg['slice_width'],
                'overlap_height_ratio': cfg['overlap_height_ratio'],
                'overlap_width_ratio': cfg['overlap_width_ratio'],
                'perform_standard_pred': cfg['perform_standard_pred'],
                'postprocess_class_agnostic': cfg['postprocess_class_agnostic'],
                'confidence_threshold': cfg['thresh'],
            },
            description=f'ScoutBot {config} model (KAZA)',
        )
    )
    log.info(f'Registered: {registry_name}')
    return str(local_path)
```

Note: the registry cache-hit early return (lines before the sources loop) is unchanged — it still appears before the sources loop and returns immediately when the registry has a valid cached entry. Only the network-fetch section changes.

- [ ] **Step 6: Run the full fast test suite**

```bash
uv run pytest -m "not slow" -v
```

Expected: all previously passing tests still pass; the 4 new/updated tests now pass. If `test_fetch_emits_observer_event_for_cached_pt_model` fails because the emitted event now includes a `provider` key that the test doesn't assert, that is fine — the test only asserts the keys it explicitly checks.

- [ ] **Step 7: Run quality gates**

```bash
uv run ruff format .
uv run ruff check .
uv run mypy scoutbot
```

Expected: no errors. If mypy complains about `last_exc` type narrowing, add `assert last_exc is not None` before the `raise RuntimeError(...) from last_exc` line.

---

## Task 2: Documentation pass + CHANGELOG

**Files:**
- Modify: `docs/configuration.rst`
- Modify: `docs/models.rst`
- Modify: `docs/security.rst`
- Modify: `CHANGELOG.rst`

**Interfaces:**
- Consumes: the completed `sources` list from Task 1 (Azure first, GCS second)

- [ ] **Step 1: Update `docs/configuration.rst`**

Find the section that documents model download / cache behavior. Add a subsection (or paragraph) explaining fallback sources. The exact location will depend on the existing structure — search for "cache" or "fetch" in the file. Add after the model cache description:

```rst
Model download sources
~~~~~~~~~~~~~~~~~~~~~~

ScoutBot tries download sources in order and uses the first that succeeds.
The built-in order is:

1. **Azure Blob Storage** (``scoutmodels.blob.core.windows.net``) — primary
2. **Google Cloud Storage** (``storage.googleapis.com/rebel-pipeline-public-models``) — fallback

If Azure is unreachable or returns an error mid-transfer, the download
automatically retries from GCS with no user action required. A warning is
logged for each skipped source. Once any source delivers the file and the
local MD5 is verified, the model is registered in the local registry and
subsequent runs resolve from cache with no network access.
```

- [ ] **Step 2: Update `docs/models.rst`**

Replace the "Hosted model files" section (currently Azure-only) with the two-provider listing:

```rst
Hosted model files
~~~~~~~~~~~~~~~~~~

PyTorch-backed configs are hosted on both Azure Blob Storage (primary) and
Google Cloud Storage (fallback). Scoutbot tries each source in order and
uses the first that succeeds. Both hosts carry matching ``.md5`` sidecar
files for integrity verification.

``v3`` — single-class detector

  *Azure (primary)*

  * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt
  * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-onecls.kaza.pt.md5

  *GCS (fallback)*

  * Model: https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt
  * MD5:   https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-onecls.kaza.pt.md5

``v3-cls`` — multi-class species detector

  *Azure (primary)*

  * Model: https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt
  * MD5:   https://scoutmodels.blob.core.windows.net/localizer/yolov8-cls.kaza.pt.md5

  *GCS (fallback)*

  * Model: https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt
  * MD5:   https://storage.googleapis.com/rebel-pipeline-public-models/scoutbot/yolov8-cls.kaza.pt.md5

Core ML configs do not have hosted ``.mlpackage`` artifacts. Scoutbot exports them on demand from
the verified source ``.pt`` model and stores the resulting bundle in ``~/.cache/scoutbot/``.
```

Also replace the "Integrity verification" subsection (currently describes a single-source 5-step flow) with the multi-source version:

```rst
Integrity verification
~~~~~~~~~~~~~~~~~~~~~~

On each ``fetch()`` call for a hosted ``.pt`` model, Scoutbot:

1. Checks the local model registry; returns immediately if a valid cached
   entry is found (no network access).
2. Tries each configured source in order (Azure first, then GCS):

   a. Downloads the ``.md5`` sidecar from that source.
   b. Compares the sidecar's MD5 against the locally cached model file.
   c. If they match, returns the cached path immediately (cache hit).
   d. Otherwise, streams the model file from that source and verifies the
      local MD5 on completion.
   e. If the download or verification fails, logs a warning and moves to
      the next source.

3. Raises ``RuntimeError`` if all sources fail.
4. Registers the successfully downloaded model in the local registry with
   the provider name recorded in the provenance block.

Core ML bundles are directory packages, so they do not use remote MD5 sidecars. Their integrity
depends on exporting from the already-verified source ``.pt`` model.
```

- [ ] **Step 3: Update `docs/security.rst`**

Replace the "Model download policy" section:

```rst
Model download policy
---------------------

Hosted model artifacts are fetched from the sources listed in
``scoutbot.model_store.CONFIGS``. PyTorch-backed models are hosted on two
providers tried in order:

1. **Azure Blob Storage** (``scoutmodels.blob.core.windows.net``) — primary
2. **Google Cloud Storage** (``storage.googleapis.com/rebel-pipeline-public-models``) — fallback

* ``v3`` uses ``yolov8-onecls.kaza.pt``
* ``v3-cls`` uses ``yolov8-cls.kaza.pt``

If the primary source is unreachable or resets the connection mid-transfer,
Scoutbot automatically retries from the fallback with a logged warning. No
user action is required. See :doc:`models` for the full URL listing.
```

Replace the "Integrity verification" section:

```rst
Integrity verification
----------------------

Both providers carry matching ``.md5`` sidecar files. On each download
attempt Scoutbot fetches the sidecar from the same source as the model,
verifies the local MD5 on completion, and raises an error rather than using
any artifact that fails verification. A source that passes the MD5 check is
registered in the local registry; subsequent runs resolve from cache with no
network access.

Core ML ``.mlpackage`` bundles are not hosted as separate remote artifacts.
They are exported locally from verified source ``.pt`` models and cached in
``~/.cache/scoutbot/``.
```

- [ ] **Step 4: Add entry to `CHANGELOG.rst`**

Add under `Unreleased` (before the existing first bullet):

```rst
* Model downloads now try multiple sources in order (Azure → GCS) and fall
  back automatically if any source is unreachable or resets the connection
  mid-transfer. GCS (``storage.googleapis.com/rebel-pipeline-public-models``)
  is the built-in fallback for all ``.pt`` models. Registry provenance now
  records which provider delivered the file. Download chunk size increased
  from 8 KB to 64 KB for better throughput on large model files.
```

- [ ] **Step 5: Run fast tests one final time to confirm nothing regressed**

```bash
uv run pytest -m "not slow" -v
uv run ruff format --check .
uv run ruff check .
uv run mypy scoutbot
```

Expected: all green.

---

## Self-Review

**Spec coverage:**
- ✅ Multiple remote sources in CONFIGS (`sources` list, Azure + GCS)
- ✅ Try in order, fall back on network failure (`requests.RequestException`)
- ✅ Fall back on MD5 mismatch (`AssertionError`)
- ✅ All sources fail → `RuntimeError` with chained cause
- ✅ Provenance records which provider was used (`RegistrySource.provider`)
- ✅ chunk_size bumped 8192 → 65536
- ✅ Existing registry cache-hit path untouched
- ✅ Core ML fetch path untouched
- ✅ `test_config_has_required_keys` updated
- ✅ 3 new fallback tests
- ✅ `docs/configuration.rst` — fallback source order documented
- ✅ `docs/models.rst` — GCS URLs added alongside Azure; integrity flow updated for multi-source
- ✅ `docs/security.rst` — download policy and integrity sections updated for two-provider fallback
- ✅ `CHANGELOG.rst`

**Placeholder scan:** None found.

**Type consistency:**
- `_download_from_source` returns `tuple[str, bool]` throughout
- `used_source` typed as `dict | None`, narrowed before use
- `RegistrySource.provider` is `Optional[str]` — consistent with `url`, `fetched_at` fields
