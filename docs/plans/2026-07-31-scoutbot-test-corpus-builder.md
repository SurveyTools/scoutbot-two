# Scoutbot Test-Corpus Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a maintainer tool that turns the internal Matetsi survey into a small, density-tiered, GPS-stripped scoutbot test corpus with frozen CPU baselines, plus a skip-if-absent regression test that asserts inference has not drifted.

**Architecture:** A single `tools/build_test_corpus.py` module of small, independently-testable pure helpers (`resolve_local`, `select_tiers`, `strip_and_verify`, provenance collectors, `build_manifest`, `assert_no_source_identifiers`, `stack_matches`) plus a thin `main()` that orchestrates: model preflight → select from JSONL → resolve local files → strip EXIF → freeze CPU baselines → write `manifest.json` / `baselines.jsonl` / `README.md` → governance scan. A `tests/test_corpus_regression.py` consumer replays the frozen baselines when a corpus is present.

**Tech Stack:** Python 3.10–3.13, `uv`, pytest, Click-free `argparse` CLI, `piexif` (EXIF strip), Pillow (dimension/decoding check, already transitive via ultralytics), `scoutbot.pipeline` (SAHI + YOLOv8).

## Global Constraints

- Runtime: always `uv run`; never bare `python`/`pytest`/`ruff`/`mypy`.
- **No git steps.** The user handles all `git add`/`commit`/`push`. Each task ends at a review checkpoint with a suggested commit message — no commit steps in this plan (repo rule overrides the skill template).
- Plans and specs live under `docs/plans/` and `docs/specs/` (repo override of skill defaults).
- Test-first for code changes (red before green).
- Comments explain WHY, not what; match the file's existing comment density and idiom.
- Never silently swallow errors — log and surface them (P3).
- Build only what the spec/plan asks; record deferrals as "Out of scope" (YAGNI).
- **Matetsi is internal-only.** Every corpus artifact must be GPS/EXIF-stripped and location-scrubbed (opaque IDs). The builder performs **no network I/O** — no upload and no model download (model must already be local; else fail with fetch instructions).
- Baselines are frozen and asserted on **CPU only** (bit-reproducible; MPS/CUDA are not).
- Density tiers are **exact inclusive integer bounds**: `empty` `c==0`, `typical` `3<=c<=6`, `dense` `9<=c<=13`, `extreme` `c>=14` (top-N desc, always includes global max). Default `--per-tier 5`.
- Manifest `schema` id is `scoutbot:test-corpus:1.0` (scoutbot namespace, not `osw:`).
- Tests import the tool as `import build_test_corpus` (`tools/` is already on `pythonpath`); the fast unit tests are **not** `slow`-marked; only the corpus-replay regression is `slow` + skip-if-absent.

---

### Task 1: `resolve_local` — deterministic full-suffix path resolution

**Files:**
- Create: `tools/build_test_corpus.py`
- Test: `tests/test_build_test_corpus.py`

**Interfaces:**
- Produces: `resolve_local(filepath: str, jsonl_source_root: str, matetsi_root: Path) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_test_corpus.py
from pathlib import Path

import pytest

import build_test_corpus as btc


def test_resolve_local_full_suffix_ignores_basename_collisions(tmp_path):
    # Two sibling dirs hold the SAME basename; full-suffix resolution must land on the right file.
    src_root = "/home/brandon/matetsi_images"
    left = tmp_path / "day1" / "flight1" / "left"
    right = tmp_path / "day1" / "flight1" / "right"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    (left / "BCK_0001.JPG").write_bytes(b"L")
    (right / "BCK_0001.JPG").write_bytes(b"R")

    got = btc.resolve_local(
        f"{src_root}/day1/flight1/right/BCK_0001.JPG", src_root, tmp_path
    )
    assert got == right / "BCK_0001.JPG"
    assert got.read_bytes() == b"R"


def test_resolve_local_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        btc.resolve_local(
            "/home/brandon/matetsi_images/day1/nope.JPG",
            "/home/brandon/matetsi_images",
            tmp_path,
        )


def test_resolve_local_foreign_prefix_raises(tmp_path):
    with pytest.raises(ValueError):
        btc.resolve_local("/mnt/e/other/x.JPG", "/home/brandon/matetsi_images", tmp_path)


def test_resolve_local_rejects_traversal(tmp_path):
    # Maintainer tool consuming path strings from a JSONL — refuse '..' escaping the root.
    with pytest.raises(ValueError, match="traversal|escapes"):
        btc.resolve_local(
            "/home/brandon/matetsi_images/../../etc/passwd",
            "/home/brandon/matetsi_images",
            tmp_path,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_build_test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_test_corpus'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# tools/build_test_corpus.py
"""Build a density-tiered, GPS-stripped scoutbot test corpus from the internal Matetsi survey.

Maintainer tool (run internally where Matetsi is available). It selects images by density tier
from the scoutbot detection JSONL, strips EXIF/GPS, freezes CPU inference baselines, and writes a
self-contained corpus (images/ + baselines.jsonl + manifest.json + README.md). It performs no
network I/O; the model must already be present locally.
"""

from __future__ import annotations

from pathlib import Path

TIERS = ("empty", "typical", "dense", "extreme")


def resolve_local(filepath: str, jsonl_source_root: str, matetsi_root: Path) -> Path:
    """Map a JSONL record's foreign absolute ``filepath`` to the local file by full relative suffix.

    Basename matching is unsafe — ~72% of Matetsi basenames collide across left/right and flights —
    so we strip the declared source-root prefix and join the *entire* remaining suffix onto
    ``matetsi_root``. That is a single deterministic path; collisions are a non-issue by
    construction. Raises if the prefix is absent or the joined path does not exist.
    """
    prefix = jsonl_source_root.rstrip("/") + "/"
    if not filepath.startswith(prefix):
        raise ValueError(
            f"JSONL filepath {filepath!r} is not under --jsonl-source-root {jsonl_source_root!r}"
        )
    suffix = filepath[len(prefix) :]
    if ".." in Path(suffix).parts:
        raise ValueError(f"Refusing traversal in JSONL suffix: {suffix!r}")
    local = matetsi_root / suffix
    root = matetsi_root.resolve()
    if not local.resolve().is_relative_to(root):
        raise ValueError(f"Resolved path escapes --matetsi-root: {local}")
    if not local.is_file():
        raise FileNotFoundError(f"Resolved local image does not exist: {local}")
    return local
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_build_test_corpus.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint/type check**

Run: `uv run ruff format tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run ruff check tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run mypy tools/build_test_corpus.py`
Expected: clean.

- [ ] **Step 6: Review checkpoint** — do not commit. Suggested message:
  `feat(tools): add test-corpus builder resolve_local (full-suffix path resolution)`

---

### Task 2: `select_tiers` — deterministic density-tier selection

**Files:**
- Modify: `tools/build_test_corpus.py`
- Test: `tests/test_build_test_corpus.py`

**Interfaces:**
- Consumes: `TIERS`
- Produces: `select_tiers(records: list[dict], per_tier: int) -> dict[str, list[dict]]` — each returned record carries an added `_count` int; buckets: `empty` (0), `typical` (3–6), `dense` (9–13), `extreme` (>=14, top-N desc). Raises `ValueError` if any tier is short.

- [ ] **Step 1: Write the failing test**

```python
def _rec(fp, n):
    return {"filepath": fp, "detections": [{} for _ in range(n)]}


def test_select_tiers_buckets_and_counts():
    records = (
        [_rec(f"/r/empty{i}.jpg", 0) for i in range(6)]
        + [_rec(f"/r/typ{i}.jpg", c) for i, c in enumerate([3, 4, 5, 6, 4, 5])]
        + [_rec(f"/r/den{i}.jpg", c) for i, c in enumerate([9, 10, 11, 12, 13, 10])]
        + [_rec(f"/r/ext{i}.jpg", c) for i, c in enumerate([20, 40, 80, 175, 30, 14])]
        # counts 1,2,7,8 fall in no tier and must be ignored:
        + [_rec("/r/gap1.jpg", 1), _rec("/r/gap7.jpg", 7), _rec("/r/gap8.jpg", 8)]
    )
    picked = btc.select_tiers(records, per_tier=3)
    assert set(picked) == set(btc.TIERS)
    for tier in btc.TIERS:
        assert len(picked[tier]) == 3
    assert all(r["_count"] == 0 for r in picked["empty"])
    assert all(3 <= r["_count"] <= 6 for r in picked["typical"])
    assert all(9 <= r["_count"] <= 13 for r in picked["dense"])
    assert all(r["_count"] >= 14 for r in picked["extreme"])
    # extreme is top-N desc → always includes the global max (175)
    assert max(r["_count"] for r in picked["extreme"]) == 175


def test_select_tiers_is_deterministic():
    records = [_rec(f"/r/den{i}.jpg", c) for i, c in enumerate([9, 10, 11, 12, 13])] + [
        _rec(f"/r/e{i}.jpg", c) for i, c in enumerate([0, 0, 0, 4, 5, 6, 20, 40, 60])
    ]
    a = btc.select_tiers(records, per_tier=2)
    b = btc.select_tiers(records, per_tier=2)
    assert [r["filepath"] for r in a["dense"]] == [r["filepath"] for r in b["dense"]]


def test_select_tiers_short_bucket_raises():
    records = [_rec("/r/e1.jpg", 0)] + [_rec(f"/r/t{i}.jpg", 4) for i in range(5)] + [
        _rec(f"/r/d{i}.jpg", 10) for i in range(5)
    ] + [_rec(f"/r/x{i}.jpg", 50) for i in range(5)]
    with pytest.raises(ValueError, match="empty"):
        btc.select_tiers(records, per_tier=5)  # only 1 empty available
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_build_test_corpus.py -k select_tiers -v`
Expected: FAIL — `AttributeError: module 'build_test_corpus' has no attribute 'select_tiers'`.

- [ ] **Step 3: Write the minimal implementation** (append to `tools/build_test_corpus.py`)

```python
def _bucket(records: list[dict], lo: int, hi: int) -> list[dict]:
    return [r for r in records if lo <= r["_count"] <= hi]


def _evenly_spaced(items: list[dict], n: int) -> list[dict]:
    """Pick ``n`` evenly-spaced items from a sorted list (not first-``n``, which clusters at an edge)."""
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def select_tiers(records: list[dict], per_tier: int) -> dict[str, list[dict]]:
    """Bucket JSONL records into density tiers and deterministically pick ``per_tier`` from each.

    Selection is over records only — never a directory walk. Bands are exact inclusive integer
    bounds derived from the Matetsi distribution. ``empty``/``typical``/``dense`` pick evenly-spaced
    across the (count, filepath)-sorted bucket; ``extreme`` (count >= 14, disjoint from ``dense``)
    takes the top-``per_tier`` descending so the global max is always included. The ``filepath``
    tiebreak orders deterministically and is never emitted. Raises if a tier is short.
    """
    recs = [{**r, "_count": len(r.get("detections", []))} for r in records]
    buckets = {
        "empty": _bucket(recs, 0, 0),
        "typical": _bucket(recs, 3, 6),
        "dense": _bucket(recs, 9, 13),
        "extreme": _bucket(recs, 14, 10**9),
    }
    out: dict[str, list[dict]] = {}
    for tier in TIERS:
        pool = sorted(buckets[tier], key=lambda r: (r["_count"], r["filepath"]))
        if tier == "extreme":
            picked = list(reversed(pool))[:per_tier]
        else:
            picked = _evenly_spaced(pool, per_tier)
        if len(picked) < per_tier:
            raise ValueError(
                f"Tier {tier!r} has only {len(picked)} candidate(s) (< --per-tier {per_tier})"
            )
        out[tier] = picked
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_build_test_corpus.py -k select_tiers -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint/type check**

Run: `uv run ruff format tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run ruff check tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run mypy tools/build_test_corpus.py`
Expected: clean.

- [ ] **Step 6: Review checkpoint** — suggested message:
  `feat(tools): add select_tiers density-tier selection (exact bands, deterministic)`

---

### Task 3: `strip_and_verify` — EXIF/GPS strip + verification, and the `piexif` dependency

**Files:**
- Modify: `tools/build_test_corpus.py`, `pyproject.toml`
- Test: `tests/test_build_test_corpus.py`

**Interfaces:**
- Produces: `strip_and_verify(src: Path, dst: Path) -> None` — copies `src`→`dst`, removes EXIF, asserts no EXIF/GPS survives and dimensions are unchanged. Raises `AssertionError` on any residual tag or size change.

- [ ] **Step 1: Add `piexif` to the `test` extra**

In `pyproject.toml`, under `[project.optional-dependencies]` `test = [ ... ]`, add:

```toml
    # EXIF/GPS stripping + verification for the internal test-corpus builder (tools/).
    "piexif",
```

Then sync: `uv sync --extra test`
Expected: `piexif` installed.

- [ ] **Step 2: Write the failing test**

```python
import piexif
from PIL import Image


def _jpeg_with_gps(path: Path, size=(64, 48)):
    Image.new("RGB", size, (10, 20, 30)).save(path, "JPEG")
    exif = {
        "0th": {piexif.ImageIFD.Make: b"TestCam"},
        "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"S"},
        "Exif": {},
        "1st": {},
        "thumbnail": None,
    }
    piexif.insert(piexif.dump(exif), str(path))


def test_strip_and_verify_removes_gps_and_keeps_dimensions(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    _jpeg_with_gps(src, size=(64, 48))
    # sanity: GPS present before strip
    assert piexif.load(str(src))["GPS"]

    btc.strip_and_verify(src, dst)

    after = piexif.load(str(dst))
    assert not after["GPS"] and not after["0th"]
    with Image.open(dst) as im:
        assert im.size == (64, 48)


def test_strip_and_verify_on_clean_jpeg_is_ok(tmp_path):
    src = tmp_path / "clean.jpg"
    dst = tmp_path / "clean_out.jpg"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(src, "JPEG")  # no EXIF
    btc.strip_and_verify(src, dst)  # must not raise
    assert dst.is_file()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_build_test_corpus.py -k strip_and_verify -v`
Expected: FAIL — `AttributeError: ... has no attribute 'strip_and_verify'`.

- [ ] **Step 4: Write the minimal implementation** (append; add imports at top of module)

Add to the import block:

```python
import shutil

import piexif
from PIL import Image
```

Append:

```python
_EXIF_IFDS = ("0th", "Exif", "GPS", "1st")


def _has_exif(exif: dict) -> bool:
    return any(exif.get(ifd) for ifd in _EXIF_IFDS) or exif.get("thumbnail") is not None


def strip_and_verify(src: Path, dst: Path) -> None:
    """Copy ``src`` → ``dst`` with EXIF removed, then assert no EXIF/GPS survives.

    ``piexif.remove`` rewrites metadata only; pixels are not re-encoded. Baselines are frozen from
    the stripped copy, so bit-losslessness is not load-bearing — but we still assert the stripped
    image decodes and keeps its dimensions, since a corrupted strip would silently change baselines.
    Governance-critical: a residual EXIF/GPS tag is a hard failure.
    """
    with Image.open(src) as im:
        src_size = im.size
    shutil.copy(src, dst)
    # Only call remove() when there is EXIF to remove; piexif.remove() on a segment-less JPEG raises.
    if _has_exif(piexif.load(str(dst))):
        piexif.remove(str(dst))
    residual = piexif.load(str(dst))
    if _has_exif(residual):
        raise AssertionError(f"EXIF/GPS survived strip in {dst}: {residual}")
    with Image.open(dst) as im2:
        if im2.size != src_size:
            raise AssertionError(f"Dimensions changed after strip: {src_size} -> {im2.size}")
        im2.load()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_build_test_corpus.py -k strip_and_verify -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Lint/type check**

Run: `uv run ruff format tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run ruff check tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run mypy tools/build_test_corpus.py`
Expected: clean. (If mypy flags missing `piexif` stubs, add `warn_unused_ignores`-safe `# type: ignore[import-untyped]` on the `import piexif` line — piexif ships no stubs.)

- [ ] **Step 7: Review checkpoint** — suggested message:
  `feat(tools): add strip_and_verify EXIF/GPS strip; add piexif to test extra`

---

### Task 4: Provenance — runtime collection, model preflight, `run_config`, `build_manifest`

**Files:**
- Modify: `tools/build_test_corpus.py`
- Test: `tests/test_build_test_corpus.py`

**Interfaces:**
- Consumes: `scoutbot.model_store.CONFIGS`, `scoutbot.model_store._md5`, `scoutbot.registry.paths.get_models_dir`, `scoutbot.__version__`
- Produces:
  - `lockfile_sha256(lock_path: Path) -> str`
  - `collect_runtime(lock_path: Path) -> dict`
  - `run_config_for(config: str) -> dict`
  - `assert_model_present(config: str) -> Path` (raises `SystemExit` with fetch instructions)
  - `build_manifest(config, per_tier, images, runtime, model_md5) -> dict`

- [ ] **Step 1: Write the failing test**

```python
import scoutbot
from scoutbot.model_store import CONFIGS


def test_run_config_for_maps_thresh_to_loc_thresh():
    rc = btc.run_config_for("v3")
    assert rc["backend_device"] == "cpu"
    assert rc["loc_thresh"] == CONFIGS["v3"]["thresh"]
    assert rc["slice_height"] == CONFIGS["v3"]["slice_height"]
    assert rc["postprocess_class_agnostic"] == CONFIGS["v3"]["postprocess_class_agnostic"]


def test_collect_runtime_has_required_keys(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"lockdata")
    rt = btc.collect_runtime(lock)
    for k in (
        "scoutbot_version", "scoutbot_commit", "python", "platform",
        "platform_system", "platform_machine", "torch", "numpy", "lockfile_sha256",
    ):
        assert k in rt
    assert rt["scoutbot_version"] == scoutbot.__version__
    assert len(rt["lockfile_sha256"]) == 64  # sha256 hex


def test_build_manifest_shape():
    images = [
        {"name": "dense_01.jpg", "tier": "dense", "md5": "a" * 32,
         "source_density": 11, "baseline_detection_count": 11},
    ]
    runtime = {"scoutbot_version": "0.5.0"}
    m = btc.build_manifest("v3", 5, images, runtime, model_md5="b" * 32)
    assert m["schema"] == "scoutbot:test-corpus:1.0"
    assert m["model"] == {"config": "v3", "name": CONFIGS["v3"]["name"], "md5": "b" * 32}
    assert m["run_config"]["loc_thresh"] == CONFIGS["v3"]["thresh"]
    assert m["runtime"] is runtime
    assert m["images"][0]["baseline_detection_count"] == 11
    assert m["per_tier"] == 5


def test_assert_model_present_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(btc, "get_models_dir", lambda: tmp_path)  # empty dir
    with pytest.raises(SystemExit, match="fetch"):
        btc.assert_model_present("v3")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_build_test_corpus.py -k "run_config or collect_runtime or build_manifest or model_present" -v`
Expected: FAIL — attributes not defined.

- [ ] **Step 3: Write the minimal implementation** (append; add imports)

Add to the import block:

```python
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version

import scoutbot
from scoutbot import model_store
from scoutbot.model_store import CONFIGS
from scoutbot.registry.paths import get_models_dir
```

Append:

```python
def lockfile_sha256(lock_path: Path) -> str:
    return hashlib.sha256(Path(lock_path).read_bytes()).hexdigest()


def _pkg(name: str) -> str:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    # WHY: provenance only — a detached/non-git checkout must not fail the build.
    # nosec B603/B607: fixed argv (no shell, no user input); `git` from PATH is intended for a
    # maintainer tool run inside the repo.
    try:
        out = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_runtime(lock_path: Path) -> dict:
    """Capture the dependency stack that makes an exact-equality baseline interpretable."""
    return {
        "scoutbot_version": scoutbot.__version__,
        "scoutbot_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "torch": _pkg("torch"),
        "torchvision": _pkg("torchvision"),
        "ultralytics": _pkg("ultralytics"),
        "sahi": _pkg("sahi"),
        "numpy": _pkg("numpy"),
        "lockfile_sha256": lockfile_sha256(lock_path) if Path(lock_path).exists() else "unknown",
    }


def run_config_for(config: str) -> dict:
    """The effective pipeline() kwargs sourced from CONFIGS (pipeline does not read these itself)."""
    cfg = CONFIGS[config]
    return {
        "backend_device": "cpu",
        "loc_thresh": cfg["thresh"],  # CONFIGS key is 'thresh'; pipeline kwarg is 'loc_thresh'
        "slice_height": cfg["slice_height"],
        "slice_width": cfg["slice_width"],
        "overlap_height_ratio": cfg["overlap_height_ratio"],
        "overlap_width_ratio": cfg["overlap_width_ratio"],
        "perform_standard_pred": cfg["perform_standard_pred"],
        "postprocess_class_agnostic": cfg["postprocess_class_agnostic"],
    }


def assert_model_present(config: str) -> Path:
    """Offline preflight: the model must already be local, else fail with fetch instructions.

    pipeline() would fetch on a cache miss, violating the no-network constraint. We check presence
    and, when the config exposes a pinned md5, integrity — never downloading. Configs without a
    pinned hash (e.g. Core ML bundles) fall back to a presence-only check.
    """
    cfg = CONFIGS[config]
    model_path = get_models_dir() / str(cfg["name"])
    hint = f"Run: uv run python -m scoutbot fetch --config {config}"
    if not model_path.exists():
        raise SystemExit(f"Model not present locally: {model_path}\n{hint}")
    pinned = cfg.get("md5")
    if pinned is not None:
        actual = model_store._md5(model_path)
        if actual != pinned:
            raise SystemExit(
                f"Model md5 mismatch for {model_path} (got {actual}, expected {pinned}).\n{hint}"
            )
    return model_path


def build_manifest(
    config: str, per_tier: int, images: list[dict], runtime: dict, model_md5: str
) -> dict:
    cfg = CONFIGS[config]
    return {
        "schema": "scoutbot:test-corpus:1.0",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "matetsi (internal; GPS-stripped; opaque IDs)",
        "per_tier": per_tier,
        "model": {"config": config, "name": str(cfg["name"]), "md5": model_md5},
        "run_config": run_config_for(config),
        "runtime": runtime,
        "images": images,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_build_test_corpus.py -k "run_config or collect_runtime or build_manifest or model_present" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint/type check**

Run: `uv run ruff format tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run ruff check tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run mypy tools/build_test_corpus.py`
Expected: clean.

- [ ] **Step 6: Review checkpoint** — suggested message:
  `feat(tools): add corpus provenance (runtime, model preflight, run_config, manifest)`

---

### Task 5: `assert_no_source_identifiers` — governance scan of emitted files

**Files:**
- Modify: `tools/build_test_corpus.py`
- Test: `tests/test_build_test_corpus.py`

**Interfaces:**
- Produces:
  - `source_segments(records: list[dict], jsonl_source_root: str, matetsi_root: Path) -> set[str]`
  - `assert_no_source_identifiers(corpus_dir: Path, forbidden: set[str]) -> None` (raises `AssertionError` on a leak)

- [ ] **Step 1: Write the failing test**

```python
def test_source_segments_collects_path_parts_and_basenames():
    records = [{"filepath": "/home/brandon/matetsi_images/141025_day1/flight1/left/BCK_0001.JPG"}]
    segs = btc.source_segments(records, "/home/brandon/matetsi_images", Path("/vol/matetsi/images"))
    assert "/home/brandon/matetsi_images" in segs
    assert "/vol/matetsi/images" in segs
    assert "141025_day1" in segs
    assert "left" in segs
    assert "BCK_0001.JPG" in segs
    assert "BCK_0001" in segs  # stem too


def test_source_segments_excludes_tier_labels():
    # A source dir that happens to equal a tier name must NOT become a forbidden token — tier
    # labels are legitimate emitted vocabulary.
    records = [{"filepath": "/home/brandon/matetsi_images/dense/left/CAM_0001.JPG"}]
    segs = btc.source_segments(records, "/home/brandon/matetsi_images", Path("/vol/x"))
    assert "dense" not in segs
    assert "CAM_0001" in segs


def test_assert_no_source_identifiers_flags_leak(tmp_path):
    (tmp_path / "manifest.json").write_text('{"note": "from 141025_day1/left"}')
    with pytest.raises(AssertionError, match="141025_day1"):
        btc.assert_no_source_identifiers(tmp_path, {"141025_day1"})


def test_assert_no_source_identifiers_allows_readme_paths(tmp_path):
    # README legitimately carries commands, env vars, and corpus-relative paths — '/' is fine.
    (tmp_path / "README.md").write_text(
        "Set `SCOUTBOT_TEST_CORPUS=/path/to/corpus` and run `uv run pytest tests/`.\n"
    )
    btc.assert_no_source_identifiers(tmp_path, {"141025_day1", "BCK_0001"})  # must not raise
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_build_test_corpus.py -k "source_segments or no_source_identifiers" -v`
Expected: FAIL — attributes not defined.

- [ ] **Step 3: Write the minimal implementation** (append)

```python
_EMITTED_FILES = ("manifest.json", "baselines.jsonl", "README.md")


def source_segments(
    records: list[dict], jsonl_source_root: str, matetsi_root: Path
) -> set[str]:
    """Forbidden source-derived tokens: the roots plus every selected record's path parts + basename.

    Deliberately does NOT include the bare '/' — the README carries commands and corpus-relative
    paths — so the scan targets source tokens, not path separators.
    """
    forbidden: set[str] = {jsonl_source_root, jsonl_source_root.rstrip("/"), str(matetsi_root)}
    for r in records:
        parts = [p for p in r["filepath"].split("/") if p]
        forbidden.update(parts)
        if parts:
            forbidden.add(Path(parts[-1]).stem)  # basename without extension
    # Tier labels are legitimate emitted vocabulary (opaque names, manifest 'tier'); never treat a
    # source segment that happens to equal a tier name as a leak.
    return {f for f in forbidden if f} - set(TIERS)


def assert_no_source_identifiers(corpus_dir: Path, forbidden: set[str]) -> None:
    """Fail the build if any emitted file leaks a source-derived token (governance-critical)."""
    for name in _EMITTED_FILES:
        path = corpus_dir / name
        if not path.exists():
            continue
        text = path.read_text()
        for token in forbidden:
            if token and token in text:
                raise AssertionError(f"Source identifier {token!r} leaked into {name}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_build_test_corpus.py -k "source_segments or no_source_identifiers" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint/type check**

Run: `uv run ruff format tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run ruff check tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run mypy tools/build_test_corpus.py`
Expected: clean.

- [ ] **Step 6: Review checkpoint** — suggested message:
  `feat(tools): add governance scan for source identifiers in emitted files`

---

### Task 6: CLI `main()` orchestration + monkeypatched end-to-end smoke test

**Files:**
- Modify: `tools/build_test_corpus.py`
- Test: `tests/test_build_test_corpus.py`

**Interfaces:**
- Consumes: every helper from Tasks 1–5, `scoutbot.pipeline`
- Produces:
  - `render_readme(config: str, per_tier: int) -> str`
  - `build_corpus(matetsi_jsonl, jsonl_source_root, matetsi_root, out_dir, per_tier, config, force=False) -> dict` (returns manifest; does all I/O; refuses a non-empty `out_dir/images` unless `force`)
  - `main(argv: list[str] | None = None) -> None` (argparse entry point)

- [ ] **Step 1: Write the failing test** (monkeypatch `pipeline` + `assert_model_present` so no Matetsi/model/network is needed)

```python
import json


def test_build_corpus_end_to_end(tmp_path, monkeypatch):
    # 1 empty + 5 typical + 5 dense + 5 extreme synthetic source images with unique suffixes.
    src_root = "/home/brandon/matetsi_images"
    matetsi_root = tmp_path / "images"
    # Neutral Matetsi-like source paths — camera-native basenames, NO tier names in the path (a tier
    # name as a source segment would otherwise be a legitimate emitted token, not a leak).
    counts = [("A", 0, 5), ("B", 4, 5), ("C", 11, 5), ("D", 50, 5)]
    records = []
    for cam, c, n in counts:
        for i in range(n):
            rel = f"day1/flight1/left/{cam}CK_{i:04d}.JPG"
            local = matetsi_root / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            _jpeg_with_gps(local, size=(48, 32))
            records.append({"filepath": f"{src_root}/{rel}", "detections": [{} for _ in range(c)]})
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))

    # No model, no real inference, no network.
    monkeypatch.setattr(btc, "assert_model_present", lambda config: tmp_path / "fake.pt")

    def fake_pipeline(img, config=None, **kw):
        assert kw["backend_device"] == "cpu"
        return 0.5, [{"l": "detection", "c": 0.5, "x": 1, "y": 2, "w": 3, "h": 4}]

    monkeypatch.setattr(btc.scoutbot, "pipeline", fake_pipeline)

    out = tmp_path / "corpus"
    manifest = btc.build_corpus(
        matetsi_jsonl=jsonl, jsonl_source_root=src_root, matetsi_root=matetsi_root,
        out_dir=out, per_tier=5, config="v3",
    )

    # Flat images/ dir, 20 opaque tier-prefixed images, all EXIF-clean.
    imgs = sorted((out / "images").glob("*.jpg"))
    assert len(imgs) == 20
    assert all(not piexif.load(str(p))["GPS"] for p in imgs)
    assert (out / "baselines.jsonl").exists()
    assert (out / "manifest.json").exists()
    assert (out / "README.md").exists()
    # Baselines carry baseline_detection_count.
    first = json.loads((out / "baselines.jsonl").read_text().splitlines()[0])
    assert first["baseline_detection_count"] == 1
    # Governance: no source token leaked (would have raised inside build_corpus otherwise).
    assert manifest["schema"] == "scoutbot:test-corpus:1.0"
    assert len(manifest["images"]) == 20


def test_build_corpus_refuses_nonempty_out_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(btc, "assert_model_present", lambda config: tmp_path / "fake.pt")
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text("")
    out = tmp_path / "corpus"
    (out / "images").mkdir(parents=True)
    (out / "images" / "stale.jpg").write_bytes(b"old")
    with pytest.raises(SystemExit, match="force"):
        btc.build_corpus(
            matetsi_jsonl=jsonl, jsonl_source_root="/r", matetsi_root=tmp_path,
            out_dir=out, per_tier=5, config="v3",
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_build_test_corpus.py -k end_to_end -v`
Expected: FAIL — `build_corpus` not defined.

- [ ] **Step 3: Write the minimal implementation** (append; add `argparse`, `json` imports)

Add to the import block:

```python
import argparse
import json
```

Append:

```python
def render_readme(config: str, per_tier: int) -> str:
    """Provenance/governance note. Opaque IDs only — no source filenames or paths."""
    return (
        "# Scoutbot Test Corpus (internal)\n\n"
        "Density-tiered scoutbot regression/benchmark corpus derived from an **internal** survey.\n\n"
        "> INTERNAL USE ONLY — GPS/EXIF-stripped, opaque IDs. **Not cleared for public "
        "distribution** until redistribution permission is established.\n\n"
        f"- Model config: `{config}`\n"
        f"- Images per tier: {per_tier} (tiers: empty, typical, dense, extreme)\n"
        "- Baselines: frozen scoutbot CPU output (`baselines.jsonl`); a drift-regression anchor, "
        "not human ground truth.\n\n"
        "## Consuming it\n\n"
        "Point the regression test at this directory:\n\n"
        "```\n"
        "SCOUTBOT_TEST_CORPUS=/path/to/this/corpus uv run pytest tests/test_corpus_regression.py\n"
        "```\n"
    )


def build_corpus(
    matetsi_jsonl: Path,
    jsonl_source_root: str,
    matetsi_root: Path,
    out_dir: Path,
    per_tier: int,
    config: str,
    force: bool = False,
) -> dict:
    """Select → resolve → strip → freeze → write corpus. No network I/O; model must be local."""
    images_dir = out_dir / "images"
    # WHY: fail fast before any expensive work. Refuse ANY non-empty output dir unless --force —
    # stale images, or a stale manifest/baselines/README left by a rebuild that failed midway, would
    # silently corrupt the corpus. With --force we clear the whole output dir, not just images/.
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(
                f"Output dir {out_dir} is non-empty; pass --force to overwrite."
            )
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    model_path = assert_model_present(config)
    model_md5 = model_store._md5(model_path) if CONFIGS[config].get("md5") else "unknown"

    records = [
        json.loads(line) for line in matetsi_jsonl.read_text().splitlines() if line.strip()
    ]
    selected = select_tiers(records, per_tier)
    rc = run_config_for(config)

    image_records: list[dict] = []
    baselines: list[dict] = []
    used_records: list[dict] = []
    for tier in TIERS:
        for idx, rec in enumerate(selected[tier], start=1):
            used_records.append(rec)
            name = f"{tier}_{idx:02d}.jpg"
            dst = images_dir / name
            strip_and_verify(resolve_local(rec["filepath"], jsonl_source_root, matetsi_root), dst)

            wic, detections = scoutbot.pipeline(
                str(dst),
                config=config,
                backend_device="cpu",
                loc_thresh=rc["loc_thresh"],
                slice_height=rc["slice_height"],
                slice_width=rc["slice_width"],
                overlap_height_ratio=rc["overlap_height_ratio"],
                overlap_width_ratio=rc["overlap_width_ratio"],
                perform_standard_pred=rc["perform_standard_pred"],
                postprocess_class_agnostic=rc["postprocess_class_agnostic"],
            )
            baselines.append(
                {
                    "name": name,
                    "tier": tier,
                    "wic": wic,
                    "detections": detections,
                    "baseline_detection_count": len(detections),
                }
            )
            image_records.append(
                {
                    "name": name,
                    "tier": tier,
                    "md5": model_store._md5(dst),
                    "source_density": rec["_count"],
                    "baseline_detection_count": len(detections),
                }
            )

    (out_dir / "baselines.jsonl").write_text(
        "".join(json.dumps(b) + "\n" for b in baselines)
    )
    runtime = collect_runtime(Path(__file__).resolve().parents[1] / "uv.lock")
    manifest = build_manifest(config, per_tier, image_records, runtime, model_md5)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "README.md").write_text(render_readme(config, per_tier))

    # Governance gate — fail the build if any source token leaked into an emitted file.
    assert_no_source_identifiers(
        out_dir, source_segments(used_records, jsonl_source_root, matetsi_root)
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a scoutbot test corpus from Matetsi.")
    parser.add_argument("--matetsi-jsonl", required=True, type=Path)
    parser.add_argument("--jsonl-source-root", required=True)
    parser.add_argument("--matetsi-root", required=True, type=Path)
    parser.add_argument("-o", "--out", required=True, type=Path)
    parser.add_argument("--per-tier", type=int, default=5)
    parser.add_argument("--config", default="v3")
    parser.add_argument(
        "--force", action="store_true", help="overwrite a non-empty output images/ dir"
    )
    args = parser.parse_args(argv)
    manifest = build_corpus(
        args.matetsi_jsonl,
        args.jsonl_source_root,
        args.matetsi_root,
        args.out,
        args.per_tier,
        args.config,
        args.force,
    )
    print(f"Wrote corpus with {len(manifest['images'])} images to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_build_test_corpus.py -k end_to_end -v`
Expected: PASS.

- [ ] **Step 5: Run the whole fast suite (guard against regressions)**

Run: `uv run pytest tests/test_build_test_corpus.py -v && uv run pytest -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Lint/type check**

Run: `uv run ruff format tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run ruff check tools/build_test_corpus.py tests/test_build_test_corpus.py && uv run mypy tools/build_test_corpus.py && uv run bandit -c pyproject.toml -r tools/`
Expected: clean.

- [ ] **Step 7: Review checkpoint** — suggested message:
  `feat(tools): add build_corpus CLI orchestration (select→strip→freeze→write→scan)`

---

### Task 7: Consumer regression test + `stack_matches` guard

**Files:**
- Modify: `tools/build_test_corpus.py` (add `stack_matches`)
- Create: `tests/test_corpus_regression.py`
- Test: `tests/test_build_test_corpus.py` (fast unit test for `stack_matches`)

**Interfaces:**
- Consumes: `build_manifest`/`collect_runtime` shapes; `scoutbot.pipeline`, `model_store._md5`, `get_models_dir`, `CONFIGS`
- Produces: `stack_matches(manifest: dict, current_runtime: dict, current_model_md5: str) -> tuple[bool, str]`

- [ ] **Step 1: Write the failing fast unit test** (in `tests/test_build_test_corpus.py`)

```python
def _manifest_for_guard():
    runtime = {
        "lockfile_sha256": "L", "torch": "2.13.0", "torchvision": "0.28.0",
        "ultralytics": "8.4.0", "sahi": "0.11.36", "numpy": "2.0.0",
        "python": "3.13.1", "platform_system": "Darwin", "platform_machine": "arm64",
    }
    return {"model": {"md5": "M"}, "runtime": runtime}, dict(runtime)


def test_stack_matches_true_on_identical_stack():
    manifest, current = _manifest_for_guard()
    ok, reason = btc.stack_matches(manifest, current, current_model_md5="M")
    assert ok and reason == ""


def test_stack_matches_false_on_model_md5():
    manifest, current = _manifest_for_guard()
    ok, reason = btc.stack_matches(manifest, current, current_model_md5="OTHER")
    assert not ok and "model" in reason


def test_stack_matches_false_on_lock_and_platform():
    manifest, current = _manifest_for_guard()
    current["lockfile_sha256"] = "DIFFERENT"
    ok, reason = btc.stack_matches(manifest, current, current_model_md5="M")
    assert not ok and "lockfile_sha256" in reason
    manifest, current = _manifest_for_guard()
    current["platform_machine"] = "x86_64"
    ok, reason = btc.stack_matches(manifest, current, current_model_md5="M")
    assert not ok and "platform_machine" in reason
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_build_test_corpus.py -k stack_matches -v`
Expected: FAIL — `stack_matches` not defined.

- [ ] **Step 3: Implement `stack_matches`** (append to `tools/build_test_corpus.py`)

```python
# Keys that must match for an exact-equality baseline comparison to be authoritative. A different
# OS/BLAS/wheel can shift the last ULP and masquerade as scoutbot drift, so any mismatch → skip.
_GUARD_KEYS = (
    "lockfile_sha256", "torch", "torchvision", "ultralytics", "sahi", "numpy",
    "python", "platform_system", "platform_machine",
)


def stack_matches(
    manifest: dict, current_runtime: dict, current_model_md5: str
) -> tuple[bool, str]:
    """Return (ok, reason). ok=True means same model + same lock + compatible platform stack."""
    if current_model_md5 != manifest["model"]["md5"]:
        return False, "model md5 differs"
    recorded = manifest["runtime"]
    for key in _GUARD_KEYS:
        if recorded.get(key) != current_runtime.get(key):
            return False, f"{key} differs (baseline {recorded.get(key)!r} != current {current_runtime.get(key)!r})"
    return True, ""
```

- [ ] **Step 4: Run to verify the unit test passes**

Run: `uv run pytest tests/test_build_test_corpus.py -k stack_matches -v`
Expected: PASS (4 assertions across 3 tests).

- [ ] **Step 5: Write the slow, skip-if-absent regression test**

```python
# tests/test_corpus_regression.py
"""Replay frozen CPU baselines from a scoutbot test corpus and assert inference has not drifted.

Slow + opt-in: skipped unless SCOUTBOT_TEST_CORPUS points at a corpus directory, so public CI
(which has no corpus) is unaffected. See docs/plans/2026-07-31-scoutbot-test-corpus-builder.md.
"""

import json
import os
from pathlib import Path

import pytest

import build_test_corpus as btc
import scoutbot
from scoutbot import model_store
from scoutbot.registry.paths import get_models_dir

_CORPUS = os.getenv("SCOUTBOT_TEST_CORPUS")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _CORPUS, reason="SCOUTBOT_TEST_CORPUS not set"),
]


def _load(corpus: Path):
    manifest = json.loads((corpus / "manifest.json").read_text())
    baselines = {
        b["name"]: b
        for b in (
            json.loads(line) for line in (corpus / "baselines.jsonl").read_text().splitlines() if line.strip()
        )
    }
    return manifest, baselines


def test_corpus_baselines_do_not_drift():
    corpus = Path(_CORPUS)
    manifest, baselines = _load(corpus)
    config = manifest["model"]["config"]

    model_path = get_models_dir() / manifest["model"]["name"]
    if not model_path.exists():
        pytest.skip(f"model {model_path} absent; run: uv run python -m scoutbot fetch --config {config}")
    current_md5 = model_store._md5(model_path)
    current_runtime = btc.collect_runtime(Path(__file__).resolve().parents[1] / "uv.lock")

    ok, reason = btc.stack_matches(manifest, current_runtime, current_md5)
    if not ok:
        pytest.skip(f"baseline frozen under a different stack ({reason}); re-run the builder to re-freeze")

    # run_config must still match what was frozen (surfaces a CONFIGS change directly).
    assert btc.run_config_for(config) == manifest["run_config"]

    rc = manifest["run_config"]
    for name, expected in baselines.items():
        wic, detections = scoutbot.pipeline(
            str(corpus / "images" / name),
            config=config,
            backend_device="cpu",
            loc_thresh=rc["loc_thresh"],
            slice_height=rc["slice_height"],
            slice_width=rc["slice_width"],
            overlap_height_ratio=rc["overlap_height_ratio"],
            overlap_width_ratio=rc["overlap_width_ratio"],
            perform_standard_pred=rc["perform_standard_pred"],
            postprocess_class_agnostic=rc["postprocess_class_agnostic"],
        )
        assert wic == expected["wic"], f"{name}: wic drifted"
        assert detections == expected["detections"], f"{name}: detections drifted"
```

- [ ] **Step 6: Verify the regression test is collected but skipped (no corpus set)**

Run: `uv run pytest tests/test_corpus_regression.py -v`
Expected: 1 skipped (`SCOUTBOT_TEST_CORPUS not set`).

- [ ] **Step 7: Lint/type check**

Run: `uv run ruff format tools/build_test_corpus.py tests/ && uv run ruff check tools/build_test_corpus.py tests/test_corpus_regression.py tests/test_build_test_corpus.py && uv run mypy tools/build_test_corpus.py`
Expected: clean.

- [ ] **Step 8: Review checkpoint** — suggested message:
  `feat(tests): add corpus drift regression (skip-if-absent) + stack_matches guard`

---

### Task 8: Documentation pass

**Files:**
- Modify: `docs/development.rst`, `docs/roadmap.rst`, `CHANGELOG.rst`

**Interfaces:** none (docs only).

- [ ] **Step 1: `docs/development.rst`** — under Testing (or a new "Test corpus" subsection), document the builder and regression test. Add:

```rst
Internal test corpus
~~~~~~~~~~~~~~~~~~~~

``tools/build_test_corpus.py`` builds a density-tiered, GPS-stripped scoutbot test corpus from the
internal Matetsi survey (maintainer-only; requires local Matetsi and a locally-present model — it
performs no network I/O). It freezes CPU inference baselines a regression test replays.

.. code-block:: bash

   uv run python tools/build_test_corpus.py \
     --matetsi-jsonl <scoutbot_results.jsonl> \
     --jsonl-source-root /home/brandon/matetsi_images \
     --matetsi-root <local>/matetsi/images \
     -o <corpus-dir> [--per-tier 5] [--config v3]

The regression test is slow and opt-in — it is skipped unless ``SCOUTBOT_TEST_CORPUS`` points at a
corpus directory, so public CI is unaffected:

.. code-block:: bash

   SCOUTBOT_TEST_CORPUS=<corpus-dir> uv run pytest tests/test_corpus_regression.py

It asserts exact CPU parity to the frozen baselines only when the model, ``uv.lock``, key library
versions, and platform match the manifest; otherwise it skips (re-run the builder to re-freeze).
```

- [ ] **Step 2: `docs/roadmap.rst`** — update the *Benchmark corpus design* item to reference the builder (density tiers + comparability metadata delivered; status in-progress) and add the deferred follow-ups: D1 hosting + public clearance (high-visibility, permission-gated), D2 human ground-truth annotation (cross-ref pipeline-design DS-5), D3 multi-model baselines (`v3-cls`/Core ML), and fold D4 per-tier benchmark metadata into the existing benchmark-`--summary-json` item. (The prior `roadmap.rst` edit from the spec review may already cover most of this — reconcile rather than duplicate.)

- [ ] **Step 3: `CHANGELOG.rst`** — under the unreleased/0.5.0 section, add an internal/testing entry:

```rst
- Added an internal test-corpus builder (``tools/build_test_corpus.py``) and an opt-in,
  skip-if-absent CPU drift-regression test (``tests/test_corpus_regression.py``, gated on
  ``SCOUTBOT_TEST_CORPUS``). Adds ``piexif`` to the ``test`` extra.
```

- [ ] **Step 4: Build the docs to verify no warnings**

Run: `uv sync --extra docs && uv run sphinx-build -M html docs docs/build`
Expected: build succeeds; no new warnings for the edited pages.

- [ ] **Step 5: Review checkpoint** — suggested message:
  `docs: document test-corpus builder, regression test, roadmap follow-ups`

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- Path resolution (`--jsonl-source-root`, full-suffix) → Task 1.
- Density tiers (exact bands, deterministic, `--per-tier 5`) → Task 2.
- EXIF strip + softened losslessness (dimensions check) → Task 3.
- Manifest `run_config`/`runtime`/lockfile hash + model preflight + `source_density` vs `baseline_detection_count` → Tasks 4 & 6.
- Governance no-source-identifiers scan → Task 5.
- Flat `images/` layout + CLI + benchmark-readiness (`--images <corpus>/images`) → Task 6.
- Regression test (slow, skip-if-absent, stack guard, exact CPU parity) → Task 7.
- Docs + `piexif` dependency → Tasks 3 & 8.
- Deferred D1–D4 → Task 8 roadmap.

**2. Placeholder scan** — no TBD/TODO; every code step carries complete code and exact commands.

**3. Type consistency** — helper signatures are stable across tasks: `resolve_local`, `select_tiers`, `strip_and_verify`, `collect_runtime`/`run_config_for`/`assert_model_present`/`build_manifest`, `source_segments`/`assert_no_source_identifiers`, `build_corpus`/`main`, `stack_matches`. `run_config_for` maps `CONFIGS['thresh']`→`loc_thresh` consistently in the builder freeze (Task 6) and the regression replay (Task 7). `_count` is added by `select_tiers` and read by `build_corpus` for `source_density`.

**Out of scope (per spec):** corpus hosting/fetch (D1), human ground-truth (D2), `v3-cls`/Core ML baselines (D3), deeper benchmark-harness/manifest integration (D4).
