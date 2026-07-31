# Scoutbot Test-Corpus Builder — Design Spec

**Date:** 2026-07-31
**Status:** approved for planning
**Topic:** A maintainer tool that turns the internal Matetsi survey into a small, density-tiered,
GPS-stripped scoutbot test corpus with frozen CPU baselines, consumed by a skip-if-absent
regression test and available for benchmarking.

## Problem

Scoutbot's only bundled imagery is `examples/` — ~7 Git-LFS frames. The slow suite runs inference
on them but only asserts "does not crash": there is no check that inference output has not
*drifted*, and no density variety, so benchmark numbers ride on a biased 7-image set. The roadmap's
*Benchmark corpus design* item asks for exactly this: a small corpus with explicit density
categories and comparability metadata.

A complete real survey is now held internally — **Matetsi** (see
`pipeline-design/docs/design/test-data-matetsi.md`): ~18k 45 MP bilateral OCC frames plus a full
scoutbot detection JSONL (18,296 images, 98,239 detections, density median 4 / p90 11 / max 175,
6% empty). It is the natural source for a representative corpus — but it is **internal-only** and
carries poaching-sensitive wildlife GPS, so anything derived from it for scoutbot must be
GPS-stripped and permission-gated before any public hosting.

## Goal

Produce a small, reproducible, density-tiered scoutbot test corpus from the internal Matetsi survey
that (a) carries frozen CPU baselines so a regression test can assert inference has not drifted, and
(b) has per-image tier metadata for comparable benchmarking — while stripping all location data and
keeping the artifact internal until permission is cleared.

## Scope (in scope)

- A maintainer builder, `tools/build_test_corpus.py`, that selects, strips, and freezes.
- A self-contained corpus directory: tiered images + `baselines.jsonl` + `manifest.json` + `README.md`.
- A skip-if-absent regression test that asserts CPU inference parity to the frozen baselines.
- Benchmark-readiness (tier metadata) so existing benchmarks can point at the corpus.
- Unit tests for the builder's pure functions (no Matetsi dependency).

## Global constraints

- Runtime: always `uv run`; never bare `python`/`pytest`/`ruff`/`mypy`.
- The user handles all `git add`/`commit`/`push`. Plans stop at a review checkpoint with a
  suggested commit message; no commit steps.
- Plans and specs live under `docs/plans/` and `docs/specs/` (repo override of skill defaults).
- Test-first for code changes (red before green).
- **Matetsi is internal-only.** Every corpus artifact must be GPS/EXIF-stripped and location-scrubbed
  (opaque IDs), and the builder performs **no network I/O** — no upload, and no model download: the
  model must already be local (see the Data-flow model preflight), else the build fails with fetch
  instructions. Public hosting is out of scope and permission-gated.
- Baselines are frozen and asserted on **CPU only** (bit-reproducible; MPS/CUDA are not).

## Locked decisions

1. **One corpus serves both regression and benchmark.** Frozen baselines + tier metadata come from
   the same tiered image set.
2. **Local / env-var hosting, skip-if-absent.** The corpus lives on disk; tests read it via
   `SCOUTBOT_TEST_CORPUS` and skip when unset. No bucket in this scope.
3. **Baselines re-run on CPU** (approach A): the Matetsi JSONL is a *selection index* only; the
   frozen truth is produced by running scoutbot on the stripped images on CPU, and the regression
   test asserts exact equality on CPU.
4. **Flat `images/` directory with opaque, tier-prefixed IDs** (`<tier>_NN.jpg`) — no Matetsi
   filenames or paths in the artifact; the tier lives in the manifest, and a *flat* directory keeps
   the corpus compatible with tools that enumerate a single image directory (the benchmark's
   `iterdir`, see §Benchmark).
5. **`v3` only** to start (single-class detector); multi-model is a documented extension.
6. **Builder lives in `tools/`** (maintainer tool, run internally where Matetsi is available).

## Density tiers

Matetsi is uniformly 45 MP, so the "large / high-resolution" roadmap axis is satisfied by
construction and **detection density** is the differentiating axis. Four tiers, bucketed from the
Matetsi scoutbot JSONL by per-image detection count:

Bands are **exact inclusive integer bounds** (not approximate), derived from the Matetsi
distribution (median 4, p90 11, max 175) so the build is deterministic:

| Tier | Selection rule (from JSONL detection count `c`) |
|---|---|
| `empty` | `c == 0` |
| `typical` | `3 <= c <= 6` |
| `dense` | `9 <= c <= 13` |
| `extreme` | the highest-`c` records (always including the global-max image) |

Continuity with the existing `examples/` is preserved by leaving `examples/` untouched (it remains
the smoke tier). Default **`--per-tier 5`** (5 images/tier → 20 total).

**Selection is deterministic.** For `empty`, `typical`, `dense`: sort the bucket by `(c, filepath)`
and pick `N` **evenly-spaced** indices (not first-`N`, which would cluster at the bucket edge). For
`extreme`: sort by `(c, filepath)` descending and take the top `N`, guaranteeing the global-max
record is included. The `filepath` tiebreak is used only internally for ordering and is never
emitted. The same inputs always yield the same corpus.

## Components

### `tools/build_test_corpus.py`

CLI:

```
uv run python tools/build_test_corpus.py \
  --matetsi-jsonl <path-to-scoutbot_results.jsonl> \
  --jsonl-source-root /home/brandon/matetsi_images \
  --matetsi-root /Volumes/.../matetsi/images \
  -o <corpus-dir> [--per-tier N] [--config v3]
```

- `--matetsi-jsonl` — the scoutbot detection JSONL (the selection index). The builder selects
  **only from JSONL records**; it never walks the image tree, so images outside the JSONL (the
  curated `*_examples/`, `test_batch_50/`) are ignored. This also pins the count to the JSONL's
  18,296 records rather than the ~18,401 in the `images/` subtree or ~18,501 in the full copy.
- `--jsonl-source-root` — the absolute prefix the JSONL `filepath`s were written under (they are
  foreign absolute paths, e.g. `/home/brandon/matetsi_images/...`). Required.
- `--matetsi-root` — the local **`images/` subtree** of the held Matetsi copy (not the top-level
  `matetsi/` directory).

**Path resolution is by full relative suffix — deterministic, not a basename search.** ~72% of
Matetsi basenames collide across left/right and flights, so basename matching is unsafe; the
*full-suffix join* sidesteps that entirely. For each record the builder strips `--jsonl-source-root`
from `filepath` and joins the remaining relative suffix (e.g. `141025_day1/day1/flight1/left/…JPG`)
onto `--matetsi-root`. The result is a single deterministic path; the builder requires it to
**exist** and fails hard otherwise. It also fails hard if `filepath` does not start with
`--jsonl-source-root` (a foreign prefix would mean the wrong source-root was passed).

Pure, unit-testable helpers (no Matetsi/network needed):
- `resolve_local(filepath, jsonl_source_root, matetsi_root) -> Path` — strip prefix, join suffix;
  raises if `filepath` lacks the prefix or the joined path does not exist.
- `select_tiers(records, per_tier) -> dict[str, list[record]]` — bucket + deterministic pick.
- `build_manifest(...) -> dict` — assemble the manifest.
- `strip_and_verify(src, dst)` — remove the EXIF block, then assert no EXIF/GPS tag survives.

The tool imports `scoutbot.pipeline` to freeze baselines, and uses **`piexif`** (added to the
`test`/tools dependency surface) to strip EXIF via `piexif.remove()` (metadata-only; it does not
re-encode pixels). Because baselines are always frozen from the **stripped** copy, the corpus is
self-consistent regardless of whether stripping is bit-lossless — so no strong losslessness claim is
made; a builder unit test asserts only that the stripped image still decodes and keeps its
dimensions.

### Corpus layout

```
<corpus-root>/
  images/                        # FLAT — tier is encoded in the name and recorded in the manifest
    empty_01.jpg  typical_01.jpg  dense_01.jpg  extreme_01.jpg  …
  baselines.jsonl     # {name, tier, wic, detections:[{l,c,x,y,w,h}], baseline_detection_count}
  manifest.json       # provenance + run config + runtime stack (schema below)
  README.md           # internal-derived, GPS-stripped, not-cleared-for-public note
```

The directory is **flat** (not `images/<tier>/`) so a single-directory enumerator finds every
image; the tier is carried by the filename prefix and the manifest.

### `manifest.json` schema

```json
{
  "schema": "scoutbot:test-corpus:1.0",
  "created": "2026-07-31T00:00:00Z",
  "source": "matetsi (internal; GPS-stripped; opaque IDs)",
  "per_tier": 5,
  "model": {"config": "v3", "name": "yolov8-onecls.kaza.20260729.pt", "md5": "<pin>"},
  "run_config": {
    "backend_device": "cpu",
    "loc_thresh": 0.45,
    "slice_height": 512, "slice_width": 512,
    "overlap_height_ratio": 0.25, "overlap_width_ratio": 0.25,
    "perform_standard_pred": false, "postprocess_class_agnostic": true
  },
  "runtime": {
    "scoutbot_version": "0.5.0", "scoutbot_commit": "<git-sha>",
    "python": "3.13.x", "platform": "<platform.platform()>",
    "torch": "2.13.0", "torchvision": "0.28.0", "ultralytics": "8.4.x",
    "sahi": "0.11.36", "numpy": "2.x",
    "lockfile_sha256": "<sha256(uv.lock)>"
  },
  "images": [
    {"name": "dense_02.jpg", "tier": "dense", "md5": "<jpg-md5>",
     "source_density": 11, "baseline_detection_count": 11}
  ]
}
```

The `schema` id uses a **scoutbot-namespaced** version tag (`scoutbot:test-corpus:MAJOR.MINOR`) —
not the ecosystem `osw:` namespace, since this is a tool-internal artifact, not a shared standard.

Provenance depth matters because the regression assertion is *exact* (§Regression test): inference
output depends on the model **and** the SAHI/threshold run config **and** the dependency stack, so
all three are captured. `run_config` records the *effective* inference parameters in force when the
baselines were frozen. `pipeline()` does not read these from `CONFIGS` — they are its kwargs — so the
builder reads them from `CONFIGS[config]` (mapping `config['thresh']` → the `loc_thresh` kwarg, plus
`slice_*`, `overlap_*`, `perform_standard_pred`, `postprocess_class_agnostic`) and passes them
explicitly to `pipeline()`, recording the same values here. Recording them makes a config change
visible in a diff and lets the regression test replay the exact run. `runtime` records the
interpreter, platform, key library versions, scoutbot version + commit, and a **`uv.lock` hash** —
enough to interpret an exact-equality failure (real drift vs. a changed stack). Per image: `source_density` is the Matetsi JSONL count (**why the image was
selected** — reference only), and `baseline_detection_count` is the number of detections in the
**frozen CPU baseline** (**what the baseline contains**); the two are recorded separately and may
legitimately differ.

## Data flow (builder)

0. **Model preflight (offline).** Before any inference, verify the target model is already present
   and valid in the OSW model store: the local model file for `config` exists **and** its md5 equals
   the **pinned expected md5 from the current model config / registry metadata** (whatever the model
   store treats as authoritative — the config's pinned hash, else the registry entry — not a
   hard-coded dict key, which some configs such as the Core ML bundles do not expose). If absent or
   mismatched, **fail with instructions** (`uv run python -m scoutbot fetch --config <config>`)
   rather than downloading — `pipeline()` would otherwise fetch on a cache miss, violating the
   no-network constraint. The preflight itself makes no network call (local file + pinned-hash check
   only; if a config exposes no pinned hash, fall back to a presence-only check).
1. **Select** — parse the Matetsi JSONL, bucket by detection count, deterministically pick `N`/tier
   (§ Density tiers). Selection is over **JSONL records only** — never a directory walk.
2. **Resolve** — for each selected record, `resolve_local()` maps its foreign absolute `filepath`
   to the local file by relative suffix under `--matetsi-root`; fail hard if the joined path does not
   exist or `filepath` lacks the `--jsonl-source-root` prefix.
3. **Copy + strip** — copy each resolved JPEG to `images/<tier>_NN.jpg` (flat, opaque ID);
   `piexif.remove()` to drop the EXIF block; **assert no EXIF/GPS tag survives** (fail hard).
4. **Freeze** — for each stripped image, `scoutbot.pipeline(img, config, backend_device='cpu',
   loc_thresh=CONFIGS[config]['thresh'], slice_height=…, …)` (run params passed explicitly from
   `CONFIGS[config]`) → `(wic, detections)`; write `baselines.jsonl` (including
   `baseline_detection_count`).
5. **Manifest + README** — write `manifest.json` (model + `run_config` + `runtime` stack + lockfile
   hash + per-image records) and a provenance/governance `README.md`.

Idempotent: same inputs → same corpus. No network I/O.

## Governance safeguards (baked in)

- **EXIF strip** + post-strip assertion that no EXIF/GPS tag survives on any image.
- **No source identifiers in any emitted file.** `manifest.json`, `baselines.jsonl`, and `README.md`
  must contain **no** original Matetsi filenames, absolute source paths, flight/day path segments,
  GPS rows, or linked-event identifiers — only opaque IDs, tier labels, integer densities, and the
  generic `source: "matetsi (internal)"` string. The builder asserts this before finishing by
  scanning emitted text for: any `--jsonl-source-root` / `--matetsi-root` substring, any original
  source path segment (the day/flight/side directory names and original basenames of the selected
  records). It does **not** blanket-forbid the `/` character — the README legitimately carries
  commands, env vars, and corpus-relative paths — so the scan targets source-derived tokens, not
  path separators. Any hit fails the build.
- **Opaque IDs** — `<tier>_NN.jpg`; the mapping back to Matetsi filenames is never written into the
  corpus.
- `README.md` classifies the corpus as internal-derived, GPS-stripped, and **not cleared for public
  distribution** until permission is established.
- Builder does zero network I/O; hosting is a separate, gated step.

## Regression test (consumption)

`tests/test_corpus_regression.py`:

- **Slow-marked** and **skipped unless `SCOUTBOT_TEST_CORPUS` points at a corpus dir** (so public
  GitHub CI is unaffected).
- Load `manifest.json` + `baselines.jsonl`.
- **Stack guard — makes the exact assertion interpretable.** Exact detection equality is only
  authoritative on the **same lock and a compatible CPU/platform stack** — a different OS, BLAS, or
  torch wheel can shift the last ULP and look like scoutbot drift. So before comparing, the test
  checks the current runtime against `manifest.runtime` and **skips with an explicit message**
  ("baseline frozen under a different stack — re-run the builder to re-freeze") on any mismatch of:
  the model md5, the `uv.lock` sha256, the recorded key library versions (`torch`, `torchvision`,
  `ultralytics`, `sahi`, `numpy`, `python`), and the platform family (`platform.system()` /
  `platform.machine()`). A pass therefore means same lock + same model + compatible platform — under
  which CPU inference is bit-reproducible and any inequality is real drift.
- When the stack matches: re-run inference by passing the manifest's `run_config` values explicitly
  as `pipeline()` kwargs — `pipeline(img, config=manifest.model.config, backend_device='cpu',
  loc_thresh=..., slice_height=..., ...)` — so the replay is independent of any drift between
  `CONFIGS` defaults and the frozen `run_config` (the test also asserts the current `CONFIGS[config]`
  still equals `run_config`, surfacing a config change directly). Then assert
  `(wic, detections) == baseline` **exactly** (CPU is bit-reproducible; the dill-removal parity work
  established this).
- Size: 5/tier × 4 tiers = 20 images × ~1 min CPU ≈ 20 min — a slow/internal gate, run where the
  corpus is present.

## Benchmark integration (light)

`benchmark_concurrency.py` discovers images with `Path.iterdir()` (top-level only), so the **flat**
`images/` layout is deliberate: `benchmark_concurrency.py --images <corpus>/images` finds every
corpus image and yields whole-corpus perf today, with **no harness change**. (A tiered
`images/<tier>/` layout would have silently found zero images under the current `iterdir` — the flat
layout avoids that trap.) Per-tier benchmarking, and recording tier labels into `--summary-json`,
requires the harness to read the manifest — a manifest-aware enhancement deferred to D4.

## Testing the builder

The builder depends on Matetsi (internal; not in CI), so **unit-test the pure functions** with tiny
synthetic inputs — no Matetsi, no network:
- `resolve_local` — with two sibling dirs holding the **same basename**, full-suffix resolution
  still lands on the correct file (demonstrating collisions are a non-issue by construction); a
  `filepath` whose joined suffix has no local file raises; a `filepath` not under
  `--jsonl-source-root` raises.
- `select_tiers` on a small synthetic JSONL → correct buckets, deterministic pick, evenly-spaced.
- `strip_and_verify` on a synthetic JPEG carrying a fake GPS EXIF tag → tag removed, verification
  passes; a negative case where a residual tag makes it raise; and an assertion that the stripped
  image still **decodes and keeps its dimensions** (the softened losslessness claim).
- `build_manifest` → schema shape, required keys, per-image records (`source_density` and
  `baseline_detection_count` both present and independent).
- A **no-source-identifiers** check over emitted `manifest.json` / `baselines.jsonl` / `README.md`
  text: no `--jsonl-source-root` / `--matetsi-root` substrings and no original source path segments
  or basenames of the selected records; a positive case confirms a README containing a command with
  `/` and an env var **passes** (the scan targets source tokens, not path separators).

The end-to-end build (needs Matetsi) is a documented manual/internal step, not a CI gate.

## Error handling / determinism

- Deterministic selection (stable sort + fixed even-spacing) → reproducible corpus.
- A missing/corrupt selected image, or an insufficiently-populated tier bucket, fails the build with
  a clear message (never silently produces a short tier).
- The GPS/EXIF assertion is a hard failure (governance-critical).

## Documentation impact

- `docs/development.rst` — document the corpus builder, `SCOUTBOT_TEST_CORPUS`, and the regression
  test (opt-in, internal).
- `docs/roadmap.rst` — link the *Benchmark corpus design* item to this work; add the deferred
  follow-ups below.
- `CHANGELOG.rst` — the builder + regression test as an internal/testing addition.
- `pyproject.toml` — `piexif` added to the test/tools dependency surface.

## Deferred / out-of-scope (explicit discussion)

Each item below is intentionally excluded from this spec, with the rationale, what it would entail,
its prerequisites, and a recommendation on roadmap placement.

### D1. Corpus hosting + public clearance
**Why deferred:** the corpus is Matetsi-derived; redistribution permission is not established, and
GPS-stripping alone does not grant rights to publish the imagery. Local/env-var consumption unblocks
the builder + regression test without a hosting or permission decision now.
**What it entails:** (a) establish imagery redistribution permission; (b) choose a home — a private
bucket first (credentialed fetch, reproducible across internal machines/CI), then optionally the
public `osw-public-models`-style bucket once cleared; (c) an MD5-pinned fetch path (reuse the model
`fetch` pattern) and a `SCOUTBOT_TEST_CORPUS` auto-fetch fallback.
**Prerequisite:** permission clearance (external/legal), not an engineering blocker.
**Roadmap:** **Yes — add a high-visibility item** (it gates turning the internal corpus into a
shared/CI asset).

### D2. Human ground-truth annotation
**Why deferred:** the frozen baselines are scoutbot's *own* CPU output — a drift-regression anchor,
not independent truth. They cannot validate accuracy, false-negative rate, or bias (that would be
circular).
**What it entails:** exhaustive independent human annotation of a Matetsi subset (species, counts,
positions), which then enables accuracy/PR-curve evaluation and feeds pipeline-design's DS-5 /
Stage-8 bias work.
**Prerequisite:** annotation effort + tooling; overlaps with pipeline-design's fixture provenance
matrix.
**Roadmap:** **Yes — add an item**, cross-referencing pipeline-design's DS ladder; note it is the
prerequisite for any *accuracy* (not just drift) claim.

### D3. `v3-cls` / Core ML baselines
**Why deferred:** starting with `v3` (single-class) keeps the first corpus and regression small and
reproducible. Multi-class (`v3-cls`) and Core ML (`v3-coreml`) add model variants and, for Core ML,
non-bit-reproducibility (MPS/ANE → tolerance, not exact).
**What it entails:** parameterize the builder/manifest over `config`, freeze per-config baselines,
and give the regression test per-config tolerance (exact for `v3`/CPU, strict-tolerance for Core ML).
The manifest already namespaces by `model.config`, so this is an extension, not a redesign.
**Prerequisite:** the `v3` corpus landing first.
**Roadmap:** **Optional item / note** — record it as a natural extension once `v3` is proven.

### D4. Deeper benchmark-harness integration
**Why deferred:** the corpus is already benchmark-ready via `--images`; wiring tier metadata *into*
benchmark summaries (so per-tier throughput and NMS-heavy behavior are recorded and comparable
across runs) is separable and belongs with the existing benchmark-schema follow-up.
**What it entails:** teach `benchmark_concurrency.py` (and `--summary-json`) to read the corpus
manifest and emit per-tier metrics; fold into the existing "record ML-stack versions in
`--summary-json`" benchmark-schema roadmap item.
**Prerequisite:** none technical; low priority.
**Roadmap:** **Fold into the existing benchmark-schema item** rather than a new one.

## Roadmap follow-ups (to add)

- Update **Benchmark corpus design** — note the builder spec/tool addresses it (density tiers +
  comparability metadata), status in-progress.
- Add **Test-corpus hosting + public clearance** (D1) — high-visibility; permission-gated.
- Add **Human ground-truth annotation of a survey subset** (D2) — cross-ref pipeline-design DS-5.
- Note **multi-model corpus baselines** (D3) under the corpus/benchmark area.
- Fold **per-tier benchmark metadata** (D4) into the existing benchmark-`--summary-json` item.
