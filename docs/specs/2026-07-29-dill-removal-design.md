# Dill Removal — Design Spec

**Date:** 2026-07-29
**Status:** approved for planning
**Topic:** Remove the runtime `dill` dependency by re-serializing the bundled KAZA `.pt` models.

## Problem

`dill>=0.4.0` is a runtime dependency in `[project.dependencies]`. Scoutbot's own code
never imports `dill` and never calls `torch.load` directly; the dependency exists only
because the two bundled KAZA `.pt` artifacts were serialized by an older Ultralytics that
embedded `dill` pickle references. When `YOLO(model_path)` unpickles a model, `dill` must be
importable or the load fails.

Carrying `dill` purely to deserialize two model files is unnecessary coupling. Re-serializing
those files with the current stack removes the references and lets the dependency go.

## Goal

Drop `dill` from the runtime dependency set, with the two `.pt` models re-serialized so they
load under `torch`/`YOLO()` without `dill` installed, and with inference output unchanged.

**This is a serialization migration, not a model change.** The v3 weights are identical
before and after; only the on-disk pickle format changes.

## Scope

In scope — the two PyTorch configs:

| Config | Cache file (current) | Cache file (new) |
|---|---|---|
| `v3` | `yolov8-onecls.kaza.pt` | `yolov8-onecls.kaza.20260729.pt` |
| `v3-cls` | `yolov8-cls.kaza.pt` | `yolov8-cls.kaza.20260729.pt` |

Out of scope:

- **Core ML configs (`v3-coreml`, `v3-cls-coreml`).** `.mlpackage` bundles are coremltools
  packages that never needed `dill`. They carry no `url`/`md5` and are exported on demand from
  their `source` `.pt`. After the new `.pt` lands they regenerate transparently; they need no
  re-host and no behavior change.
- **Model weights / training.** Untouched.
- **The dependency currency policy** and other roadmap items.

## Global constraints

- Runtime: always `uv run`; never bare `python`/`pytest`/`ruff`/`mypy`.
- The user handles all `git add`/`commit`/`push`. Plans stop at a review checkpoint with a
  suggested commit message; no commit steps.
- Test-first for code changes (red before green).
- Uploading artifacts to Azure/GCS is an outward-facing action performed by the **user**, not
  the agent. The agent produces and verifies artifacts locally and hands off an upload checklist.
- Artifacts carry a neutral build-date revision tag, not an implementation-named suffix. The tag
  is the re-serialization (build) date `20260729` in `YYYYMMDD` form — defined by the build
  date, not the upload date, so Azure and GCS carry identical, stable filenames.

## Decisions (locked)

1. **Re-serialization: Ultralytics `model.save()`** (approach A). Load the old `.pt` with
   `dill` available, then `YOLO(old).save(new)`. Ultralytics re-pickles the whole model object
   through the current stack without `dill` closures, preserving all metadata (class names,
   task, args). Rejected: hand-rebuilding a `state_dict` checkpoint (avoidable reconstruction
   risk) and a blind `torch.save(torch.load(...))` round-trip (does not reliably drop `dill`).
2. **Hosting: user uploads; GCS becomes primary, Azure fallback.** The current source order
   (Azure primary, GCS fallback) is swapped because the user has more control over the GCS
   resource. Same host directories (`…/scoutbot/`, `…/localizer/`); the artifact is
   disambiguated by filename suffix, not a new path segment.
3. **Naming: a neutral build-date revision tag on the filename, not a `/v2/` path or an
   implementation-named suffix.** A path like `/v2/` would collide with the model's own `v3`
   lineage, and a suffix like `.nodill` names an implementation fix that will read strangely
   once all artifacts are dill-free. The tag is the re-serialization date `20260729`
   (`YYYYMMDD`), e.g. `yolov8-onecls.kaza.20260729.pt`; it stays meaningful if the artifact is
   ever rebuilt for another serialization/toolchain reason. A distinct filename also means a
   user's local prior copy (`yolov8-*.kaza.pt`) is never overwritten or confused — the
   re-serialized file sits beside it under a new name. The tag is defined by the build date, not
   the upload date, so Azure and GCS carry identical names.
4. **Cache invalidation: pin the expected MD5 in `CONFIGS`.** Add an `md5` field per `.pt`
   config (the new file's hash) and make `fetch()` treat it as authoritative, so upgraded code
   auto-invalidates any stale cache. This also pins model integrity in code, which the current
   sidecar-only trust model lacks.

## Architecture / components

### Re-export procedure (one-shot, maintainer)

A documented, exact procedure captured in the implementation plan — **not** a committed script,
because after `dill` is removed such a script could not run in the project env and would be dead
code. Run in this repo's env (which still has `dill`) before the dependency is removed. For each
of `v3`, `v3-cls`:

1. `scoutbot fetch --config <cfg>` to obtain the current `.pt`.
2. `YOLO(old_pt).save(new_pt)` where `new_pt` is the date-tagged filename.
3. Verify (see Acceptance gate) and compute the MD5.
4. Write the `.md5` sidecar and the host-directory `INDEX.txt`.

Deliverables handed to the user: two date-tagged `.pt` files, two `.pt.md5` sidecars, one
`INDEX.txt`, and an upload checklist naming the exact GCS `…/scoutbot/` and Azure
`…/localizer/` object paths ("upload these exact bytes").

### `CONFIGS` (`scoutbot/model_store.py`)

For `v3` and `v3-cls`:

- `name` → the date-tagged filename (old `…kaza.pt` stays intact on disk).
- `sources` reordered: GCS first, Azure second.
- `url` / `md5_url` → the new date-tagged filenames in the same directories.
- New `md5` field holding the pinned hash of the new file.

Core ML configs are unchanged and carry no `md5` key.

### `fetch()` invalidation (`scoutbot/model_store.py`)

Make the pinned hash authoritative while leaving Core ML and unpinned configs behaving exactly
as today:

- **Registry short-circuit** (currently `_md5(cached) == entry.md5`): return the cached file
  only when `_md5(cached) == cfg['md5']`, falling back to `entry.md5` when a config has no
  `md5` pin (Core ML). A stale cache fails the check and falls through to download.
- **Post-download assertion:** after a successful source download, assert the hosted sidecar
  hash equals `cfg['md5']` (when pinned). A bad or partial upload fails loudly at fetch instead
  of silently serving mismatched bytes.

### Host-directory index

A plain-text `INDEX.txt` uploaded into each host directory, listing each model file, its MD5,
what it is, and the provenance note that the weights are identical to the earlier
`yolov8-*.kaza.pt` files and only the serialization changed.

### Dependency removal

Drop `dill>=0.4.0` and its explanatory comment from `pyproject.toml`; `uv lock`; re-sync.

## Data flow / sequencing

The one hard ordering constraint: `CONFIGS` will reference filenames that do not exist on the
hosts until the user uploads them. The work therefore splits at the upload hand-off.

**Pre-upload — agent produces and verifies locally:**

1. Re-export the two date-tagged files; produce sidecars, `INDEX.txt`, and pinned hashes.
2. Add the `fetch()` md5-pin invalidation logic + unit tests (mocked network).
3. Update `CONFIGS` (new filenames, GCS-primary, pinned `md5`).
4. Prime the local cache with the produced date-tagged files and run the **slow suite with
   `dill` uninstalled** — proving dill-removal and inference parity before anything is hosted.

**Hand-off:** deliver the artifacts + upload checklist. **User uploads to GCS and Azure.**

**Post-upload — real-network validation:**

5. Clean-cache `scoutbot fetch --config v3` / `v3-cls`: confirms GCS-primary resolves, Azure
   fails over, and the hosted sidecar hash matches the pinned `md5`.
6. Remove `dill`, `uv lock`, re-sync; run the full slow suite (user's Apple Silicon) and
   `uv audit --locked`.

## Error handling & rollback

- Re-export fails the no-dill load or the parity check → stop, keep `dill`, investigate. Old
  artifacts are untouched throughout.
- Post-upload hash mismatch → the pinned-md5 assertion fails loudly at fetch; fix the upload,
  no silent bad state.
- **Rollback** = revert the CONFIGS/`pyproject` commit: the old `yolov8-*.kaza.pt` files remain
  at their current (unsuffixed) URLs, so reverting fully restores the prior working state.

## Testing

- **Unit** (`tests/test_model_store.py`):
  - short-circuit re-downloads when the cached hash ≠ `cfg['md5']`;
  - short-circuit still serves when they match;
  - fetch raises when the hosted sidecar ≠ pinned `md5`;
  - a Core ML config (no `md5` key) keeps its current behavior.
- **No-dill gate (acceptance):** the slow inference suite passes in an environment where
  `dill` is not installed — first pre-upload with a primed cache, then again after the
  dependency is removed. Optionally a fast test that blocks `import dill` via `sys.modules`
  and loads a small re-saved fixture.
- **Parity:** the re-export step asserts identical detections (old vs new) on the `examples/`
  corpus, **CPU-only** for exact equality (MPS/CUDA comparisons use a strict tolerance); the
  existing slow inference assertions guard it thereafter.

## Acceptance gate

Both must hold for each re-exported file before upload:

1. It loads via `YOLO()` with `dill` **uninstalled**.
2. It produces **identical detections** to the old model on the `examples/` corpus. The parity
   comparison runs **CPU-only** (device `cpu`), on the same torch/Ultralytics stack, same
   config, and same corpus, where bitwise-identical output is expected because the weights are
   unchanged. Any comparison that involves an MPS or CUDA path instead uses a **strict numeric
   tolerance** rather than exact equality, since those backends are not bit-reproducible.

## Documentation

The GCS-primary / Azure-fallback swap is user-visible and must be surfaced across the docs, not
only in the changelog:

- `docs/models.rst` — new date-tagged filenames, provenance (same weights, re-serialized to
  remove the `dill` dependency), pinned MD5s, and the GCS-primary/Azure-fallback source order.
- `docs/configuration.rst` — the download source order (GCS primary, Azure fallback) as a
  user-facing behavior.
- `docs/security.rst` — model integrity is now pinned in code (hash in `CONFIGS`), an integrity
  strengthening over sidecar-only trust, and the trusted source order.
- `CHANGELOG.rst` — `dill` removed; models re-serialized (same weights); GCS primary; model-hash
  pinning.
- `docs/roadmap.rst` — move *Model re-export / dill removal* to the completed archive; note the
  source-order swap.
- `pyproject.toml` — `dill` dependency and comment removed.
- `docs/installation.rst` — a line on automatic re-download on upgrade if warranted.

## Out of scope (recorded deferrals)

- Re-hosting or re-generating the Core ML `.mlpackage` bundles (they are already dill-free and
  regenerate on demand).
- Migrating the sidecar-based integrity model to a stronger scheme beyond the in-code MD5 pin
  (e.g. signed manifests / SHA-256); the MD5 pin is the minimal change this work requires.
