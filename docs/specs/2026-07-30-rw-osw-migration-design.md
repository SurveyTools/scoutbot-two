# rw → osw Namespace Migration — Design Spec

**Date:** 2026-07-30
**Status:** approved for planning
**Topic:** Rebrand the ecosystem namespace in scoutbot-2026 from `rw` ("Rebel Workflow") to
`osw` ("Overhead Survey Workflow"): environment variables, the shared registry directory, the
registry schema id, and the public GCS bucket.

## Problem

scoutbot's local model registry is shared ecosystem infrastructure branded under the old
"Rebel Workflow" (`rw`) name: the `RW_REGISTRY_PATH` / `RW_MODELS_DIR` environment variables,
the `~/.rebel-workflow/` directory, the `rw:local-model-cache:1.0` schema id, and the
`rebel-pipeline-public-models` GCS bucket. The ecosystem is rebranding to **Overhead Survey
Workflow** (`osw`, `overheadsurvey.org`, namespace `osw = https://overheadsurvey.org/ns/workflow/`).
scoutbot is the first adopter and is heading toward a public release, so its user-facing names
should be consistent with the new namespace before it goes public.

## Goal

Replace every `rw`/`rebel` ecosystem-naming token in scoutbot with its `osw` equivalent, do a
clean break (no backward-compatibility with the old names), harden registry reads against
foreign files, and document the registry schema versioning contract for the ecosystem.

## Scope

In scope (scoutbot-2026 only):

- Env vars: `RW_REGISTRY_PATH` → `OSW_REGISTRY_PATH`, `RW_MODELS_DIR` → `OSW_MODELS_DIR`.
- Registry directory default: `~/.rebel-workflow/` → `~/.osw/`.
- Registry schema id: `rw:local-model-cache:1.0` → `osw:local-model-cache:1.0`.
- GCS bucket: `rebel-pipeline-public-models` → `osw-public-models` (path prefix `/scoutbot/`
  unchanged; the new bucket is already live and verified — see Validation).
- **Public-release metadata**: the `authors` entry in `pyproject.toml` (`Rebel Workflow Team`)
  and `author` in `docs/conf.py` (`Rebel Workflow Team`) → `Overhead Survey Workflow Team`.
- A registry **schema-guard** with a documented major/minor versioning contract.
- All source, tests, and user-facing docs that reference the above (see **Audit scope** for the
  exact boundary).

Out of scope:

- **Azure fallback** URLs/upload — still deferred (roadmap: *Dill-free model Azure fallback
  upload + failover validation*). The Azure container name (`localizer`) is not `rw`-branded and
  does not change here.
- **Sibling ecosystem repos** (`rw_modules`, `rw_survey_design`, `rw_runner_core`, etc.) — they
  still use `rw`; scoutbot adopts `osw` first. Cross-repo migration is a separate effort.
- **`platformdirs` / native per-OS locations** — deliberately not adopted; `~/.osw` mirrors the
  existing `~/.rebel-workflow` and `~/.cache/scoutbot` dotdir pattern (uniform on Windows too).
  An ecosystem-wide XDG/platformdirs decision is deferred.
- The bucket's non-catalog objects (`scout.loc.*.onnx`, the old undated `yolov8-*.kaza.pt`) —
  not referenced by scoutbot's `CONFIGS`.
- **Historical planning docs** — `docs/plans/**` and prior `docs/specs/**` (currently
  `docs/plans/2026-05-09-scoutbot-registry.md`, `docs/plans/2026-07-05-multi-source-model-fetch.md`,
  `docs/plans/2026-07-29-dill-removal.md`, ~53 references). These are **archival records of what
  was true when written** and are deliberately **not** rewritten; renaming them would falsify the
  history. Only this spec is exempt (it describes the new names). See **Audit scope**.

## Audit scope (completion criteria)

The rename is "complete" when every `rw`/`rebel` ecosystem-naming token is gone from the
**live surface** — source, tests, and user-facing docs/metadata — while archival planning docs
are left intact. Run:

```bash
grep -rniE "RW_|rebel[ _-]workflow|rebel-pipeline|\brw:" \
  scoutbot/ tests/ docs/*.rst README.md CONTRIBUTING.md CHANGELOG.rst pyproject.toml docs/conf.py
```

The completion target is zero **stale/live** references. Documenting and testing a rename,
however, *requires naming the old thing*, so a small set of matches are **intentional
old-name references** and are expected to remain:

- **Guard-rejection tests** — tests that write a `rw:local-model-cache:*` schema (or otherwise
  name an old identifier) to prove the schema-guard rejects a pre-migration registry
  (e.g. `test_guard_rejects_old_rw_family`, and the CLI clean-error test's fixture data).
- **Absence assertions** — tests that assert an old token is *gone*
  (e.g. `assert 'rebel-pipeline-public-models' not in url`).
- **Changelog / rename history** — the `CHANGELOG.rst` bullet documenting the rename, which
  names "Rebel Workflow" and `~/.rebel-workflow` to describe what changed and what is orphaned.

So the pass criterion is: **every remaining match is one of the intentional categories above; no
stale/live reference survives.** Confirm by filtering the intentional matches out and expecting
an empty result:

```bash
grep -rniE "RW_|rebel[ _-]workflow|rebel-pipeline|\brw:" \
  scoutbot/ tests/ docs/*.rst README.md CONTRIBUTING.md CHANGELOG.rst pyproject.toml docs/conf.py \
  | grep -vE "test_guard_rejects_old_rw_family|schema.{0,4}rw:local-model-cache|not in gcs\['url'\]|Rebel Workflow|~/\.rebel-workflow.* orphaned"
```

The `rebel[ _-]workflow` class matches the spaced "Rebel Workflow" prose form (e.g. the
`pyproject.toml` / `docs/conf.py` authors and the `scoutbot.py` CLI docstring) as well as the
`rebel-workflow` / `rebel_workflow` path/identifier forms — case-insensitive via `-i`.

`docs/plans/` and `docs/specs/` (except this file) are intentionally excluded from that audit.
The Wild Me / WildMeOrg attribution in `README.md` and `pyproject.toml` is unrelated to the
`rw` brand and is left unchanged.

## Global constraints

- Runtime: always `uv run`; never bare `python`/`pytest`/`ruff`/`mypy`.
- The user handles all `git add`/`commit`/`push`. Plans stop at a review checkpoint with a
  suggested commit message; no commit steps.
- Plans and specs live under `docs/plans/` and `docs/specs/` (repo override of skill defaults).
- Test-first for code changes (red before green).
- **Clean break**: no `RW_*` / `~/.rebel-workflow` backward-compatibility aliases. The user base
  is <5 and directly reachable; the old registry is simply orphaned (models remain in
  `~/.cache/scoutbot/`, and the first fetch re-registers under `~/.osw/`).
- Target names are fixed: dir `~/.osw`, env prefix `OSW_`, schema id family
  `osw:local-model-cache`, bucket `osw-public-models`.

## Locked decisions

1. **Names:** `~/.osw` + `OSW_` prefix. Chosen over `~/.config/osw` / `~/.local/share/osw`
   (XDG) and `platformdirs` because scoutbot already standardizes on plain dotdirs under `$HOME`
   (`~/.cache/scoutbot`, `~/.rebel-workflow`), so `~/.osw` is a zero-behavior-change swap that is
   uniform on Windows and needs no new dependency or platform branching.
2. **Clean break** — no compatibility shim, no old-registry migration.
3. **Schema id** → `osw:local-model-cache:1.0`.
4. **Schema-guard** with **family match + major enforcement** (see below).
5. **Bucket rename** folded into this spec (the `osw-public-models` bucket is already uploaded
   and hash-verified); Azure remains deferred.

## Component changes

### `scoutbot/registry/paths.py`

Both functions rename the env var and the default directory:

- `get_registry_path()`: `OSW_REGISTRY_PATH`, default `~/.osw/model-registry.json`.
- `get_models_dir()`: `OSW_MODELS_DIR`, default `~/.osw/models`.

### `scoutbot/registry/__init__.py`

- `SCHEMA_ID = 'osw:local-model-cache:1.0'`.
- Add the schema-guard to `_read_raw()` (see next section).

### `scoutbot/model_store.py`

- The four `CONFIGS` URLs (`v3` and `v3-cls`, `url` + `md5_url`) change host from
  `rebel-pipeline-public-models` to `osw-public-models`. Path prefix `/scoutbot/`, filenames,
  and pinned MD5s are unchanged. Azure fallback URLs are untouched.

### `scoutbot/scoutbot.py`

- Line 438 docstring: "Rebel Workflow model registry" → "Overhead Survey Workflow (OSW) model
  registry".

### Public metadata

- `pyproject.toml` `authors`: `{name = "Rebel Workflow Team"}` → `{name = "Overhead Survey
  Workflow Team"}` (the `{name = "Wild Me"}` co-author is unrelated and stays).
- `docs/conf.py`: `author = 'Rebel Workflow Team'` → `author = 'Overhead Survey Workflow Team'`
  (this also feeds the rendered docs copyright/footer).

(The exact ownership string is the maintainer's call; `Overhead Survey Workflow Team` is used
here as the direct analog of the old name.)

## Schema-guard and versioning contract

### Guard behavior (`_read_raw()`)

Constants in `scoutbot/registry/__init__.py`:

- `SCHEMA_ID = 'osw:local-model-cache:1.0'`
- `SCHEMA_FAMILY = 'osw:local-model-cache'`
- `SCHEMA_MAJOR = 1`

Exception: a dedicated `class RegistrySchemaError(RuntimeError)` (defined in
`scoutbot/registry/__init__.py`, exported so tests and callers can catch it). Every refusal
raises `RegistrySchemaError`; every message names the registry path, the offending value, the
expected `osw:local-model-cache:{SCHEMA_MAJOR}.*`, and the `OSW_REGISTRY_PATH` escape hatch.

A **missing** file returns a fresh `{'schema': SCHEMA_ID, 'models': {}}` as today (no guard). For
an **existing** file, `_read_raw()` evaluates the following in order and raises
`RegistrySchemaError` on the first failure:

| # | Condition | Outcome |
|---|---|---|
| 1 | File is not valid JSON (`json.JSONDecodeError`) or not a JSON object (`dict`) | refuse — corrupt/foreign |
| 2 | `schema` key missing, or its value is not a `str` | refuse — not an OSW registry |
| 3 | `schema` does not split into exactly three `:`-separated parts (`family.head:family.tail:version`) — e.g. bare `osw:local-model-cache`, or extra colons | refuse — malformed schema id |
| 4 | The family (first two parts joined, `osw:local-model-cache`) ≠ `SCHEMA_FAMILY` — includes any old `rw:*` id and any other family | refuse — wrong family |
| 5 | The version part is not `MAJOR.MINOR` with **both** integers — e.g. `1` (no minor), `1.x`, `1.2.3`, empty | refuse — malformed version |
| 6 | `MAJOR` ≠ `SCHEMA_MAJOR` (e.g. `2.0`) | refuse — incompatible major |
| — | Otherwise (family match, integer `MAJOR == SCHEMA_MAJOR`, any `MINOR`) | **accept**; unknown top-level/entry fields are ignored |

The guard only ever *reads and rejects* — it never rewrites or deletes an unrecognized file, so
coexistence holds: scoutbot touches only `model-registry.json` and `models/` within `~/.osw`,
never the directory as a whole.

### CLI surface

The registry-backed CLI commands (`scoutbot fetch`, `scoutbot models`, `scoutbot models info`,
`scoutbot register`) catch `RegistrySchemaError` at the command boundary and exit non-zero with
the exception's message (no Python traceback), so a foreign/incompatible registry produces a
clean, actionable CLI error rather than a stack dump.

### Versioning contract (documented in `docs/security.rst`)

The registry schema id is `osw:local-model-cache:MAJOR.MINOR`. The compatibility contract, which
all OSW ecosystem tools sharing the registry are expected to honor:

- **MINOR bump** (`1.0` → `1.1`): additive and backward-compatible — new *optional* fields only.
  Older readers ignore unknown fields; newer readers tolerate missing ones. No reader breaks, so
  tools at different minor versions interoperate on the same registry file.
- **MAJOR bump** (`1.x` → `2.0`): breaking — field removal/rename or changed semantics. A reader
  built for major `N` must **refuse** a file at major `≠ N` rather than risk misreading it.
  Adopting a new major must be coordinated across the tools that share the registry.

## Data flow / behavior

Unchanged except for names: `fetch()` resolves the registry via `get_registry_path()` (now
`~/.osw`/`OSW_REGISTRY_PATH`), downloads from `CONFIGS` sources (GCS now `osw-public-models`,
primary), verifies against the pinned MD5, and registers under the `osw:` schema id. On an
existing machine, the old `~/.rebel-workflow` registry is not read; scoutbot creates and
populates `~/.osw` on first fetch.

## Error handling & rollback

- Foreign/incompatible registry file → the schema-guard refuses with an actionable message; the
  user relocates via `OSW_REGISTRY_PATH`. scoutbot never clobbers the file.
- The new bucket is verified live before the `CONFIGS` swap, so the URL change is not gated on
  external timing (unlike Azure).
- **Rollback** = revert the commit: the old `rebel-pipeline-public-models` bucket and
  `~/.rebel-workflow` registry are left intact, so reverting fully restores prior behavior.

## Testing

- **`tests/test_registry.py`**:
  - Update the default-path assertions to `~/.osw/model-registry.json` and `~/.osw/models`, and
    the env-var names to `OSW_REGISTRY_PATH` / `OSW_MODELS_DIR` (the existing
    `test_get_models_dir_default` / `_from_env` tests keep `get_models_dir()` covered).
  - Schema-guard cases, each asserting `pytest.raises(RegistrySchemaError)` unless noted, mapping
    one-to-one to the guard table: non-JSON file; JSON that is not an object; missing `schema`;
    non-string `schema`; malformed id (bare `osw:local-model-cache`); old `rw:local-model-cache:1.0`
    (wrong family); malformed version (`...:1`, `...:1.x`); incompatible major (`...:2.0`); **plus
    a passing** same-major newer-minor file (`...:1.9` with an extra unknown field) that reads and
    ignores the unknown field; and a normal `1.0` round-trip.
- **`tests/test_loc.py`, `tests/test_model_store.py`, `tests/test_pipeline.py`,
  `tests/test_cli.py`**: swap all `RW_*` env-var references to `OSW_*` in isolation fixtures
  (~13 references, including the defensive `OSW_MODELS_DIR` set in `test_model_store.py`'s
  `isolated_registry` fixture).
- **CLI**: a test that a foreign/incompatible registry surfaces a clean non-zero CLI error (the
  `RegistrySchemaError` message, no traceback) from a registry-backed command.
- Full fast suite green; slow suite unaffected (names only).

## Validation

- The new bucket is already verified: both date-tagged `.pt` files at
  `osw-public-models/scoutbot` have content == sidecar == pinned MD5, and `INDEX.txt` is present.
- Post-change, run a clean-room fetch (isolated `OSW_REGISTRY_PATH` + `CACHE_DIR`, both configs)
  asserting the model resolves from `osw-public-models` (GCS primary) and the pinned hashes
  verify.
- `OSW_MODELS_DIR` / `get_models_dir()` is **API-only**: it belongs to the registry path API and
  is covered by `test_registry.py`, but the model-fetch path does not consume it (models cache
  under `~/.cache/scoutbot/`, and the registry stores absolute model paths). The clean-room fetch
  validation therefore exercises `OSW_REGISTRY_PATH` only; `OSW_MODELS_DIR` is validated by unit
  tests, not the fetch flow.
- Run the full gate: `ruff format --check` / `ruff check` / `mypy scoutbot tools/visualise.py` /
  `uv audit --locked` / `bandit` / `pytest -m "not slow"` (+ slow where feasible).

## Documentation

- `docs/security.rst` — GCS bucket name; the **registry schema versioning contract** (new
  section); the schema-guard behavior.
- `docs/configuration.rst` — env-var names (`OSW_*`), registry directory (`~/.osw`), bucket URL.
- `docs/models.rst` — bucket URLs.
- `docs/roadmap.rst` — the registry-removal command (`rm ~/.osw/model-registry.json`).
- `CHANGELOG.rst` — **reconcile the existing `Unreleased` bullets so the section is
  self-consistent, not just append.** These features ship for the first time in this release, so
  the notes should read in final form: update the multi-source-fetch bullet that names
  `storage.googleapis.com/rebel-pipeline-public-models` and the registry bullet that names
  `~/.rebel-workflow/model-registry.json` (and any `RW_*` env vars) to the new `osw` names. Then
  add a dedicated bullet documenting the `rw` → `osw` rename itself (env vars, registry dir,
  schema id, GCS bucket, clean break, schema-guard). After this, the **Audit scope** grep must
  return zero for `CHANGELOG.rst`. (The unrelated historical `[tool.uv.audit]` wording is
  reconciled separately in the release-notes consolidation, not here.)
