# Prometheus-Survey-Hub `scoutbot-2026` → `scoutbot-two` Migration Plan

> Cross-repo text sweep in **prometheus-survey-hub** accompanying the scoutbot
> repo rename (`scoutbot-2026` → `scoutbot-two`). Authored in scoutbot-two per
> the repo's `docs/plans/` convention; the edits land in the Hub repo.
> **No git steps** — stop at review state, hand back a suggested commit message.

**Goal:** Update all live/user-facing/runtime references to the old repo name
in the Hub, keeping runtime behavior intact, while applying four decisions made
during planning.

**Scope:** 14 grep hits across runtime code, config, tests, and docs (~17
edits). One behavioral change (the provenance label string). Independent of
the scoutbot in-repo sweep — can run any time after the GitHub rename.

## Decisions (locked during planning, 2026-08-14)

| # | Decision | Choice |
|---|----------|--------|
| Q1 | Stored provenance label `scoutbot-2026/{config}` | → `scoutbot/{config}` — matches the Python **package** name (unchanged by the rename); repo suffix stays out of provenance |
| Q2 | `DEFAULT_VENV` convention path | → `~/repos/scoutbot-two/.venv` (straight rename of the convention) |
| Q3 | Vestigial `config/default.yaml` `scoutbot_venv` key | Update the string in place (do not prune the orphaned consumer in this sweep) |
| Q4 | Roadmap/planning docs naming the workstream | Update to the new name |

## Context notes (verified during planning)

- The active detection pipeline (`api/services/*` → `scoutbot_detector.resolve_scoutbot()`)
  resolves the venv from `$SCOUTBOT_VENV` else `DEFAULT_VENV`. It does **not**
  read `config/default.yaml`'s `scoutbot_venv`.
- `config/default.yaml:74 scoutbot_venv` is read **only** by
  `survey_backbone/hh/detection/detector.py`, which is **not imported anywhere**
  (orphaned; its own hardcoded default is the older `~/wildme/scoutbot-venv`).
  Hence Q3 = update-in-place, and a follow-up cleanup is logged below.
- On the current machine neither `~/repos/scoutbot-2026` nor `~/repos/scoutbot-two`
  exists and `$SCOUTBOT_VENV` is unset — the default path is a documented
  fallback, so nothing here breaks at runtime as long as the real environment
  sets `$SCOUTBOT_VENV`.

---

## Task 1: Apply the Hub sweep

All edits are in `prometheus-survey-hub` (branch `scoutbot-two-updates`).

### Group A — Provenance label (runtime + tests, change in lockstep)

Behavioral change: stored job/import provenance goes from `scoutbot-2026/v3`
→ `scoutbot/v3`. The tests below already pin the string, so update code and
tests together (test-update-in-lockstep, not red→green).

- `api/services/import_service.py:289` — `f"scoutbot-2026/{configs.pop()}"` → `f"scoutbot/{configs.pop()}"`
- `api/services/job_service.py:308` — `f"scoutbot-2026/{sb_config}"` → `f"scoutbot/{sb_config}"`
- `tests/test_import_jobs.py:83` — assert `"scoutbot-2026/v3"` → `"scoutbot/v3"`
- `tests/test_detect_job.py:160` — assert `"scoutbot-2026/v3"` → `"scoutbot/v3"`

### Group B — Venv path (`~/repos/scoutbot-2026/.venv` → `~/repos/scoutbot-two/.venv`)

- `survey_backbone/scoutbot_detector.py:34` — `DEFAULT_VENV`
- `survey_backbone/scoutbot_detector.py:10` — docstring resolution note
- `survey_backbone/scoutbot_detector.py:56` — FileNotFoundError message tail (`…at ~/repos/scoutbot-two/.venv.`)
- `README.md:50` — resolution note
- `config/default.yaml:74` — `scoutbot_venv:` value (vestigial; update in place per Q3)

### Group C — Repo-name tags in prose / errors (`scoutbot-2026` → `scoutbot-two`)

- `survey_backbone/scoutbot_detector.py:2` — module docstring tag `(scoutbot-2026)`
- `survey_backbone/scoutbot_detector.py:54` — "Install scoutbot-2026" → "Install scoutbot-two"
- `config/default.yaml:71` — comment tag `(scoutbot-2026)`
- `api/services/detection_service.py:358` — docstring tag `(scoutbot-2026)`
- `README.md:47` — project name **and** URL → `https://github.com/SurveyTools/scoutbot-two`

### Group D — Roadmap / planning docs (Q4: update)

- `docs/ROADMAP.md:81` — "scoutbot-2026 migration (P1)" → "scoutbot-two migration (P1)"
- `docs/PIPELINE_ALIGNMENT.md:73` — same workstream name
- `docs/PIPELINE_ALIGNMENT.md:76` — `~/repos/scoutbot-2026` API reference

### Verify

Run the two affected test files in the **Hub venv** (not scoutbot's):

```bash
pytest tests/test_import_jobs.py tests/test_detect_job.py -q
```

Then confirm zero remaining references:

```bash
grep -rIn "scoutbot-2026" --exclude-dir=.git --exclude-dir=.venv .
```

Expected: both test files pass; grep returns nothing.

---

## Out of scope (record, don't do here)

- **Prune orphaned `hh/detection` scoutbot path.** `config/default.yaml`'s
  `scoutbot_venv` key and `survey_backbone/hh/detection/detector.py` are dead
  (unimported). Removing them is a separate cleanup, tracked so the stale
  wiring isn't forgotten.
- **Provenance-format changelog note.** If the Hub adopts a changelog for
  output-schema changes, the `scoutbot-2026/v3` → `scoutbot/v3` label change
  warrants a one-line entry. Hub is pre-release, so no back-compat handling for
  existing stored records is required.

## Handoff

No `git add` / `git commit`. Stop at review state and provide a suggested
commit message, e.g.:

```
Rename scoutbot-2026 references to scoutbot-two

Sweep Hub references to the renamed scoutbot repo. Provenance label
changes to package-name form (scoutbot/{config}); venv default and docs
track the new repo name. Orphaned hh/detection config key updated in
place (pruning tracked separately).
```
