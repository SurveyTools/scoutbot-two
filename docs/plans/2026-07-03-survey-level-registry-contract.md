# Survey-Level Registry Contract Draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` if this draft is turned into an implementation
> plan later.

**Goal:** Sketch the future survey-level registry contract that will eventually live in
`pipeline-design`, while keeping ScoutBot scoped to the current offline-local registry work.

**Non-goal:** Do not move the survey-level contract into ScoutBot implementation code. This draft is
only for the shared design boundary, terminology, and roadmap placement.

**Ownership split:**

* ScoutBot owns the local registry cache, offline fetch behavior, and CLI conveniences for choosing
  a model.
* `pipeline-design` owns the canonical survey-level `model_registry.json` schema, the model-family
  contract, lineage semantics, compatibility rules, and the Stage 3 boundary with run-level
  provenance.

**Draft contract concepts to carry forward:**

* `model_family_id` as the stable identifier for a biological/model lineage
* `model_release_id` or `model_variant_id` for a concrete exported artifact
* parent/child links between a base model and fine-tuned or re-exported descendants
* versioned aliases so old names can resolve to the same family or release
* compatibility metadata for supported stages, geometry expectations, and backend constraints
* survey-level provenance for ownership, source, and artifact identity
* run-level provenance for threshold, backend choice, and per-run inference settings

**Roadmap intent:** This is a version-3-or-later contract. The current ScoutBot registry plan should
remain the offline-first implementation step, and the richer registry schema should be defined
before any future registry rewrite is attempted.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `docs/roadmap.rst` | Add the survey-level registry contract note and the pipeline-design ownership split |

---

## Task 1: Draft the survey-level contract boundary

**Files:**
- Modify: `docs/roadmap.rst`

- [ ] Add a roadmap subsection that explains the v3+ survey-level registry contract and why it
  belongs in `pipeline-design`.
- [ ] Keep the existing ScoutBot registry work framed as the local/offline implementation step, not
  the final registry schema.

## Task 2: Make the contract shape explicit

**Files:**
- Modify: `docs/roadmap.rst`

- [ ] Enumerate the contract concepts that need to survive into the canonical survey-level schema:
  family IDs, lineage, aliases, compatibility metadata, and the survey-level/run-level split.
- [ ] Keep run-level provenance separate from the survey-level registry record so the boundary stays
  clear for Stage 3 and downstream consumers.

## Task 3: Final review

**Files:**
- Modify: `docs/roadmap.rst`

- [ ] Re-read the registry plan and the Stage 3 / schema-family docs to confirm the split remains
  consistent.
- [ ] Documentation pass completed: reviewed README/Sphinx docs/changelog/roadmap/security docs
      relevant to this change, updated them or recorded why no doc update was needed.
