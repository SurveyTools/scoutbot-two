# GitHub Pages Docs Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the rendered ScoutBot Sphinx docs to GitHub Pages on merges to `main` and route the README to the live site instead of raw `.rst` sources.

**Architecture:** Keep the Sphinx source of truth in `docs/*.rst`, but build and deploy HTML from GitHub Actions so readers land on rendered pages. Use a dedicated Pages workflow on `main`, keep the existing test and quality workflows separate, and update the README to link to the published site root and key entry pages. If Read the Docs stays configured, treat it as secondary; the branch’s primary docs surface is the GitHub Pages site.

**Tech Stack:** GitHub Actions, GitHub Pages, Sphinx, `uv`, `sphinx-click`.

---

## Task 1: Add a GitHub Pages deployment workflow

**Files:**
- Create: `.github/workflows/docs-pages.yml`

**Goal:** Build the Sphinx HTML site on pushes to `main` and deploy it to GitHub Pages.

- [x] **Step 1: Write the workflow**

Add a workflow that builds docs from `docs/` and publishes the rendered HTML:

```yaml
name: Docs Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v7
        with:
          lfs: true
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v8.2.0

      - name: Set up Python 3.12
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Install docs dependencies
        run: uv sync --extra docs

      - name: Build Sphinx HTML
        run: uv run sphinx-build -b html docs docs/_build/html

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v4
        with:
          path: docs/_build/html

  deploy:
    needs: build
    runs-on: ubuntu-22.04
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v5
```

- [x] **Step 2: Verify the workflow syntax matches the repo’s current Actions style**

Confirm the workflow uses the same modern Actions family as the existing CI:

```bash
sed -n '1,120p' .github/workflows/testing.yml
sed -n '1,120p' .github/workflows/quality.yml
```

Expected: the Pages workflow follows the same `actions/checkout@v7`, `actions/setup-python@v6`, and `astral-sh/setup-uv@v8.2.0` pattern.

- [x] **Step 3: Confirm the workflow only runs on main and manual dispatch**

Run a grep check:

```bash
rg -n "workflow_dispatch|push:|branches: \[main\]" .github/workflows/docs-pages.yml
```

Expected: the site deploys only on `main` merges and manual triggers.

---

## Task 2: Route the README to rendered docs

**Files:**
- Modify: `README.md`

**Goal:** Stop sending readers to raw `.rst` files from the README and point them at the rendered site instead.

- [x] **Step 1: Replace source-doc links with the published site root**

Update the README’s docs-routing table so the main reader paths point to the live site rather than repository source files. Use the rendered docs root plus page names, for example:

```md
| I want to... | Start here |
|---|---|
| Run ScoutBot directly on images | [ScoutBot docs](https://surveytools.github.io/scoutbot-2026/), [Installation](https://surveytools.github.io/scoutbot-2026/installation.html), [Usage](https://surveytools.github.io/scoutbot-2026/usage.html), [Configuration](https://surveytools.github.io/scoutbot-2026/configuration.html) |
| Integrate ScoutBot into a larger pipeline | [Usage](https://surveytools.github.io/scoutbot-2026/usage.html), [API reference](https://surveytools.github.io/scoutbot-2026/api.html), [CLI reference](https://surveytools.github.io/scoutbot-2026/cli.html) |
| Understand outputs and analysis limits | [Usage](https://surveytools.github.io/scoutbot-2026/usage.html), [Configuration](https://surveytools.github.io/scoutbot-2026/configuration.html), [Performance](https://surveytools.github.io/scoutbot-2026/performance.html) |
| Contribute code or prepare releases | [Development](https://surveytools.github.io/scoutbot-2026/development.html), [Roadmap](https://surveytools.github.io/scoutbot-2026/roadmap.html), [Changelog](https://surveytools.github.io/scoutbot-2026/changelog.html) |
| Add models or investigate performance | [Models](https://surveytools.github.io/scoutbot-2026/models.html), [Core ML](https://surveytools.github.io/scoutbot-2026/coreml.html), [Performance](https://surveytools.github.io/scoutbot-2026/performance.html), [Roadmap](https://surveytools.github.io/scoutbot-2026/roadmap.html) |
```

Use the repository’s published Pages URL as the base for all README links.

- [x] **Step 2: Point the top-level docs section at the rendered site**

Update the README’s documentation note so it tells readers to use the live site, not the source tree:

```md
## Documentation

Read the rendered ScoutBot docs at https://surveytools.github.io/scoutbot-2026/.
```

Keep the local build command for maintainers only.

- [x] **Step 3: Verify no README links still point to raw `.rst` docs**

Run:

```bash
rg -n "docs/[a-z0-9_-]+\\.rst" README.md
```

Expected: no user-facing README routing links point to Sphinx source files.

---

## Task 3: Align docs navigation with the published site

**Files:**
- Modify: `docs/index.rst`
- Modify: `docs/roadmap.rst` if needed
- Modify: `docs/development.rst` if needed

**Goal:** Make the rendered site the documented home for users and keep maintainer notes accurate.

- [x] **Step 1: Add a short site-oriented note to the docs landing page**

If the docs landing page should mention where to find the hosted site, add a small note near the top of `docs/index.rst`:

```rst
ScoutBot docs are published on GitHub Pages; this repository contains the Sphinx source.
```

Keep the toctree unchanged unless the site layout itself needs changes.

- [x] **Step 2: Update roadmap or development notes if they still talk about local-only docs**

Search for any language that implies the docs are only meant to be built locally:

```bash
rg -n "build the Sphinx docs locally|Open `docs/build/html/index.html`|RTD|Read the Docs" docs README.md
```

If any user-facing text still suggests local-only docs, rewrite it to reference the hosted site and keep the local build command as a maintainer path.

- [x] **Step 3: Verify the site still builds cleanly**

Run:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Expected: clean Sphinx build, including `sphinx-click` pages like `cli.rst`.

---

## Task 4: Decide whether Read the Docs stays enabled

**Files:**
- Modify: `.readthedocs.yaml` or remove it if it is no longer needed
- Modify: `docs/roadmap.rst` if the hosting decision should be recorded

**Goal:** Avoid running two doc hosting systems unless there is a clear reason to keep both.

- [x] **Step 1: Choose the hosting policy**

If GitHub Pages becomes the canonical docs site, decide whether Read the Docs remains:

```md
Option A: keep RTD as a secondary mirror.
Option B: deprecate RTD and leave GitHub Pages as the only hosted docs site.
```

Document the choice in the plan execution notes before changing files.

- [x] **Step 2: If RTD is kept, confirm it still builds the same source**

Make sure `.readthedocs.yaml` continues to install the `docs` extra and point at `docs/conf.py`:

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.10"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

- [x] **Step 3: If RTD is removed, delete the config file and stop mentioning it in docs**

If the project chooses GitHub Pages only, remove `.readthedocs.yaml` and update any docs that still mention RTD.

---

## Task 5: Validate the published-docs path

**Files:**
- Read/build docs output
- Read: `.github/workflows/docs-pages.yml`, `README.md`, `docs/index.rst`

**Goal:** Prove the site builds and the routing text matches the new hosting model.

- [ ] **Step 1: Build the docs locally**

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

- [ ] **Step 2: Check for stale source-doc links in the README**

```bash
rg -n "docs/[a-z0-9_-]+\\.rst" README.md
```

Expected: no public routing links point to raw `.rst` files.

- [ ] **Step 3: Check the workflow paths**

```bash
rg -n "upload-pages-artifact|deploy-pages|docs/_build/html" .github/workflows/docs-pages.yml
```

Expected: the workflow uploads and deploys the Sphinx HTML output directory.

- [ ] **Step 4: Confirm the rendered site will include the raw Sphinx directives**

Because the site is HTML, `sphinx-click` and other directives should render as documentation, not source text. Verify by opening the generated output locally:

```bash
python - <<'PY'
from pathlib import Path
index = Path('docs/_build/html/cli.html')
print(index.exists())
PY
```

Expected: `True`.

---

## Documentation Pass

- [ ] Update `CLAUDE.md` and `docs/development.rst` if the new site hosting model changes the docs ownership or contribution workflow.
- [ ] Update `docs/roadmap.rst` to record whether GitHub Pages is now the canonical docs site and whether RTD is retired or retained.
- [ ] Update `CHANGELOG.rst` if the hosting change is user-visible or part of the release notes.
