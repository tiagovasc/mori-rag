# Basics

Mori is a standalone, open-source retrieval and answer system. It combines semantic passage search and Jev relevance ranking with complete source units and ranked summaries. The public entry point is [README.md](README.md); implementation and release status belong in [CHANGELOG.md](CHANGELOG.md).

# Table of Contents

Basics; Table of Contents; Maintenance; What Belongs Here; How This Folder Works; How to Add or Edit; Changelog; Reminders; MISC.

# Maintenance

Keep this guide, public README, configuration reference and tests current when behavior changes. Preserve original source units and historical query packets. Never publish generated corpus data, credentials or private verification records.

# What Belongs Here

Reusable application source, documentation, provider adapters, authored example units and tests. Existing research corpora stay with their original projects. Local release evidence and operational notes are ignored under `.local/`.

# How This Folder Works

`mori/` owns configuration, ingestion, storage, providers, retrieval and local UI. `docs/` explains contracts and extension points. `examples/` contains explicitly distributable synthetic sources and configurations. `tests/` verifies actual pipeline boundaries without paid calls.

# How to Add or Edit

Follow [AGENTS.md](AGENTS.md). Update the relevant public guide and behavioral tests together. Local corpus and run directories default to `.mori/`; do not commit them. Public release and limitations are recorded in the changelog.

# Changelog

22 October 2026: Created as a separate project combining reusable ideas from prior retrieval, summary-ranking and full-source systems. Existing projects remain unchanged.

# Reminders

The relevance scale is a ranking signal, not a probability of correctness. Full sources must fit explicit context budgets; never silently truncate. Local UI is loopback-only and is not a hosted multi-user service.

# MISC

Provider-specific authentication belongs in the user's environment. Public-safe examples do not depend on any private corpus or workspace path.
