# Basics

Mori is a standalone, open-source retrieval and answer system. It combines semantic passage search and Jev relevance ranking with complete source units and ranked summaries. The public entry point is [README.md](README.md); implementation and release status belong in [CHANGELOG.md](CHANGELOG.md).

# Table of Contents

Basics; Table of Contents; Maintenance; What Belongs Here; How This Folder Works; How to Add or Edit; Changelog; Reminders; MISC.

# Maintenance

**Independent Git boundary — 22 October 2026:** Mori's working folder is its own repository, using only `https://github.com/tiagovasc/mori-rag.git`. The parent Claude repository excludes the entire `Projects/Mori/` tree. Their indexes, branches, histories and remotes remain independent; this is not a submodule or shared worktree. Verify the repository root before staging. Workspace housekeeping must not stage Mori's code or push its remote. `.local/public-source/` is a preserved initial-publication snapshot, not the ongoing editing/publishing authority.

Keep this guide, public README, configuration reference and tests current when behavior changes. Preserve original source units and historical query packets. Never publish generated corpus data, credentials or private verification records.

# What Belongs Here

Reusable application source, documentation, provider adapters, authored example units and tests. Existing research corpora stay with their original projects. Local release evidence and operational notes are ignored under `.local/`.

# How This Folder Works

`mori/` owns configuration, ingestion, storage, providers, retrieval and local UI. `docs/` explains contracts and extension points. `examples/` contains explicitly distributable synthetic sources and configurations. `tests/` verifies actual pipeline boundaries without paid calls.

# How to Add or Edit

Follow [AGENTS.md](AGENTS.md). Update the relevant public guide and behavioral tests together. Local corpus and run directories default to `.mori/`; do not commit them. Public release and limitations are recorded in the changelog.

# Changelog

23 October 2026 menu refinement: source checkboxes are inside the collapsed **Sources** disclosure immediately above **Evidence settings**, with selection counts visible when closed. The public README documents usage; `CHANGELOG.md` records verification.

23 October 2026 source-selection follow-up: the main composer has source checkboxes with all/none controls, selected-source budgets, server-enforced filtering and saved-selection restoration. Generic adapters can group compatible collections and keep their prior inquiries visible. Empty/unavailable sources fail closed. The same-origin selection route avoids cross-port form redirects. Current verification and limits are recorded in `CHANGELOG.md`; private operational setup and inventories remain ignored.

23 October 2026 integration closeout: 72 automated tests pass, including ONNX truncation/fingerprint safeguards, Codex environment isolation, grouped citation handling and rejected-answer preservation. The updated package and authored live-provider fixture are verified. Earlier checkpoints below are historical; `CHANGELOG.md` owns compatibility and verification details.

23 October 2026 validation: JSONL ingestion now preserves Unicode separators within source strings. All 61 tests pass, a fresh wheel builds and installs with packaged assets, and the authored fixture passes real local embeddings, Jev scoring and Codex answering. Detailed current scope and publication status remain in `CHANGELOG.md`.

22 October 2026 follow-up audit: fixes and regression coverage for optional-stage budgets, source attribution and metadata, embedding/summary integrity, DOCX extraction, provider redirects, public-source DNS pinning, strict upload switches and initial UI polling. Detailed behavior and verification are recorded in `CHANGELOG.md`; prior release verification below is historical. Regression modules under the existing `tests/` group include `test_pipeline_regressions.py` and `test_fetch_security.py`. Private continuation and reversible diff records remain under `.local/`.

22 October 2026 completion: public source is https://github.com/tiagovasc/mori-rag. Thirty tests and cross-platform CI pass; real MiniLM/Jev/Codex smoke verification and desktop/mobile/upload/source-link checks passed. MiniLM/OpenAI embedding selection is implemented; OpenAI contract is mock-tested. Public-safe fixtures only; private release receipts remain under `.local/`. Public README and CHANGELOG own the supported behavior and limitations.

22 October 2026: Created as a separate project combining reusable ideas from prior retrieval, summary-ranking and full-source systems. Existing projects remain unchanged.

# Reminders

The relevance scale is a ranking signal, not a probability of correctness. Full sources must fit explicit context budgets; never silently truncate. Local UI is loopback-only and is not a hosted multi-user service.

# MISC

Provider-specific authentication belongs in the user's environment. Public-safe examples do not depend on any private corpus or workspace path.

Audit lessons: deduplicate retrieval text within its unit so identical words do not erase distinct source context; retain metadata in the actual model prompt, not only in the JSON packet. Validate all summary vectors and import identities before committing or making avoidable provider calls. A disabled stage must also disappear from preflight cost. Run test discovery from this repository root: a parent directory can resolve an unrelated installed `tests` package. Windows restricted loopback access can fail even while a test server is running; preserve the socket diagnostic and use the approved scoped test execution rather than changing the server's local-only bind.

Navigation lesson: a successful HTTP redirect does not prove a form works in a browser. `form-action 'self'` can block a redirect to another port. Keep selection in the same origin and verify the actual click. Source filtering must happen before every retrieval branch, not only when rendering results; keep manual/URL selections stable across polling and restore saved selections explicitly.
