# Changelog

## 0.1.0 — 2026-10-22

Initial source-aware RAG release: immutable portable source units and summaries, exact chunk offsets, local/ONNX/HTTP embeddings, independent Jev passage/summary ranking, a normalized 0–100 qualitative rubric, optional disabled LLM reranking, configurable full-source/summary/passage packets, context budgets, local upload UI, portable answer handoff and an optional Codex SDK bridge.

Verification is in progress. Offline tests have exercised the actual 500 → 100 → 20 mechanics. Final live smoke, interface checks, package/build checks and publication audit are recorded below when complete. No broad retrieval-quality or academic-accuracy benchmark is claimed.

### Verification closeout

The implementation-stage note above is superseded by these completed checks. All 27 initial regression tests passed, including the real local HTTP server, upload ingestion, request-origin/token checks, static-file boundaries and the 500 → 100 → 20 fixture. Python wheel build and frontend/bridge JavaScript syntax checks passed. Later embedding-preset checks are recorded in the final release receipt.

A bounded live query used real MiniLM ONNX embeddings (384 dimensions), real Jev with ten qualitative levels, and the real Codex SDK with `gpt-6.1-sol` / low. Seven fictional units yielded 17 retrieved/rated passages and seven rated summaries. Final evidence contained one complete original, five other summaries and fourteen passages; the requested cap of twenty was explicitly reported as a shortfall after full-unit exclusion. The 2,836-character answer passed citation-ID/paragraph checks and a manual source spot-check. The heavier reranker remained disabled. This is a functional smoke test, not a comparative retrieval or academic-quality benchmark.

The actual interface was inspected at desktop and 390px phone width, with no horizontal overflow (375px body width). Budget review, saved-answer opening and a citation opening its exact original passage worked. A real browser upload completed as seven unchanged units, without re-embedding existing sources. Provider secrets and private verification records are excluded from public source.

MiniLM and OpenAI have explicit initializer/configuration presets. OpenAI uses `text-embedding-3-large` with configurable output dimensions and a separately supplied API key. Its contract is tested with mocks; no live OpenAI call, API-key reuse or corpus-specific speed/quality comparison was performed. Changing models requires a separate index and explicit re-embedding.

The wheel is an installation artifact; source is published on GitHub, not claimed released to PyPI. Hosted multi-user access, production ANN connectors, audio transcription and automated media downloading are outside this local engine; adapter boundaries are documented explicitly.

### Implementation notes

- Final embedding-preset regression expansion: all 30 tests passed. The OpenAI fixture verifies model, endpoint, float output format, requested dimensions, response order/normalization, missing-key failure and shape rejection. The initializer preserves existing configurations. MiniLM remains the local speed-first option; no live OpenAI API call was made.

- SQLite context managers commit/roll back but do not close a connection. Explicit closing is required to avoid Windows file-handle leaks; the regression suite exercises actual database cleanup.
- The optional Node adapter's dependency installation is run from its own directory. The initial relative-prefix installation did not resolve the package directory in this host; no failed dependency result is treated as a completed install.
