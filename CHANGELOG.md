# Changelog

## Source selection in the chat — 2026-10-23

The main question composer now offers source checkboxes, Select all / Select none, and selected unit/passage counts. Budgets and queries use the same server-validated selection across full units, summaries, passages and neighbors. Empty, unavailable and duplicate selections are rejected before provider initialization. Manual and URL selections survive polling; saved inquiries restore their recorded source scope without overwriting an explicit URL choice on initial load.

The reference store supplies read-only source views. Collection adapters can expose named groups, disable uploads and retain inquiries from trusted historical store roots. `/open?collection=<id>` redirects within the same origin, avoiding form-policy failures from cross-port navigation. Historical runs remain readable.

Verification: **92 public tests** pass, including ten backend selection and ten frontend interaction regressions. Actual browser checks reproduce the prior form failure and verify its repaired route, checkbox combinations, empty-selection guard, selected budgets, saved inquiries and citation navigation. Desktop and 390px layouts were inspected without horizontal overflow. A bounded combined-source live query returned only selected evidence with exact text, metadata and offsets. Private fixtures and runtime data remain excluded from the public repository. This is functional validation, not a retrieval-quality benchmark.

## Validation follow-up — 2026-10-23

Valid JSONL source strings can contain literal Unicode next-line, line-separator and paragraph-separator characters. The importer now splits records only at ASCII newlines, preserving those characters inside source text, summaries and metadata. Synthetic coverage includes LF/CRLF records and a UTF-8 BOM.

All **72 automated tests pass** on Windows/Python 3.12. A fresh wheel builds and installs into an isolated directory with the UI and both embedding presets present. A fresh authored-fixture run passes actual local MiniLM embeddings, Jev scoring and Codex answering. This is integration verification, not an answer-quality benchmark. No new visual browser pass or remote CI result is claimed here.

Additional failures found during integration are corrected:

- ONNX tokenizers disable serialized truncation before the input-length check. A new ONNX tokenization fingerprint prevents reuse of potentially clipped older embeddings; rebuild those indexes in a separate directory. Other provider fingerprints remain compatible.
- The optional Codex bridge passes only runtime/authentication environment variables, excluding unrelated provider credentials and startup/proxy overrides. Shell execution and hooks are disabled. Hosted inference, Codex session retention and inherited tool/configuration limitations are documented.
- Known citation groups and ranges are converted to individually linked IDs while retaining page notes and the raw answer. Unknown, reversed, cross-family or malformed references still fail validation. Rejected candidates are retained for local diagnosis and never served as validated answers.

The earlier audit record below is historical; the verification above applies to the current source.

## Unreleased — correctness and reliability audit, 2026-10-22

The initial release's 30-test suite left several boundary failures uncovered. This follow-up fixes the local working source; no new public release or remote CI result is claimed.

### Corrected behavior

- Passage-only queries skip summary ranking and its call budget. Only the active final-passage cap constrains the Jev shortlist; unconfigured answer/rerank requests fail before provider work.
- Identical text from different units keeps its distinct attribution and neighboring context. Full-unit exclusion cannot erase another unit's copy before ranking. Exact repeats within one unit remain deduplicated.
- Stable unit IDs and exact metadata reach the readable context and final model prompt, including authorship and extraction/summary caveats. Metadata counts toward the context budget and must be finite JSON.
- Unchanged imports with generated summaries reuse their persisted summary/provenance without another model call. All import identities are validated before summary generation, and changed sources remain immutable.
- Every summary vector is normalized and dimension-checked before pooling, with bounded embedding batches. Extreme finite values no longer collapse to zero vectors. Advertised embedding model substitutions and explicit-dimension/default mismatches fail; legacy indexes remain readable without rewriting them.
- DOCX extraction retains visible tabs and line breaks and omits deleted revisions and field instructions.
- Provider redirects are rejected before forwarding credentials or source data. Public-source fetches connect to validated public IPs with TLS hostname verification and repeat checks at redirects, closing the DNS re-resolution gap.
- Upload summary switches require JSON booleans. The interface waits for initial server state before enabling operations, avoiding the startup polling race. HTTP test failures now expose useful startup/loopback diagnostics and clean up their own process.

### Verification and limits

All **60 automated tests passed** on Windows/Python 3.12: 36 core tests, 12 pipeline regressions, five mocked source-transport tests and seven web/UI tests. This includes actual loopback uploads, saved-source checks, the 500 → 100 → 20 mechanics and a Node execution of the real frontend script with delayed startup state. After the final explicit HTTP-response cleanup, the five transport tests passed again. Frontend/bridge JavaScript syntax and whitespace checks pass. Synthetic fixtures only; no real corpus changes or paid provider calls.

Browser visual inspection, new live provider calls, package rebuild and remote cross-platform CI were not repeated in this audit. The existing local code remains dependency-compatible, but no rebuilt distribution or deployment is claimed. Source identity's legacy hexadecimal-file-hash convention is documented in `docs/UNITS.md` and preserved to avoid invalidating existing immutable units. Mutable provider aliases, semantic answer accuracy and prompt-injection resistance remain limitations documented in the architecture/security guides.

### Preservation

Recoverable pre-audit code and guidance: project commit `e47a7ad`. No corpus/index migration, parent-repository staging, deletion of user files or public push occurred. The containing workspace still ignores the whole project. Local audit evidence preserves the exact OLD/NEW diff and file hashes. To inspect the prior state without replacing current work, use `git show e47a7ad:<path>`; to reverse this audit later, first preserve intervening edits and apply the audit diff in reverse after a successful `git apply --reverse --check`. Newly added regression modules have OLD `[not present]` in that diff.

## Repository boundary — 2026-10-22

The project working folder now has its own independent Git root, index, objects and remote. The containing workspace excludes this complete project tree. No submodule, shared worktree, linked object store or parent index mutation was introduced. The initial publishing checkout remains preserved as historical evidence; ongoing releases use this project root with an explicit boundary guard. Its initial release history was preserved when setting up the working repository. No application behavior changed.

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
