# Architecture

1. Validate explicit original units and complete summaries. Reject missing fields, conflicting IDs and changed model fingerprints.
2. Split originals into deterministic overlapping windows with exact start/end offsets. Embed original windows and all summary segments with matching document semantics. Commit the complete import transaction only after every embedding succeeds.
3. Preflight the maximum rating decision count and any required answer/rerank provider. Query the embedding model with its configured query prefix; rank exact cosine similarity. Deduplicate whitespace-normalized exact text within each unit, continuing down the ranking to fill the candidate cap. Identical words from different units retain their distinct identity and context.
4. Jev rates each passage against the original question. Optional same-unit neighbors affect rating context only when enabled. Independently rate complete summaries, optionally after semantic summary preselection. When both `full_units` and `summaries` are zero, skip summary selection, rating and its call budget entirely.
5. Select full units from the summary ranking and exclude their IDs from other summary selections. Optionally exclude their passages before filling the Jev shortlist; this backfills from the ranked candidate pool rather than returning duplicate evidence.
6. Either take the configured final passage cap directly, or ask the optional LLM to order every shortlist ID exactly once and retain its top `rerank_k`. Missing, duplicated or invented IDs fail explicitly.
7. Assemble full originals, separately labelled summaries and exact original passage windows. Carry stable unit IDs and source metadata into both readable context and the model handoff, including authorship, URLs and extraction caveats. Add answer-stage neighbors independently from rating neighbors. Check the full rendered character budget, including metadata, before publishing context.
8. Save the portable context and exact answer prompt. Optionally call the configured answer provider. Validate supplied citation IDs and require citations on substantive paragraphs before publishing an answer.

## Failure and preservation

Corpus imports use a transaction; failed imports publish no partial new units. Queries preserve requests and retrieved inputs even when later stages fail. Jev caches every successful decision as it finishes, making explicit retry inexpensive. There is no silent cosine fallback when Jev fails and no automatic restart of interrupted model generation.

Every source ID is immutable. The version hash binds title, source identity, original, summary and metadata. Identical reimports avoid embedding calls. Model fingerprints include provider, requested model/revision, endpoint when relevant, task prefixes, chunk policy, normalization and vector dimensions. Use pinned model revisions for reproducibility; a mutable server-side alias can still change despite a stable requested name. The current reference implementation does not hash every model weight on every startup.

When `--summarize` reimports an unchanged unit, its persisted summary and generated provenance are reused without another model call. Source-field changes still require a new version. Every summary-segment vector is validated and normalized before pooling; segments use the configured embedding batch size. Legacy fingerprints that omit a requested dimension are treated as the equivalent default of zero; switching between an explicit dimension request and the provider default is rejected.

## Reference implementation and scale

Python owns orchestration; providers are small independent adapters. SQLite stores units, exact chunk text, normalized vectors and decision cache. Exact cosine scoring is dependency-free but scans the collection; it is not an ANN index. Threaded relevance calls have configurable bounded concurrency. The local server admits one import/query job at a time; CLI callers should avoid concurrent writes to one data directory.

Provider calls have bounded input bytes, timeouts and retries. The preflight distinguishes decisions from possible HTTP attempts: retries can multiply actual calls and billing. API model aliases and remote behavior are outside local reproducibility guarantees. Cached requests use the requested model; use explicit versions or a fresh corpus/cache directory to force a new evaluation after a provider change.

Use the database/export and provider contracts in [ADAPTERS.md](ADAPTERS.md). Do not imply that changing a database URL alone changes the reference SQLite implementation into a scalable multi-tenant service.
