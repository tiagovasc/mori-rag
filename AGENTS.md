# Building a corpus with Mori

Mori's public contract is `mori.context.v1`. Read `docs/UNITS.md` and `docs/ADAPTERS.md` before adding sources. Never use this repository's example sources as factual research evidence: they are invented teaching fixtures.

1. Identify the user's source collection, retrieval question, privacy requirements and allowed provider destinations. Do not upload private sources just because a provider is configured.
2. Inventory originals. Define `source_id` for the work and stable `id` for each independently readable unit. A book normally becomes chapters; a video or podcast normally becomes one lecture/episode, or explicit timestamped segments if too large. Keep introductory material attached to the argument it introduces. Preserve unit order and parent identity in metadata.
3. Extract original text without rewriting it. Preserve page markers, time markers, authorship, version, license, original URL and extraction/OCR limitations. Do not claim an unavailable transcript was obtained. A database row containing a complete article can be a unit; a row containing an isolated chunk cannot.
4. Produce or import a faithful summary for every unit. Summaries should cover the whole unit, including qualifications and counterarguments. Record the summary model or human author. Do not use an opening excerpt as an implicit summary.
5. Write UTF-8 JSONL using the schema in `docs/UNITS.md`. Validate all units before embedding. Use the same embedding model/revision, task prefixes and dimensions for documents and queries. The index fingerprint prevents accidental mixing.
6. Run `mori ... ingest`, then inspect `status` and `plan`. Default retrieval is 500, Jev shortlist 100, final passages 100, full units 1 and summaries 5. Heavier reranking is disabled. Do not silently turn it on or change the requested model.
7. Build evidence first. Inspect `context.json`: all originals must match stored text and offsets; summary-only material must remain labelled. If a full unit exceeds budget, ask the user to choose an explicit smaller unit, lower counts or a larger supported model budget. Never silently truncate a chapter.
8. Generate an answer only after evidence selection. Final citation validation checks supplied IDs, not whether the interpretation is true. Independently review quotations, attribution, argument and evidence coverage before academic/public use.
9. Preserve request, ranking, rubric, source hashes, context and answer. Reuse successful cached ratings on retry. A changed source needs a new unit version; do not overwrite prior inquiries.

## Extending the implementation

Keep retrieval, embeddings, relevance, reranking, summary generation and final answering independent. A database adapter implements the Store contract; a model adapter implements `encode`, `score` or `complete`/`json`. Tests must prove no cross-unit neighbor expansion, no fabricated IDs, explicit budget failures and correct optional-stage behavior. Do not couple the engine to a particular author, topic, database brand or account.

Never commit `.mori/`, `.local/`, uploaded documents, credentials, query histories or provider logs. Run `python -m unittest discover -s tests -v` before submitting changes. Public claims of superiority require a reproducible evaluation against a named baseline.
