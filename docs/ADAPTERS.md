# Adapting Mori to a collection or database

The stable ingestion boundary is UTF-8 JSONL containing complete original units and their summaries. It is intentionally independent of a database vendor. Do not export rows containing only isolated chunks and call them complete originals.

## Existing SQL database

Map your schema explicitly:

```python
import json

# db is a connection the user has explicitly authorized.
rows = db.execute("SELECT work_id, chapter_id, title, text, summary FROM chapters")
with open("units.jsonl", "w", encoding="utf-8") as output:
    for work, chapter, title, original, summary in rows:
        output.write(json.dumps({
            "source_id": str(work), "id": f"{work}:{chapter}:v1",
            "title": title, "text": original, "summary": summary,
            "kind": "chapter", "locator": str(chapter)
        }, ensure_ascii=False) + "\n")
```

Then run `mori ingest units.jsonl`. This route re-embeds the exported units with your configured model; it does not silently reinterpret incompatible existing vectors. A JSONL export is a snapshot, not a live bidirectional database sync.

## External vector database

Replace the storage/search contract when necessary. The reference `Store` supplies `units()`, `chunks()`, `fingerprint()`, `stats()`, `cache_get()`, `cache_put()` and a local `root` for immutable run artifacts. `query()` accepts that object by interface rather than checking its concrete type. Returned units need complete originals, summaries, metadata and identity hashes; chunks need exact text, IDs, unit IDs, sequence, offsets and normalized vectors. Use `ingest()`/`validate_unit()` to enforce the same schema at adapter boundaries.

In new sentence-strategy indexes, each row from `chunks()` is a readable parent passage. `search_vectors` holds its normalized child vectors and `embedding_spans` holds the corresponding `{seq, start, end}` records, with absolute offsets into the original unit. Both arrays must be nonempty and aligned, with contiguous sequences and complete ordered source coverage. Vectors must be finite, nonzero and unit-normalized. The fingerprint declares `retrieval_pooling = "max-child-cosine-v1"`. Retrieval computes the maximum child cosine, retains one parent candidate and records the winning child as `retrieval_match = {seq, start, end, score}`; sequence breaks equal-score child ties. Do not replace the children with one averaged vector or emit each child as a separate evidence passage.

Units can similarly provide `summary_search_vectors` and `summary_embedding_spans`, with offsets relative to the summary, under `summary_pooling = "max-child-cosine-v1"`. These vectors are used only when semantic summary preselection is enabled. Jev always receives the complete summary. A declared new representation with missing/empty child data fails rather than falling back to legacy vectors. Explicit legacy word-strategy indexes retain their old vector representation and fingerprint.

The saved context keeps the target and optional neighbors for provenance, plus a contiguous `context = {text, start, end}` slice of the same original unit. This combined slice represents overlapping neighboring passages once in the prompt/source reader; do not concatenate duplicated overlap text. Vector arrays are internal search data and are excluded from query artifacts. Preserve winning child offsets, parent identity, source selection and full original text across adapter boundaries.

The current query implementation performs exact cosine search over `chunks()`. Supporting an ANN service efficiently requires replacing that retrieval step and fetching same-unit neighbors explicitly; returning millions of vectors through this method is not a scalable adapter. Preserve stable IDs, original source retrieval, explicit filters and deterministic tie handling. Declare approximate retrieval's recall tradeoff and test against an exact reference. No Supabase, pgvector, Qdrant or other remote connector is claimed to ship in this initial version.

## Source selection in the research room

The web interface also requires `source_catalog() -> [{id, label, units, chunks}]` and `select_sources(ids) -> selected_store`. The reference store groups by `source_id`; an adapter may group compatible existing indexes into named collections. A selected view must restrict `units()`, `chunks()` and `stats()` together, preserve original identities/fingerprints and reject empty, unknown or duplicate IDs before provider work. It may share caches, but must never broaden a selection when returning full sources, summaries or neighbors. Its `selection` catalog is saved as `sources` in the request and context packet.

`create_server(store, config, port, allow_uploads=False)` binds the research interface with uploads disabled, without running its event loop; `serve()` remains the ordinary blocking entry point. Queries still write their own cache and run artifacts. An optional `run_roots()` returns trusted local store roots whose historical inquiries should appear alongside new runs. The first root wins duplicate run IDs. Only the adapter supplies these paths; clients cannot choose filesystem roots. Legacy `/open?collection=<id>` links resolve to a same-origin chat selection.

## Websites and Wikipedia

`mori fetch` imports a single explicitly supplied public HTTPS URL, with redirect checks and a byte cap. HTML extraction removes script/style/navigation text but is not a site-specific article parser. Inspect the result; boilerplate and extraction errors can remain. Preserve source URLs, revision/date and actual license. Wikipedia revision exports or an API-derived JSONL adapter are preferable when reproducibility matters. The importer does not bypass access controls or crawl a domain.

Each source connection uses the public numeric address validated for that request, with normal TLS certificate/hostname verification against the URL host. Redirects are resolved and checked again before connecting. This prevents a second DNS lookup from redirecting the connection to a private address. Source fetching connects directly and does not use environment HTTP proxies.

## YouTube and podcasts

Obtain an authorized transcript or subtitle file through the source's normal export, a user-provided transcript, or a collection-specific connector. Import VTT/SRT/TXT and keep episode/video URLs and timestamps in unit metadata. A podcast page or audio download is not itself a transcript. Transcription, platform authentication, paywall handling and media downloading are deliberately separate from this retrieval engine. An agent can add a connector under this contract without changing ranking or answer assembly.

## Books, papers and files

Use original text or reviewed OCR. For long books, generate an explicit unit manifest at chapter boundaries. DOCX extraction preserves text but not pagination; PDF extraction adds page markers and does not invent OCR. Check quotations against the source document before publication.

## Provider interfaces

The research room exposes the configured answer model and additional `llm_models` as a dropdown in Settings. The default `llm_model` is included automatically. Configure only models supported by the same LLM adapter, credentials and input budget; the menu does not discover provider availability or adjust context limits. The server validates each requested model against its startup catalog before provider work. A selection applies to that inquiry's answer and optional reranking; imports and future requests without an override retain the server default. Clients cannot change the catalog, provider type, endpoint, credentials or command.

Context-budget mode adds an optional `context_budget` object to the existing `mori.context.v1` packet. It accounts for the serialized answer prompt, output/overhead reserves, required full-unit/summary bundle, packed passages and active limits. Old packets remain readable without it. Candidate selection remains bounded by retrieval, relevance and the Jev shortlist; reranking can reorder that shortlist but its fixed final count is ignored in budget mode. The current byte-based token estimate and command-provider caveats are documented in `PROVIDERS.md`; do not treat the estimate as measured provider usage.

- Embedding provider: `.fingerprint` plus `encode(texts, query=False) -> normalized vectors`. Return one vector per input, in input order, with consistent dimensions. Optional query expansion sends the original question followed by usable distinct variants in one `encode(..., query=True)` batch, so adapters must support multiple query inputs. All use the same query-task semantics, model and dimensions as ordinary retrieval; document/query semantics must remain compatible with the stored index. If embedding the expanded batch fails, Mori retries only the original question; failure of that original query still propagates.
- Relevance provider: `score(question, text, purpose) -> {score: 0..100, label, ...}`. Fail on provider errors; never substitute a neutral or zero score for a failed request.
- LLM provider: `complete(task, prompt) -> text`, `json(task, prompt) -> parsed JSON`. `task` separates summary generation (`summarize`), optional query expansion (`query_expansion`), optional reranking (`rerank`) and answering (`answer`). Command providers receive `{task, prompt, model}` on standard input and return `{text}`. The built-in HTTP and command transports use `query_expansion_model`, when configured, only for `query_expansion`; other tasks retain `llm_model`. An injected LLM adapter must implement its own task/model routing.

Query expansion requests a JSON object with four string fields: `improved`, `specific`, `broad` and `hypothetical_answer`. The engine accepts nonempty strings of at most 4000 characters, discards unusable fields and deduplicates variants against the original question and one another after whitespace normalization and case folding. Failed or wholly unusable responses fall back to the original question. This optional source-free call also runs for evidence-only queries when enabled. Generated text is retrieval input, never evidence: relevance scoring, reranking and answering retain the original question. The bounded interleaved candidate union preserves source selection and attribution; saved similarities and winning-child offsets remain those for the original query. See [evaluation](EVALUATION.md) for the fixture coverage of these boundaries.

Inject providers into `ingest()`/`query()` for an integration or evaluation. Tests use explicit fixtures, never claim those are semantic models or paid providers.

For sentence-strategy ingestion, an embedding adapter may define `embedding_spans(text) -> [{seq, start, end, text}]` on its class, using offsets relative to its input. Return complete ordered coverage with exact source slices and validate capacity against the full prefixed input, including special tokens. Built-in local/ONNX providers use the real tokenizer; the HTTP provider explicitly fingerprints its UTF-8 byte estimate because it has no local tokenizer. An adapter without this declared method receives whole parent passages and is fingerprinted as `adapter-whole-parent-v1`; that adapter remains responsible for rejecting inputs beyond its own model capacity. Mori does not infer a safe word/token ratio for it. Include representation and tokenizer-policy changes in the adapter fingerprint so incompatible vectors cannot silently mix.
