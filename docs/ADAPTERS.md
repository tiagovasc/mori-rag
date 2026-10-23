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

The current query implementation performs exact cosine search over `chunks()`. Supporting an ANN service efficiently requires replacing that retrieval step and fetching same-unit neighbors explicitly; returning millions of vectors through this method is not a scalable adapter. Preserve stable IDs, original source retrieval, explicit filters and deterministic tie handling. Declare approximate retrieval's recall tradeoff and test against an exact reference. No Supabase, pgvector, Qdrant or other remote connector is claimed to ship in this initial version.

## Websites and Wikipedia

`mori fetch` imports a single explicitly supplied public HTTPS URL, with redirect checks and a byte cap. HTML extraction removes script/style/navigation text but is not a site-specific article parser. Inspect the result; boilerplate and extraction errors can remain. Preserve source URLs, revision/date and actual license. Wikipedia revision exports or an API-derived JSONL adapter are preferable when reproducibility matters. The importer does not bypass access controls or crawl a domain.

Each source connection uses the public numeric address validated for that request, with normal TLS certificate/hostname verification against the URL host. Redirects are resolved and checked again before connecting. This prevents a second DNS lookup from redirecting the connection to a private address. Source fetching connects directly and does not use environment HTTP proxies.

## YouTube and podcasts

Obtain an authorized transcript or subtitle file through the source's normal export, a user-provided transcript, or a collection-specific connector. Import VTT/SRT/TXT and keep episode/video URLs and timestamps in unit metadata. A podcast page or audio download is not itself a transcript. Transcription, platform authentication, paywall handling and media downloading are deliberately separate from this retrieval engine. An agent can add a connector under this contract without changing ranking or answer assembly.

## Books, papers and files

Use original text or reviewed OCR. For long books, generate an explicit unit manifest at chapter boundaries. DOCX extraction preserves text but not pagination; PDF extraction adds page markers and does not invent OCR. Check quotations against the source document before publication.

## Provider interfaces

- Embedding provider: `.fingerprint` plus `encode(texts, query=False) -> normalized vectors`. The query/document task semantics must match the stored index.
- Relevance provider: `score(question, text, purpose) -> {score: 0..100, label, ...}`. Fail on provider errors; never substitute a neutral or zero score for a failed request.
- LLM provider: `complete(task, prompt) -> text`, `json(task, prompt) -> parsed JSON`. `task` separates summary generation, optional reranking and answering.

Inject providers into `ingest()`/`query()` for an integration or evaluation. Tests use explicit fixtures, never claim those are semantic models or paid providers.
