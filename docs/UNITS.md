# Source units

A source is a work. A unit is a complete, independently readable segment of that work. A passage is readable original evidence inside that unit; one or more smaller search spans provide its embedding representation. This distinction controls what can be read in full, what is shown as passage evidence and what a summary is actually summarizing.

| Material | Source | Typical unit | Useful locator |
|---|---|---|---|
| Book | Edition of the book | Chapter or explicit subsection | Chapter and page range |
| YouTube | Video or lecture series | One lecture/video | Original URL and timestamps |
| Podcast | Series or episode | Episode or explicit segment | Episode URL and time range |
| Academic article | Version of the paper | Whole paper or named section | DOI, version and page range |
| Wikipedia | Article/revision | Article or named section | Revision URL and section |
| SQL collection | Dataset | Complete article/record with argument | Stable record ID and revision |

## Required JSONL fields

- `id`: stable unique ASCII unit ID, maximum 160 characters; version it when source or summary changes.
- `source_id`: stable parent-work identity. Multiple chapters may share it.
- `title`: human-readable unit title.
- `text`: the complete original unit, preserving wording and locators.
- `summary`: a faithful summary of the entire unit, never an automatically assumed first paragraph.

Optional fields are `kind`, `locator` and a free JSON `metadata` object. Recommended metadata: author, date, edition, URL, license, collection, source hash, OCR/transcription limitations, summary author/model, parent order and timestamp bounds. Metadata is carried into structured context unchanged and rendered as JSON in the readable evidence and model handoff. It is untrusted source data and counts toward the context budget. Non-finite numbers and non-JSON values are rejected before provider calls.

## Choosing boundaries

Keep an argument with its supporting explanation and qualifications. A chapter is often a better unit than a page; an entire anthology is often worse than a chapter. Do not split solely to make a model's context window appear sufficient. If the natural unit is too large, define a meaningful subsection and retain its parent and position.

`--split headings` splits Markdown at level-two headings. It is an explicit convenience, not an intelligent chapter detector. Review the resulting boundaries. PDF page markers aid location but do not automatically define chapters. VTT/SRT timecodes remain in the original text.

The chunker preserves exact character offsets in extracted unit text. Chunk identity includes the unit ID, version hash and offsets. It collapses whitespace-only duplicate passages within the same unit during retrieval; identical passages from distinct units remain separate because their attribution and context can differ. It does not merge similar arguments or paraphrases. Neighbors are always restricted to the same unit. Character offsets refer to extracted text, not original PDF byte offsets or an inferred page number.

## Passages, search spans and compatibility

New indexes default to `chunk_strategy = "sentence"`, `chunk_words = 400` and `chunk_overlap = 50`. These are approximate readable-passage and overlap targets. The splitter prefers sentence/paragraph boundaries and permits up to 25% extra words to finish a passage or absorb a short tail. Overlap uses complete boundaries where possible, so its actual size varies. Short complete units remain short; an unpunctuated sentence beyond the passage budget falls back to a word boundary and records that exception. The detector handles common abbreviations but is a heuristic, not a language-independent sentence parser. Inspect OCR, mathematical notation, lists and punctuation-poor transcripts.

Each passage is divided into exact, nonoverlapping search spans before embedding. Local and ONNX providers count the actual tokenized input including document prefixes and special tokens. The default `embedding_span_tokens = 64` is a configurable retrieval target, capped by the provider's configured/model input limit; MiniLM's input ceiling is 256 tokens. These are separate settings, and changing the retrieval target requires a new index. No source text is reconstructed from decoded tokens. An unbreakable word that cannot fit fails explicitly. A compatible HTTP provider has no local tokenizer contract: its span estimate uses UTF-8 bytes and is a conservative heuristic, not verified token accounting. Configure and test that provider's input limits separately.

Retrieval uses the maximum similarity across a passage's search spans, then returns the passage once. Jev and the answer receive its complete original text; a small matching span is a retrieval key, not the entire cited evidence. Zero `rating_neighbors` or `answer_neighbors` therefore means no *additional passages*, not a loss of the current passage's context. Full-unit and summary branches remain separate.

The new representation participates in the embedding fingerprint. Rebuild into a separate index directory, validate exact source coverage and retrieval, then switch configurations; do not overwrite an existing index or silently reuse its vectors. Preserve the old index and run directories for rollback and historical citation reading. Explicit `chunk_strategy = "words"` with the original model, prefixes, token limit and word-window settings retains the legacy fingerprint and behavior. Changing from the old 120/20 word defaults to the new 400/50 sentence defaults is an index migration, not merely a display preference.

Compatibility note: automatically extracted files retain the initial release's identity algorithm. Their `metadata.original_sha256` is SHA-256 of the UTF-8 hexadecimal representation of the file bytes, rather than the conventional SHA-256 of the raw bytes. Do not compare that legacy field directly with a file-checksum utility. IDs and saved unit hashes are preserved; changing the algorithm requires a versioned import/migration rather than silently rewriting an existing corpus.

## Summaries are evidence of a different kind

Mori ranks complete summaries independently from chunks. The strongest ranked unit supplies the original in full; the next distinct units supply summaries. A summary-only unit is labelled as such in the final prompt and citations. It must not be represented as directly reading its original.

Ingestion can generate missing summaries with `--summarize`, using the configured LLM. Model prompts and provider configuration should be reviewed for the data being sent. Imported summaries should include provenance in metadata. Source truth is never rewritten to match a generated summary.

Repeating a summary-generating import of the same source fields reuses the saved summary and provenance; it does not regenerate a different summary under the existing immutable ID. Changed originals or metadata still need a new version. DOCX extraction keeps visible text, tabs and line breaks and omits deleted revisions and field instructions. It does not preserve page layout or replace visual review of the original document.

For semantic summary preselection, the whole summary is represented by nonoverlapping search spans. New sentence-strategy indexes use the strongest matching summary span, so a narrow relevant part is not averaged away; Jev still sees the complete selected summary. Legacy word-strategy indexes retain the normalized mean representation. Semantic preselection is a recall heuristic and should be evaluated; `summary_candidates = 0` skips that preselection and scores every complete summary directly.
