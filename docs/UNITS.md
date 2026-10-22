# Source units

A source is a work. A unit is a complete, independently readable segment of that work. A chunk is a retrieval window inside the unit. This distinction controls what can be read in full and what a summary is actually summarizing.

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

Optional fields are `kind`, `locator` and a free `metadata` object. Recommended metadata: author, date, edition, URL, license, collection, source hash, OCR/transcription limitations, summary author/model, parent order and timestamp bounds. Metadata is carried into final context unchanged.

## Choosing boundaries

Keep an argument with its supporting explanation and qualifications. A chapter is often a better unit than a page; an entire anthology is often worse than a chapter. Do not split solely to make a model's context window appear sufficient. If the natural unit is too large, define a meaningful subsection and retain its parent and position.

`--split headings` splits Markdown at level-two headings. It is an explicit convenience, not an intelligent chapter detector. Review the resulting boundaries. PDF page markers aid location but do not automatically define chapters. VTT/SRT timecodes remain in the original text.

The chunker preserves exact character offsets in extracted unit text. Chunk identity includes the unit ID, version hash and offsets. It collapses whitespace-only duplicate passages during retrieval but does not merge similar arguments or paraphrases. Neighbors are always restricted to the same unit. Character offsets refer to extracted text, not original PDF byte offsets or an inferred page number.

## Summaries are evidence of a different kind

Mori ranks complete summaries independently from chunks. The strongest ranked unit supplies the original in full; the next distinct units supply summaries. A summary-only unit is labelled as such in the final prompt and citations. It must not be represented as directly reading its original.

Ingestion can generate missing summaries with `--summarize`, using the configured LLM. Model prompts and provider configuration should be reviewed for the data being sent. Imported summaries should include provenance in metadata. Source truth is never rewritten to match a generated summary.

For semantic summary preselection, the whole summary is embedded in nonoverlapping segments and their normalized mean is stored. This prevents silent truncation of long summaries. Jev still sees the complete selected summary. This average embedding is a recall heuristic, not a replacement for the summary's meaning.
