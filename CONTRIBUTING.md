# Contributing

Use Python 3.11+. Install the core in editable mode and run `python -m unittest discover -s tests -v`. Frontend edits also require `node --check mori/static/app.js` and a real local browser check at desktop and phone widths.

Keep changes independent of private corpora and account-specific paths. Add a provider behind the existing embedding/relevance/LLM interfaces; document its input contract, privacy behavior, token limits and how to reproduce a smoke test. New source adapters must preserve original unit boundaries and provenance rather than rewrite text during ingestion.

Never turn provider errors into fabricated relevance scores, silently clip originals, assume summary-only material was directly read, or change stored unit text in place. A bug fix must include a behavioral regression test. Preserve stable evidence IDs and old query packets.

Public claims about quality, speed or cost need a reproducible evaluation, a defined baseline and limitations. Do not commit user datasets, `.mori/`, `.local/`, upload contents, credentials or provider logs. The MIT license covers contributions to Mori, not third-party material imported by users.
