# Security and data handling

Mori is a local, single-user research tool. The HTTP server binds to `127.0.0.1`, validates the exact Host and Origin for mutations, and requires a random per-process request token. It has no accounts, hosted access control, durable distributed job queue or multi-tenant isolation. Do not expose it through a public tunnel as a production service.

The browser cannot set provider URLs, credentials or commands. Source text and generated answers are rendered as text, not executed as HTML. Static assets have an explicit allowlist and a restrictive content security policy. Run IDs and upload types are validated; uploads get generated storage filenames. Only one import/query job runs at a time in the local server.

The redesigned interface's endpoints follow the same rules. Stop (`/api/cancel`) and verdict saves are mutations with the token, Host and Origin checks. A verdict is accepted only for a citation ID present in that inquiry's saved context, with the value yes, partly or no, or none to clear it. `/api/unit` returns one unit's text from the room's own collections and requires exactly one ID. Fonts are served only from a fixed allowlist. The live rating board holds short passage excerpts in memory for the running job only; it is not written to run folders, logs or progress observers.

Source and provider material is untrusted. Prompt instructions tell models not to follow source-embedded instructions, but that is not a complete prompt-injection defense. The optional Codex bridge requests a read-only sandbox, disables sandbox network/web access, and uses a fresh working directory. These restrictions apply to agent tools: Codex inference is still hosted and receives the supplied prompt. The SDK may retain prompts in the user's Codex session history outside Mori's data directory. Trusted local command configuration is still capable of running code: inspect it before using someone else's config.

Credentials come from named environment variables. Mori does not scan unrelated env files or user directories for keys. HTTP/provider errors withhold response bodies and command stderr, since those can include secrets. Do not put secrets directly in TOML fields or model names. The configuration snapshot is part of the saved query record.

Provider endpoints must be final URLs: HTTP redirects are rejected so bearer credentials and source data are never forwarded to another endpoint. Explicit public-source fetching separately supports checked redirects, pins connections to validated public IP addresses, retains TLS hostname verification, and bypasses environment proxies. Upload `summarize` and query `answer` switches require real JSON booleans; a string such as `"false"` cannot activate model calls.

Choose provider destinations by the material each stage receives. Running the interface on localhost does not make hosted inference local.

| Stage | What the configured provider receives |
|---|---|
| Prompt extension (off by default) | Your original question and instructions for generating search phrasings. No passages or summaries. |
| Embeddings | At import: source and summary spans. At query time: the original and, when enabled, usable generated phrasings. Local embedding models process these on your machine. |
| Jev relevance | The original question and each candidate passage with its configured neighbors, or a complete summary. Generated phrasings are not sent to Jev. |
| Optional reranking and answering | The reranker receives the original question and shortlisted passage evidence; the answer model receives the original question and assembled evidence. |
| Optional summary generation | The original unit to summarize. |

Keep the data directory private: it contains full texts, vectors, questions, saved search phrasings, cached decisions, answers and citation verdicts. The supplied Git ignore rules cover `.mori/`, `.mori-*/` and `.local/`. If you choose a different `--data` directory inside a checkout, exclude it explicitly before committing. Each inquiry is preserved; there is no automatic retention purge.

Explicit URL import permits public HTTPS only, rechecks redirects and pins connections to the validated public address. It is a convenience for local authorized fetching, not a hardened hostile-network crawler. For untrusted automated feeds or a public service, also use an isolated egress-controlled fetcher with site-specific policies and resource limits.

The importer does not grant rights to redistribute a work. Keep actual source/model licenses and respect access conditions. Cite summary-only sources honestly. Mechanical citation checks establish identity, not truth or entailment.

Report suspected vulnerabilities privately to the repository owner using GitHub's private reporting feature when available; do not post credentials or private source text in an issue.
