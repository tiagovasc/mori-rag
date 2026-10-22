# Security and data handling

Mori is a local, single-user research tool. The HTTP server binds to `127.0.0.1`, validates the exact Host and Origin for mutations, and requires a random per-process request token. It has no accounts, hosted access control, durable distributed job queue or multi-tenant isolation. Do not expose it through a public tunnel as a production service.

The browser cannot set provider URLs, credentials or commands. Source text and generated answers are rendered as text, not executed as HTML. Static assets have an explicit allowlist and a restrictive content security policy. Run IDs and upload types are validated; uploads get generated storage filenames. Only one import/query job runs at a time in the local server.

Source and provider material is untrusted. Prompt instructions tell models not to follow source-embedded instructions, but that is not a complete prompt-injection defense. The optional Codex bridge requests a read-only environment without network/web access and a fresh working directory. Trusted local command configuration is still capable of running code: inspect it before using someone else's config.

Credentials come from named environment variables. Mori does not scan unrelated env files or user directories for keys. HTTP/provider errors withhold response bodies and command stderr, since those can include secrets. Do not put secrets directly in TOML fields or model names. The configuration snapshot is part of the saved query record.

Jev receives candidate passages, optional neighbors, summaries and the question. Hosted embedding providers receive source/summary segments and queries. A final model receives the complete selected evidence; summary generation receives the original unit. Decide which destinations are allowed for your data before running imports or queries.

`.mori/` and uploaded sources contain private full texts, vectors, questions, cached decisions and answers. They are ignored by Git. Back them up privately if needed. Each query is preserved; there is no automatic retention purge or account deletion workflow in this local app.

Explicit URL import permits public HTTPS only and rechecks redirects. It is a convenience for local authorized fetching, not a hardened hostile-network crawler; DNS resolution and the eventual connection are separate operations. For untrusted automated feeds or a public service, use an isolated egress-controlled fetcher that pins resolved public addresses, blocks DNS rebinding and enforces site-specific policies.

The importer does not grant rights to redistribute a work. Keep actual source/model licenses and respect access conditions. Cite summary-only sources honestly. Mechanical citation checks establish identity, not truth or entailment.

Report suspected vulnerabilities privately to the repository owner using GitHub's private reporting feature when available; do not post credentials or private source text in an issue.
