# Providers

## MiniLM or OpenAI

Use `mori init --embeddings minilm` for the speed-first local preset, or `mori init --embeddings openai` for the quality-first hosted preset (`--output` selects the new TOML file). MiniLM's compact local model usually favors speed and cost; the larger OpenAI model is the higher-quality option to evaluate when recall and semantic distinctions matter more. This relative guidance is not a corpus-specific benchmark or a guarantee about network latency.

The OpenAI preset uses `text-embedding-3-large`, `https://api.openai.com/v1/embeddings`, and `OPENAI_API_KEY`; it requests 3,072 float dimensions. `embedding_dimensions` can request a supported smaller size, trading representation capacity for storage/search cost. The returned shape must match the requested size. Use a new data directory and re-embed when changing any model/dimension setting. [Official guide](https://developers.openai.com/api/docs/guides/embeddings)

MiniLM was verified live locally. OpenAI's request/response contract is covered by mocked provider tests; no live OpenAI embedding call or speed/quality comparison is claimed in this release.

## Local semantic embeddings

Install `.[local]` and use `embedding = "local"`. The default model is `sentence-transformers/all-MiniLM-L6-v2`. The selected Sentence Transformer's normal pooling/projection is used, with normalized output. Remote custom model code is not enabled. Pin `embedding_revision` when reproducibility matters.

Inputs are checked against the model's token limit before inference. Overlong passages fail instead of being silently clipped; reduce `chunk_words` and create a new index, or select a suitable longer-context embedding model. Query/document prefixes are separately configurable and included in the fingerprint.

## Existing ONNX model

Install `.[onnx]` and point `embedding_model` at a directory containing `tokenizer.json` and the configured ONNX model file:

```toml
embedding = "onnx"
embedding_model = "/path/to/model-directory"
embedding_revision = "your-pinned-revision"
onnx_model_file = "onnx/model_quantized.onnx"
onnx_pooling = "mean"
embedding_token_limit = 256
```

Mean pooling uses attention-masked `last_hidden_state` and is appropriate for the MiniLM example. It is not a universal model rule. Models with a trained projected `sentence_embedding` output require `onnx_pooling = "sentence_embedding"`. Never replace a trained projection with raw-token pooling. Obtain model weights from a trusted source and follow their license. Mori bundles no weights.

Mori disables any truncation stored in `tokenizer.json` before checking the full tokenized input against `embedding_token_limit`. Set that limit only to a capacity verified for your actual model. ONNX fingerprints now include `onnx_tokenization = "untruncated-v1"`; older ONNX indexes may contain silently clipped embeddings and must be rebuilt in a separate data directory. Original units and previous query records need not be deleted. Other embedding providers retain their existing fingerprint compatibility.

## Compatible embedding endpoint

```toml
embedding = "http"
embedding_model = "your-embedding-model"
embedding_url = "http://127.0.0.1:11434/v1/embeddings"
embedding_key_env = "MORI_EMBEDDING_KEY"
```

The endpoint accepts `{model, input: [text, ...]}` and returns `data: [{index, embedding}]`. Set the named environment variable if authentication is needed. Returned indices, vector shape and finite values are validated. Remote providers may apply their own tokenization/truncation; verify their contract rather than assuming Mori can detect hidden server behavior.

Provider HTTP redirects are rejected for embeddings, relevance and answering. Configure the final endpoint directly; Mori will not forward credentials or source data to a redirect destination. Summary embeddings use bounded batches, and every segment must have the same dimension before pooling. Normalization also rejects zero or invalid vectors without silently producing zero vectors from floating-point overflow.

If an embedding response supplies a `model` field, it must match `embedding_model` exactly. Configure a gateway's concrete returned identity instead of an alias that resolves to another name. An endpoint that omits its model identity cannot provide this additional check; pinned model revisions and endpoint controls still matter.

## Jev / TypeSafe

Defaults: `jev_url = "https://api.typesafe.ai/v1/systemone"`, `jev_model = "jev-latest"`, `jev_key_env = "TYPESAFE_API_KEY"`. Every item gets one `score` question using the full configured rubric. This adapter preserves distributions, fractional expected score, confidence and returned model identity. Use 2–10 nonempty rubric levels. The endpoint and model can be changed explicitly for a compatible gateway.

The source text and original question are sent to the selected service. Credentials are read from the named environment variable only; Mori never looks for a private workspace env file. The default request-byte budget is an application guard and may be lower than a provider's limits. Full summaries exceeding it stop with an explicit message.

## Compatible answer endpoint

```toml
llm = "http"
llm_url = "http://127.0.0.1:11434/v1/chat/completions"
llm_model = "your-answer-model"
llm_key_env = "MORI_LLM_KEY"
llm_max_output_tokens = 4096
```

The endpoint accepts `model`, `messages` and `max_tokens`, and returns `choices[0].message.content`. A `finish_reason` of `length` fails rather than publishing a truncated answer. Provider-specific parameter mappings belong in another adapter; not all hosted models accept this schema.

## Trusted command / Codex

`llm = "command"` invokes `llm_command` as an argument array with `shell=False`. It receives one JSON object on stdin: `{task, prompt, model}`. Return `{text}` as the only JSON object on stdout; diagnostics belong on stderr. `task` is `summarize`, `rerank` or `answer`. Commands are trusted server configuration and cannot be set through uploads or browser settings. Do not interpolate source text into shell commands.

`adapters/codex/bridge.mjs` implements that contract using the official Codex SDK. Install with `npm ci --ignore-scripts` inside its directory, authenticate Codex separately, then use `examples/codex.toml` from the project root. The model and reasoning effort (`MORI_CODEX_EFFORT`) are configurable. This is a local user workflow, not a promise that a subscription can host an unrestricted public API.

The bridge sends the supplied source/prompt to the authenticated Codex service. Its sandbox network restriction controls agent tools, not model inference. Codex may save prompt content in its own session history outside the Mori data directory. Keeping a corpus out of Git does not make hosted processing offline; choose allowed data destinations explicitly.

The subprocess receives an explicit runtime and Codex authentication environment allowlist; unrelated credentials such as the Jev key, Node startup hooks and proxy variables are not inherited. Shell, unified execution and hooks are disabled as additional restrictions. Existing Codex configuration and possible MCP tools still require review; this adapter does not claim complete tool isolation.

## Manual handoff

`llm = "none"` still saves ranked `context.json`, `context.md` and `answer_prompt.txt`. Give the prompt to the LLM of your choice. This mode does not generate missing summaries or perform optional LLM reranking. It is the default because an application should not silently choose a paid answer provider.
