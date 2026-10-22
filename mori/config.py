"""Validated portable configuration. No credentials are serialized here."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import tomllib


@dataclass
class Config:
    retrieval_k: int = 500
    jev_k: int = 100
    final_passages: int = 100
    full_units: int = 1
    summaries: int = 5
    min_relevance: float = 40.0
    summary_min_relevance: float = 40.0
    summary_candidates: int = 0  # 0 = score every unit's complete summary
    rerank_enabled: bool = False
    rerank_k: int = 20
    rating_neighbors: int = 0
    answer_neighbors: int = 0
    exclude_full_unit_passages: bool = True
    chunk_words: int = 120
    chunk_overlap: int = 20
    context_max_chars: int = 180000
    llm_max_input_chars: int = 200000
    unit_max_chars: int = 500000
    request_max_bytes: int = 120000
    max_rating_calls: int = 2000
    workers: int = 8
    timeout: int = 90
    retries: int = 2
    embedding: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = ""
    onnx_model_file: str = "onnx/model_quantized.onnx"
    onnx_pooling: str = "mean"
    embedding_token_limit: int = 256
    embedding_url: str = "http://127.0.0.1:11434/v1/embeddings"
    embedding_key_env: str = "MORI_EMBEDDING_KEY"
    query_prefix: str = ""
    document_prefix: str = ""
    embedding_batch: int = 32
    embedding_dimensions: int = 0
    jev_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"
    jev_key_env: str = "TYPESAFE_API_KEY"
    llm: str = "none"
    llm_model: str = ""
    llm_url: str = "http://127.0.0.1:11434/v1/chat/completions"
    llm_key_env: str = "MORI_LLM_KEY"
    llm_command: list[str] = field(default_factory=list)
    llm_max_output_tokens: int = 4096
    rubric: list[str] = field(default_factory=lambda: [
        "Unrelated: no substantive connection to the question.",
        "Incidental: shares words but no useful evidence.",
        "Background: establishes broad context only.",
        "Tangential: touches a related issue indirectly.",
        "Partial: helps explain one minor aspect.",
        "Useful: addresses a substantive part of the question.",
        "Strong: directly explains a central aspect with supporting detail.",
        "Very strong: closely answers a main issue with meaningful context.",
        "Excellent: directly answers most of the question with specific evidence.",
        "Exact: directly and comprehensively addresses the requested issue.",
    ])

    def validate(self):
        for name in ("retrieval_k", "jev_k", "final_passages", "rerank_k", "chunk_words",
                     "context_max_chars", "llm_max_input_chars", "unit_max_chars", "request_max_bytes",
                     "max_rating_calls", "workers", "timeout", "embedding_batch", "llm_max_output_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("full_units", "summaries", "summary_candidates", "rating_neighbors", "answer_neighbors", "chunk_overlap", "retries", "embedding_dimensions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("min_relevance", "summary_min_relevance"):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"{name} must be between 0 and 100")
        for name in ("rerank_enabled", "exclude_full_unit_passages"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.jev_k > self.retrieval_k or self.final_passages > self.jev_k or self.rerank_k > self.jev_k:
            raise ValueError("Passage counts must follow retrieval_k >= jev_k >= final_passages/rerank_k")
        if self.chunk_overlap >= self.chunk_words:
            raise ValueError("chunk_overlap must be smaller than chunk_words")
        if not isinstance(self.rubric, list) or not 2 <= len(self.rubric) <= 10 or any(not isinstance(x, str) or not x.strip() for x in self.rubric):
            raise ValueError("Jev requires 2–10 nonempty ordered rubric descriptions")
        if self.embedding not in ("local", "onnx", "http", "demo") or self.llm not in ("none", "http", "command"):
            raise ValueError("Unknown provider")
        if type(self.embedding_token_limit) is not int or self.embedding_token_limit < 8 or self.onnx_pooling not in ('mean', 'sentence_embedding'):
            raise ValueError("Invalid embedding token limit or ONNX pooling")
        if not isinstance(self.llm_command, list) or any(not isinstance(x, str) or not x for x in self.llm_command):
            raise ValueError("llm_command must be an argument array, never a shell command string")
        if self.llm == "command" and not self.llm_command:
            raise ValueError("command provider needs llm_command")
        if self.workers > 64 or self.retrieval_k > 10000 or self.rating_neighbors > 10 or self.answer_neighbors > 10:
            raise ValueError("Requested concurrency, retrieval or neighbors exceeds the reference implementation limit")
        return self

    def public(self):
        return asdict(self)


def load(path=None, overrides=None):
    values = tomllib.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if set(values) - set(Config.__dataclass_fields__):
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(set(values) - set(Config.__dataclass_fields__))))
    values.update(overrides or {})
    return Config(**values).validate()
