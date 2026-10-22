# Configuration

Every `Config` dataclass field in `mori/config.py` is accepted as a top-level TOML field. Unknown fields and invalid types fail. `examples/mori.toml` is the normal starting point; omitted fields retain defaults.

| Setting | Default | Meaning |
|---|---:|---|
| `retrieval_k` | 500 | Maximum unique semantic passage candidates |
| `jev_k` | 100 | Maximum relevant passages after Jev and full-unit exclusion |
| `final_passages` | 100 | Final cap when heavier reranking is off |
| `rerank_enabled` | false | Additional LLM passage ordering |
| `rerank_k` | 20 | Final cap when heavier reranking is on |
| `full_units` | 1 | Complete originals chosen through the summary branch |
| `summaries` | 5 | Other, distinct unit summaries |
| `summary_candidates` | 0 | 0 scores all summaries; positive values semantically preselect that many |
| `min_relevance` | 40 | Passage threshold, normalized 0–100 |
| `summary_min_relevance` | 40 | Summary/full-source relevance threshold |
| `rating_neighbors` | 0 | Number of chunks on each side shown to Jev |
| `answer_neighbors` | 0 | Number on each side included in the final packet |
| `exclude_full_unit_passages` | true | Avoid repeating units already included in full |
| `chunk_words` / `chunk_overlap` | 120 / 20 | Deterministic whitespace-window sizes |
| `context_max_chars` | 180000 | Maximum complete rendered evidence packet |
| `llm_max_input_chars` | 200000 | Maximum prompt sent to a final/summary/rerank provider |
| `unit_max_chars` | 500000 | Import boundary guard; define smaller units explicitly if exceeded |
| `max_rating_calls` | 2000 | Maximum planned decisions before starting; retries may multiply HTTP calls |
| `workers` | 8 | Concurrent relevance decisions |
| `timeout` | 90 | Provider/command timeout, seconds |
| `retries` | 2 | Additional HTTP retries after the first attempt |
| `request_max_bytes` | 120000 | Jev request body guard, measured in UTF-8 bytes |

Counts satisfy `retrieval_k >= jev_k >= final_passages` and `jev_k >= rerank_k`. The reference implementation caps semantic candidates at 10,000, workers at 64 and neighbor radius at 10. These are implementation guards, not provider promises. Zero full units/summaries are allowed for explicit ablation, but the normal design uses both branches.

## Qualitative anchors and numeric caps

The default ten levels have evenly spaced numeric anchors at `0, 11.11, 22.22, …, 100` relevance points. The text explains the decision at every anchor. The following ranges are interpretations of those anchors, not new API categories or correctness probabilities:

| Normalized range | Qualitative interpretation |
|---|---|
| 0–<22.22 | Unrelated to incidental connection |
| 22.22–<44.44 | Background or tangential connection |
| 44.44–<66.67 | Partial to useful substantive evidence |
| 66.67–<88.89 | Strong to very strong direct relevance |
| 88.89–100 | Excellent to exact substantive match |

The provider is asked to choose across the ten fully described ordinal levels; the expected score gives a continuous normalized result. A modal category and expected score may differ when the distribution spans categories. Preserve both. If you replace the rubric, document your own anchors and evaluate your own thresholds rather than inheriting these ranges blindly.

Every API criterion includes its numeric display anchor and the surrounding midpoint-to-midpoint band alongside the qualitative description. For example, level 7 has anchor 77.78 and band 72.22–83.33. Those bands describe ordinal anchors; the broader ranges above summarize neighboring meanings. Neither is an accuracy calibration.

No threshold should be interpreted as universal sufficiency for academic use. Change a cap only after checking the resulting evidence against real questions and original sources.
