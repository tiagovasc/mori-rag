# Evaluation

This release tests the evidence pipeline, not a universal claim of superiority. No numerical quality improvement over a benchmark is claimed.

## Automated contract tests

Run `python -m unittest discover -s tests -v`. The suite covers real local SQLite persistence and exact source slices, a 600-passage fixture exercising 500 → 100 → 20 selection, disabled/enabled reranking, complete-unit exclusion, summary identity, neighbor boundaries, import immutability, fingerprint mismatch, budget preflight, explicit overflow, malformed provider responses, decision caching and citation validation.

Network-independent tests inject labelled hash-vector/scoring/answer fixtures. They prove behavior at boundaries, not semantic retrieval quality, Jev accuracy or model reasoning quality. They must not be described as real API tests.

[Prompt extension tests](../tests/test_query_expansion.py) use distinguishable vectors to show that different phrasings can find different passages while sharing one candidate cap. They check source filtering, deduplication, fallback after a failed rewrite or embedding batch, and the exact question passed to Jev, reranking and answering. Separate provider tests check that choosing an extension model leaves the answer model unchanged. These checks do not establish that the extra passages are more relevant.

## Live smoke verification

The author release process uses the redistributable fictional Northbridge corpus, real local semantic embeddings, real Jev decisions and a real final-model call. It checks packet identities, summary/full-source boundaries, citations and saved output. Final measured counts and limitations belong in the release changelog once that run completes.

One small successful query is not an academic-quality benchmark. A useful evaluation should include domain-specific positive/negative queries, reviewed source relevance, relevant material absent from summaries, long-unit context failures and questions that cannot be answered from the collection.

## Ablations to run on your own corpus

Keep the same sources, question and embedding candidates while comparing: passages only; passages plus summaries; complete unit plus summaries plus passages; Jev ranking with and without the optional heavier reranker; neighbors off versus on. Compare answer correctness, omitted qualifications, source coverage, retrieval recall, total provider cost and latency. Rate outputs without exposing the method label to the reviewer when feasible.

Evaluate query expansion separately with the same sources, original questions, embedding model, candidate limit and downstream settings, allowing the retrieved candidates to change. Compare expansion off versus on, record usable variants and fallback frequency, and include the extra expansion call and query embeddings in cost and latency. Synthetic union tests do not establish whether the added candidates are useful on a real collection.

Save question sets, source versions, prompts, provider model versions, relevance thresholds, cache state and full evidence packets. Count all summary-generation/rating/answer calls, including retries and failed attempts. Do not report an API-equivalent estimate as actual subscription billing.
