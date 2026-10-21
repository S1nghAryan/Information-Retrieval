# Existing-method improvement experiment

Exploratory development only; dev was used in earlier tuning. No test labels or rankings are read.

Three fixed query policies: unchanged, remove common question words, remove English stopwords while retaining not/no/never. Empty filtered queries fall back to original terms. Passage indexes, stemming and ranking formulas stay fixed. The semantic ranking stays unchanged; RRF uses the candidate BM25 ranking with semantic weight 4 and k=60.

| Method | Selected query policy | Baseline validation nDCG@10 | Candidate validation nDCG@10 | Baseline R@10 | Candidate R@10 |
|---|---|---:|---:|---:|---:|
| tfidf | baseline | 0.247272 | 0.247272 | 0.331760 | 0.331760 |
| bm25 | baseline | 0.375273 | 0.375273 | 0.441199 | 0.441199 |
| hybrid_chunked_two | baseline | 0.541521 | 0.541521 | 0.577537 | 0.577537 |

No automatic promotion or dashboard replacement. This split was previously inspected and is not fresh held-out evidence. Baseline scores reproduce the original evaluator. Exact policy lists, split IDs and selection/validation results are in query-experiment-dev.json.
