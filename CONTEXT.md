# Current project context

Updated 20 October 2026. This file supersedes earlier implementation notes.

The user requested removing extra methods, retaining the stronger metric version per method, removing the cross-encoder and trying to improve the original methods. nDCG@10 was explicitly stated as the selection criterion.

Active method IDs (latest user request restores standard RRF): `tfidf`, `bm25`, `semantic_chunked`, `hybrid_chunked_two`, `hybrid`. Standard RRF uses equal weights across TF-IDF/BM25/original semantic, k=60, and has dev-only saved metrics. Default RRF uses BM25/semantic weights 1:4 and k=60. Saved dev nDCG@10 for RRF is 0.472743; test is 0.615327. These original scores were not changed. The two-way RRF is a disclosed deviation from the PPT's equal-weight three-component formula, retained according to the latest user instruction.

The backend reads C:\ARYAN\RAG, which must remain unchanged. Synced sources are read-only. Live search is dev only. Qwen3:4b currently selects validated quotations; free-form synthesis remains deferred.

New development-only query-cleaning experiments found no winning change on selection nDCG@10. All three selected the unchanged policy. Exact records are in submission/query-experiment-dev.json. Do not tune against inspected test data.

Historical records were preserved in the sibling project folder `history-before-four-methods-2026-10-20`, outside the active app and submission. The removed cross-encoder code and downloaded weights were deleted. Only standard RRF was explicitly restored afterward. Do not restore other retired methods.

Preserve English-only copy, the light/slate/blush theme, cartoon outlines and the ownership footer naming Aryan Singh, Aryan Masrani and Chirag Jotwani.

Answer context now includes up to ten retrieved excerpts (550 UTF-8 bytes each), with an 8192-token model window and up to five verified quotations. Sources highlight exact answer quotes in pink and matching keywords in lavender. Live indoor-plants check supplied ten contexts and returned a verified source-1 quotation without warnings; broader context does not guarantee additional citations.
