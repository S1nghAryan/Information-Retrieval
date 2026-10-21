# Five-method demonstration

1. Start Ollama, start-backend.ps1 and npm run dev with the original indexes available.
2. Ask “Which plants grow well indoors?” using TF-IDF, BM25, Sentence Transformer, weighted RRF and Standard RRF (all three).
3. Explain chunked semantic retrieval and the weighted RRF formula: BM25 weight 1, semantic weight 4, k=60. Then demonstrate standard RRF: equal weights for TF-IDF/BM25/original semantic with k=60, matching the PPT formula.
4. Show the top ten sources, validated quotations and exact excerpts supplied to Qwen. Explain the extractive generation limitation.
5. Show five development methods and four test methods (standard RRF is dev-only), a comparison metric, the PR selector and CSV export.
6. Explain that no new query-cleaning candidate improved selection scores; saved test results remain the original frozen results.
7. Show the report and distinguish measured results from pending print review and independent answer evaluation.

First-query index loading is separate from warm search time. Preserve the light theme and ownership footer.

Pending: final presentation rehearsal, printed-page review and institution-specific identifiers.
