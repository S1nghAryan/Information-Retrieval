# Remaining project steps

Updated 20 October 2026, after restoring standard three-way RRF at the user's request.

## Current change

1. [x] **Retain requested methods and remove other duplicates/cross-encoder.**
   - Why: keep one version of each requested retrieval family.
   - Retained: TF-IDF, BM25, chunked Sentence Transformer, two-way weighted RRF (1:4, k=60), plus restored standard three-way RRF (equal weights, k=60). Selection uses nDCG@10, not a claim of winning every metric.
2. [x] **Try improvements within the retained methods.**
   - Why: improve retrieval without adding another ranking family.
   - Result: two query-cleaning alternatives were tested for TF-IDF, BM25 and their effect on RRF. The unchanged configuration won each selection comparison. No inflated or invented gains were published.
3. [x] **Verify the reduced API, UI, report and package.**
   - Why: removed methods must not survive in menus, exports, defaults or runnable backend paths.
   - Complete when: five-mode parity, rejected retired API IDs, frontend build, browser inspection and package hashes pass.

## Submission work

4. [x] **Desktop main-flow keyboard check.**
   - Why: submission, citations and evaluation controls should work without a mouse.
   - Completed before the method reduction; the native controls remain. See keyboard-verification.md. Broader accessibility certification is not claimed.
5. [ ] **Review the report in print preview.**
   - Why: screen inspection cannot establish page breaks, clipping and printed table legibility.
   - Browser local-file access was blocked by URL policy. Manual print preview remains pending.
6. [ ] **Run the final five-method presentation.**
   - Why: the assignment requires a live demonstration in addition to code.
   - Show one common question using each method, explain both RRF formulas, display sources, metrics, PR graphs and CSV export.
7. [ ] **Confirm required student/course identifiers and submission destination.**
   - Why: the local project does not specify institutional details.
   - The three requested names are present. No external submission is authorized.
8. [ ] **Confirm the expected LLM answer format.**
   - Why: the PPT asks for LLM generation; current source quotation selection is narrower than synthesized prose.
   - A synthesized mode needs an independent faithfulness evaluation before deployment.

## Optional further accuracy work

9. [ ] **Independent answer-quality set and query-focused excerpts.**
   - Why: quotation matching proves provenance, not relevance or completeness; useful source content may fall beyond the excerpt budget.
   - Use separately reserved examples, preserve original wording and conditions, and record failures.
10. [ ] **Further original-method tuning with a new validation protocol.**
    - Why: repeated tuning on the existing dev split risks overfitting; test data has already been inspected.
    - Possible work: BM25 parameter sweeps and semantic chunk-size comparisons with separate candidate indexes. Do not report a gain until measured, and do not overwrite original artifacts.
11. [ ] **Broader latency, failure and device checks if claiming production readiness.**
    - Why: bounded local checks do not establish concurrent load, cold-start behavior on other hardware or universal accessibility.

Historical scope and retired-method evidence are archived outside the app. They are not current completion claims.
