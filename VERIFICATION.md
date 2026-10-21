# Verification — four-method scope

Updated 20 October 2026.

- New exploratory query experiment completed over 417 development queries. Three query policies were fixed before evaluation. Selection used 333 questions and reporting used 84; this previously inspected dev split is not independent. Unchanged query processing won for TF-IDF, BM25 and RRF. Baseline nDCG reproduced saved evaluation results.
- Menus, API choices, default method and metric export now retain only TF-IDF, BM25, chunked Sentence Transformer and two-way RRF.
- Removed cross-encoder implementation and downloaded model. Archived historical diagnostic/evaluation records outside the active app and submission.
- Previous desktop keyboard flow passed; evidence remains in submission/keyboard-verification.md. Report print review and independent answer-quality scoring remain pending.
- Reduced-scope integration/build/browser/package checks are recorded below after completion.

## Completed reduced-scope checks

- All four active API methods exactly matched saved top-ten IDs for the bounded development parity query.
- Removed method IDs returned 422. Blank query and expired retrieval returned the expected validation responses.
- A live default-RRF plants question returned a valid source quotation with no warnings through the frontend proxy.
- Production frontend build passed (24 modules). Python compilation passed for changed server, experiment and report scripts.
- Browser chat selector contains exactly four choices, defaults to RRF, and both evaluation splits display four saved rows. Test summary correctly shows nDCG@10 0.6153 and RRF Success@5 85.33%.
- Screenshot: submission/four-methods-dashboard.png. Original layout/theme preserved.

## Selected nDCG card — 21 October 2026

The nDCG summary now follows the evaluation Method selection and labels itself SELECTED nDCG@10. Choosing a chat retrieval method also selects that method for evaluation. The score and name use the same resolved row as the PR curve, including its available-method fallback when changing splits. The comparison chart still highlights the actual best method independently. Production build passed. Browser visual verification was blocked by its connection-error page; no screenshot verification is claimed for this change.

Standard RRF (all three) has been restored in the API, frontend and development snapshot; five live choices, five dev rows and four measured test rows. All five bounded top-ten parity checks passed in the preceding run. Its saved test scores remain unavailable.

## Metric-driven summary — 21 October 2026
The top selected-score card now uses labels[metric] and the selected method row metrics[metric]. Changing the comparison metric, difference-chart metric or heatmap heading updates the card through the shared metric state. Scores use the same four-decimal 0–1 scale as the comparison chart, including Success@5. Production build passed; browser confirmation remains unavailable in the automation tab.
