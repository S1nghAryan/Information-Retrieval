# Bounded answer-quality review — 20 October 2026

This review covers five local questions used while refining the implementation. They are development regression examples, not an independent answer-accuracy benchmark. The original outputs are preserved in answer-validation-before.json; the current API outputs and supplied contexts are in answer-validation.json.

| Case | Before | Current observation | Remaining limitation |
|---|---|---|---|
| Indoor plants | Free-form summary of three sources | Exact, attributed quotation about ferns, ivies and pothos | Selects only one useful source; not an exhaustive plant guide |
| Day hike | Truncated response | Complete source quotation retains first-aid advice and the explicit camping condition | Related evidence, not a complete day-hike packing list |
| Homemade bread | Personal experience presented as advice | First-person quotation, without inventing the qualifier “homemade” or a preservation guarantee | Source does not specify homemade bread; excerpt cannot fully answer the specific question |
| Mars weather | Refusal with unnecessary interpretation of unrelated passages | Concise insufficient-evidence response | Abstention quality is not established beyond this example |
| No lexical matches | No-match response | No-match response; no model request | Only a bounded empty-result test |

The final five-case run had no output-check warnings. Every displayed quotation matched the cited supplied excerpt. These checks establish provenance, not the correctness of source content. The sentence-expansion heuristic uses punctuation; it is not a semantic or grammatical verifier. Adjacent sentences outside the selected span and evidence outside the 550-byte excerpt can still matter.

Six deterministic test methods cover fabricated quotations, source IDs, duplicate citations, partial selections that omit a negation, truncated output followed by a successful retry, bounded failure, abstention, and rejection of exposed reasoning. All seven live retrieval methods also matched their saved top-ten passage IDs on one existing development query. No retrieval scoring change was made.

The current answer mode deliberately uses extractive evidence. A future free-form synthesis stage needs a separate faithfulness evaluation and stronger support checking before it can replace this mode. The next retrieval experiments should address ranking and context selection on development data; no test labels were used for the new diagnosis.
