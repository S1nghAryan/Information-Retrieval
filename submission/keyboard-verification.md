# Keyboard verification — 20 October 2026

Scope: desktop in-app browser, current live local app. This is a bounded keyboard usability check, not a screen-reader or cross-device accessibility certification.

- Tab reached brand, New question, chat/evaluation navigation, three suggested questions, evaluation link and question input in order.
- Entered “Which plants grow well indoors?”, selected TF-IDF using Home, tabbed to submit and activated it with Enter. A cited Qwen answer appeared. Cold index loading was 54.7 seconds; generation was 13.9 seconds in that observed run. Later warm rendering showed lower timings; neither is a controlled benchmark.
- Enter on citation [1] opened source-1 (passage 19493).
- Space and Enter toggled its native summary. Focus had a visible red outline (computed 2.4 px at the current scale).
- Tab traversed the remaining source summaries through source 10 and then reached the question field. The exact-excerpts disclosure also opened with Enter and displayed visible focus.
- Tab traversed development/test controls, CSV, comparison selector, PR selector, six heatmap metric headings, difference selector and scrollable results region, then returned to the page start. No focus trap was observed.
- Keyboard selected held-out test, MAP@100 and a PR curve (historical check before the method reduction); displayed selected state and chart values changed accordingly.
- Some browser keyboard interactions changed views unexpectedly during testing. Final source toggling and forward navigation were repeated using locator-targeted keyboard events and verified from the DOM and screenshot.

Evidence: keyboard-focus-proof.png. No application code change was needed for that keyboard check. Method choices were subsequently reduced to four; see VERIFICATION.md for the current scope. Print review, final assignment presentation and institution-specific details remain separate checklist items.
