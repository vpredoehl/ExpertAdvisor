---
title: "Phase 23A3 Layout-7 Inference Worker Publication, Registry Cutover, and Controlled Production Recovery"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase23A3_Layout7InferenceWorkerPublicationRegistryCutoverAndControlledRecovery_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 23A3 Layout-7 Inference Worker Publication, Registry Cutover, and Controlled Production Recovery

**GO WITH PREREQUISITES** — publication and registry cutover succeeded; controlled production recovery was correctly not dispatched.

- `git status --short`: clean
- `git log -1 --oneline`: `30c891f Document Phase 23A3 layout-7 worker cutover`
- Amended Phase 23A2 baseline: `ef51c80b4543d8c1b6a2b0bf7af404b12f44b838`
- Phase 23A3 documentation commit: `30c891f`

The amended baseline and original `3b7134a…` are sibling commits with the same parent. Tree comparison proved the delta is documentation-only: one report relocation and one Markdown continuation file; no runtime/build/test/source changes.

Validated Release worker:

- SHA: `35bdf9dab45fa636efea379f1b7a42a7ca61f5c0595f7cac2aabd0b2f6ec706b`
- A/B regression: passed four paths with exact durable digest `60f42065747e58843ffc11763fd5c7da7c1ce53820c5735b7344f132bc6ef1f8`

Registry cutover:

- Before: registry `23f8109d…d96d92`; old artifact [lstm-infer-worker](/Volumes/Developer%20SSD/ExpertAdvisor/Builds/SemanticWorkers/layout7/infer/bdb4d905b5badedfcaa3706bed9a00ec03762800/11f822ee12497781eca6fc942ddb422224cd9f5ed6e41291d0cf2a79dd778941/lstm-infer-worker), SHA `11f822ee…78941`
- After: registry `78ac5c4e…745c69`; new artifact [lstm-infer-worker](/Volumes/Developer%20SSD/ExpertAdvisor/Builds/SemanticWorkers/layout7/infer/ef51c80b4543d8c1b6a2b0bf7af404b12f44b838/35bdf9dab45fa636efea379f1b7a42a7ca61f5c0595f7cac2aabd0b2f6ec706b/lstm-infer-worker), SHA `35bdf9…c706b`
- Independent resolver validation against the operational registry passed. The old artifact remains byte-identical; only layout-7/infer changed.

Recovery boundary:

- 648/649 remain `failed:infer`, models 1914/1916, no new attempts/results/profitability rows.
- 1129/1130 remain failed signal-5 historical attempts.
- 650/651 remain active train attempts 1131/1132.
- Live scheduler remains single-instance, `train=2 infer=0 analyze=0`, with the old startup-time registry snapshot.

The supported requeue commands were dry-run successfully but not applied. Recovery requires an approved, fenced scheduler handoff/reload that proves the new worker SHA and admits only 648/649, returning inference capacity to zero immediately afterward. Do not enable normal production inference/analyze yet.

Changed files:

- [Phase 23A3 report](/Volumes/Developer%20SSD/ExpertAdvisor/docs/Phase23/LSTM_Phase23A3_Layout7InferenceWorkerPublicationRegistryCutoverAndControlledRecovery.md)
- [Run output](/Volumes/Developer%20SSD/ExpertAdvisor/docs/Phase23/LSTM_Phase23A3_Layout7InferenceWorkerPublicationRegistryCutoverAndControlledRecovery_Output.md)

Release builds and all required focused regressions passed. `git diff --stat` is empty after commit.