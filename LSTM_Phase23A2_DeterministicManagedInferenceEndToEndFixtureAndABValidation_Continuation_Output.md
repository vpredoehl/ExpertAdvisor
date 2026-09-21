---
title: "Phase 23A2 Deterministic Managed-Inference End-to-End Fixture and Compatibility/Standalone A/B Validation — Continuation"
document_type: "architecture review"
status: "final"
generated_from: "Phase23A2_Continuation.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 23A2 Deterministic Managed-Inference End-to-End Fixture and Compatibility/Standalone A/B Validation — Continuation

Phase 23A2: GO.

Committed:
- `3b7134a Validate deterministic managed inference A/B`
- Full SHA: `3b7134a9ca98496d0575d1025dc9d76674cde28d`

Implemented deterministic disposable LSTM/Forex fixture and A/B harness. It runs compatibility `LSTM_Release` and standalone `lstm-infer-worker` twice each through real scheduler-created exact attempts, detached materialization, evaluation, atomic result/profitability persistence, and orphan recovery.

Results:
- Compatibility: passed twice.
- Standalone: passed twice; no SIGTRAP.
- Durable A/B result: exact normalized semantic digest matched in all runs: `60f42065747e58843ffc11763fd5c7da7c1ce53820c5735b7344f132bc6ef1f8`.
- Atomicity: each run had exactly one completed final result and one linked profitability observation; state reached `pending:analyze`; attempt reached `completed:process_missing_result_recovered`.
- Release builds passed: LSTM Release, LSTM Infer Worker, LSTM Scheduler Bundle.
- Focused Phase 22/23, registry/publisher, exact-attempt/orphan recovery, and economic snapshot/feature regressions passed.

A/B executable SHA-256:
- `LSTM_Release`: `b98d2b134fa25e4d3cf86bc4a771b86016f21398666f1ff6e7f307d31463ebd1`
- `lstm-infer-worker`: `1a32e91fed1aa9be405a8b6103a4c135617cd5863863972cdd177aa9b94391c4`

Production safety verified:
- 648/649 remain `failed:infer`.
- Attempts 1129/1130 remain historical `failed:parent_observed_exit:-5`.
- 650/651 remain `running:train`; train-only scheduler unchanged.
- Operational registry SHA unchanged: `23f8109d…`
- Published layout-7 worker SHA unchanged: `ac5023fa…`
- No disposable Phase23A2 databases remained.

Changed files:
- [A/B harness](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/LSTMPhase23A2ManagedInferenceEndToEndABTests.sh)
- [Fixture seeder](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/Phase23A2ManagedInferenceFixtureSeeder.cpp)
- [Snapshot regression compatibility fix](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/EconomicCalendarSnapshotTests.sh)
- [Validation report](/Volumes/Developer%20SSD/ExpertAdvisor/docs/Phase23/LSTM_Phase23A2_DeterministicManagedInferenceEndToEndFixtureAndABValidation.md)
- [Review output](/Volumes/Developer%20SSD/ExpertAdvisor/LSTM_Phase23A2_DeterministicManagedInferenceEndToEndFixtureAndABValidation_Output.md)

`git status --short`: clean.

`git diff --stat d54b89b..HEAD`: 5 files, 580 insertions, 5 deletions.

Next: do not publish or recover anything yet. A separate authorized phase should publish a new immutable standalone worker, perform registry/cutover verification, then decide whether to recover 648/649 with fresh scheduler-created attempts.