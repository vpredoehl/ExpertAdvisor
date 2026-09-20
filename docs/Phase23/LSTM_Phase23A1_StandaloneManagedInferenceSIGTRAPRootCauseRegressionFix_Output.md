---
title: "Phase 23A1 Standalone Managed Inference SIGTRAP Root-Cause and Regression Fix — Continuation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_Phase23A1_StandaloneManagedInferenceSIGTRAPRootCauseRegressionFix_Continuation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Phase 23A1 Standalone Managed Inference SIGTRAP Root-Cause and Regression Fix — Continuation

NO-GO — commit `c790c86 Fix standalone managed inference optional symbol fallback`.

Proven root cause: scheduler-managed inference omits `--symbol`, while models 1914/1916 persist `train_symbol_meta`. `value_or(*request.requestedSymbol)` eagerly dereferenced the empty fallback optional under Release optimization, causing SIGTRAP before binding/persistence. The fix selects the fallback lazily.

The standalone exact-attempt reproduction now passes that boundary and fails normally on the fixture’s missing Forex economic-event history—not SIGTRAP. Full result/profitability persistence and a dynamic compatibility A/B remain unverified, so no worker publication or production recovery is authorized.

Files changed: managed inference application/runtime diagnostics, persisted-model diagnostics, Release worker composition, Phase 23A1 regression test, and [Phase 23A1 report](/Volumes/Developer SSD/ExpertAdvisor/docs/Phase23/LSTM_Phase23A1_StandaloneManagedInferenceSIGTRAPRootCauseRegressionFix_Output.md).

Validated:

- Phase 22Z1/Z3/Z4/Z5 tests, Phase 23A1 test, semantic publisher, and scheduler zero-worker limits: passed.
- Release builds: `LSTM Infer Worker`, `LSTM Release`, `LSTM Scheduler Bundle`: passed.
- Disposable databases, temporary registries, and detached worktree: cleaned up.

Production remains immutable: registry SHA-256 is unchanged; 648/649 remain failed/infer; attempts 1129/1130 remain failed with signal 5; live train-only scheduler and 650/651 were untouched.

Recovery recommendation: do not requeue 648/649. Next phase should provide a deterministic disposable Forex/economic-event fixture, complete standalone and compatibility exact-attempt A/B through result/profitability commit, then seek separate approval to publish a new immutable worker and recover with new attempts.

`git status --short`: clean.
`git diff --stat HEAD^..HEAD`: 7 files, 285 insertions, 7 deletions.