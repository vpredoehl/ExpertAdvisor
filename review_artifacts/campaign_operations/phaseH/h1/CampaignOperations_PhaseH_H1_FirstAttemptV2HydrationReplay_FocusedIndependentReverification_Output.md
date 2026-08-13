---
title: "Campaign Operations Phase H H1 First Attempt V2 Hydration and Replay Focused Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_FirstAttemptV2HydrationReplay_FocusedIndependentReverification_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H H1 First Attempt V2 Hydration and Replay Focused Independent Reverification

## 1. Executive Summary

The correction is incomplete. Repository hydration is graph-complete, but SQL replay still accepts a later Attempt V2 when an earlier enablement-chain predecessor is corrupted.

## 2. Root Defect Reviewed

The required invariant is that replay of Attempt #2 validates the complete immutable evidence rooted at Attempt #1. SQL replay does not validate the complete predecessor chain of Attempt #1’s enablement.

## 3. Repository Hydration Assessment

Pass. `LoadEnablementEventById` recursively reconstructs each predecessor, validates canonical/hash, typed mirrors, audit, kind/version continuity, and genesis conditions. `FindRequestProductionAdmission` then hydrates complete Attempt #1 evidence and its acquisition audit.

## 4. SQL Replay Assessment

Fail. Replay loads `first_enablement_predecessor` only one level deep. It checks only:

- existence / ID
- `event_kind = disable`
- resulting version
- canonical text matching the child’s predecessor mirror

It does not reconstruct or validate that predecessor’s hash, typed mirrors, scheduler/build fields where applicable, enablement audit, or its own predecessor chain.

See [migration replay validation](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3018) and its limited predecessor predicate at [line 3273](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/055_campaign_operations_production_admission_foundation.sql:3273).

## 5. Adversarial Counterexample Assessment

A remaining successful-replay path exists:

1. Enablement E1 → disablement D2 → enablement E3.
2. Admission A and Attempt #1 are created under E3.
3. Recovery occurs; Attempt #2 reuses Admission A.
4. Under the same isolated corruption mechanism used by the tests, corrupt D2’s `enablement_identity_hash` to a syntactically valid but incorrect FNV value.
5. Replay Attempt #2.

Replay succeeds: D2’s canonical text, kind, and version remain valid, which are its only replay checks. Repository hydration fails because it recursively reconstructs D2 and validates its stored hash.

The same escape applies to D2 typed-mirror corruption, D2 audit corruption, or corruption/missing evidence farther back in D2’s predecessor chain.

## 6. Regression Assessment

No regression was observed in the existing isolated suite for immutable-admission reuse, post-recovery reacquisition, cross-principal replay, protected-function preflight, role graph, default-off behavior, or scheduler generation 52.

## 7. Test Assessment

The new Attempt #2/Attempt #1 tests are substantive: they mutate persisted evidence, prove the mutation occurred, preserve Attempt #2, require SQLSTATE `23514`, and roll back each mutation.

They are nevertheless insufficient. The fixture exercises a genesis enablement only; it never creates E1 → D2 → E3 before Attempt #1. Therefore it cannot detect the replay/hydration divergence for predecessor-chain corruption.

## 8. Commands Executed

- Read `AGENTS.md`
- Inspected active scheduler/training/inference processes
- Read repository hydration, replay SQL, correction diff, and tests
- `git diff --check`
- `git status --short`
- `git diff --stat`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`

## 9. Tests Executed

`Tests/CampaignOperationsPhaseH1MigrationTests.sh` — passed using its disposable PostgreSQL cluster. It includes the new direct first-attempt corruption harness, repository hydration checks, and broader H1 regressions.

## 10. Deferred Checks

No release build was run because active scheduler workers use the shared Release product. A dedicated executable non-genesis enablement-chain corruption fixture is absent; creating one would modify the test suite, contrary to this review’s no-modification scope.

## 11. Findings

- **High:** SQL replay does not recursively validate Attempt #1’s enablement predecessor chain.
- **High:** A corrupted predecessor hash, typed mirror, audit, or deeper predecessor can permit Attempt #2 replay.
- The existing tests pass but do not cover this path.
- No files were modified during this review.
- `git diff --check` passed.
- `git status --short` remains pre-existing H1 work: the six correction files are `AM`, alongside the broader staged H1 baseline and four untracked prior-review outputs.
- `git diff --stat`: 6 correction files, 868 insertions, 121 deletions.

## 12. Final Disposition

FIRST_ATTEMPT_V2_REMAINING_DEFECTS_FOUND