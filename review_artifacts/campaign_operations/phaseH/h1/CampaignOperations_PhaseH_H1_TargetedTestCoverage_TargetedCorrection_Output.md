---
title: "Campaign Operations Phase H H1 Targeted Test Coverage Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_TargetedTestCoverage_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H H1 Targeted Test Coverage Targeted Correction

Implemented all five targeted coverage corrections. No production code changed.

Files changed:

- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sql`
- `Tests/CampaignOperationsPhaseH1RepositoryTests.cpp`

| Item | Coverage added/tightened | Exact result |
|---|---|---|
| Protected catalog mutations | Disposable live `pg_proc` mutations for language, `provariadic`, and hostile `search_path`; rollback/snapshot verified. | Language/config: `55000 H1A008`; variadic: `42501 H1A005 protected function default or variadic mismatch`. |
| First-Attempt mirrors | Corrupts first Attempt admission ID, enablement ID, and build-hash mirror, then invokes actual later Attempt replay. | `23514 production acquisition replay evidence corrupt`; Attempt #2 unchanged and corruption rolled back. |
| Readiness negative controls | Adds unrelated role edge and unrelated Completion check constraint. Relevant dependency remains included; unrelated catalog state leaves snapshot/readiness unchanged. | Exact snapshot/readiness preservation. |
| Duplicate first Attempt | Disables only disposable-fixture uniqueness protections, clones persisted Attempt #1, proves two authorities, invokes replay, then rolls back. | `23514 production acquisition replay evidence corrupt`; no silent selection or repair. |
| Readiness diagnostics | Refactored corruption helper to require `persistenceCorruption` plus one exact diagnostic per mutation. | E.g. `campaign_operations_admission_corrupt`, `campaign_operations_attempt_v2_corrupt`, `campaign_operations_enablement_missing`. |

Validation passed:

- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
  - Includes replay/recovery, repository readiness compilation/tests, lock tests, Phase 1–5 regression, ACL-origin, traceability, and reference-graph validation.
- `bash -n Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
- `git diff --check`
- `git diff --cached --check`

The full harness’s partial-final reference graph correctly reported `NOT_READY_FOR_REVERIFICATION` only because final-assurance artifacts were intentionally not generated in partial-final mode; all correction-relevant validations passed.

No evidence/generated artifacts were changed. Release Xcode build was not run because active `LSTM_Release` training and scheduler workers were detected; the H1 harness compiled the affected C++ tests directly.

`git diff --stat`: 3 files changed, 214 insertions, 42 deletions.
`git status --short`: 130 entries total; the three correction files are `AM`, with the remaining staged H1 baseline and review output pre-existing.

READY_FOR_TARGETED_TEST_COVERAGE_INDEPENDENT_REVERIFICATION