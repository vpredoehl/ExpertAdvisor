---
title: "Campaign Operations Phase 5 Architectural Phase G Targeted Correction and Verification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase5_ArchitecturalPhaseG_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 5 Architectural Phase G Targeted Correction and Verification

## Findings

### BLOCKER

- None remaining in the targeted Phase G scope.

### HIGH

- Resolved: completion canonicals previously omitted exact authorization, reservation-event, dispatch-outcome, owner, cancellation-settlement, and reconciliation-resolution evidence.
- Resolved: completion lacked reconnect/canonical-lookup recovery for broken connections and uncertain commits.
- Resolved: post-completion database gates did not cover every Phase B–F mutation table or serialize correctly with completion.

### MEDIUM

- Resolved: classification precedence and contradiction coverage were incomplete.
- Resolved: reconciliation observation acquired a request lock without first acquiring the campaign lock.
- Resolved: completion locked only one authorization domain instead of both applicable domains.

### LOW

- No open Phase G defect identified.

### NONBLOCKING_FOLLOWUP

- Process-level scheduler/global-control integration suites were not run because four real training workers were active.
- The Release build succeeded but emitted existing libpqxx `exec_params` deprecation warnings, including 370 from `ExperimentScheduler.cpp`. Phase G-focused strict builds were warning-free.
- A separate independent verification is still required. This report does not declare commit readiness.

## 1. Decision and exact scope

The eight requested correction areas are complete and ready for independent review.

The work stayed within migration 054, completion locking/replay/evidence, privileges, Phase G terminal gates, classification, concurrency tests, and the narrow Phase F reconciliation lock-order correction. No Phase H enablement, scheduler behavior, lifecycle mutation, scientific policy, refunds, automatic polling, backups, or production data were changed.

## 2. Files changed in this correction

Materially edited during this correction run:

- [054_campaign_operations_completion_and_audit.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql>)
- [CampaignOperationsCompletion.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletion.cpp>)
- [CampaignOperationsCompletion.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletion.hpp>)
- [CampaignOperationsCompletionRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionRepository.cpp>)
- [CampaignOperationsCompletionService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.cpp>)
- [CampaignOperationsCompletionService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.hpp>)
- [CampaignOperationsControlRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsControlRepository.cpp>)
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [CampaignOperationsTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsTests.cpp>)
- [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp>)
- [CampaignOperationsPhase5MigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase5MigrationTests.sql>)
- [CampaignOperationsPhase5.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignOperationsPhase5.rst>)
- [CampaignOperations_Phase5_Operational_Completion_Audit.md](</Volumes/Developer SSD/ExpertAdvisor/docs/architecture/CampaignOperations_Phase5_Operational_Completion_Audit.md>)

All unrelated pre-existing Phase F/G worktree changes were preserved.

## 3. Migration 054 corrections

Migration 054 now provides:

- Additive application and direct replay after 053.
- One completion event per campaign.
- One exact, same-transaction audit reference, enforced by a deferred constraint trigger.
- Restrictive foreign keys and contract-version, enum, hash, canonical-size, arithmetic, and evidence checks.
- Owner-enforced update, delete, and truncate rejection for completion and audit history.
- Logical archival only through `campaign_operations_completion_status_v1`.
- Nineteen `BEFORE INSERT` completion gates and two guarded projection `BEFORE UPDATE` gates.
- Campaign-row locking inside the common completion gate, closing the uncommitted-completion race.
- Pinned function search paths.
- `campaign_operations_completion_writer NOLOGIN`.
- Explicit revocation from `pqxx`, `PUBLIC`, readers, and unrelated Campaign Operations roles.
- Column-scoped completion/audit inserts, required source reads, sequence usage, and only the required lock/evidence functions.
- No production runtime capability assignment.

Migration 050 initially failed when attempted without an explicit transaction because it contains `LOCK TABLE`. It was rerun correctly with `psql -1`; migrations 050–054, direct 054 replay, and catalog tests then passed.

## 4. Locking and transaction proof

The completion transaction now acquires:

```text
authorization:
  adopt_existing_pending_and_control
  dispatch_full_materialization
→ budget
→ campaign/completion
→ reservations ordered by reservation_id
→ requests ordered by operational_request_id
```

Both authorization domains use stable sorted acquisition, followed by the budget head and campaign row. Reservation and request rows are locked ascending. Evidence is loaded only after the complete lock set is held.

`SERIALIZABLE` remains supplemental. Every `40001` or `40P01` retry starts a new connection transaction, reacquires all locks, and rebuilds the canonical evidence.

`InsertReconciliationObservation` now locks the campaign before its request candidate, removing its previous campaign/request inversion. Database completion gates also lock the campaign row before checking completion.

## 5. Replay, idempotency, and uncertain commits

The service now supports a connection factory and bounded reconnection:

- Exact replay returns `existing_identical`.
- Changed operation key, actor, reason, or evidence returns `conflicting_replay` without mutation.
- Concurrent identical attempts converge on one event row.
- `23505` uniqueness races reload and compare the full event and canonical text.
- Hash equality never establishes identity.
- `40001` and `40P01` retry the complete transaction.
- Pre-append connection loss reconnects, proves absence, then retries boundedly.
- Post-commit lost response reconnects and performs canonical lookup before any append retry.
- A matching stored canonical returns the existing event.
- A changed stored canonical conflicts.
- An outcome that cannot be proven present or absent fails closed as ambiguous.

The repository suite deterministically injected both a pre-append `pqxx::broken_connection` and a post-commit lost response in the same completion request; the final result was one row and `existing_identical`.

## 6. Immutable evidence and canonical proof

The completion identity binds exact ordered evidence for:

- Campaign canonical.
- Both authorization chains, including event IDs, versions, kinds, scopes, effective times, and complete canonicals.
- Budget head and arithmetic.
- Reservations and every ordered reservation event.
- Requests, dispatch attempts, and exact attempt outcomes.
- Bindings and downstream control owners.
- Cancellation requests and settlements.
- Reconciliation observations and resolutions.
- Per-member lifecycle status, phase, and timestamp evidence.
- Administrative terminal state and classification.
- Actor, completion-writer capability, and reason.

Golden-vector, reconstruction, changed-evidence, canonical hydration, and hash-collision tests passed. A supplied matching hash with different canonical text is rejected.

## 7. Privilege proof

Catalog and runtime tests established:

- Completion writer is `NOLOGIN`.
- `pqxx` is not a member and cannot assume it.
- Completion writer has no update, delete, or truncate right on completion history.
- It has no mutation privilege over authorization, budget, reservation, request, dispatch, cancellation, reconciliation, lifecycle, scheduler, process, or recommendation authority.
- Completion insert access is column-scoped.
- Lifecycle access is restricted to `experiment_id`, `status`, `phase`, and `updated_at`.
- Readers/auditors receive only approved completion history/status reads and evidence evaluators.
- `PUBLIC`, `pqxx`, readers, and unrelated roles cannot execute privileged validation or gate functions.
- Unrelated roles cannot forge completion audit references.
- Owner-level attempts to update or truncate immutable history are rejected by triggers.

## 8. Post-completion mutation-gate matrix

| Mutation path | After completion |
|---|---|
| Governance provenance append | New/changed fact rejected; existing service replay remains idempotent |
| Authorization grant/revoke/expiry/supersession | Rejected |
| Budget grant/amend/revoke/supersession | Rejected |
| Reservation creation | Rejected |
| Reservation events, release, expiry, permanent-failure settlement | Rejected |
| Reservation commitment | Rejected |
| Operational-request creation | Rejected |
| Guarded reservation/request projection update | Rejected |
| Dispatch lease acquisition | Rejected |
| Dispatch-attempt insertion | Rejected |
| Dispatch-attempt outcome | Rejected |
| Binding insertion | Rejected |
| Downstream control-owner insertion | Rejected |
| Pause/resume | Rejected |
| Cancellation request | Rejected |
| Cancellation settlement | Rejected |
| Reconciliation observation | Rejected |
| Reconciliation resolution | Rejected |
| Owning-service reconciliation recovery | Rejected |
| Restart/replay | Identical committed fact returns existing; changed replay conflicts |
| Completion replay | Exact returns existing event; changed canonical conflicts |
| Read-only completion/status | Allowed |
| Ordinary lifecycle retry/requeue | Allowed outside Campaign Operations; display-only state may change |

The catalog matrix verifies the common gate on every direct and indirectly keyed mutation table. Runtime tests verify representative direct and indirect inserts, guarded updates, service replay, and deterministic post-completion contention.

## 9. Classification truth table

| Evidence | Result |
|---|---|
| Unsettled obligations | Block before classification |
| Unbound permanently failed request | `operational_request_failed` |
| Permanently failed request plus binding | Invalid contradictory evidence |
| Failed + completed | `mixed_terminal_outcomes` |
| Failed + cancelled | `mixed_terminal_outcomes` |
| Failed + never dispatched | `mixed_terminal_outcomes` |
| All failed | `downstream_failure` |
| Completed + cancelled | `terminal_partial_completion` |
| Completed + never dispatched | `terminal_partial_completion` |
| All cancelled | `all_scope_cancelled` |
| All never dispatched | `all_scope_cancelled` |
| All completed and bound | `all_downstream_completed` |
| Duplicate binding | Block |
| Duplicate/conflicting owner or lifecycle evidence | Block |
| Inconsistent budget arithmetic | Block |
| Unresolved cancellation | Block |
| Unresolved reconciliation | Block |

The classifier remains independent of scientific success and downstream lifecycle policy.

## 10. Concurrency-race matrix

| Race | Deterministic result |
|---|---|
| Identical completion vs identical completion | One `recorded`, one `existing_identical`, one row |
| Conflicting completion vs completion | One immutable winner; loser reloads full canonical and conflicts |
| Authorization transition vs completion | Serialized by authorization-domain locks and campaign gate |
| Budget mutation vs completion | Serialized by budget-head lock and campaign gate |
| Reservation release/expiry/commit vs completion | Campaign-first serialization; loser blocked or completion rebuilds evidence |
| Request terminalization vs completion | Campaign/request order; no partial canonical |
| Dispatch lease acquisition vs completion | Campaign gate serializes; post-completion acquisition rejected |
| Dispatch outcome vs completion | Same campaign gate; immutable outcome is either included or rejected |
| Binding/control-owner insertion vs completion | Included before completion or rejected after completion |
| Cancellation request vs completion | Included/blocking or rejected after winner |
| Cancellation settlement vs completion | Direct independent-connection race passed |
| Reconciliation observation vs completion | Direct independent-connection race passed |
| Reconciliation resolution/recovery vs completion | Direct reconciliation race passed; completion blocked until resolved |
| Lifecycle retry/requeue vs completion | Allowed outside Campaign Operations; only current display changes |
| Four simultaneous Phase F mutations vs held completion locks | All blocked until release, then all rejected with `23514`; one completion row |

No duplicate completion, partial truth, or observed deadlock occurred.

## 11. Scope and scheduler/lifecycle isolation

Before binaries or integration tests:

- Four active unmanaged `LSTM_Release --train` workers were found: PIDs 38330, 38400, 39259, and 39540.
- Read-only scheduler status reported authority vacant, scheduler absent, protocol generation 52 pending.
- The live DerivedData binary was not overwritten.
- The Release build used `DerivedData/ExpertAdvisor-PhaseGCorrections`.
- No scheduler, worker, training, inference, analysis, or process-level integration command was launched.
- No production rows or backup files were modified.
- Backup status remained clean; observed hashes were:
  - `LSTM_latest.dump`: `ebcfa55680f4db5fb136527ceff1039f97a139fcdcbd4574186ef341655fac31`
  - `LSTM_latest.dump.json`: `80bd1559f4f25f2502f39690e7a11136aaa9fbe3a9772dabde4a3618461ba87f`

All disposable databases were removed after verification.

## 12. Tests and build commands

Passed:

- Strict Phase G domain/classification compile and execution:

```text
xcrun clang++ -std=c++20 -O1 -Wall -Wextra -Wpedantic -Werror \
  -Wshorten-64-to-32 -Wno-c++23-attribute-extensions \
  -I Sources -I Headers \
  Tests/CampaignOperationsTests.cpp \
  Sources/CampaignOperations.cpp \
  Sources/CampaignOperationsDispatch.cpp \
  Sources/CampaignOperationsControl.cpp \
  Sources/CampaignOperationsCompletion.cpp \
  Sources/ExperimentRecommendation.cpp \
  -o /tmp/CampaignOperationsTests-phaseg-correction

/tmp/CampaignOperationsTests-phaseg-correction
```

- Strict Phase G repository compile and execution:

```text
xcrun clang++ -std=c++20 -O0 -g -Wall -Wextra -Wpedantic -Werror \
  -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
  -I Sources -I Headers $(pkg-config --cflags libpqxx) \
  Tests/CampaignOperationsRepositoryTests.cpp \
  Sources/CampaignOperations.cpp \
  Sources/CampaignOperationsRepository.cpp \
  Sources/CampaignOperationsService.cpp \
  Sources/CampaignOperationsDispatch.cpp \
  Sources/CampaignOperationsDispatchRepository.cpp \
  Sources/CampaignOperationsControl.cpp \
  Sources/CampaignOperationsControlRepository.cpp \
  Sources/CampaignOperationsControlService.cpp \
  Sources/CampaignOperationsCompletion.cpp \
  Sources/CampaignOperationsCompletionRepository.cpp \
  Sources/CampaignOperationsCompletionService.cpp \
  Sources/ExperimentRecommendation.cpp \
  Sources/ExperimentRecommendationConversionWorkflow.cpp \
  Sources/ExperimentRecommendationConversionWorkflowRepository.cpp \
  $(pkg-config --libs libpqxx) -pthread \
  -o /tmp/CampaignOperationsRepositoryTests-phaseg-correction
```

Final repository run:

```text
LSTM_TEST_DB_NAME=phaseg_correction_test_20260731_27925 \
  /tmp/CampaignOperationsRepositoryTests-phaseg-correction
```

Result: exit 0; database removed.

- Restored-schema migrations 050→054 under `psql -1`: passed.
- Direct migration 054 replay: passed.
- `Tests/CampaignOperationsPhase5MigrationTests.sql`: passed.
- Catalog/ACL/trigger checks: passed; 21 mutation-gate triggers found.
- Phase F control, dispatch, cancellation, settlement, and reconciliation regressions within the repository suite: passed.
- Lost-response, pre-append broken connection, reconnect, replay, conflict, and race tests: passed.
- Recommendation campaign launch pure tests: passed.
- Recommendation outcome-policy tests: passed after adding their complete source dependency set.
- `SchedulerCanonicalPathTests.sh`: passed.
- `SchedulerChildStatusTests`: passed.
- `ExperimentCurrentOperationTests`: passed.
- `GlobalExperimentControlTests`: passed with third-party deprecation diagnostics suppressed.
- Phase 2, 4, and 5 CLI parser suites: passed.
- `bash -n` on three Campaign Operations shell suites: passed.
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`: passed.
- `git diff --check`: passed.
- `git diff --cached --check`: passed.
- Untracked-file trailing-whitespace scan: passed.

Release build:

```text
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor-PhaseGCorrections \
  -jobs 1 build
```

Result: `** BUILD SUCCEEDED **`.

Not run:

- Process-level scheduler ownership/global-control integration suites, due to the four active training workers.
- `ExperimentRecommendationCampaignLaunchRepositoryTests` was not rerun separately; Campaign Operations repository integration plus pure launch and outcome-policy suites covered the touched completion/lifecycle boundary.

Diagnostic attempts subsequently corrected:

- Migration 050 without `psql -1`: rejected because `LOCK TABLE` requires a transaction.
- Two outcome-policy link attempts omitted transitive sources; the complete-source command passed.
- Initial global-control compile omitted libpqxx flags, then exposed existing deprecation warnings under `-Werror`; the isolation unit passed using the project’s third-party-warning suppression.
- First extension of the pre-commit fault test asserted hook invocations rather than injected faults; corrected test passed.

## 13. Residual risks

- Independent verification remains mandatory.
- Process-level scheduler suites remain unverified in this run for worker safety.
- Existing non-Phase-G libpqxx deprecation warnings remain; correcting them would be unrelated refactoring outside this task.
- Broken connections were tested through deterministic `pqxx::broken_connection` injection rather than physically terminating a PostgreSQL TCP session.

## 14. Exact `git status --short`

```text
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_Phase4_ArchitecturalPhaseF_Implementation_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_Phase4_ArchitecturalPhaseF_IndependentReview_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-25/CampaignOperations_ThreeFixWorktree_DualRemote_Integration_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalFocusedCorrection2_Implementation_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalFocusedCorrection2_IndependentCEEVerification_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalFocusedCorrection_Implementation_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_Phase4_FinalIndependentCEEVerification_Output.md
 A ArchitectureReviews/CampaignOperations/05_Phase4_Implementation/2026-07-26/CampaignOperations_ThreeFixWorktree_DualRemote_Integration_Output.md
 M Database/README.md
 A Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/CampaignOperations.cpp
 M Sources/CampaignOperations.hpp
 A Sources/CampaignOperationsControl.cpp
 A Sources/CampaignOperationsControl.hpp
 A Sources/CampaignOperationsControlRepository.cpp
 A Sources/CampaignOperationsControlRepository.hpp
 A Sources/CampaignOperationsControlService.cpp
 A Sources/CampaignOperationsControlService.hpp
 M Sources/CampaignOperationsDispatchRepository.cpp
 M Sources/CampaignOperationsDispatchService.cpp
 M Sources/CampaignOperationsService.cpp
 M Sources/ExperimentScheduler.cpp
 M Tests/CampaignOperationsPhase2CliTests.sh
 A Tests/CampaignOperationsPhase4CliTests.sh
 A Tests/CampaignOperationsPhase4MigrationTests.sql
 M Tests/CampaignOperationsRepositoryTests.cpp
 M Tests/CampaignOperationsTests.cpp
 M Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp
 M docs/CampaignOperationsPhase3.rst
 A docs/CampaignOperationsPhase4.rst
 M docs/architecture/CampaignOperations_Phase3_Durable_Dispatch.md
 A docs/architecture/CampaignOperations_Phase4_Controls_Cancellation_Reconciliation.md
 M docs/architecture/README.md
 M docs/architecture/Volume_XII_Database.md
 M docs/architecture/Volume_X_Research_Automation.md
 M docs/architecture/adr/ADR-0015-cancellation-reconciliation-and-recovery.md
 M docs/architecture/adr/ADR-0017-campaign-privileges-and-audit.md
?? CampaignOperations_Phase5_ArchitecturalPhaseG_Implementation_Output.md
?? CampaignOperations_Phase5_ArchitecturalPhaseG_IndependentFocusedVerification_Output.md
?? CampaignOperations_PhaseF_SchedulerHardened_Integration_IndependentVerification_Output.md
?? Database/migrations/054_campaign_operations_completion_and_audit.sql
?? Sources/CampaignOperationsCompletion.cpp
?? Sources/CampaignOperationsCompletion.hpp
?? Sources/CampaignOperationsCompletionRepository.cpp
?? Sources/CampaignOperationsCompletionRepository.hpp
?? Sources/CampaignOperationsCompletionService.cpp
?? Sources/CampaignOperationsCompletionService.hpp
?? Tests/CampaignOperationsPhase5CliTests.sh
?? Tests/CampaignOperationsPhase5MigrationTests.sql
?? docs/CampaignOperationsPhase5.rst
?? docs/architecture/CampaignOperations_Phase5_Operational_Completion_Audit.md
```

## 15. Complete diff statistics

Exact tracked `git diff --stat` summary:

```text
38 files changed, 11330 insertions(+), 187 deletions(-)
```

Exact cached summary:

```text
0 files changed, 0 insertions, 0 deletions
```

Untracked files:

```text
14 files, 3921 lines
```

Combined worktree accounting, treating untracked lines as additions:

```text
52 paths, 15251 added lines, 187 deleted lines
```

The tracked stat is dominated by the pre-existing Phase F/G worktree. No files were staged, committed, reset, cleaned, or stashed.

PHASE_G_CORRECTIONS_COMPLETE_READY_FOR_INDEPENDENT_REVIEW