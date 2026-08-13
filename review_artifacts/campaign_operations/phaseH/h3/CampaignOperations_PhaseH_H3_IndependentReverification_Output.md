---
title: "Campaign Operations Phase H H3 Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H3_IndependentReverification_Continuation_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# Campaign Operations Phase H H3 Independent Reverification

# H3 Independent Reverification Report

## 1. Executive verdict

H3 is not ready. One independently reproduced HIGH correctness/replay defect remains in Manager source-evidence recovery.

The prior interruption did not affect the assessment. Preserved evidence was recovered from the transcript and `output.txt`, and the current tree was inspected and exercised.

## 2. Normative scope reviewed

Reviewed against the original prompt and its authoritative architecture, ADRs, H1/H2 baseline, H3 reports, current implementation, migration 058, CLI, tests, documentation, status, and diff.

The previously missing independent-output Markdown file was not present; `output.txt` contained the more complete interrupted session.

## 3. H3 architecture mapping

The implementation correctly provides:

- Bounded run-once with maximum 100.
- One optimistic candidate snapshot.
- Sequential request processing.
- Deterministic request/version operation identities.
- Common Phase E dispatch-engine reuse.
- Local-failure continuation and global-stop behavior.
- No daemon, polling, Manager sleep loop, autostart, supervision, worker control, or other H4 authority.

## 4. Candidate snapshot

PASS.

[SelectDispatchCandidatesForManager](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchRepository.cpp:219>) uses one `REPEATABLE READ, READ ONLY` transaction, orders by `operational_request_id`, applies the existing future-actions predicate, limits by the reviewed bound, and commits before processing. No sequence, advisory lock, tuple lock, `FOR UPDATE`, `SKIP LOCKED`, or durable batch identity is used.

## 5. Deterministic identity

PASS.

[BuildManagerRequestOperationIdentity](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsManager.cpp:18>) implements the frozen field order, UTF-8 octet framing, FNV-1a-64 representation, positive version, and exact `mgr-v1:<hash>:<version>` key.

Independent current-tree vector result:

```text
H3_SOURCE=campaign_operations_manager_request_operation_v1;request_identity_canonical=10:request-µ;expected_request_version=7
H3_HASH=fnv1a64:4f1c6a7f9ed711f5
H3_KEY=mgr-v1:fnv1a64:4f1c6a7f9ed711f5:7
H3_IDENTITY_VECTORS_PASS
```

## 6. Common Phase E engine

PASS with the blocking recovery defect below.

Run-once calls the Manager production adapter, which uses the same internal `RunDispatchAdapter` used by H2 and the isolated adapter. No duplicate acquisition or handoff implementation was introduced. Test fixtures remain compile-time gated.

## 7. Local continuation

PASS for covered cases.

The current disposable harness dynamically demonstrated three request-local failures, continuation through later requests, persistence of earlier successes, and absence of partial evidence:

```text
H3_A_LOCAL_CONTINUATION ... three_distinct_local_failures=PASS ...
```

## 8. Global-stop behavior

PASS for the four required classes.

Current actual Manager-path execution passed:

- Global disable.
- Scheduler-protocol ineffectiveness.
- Privilege failure.
- Database-wide failure.

Scheduler diagnostics are evaluated before generic ineffective-enablement diagnostics in [GlobalReason](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsManagerService.cpp:24>).

## 9. Multi-Manager concurrency

PASS for the covered same-request race.

Two Managers selected the same request/version and deterministic identity. The result was one new operation and one exact replay, with direct backend and `pg_blocking_pids()` evidence.

## 10. Unrelated-request concurrency

PASS.

The harness demonstrated that request Y completed while request X was blocked, without a Manager-global mutex or process-wide serialization primitive.

## 11. Migration 058 persistence/invariants

FAIL — HIGH finding H3-RV-001.

Migration 058 validates a source row when one is inserted, but has no reverse/deferred COMMIT-time guard requiring Manager evidence for a Manager-shaped Attempt V2. See [migration 058](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/058_campaign_operations_h3_manager_run_once.sql:65>).

Consequently:

- A Manager-shaped Attempt V2 can commit with zero Manager source rows.
- [FindRecoverableProductionDispatchLease](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchRepository.cpp:721>) returns that lease without checking Manager source evidence.
- [LookupRecoverableLease](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:193>) receives no expected Manager source canonical.
- The handoff path proceeds after validating only Attempt/lease equality.
- Source evidence is checked only after a complete binding already exists in [LookupBeforeRetry](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsDispatchService.cpp:95>).

This violates the required bidirectional completeness, missing-evidence fail-closed behavior, and “Manager Attempt V2 cannot commit without required source evidence” contract.

## 12. Replay/retry/uncertain-commit

FAIL as a consequence of H3-RV-001.

Independent disposable reproduction:

1. The Manager snapshot selected request 71/version 3.
2. A caller-keyed H2 production dispatch used the deterministic Manager key and actor.
3. Process interruption was injected immediately after acquisition commit.
4. Durable state was `Attempt V2=1, Manager source=0, binding=0`.
5. The actual Manager run-once path recovered that lease and committed the binding without source evidence.

Result:

```text
H3_MISSING_MANAGER_SOURCE_RECOVERY_REPRO PASS
pre=1|0|0
post=1|0|1
manager_outcome=dispatch_result
replay=new_operation
```

A later Manager replay then encounters the complete-binding source check and rejects the operation because the source row is missing. Thus the initial recovery reports success, but its own subsequent deterministic replay fails closed.

Existing H2 retry limits, SQLSTATE selection, fresh-connection uncertain-commit handling, and complete-binding full-canonical comparison otherwise remain intact.

## 13. Privilege/deployment

No separate HIGH/MEDIUM privilege defect was found.

Migration 058 is additive, grants no LOGIN or scheduler authority, and does not extend the Manager’s accepted role tuple. PUBLIC and unrelated dispatcher access are revoked. The finding concerns persistence completeness, not privilege escalation.

## 14. Harness quality

The A–J harness is isolated, deterministic, uses explicit barriers and PostgreSQL blocking evidence, and passed on the current tree.

However, scenario H tests tampering and missing evidence only after normal complete Manager commits. It does not cover a Manager-shaped caller-keyed Attempt V2 interrupted between acquisition and handoff. Therefore its PASS marker does not establish the required reverse completeness or recoverable-lease guarantee.

## 15. Regression/build evidence

| Verification | Result |
|---|---|
| H3 disposable runtime/concurrency A–J | PASS |
| H3 identity vectors | PASS |
| H3 CLI parser | PASS |
| H3 structural contract tests | PASS |
| H2 workflow baseline used by H3 harness | PASS |
| Targeted missing-source recovery reproduction | DEFECT REPRODUCED |
| `git diff --check` | PASS |
| Release build | Prior completion evidence: `BUILD SUCCEEDED`; not rerun against the shared DerivedData executable because an active scheduler and two training workers were present |

Current H3 sources were compiled repeatedly with `-Wall -Wextra -Werror` by the disposable verification.

## 16. Worktree/scope

No repository files were modified by this reverification. There are no staged changes.

Current `git status --short`:

```text
 M Database/README.md
 M Database/backups/LSTM_latest.dump.json
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/CampaignOperationsDispatchRepository.cpp
 M Sources/CampaignOperationsDispatchRepository.hpp
 M Sources/CampaignOperationsDispatchService.cpp
 M Sources/CampaignOperationsDispatchService.hpp
 M Sources/ExperimentScheduler.cpp
 M Tests/CampaignOperationsPhaseH1CliTests.sh
 M Tests/CampaignOperationsPhaseH1MigrationTests.sh
 M Tests/CampaignOperationsPhaseH1WorkflowLockTests.cpp
 M Tests/CampaignOperationsPhaseH2ConcurrencyTests.sh
 M Tests/CampaignOperationsPhaseH2ReplayAssociationTests.sh
 M Tests/CampaignOperationsPhaseH2WorkflowTests.sh
?? CampaignOperations_PhaseH_H3_Implementation_Output.md
?? CampaignOperations_PhaseH_H3_RuntimeConcurrencyHarness_VerificationCompletion_Output.md
?? CampaignOperations_PhaseH_H3_RuntimeConcurrency_Harness_Output.md
?? Database/migrations/058_campaign_operations_h3_manager_run_once.sql
?? Sources/CampaignOperationsManager.cpp
?? Sources/CampaignOperationsManager.hpp
?? Sources/CampaignOperationsManagerService.cpp
?? Sources/CampaignOperationsManagerService.hpp
?? Tests/CampaignOperationsPhaseH3CliTests.sh
?? Tests/CampaignOperationsPhaseH3ContractTests.sh
?? Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.cpp
?? Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.sh
?? Tests/CampaignOperationsPhaseH3Tests.cpp
?? docs/CampaignOperationsPhaseH3.rst
```

Current `git diff --stat`:

```text
14 files changed, 365 insertions(+), 17 deletions(-)
```

Untracked H3 files are not represented in that statistic. Migration 058’s current SHA-256 is `0c6f6401992575b170a5066d3e1bf2285265cea060d98122b1170d9875b43920`.

## 17. Documentation consistency

[CampaignOperationsPhaseH3.rst](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignOperationsPhaseH3.rst:38>) states that complete source canonical evidence is stored and byte-compared on replay. That statement is false for the reproduced recoverable-lease path and must be reconciled after correction.

The H4 exclusions remain accurate.

## 18. Findings

| Severity | ID | Finding |
|---|---|---|
| HIGH | H3-RV-001 | Manager run-once can recover and bind a Manager-shaped Attempt V2 lacking mandatory Manager source evidence; later exact replay fails |
| INFORMATIONAL | — | A disposable PostgreSQL process left by the interrupted run remains under `/tmp/ea-h1-pg.79WDDQ`; it is isolated from production and did not affect the result |

No BLOCKER or separate MEDIUM finding was identified.

## 19. Minimum correction scope

Without adding any H4 capability:

1. Add PostgreSQL COMMIT-time bidirectional completeness requiring exactly one exact source-evidence row for every Manager-identity Attempt V2.
2. Reserve/reject the Manager operation-key namespace on caller-keyed H2 acquisition unless the complete Manager source evidence participates in the same transaction.
3. Pass and full-compare Manager source identity on every recoverable-lease and conflicting-replay path before handoff; never backfill or accept an existing caller-keyed attempt.
4. Add a disposable regression matching the reproduced acquisition-commit/process-interruption/run-once sequence.
5. Update the H3 documentation and harness claims.

## 20. Final disposition

H3_INDEPENDENT_REVERIFICATION_FAILED