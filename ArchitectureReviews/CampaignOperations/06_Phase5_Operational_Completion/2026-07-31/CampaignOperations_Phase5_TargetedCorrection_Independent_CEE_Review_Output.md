---
title: "Campaign Operations Phase 5 Targeted Correction Independent CEE Review"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase5_TargetedCorrection_Independent_CEE_Review_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 5 Targeted Correction Independent CEE Review

# Independent Verification Result

**NOT READY FOR COMMIT**

Three HIGH-severity correctness defects remain, plus one MEDIUM mutation-gate gap. The implementation is otherwise strong in locking order, classification, immutable-row protection, role separation, and concurrent completion uniqueness.

## Executive Summary

| Severity | Finding |
|---|---|
| HIGH | Actual uncertain commits (`pqxx::in_doubt_error`) are not recovered. |
| HIGH | Normal replay does not perform the required full-canonical comparison. |
| HIGH | Migration 054 permits the completion writer to persist unrelated or false identity/hash values permanently. |
| MEDIUM | The Phase F lifecycle-cancellation mutation path lacks the claimed completion gate. |

## Findings by Review Area

### 1. Migration 054

#### HIGH — Persisted completion identity is not database-enforced

The completion table only checks that evidence and completion hashes match the textual FNV shape. The validation trigger recomputes and compares evidence canonical text, but never validates:

- Any `*_evidence_hash` against its canonical.
- `completion_identity_canonical` against the completed row.
- `completion_identity_hash` against `completion_identity_canonical`.

See [054_campaign_operations_completion_and_audit.sql:73](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:73>) and the trigger comparison ending at [line 876](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:876>).

This is reachable through granted authority: the completion writer receives column-scoped insert permission for all of those identity fields at [line 1234](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:1234>). It can supply current valid canonical evidence while persisting arbitrary shape-valid hashes and an unrelated completion identity.

The C++ repository detects the corruption when hydrating the row, but that is too late: the one-per-campaign row is immutable and blocks replacement. Therefore SQL can permanently create an unreadable authoritative completion event.

Migration ordering, FK structure, arithmetic constraints, uniqueness, immutable update/delete/truncate triggers, indexes, ownership, and transactional rollback expectations otherwise appear correct.

### 2. Completion Locking

No independent defect found in the normal locking path.

The implementation acquires both authorization domains in stable order, then budget, campaign, ordered reservations, and ordered requests. The common completion gate locks the campaign row, closing the uncommitted-completion/child-insert race. Unique campaign completion plus the campaign lock provides exactly-once persistence under ordinary concurrent execution.

### 3. Replay / Idempotency

#### HIGH — Actual uncertain commits bypass recovery

`transaction.commit()` can throw `pqxx::in_doubt_error`, which is a direct subtype of `pqxx::failure`, not `pqxx::broken_connection` or `pqxx::sql_error`. The factory overload catches only the latter two at [CampaignOperationsCompletionService.cpp:234](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.cpp:234>).

Consequently, if the connection is lost while PostgreSQL’s commit result is uncertain, the exception escapes without reconnecting or performing the canonical outcome lookup. The CLI can report failure even though completion committed.

The installed libpqxx definition confirms the separate hierarchy at [except.hxx:157](/opt/homebrew/Cellar/libpqxx@7.10.1/7.10.1/include/pqxx/except.hxx:157).

The existing test at [CampaignOperationsRepositoryTests.cpp:5180](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:5180>) throws `broken_connection` only after `commit()` has already returned. It verifies lost-response recovery, but cannot verify libpqxx’s uncertain-commit exception path.

#### HIGH — Fresh replay compares only three request fields

When a completion already exists, the normal path compares only operation key, actor, and reason at [CampaignOperationsCompletionService.cpp:107](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.cpp:107>). The reconnect lookup does the same whenever no prior in-process `attemptedEvent` exists at [line 152](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsCompletionService.cpp:152>).

Thus a fresh invocation does not compare current terminal evidence/classification with the complete stored canonical. For example, lifecycle evidence may legitimately change after completion; the same key/actor/reason is still returned as `existing_identical` without evaluating the changed payload.

This conflicts with the accepted completion contract that different evidence/classification conflicts at [CampaignOperations_Revised_Architecture_Output.md:980](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:980>) and “a changed terminal payload conflicts” at [line 1197](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:1197>).

The full event comparison is performed only when `attemptedEvent` survived an earlier attempt in the same function call.

### 4. Immutable Completion Evidence

The constructed C++ event captures exact ordered canonical evidence for authorization heads/chains, budget, reservations and events, requests and dispatch outcomes, bindings and control owners, lifecycle, cancellation settlements, and reconciliation resolutions.

However, the HIGH database-enforcement defect above means the persisted evidence hashes and overall completion identity are not guaranteed to represent that evidence. Immutability then preserves the corruption rather than protecting a verified identity.

### 5. Privilege Model

No separate role-escalation defect found.

Verified statically:

- Completion writer is `NOLOGIN`.
- `pqxx` membership is revoked and rejected.
- `PUBLIC`, unrelated roles, and readers lose mutation rights.
- Completion writer lacks update/delete/truncate, lifecycle mutation, settlement, resolution, scheduler, and recommendation mutation authority.
- Tables, functions, sequences, and view are assigned to the intended NOLOGIN owner.
- Security-definer search paths are pinned.

The HIGH identity-enforcement defect remains security-relevant because the permitted completion insert itself can persist malformed identity data.

### 6. Mutation Gates

#### MEDIUM — Phase F lifecycle-cancellation path is omitted

Migration 054’s gate list at [054_campaign_operations_completion_and_audit.sql:1094](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/054_campaign_operations_completion_and_audit.sql:1094>) does not include `experiment_lifecycle_cancellation_event`.

The Phase F security-definer function can update `experiment` and append this event without checking completion at [053_campaign_operations_controls_cancellation_reconciliation.sql:1437](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql:1437>). Its replay branch also returns an existing event before comparing the supplied canonical at [line 1471](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/053_campaign_operations_controls_cancellation_reconciliation.sql:1471>).

In a fully consistent settled cancellation, uniqueness normally makes a post-completion call a no-op, limiting practical impact. Nevertheless, this is a Phase F authoritative mutation path without the independently claimed Phase G terminal gate.

The migration test repeats the same incomplete table list at [CampaignOperationsPhase5MigrationTests.sql:230](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhase5MigrationTests.sql:230>), so it passes despite the omission.

### 7. Classification

No defect found.

The SQL and C++ precedence agree for:

- Unbound permanent request failure.
- Failed plus completed/cancelled scope.
- All downstream failed.
- Completed plus cancelled scope.
- All cancelled/never dispatched.
- All downstream completed.

Contradictory cardinality, nonterminal lifecycle, ambiguous dispatch evidence, held reservations, unsettled cancellation, unresolved reconciliation, and impossible request/reservation combinations block before classification.

### 8. Concurrency

No additional race or deadlock defect was established in normal workflows.

Campaign-first terminal gates serialize completion against reservation/request transitions, dispatch evidence, binding, cancellation settlement, and reconciliation. Serializable isolation protects multi-table evidence against child-row write skew. Concurrent identical completions converge through the campaign lock and unique campaign constraint.

The unresolved `in_doubt_error` path remains the critical commit-boundary concurrency/recovery defect.

## Testing Review

The tests meaningfully cover ordinary completion, blockers, classification, concurrent duplicate completion, cancellation/reconciliation races, immutable-row triggers, and simulated lost responses.

Material gaps:

- No `pqxx::in_doubt_error` test.
- The lost-response test throws only after a successful commit.
- No fresh replay after post-completion evidence changes.
- No valid direct completion-writer insert with mismatched evidence hashes or completion identity.
- The migration mutation-gate matrix omits the same lifecycle-cancellation table as the migration.
- Later lifecycle change is tested only for status display at [ExperimentRecommendationCampaignLaunchRepositoryTests.cpp:3925](</Volumes/Developer SSD/ExpertAdvisor/Tests/ExperimentRecommendationCampaignLaunchRepositoryTests.cpp:3925>), not subsequent replay classification.

## Overall Assessment

The core completion decision, classification, lock ordering, immutable row protection, and least-privilege structure are substantially correct. Commit readiness fails because uncertain commits can escape recovery and the database permits a permanently malformed completion identity. Canonical replay is also incomplete relative to the accepted contract.

## Validation Performed

No files were changed.

Passed:

```text
Strict CampaignOperations domain/classification compile and execution
bash -n Tests/CampaignOperationsPhase5CliTests.sh
plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj
git diff --check
```

Repository integration, migration execution, Release build, and process suites were not run because a live scheduler and 12 training/inference workers were active.

## Worktree State

`git status --short` reports:

- 38 tracked changed/added paths.
- 16 untracked paths.
- No staged changes.
- Phase 5 migration, completion sources, tests, and documentation remain untracked.

`git diff --stat`:

```text
38 files changed, 11330 insertions(+), 187 deletions(-)
```

Final decision: **NOT READY FOR COMMIT**.