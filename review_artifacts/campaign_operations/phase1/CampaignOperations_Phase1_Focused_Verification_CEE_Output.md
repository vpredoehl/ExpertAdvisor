---
title: "Campaign Operations Phase 1 Focused Verification CEE Review"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_Phase1_Focused_Verification_CEE_Prompt.txt"
reasoning_effort: "high"
model: "default"
---

# Campaign Operations Phase 1 Focused Verification CEE Review

## Executive verdict

The implementation is now ready. The review initially found:

- migration assertions that did not verify exact physical keys;
- oversized-canonical tests that shared parent fixtures and lacked repository-specific atomicity assertions;
- a real concurrency defect under `REPEATABLE READ`, where the loser leaked `pqxx::unique_violation`.

All three were corrected and the final focused PostgreSQL suite passed.

## 1. Authorization-audit provenance reference

**PASS — the change is valid and is the smallest safe correction.**

- Provenance creation already writes an audit row with `governance_provenance_event_id`: [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:645>).
- `campaign_operations_audit_provenance_uidx` permits only one non-null audit reference per provenance event: [045_campaign_operations_foundation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/045_campaign_operations_foundation.sql:356>). Reusing that ID on an authorization audit would therefore collide.
- Authorization audit insertion now passes `NULL` provenance and the new authorization ID: [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:786>).
- The cause-shape check permits this and requires `authorization_event_id`: [045_campaign_operations_foundation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/045_campaign_operations_foundation.sql:335>).
- The behavior has a direct assertion: [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:929>).
- The immutable authorization row retains the provenance FK, canonical, hash, campaign, and prerequisite policy. Hydration revalidates them: [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:327>).
- Runtime roles have no update/delete/truncate authority, and the audit remains a same-transaction reference rather than current-state authority: [045_campaign_operations_foundation.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/045_campaign_operations_foundation.sql:575>).
- The architecture requires causal fields only where applicable, explicitly requires authorization event ID, and leaves domain events authoritative: [CampaignOperations_Revised_Architecture_Output.md](</Volumes/Developer SSD/ExpertAdvisor/ArchitectureReviews/CampaignOperations/02_CEE/CampaignOperations_Revised_Architecture_Output.md:1386>).

No reader, report, or documented Phase 1 contract requires direct provenance attachment to an authorization audit row. No audit uniqueness constraint was removed.

## 2. Exact migration catalog assertions

**PASS after correction.**

The migration test now verifies:

- the three removed objects are absent by relation and constraint identity;
- every unique B-tree index—constraint-backed or standalone—is inspected through `pg_index`;
- none has the relevant canonical column as a uniqueness key: [CampaignOperationsMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsMigrationTests.sql:101>);
- all four natural constraints have the exact ordered key arrays, valid/ready backing indexes, and no expressions or predicates: [CampaignOperationsMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsMigrationTests.sql:155>);
- each canonical hash index is on the correct table and column, valid, ready, B-tree, and non-unique: [CampaignOperationsMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsMigrationTests.sql:207>);
- the active trigger function contains prerequisite-policy equality using a whitespace- and case-tolerant catalog definition assertion: [CampaignOperationsMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsMigrationTests.sql:260>).

Migration 045 was applied twice before these assertions: [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1550>). The final run passed.

## 3. Concurrent authorization loser

**PASS after fixing a confirmed defect.**

### Interleaving analysis

- Both successor objects bind the same persisted grant ID/canonical/hash and chain version 2: [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1172>).
- The advisory lock is acquired before natural-version and head reads: [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:697>).
- Under `READ COMMITTED`, the waiter sees the winner’s natural version after lock acquisition and returns `campaign_operations_authorization_conflict`.
- Under `REPEATABLE READ`, the waiter retains its earlier snapshot, misses the winner, and reaches the chain uniqueness constraint. The adversarial probe reproduced an uncaught `pqxx::unique_violation`.
- The authorization insert now translates any uniqueness loser at that insert into `ErrorCode::persistenceConflict`: [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:741>).
- Both isolation paths are now run as independent two-connection races: [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1141>) and [invocations](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1688>).

Each race verifies:

- exactly one recorded successor;
- exactly one deterministic domain conflict;
- two authorization rows total, including the initial grant;
- exactly two authorization audit rows;
- exactly one current head;
- no unexpected database exception.

Because the event and audit use the caller-owned transaction, the failed loser leaves neither a partial authorization row nor audit row.

## 4. Oversized-canonical matrix

The generator is deterministic xorshift output over a 64-character alphabet: [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:610>). Final authoritative canonicals are asserted above 8 KiB and below the 128 MiB domain limit.

| Repository | Persist | Reload | Replay | Conflict | Atomicity |
|---|---|---|---|---|---|
| Campaign | PASS | PASS | PASS | PASS | PASS |
| Provenance | PASS | PASS | PASS | PASS | PASS |
| Authorization | PASS | PASS | PASS | PASS | PASS |

Evidence:

- Campaign uses its own dedicated 16 KiB materialization payload and campaign fixture; row/audit counts remain 1 after conflict: [lines 1276–1328](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1276>).
- Provenance uses a separate short-parent campaign and its own 16 KiB governance payload embedded in ratification, review, and proposal canonicals; row/audit counts remain 1/2: [lines 1330–1407](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1330>).
- Authorization uses another dedicated 16 KiB campaign payload plus a repository-local incompressible 4096-byte reason, without relying on provenance; row/audit counts remain 1/2: [lines 1409–1503](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1409>).

The repositories no longer share one oversized authoritative fixture, so one repository’s assertions cannot satisfy another’s coverage.

## 5. Migration 045 history

**PASS for every known local real environment.**

The migrator derives identifier `045` from the filename and records it in `schema_migrations(version, filename, checksum, applied_at)`: [migrate_lstm_db.sh](</Volumes/Developer SSD/ExpertAdvisor/migrate_lstm_db.sh:19>) and [version/checksum handling](</Volumes/Developer SSD/ExpertAdvisor/migrate_lstm_db.sh:33>).

Read-only findings:

- `LSTM`: ledger exists; zero rows for version `045`; latest recorded version is `033`; zero Campaign Operations tables.
- `forex`, `results`, and `postgres`: no migration ledger and zero Campaign Operations tables.
- `phase4c_step6_verify`: empty disposable database, no ledger, zero remaining schemas/tables after testing.
- Only one local PostgreSQL cluster was discoverable, listening on port 5432.
- No local or remote Git ref contains migration 045 or the Campaign Operations implementation. The migration remains untracked and unreleased.
- Campaign Operations roles exist cluster-wide from disposable testing, but have no `pqxx` membership and no persisted Campaign Operations schema or ledger entry.

Editing 045 in place is safe for every known environment. If an unknown external environment has already recorded an older 045, the migrator will detect a checksum mismatch and will not replay it; that environment would require a new corrective migration.

Read-only operator check for any external environment:

```bash
psql -X -v ON_ERROR_STOP=1 ... -c \
"BEGIN READ ONLY;
 SELECT version, filename, checksum, applied_at
 FROM schema_migrations
 WHERE version = '045';
 COMMIT;"
```

## Files changed by this review

- [CampaignOperationsRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/CampaignOperationsRepository.cpp:741>)
- [CampaignOperationsMigrationTests.sql](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsMigrationTests.sql:101>)
- [CampaignOperationsRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsRepositoryTests.cpp:1141>)

No migration, documentation, CLI, scheduler, experiment, governance, or later-phase behavior was changed.

## Commands and results

Inspection commands included `git status`, `git diff`, `rg`, `nl`, `sed`, `wc`, `git log --all`, `git rev-list --all --objects`, Xcode project listing, PostgreSQL process/socket discovery, and read-only catalog queries. All completed as expected.

Notable verification commands/results:

- `ps` inspection: active scheduler and six training workers found; none interrupted.
- Read-only database enumeration and ledger queries: results documented above.
- Initial warning-clean compilation attempts exposed missing include/link inputs; the completed focused compile with `-Wall -Wextra -Werror` passed.
- Initial `REPEATABLE READ` probe: failed with the expected uncaught `pqxx::unique_violation`, confirming the defect.
- Final focused repository executable: passed with both isolation races, all oversized fixtures, migration 045 applied twice, and catalog tests.
- `plutil -lint ExpertAdvisor.xcodeproj/project.pbxproj`: passed.
- `git diff --check`: passed.
- Explicit trailing-whitespace scan of affected untracked files: passed.
- Final disposable database cleanup: zero non-system tables and zero test schemas.
- Final `LSTM` read-only recheck: zero migration-045 rows and zero Campaign Operations tables.

A broad Xcode build was not rerun because the active scheduler may launch from the shared build output and six training workers were active. The directly affected sources were compiled warning-clean and executed through the focused suite.

## `git status --short`

```text
 M Database/README.md
 M ExpertAdvisor.xcodeproj/project.pbxproj
?? CampaignOperations_Phase1_Implementation_Output.md
?? CampaignOperations_Phase1_Independent_Review_Output.md
?? CampaignOperations_Phase1_Targeted_Correction_Implementation_Output.md
?? CampaignOperations_Phase1_Targeted_Followup_CEE_Output.md
?? Database/migrations/045_campaign_operations_foundation.sql
?? Sources/CampaignOperations.cpp
?? Sources/CampaignOperations.hpp
?? Sources/CampaignOperationsRepository.cpp
?? Sources/CampaignOperationsRepository.hpp
?? Tests/CampaignOperationsMigrationTests.sql
?? Tests/CampaignOperationsRepositoryTests.cpp
?? Tests/CampaignOperationsTests.cpp
```

## `git diff --stat`

`git diff --stat` excludes untracked files:

```text
 Database/README.md                      | 20 +++++++++++++++++++-
 ExpertAdvisor.xcodeproj/project.pbxproj | 16 ++++++++++++++++
 2 files changed, 35 insertions(+), 1 deletion(-)
```

## Remaining limitations

- No remote/staging database configuration was available; only all discoverable local databases were inspected.
- A full Debug/Release Xcode rebuild was intentionally omitted due to active production-like scheduler/training activity.
- The working tree remains uncommitted and most Campaign Operations files remain untracked.

**READY TO COMMIT CAMPAIGN OPERATIONS PHASE 1**