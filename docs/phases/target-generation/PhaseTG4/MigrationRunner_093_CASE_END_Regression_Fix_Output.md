---
title: "Migration Runner 093 CASE END Regression Fix"
document_type: "architecture review"
status: "final"
generated_from: "MigrationRunner_093_CASE_END_Regression_Fix_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Migration Runner 093 CASE END Regression Fix

Implemented the migration-runner fix without modifying migration 093.

- Parser correction: bare `END;` is accepted only while lexically inside an open SQL `CASE`; at statement level it remains fail-closed as transaction control. `END WORK`, `END TRANSACTION`, and chain variants remain rejected regardless of CASE depth. Lexing ignores comments and quoted/dollar-quoted content.
- Added regression fixtures/tests for multiline CASE, unmatched `END;`, BEGIN/END wrappers, existing unmatched COMMIT, and actual checked-in 086→093 execution.
- The disposable 093 test verifies SHA-256 `7c0d60c3e8a40a109c1e3f8bb629c6f8f092e4cd391608590440712c8e9937f7` is recorded and confirms `resume_requested=true` becomes origin `operator`.

Files changed:

- `migrate_lstm_db.sh`
- `Tests/MigrationRunnerAtomicityTests.sh`
- `Tests/fixtures/migration_runner/case_expression/060_multiline_case.sql`
- `Tests/fixtures/migration_runner/internal/031_internal_end_transaction.sql`
- `Tests/fixtures/migration_runner/transaction_control/051_unmatched_end.sql`

Validation run:

- `bash -n migrate_lstm_db.sh Tests/MigrationRunnerAtomicityTests.sh` — passed
- `bash Tests/MigrationRunnerAtomicityTests.sh` — passed
- `git diff --check` — passed

No production migrations, production database, scheduler, or LSTM executable were touched.

`git status --short`:

```text
 M Tests/MigrationRunnerAtomicityTests.sh
 M migrate_lstm_db.sh
?? Tests/fixtures/migration_runner/case_expression/
?? Tests/fixtures/migration_runner/internal/031_internal_end_transaction.sql
?? Tests/fixtures/migration_runner/transaction_control/051_unmatched_end.sql
```

`git diff --stat`:

```text
 Tests/MigrationRunnerAtomicityTests.sh |  68 +++++++++++++++++-
 migrate_lstm_db.sh                     | 122 +++++++++++++++++++++++++++++----
 2 files changed, 174 insertions(+), 16 deletions(-)
```

Recommended commit message:

```text
Fix migration runner handling of CASE END terminators
```