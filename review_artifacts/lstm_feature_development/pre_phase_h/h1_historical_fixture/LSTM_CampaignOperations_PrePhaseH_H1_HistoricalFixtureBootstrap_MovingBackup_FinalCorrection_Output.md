---
title: "LSTM Campaign Operations Pre-Phase-H H1 Historical Fixture Bootstrap and Moving Backup Final Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_PrePhaseH_H1_HistoricalFixtureBootstrap_MovingBackup_FinalCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Campaign Operations Pre-Phase-H H1 Historical Fixture Bootstrap and Moving Backup Final Correction

Implemented the narrow historical-fixture correction.

- Changed [CampaignOperationsPhaseH1MigrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh) and [CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh).
- Both fixtures now:
  - create only inert predecessor roles needed through 054 (not H1/055 authority roles);
  - create the disposable database owned by `vjp`;
  - restore the immutable schema-049 dump;
  - fail closed on metadata (`049`, `939e126`, timestamp) and the required SHA-256;
  - run 050–052 under `SET ROLE vjp`, restoring the historical owner of `experiment_scheduler_protocol`.
- The protected-function fixture no longer references `LSTM_latest.dump`; targeted historical paths contain zero mutable-backup references.

Root causes:

1. The disposable DB was owned by `campaign_manager_login`, so switching to historical `vjp` lacked `public` schema creation authority. Making `vjp` the disposable DB owner reproduces the historical migration principal without adding an ACL grant.
2. `CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh` still restored the mutable latest backup. It now uses the exact schema-049 fixture and upgrades through 054.

Verification:

- `bash -n` for both scripts: exit 0.
- `git diff --check`: exit 0.
- `bash Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`: exit 0.
- `bash Tests/CampaignOperationsPhaseH1MigrationTests.sh`: reaches migration 055 and reports:
  - `H1_ACL_CATALOG_V3_OK rows=41`
  - `Campaign Operations Phase H1 migration tests passed`
  - H1REG027 passes.

The H1 wrapper’s final exit status remains 1 due to an independent post-upgrade H2-audit parse error after H1 success:

```text
ERROR:  42601: syntax error at or near ":"
LINE 4: AND checksum=:'h2_checksum')
```

This originates in `Scripts/CampaignOperationsH1DeploymentAudit.sh:179`, outside the requested scope; no H1REG027/catalog contract was changed.

Migration 055 is byte-identical to HEAD:

```text
1b13d3a64336d7cbd55c935396ec42c4c06320105677829f0cf405733e5715fe
```

`git diff --stat`:

```text
Tests/CampaignOperationsPhaseH1MigrationTests.sh   | 86 +++++++++++++++++-----
Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh | 81 +++++++++++++++-----
2 files changed, 130 insertions(+), 37 deletions(-)
```

`git status --short` retains the pre-existing untracked metadata/review artifacts; the only tracked diffs are the two fixture scripts above.