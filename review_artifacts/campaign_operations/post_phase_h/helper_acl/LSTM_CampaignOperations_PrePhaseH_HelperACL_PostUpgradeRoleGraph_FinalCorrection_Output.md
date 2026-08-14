---
title: "LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade Role-Graph Final Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_PrePhaseH_HelperACL_PostUpgradeRoleGraph_FinalCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade Role-Graph Final Correction

Implemented only the requested residual correction.

Changed:

- [CampaignOperationsH1DeploymentAudit.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Scripts/CampaignOperationsH1DeploymentAudit.sh:169)
  - Applies the existing post-H1 accommodation to `post-upgrade`.
  - Preserves H1 boundary-role edge rejection.
  - Delegates later role validation to H2 when migration 056 is present.

- [CampaignOperationsPrePhaseHMigrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPrePhaseHMigrationTests.sh:19)
  - Adds the accepted four-LOGIN production capability graph.
  - Adds a prohibited H1 boundary-membership regression.

Verification:

- `bash -n`, `sh -n`: PASS
- `git diff --check`: PASS
- Migration-064 frozen-H1 rewrite/direct-call scan: PASS
- Migration 055 unchanged: CONFIRMED
- Migration-064 two-entry ACL contract: unchanged and not broadened
- Focused/H1/H2 runtime regressions: BLOCKED by invalid disposable fixtures:
  - missing objects/functions in the catalog-only fixture;
  - PostgreSQL `plpgsql.dylib` load failure in the full fixture;
  - stale H2 fixture missing migration-059/dispatch-service role.

`git status --short` and `git diff --stat` show additional pre-existing worktree changes unrelated to this correction.

FINAL_RESIDUAL_CORRECTION_STATUS=BLOCKED