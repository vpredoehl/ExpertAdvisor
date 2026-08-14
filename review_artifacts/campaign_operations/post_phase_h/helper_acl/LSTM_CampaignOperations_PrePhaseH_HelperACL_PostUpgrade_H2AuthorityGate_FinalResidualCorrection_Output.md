---
title: "LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade H2 Authority Gate Final Residual Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_PrePhaseH_HelperACL_PostUpgrade_H2AuthorityGate_FinalResidualCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade H2 Authority Gate Final Residual Correction

# Final Residual Correction Review

`FINAL_RESIDUAL_CORRECTION_STATUS=BLOCKED`

1. Files changed:

- [Scripts/CampaignOperationsH1DeploymentAudit.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Scripts/CampaignOperationsH1DeploymentAudit.sh)
- [Tests/CampaignOperationsPhaseH2WorkflowTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.sh)
- [Tests/CampaignOperationsPrePhaseHMigrationTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPrePhaseHMigrationTests.sh)

2. Root cause:

`post-upgrade` filtered ordinary H1A003 graph findings before proving migration 056/H2 authority existed.

3. Correction:

The wrapper now checks migration 056 by exact version, filename, and SHA-256 before filtering. Without H2 authority, ordinary graph delegation fails with:

`H2A004 ... post-upgrade role-graph delegation requires H2 / migration-056 authority`

Boundary-authority graph findings remain H1A003 and are never delegated.

4. Test coverage added:

- H2-present fixture: invokes the H1 wrapper and asserts the delegated H2 audit passes.
- H2-absent fixture: removes only the disposable 056 ledger row and asserts deterministic H2A004 failure.
- Boundary-hostile fixture: asserts H1A003 rejection.

5. Verification:

PASS:

```text
bash -n Scripts/CampaignOperationsH1DeploymentAudit.sh
bash -n Tests/CampaignOperationsPhaseH2WorkflowTests.sh
bash -n Tests/CampaignOperationsPrePhaseHMigrationTests.sh
git diff --check
Tests/CampaignOperationsPhaseHProductionConnectionTests.sh
```

Frozen migration 055 and migration 056 have no tracked diffs. No ACL or membership broadening was added.

Runtime proof is blocked: the H2 workflow cannot reach the wrapper because the existing H1 fixture fails during schema restore with missing cluster-global role `campaign_operations_h1_boundary_authority`. The available pre-Phase-H fixture is incomplete for the 064 overlay.

`git diff --stat`:

```text
Database/README.md                                  |  15 +-
Database/backups/LSTM_latest.dump                   |   4 +-
Database/backups/LSTM_latest.dump.json              |  14 +-
Scripts/CampaignOperationsH1DeploymentAudit.sh      |  85 ++++++++++-
Tests/CampaignOperationsAdmissionTopLevelTests.sh   |  29 +++-
Tests/CampaignOperationsPhaseH2WorkflowTests.sh     |  36 +++++
Tests/CampaignOperationsPrePhaseHPrivilegeTests.sql | 159 ++++++++++++++++++---
```

Existing unrelated worktree changes remain untouched.