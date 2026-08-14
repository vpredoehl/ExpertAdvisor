---
title: "LSTM Campaign Operations Pre-Phase-H Helper ACL Frozen H1 Audit Final Residual Correction"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_PrePhaseH_HelperACL_FrozenH1Audit_FinalResidualCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# LSTM Campaign Operations Pre-Phase-H Helper ACL Frozen H1 Audit Final Residual Correction

Implemented the residual correction.

- Root cause: migration 064 rewrote the frozen H1 audit via `pg_get_functiondef()` and directly invoked it, causing H1A003 on the post-H1 role graph.
- Migration 064 now retains only the exact two-entry ACL grant:
  `campaign_operations_budget_administrator` and `campaign_operations_request_acceptor` → `public.lock_campaign_operations_campaign(bigint)`.
- Frozen migration 055 and its audit function remain unchanged. Migration 064 no longer rewrites or invokes H1A003.
- Added independent 064 checks for ownership, security definer/search path, grant options, forbidden helper grants, PUBLIC/pqxx access, LOGIN boundaries, and Phase-H authority.
- The audit wrapper validates the separate 064 overlay and tolerates only the two exact ACL deltas when 064 is recorded.

Files touched for this correction:

- [migration 064](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/064_campaign_operations_pre_phase_h_helper_acl_reconciliation.sql>)
- [064 compatibility manifest](</Volumes/Developer SSD/ExpertAdvisor/Database/manifests/064_campaign_operations_pre_phase_h_acl_manifest.sql>)
- [H1 audit wrapper](</Volumes/Developer SSD/ExpertAdvisor/Scripts/CampaignOperationsH1DeploymentAudit.sh>)
- [focused migration test](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPrePhaseHMigrationTests.sh>)
- [privilege regression](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPrePhaseHPrivilegeTests.sql>)
- [Database README](</Volumes/Developer SSD/ExpertAdvisor/Database/README.md>)

Validation:

- Read-only 064 overlay against live LSTM: PASS.
- Live campaign-3 baseline unchanged: `3|1|1|1`.
- Shell syntax checks: PASS.
- Frozen-H1 rewrite/call scan: PASS.
- `git diff --check`: PASS.
- Migration 055: unchanged.
- Release build not run; changes are SQL/scripts/tests/docs only.
- Full disposable migration and real command regressions were not executed because no disposable 055 fixture was available and an active training worker is running.

The partially applied predecessor is handled by replay-safe `GRANT` statements; migration 064 does not insert its ledger row.