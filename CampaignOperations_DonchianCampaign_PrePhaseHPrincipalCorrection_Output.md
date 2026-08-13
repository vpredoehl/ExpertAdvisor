---
title: "Campaign Operations Pre-Phase-H Principal Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_DonchianCampaign_PrePhaseHPrincipalCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# Campaign Operations Pre-Phase-H Principal Correction

Implementation complete.

- Root cause: combined authority wiring and view-owner ACL defects. Pre-Phase-H commands were hard-wired to `pqxx`; sealed ownership removed `campaign_operations_owner`’s underlying request-table read needed by status views.
- Correction:
  - Added required `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER` routing.
  - Rejects superusers and Phase-H production-capable principals.
  - Added migration 062 restoring view-owner `SELECT` and revoking accidental `pqxx` Phase 2 reads.
  - Added privilege-boundary regression SQL and documentation.
- Phase-H production routing remains unchanged.

Files changed:

- `Sources/ExperimentScheduler.cpp`
- `Database/migrations/062_campaign_operations_pre_phase_h_view_access.sql`
- `Tests/CampaignOperationsPrePhaseHPrivilegeTests.sql`
- `Tests/CampaignOperationsPhaseHProductionConnectionTests.sh`
- `docs/CampaignOperationsPhase2.rst`
- `Database/README.md`

Tests:

- Phase 2 CLI parser: passed.
- Phase-H routing test: passed.
- Release build: passed; only existing warnings.
- `git diff --check`: passed.
- Missing principal and superuser rejection: passed.
- SQL privilege regression requires a disposable test database and was not run.

Deployment required: apply migration 062, provision a dedicated non-superuser pre-Phase-H LOGIN with the required Phase 2 capability memberships, and set `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER`. Do not use `vjp`, `pqxx`, or a Phase-H production LOGIN.

Final judgment: PASS for the implementation; live independent reverification remains pending migration/deployment.