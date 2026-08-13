---
title: "Campaign Operations Operational Campaign Admission Binding Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_OperationalCampaignAdmissionBinding_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# Campaign Operations Operational Campaign Admission Binding Correction

Implemented the Campaign Operations admission correction.

- Root cause: no production caller connected Phase 4D materializations to `PersistOperationalCampaign()`.
- Owner: Campaign Operations service/workflow, per ADR-0010; CLI only parses/routes.
- New command: `--campaign-operations-admit MATERIALIZATION_ID --campaign-operations-actor ACTOR --campaign-operations-reason REASON --yes`
- Path: CLI → `AdmitOperationalCampaign()` → authoritative materialization loader → `BuildOperationalCampaign()` → `PersistOperationalCampaign()`.
- Idempotency: `recorded`, `existing_identical`, or conflict; creator capability is enforced even on replay.
- Routing: uses `CampaignOperationsPrePhaseHConnectionString()`.
- Authorization: extends `campaign_operations_pre_phase_h_login` with only `campaign_operations_campaign_creator`; no Phase-H authority.
- Side effects: no budget, request, dispatch, experiment, activation, or scheduler changes.

Changed core files include:

`CampaignOperationsService.*`, `CampaignOperations.*`, `CampaignOperationsRepository.cpp`, `ExperimentScheduler.cpp`, focused repository/CLI/routing tests, `docs/CampaignOperationsPhase2.rst`, and `Database/README.md`.

Validation:

- Release build: passed.
- Phase 2 CLI tests: passed.
- Phase H1 CLI tests: passed.
- Phase-H routing tests: passed.
- Repository test syntax with `-Wall -Wextra -Werror`: passed.
- `git diff --check`: passed.
- Broad H1 harness: blocked by local disposable-cluster setup missing `campaign_operations_h1_boundary_authority`; no live production data was modified.

Deployment required:

```sql
GRANT campaign_operations_campaign_creator
TO campaign_operations_pre_phase_h_login;
```

Final judgment: PASS for the implementation correction; live privilege verification remains pending that deployment grant and the blocked disposable-cluster harness setup.