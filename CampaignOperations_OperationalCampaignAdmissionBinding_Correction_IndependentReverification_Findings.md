---
title: "Campaign Operations Operational Campaign Admission Binding Correction — Independent Reverification"
document_type: "independent reverification"
status: "final"
date: "2026-08-13"
---

# Campaign Operations Operational Campaign Admission Binding Correction
## Independent Reverification Findings

### Final judgment

**IMPLEMENTATION: PASS**

**DEPLOYMENT / LIVE ACTIVATION: CONDITIONALLY READY, pending the explicit deployment grant and live privilege/admission verification.**

No correctness blocker was found in the supplied implementation package. The correction closes the identified architectural gap by adding one explicit Campaign Operations admission command that converts an existing, authoritative Phase 4D campaign materialization into the immutable Campaign Operations operational-campaign identity through the pre-existing canonical builder and persistence authority.

The remaining item is operational rather than an implementation defect: the deployed `campaign_operations_pre_phase_h_login` must receive `campaign_operations_campaign_creator` in addition to its existing budget-administrator and request-acceptor capabilities before the command can succeed in the live database. The package intentionally leaves that grant as an explicit deployment action rather than embedding it in migration 062.

### Reverified architecture and authority boundary

The new production path is appropriately owned by the Campaign Operations service/workflow, while `ExperimentScheduler.cpp` remains the CLI parser/router. The path is:

`--campaign-operations-admit MATERIALIZATION_ID`
→ `RunCampaignOperationsCommand()`
→ `RunOperationalCampaignAdmissionCommand()`
→ `AdmitOperationalCampaign()`
→ authoritative Phase 4D materialization load
→ `BuildOperationalCampaign()`
→ `PersistOperationalCampaign()`.

This is the correct boundary for the missing materialization-to-operational-campaign transition. It does not overload budget administration, request acceptance, scheduler startup, manager execution, or status commands.

The command is classified as a durable Campaign Operations mutation, so the existing parser rules require `--yes`, reject `--dry-run`, and require both `--campaign-operations-actor` and `--campaign-operations-reason`.

### Authoritative input and canonical construction

`AdmitOperationalCampaign()` accepts only the materialization ID from the caller. It reloads the persisted Recommendation Campaign materialization and rejects a nonexistent materialization. It does not trust caller-supplied canonical text, hashes, member counts, action kinds, scope kinds, or contract versions.

The service then constructs the operational campaign with the existing `BuildOperationalCampaign()` primitive. That preserves the existing canonical contract:

- origin: `phase4d_materialization_v1`
- action: `dispatch_full_materialization`
- scope: `complete_materialization`
- exact persisted materialization contract version
- exact persisted materialization canonical identity/hash
- exact selected-member count.

The repository revalidates the materialization binding before first insert, so the service-layer load is not the sole integrity check.

### Persistence and idempotency

`PersistOperationalCampaign()` now enforces `campaign_operations_campaign_creator` before any replay/insert path. This is important because an unauthorized caller cannot exploit the `existing_identical` path to observe a successful privileged operation.

The persistence sequence remains sound:

1. validate the canonical operational campaign;
2. acquire the materialization-scoped transaction advisory lock;
3. load any existing campaign for that materialization;
4. return `existing_identical` only for an exact match;
5. reject changed payload as `persistenceConflict`;
6. validate the durable Phase 4D materialization binding;
7. insert one operational campaign;
8. record one `campaign_created` audit event with the creator capability.

The database uniqueness constraint on recommendation materialization plus the advisory lock and exact replay comparison preserve one operational campaign per materialization.

Focused repository evidence in the package covers first creation, exact replay, changed-payload conflict, rollback behavior, missing materialization rejection, capability denial on replay, deterministic command output, and the absence of downstream budget/reservation/request side effects.

### Least privilege and routing

The new admission command is routed through `CampaignOperationsPrePhaseHConnectionString()` and the existing `ValidateCampaignOperationsPrePhaseHPrincipal()` preflight.

That preflight fails closed if `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER` is missing and rejects a superuser or a principal inheriting any Phase-H production capability. The admission route therefore does not fall back to ordinary `pqxx`, does not use `LSTM_DB_USER`, and does not acquire a Phase-H production principal.

The intended deployment principal is the existing dedicated `campaign_operations_pre_phase_h_login`, extended with exactly one additional NOLOGIN capability:

```sql
GRANT campaign_operations_campaign_creator
TO campaign_operations_pre_phase_h_login;
```

The privilege regression included in the package expects that LOGIN to have exactly these three Campaign Operations capabilities:

- `campaign_operations_campaign_creator`
- `campaign_operations_budget_administrator`
- `campaign_operations_request_acceptor`

and no Phase-H production capability membership.

This is a coherent least-privilege deployment model for the existing reviewed pre-Phase-H service. It intentionally concentrates the three sequential pre-Phase-H operator functions in one dedicated LOGIN while still excluding authorization, control, recovery, completion, dispatch, transactional Phase 5, and Phase-H production authority.

### Migration 062

Migration 062 is correctly *not* repurposed as an admission-capability deployment migration. Its scope remains restoring the status-view owner's required read access after H1 ownership sealing and removing inappropriate `pqxx` view/request-table access.

It does not create a LOGIN and does not grant `campaign_operations_campaign_creator`. That is consistent with the project's explicit deployment-capability model and with the documentation requiring a separate reviewed grant.

### Side-effect isolation

The admission service itself creates only the operational campaign row and its audit row. The focused repository tests explicitly verify that admission does not create:

- budget ledger entries,
- reservations,
- operational requests.

The CLI help and service path add no execution, activation, experiment, scheduler, or worker action. The new command returns immediately after campaign admission and does not invoke any downstream Campaign Operations phase.

Therefore the correction preserves the required staged workflow:

materialization
→ explicit operational campaign admission
→ budget grant
→ request acceptance
→ production admission/dispatch.

### Test and evidence assessment

The supplied correction report records successful:

- Release build;
- Phase 2 CLI parser tests;
- Phase H1 CLI tests;
- Phase-H production/principal routing tests;
- repository-test syntax compilation with strict warnings;
- `git diff --check`.

The independent package also contains focused test source for the admission semantics and privilege boundary. The broad H1 disposable-cluster harness was not completed because the local disposable cluster lacked `campaign_operations_h1_boundary_authority`; that is an environment/setup limitation, not evidence of an admission-path failure.

This reverification is based on the supplied source/test package and recorded test outputs. I did not execute the macOS/Xcode/PostgreSQL test suite from this sandbox.

### Residual deployment gate

Before using admission against materialization 2 in the live database, verify the deployment role and then perform the first controlled admission.

Required role change:

```sql
GRANT campaign_operations_campaign_creator
TO campaign_operations_pre_phase_h_login;
```

Then verify that the LOGIN has the three intended pre-Phase-H capabilities and no Phase-H production capability. Only after that should the live command be issued.

An expected successful first admission should report `CAMPAIGN_OPERATIONS_ADMISSION` with `disposition=recorded`; an immediate exact retry should report the same `operational_campaign_id` with `disposition=existing_identical`.

### Conclusion

The correction is **independently accepted as PASS at the implementation level**.

No source-level residual correction is required from the supplied evidence. The next action is deployment of the creator capability to the already reviewed pre-Phase-H LOGIN, followed by a controlled live privilege check and admission/replay verification. Until that grant is applied, the implementation is correct but the production admission path is intentionally not usable.
