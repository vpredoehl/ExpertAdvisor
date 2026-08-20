---
title: "Campaign Operations Donchian Campaign Pre-Phase-H Principal Correction — Independent Reverification Findings"
document_type: "independent reverification findings"
status: "final"
date: "2026-08-12"
scope: "Packaged implementation and supplied verification evidence"
---

# Campaign Operations Donchian Campaign Pre-Phase-H Principal Correction
## Independent Reverification Findings

## Verdict

**PASS — implementation correction verified, with live deployment verification still required before operational use.**

I found no implementation-level blocker in the packaged correction. The change is internally consistent with the intended least-privilege boundary: pre-Phase-H Campaign Operations commands no longer use the generic `pqxx` connection, require an explicitly selected deployment LOGIN, reject superusers and Phase-H production-capable principals, restore the status-view owner's required underlying-table read, and remove the accidental direct Phase 2 reads from `pqxx`.

This finding does **not** certify the live database after migration 062, because the supplied evidence explicitly states that the disposable-database SQL privilege regression was not run and that migration 062 had not yet been applied during the implementation review. Therefore the implementation can proceed to deployment/reverification, but the production activation step should remain gated on the live privilege checks described below.

## Evidence reviewed

The reverification package identifies branch `lstm-feature-development` at baseline HEAD `843c967b56522826a3f9b7b29c444c9bf7070479`, includes the complete working-tree diff, the correction output, the new migration, privilege regression SQL, routing regression shell test, Phase 2 documentation, and database migration documentation.

`git diff --check` is clean in the supplied evidence.

## Finding 1 — Pre-Phase-H principal routing is corrected

`CampaignOperationsPrePhaseHConnectionString()` now requires `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER`. There is no fallback to `LstmDbConnectionString()` or `pqxx`. A missing or empty environment variable throws before a Campaign Operations workflow command is opened.

This is the correct fail-closed behavior for the discovered defect. It prevents the ordinary LSTM runtime principal from silently becoming the authority used for Phase 2 Campaign Operations.

**Result: PASS.**

## Finding 2 — Phase-H production authority is separated from the new principal

`ValidateCampaignOperationsPrePhaseHPrincipal()` opens the selected connection and rejects the current principal when it is either a PostgreSQL superuser or a member of any enumerated Phase-H production authority: `campaign_operations_production_enabler`, `campaign_operations_production_disabler`, `campaign_operations_production_dispatcher`, `campaign_operations_production_dispatch_service`, `campaign_operations_production_phase5_transactional`, `campaign_operations_production_reader`, or `campaign_operations_scheduler_protocol_evidence_reader`.

The validation occurs before the pre-Phase-H Campaign Operations command dispatch body. The established Phase-H production connection function and its dedicated production environment variables remain separate.

**Result: PASS.**

## Finding 3 — Migration 062 is narrowly scoped and least-privilege

`062_campaign_operations_pre_phase_h_view_access.sql` revokes all privileges on `campaign_operations_budget_status_v1` and `campaign_operations_request_status_v1` from `pqxx`, revokes `SELECT` on `campaign_operations_operational_request` from `pqxx`, and grants `SELECT` on `campaign_operations_operational_request` to `campaign_operations_owner`.

The grant to `campaign_operations_owner` is necessary because the Phase 2 status views remain owned by that NOLOGIN owner while migration 055 transferred ownership of the underlying operational-request table to the H1 boundary authority. The migration creates no LOGIN, grants no Phase 2 capability membership, and grants no Phase-H production privilege.

The correction is appropriately additive rather than modifying an already-applied earlier migration.

**Result: PASS.**

## Finding 4 — Privilege regression coverage matches the intended boundary

`Tests/CampaignOperationsPrePhaseHPrivilegeTests.sql` checks that `campaign_operations_owner` can select the sealed operational-request table; `pqxx` cannot directly select the two Phase 2 status views; `pqxx` cannot select/insert/update/delete the operational-request table; `pqxx` cannot execute the production dispatch V2 transition and does not inherit Phase-H production roles; `campaign_operations_reader` can execute both status views; `campaign_operations_budget_administrator` can execute the budget status view; `campaign_operations_request_acceptor` can execute the request status view; Phase-H capability roles remain NOLOGIN; and the manager LOGIN remains separated from Phase 2 capabilities.

That is good coverage of both the view-owner defect and the authority boundary that caused the original failure.

The supplied implementation evidence says this SQL regression was **not executed** because it requires a disposable database. That is the principal remaining verification gap.

**Result: PASS for test design; execution pending.**

## Finding 5 — Documentation is consistent with the code

`docs/CampaignOperationsPhase2.rst` now states that the ordinary `pqxx` connection is not a Phase 2 principal; `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER` is required; there is no `pqxx` fallback; `LSTM_DB_USER` does not select the pre-Phase-H principal; superusers and Phase-H production-capable principals are rejected; and the deployment LOGIN may receive the intentionally reviewed Phase 2 NOLOGIN capabilities but must not be a Phase-H production LOGIN.

`Database/README.md` documents migration 062 as a view-owner read correction that creates no LOGIN or capability grant.

**Result: PASS.**

## Finding 6 — Supplied focused verification is credible but not yet a live deployment proof

The correction output records successful Phase 2 CLI parser testing, Phase-H routing testing, Release build, `git diff --check`, missing-principal rejection, and superuser rejection. The implementation transcript also records the rebuilt binary failing closed when the pre-Phase-H principal is absent and rejecting `vjp` as a superuser, while leaving scheduler/worker/experiment state untouched.

However, the package explicitly states that migration 062 had not been applied and the SQL privilege regression had not been executed on a disposable/live-equivalent database.

**Result: PASS with deployment gate.**

## Required live deployment reverification

Before using the corrected pre-Phase-H Campaign Operations path operationally:

1. Back up the database because migration 062 changes schema-level privilege state.
2. Apply migration 062.
3. Provision a dedicated non-superuser LOGIN for `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER`.
4. Grant only the Phase 2 capability memberships intentionally required for that service.
5. Confirm the LOGIN has no Phase-H production capability membership.
6. Run `Tests/CampaignOperationsPrePhaseHPrivilegeTests.sql` against a migration-equivalent disposable database, or perform equivalent live catalog and command checks.
7. Verify `--campaign-operations-budget-status` and `--campaign-operations-request-status` succeed through the reviewed pre-Phase-H LOGIN.
8. Re-run `--campaign-operations-production-readiness` and confirm the Phase-H production path remains ready and unchanged.

Do not use `vjp`, `pqxx`, or any Phase-H production LOGIN for `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER`.

## Final disposition

**Independent reverification: PASS for the implementation.**

**Residual risk:** deployment-state verification only. No code correction is presently indicated by the supplied evidence.

**Recommended next action:** archive these findings, apply/back up for migration 062, provision the dedicated pre-Phase-H LOGIN, execute the privilege regression/live command checks, and then refresh production build authorization if the schema-change archival commit changes the authorized source/build identity.
