---
title: "Campaign Operations H1 SECURITY DEFINER Ownership Sealing — Independent Reverification Findings"
document_type: "independent reverification"
status: "PASS"
date: "2026-08-13"
---

# Campaign Operations H1 SECURITY DEFINER Ownership Sealing — Independent Reverification Findings

## Verdict

**PASS — no blocking defect found.**

The correction is narrowly scoped to the actual ownership/dependency failure. Migration 063 restores only the four `SELECT` privileges that migration 055's H1 contract already intended `campaign_operations_owner` to retain, while preserving H1 relation ownership under `campaign_operations_h1_boundary_authority` and withholding mutation privileges. The supplied regression coverage also reaches the real user-facing admission command and verifies idempotent replay and non-interference with experiment/scheduler-worker state.

I find no basis for moving `enforce_campaign_operations_completion_boundary_consistent()` into the literal H1 sealed-authority function manifest. Retaining it under `campaign_operations_owner` is consistent with its role as a read-only deferred consistency checker, provided the owner retains the exact read dependencies it requires. Migration 063 now explicitly freezes that ownership, `SECURITY DEFINER` status, and hardened search path.

## 1. Migration 063: ownership/sealing correction

**Verified.**

Migration 063 grants:

- `SELECT` on `campaign_operations_dispatch_attempt`
- `SELECT` on `campaign_operations_dispatch_audit_reference_event`
- `SELECT` on `campaign_operations_completion_event`
- `SELECT` on `campaign_operations_completion_audit_reference_event`

to `campaign_operations_owner` only.

The migration then independently asserts that, for all four relations:

- `campaign_operations_owner` has `SELECT`;
- it does **not** have `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, `REFERENCES`, or `TRIGGER`;
- relation ownership remains `campaign_operations_h1_boundary_authority`.

It also rejects accidental completion-history `SELECT` for `pqxx` and `campaign_operations_campaign_creator`.

This is the correct repair shape. It restores the execution rights of existing owner-defined `SECURITY DEFINER` readers without transferring H1 relation ownership or granting a mutation path.

**Finding:** no privilege broadening or ownership regression identified.

## 2. `enforce_campaign_operations_completion_boundary_consistent()` and the H1 literal manifest

**Verified decision: do not add it to the H1 sealed-authority manifest.**

The function is a deferred trigger checker. Its operation is observational: it reads `campaign_operations_completion_event`, compares existence of the completion fact with `NEW.completion_boundary_closed`, and raises on mismatch. It does not create, mutate, authorize, dispatch, or otherwise own an H1 transition.

Migration 055 deliberately transferred a selected set of H1 transition/guard functions to `campaign_operations_h1_boundary_authority`; this function was not in that sealed set. The correction does not reinterpret that architecture merely because later table ownership removed a read privilege on which the function depended.

Migration 063 now provides the right local contract for this non-manifest function:

- owner must remain `campaign_operations_owner`;
- it must remain `SECURITY DEFINER`;
- its `proconfig` must remain exactly `search_path=pg_catalog, public, pg_temp`.

That is preferable to expanding the H1 literal protected-function set after the fact.

**Finding:** no manifest omission. Adding the function to the sealed H1 authority manifest would be a broader authority change than required by the defect.

## 3. Remaining `campaign_operations_owner` SECURITY DEFINER functions

**Verified with no additional ownership transfer required.**

The correction's dependency audit identifies the functions whose effective execution was affected by loss of the restored H1 read tuples:

- `campaign_operations_completion_blockers(bigint)`
- `campaign_operations_completion_classification(bigint)`
- `campaign_operations_completion_evidence_text(bigint,text)`
- `campaign_operations_future_actions_allowed(bigint)`
- `enforce_campaign_operations_completion_boundary_consistent()`

The correction transcript further demonstrates execution as `SET LOCAL ROLE campaign_operations_owner` for representative affected readers: `campaign_operations_future_actions_allowed`, `campaign_operations_completion_blockers`, and `campaign_operations_completion_evidence_text` execute after migration 063. A call to `campaign_operations_completion_classification` raised the domain-expected "completion remains blocked" result rather than a privilege error; that is evidence that the ownership dependency itself is repaired.

The remaining owner-defined `SECURITY DEFINER` functions were not indiscriminately migrated. That is the correct result: ownership should change only where the H1 authority contract specifically requires sealed transition/guard ownership, not merely because a function happens to use `SECURITY DEFINER`.

**Finding:** no second instance of the same ACL/ownership-dependency defect was established beyond the read paths repaired by migration 063.

## 4. Real top-level admission regression

**Verified.**

`Tests/CampaignOperationsAdmissionTopLevelTests.sh` invokes the actual Release binary:

`LSTM_Release --campaign-operations-admit MATERIALIZATION_ID --campaign-operations-actor ... --campaign-operations-reason ... --yes`

with `CAMPAIGN_OPERATIONS_PRE_PHASE_H_DB_USER` set to the deployment login.

The regression verifies:

- materialization fixture completeness before execution;
- no existing operational campaign for the fixture;
- first command returns `disposition=recorded`;
- exactly one operational campaign exists afterward;
- exactly one `campaign_created` audit reference exists;
- persisted materialization member count matches source materialization;
- no completion event is spuriously created;
- replay returns `disposition=existing_identical`;
- replay does not duplicate campaign or audit rows;
- experiment table snapshot is unchanged;
- scheduler-worker-attempt snapshot is unchanged;
- the pre-Phase-H login remains non-superuser, lacks H1 production roles, and cannot read H1 completion history.

This is the correct regression boundary for the failure that was originally invisible to repository-only testing.

**Finding:** top-level dispatch/admission coverage is adequate for this defect.

## 5. Originally failing admission path

**Verified from execution evidence.**

The correction evidence records the repaired materialization-2 path successfully creating one operational campaign and one `campaign_created` audit row, with the deferred completion-boundary trigger committing successfully. Replay returned `existing_identical`.

The evidence also records:

- 2 materialization members;
- experiment snapshot unchanged at 330 rows;
- worker-attempt snapshot unchanged at 620 rows;
- no scheduler process started;
- no forbidden H1 production authority gained by the pre-Phase-H login.

The migration checksum reported and independently computed in the correction run is:

`1f99017a11b39f5ff58bbf9d24b297d9b52308435cc4d073b75f7d39ed28386b`

**Finding:** the original runtime failure is closed by the correction rather than bypassed.

## 6. Security review

No privilege escalation was found in the correction.

The important security properties remain:

- H1 relations remain owned by `campaign_operations_h1_boundary_authority`.
- `campaign_operations_owner` is given read access, not relation ownership.
- No mutation ACL is added by migration 063.
- The ordinary pre-Phase-H login is not granted completion-table `SELECT`.
- The admission CLI remains routed through the dedicated pre-Phase-H principal.
- The affected deferred trigger still executes through its prior `SECURITY DEFINER` owner and hardened search path.
- No scheduler/lifecycle side effect is introduced by admission.

I found no missing ACL tuple required by the demonstrated owner-read paths after the four restored grants.

## 7. Residual observations

Two non-blocking observations remain.

First, `Tests/CampaignOperationsAdmissionTopLevelTests.sh` is intentionally environment-backed rather than hermetic. It requires a prepared materialization and refuses to run after that materialization has already been admitted. This is reasonable for a deployment-path regression, and it performs its own replay test during the first successful run, but repeatable CI execution will require a fresh/disposable database fixture.

Second, migration 063's inline negative ACL check names `pqxx` and `campaign_operations_campaign_creator`, while the separate pre-Phase-H privilege regression checks the actual deployed login. The broader H1 migration/deployment audit remains the authority for the full role graph. I do not consider this a coverage gap requiring expansion of migration 063 itself.

Neither observation changes the PASS verdict.

## Final disposition

**Independent reverification: PASS.**

The ownership/sealing defect is corrected at the narrow ACL boundary, the H1 sealed-authority model is not unnecessarily expanded, the affected owner-defined `SECURITY DEFINER` read paths are restored, and the real top-level admission command now has regression evidence covering the original failure.

Recommended next action: archive this finding with the correction output and proceed to the next deployment/activation step. No additional corrective implementation is required for this issue.
