---
title: "LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade H2 Authority Gate - Independent Reverification Findings"
document_type: "independent reverification"
status: "blocked_on_runtime_evidence"
date: "2026-08-13"
source_package: "LSTM_CampaignOperations_PrePhaseH_HelperACL_PostUpgrade_H2AuthorityGate_IndependentReverification_Input.tar.gz"
---

# LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade H2 Authority Gate
## Independent Reverification Findings

## Verdict

**The previously identified post-upgrade H2-authority gating defect is corrected at source level. No residual code defect was found in the scoped correction. Final closure remains BLOCKED only because the uploaded package does not contain enough of the H2 runtime fixture/audit dependency set to independently execute the required integration proof.**

`FINAL_REVERIFICATION_STATUS=BLOCKED`

`SCOPED_CODE_CORRECTION=PASS`

`RESIDUAL_CODE_DEFECT_COUNT=0`

`RUNTIME_EVIDENCE_BLOCKER_COUNT=1`

The correction now establishes the exact migration-056 ledger identity before it suppresses/delegates ordinary post-H1 `graph:` findings. When that exact H2 authority is absent and ordinary non-boundary graph findings exist, the wrapper fails closed with deterministic `H2A004`. When H2 is present, ordinary later deployment graph findings are delegated and the H2 audit is invoked. Any finding involving `campaign_operations_h1_boundary_authority` remains under H1 and fails as `H1A003`.

## Source Package Integrity

The uploaded archive was extracted and its included SHA-256 manifest was independently checked.

- Manifest entries checked: **22**
- Hash mismatches: **0**
- `git_diff_check.txt`: empty
- `bash -n Scripts/CampaignOperationsH1DeploymentAudit.sh`: PASS
- `bash -n Scripts/CampaignOperationsH2DeploymentAudit.sh`: PASS
- `bash -n Tests/CampaignOperationsPhaseH2WorkflowTests.sh`: PASS
- `bash -n Tests/CampaignOperationsPrePhaseHMigrationTests.sh`: PASS

The supplied migration-055 SHA-256 is:

`1b13d3a64336d7cbd55c935396ec42c4c06320105677829f0cf405733e5715fe`

This matches the hard-coded frozen-H1 checksum required by the supplied H2 deployment audit script.

## Reverification of the Previously Identified Defect

### 1. Exact H2 authority is established before ordinary graph delegation

The corrected H1 wrapper now computes the checked-in migration-056 SHA-256 and, at `post-upgrade`, queries `schema_migrations` for all three exact identity fields:

- `version='056'`
- `filename='056_campaign_operations_h2_privilege_deployment_contract.sql'`
- `checksum=<SHA-256 of the checked-in migration 056>`

This occurs before the ordinary H1 role-graph filtering branch.

That corrects the prior ordering defect, where ordinary post-H1 graph findings were filtered first and the existence of H2 authority was considered only later.

**Result: PASS.**

### 2. H2-absent post-upgrade now fails closed

The wrapper contains the required fail-closed condition:

- stage is `post-upgrade`;
- exact migration-056 identity is not present;
- at least one `graph:` finding exists;
- no `graph:` finding involves `campaign_operations_h1_boundary_authority`.

The wrapper then terminates through:

`diagnostic=H2A004`

with the deterministic detail:

`post-upgrade role-graph delegation requires H2 / migration-056 authority`

Crucially, this happens before the ordinary graph findings can be filtered away.

The focused H2 workflow test also contains the correct negative regression:

1. capture the disposable target's migration-056 checksum;
2. delete only the `schema_migrations` row for version 056;
3. invoke the H1 wrapper at `post-upgrade`;
4. require failure;
5. require `diagnostic=H2A004`;
6. require the exact H2-authority diagnostic;
7. restore the exact migration-056 ledger row.

The pre-Phase-H migration test independently covers the same fail-closed semantic on a 064-overlaid database without migration 056.

**Result: PASS by direct source and regression-design inspection.**

### 3. H2-present post-upgrade delegates only after authority proof

The ordinary graph filter is now conditional on:

`stage == post-upgrade && h2_deployment_installed == t`

for the post-upgrade case.

After the H1 role checks complete, the wrapper's `post-upgrade` branch invokes:

`Scripts/CampaignOperationsH2DeploymentAudit.sh --stage post-upgrade ...`

only when that exact H2 ledger identity was already established.

Because the script is running under `set -euo pipefail`, failure of the delegated H2 audit prevents an enclosing H1 success result.

The H2 workflow regression requires both:

`H2_DEPLOYMENT_AUDIT_OK stage=post-upgrade`

and:

`H1_DEPLOYMENT_AUDIT_V1_OK stage=post-upgrade`

for the H2-present positive case.

Therefore success is no longer equivalent to “H1 ignored the graph”; it requires the separately versioned H2 audit to succeed.

**Result: PASS by direct source and regression-design inspection.**

### 4. H1 boundary-authority isolation remains non-delegable

The graph filter still retains any graph finding containing:

`campaign_operations_h1_boundary_authority`

using the existing retained-edge rule.

Therefore a boundary-authority finding remains in `role_findings` and reaches the H1 failure path:

`diagnostic=H1A003`

before the post-upgrade H2 delegation branch can execute.

The focused pre-Phase-H migration test explicitly creates a hostile LOGIN, grants it `campaign_operations_h1_boundary_authority`, invokes the post-upgrade wrapper, and requires `diagnostic=H1A003`.

This is the required authority split:

- ordinary accepted post-H1 deployment graph: H2 may validate it, but only when H2 is proven installed;
- H1 boundary-authority reachability: always H1-owned and never delegated.

**Result: PASS.**

## Scope Preservation

The packaged `current_worktree.diff` shows no tracked modification to:

- `Database/migrations/055_campaign_operations_production_admission_foundation.sql`
- `Database/migrations/056_campaign_operations_h2_privilege_deployment_contract.sql`

The source-level H2-authority-gate correction is confined to the wrapper/test path. No new helper ACL grant or production role membership was introduced by this final correction.

The supplied migration 064 remains the previously accepted narrow compatibility overlay. Nothing in this H2-authority-gate correction broadens that two-entry helper ACL contract.

**Result: PASS.**

## Runtime Evidence Blocker

The source correction is internally consistent, but the uploaded reverification package is **not self-contained enough to execute the complete H2-present integration branch independently in this environment**.

The supplied `Scripts/CampaignOperationsH2DeploymentAudit.sh` directly depends on artifacts that are not included in the package, including at least:

- `Tests/CampaignOperationsPhaseH2DeploymentAudit.sql`
- `Database/migrations/057_campaign_operations_h2_production_bind_state_contract.sql`
- `Database/migrations/059_campaign_operations_direct_sql_readiness_boundary.sql`
- `Database/manifests/059_campaign_operations_direct_sql_boundary_explicit_acl.tsv`
- `Database/manifests/059_campaign_operations_direct_sql_boundary_manifest.sha256`
- `Scripts/CampaignOperationsH2ManifestValidator.sh`

The broader workflow test also requires a disposable PostgreSQL cluster/bootstrap environment and additional repository fixtures not contained in the archive.

Accordingly, I can independently prove from the supplied source that the original authority gap is removed, but I cannot truthfully claim that the H2-present, H2-absent, and hostile-boundary runtime branches were re-executed here against a live PostgreSQL fixture.

This is an **evidence/package blocker, not a newly discovered implementation defect**.

## Comparison to the Previous Independent Finding

The prior residual finding was:

`R-064-POSTGRAPH-001 — post-upgrade suppresses H1A003 before H2 authority is proven present`

That condition no longer exists.

The relevant ordering is now:

1. calculate exact checked-in migration-056 checksum;
2. at `post-upgrade`, establish exact 056 ledger identity;
3. if H2 is absent and an ordinary graph requires delegation, fail `H2A004`;
4. only if H2 is present, filter ordinary post-H1 graph findings from frozen-H1 interpretation;
5. retain boundary-authority graph findings;
6. fail retained H1 graph findings as `H1A003`;
7. invoke the H2 deployment audit when H2 is present;
8. allow enclosing H1 success only after the delegated H2 audit succeeds.

That is the required safe ordering.

## Final Assessment

The narrow correction requested after the previous reverification has been implemented correctly.

There is **no basis for another code correction** from the supplied source.

The next action should be runtime closure, not further editing: run the focused workflow in a fixture that has the complete H1/H2 cluster-role/bootstrap dependencies and capture proof of the three required branches:

1. exact 056 present + legitimate deployment graph -> delegated H2 PASS;
2. 056 absent + ordinary deployment graph -> deterministic H2A004 fail-closed;
3. H1 boundary-authority membership -> H1A003 rejection.

If those runtime branches pass, this item can move directly from `BLOCKED` to final `PASS`.

## Machine-Readable Summary

`FINAL_REVERIFICATION_STATUS=BLOCKED`

`SCOPED_CODE_CORRECTION=PASS`

`POST_UPGRADE_H2_AUTHORITY_PRECHECK=PASS`

`H2_ABSENT_FAIL_CLOSED=PASS_SOURCE`

`H2_PRESENT_DELEGATION=PASS_SOURCE`

`H1_BOUNDARY_ISOLATION=PASS_SOURCE`

`MIGRATION_055_UNCHANGED=PASS`

`MIGRATION_056_UNCHANGED=PASS`

`MIGRATION_064_SCOPE_PRESERVED=PASS`

`SHELL_SYNTAX=PASS`

`PACKAGE_SHA256_INTEGRITY=PASS`

`INDEPENDENT_RUNTIME_REEXECUTION=BLOCKED_PACKAGE_INCOMPLETE`

`RESIDUAL_CODE_DEFECT_COUNT=0`

`RUNTIME_EVIDENCE_BLOCKER_COUNT=1`
