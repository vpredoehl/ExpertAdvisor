---
title: "LSTM Campaign Operations Pre-Phase-H Helper ACL Final Residual Correction - Independent Reverification Findings"
document_type: "independent reverification"
status: "residual_blocker"
date: "2026-08-13"
---

# LSTM Campaign Operations Pre-Phase-H Helper ACL Final Residual Correction
## Independent Reverification Findings

## Verdict

**NOT YET CLOSED — one narrow residual deployment-audit integration blocker remains.**

The final residual correction successfully fixes the original migration-064 defect: migration 064 itself no longer rewrites or directly invokes the frozen migration-055 H1 deployment audit, and the intended two-entry helper ACL repair remains narrow and well sealed.

However, the current `Scripts/CampaignOperationsH1DeploymentAudit.sh` still evaluates the frozen H1 role-graph rule (`H1A003`) without post-H1 graph accommodation at `post-upgrade`. It only suppresses the accepted H2/H3/H4 LOGIN-to-production-capability graph at `pre-enablement`.

That means the focused migration test can pass `--stage post-upgrade` only because its disposable fixture does **not** reproduce the actual deployed post-H1 production LOGIN memberships. In the real deployed graph, which contains manager/enabler/disabler/dispatch-service LOGIN memberships to H1 protected production roles, the wrapper's early `role_findings` block will still raise `H1A003` before the migration-064 overlay is reached.

This is the remaining compatibility gap.

## Confirmed Correct

### 1. Migration 064 preserves the frozen migration-055 audit object

The current migration 064 contains no `pg_get_functiondef()` rewrite of
`campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)` and no direct
call requiring that historical audit to accept current post-H1 state.

The focused migration test explicitly hashes the frozen H1 audit definition
before and after repeated application of 064 and requires equality.

**Result: PASS.**

### 2. Exact two-entry ACL closure is retained

Migration 064 grants only:

- `campaign_operations_budget_administrator` -> `public.lock_campaign_operations_campaign(bigint)` EXECUTE
- `campaign_operations_request_acceptor` -> `public.lock_campaign_operations_campaign(bigint)` EXECUTE

The migration independently checks the campaign-lock helper remains:

- owned by `campaign_operations_h1_boundary_authority`;
- `SECURITY DEFINER`;
- pinned to `search_path=pg_catalog, public`.

**Result: PASS.**

### 3. Adjacent helper and authority expansion is rejected

Migration 064, the 064 compatibility manifest, and privilege regression collectively reject:

- PUBLIC helper EXECUTE;
- `pqxx` helper EXECUTE;
- dispatcher helper EXECUTE;
- Phase-5 helper EXECUTE;
- grant options;
- either Phase-A-G capability receiving any of the other four sealed lock helpers;
- pre-Phase-H LOGIN superuser status;
- pre-Phase-H LOGIN membership in `campaign_operations_owner`;
- pre-Phase-H LOGIN membership in Phase-H production roles;
- direct production transition authority.

**Result: PASS by static contract inspection.**

### 4. Partially applied predecessor and replay shape is correct

`GRANT EXECUTE` is replay-safe. The focused migration test deliberately applies
064 three times before writing the migration ledger row, with the second apply
representing the observed live partial predecessor where both grants already
exist but `schema_migrations` has no version 064 row.

The test also checks that migration 064 itself does not write its own ledger row.

**Result: PASS by test design/static inspection.**

### 5. Separate 064 overlay is narrow

`Database/manifests/064_campaign_operations_pre_phase_h_acl_manifest.sql`
treats migration 055 as the immutable base and checks only the explicit
post-H1 compatibility overlay. It does not redefine H1A003 or mutate the frozen
H1 function.

**Result: PASS.**

## Residual Blocker

### R-064-001 — `post-upgrade` H1 wrapper still rejects the legitimate post-H1 deployment role graph

`Scripts/CampaignOperationsH1DeploymentAudit.sh` computes recursive role-graph
findings for all H1 protected roles before migration-064 overlay handling.

The script contains a special accommodation only for:

```text
stage == pre-enablement
```

At that stage it filters ordinary post-H1 graph edges and delegates exact
deployment-role validation to the H2 audit.

There is no corresponding post-H1 accommodation for:

```text
stage == post-upgrade
```

Yet `Tests/CampaignOperationsPrePhaseHMigrationTests.sh` records migration 064
and then explicitly expects:

```text
CampaignOperationsH1DeploymentAudit.sh --stage post-upgrade
```

to succeed.

The disposable test fixture creates the production capability roles but does
not recreate the actual deployed production LOGIN memberships. Therefore its
`post-upgrade` success does not prove compatibility with the real current
deployment graph.

In the real graph already observed during deployment, examples include:

```text
campaign_operations_enabler_login
  -> campaign_operations_production_enabler
  -> campaign_operations_production_reader
  -> campaign_operations_scheduler_protocol_evidence_reader

campaign_operations_disabler_login
  -> campaign_operations_production_disabler
  -> campaign_operations_production_reader

campaign_operations_manager_login
  -> campaign_operations_production_dispatcher
  -> campaign_operations_production_phase5_transactional
  -> campaign_operations_production_reader
  -> campaign_operations_scheduler_protocol_evidence_reader

campaign_operations_dispatch_service_login
  -> campaign_operations_production_dispatch_service
  -> campaign_operations_production_phase5_transactional
```

Those are exactly the kinds of edges the unfiltered `post-upgrade`
`role_findings` logic classifies as H1A003.

### Why this matters

The original residual defect was that a frozen H1 audit was being asked to
validate a later post-H1 deployment graph. Migration 064 no longer does that,
which is correct. But the supported deployment wrapper still does so at
`post-upgrade`.

Therefore the correction is structurally sound at the migration/ACL layer but
the deployment-audit integration is not yet fully closed.

## Required Narrow Correction

Do **not** change migration 055 or H1A003 semantics.

Instead, update the H1 deployment wrapper so that post-H1 deployment stages
which legitimately occur after H2 role installation validate the later role
graph through the separately versioned H2 deployment authority, while retaining
the H1 boundary-owner isolation check.

The correction should be as narrow as the existing `pre-enablement` treatment.
Then extend the focused disposable test so its `post-upgrade` fixture includes
the exact accepted production LOGIN memberships before invoking the wrapper.
That proves the wrapper accepts the real deployment shape rather than an
artificially empty production-role graph.

## Evidence Quality / Tests

The supplied package shows:

- live read-only 064 overlay: PASS;
- live campaign-3 baseline unchanged;
- shell syntax checks: PASS;
- frozen-H1 rewrite/call scan: PASS;
- `git diff --check`: PASS;
- migration 055 unchanged;
- no Release rebuild required because the residual correction is SQL/scripts/tests/docs only.

The prior correction report also explicitly states that the full disposable
migration and real command regressions were not executed in that run. The
current source now contains stronger focused regression logic, but the uploaded
reverification package does not provide fresh runtime output proving that full
test against a fixture matching the actual post-H1 deployment role graph.

## Final Classification

**Residual blocker: R-064-001**

The helper ACL correction itself is accepted. Closure should wait for one more
very narrow correction to the deployment-audit stage handling and a regression
fixture that reproduces the real production LOGIN-to-capability graph.
