---
title: "LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade Role-Graph - Independent Reverification Findings"
document_type: "independent reverification"
status: "residual_blocker"
date: "2026-08-13"
source_package: "LSTM_CampaignOperations_PrePhaseH_HelperACL_PostUpgradeRoleGraph_Reverification_Input.tar.gz"
---

# LSTM Campaign Operations Pre-Phase-H Helper ACL Post-Upgrade Role-Graph
## Independent Reverification Findings

## Verdict

**NOT YET CLOSED — the two-entry ACL correction remains sound, but one narrow role-graph enforcement defect remains in the `post-upgrade` wrapper path.**

The latest correction fixes the previously identified real-deployment compatibility problem in the positive case: `Scripts/CampaignOperationsH1DeploymentAudit.sh` no longer automatically rejects ordinary post-H1 deployment LOGIN edges at `post-upgrade`, and it preserves explicit rejection of reachability involving `campaign_operations_h1_boundary_authority`.

However, the filtering is applied **unconditionally for every `post-upgrade` audit before the script establishes that migration 056/H2 authority is installed**. The later H2 audit is conditional on a valid migration-056 ledger row. Consequently, a database with no migration 056 can enter `post-upgrade`, have arbitrary non-boundary H1 protected-role graph edges suppressed from H1A003, and receive no H2 graph validation at all.

That is broader than the required correction. The required compatibility rule was to delegate legitimate later role-graph evolution to the separately versioned H2 authority **when H2 is installed**, while retaining the frozen H1 role-graph rule otherwise.

**Independent closure status: RESIDUAL BLOCKER.**

## Confirmed Correct

### 1. Migration 064 remains a narrow two-entry ACL repair

`Database/migrations/064_campaign_operations_pre_phase_h_helper_acl_reconciliation.sql` grants only:

- `campaign_operations_budget_administrator` -> `public.lock_campaign_operations_campaign(bigint)` EXECUTE;
- `campaign_operations_request_acceptor` -> `public.lock_campaign_operations_campaign(bigint)` EXECUTE.

The migration also verifies that the helper remains owned by `campaign_operations_h1_boundary_authority`, remains `SECURITY DEFINER`, and retains `search_path=pg_catalog, public`. It rejects PUBLIC, `pqxx`, dispatcher, Phase-5 helper execution, grant options, adjacent helper grants, pre-Phase-H LOGIN production membership, owner membership, and superuser status.

**Result: PASS by direct source inspection.**

### 2. Frozen migration-055 audit is no longer rewritten by migration 064

The supplied migration 064 contains no `pg_get_functiondef()` rewrite and no direct invocation of `campaign_operations_h1_deployment_audit_v1(text,boolean,boolean)`. The focused test hashes that frozen function definition before and after repeated 064 application and requires equality.

**Result: PASS by source/test inspection.**

### 3. The 064 overlay remains separate from frozen H1 authority

`Database/manifests/064_campaign_operations_pre_phase_h_acl_manifest.sql` treats 055 as the immutable base and checks only the explicit compatibility overlay. It rejects adjacent helper expansion and pre-Phase-H authority escalation.

**Result: PASS.**

### 4. The latest correction preserves H1 boundary-authority isolation

The wrapper's post-H1 graph filtering retains every graph finding that contains `campaign_operations_h1_boundary_authority`:

```bash
awk '!/^graph:/ || /campaign_operations_h1_boundary_authority/'
```

The focused test adds a hostile LOGIN, grants it `campaign_operations_h1_boundary_authority`, and requires `post-upgrade` to fail with `diagnostic=H1A003`.

**Result: PASS by structural inspection.**

### 5. Positive post-H1 deployment graph coverage was materially improved

`Tests/CampaignOperationsPrePhaseHMigrationTests.sh` now creates synthetic enabler, disabler, manager, and dispatch-service LOGINs with the deployment capability memberships observed in the deployed environment before exercising `--stage post-upgrade`.

This corrects the earlier weakness where the disposable fixture had an artificially empty production LOGIN graph.

**Result: PASS as a regression-design improvement.**

## Residual Blocker

### R-064-POSTGRAPH-001 — `post-upgrade` suppresses H1A003 before H2 authority is proven present

The relevant wrapper ordering is:

- lines 169-179: if stage is `pre-enablement` **or `post-upgrade`**, remove every `graph:` finding that does not mention `campaign_operations_h1_boundary_authority`;
- lines 181-191: evaluate the remaining H1 role findings;
- lines 262-278: only afterwards, query `schema_migrations` for migration 056 and call `CampaignOperationsH2DeploymentAudit.sh` if that exact ledger row exists.

Therefore, at `post-upgrade` with migration 056 absent:

1. ordinary protected-role graph findings are removed;
2. H1A003 does not evaluate them;
3. `h2_deployment_installed` is false;
4. H2 audit is not invoked;
5. the wrapper can continue without any authority validating those edges.

This is a real enforcement gap, not merely missing runtime evidence.

### The focused test currently masks this gap instead of detecting it

The focused test deliberately creates the accepted H2-style deployment LOGIN graph at lines 19-84, but its stated predecessor is migration 055 and it explicitly expects no 064 ledger row before the test. It does **not** establish a migration-056 ledger row before the `post-upgrade` wrapper invocation at lines 172-175.

Thus its `deployed_login_graph=PASS` case demonstrates that the wrapper tolerates these edges **without H2 authority being installed**, which is the opposite of the intended delegation condition.

A stronger negative regression is required: on a 055+064 database with **no 056 ledger row**, add a non-boundary protected-role edge (for example a LOGIN -> `campaign_operations_production_reader`) and require `--stage post-upgrade` to fail H1A003. Then establish the authentic 056/H2 state and prove that the same accepted deployment graph succeeds only through H2 validation.

## Required Narrow Correction

Do not change migration 055, migration 064, or the two-entry ACL overlay.

Change only the `post-upgrade` role-graph accommodation so that ordinary H1 graph findings are filtered **only after the wrapper has proven migration 056 is installed and will invoke the H2 deployment audit**. Equivalent safe shapes include:

- determine `h2_deployment_installed` before H1 role filtering, and apply the non-boundary graph filter at `post-upgrade` only when it is true; or
- retain ordinary H1A003 enforcement at `post-upgrade` by default, then use a specifically H2-authorized branch when the exact migration-056 ledger identity is present.

`pre-enablement` may continue using its existing separately versioned H2 path if that stage's prerequisite contract already guarantees H2 presence.

The focused test should then cover both sides:

1. **056 absent:** non-boundary protected role edge -> H1A003 failure;
2. **056 present:** exact accepted deployment LOGIN graph -> H2 audit success;
3. **056 present + hostile H1 boundary edge:** H1A003 failure;
4. **056 present + unsupported deployment graph edge / ADMIN OPTION:** H2 audit failure.

## Evidence Limitation in the Uploaded Reverification Package

The package reports `Database/migrations/059_campaign_operations_h4_continuous_manager.sql` as missing and does not include `Scripts/CampaignOperationsH2DeploymentAudit.sh`. Because the corrected wrapper explicitly delegates to that H2 script when migration 056 is installed, those omitted artifacts prevent independent source-level confirmation that the current H2 audit accepts the complete later deployed graph, including dispatch-service/H4 evolution.

This evidence limitation would by itself make final runtime closure unproven. It does **not** cause the verdict above; the unconditional `post-upgrade` filtering defect is independently visible in the supplied wrapper and focused test.

## Other Checks

- `git_diff_check.txt`: empty, so the packaged worktree passes `git diff --check`.
- Latest Codex correction output explicitly reports focused/H1/H2 runtime regressions as `BLOCKED`; therefore no fresh runtime PASS evidence was supplied that overrides the structural finding.
- The missing migration-059 file in the package is consistent with the packaging terminal output and should be included in the next reverification package if it is part of current H2/H4 role-graph authority.

## Final Assessment

The helper ACL compatibility correction itself is now well constrained and should be preserved. The remaining defect is narrower: the deployment wrapper has moved from **rejecting legitimate post-H1 graph evolution** to **tolerating that evolution too early**, before the later authority is proven present.

One small wrapper/test correction should be sufficient. After that correction, rerun the focused migration test with both 056-absent and authentic 056-present fixtures, then independently reverify the H2 delegation path.

**FINAL_REVERIFICATION_STATUS=RESIDUAL_BLOCKER**
**RESIDUAL_ITEM_COUNT=1**
**MIGRATION_064_TWO_ENTRY_ACL=PASS**
**FROZEN_H1_AUDIT_IMMUTABILITY=PASS**
**H1_BOUNDARY_ISOLATION=PASS**
**POST_UPGRADE_H2_DELEGATION_GATING=FAIL**
