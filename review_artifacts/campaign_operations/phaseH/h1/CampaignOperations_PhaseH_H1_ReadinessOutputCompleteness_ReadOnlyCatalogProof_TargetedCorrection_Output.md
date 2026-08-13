---
title: "Campaign Operations Phase H H1 Readiness Output Completeness Read-Only Catalog Proof Targeted Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_ReadinessOutputCompleteness_ReadOnlyCatalogProof_TargetedCorrection_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H H1 Readiness Output Completeness Read-Only Catalog Proof Targeted Correction

## 1. Executive Summary

Corrected the readiness read-only proof by extending `ReadinessEvidenceJson` with deterministic, readiness-relevant PostgreSQL catalog evidence. No production readiness semantics or migration changed.

## 2. Remaining Defect Corrected

The before/after proof now covers the catalog inputs directly used by readiness, not only application tables.

## 3. Exact Catalog Dependency Trace

- `pg_constraint`: readiness reads matching `CHECK` constraints on `public.campaign_operations_completion_event`, using `contype` and `pg_get_constraintdef(oid)` to derive Completion proof version.
- `pg_roles`: `campaign_operations_has_explicit_role_v1` resolves the `session_user` role and names of recursively reached roles.
- `pg_auth_members`: the helper recursively follows `member → roleid` membership edges.
- Additional role-helper catalogs: none. The helper does not explicitly read role attributes, `pg_authid`, `pg_db_role_setting`, or membership option flags.

Readiness invokes the helper for eight roles: four required production roles and four prohibited roles.

## 4. Stable Catalog Snapshot Design

Updated [CampaignOperationsPhaseH1RepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:39>) to add `catalog` evidence:

- Completion constraints: schema, relation, constraint name, type, normalized `pg_get_constraintdef` output; ordered by schema/relation/name.
- Roles: all names in the helper’s recursive closure from `session_user`; ordered by role name.
- Memberships: all edges whose member is in that closure, represented by member/granted role names; ordered by both names.

It excludes OIDs, physical state, unrelated roles/edges, role attributes, and membership options because readiness does not use them.

## 5. Read-Only Proof Update

The existing authentic flow remains:

1. Load disposable H1 state.
2. Capture `ReadinessEvidenceJson`.
3. Run real `RunProductionReadinessCommand` (`REPEATABLE READ, READ ONLY`).
4. Capture again.
5. Assert exact equality.

Application evidence remains covered: migration, scheduler, enablement/audit, admission, attempts/audit, requests, completion/audit, and reconciliation state. Catalog evidence is now included.

## 6. Role-Helper Coverage Assessment

Added a rollback-scoped transitive role mutation test:

- Creates an intermediate NOLOGIN role.
- Grants `campaign_operations_production_reader` to it.
- Grants it to `campaign_manager_login`.
- Verifies both recursive membership edges are present in the snapshot.
- Verifies the snapshot changes and loaded readiness changes `readerMember` to true.

This proves the catalog snapshot captures the helper’s real recursive inputs.

## 7. Completion Constraint Coverage Assessment

The existing disposable constraint mutation now additionally proves that:

- replacing the Completion version check changes the catalog snapshot;
- the snapshot’s stored normalized definition equals `pg_get_constraintdef`;
- readiness reports `completion_nested_v2_proof_version=2`.

## 8. Tests Added/Updated

Updated only the pass-owned file:

- [Tests/CampaignOperationsPhaseH1RepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1RepositoryTests.cpp:39>)

## 9. Tests Executed and Exact Results

Passed:

- `clang++ -std=c++20 -Wall -Wextra -Werror ... CampaignOperationsPhaseH1RepositoryTests.cpp ...`
- `Tests/CampaignOperationsPhaseH1MigrationTests.sh` — exit `0`
  - H1 repository/service tests passed.
  - Authentic readiness command before/after proof passed.
  - Phase 1–5 repository/service/completion regression passed.
  - H1 deployment audit and scheduler migration/policy checks passed.
- `git diff --check -- Tests/CampaignOperationsPhaseH1RepositoryTests.cpp` — clean.

## 10. Regression Assessment

No production readiness implementation changed. Existing blockers, output, isolation level, role graph behavior, hydration behavior, and H1 contracts remain intact.

## 11. Migration / Checksum / Manifest Assessment

Migration 055, checksums, and manifests were not changed by this pass. The H1 manifest validation within the disposable suite passed.

## 12. Deferred Checks and Exact Reason

No shared Release build or CLI run: active scheduler and training workers were detected, so relinking or executing the shared `LSTM_Release` was unsafe. The focused isolated compile and disposable PostgreSQL suite were used instead.

No broad final evidence sweep was run, per scope. The suite’s existing evidence-graph report remains outside this correction’s scope.

## 13. Files Changed

Pass-owned change:

- `Tests/CampaignOperationsPhaseH1RepositoryTests.cpp`

## 14. Worktree / Staging Assessment

No files were staged, unstaged, committed, reset, or discarded. The pre-existing broad H1 staged/unstaged worktree was preserved.

`git diff --stat` reports the existing unstaged aggregate:

```text
9 files changed, 1345 insertions(+), 158 deletions(-)
```

The relevant status entry remains:

```text
AM Tests/CampaignOperationsPhaseH1RepositoryTests.cpp
```

## 15. Remaining Findings

No remaining readiness-output-completeness proof defect found. Independent reverification was not performed.

## 16. Final Disposition

READY_FOR_READINESS_OUTPUT_COMPLETENESS_READONLY_PROOF_INDEPENDENT_REVERIFICATION