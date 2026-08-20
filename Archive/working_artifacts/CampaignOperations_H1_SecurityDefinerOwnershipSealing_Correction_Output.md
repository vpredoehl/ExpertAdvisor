---
title: "Campaign Operations H1 SECURITY DEFINER Ownership Sealing Correction"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_H1_SecurityDefinerOwnershipSealing_Correction_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-luna"
---

# Campaign Operations H1 SECURITY DEFINER Ownership Sealing Correction

Implemented the narrow ACL correction.

1. Root cause

Migration 055 sealed four H1 relations but live ACL state lacked the explicit read grants already required by its contract. Owner-defined readers and the deferred completion-boundary trigger therefore failed under `campaign_operations_owner`.

2. Files changed

- [063 additive migration](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/063_campaign_operations_h1_owner_read_acl_reconciliation.sql:1>)
- [Top-level admission regression](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsAdmissionTopLevelTests.sh:1>)
- [H1 ACL regression](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsH1OwnerReadAclTests.sql:1>)
- [Pre-Phase-H negative privilege assertions](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignOperationsPrePhaseHPrivilegeTests.sql:82>)
- [Database documentation](</Volumes/Developer SSD/ExpertAdvisor/Database/README.md:116>)

Existing unrelated worktree changes were preserved.

3. Exact correction

Migration 063 restores `SELECT` only for `campaign_operations_owner` on:

- `campaign_operations_dispatch_attempt`
- `campaign_operations_dispatch_audit_reference_event`
- `campaign_operations_completion_event`
- `campaign_operations_completion_audit_reference_event`

Ownership remains with `campaign_operations_h1_boundary_authority`; no mutation privileges or ordinary-role privileges were added.

4. H1 authority manifest decision

`enforce_campaign_operations_completion_boundary_consistent()` remains owned by `campaign_operations_owner` and is not added to the literal H1 function manifest.

Migration 055 explicitly defines the former owner’s narrow read ACL contract and intentionally excludes this read-only consistency function from the sealed-function list. Its hardened search path and `SECURITY DEFINER` status remain unchanged.

5. Remaining function audit

Same ownership/dependency defect:

- `campaign_operations_completion_blockers`
- `campaign_operations_completion_classification`
- `campaign_operations_completion_evidence_text`
- `campaign_operations_future_actions_allowed`
- `enforce_campaign_operations_completion_boundary_consistent`

All are fixed by the restored owner-read ACL.

Correct as-is:

- `campaign_operations_completion_identity_valid`
- lifecycle cancellation helper
- budget, control, cancellation, reconciliation, and reservation enforcement functions
- `guard_campaign_operations_completion_boundary_mutation`

No indiscriminate ownership migration was performed.

6. Tests and build

Passed:

- Release `xcodebuild`
- Campaign Operations Phase 2, 4, and 5 CLI tests
- Phase-H connection-routing test
- H1 owner-read ACL SQL regression
- Pre-Phase-H privilege regression
- Real top-level admission regression

The H1 deployment audit still reports pre-existing production-login role-graph edges in this development database. No role membership was changed.

7. Admission evidence

Against `LSTM` development state:

- Materialization 2: 2 members
- Exactly one operational campaign persisted: ID 2
- Exactly one `campaign_created` audit reference persisted
- Deferred boundary trigger committed successfully
- Replay returned `existing_identical`
- Experiment snapshot unchanged: 330 rows
- Worker-attempt snapshot unchanged: 620 rows
- No scheduler process was running or started
- Pre-Phase-H login remained non-superuser without H1 production roles or completion-table `SELECT`

The database is now intentionally no longer pristine for materialization 2, so the literal operator command should only be rerun against an untouched development/test state:

```bash
./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
  --campaign-operations-admit 2 \
  --campaign-operations-actor vjp \
  --campaign-operations-reason \
  "Admit approved CADCHF H4 Donchian-20 enabled versus zero-ablation materialization 2 into Campaign Operations" \
  --yes
```

Migration 063 checksum:

```text
1f99017a11b39f5ff58bbf9d24b297d9b52308435cc4d073b75f7d39ed28386b
```

`git diff --check` passed.