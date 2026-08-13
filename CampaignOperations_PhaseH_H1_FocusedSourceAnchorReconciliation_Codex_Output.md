---
title: "Campaign Operations Phase H H1 Focused Source Anchor Reconciliation"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PhaseH_H1_FocusedSourceAnchorReconciliation_Codex_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# Campaign Operations Phase H H1 Focused Source Anchor Reconciliation

Stopped without modifying files: two genuine source-text changes were found, so anchors/digests were not updated.

Genuine changes:

`PHASEH-INCREMENTS` — `docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md`

- Frozen: H4 is excluded pending separate acceptance.
- Current: H4 is accepted by ADR-0020 as an external, deployment-owned supervisor, with new operational details and limits.

```diff
- Excluded from the initial implementation. It requires a separate operational ADR...
+ ADR-0019 excluded H4 from its initial implementation. ADR-0020 later accepts
+ H4 solely as external, deployment-owned ... supervision ...
```

`VOLUMEXII-MIGRATION` — `docs/architecture/Volume_XII_Database.md`

```diff
- implemented by Phase H Step 1
+ implemented by Phase H H1
- Migration 055 must add no login membership, backfill no request...
+ Migration 055 adds no login membership, backfills no request...
```

All checked clauses:

`ADR19-AUTHORITY`, `ADR19-MANAGER-BOUNDARY`, `ADR19-SCHEDULER-BOUNDARY`, `ADR19-SCOPE`, `ADR19-COMPATIBILITY`, `ADR19-H1-BOUNDARY`, `ADR19-VERIFICATION`, `ADR19A-IDENTIFIER`, `ADR19A-SEALED-OWNER`, `ADR19A-DEFINERS`, `ADR19A-REPLAY`, `ADR19A-LOCK-ORDER`, `ADR19A-MIGRATION`, `ADR19A-HISTORY`, `ADR19B-FAILURE-CONTRACT`, `ADR19B-ROLE-IDENTITY`, `ADR19B-ROLE-GRAPH`, `ADR19B-OWNERSHIP`, `ADR19B-ALL-SCHEMA`, `ADR19B-ACL`, `ADR19B-DEPLOYMENT-AUDIT`, `ADR19B-RESTORE`, `ADR19B-LOCK-EVIDENCE`, `ADR19B-NEGATIVE-AUTHENTICITY`, `ADR19B-HISTORICAL`, `ADR19B-INERTNESS`, `PHASEH-LOCK-ORDER`, `PHASEH-PRIVILEGES`, `PHASEH-MIGRATION`, `PHASEH-INCREMENTS`, `PHASEH-VERIFICATION`, `VOLUMEXII-MIGRATION`, `VOLUMEXII-SEALED`, `VOLUMEXII-AUDIT`, `VOLUMEXII-TESTING`, `VOLUMEXII-PERMISSIONS`.

Verified pure anchor drift candidates were found for `PHASEH-LOCK-ORDER`, `PHASEH-PRIVILEGES`, `PHASEH-MIGRATION`, `PHASEH-VERIFICATION`, `VOLUMEXII-SEALED`, `VOLUMEXII-AUDIT`, `VOLUMEXII-TESTING`, and `VOLUMEXII-PERMISSIONS`, but none were changed because the genuine changes require stopping.

- Changed anchors: none.
- `canonical_excerpt` / `canonical_clause_digest`: unchanged for every row.
- `AUTHORITY_DIGEST`: unchanged, `34be85602ebf1c492530ebf17955aaaa2e407c1ee8d7cbc569627f6d8934df4f`.
- TSV SHA-256 remains the same value.

Validation executed:

- `git diff --check` — exit 0
- `python3 -m py_compile Scripts/CampaignOperationsH1EvidenceAuthority.py` — exit 0
- `python3 Tests/CampaignOperationsPhaseH1EvidenceAuthorityTests.py` — exit 1; first failure remains `H1A104 key=PHASEH-LOCK-ORDER ... altered-canonical-excerpt`
- `bash -n Tests/CampaignOperationsPhaseH1MigrationTests.sh` — exit 0
- Migration test script not run, because reconciliation is blocked by genuine normative-source drift.

Focused-task diff summary: no files modified.