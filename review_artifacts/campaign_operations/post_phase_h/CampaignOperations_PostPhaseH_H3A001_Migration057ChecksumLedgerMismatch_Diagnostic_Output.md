---
title: "Campaign Operations Post-Phase-H H3A001 Migration 057 Checksum Ledger Mismatch Diagnostic"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PostPhaseH_H3A001_Migration057ChecksumLedgerMismatch_Diagnostic_Prompt.txt"
reasoning_effort: "medium"
model: "gpt-5.6-terra"
---

# Campaign Operations Post-Phase-H H3A001 Migration 057 Checksum Ledger Mismatch Diagnostic

## Verdict

H3 fails because migration 058 hard-codes the obsolete SHA-256 of migration 057 from `HEAD`, while the authoritative H1/H2 working-tree correction intentionally changed migration 057’s bytes. The disposable H3 ledger is correct for the current migration; it is not seeded incorrectly.

## Exact H3A001 check

Only [058_campaign_operations_h3_manager_run_once.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/058_campaign_operations_h3_manager_run_once.sql:10) emits `H3A001`.

At lines 10–16 it requires a `schema_migrations` row with all three exact values:

- `version = '057'`
- `filename = '057_campaign_operations_h2_production_bind_state_contract.sql'`
- `checksum = '4360f72635e5a6135ab355071f9dff8196b9c1612be2b87e13294a41d65c89af'`

It compares ledger fields only. It does not hash migration bytes, inspect a manifest, or use migration 057’s embedded predecessor checksum.

## Migration 057 identity

- Filename: `057_campaign_operations_h2_production_bind_state_contract.sql`
- Working-tree SHA-256: `ab58c6e7433bd81925497de9b4eb825fa0c8b908d202da61a261cfd4b18368fb`
- `HEAD` SHA-256: `4360f72635e5a6135ab355071f9dff8196b9c1612be2b87e13294a41d65c89af`
- Working-tree and `HEAD` bytes are not identical: exactly one line changed.
- Migration 057’s only embedded checksum is its predecessor-056 guard, at [057…sql:13](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/057_campaign_operations_h2_production_bind_state_contract.sql:13): `e382ea14…4310`. It is not a self-checksum.
- No manifest records a migration-057 checksum. The H2 manifest covers migration 056 plus its ACL TSV.
- There is no separate H3 test constant for 057; migration 058’s SQL literal is the effective H3 expected value.

## Checksum authority comparison

| Authority | Value |
|---|---|
| Current migration 057 bytes | `ab58c6e7…368fb` |
| `HEAD` migration 057 bytes | `4360f726…c89af` |
| Migration 057 embedded 056 predecessor checksum | `e382ea14…4310` |
| H2 audit’s 057 authority | Dynamic SHA-256 of current 057 bytes |
| H3/058 required 057 ledger checksum | `4360f726…c89af` |
| Disposable H3 ledger row 057 | `ab58c6e7…368fb` |

The first divergence is migration 058’s hard-coded `4360f726…c89af` versus the authoritative working-tree migration-057 SHA `ab58c6e7…368fb`.

## Disposable H3 ledger state

The H3 scripts first invoke the H2 workflow. That workflow:

1. Seeds all migration rows except 056 and 057 with current bytes.
2. Runs [migrate_lstm_db.sh](/Volumes/Developer%20SSD/ExpertAdvisor/migrate_lstm_db.sh:40), which hashes and installs 056 and 057.
3. Explicitly asserts that the 057 ledger row equals the current filename and SHA at [CampaignOperationsPhaseH2WorkflowTests.sh:81](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH2WorkflowTests.sh:81).

Immediately before attempting 058, the retained row is:

- Version: `057`
- Filename: `057_campaign_operations_h2_production_bind_state_contract.sql`
- Checksum: `ab58c6e7433bd81925497de9b4eb825fa0c8b908d202da61a261cfd4b18368fb`
- `applied_at`: runner default (`now()` at the 057 installation)
- Origin: actual disposable migration-runner execution, not a seed/restore row.

Both H3 scripts remove only the synthetic 058 ledger row; they preserve 057. See [compatibility setup](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH3CompatibilityTests.sh:28) and [058 execution setup](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH3Migration058ExecutionTests.sh:28).

## First point of divergence

The H1REG027-authoritative correction changed migration 057’s 056-predecessor guard from `4616e269…25cc` to `e382ea14…4310`, documented in [the correction record](/Volumes/Developer%20SSD/ExpertAdvisor/CampaignOperations_PostPhaseH_H1REG027_ManifestCatalogConsistencyCorrection_Output.md:25).

That necessary one-line migration-057 change changed its file SHA from `4360f726…c89af` to `ab58c6e7…368fb`. Migration 058’s predecessor check was not propagated.

## Root cause

This is not a bad H3 fixture, inherited database state, manifest artifact, or H2 ledger mutation.

It is a stale migration-058 prerequisite literal: 058 validates the old migration-057 identity (`HEAD`) instead of the current authoritative H1/H2 migration-057 identity.

## Smallest correction target

[058_campaign_operations_h3_manager_run_once.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/058_campaign_operations_h3_manager_run_once.sql:15): update only its required predecessor-057 checksum literal to the authoritative current SHA `ab58c6e7433bd81925497de9b4eb825fa0c8b908d202da61a261cfd4b18368fb`.

Do not alter migration 057 or the disposable ledger to fit the stale expected value.

## Files that would need modification

- `Database/migrations/058_campaign_operations_h3_manager_run_once.sql`

No test fixture or manifest change is evidenced as necessary for this narrow consistency correction; the H3 test scripts derive their installed migration checksums dynamically.

## Files that must not be modified

- `Database/migrations/057_campaign_operations_h2_production_bind_state_contract.sql`
- H1 manifests and migration 055
- Migration 056, H2 audit semantics, and H2 workflow predecessor/version fixture semantics
- H3 fixture ledger setup
- Production database contents, ownership, grants, or deployment state

## Recommended validation sequence after correction

1. Focused 058 predecessor-checksum/ledger regression.
2. H2 workflow.
3. H3 compatibility.
4. H3 migration-058 execution regression.
5. Independent reverification, including a fresh disposable migration ledger check.

## Independent reverification readiness

Not ready until migration 058’s stale 057 checksum literal is reconciled. The correction is narrowly isolated and preserves the successful H1/H2 state.

`H3A001_ROOT_CAUSE=stale_migration_058_preflight_checksum_for_authoritatively_modified_057`