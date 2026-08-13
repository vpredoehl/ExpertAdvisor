---
title: "Campaign Operations Post-Phase-H H1REG027 Diagnostic Review"
document_type: "architecture review"
status: "final"
generated_from: "CampaignOperations_PostPhaseH_H1REG027_DiagnosticReview_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# Campaign Operations Post-Phase-H H1REG027 Diagnostic Review

# H1REG027 Diagnostic Result

## Classification

REAL MANIFEST/CATALOG DRIFT

## Executive Finding

H1REG027 is pre-existing manifest/catalog drift. The seven required `pqxx` column-level UPDATE grants exist exactly, but the frozen H1 manifest expects `experiment_scheduler_protocol` owner `campaign_manager_login`; the restored/migrated catalog owns it as `vjp`. The comparator therefore filters out all seven observed tuples and raises `missing-observed-tuples`.

## Current-vs-Pre-H1A006 Comparison

| Migration 055 bytes | Suite exit | H1REG027 | Captured scheduler ACLs |
|---|---:|---|---|
| Current H1A006-corrected | 1 | `missing-observed-tuples` | 7 column UPDATE tuples, owner `vjp` |
| `.pre_h1a006_fix` original | 1 | `missing-observed-tuples` | Identical 7 tuples, owner `vjp` |

Both isolated runs reached the catalog pipeline. After omitting volatile capture fields (capture/run/cluster/query-execution identities), the complete observed ACL tuple files were identical.

Thus H1REG027 occurs with the original migration bytes as well as the corrected bytes.

## Missing Tuple

The manifest generates these seven expected tuples, each with owner `campaign_manager_login`:

- `public.experiment_scheduler_protocol.cutover_state`
- `...cutover_completed_at`
- `...cutover_completed_by`
- `...cutover_executable_path`
- `...cutover_process_evidence`
- `...failure_diagnostic`
- `...updated_at`

Exact tuple shape for each:

```text
column_acl | column | <identity> | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
```

Observed catalog state for each is identical except owner:

```text
column_acl | column | <identity> | vjp | pqxx | UPDATE | false | explicit | "" | r
```

This is not a missing PostgreSQL privilege. `reconcile_acl_catalog()` filters observed rows by an object key containing the owner; expected owner `campaign_manager_login` matches zero observed rows owned by `vjp`, then raises `missing-observed-tuples`.

## Source Trace

1. `Tests/fixtures/CampaignOperationsH1Requirements.tsv` declares `H1-ACL-SCHEDULER-COLUMNS` as fixture `H1REG027`.
2. `Tests/fixtures/CampaignOperationsH1Fixtures.tsv` maps `H1REG027` to `GEN-ACL-MANIFEST`.
3. `Tests/fixtures/CampaignOperationsH1Traceability.tsv`, evidence obligations, runtime records, and graph edges consistently map H1REG027 to this ACL-catalog evidence path.
4. `Tests/CampaignOperationsPhaseH1MigrationTests.sh` installs/replays 055, then invokes `Scripts/CampaignOperationsH1CatalogPipeline.py` before later negative tests.
5. The pipeline runs `CampaignOperationsH1AclCatalogGenerator.py`, which captures `pg_class.relowner`, `relacl`, and `pg_attribute.attacl`.
6. `CampaignOperationsH1AclCatalog.py` derives expected column tuples from `055_campaign_operations_h1_column_acl.tsv`, obtaining the table owner from `055_campaign_operations_h1_object_inventory.tsv`.
7. That inventory names `campaign_manager_login`; catalog capture reports `vjp`. The owner-bearing filter produces zero candidate observed tuples and raises the reported error.

## PostgreSQL ACL Evidence

`public.experiment_scheduler_protocol` in both disposable runs:

```text
pg_class.relowner = vjp
pg_class.relacl   = {vjp=arwdDxtm/vjp,pqxx=r/vjp}
```

- Table-level `pqxx UPDATE`: absent.
- Table-level `pqxx SELECT`: present.
- Column-level `pqxx UPDATE`: present for all seven expected columns.
- Missing expected columns: none.
- Extra `pqxx UPDATE` column privileges: none.
- Column ACL representation: explicit `pg_attribute.attacl`, each `{pqxx=w/vjp}`.
- Grantee: `pqxx`.
- Grantor: `vjp`.
- Privilege: `UPDATE`.
- Grant option: false.
- Origin: explicit, not default ACL expansion.

The schema-only backup independently confirms the same pre-H1 catalog contract: `experiment_scheduler_protocol` is owned by `vjp` and already contains the same seven `pqxx` UPDATE grants. The test restores with `--no-privileges`, preserving ownership, then migration 052 reissues the grants.

Migration 052 establishes these exact seven column grants. Migration 055 adds `scheduler_evidence_contract_version` and revokes only the separate scheduler-evidence-owner role; it neither changes table ownership nor modifies `pqxx` scheduler UPDATE grants.

## Relationship to H1A006 Correction

The current H1A006 diff only changes protected-function preflight/sealing logic:

- a predecessor function ACL array;
- `pg_proc.proacl` validation;
- a REVOKE on one function from `campaign_operations_dispatcher`.

It contains no `experiment_scheduler_protocol` ownership or ACL operation.

Therefore:

- The predecessor ACL array cannot affect non-function ACL capture.
- The sealing-time dispatcher REVOKE cannot affect H1REG027.
- H1A006 did not cause this failure.
- In the actual H1 migration-suite comparison, original bytes also reach and fail H1REG027. It is not newly exposed by this correction within this harness; it was pre-existing and independently reproducible.

## Recommended Next Correction

Update the frozen manifest to match the actual established scheduler protocol ownership (`vjp`), rather than changing the database owner.

Required files:

- `Database/manifests/055_campaign_operations_h1_object_inventory.tsv` — change this object’s owner to `vjp`.
- `Database/manifests/055_campaign_operations_h1_acl_manifest.sql` — change the matching object-manifest owner to `vjp`.
- `Database/manifests/055_campaign_operations_h1_manifest.sha256` — regenerate the manifest-set digest.
- `Database/migrations/055_campaign_operations_production_admission_foundation.sql` — update only its embedded `H1_MANIFEST_DIGEST_SHA256` marker.

A `/tmp` manifest-only simulation with those four corresponding changes passed manifest validation, reconciled H1REG027 as `equal` with seven tuples, and produced no scheduler rows from the authoritative SQL manifest.

Do not add `ALTER TABLE ... OWNER TO campaign_manager_login` as a narrow test fix: it changes the established catalog contract and would require broader ACL normalization.

## Regression Scope After Correction

Minimum focused commands:

```bash
Scripts/CampaignOperationsH1ManifestValidator.sh
bash Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh
bash Tests/CampaignOperationsPhaseH1MigrationTests.sh
```

Before production activation resumes, rerun the relevant downstream H1/H2/H3 coverage:

```bash
bash Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh
bash Tests/CampaignOperationsPhaseH2PrivilegeDeploymentTests.sh
bash Tests/CampaignOperationsPhaseH2WorkflowTests.sh
bash Tests/CampaignOperationsPhaseH3CompatibilityTests.sh
bash Tests/CampaignOperationsPhaseH3Migration058ExecutionTests.sh
```

## Safety Statement

No repository files were modified. No production database or persistent PostgreSQL cluster was modified. Both PostgreSQL clusters used for diagnosis were created under `/tmp` and stopped after inspection.

Repository status remained:

```text
 M Database/migrations/055_campaign_operations_production_admission_foundation.sql
 M Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh
?? Database/migrations/055_campaign_operations_production_admission_foundation.sql.pre_h1a006_fix
?? Tests/.CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.debug.sh
?? Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh.pre_h1a006_fix
```

`git diff --check` passed. `git diff --stat` reports the pre-existing local H1A006 work only: 2 files, 184 insertions, 4 deletions.

H1REG027_DIAGNOSTIC=MANIFEST_CATALOG_DRIFT