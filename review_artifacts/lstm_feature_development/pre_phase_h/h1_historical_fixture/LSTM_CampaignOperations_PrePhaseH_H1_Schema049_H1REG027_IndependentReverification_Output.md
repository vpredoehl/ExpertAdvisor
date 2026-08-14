---
title: "LSTM Campaign Operations Pre-Phase-H H1 Schema-049 H1REG027 Independent Reverification"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignOperations_PrePhaseH_H1_Schema049_H1REG027_IndependentReverification_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-terra"
---

# LSTM Campaign Operations Pre-Phase-H H1 Schema-049 H1REG027 Independent Reverification

## 1. Verdict

**PASS WITH REQUIRED CORRECTION** for the schema-049 fixture approach.

The fixture direction is correct, but H1REG027 is a **test-bootstrap/prerequisite reconstruction defect**, not a missing `pqxx` privilege and not a frozen-manifest defect. The harness applies migrations 050–052 as `campaign_manager_login`; the authoritative migration path uses `vjp`, which is the frozen H1 owner contract.

No files or database state were changed during this review.

## 2. Exact H1REG027 Mismatch

The comparison tuple fields are:

`tuple_kind, object_class, object_identity, owner, grantee, privilege, grant_option, origin, default_scope, object_kind`

Expected tuples, derived from `055_campaign_operations_h1_column_acl.tsv` plus the inventory owner:

```text
column_acl | column | public.experiment_scheduler_protocol.cutover_completed_at       | vjp | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_completed_by       | vjp | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_executable_path    | vjp | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_process_evidence   | vjp | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_state              | vjp | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.failure_diagnostic          | vjp | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.updated_at                  | vjp | pqxx | UPDATE | false | explicit | "" | r
```

Observed catalog tuples:

```text
column_acl | column | public.experiment_scheduler_protocol.cutover_completed_at       | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_completed_by       | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_executable_path    | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_process_evidence   | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.cutover_state              | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.failure_diagnostic          | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
column_acl | column | public.experiment_scheduler_protocol.updated_at                  | campaign_manager_login | pqxx | UPDATE | false | explicit | "" | r
```

Every field other than `owner` matches. The raw ACLs are `{pqxx=w/campaign_manager_login}`; their grantor is not part of the canonical comparison tuple.

The reconciler filters observed rows using a key that includes `owner` before comparing privileges. Therefore all seven observed rows are discarded, producing the generic `missing-observed-tuples` error. This is not a grantee, privilege, grant-option, column-set, relation-identity, or ACL-origin mismatch.

## 3. Root Cause

`experiment_scheduler_protocol` does not exist in schema 049. Migration 052 creates it and grants the seven column UPDATE privileges:

- [052 migration](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql:7) creates the table.
- [052 migration](/Volumes/Developer%20SSD/ExpertAdvisor/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql:375) grants `pqxx` SELECT and the seven UPDATE columns.
- Migrations 050–055 contain no ownership transfer for this table.
- Migration 055 only adds its evidence-version column and revokes a separate evidence-owner role; it does not change this owner or these grants.

The local harness initializes PostgreSQL as `campaign_manager_login` and directly runs 050–052 as that role ([test](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:85), [prerequisites](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1MigrationTests.sh:120)). PostgreSQL consequently assigns the new table to `campaign_manager_login`.

The documented migration runner defaults to the local migration identity (`$USER`, with `vjp` fallback), and the historical post-052/054 evidence has this table owned by `vjp`. The frozen H1 inventory and SQL manifest consistently require `vjp` ([inventory](/Volumes/Developer%20SSD/ExpertAdvisor/Database/manifests/055_campaign_operations_h1_object_inventory.tsv:4), [SQL manifest](/Volumes/Developer%20SSD/ExpertAdvisor/Database/manifests/055_campaign_operations_h1_acl_manifest.sql:11)).

The catalog code correctly captures `pg_class.relowner` and uses it in the canonical tuple ([generator](/Volumes/Developer%20SSD/ExpertAdvisor/Scripts/CampaignOperationsH1AclCatalogGenerator.py:83)). The comparator correctly includes owner in its pre-comparison filter ([reconciler](/Volumes/Developer%20SSD/ExpertAdvisor/Scripts/CampaignOperationsH1AclCatalog.py:183)).

## 4. Historical / Authority Analysis

- **Schema 049:** no `experiment_scheduler_protocol` object exists. The recovered dump SHA-256 matches the supplied value: `ebcfa556…655fac31`.
- **Migration 052:** is authoritative for creating the table and the seven grants.
- **Migrations 053–055:** do not establish this table’s owner or replace the seven grants.
- **Frozen H1 contract:** requires owner `vjp`; do not change it to accommodate a test running the prerequisite creation under a different role.

The old `LSTM_latest.dump` masked the defect. Its archive already contains `experiment_scheduler_protocol` owned by `vjp`; `CREATE TABLE IF NOT EXISTS` in migration 052 then does not transfer ownership. Restoring that mutable post-049 state made the test appear to reconstruct the intended ownership when it actually inherited it.

## 5. Smallest Correct Correction

Do not modify the H1 ACL manifests or weaken the ACL contract.

1. Keep the schema-049 restore.
2. Run prerequisite migrations 050–052 under `SET ROLE vjp` in the disposable cluster, while retaining `campaign_manager_login` as the bootstrap/superuser connection and retaining its role for 053–055.
3. Add a focused post-052 assertion that `public.experiment_scheduler_protocol` is owned by `vjp`, before H1 installation.
4. Preserve the existing seven grants exactly.

This corrects the test reconstruction principal rather than relying on a post-H1 object owner. It also avoids changing already-applied migration 052 bytes or the H1 checksum chain.

## 6. Residual Moving-State Audit

One direct H1-assurance dependency remains:

- [CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh](/Volumes/Developer%20SSD/ExpertAdvisor/Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh:135) still restores `Database/backups/LSTM_latest.dump`.

It should use the fixed schema-049 fixture and reconstruct through 050–054 under the same authoritative migration identity, or use a separately tracked immutable schema-054 fixture. It currently has no schema-version or digest guard.

The main H1 migration harness no longer references `LSTM_latest.dump`, but its current fixture work is incomplete:

- `LSTM_schema_049.dump` is ignored by `.gitignore` and not tracked by Git LFS.
- `LSTM_schema_049.dump.json` is untracked.
- Its `backup_path` incorrectly says `Database/backups/LSTM_latest.dump`.
- The harness checks only a grep match for `"schema_version": "049"` and does not verify the dump hash.

Necessary deterministic fixture guards: track the dump through Git LFS, correct the metadata path, record the supplied SHA-256 in metadata, and verify exact JSON fields plus the dump SHA-256 before restore.

## 7. Reverification Plan

After correction:

```bash
Scripts/CampaignOperationsH1ManifestValidator.sh
python3 Tests/CampaignOperationsPhaseH1AclCatalogIndependenceTests.py
bash Tests/CampaignOperationsPhaseH1MigrationTests.sh
```

The final command is the required gate. It should pass H1REG027 with seven reconciled scheduler-column tuples and retain all other ACL assertions.

Then correct the protected-function preflight fixture dependency and run:

```bash
bash Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh
```

## 8. Files Inspected

- `Tests/CampaignOperationsPhaseH1MigrationTests.sh`
- `Tests/CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh`
- `Scripts/CampaignOperationsH1AclCatalog.py`
- `Scripts/CampaignOperationsH1AclCatalogGenerator.py`
- `Scripts/CampaignOperationsH1CatalogPipeline.py`
- `Scripts/CampaignOperationsH1ManifestValidator.sh`
- `migrate_lstm_db.sh`
- migrations 050–055
- H1 object, explicit ACL, column ACL, and SQL ACL manifests
- `Database/backups/LSTM_schema_049.dump` and metadata JSON
- preserved cluster’s observed catalog TSV/payload and read-only catalog queries
- Git history including `23b412e`, `957fcbd`, and `491e755`

Working tree remains unchanged by this review:

```text
 M Tests/CampaignOperationsPhaseH1MigrationTests.sh
?? Database/backups/LSTM_schema_049.dump.json
?? review_artifacts/lstm_feature_development/scheduler_orphan_recovery/LSTM_Scheduler_TrainOrphanCheckpointRecovery_IndependentReverification_Output.md
```

`git diff --stat`: one pre-existing test file changed, `6 insertions(+), 1 deletion(-)`.