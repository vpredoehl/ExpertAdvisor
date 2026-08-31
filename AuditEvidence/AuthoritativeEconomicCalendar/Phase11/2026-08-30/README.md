# PCE production-readiness evidence — Phase 11

This directory is a no-write review package generated from commit
`ccf4a6499e4555b8ec3c94fdcc0239c55a7b5e8b` and the read-only production
catalog observed on 2026-08-30.  It does not attest that migration 088 or any
release-actual row has been deployed to production.

## Decisions

- Migration 088: `BLOCKED`.  The migration itself is compatible with the
  current production schema and passed transactional apply/rollback tests, but
  production is registered only through migration 084.  The established
  `migrate_lstm_db.sh` workflow would apply unrelated migrations 085, 086, and
  087 together with 088.  A migration-088-only deployment gate therefore does
  not yet exist.  Do not bypass the migration ledger or register predecessors
  without applying and independently verifying them.
- PCE evidence payload: `NEEDS_REVIEW`.  All 183 certified initials are
  import-eligible and the payload is deterministic, but migration 088 is not
  deployed and the existing importer intentionally refuses production writes.
  A separately reviewed production execution authority/path is still needed.
- Width-75 production training: `NOT_APPROVED_IN_THIS_PHASE`.

## Evidence files

- `pce-production-import-manifest.jsonl`: 183 canonical migration-088 payload
  rows, ordered by availability, event ID, and observation ID.
- `pce-production-readiness.json`: row-level current catalog reconciliation,
  classifications, exclusions, coverage, and manifest hash.
- `pce-production-import.sql`: the exact append-only transaction validated in
  disposable databases.  It is review evidence, not authorization to execute
  direct SQL against production.

Manifest SHA-256:

```text
ae151f05e287a4c2cf0d3ee0e8928fe8cab3f8e08c16fb9c75dbf7d85fbd8ce8
```

Two independent production-catalog runs produced byte-identical manifest,
audit, and SQL files.  A first disposable import inserted 183 rows.  The
second run reproduced the same manifest bytes/hash, classified 183
duplicate-identical rows, generated an empty `BEGIN`/`COMMIT` transaction,
and mutated nothing.

## Coverage and reconciliation

- Catalog classifications: 168 `READY`, 15 `MISSING_CONSENSUS`, and zero
  missing events, identity mismatches, incompatible consensus rows, invalid
  candidate semantics, existing conflicts, or existing duplicates.
- All 183 identity-valid actuals remain import-eligible.  Missing consensus is
  not a defect in authoritative actual evidence; it makes surprise unavailable
  rather than zero for those 15 events.
- Historical 2010-01-01 through 2025-01-01: 179 events, 167 initials,
  93.2961% actual coverage, 152 jointly usable rows, 84.9162% usable coverage.
- Post-2025: 17 events, 16 initials, 94.1176% actual coverage, 16 jointly
  usable rows, 94.1176% usable coverage.
- Certified range: 2010-02-01 through 2026-06-25.
- Exclusions: 12 inexact `less than 0.1%` observations and one combined-month
  October/November 2025 observation.  None was converted.

## Future migration-088 operator gate

Do not execute this gate while production remains at migration 084.

Preconditions:

1. Repository is clean at an explicitly approved commit.
2. Migrations 085–087 have been independently approved, applied, and verified,
   so `schema_migrations` ends at 087 and 088 is absent.
3. No conflicting schema maintenance is active; the production role and
   `pqxx` role match the audited privilege model.
4. Create the repository-convention schema+data rollback point when required:

   ```sh
   ./DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release \
     --backup-database \
     --backup-output=Database/backups/LSTM_latest.dump
   ```

5. Independently approve the manifest SHA-256 above.  The manifest is not
   imported during the migration gate.

Once the ledger ends at 087, the established migration command is:

```sh
LSTM_DB_HOST=127.0.0.1 \
LSTM_DB_NAME=LSTM \
LSTM_DB_ADMIN_USER=vjp \
./migrate_lstm_db.sh
```

The expected result is exactly one apply (`088`) and all earlier migrations
skipped with matching checksums.  The migration runner wraps migration 088 and
its ledger registration in one transaction.  On any SQL error the transaction
rolls back.

Immediately verify read-only:

```sql
BEGIN READ ONLY;
SELECT version, filename, checksum, applied_at
FROM schema_migrations WHERE version='088';
SELECT to_regclass('public.economic_event_release_actual'),
       to_regclass('public.economic_event_feature_release_actual'),
       to_regprocedure('public.validate_economic_event_release_actual()'),
       to_regprocedure('public.reject_economic_event_release_actual_mutation()');
SELECT has_table_privilege('pqxx','public.economic_event_release_actual','SELECT'),
       has_table_privilege('pqxx','public.economic_event_release_actual','INSERT'),
       has_table_privilege('pqxx','public.economic_event_release_actual','UPDATE'),
       has_table_privilege('pqxx','public.economic_event_release_actual','DELETE'),
       has_table_privilege('pqxx','public.economic_event_feature_release_actual','SELECT');
SELECT count(*) FROM economic_event_release_actual;
COMMIT;
```

Expected privileges are `true, false, false, false, true`; expected initial
row count is zero.  Stop after verification.  Do not combine the migration and
PCE evidence import gates.

## Future PCE import operator gate

Do not use direct production SQL merely because the payload in this directory
validated.  The existing authoritative importer has an intentional disposable
database guard.  A future reviewed increment must authorize a narrow
production execution path through that workflow (or an equivalent approved
service/repository transaction) without weakening the guard generally.

Required preconditions are:

1. Migration 088 is independently verified and its table is empty.
2. This exact manifest hash and expected count of 183 are approved.
3. Re-running the read-only reconciliation yields 183 import-eligible rows,
   zero unexpected existing rows, zero conflicts, zero identity mismatches,
   and byte-identical manifest output.
4. The import authority has only the separately reviewed INSERT capability
   needed for the exact columns; normal `pqxx` remains read-only.
5. The approved workflow executes the reviewed payload in one transaction,
   with no update, delete, upsert, or conflict suppression, and fails the whole
   transaction on any unexpected conflict.

Post-import read-only verification must check:

```sql
BEGIN READ ONLY;
SELECT count(*) AS rows,
       count(*) FILTER (WHERE publication_state='initial' AND revision_sequence=0) AS initials,
       count(*) FILTER (WHERE publication_state='revision') AS revisions,
       min(available_at), max(available_at)
FROM economic_event_release_actual;
SELECT source_agency, count(*)
FROM economic_event_release_actual GROUP BY source_agency ORDER BY source_agency;
SELECT count(*) FILTER (
           WHERE source_artifact_sha256 IS NULL OR source_provenance IS NULL
       ) AS null_provenance
FROM economic_event_release_actual;
SELECT count(*) AS feature_rows,
       count(DISTINCT economic_event_id) AS feature_events
FROM economic_event_feature_release_actual;
COMMIT;
```

Expected PCE-only results are 183 rows, 183 initials, zero revisions, BEA=183,
2010-02-01 through 2026-06-25, zero null provenance, and 183 distinct
feature-view events.  The workflow must also compare every persisted
event/revision/source-observation identity to the approved manifest and repeat
the 152 historical plus 16 post-2025 joint semantic-compatibility audit.

An import transaction failure rolls back all attempted inserts.  After commit,
release-actual evidence is immutable: routine rollback is not UPDATE or DELETE.
Bad committed evidence requires a separately reviewed immutable-evidence
correction design.
