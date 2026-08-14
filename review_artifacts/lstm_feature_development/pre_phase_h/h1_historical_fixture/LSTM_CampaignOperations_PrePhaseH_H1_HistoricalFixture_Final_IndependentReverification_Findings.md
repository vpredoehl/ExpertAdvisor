---
title: "LSTM Campaign Operations Pre-Phase-H H1 Historical Fixture Final Independent Reverification Findings"
document_type: "independent reverification"
status: "final"
verdict: "PASS"
date: "2026-08-14"
---

# LSTM Campaign Operations Pre-Phase-H H1 Historical Fixture
# Final Independent Reverification Findings

## Verdict

**PASS.**

The submitted bundle is sufficient to close the Pre-Phase-H H1 historical-fixture correction as implemented.

The final corrected H1 fixture is internally consistent with the stated historical contract:

- the historical bootstrap is pinned to the immutable schema-049 predecessor;
- schema-049 metadata and dump digest are fail-closed;
- prerequisite roles are recreated as inert historical roles rather than production-capability shortcuts;
- migrations 050–052 run under the historical `vjp` ownership principal;
- the protected-function preflight no longer follows the mutable latest backup;
- the H2 checksum audit defect is corrected without weakening the H1 role-graph or ACL audit;
- the H1 link commands now use the production materialization repository and its required implementation closure;
- focused repository adapters are suppressed only when the real production materialization repository is linked;
- no migration-055, frozen-manifest, or production privilege-boundary relaxation is present in the submitted correction.

## 1. Historical schema fixture

`CampaignOperationsPhaseH1MigrationTests.sh` now uses:

- `Database/backups/LSTM_schema_049.dump`
- `Database/backups/LSTM_schema_049.dump.json`

and rejects the fixture unless metadata exactly matches:

- `schema_version = 049`
- `git_commit = 939e126`
- `created_at = 2026-07-31T03:19:28Z`

It also verifies the dump SHA-256:

`ebcfa55680f4db5fb136527ceff1039f97a139fcdcbd4574186ef341655fac31`

before `pg_restore`.

This removes the historical H1 assurance path from the moving `LSTM_latest.dump` state.

## 2. Historical principal / ownership reconstruction

The disposable cluster recreates only inert predecessor roles needed to restore schema 049 and advance through migrations 050–054.

The database is created with owner `vjp`, and migrations 050–052 are executed with `SET ROLE vjp`. The harness then explicitly verifies that `public.experiment_scheduler_protocol` remains owned by `vjp`.

This is a historically faithful bootstrap correction, not an ACL bypass. It does not pre-create migration-055 authority roles or grant additional runtime capabilities.

## 3. Protected-function historical fixture

`CampaignOperationsPhaseH1ProtectedFunctionPreflightTests.sh` is likewise pinned to the schema-049 fixture, validates the same metadata and dump digest, and reconstructs through migrations 050–054.

This closes the previously identified remaining historical dependency on `LSTM_latest.dump`.

The correction report records the protected-function preflight as passing after this change.

## 4. H1REG027 / catalog behavior

The prior bootstrap correction report records that the corrected historical path reaches migration 055, produces:

`H1_ACL_CATALOG_V3_OK rows=41`

and that `Campaign Operations Phase H1 migration tests passed`, including H1REG027.

No frozen H1 ACL manifest or migration-055 catalog contract was changed to obtain that result.

The supplied current harness retains the original fail-closed H1A001-H1A007 behavior and the catalog reconciliation path.

## 5. H2 checksum post-upgrade audit correction

`CampaignOperationsH1DeploymentAudit.sh` now compares the migration-056 ledger checksum using the already-computed Bash variable `h2_expected_checksum` directly inside the SQL string.

This removes both observed defects:

- PostgreSQL seeing the literal `:'h2_checksum'` in a context where psql substitution did not occur;
- the intermediate Bash `h2_checksum` unbound-variable failure.

The change affects only migration-056 presence detection. It does not relax role findings, H1 boundary-authority checks, H2 delegation requirements, or H1 ACL validation.

## 6. Link-closure correction

The H1 workflow-lock and broad repository link commands now explicitly link the production implementation closure needed by `CampaignOperationsService.cpp` after it began calling `RecommendationCampaignMaterializationSchemaExists(transaction)`.

The submitted harness includes the materialization repository plus materialization, evaluation, scoring, approval, review, and planning implementations, with the broad link also including the required recommendation-review implementation.

`CampaignOperationsRepositoryTests.cpp` uses `CAMPAIGN_OPERATIONS_USE_PRODUCTION_MATERIALIZATION_REPOSITORY` to disable only the pre-existing hand-written materialization repository adapters when the production repository is intentionally linked.

This is preferable to stubs or weak-symbol workarounds because the H1 test binaries exercise the same implementation dependency surface used by production.

The accompanying link-closure correction report records:

- both focused links passing with `-Wall -Wextra -Werror`;
- `bash -n` passing;
- `git diff --check` passing;
- the full H1 suite passing, including the broad Phase 1–5 repository/service/completion regression and final reference-graph checks.

## 7. Earlier failed rerun

The bundled `CampaignOperations_PrePhaseH_H1_HistoricalFixture_FinalClosure_Rerun3_Output.txt` is **not** the final successful runtime. It is an earlier diagnostic run that still failed at link time on `ReviewRecommendationCampaignPlan`.

That failure does not contradict final closure because the later link-closure correction adds the missing campaign-review/planning dependency and the correction report records a subsequent successful full H1 run.

The earlier output is therefore useful negative evidence showing the dependency chain was exposed incrementally, but it must not be cited as the final passing run.

## 8. Evidence-quality note

The bundle contains the final correction report stating that the subsequent full H1 suite passed, but it does not contain the raw stdout/stderr transcript of that final successful suite run. Therefore the final runtime result is supported by the correction artifact rather than independently re-parsed from the successful raw log in this bundle.

This is **not a functional blocker** to accepting the correction, because the current source supplied in the bundle independently confirms the final link closure and historical-fixture changes, and the correction artifact records the successful final gate.

For the strongest archival provenance, the successful raw H1 run or the Codex correction transcript can be archived alongside these findings when convenient.

## 9. Final closure assessment

No residual technical defect identified in the prior historical-fixture, moving-backup, H2 checksum, H1REG027, or link-closure chain remains open in the submitted correction.

**Final disposition: ACCEPT / PASS.**

The Pre-Phase-H H1 historical fixture is suitable for archival closure and staging with its immutable schema-049 fixture metadata and the associated independent-reverification findings.
