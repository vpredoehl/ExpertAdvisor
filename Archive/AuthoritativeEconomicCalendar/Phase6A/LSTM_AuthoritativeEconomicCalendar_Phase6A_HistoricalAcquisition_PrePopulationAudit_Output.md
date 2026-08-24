# Phase 6A: Full Historical Acquisition + Pre-Population Audit — Continuation Report

## 1. Executive summary

This continuation recovered and retained the interrupted Phase 6A work, completed the remaining Census acquisition, repaired narrowly evidenced BEA/Census/Federal Reserve historical parser and enumeration defects, added deterministic per-entry and whole-batch manifest auditing, and exercised the accepted corpus through the common import service in a disposable PostgreSQL database.

The non-BLS acquisition enumerated 2,007 first-party artifacts. All 2,007 hashes validated; 1,993 artifacts parsed; 14 failed closed; 12 parsed candidates failed whole-batch identity/timestamp validation; and 1,981 occurrences entered the final accepted manifests. The existing BLS dataset contains another 813 exact occurrences, for 2,794 accepted/persisted occurrences across the supported inventory.

The final accepted non-BLS corpus imported 1,981 rows into a fresh disposable database. An exact repeat produced 1,981 unchanged rows, zero inserts, zero updates, and zero rejects. Repository conflict, rollback, same-time/different-family, and cross-agency isolation regressions passed. The required LSTM Debug build succeeded.

Phase 6A is nevertheless not ready for Phase 6B. Blocking facts are: no committed reproducible BLS acquisition/import corpus exists; DOL/ETA's current authoritative archive exposes only 46 releases from 2025 rather than 2010-current; five authoritative Census release-ID collision groups required 10 otherwise parseable entries to be excluded; two distinct Federal Reserve FOMC statements share an indistinguishable date-only family timestamp and were excluded; and 14 authoritative artifacts remain intentionally failed closed.

## 2. Recovery from interrupted run

Recovery was performed before edits. The interrupted source changes, untracked audit harness, downloaded artifacts, manifests, audits, accepted manifests, logs, and binaries under `DerivedData/Development/Phase6A` were inspected and retained.

The interrupted transcript had stated that disposable-database creation failed and left no database. Current catalog state was authoritative and showed `ea_economic_calendar_phase6a_20260824_001`, owned by `pqxx`, with migration 072 applied and zero `economic_event` rows. It was reused/recreated only as a disposable database and was dropped at completion.

Recovered work included BEA/Census/DOL archive caching, Federal Reserve annual press and Beige Book archive enumeration, intermeeting statement discovery, minutes/statement classification, expanded historical Beige Book URL support, narrow Census historical header handling, Federal Reserve minutes meeting association, conservative date-only semantics, and the first version of the manifest audit tooling.

## 3. Initial continuation repository state

- Working directory: `/Volumes/Developer SSD/ExpertAdvisor`
- Branch: `lstm-feature-development`
- HEAD: `cb3bb723e0e879630f877ed811bc2002a075e516`
- Initial tracked modifications: 9 files, 396 insertions, 26 deletions.
- Initial untracked files: `Scripts/audit_economic_event_manifest.py`, `Tests/EconomicEventManifestAudit.cpp`, and `Tests/EconomicEventManifestAudit.sh`.
- Initial `git diff --check`: clean.
- Existing Phase 6A artifacts: approximately 408 MB at recovery.
- Existing audits: BEA 383/358 accepted, Census retail 198/198, DOL/ETA 46/46, Federal Reserve 407/398.
- Existing disposable DB: present, schema applied, zero rows.

The interrupted checkpoint had not completed five Census families, whole-corpus collision validation, complete disposable import, final regressions, the required build command, database cleanup, or this report.

## 4. Phase 1-5 supported family inventory

| Agency | Supported repository families | Path |
|---|---|---|
| BLS | CPI, EMPLOYMENT, EMPLOYMENT_ANNUAL, JOLTS, PPI | Existing persisted Phase 1 dataset/read repository; no committed importer or corpus |
| DOL/ETA | WEEKLY_CLAIMS | DOL/ETA manifest adapter and common importer |
| BEA | GDP_ADVANCE, GDP_SECOND, GDP_THIRD, PERSONAL_INCOME_OUTLAYS | BEA manifest adapter and common importer |
| Census | CONSTRUCTION_SPENDING, DURABLE_GOODS_ADVANCE, MANUFACTURERS_ORDERS, NEW_RESIDENTIAL_CONSTRUCTION, NEW_RESIDENTIAL_SALES, RETAIL_SALES_ADVANCE | Census manifest adapter and common importer |
| Federal Reserve | BEIGE_BOOK, FOMC_MINUTES, FOMC_STATEMENT | Federal Reserve manifest adapter and common importer |

No additional families were invented. In particular, Board discount-rate minutes are not classified as FOMC minutes.

## 5. Existing Phase 6A work recovered and retained

The continuation retained the existing modifications in all acquisition scripts and adapters. It verified the Federal Reserve archive test that discovers the May 9, 2010 intermeeting statement, the exclusion of unrelated Board minutes, the Federal Reserve date-only behavior, Census historical release-header behavior, and all existing hash-bearing artifacts and manifests.

Existing valid downloads were reused. Census partial directories were resumed rather than replaced. Existing source bytes were hash checked, existing `pdftotext` output was regenerated and byte-compared, and deterministic manifest content was checked.

## 6. New/final Phase 6A changes

- Completed Census acquisition for all six supported families, adding retry/backoff, explicit resume, request delay, deterministic reuse, extraction comparison, and manifest comparison behavior.
- Narrowed Census parsing for exact historical OCR/header variants, the 2013 shutdown combined releases, the 2018 delayed-release identity, and date-only `FOR IMMEDIATE RELEASE` headers.
- Narrowed BEA parsing for literal UTF-8 nonbreaking space, historical numeric-only release IDs in the release-ID slot, punctuation/header variants, colon/`for` PIO headings, combined-month PIO, and shutdown-era `initial`/`updated` GDP labels.
- Completed Federal Reserve archive enumeration/classification and retained strict statement/minutes/Beige Book separation.
- Separated provenance, parse, and validation status in the manifest audit.
- Added whole-batch duplicate source-ID and agency/family/timestamp validation before accepted-manifest emission.
- Added deterministic manifest-audit unit coverage.
- Produced combined Census agency manifest/audit so cross-family identity collisions are visible before import.

## 7. Acquisition/enumeration architecture by agency

- BLS: represented by the existing authoritative persisted dataset and read-only repository path. No second manifest architecture was introduced. The checkout still has no committed BLS acquisition/import corpus.
- DOL/ETA: enumerates immutable links exposed by `https://oui.doleta.gov/unemploy/archive.asp`; no guessed weekly URL construction.
- BEA: paginates the official BEA news archive by supported product ID and filters canonical first-party occurrence URLs by year.
- Census: enumerates the six official family-specific historical-release indexes and acquires the linked PDFs; text is deterministically extracted with a recorded extractor identity.
- Federal Reserve: enumerates annual Federal Reserve press indexes for FOMC statements/minutes and the official Beige Book archive/year indexes; canonical occurrence types drive acquisition and parsing.

## 8. Historical acquisition boundaries

The requested lower boundary was 2010-01-01. Current acquisition was performed on 2026-08-24 and stops at the latest occurrence exposed by each authoritative archive, not at a fabricated schedule date.

| Agency/family | First accepted release | Last accepted release |
|---|---:|---:|
| BLS (all) | 2010-01-08 | 2026-08-13 |
| BEA GDP | 2010-01-29 | 2026-06-25 |
| BEA PIO | 2010-02-01 | 2026-06-25 |
| Census families | 2010-02-12 to 2010-03-04 | 2026-06-25 to 2026-08-18 |
| DOL/ETA weekly claims | 2025-01-02 | 2025-12-31 |
| Federal Reserve | 2010-01-06 | 2026-08-19 |

DOL/ETA is not a 2010-current boundary: its live first-party archive exposed only 2025 occurrence links. BLS is current in production but not reproducible from committed repository artifacts.

## 9. Full corpus counts

Manifest corpus:

- Enumerated occurrences/artifacts: 2,007
- Authoritative source artifacts acquired: 2,007
- Acquisition failures among enumerated links: 0
- Provenance hashes validated: 2,007
- Parsed candidates: 1,993
- Parse failures: 14
- Validation failures after parsing: 12
- Accepted non-BLS occurrences: 1,981
- Existing accepted/persisted BLS occurrences: 813
- Combined accepted/persisted supported inventory: 2,794
- First-pass accepted-manifest inserts: 1,981
- Second-pass unchanged rows: 1,981
- Final accepted-import conflicts: 0
- Duplicate source-ID groups in enumerated candidates: 5 groups / 10 candidates
- Cross-family source-ID collision groups: 3 groups / 6 candidates
- Exact semantic duplicate Census publication groups: 2 groups / 4 candidates
- Possible same-family/same-time Federal Reserve duplicate group: 1 group / 2 distinct statements

Accepted counts by family:

| Agency/family | Accepted |
|---|---:|
| BLS/CPI | 199 |
| BLS/EMPLOYMENT | 199 |
| BLS/EMPLOYMENT_ANNUAL | 17 |
| BLS/JOLTS | 199 |
| BLS/PPI | 199 |
| BEA/GDP_ADVANCE | 63 |
| BEA/GDP_SECOND | 63 |
| BEA/GDP_THIRD | 64 |
| BEA/PERSONAL_INCOME_OUTLAYS | 191 |
| CENSUS/CONSTRUCTION_SPENDING | 189 |
| CENSUS/DURABLE_GOODS_ADVANCE | 196 |
| CENSUS/MANUFACTURERS_ORDERS | 193 |
| CENSUS/NEW_RESIDENTIAL_CONSTRUCTION | 191 |
| CENSUS/NEW_RESIDENTIAL_SALES | 191 |
| CENSUS/RETAIL_SALES_ADVANCE | 198 |
| DOL_ETA/WEEKLY_CLAIMS | 46 |
| FEDERAL_RESERVE/BEIGE_BOOK | 124 |
| FEDERAL_RESERVE/FOMC_MINUTES | 134 |
| FEDERAL_RESERVE/FOMC_STATEMENT | 138 |

Combined accepted/persisted counts by release year:

| Year | Count | Year | Count |
|---:|---:|---:|---:|
| 2010 | 162 | 2019 | 159 |
| 2011 | 166 | 2020 | 170 |
| 2012 | 169 | 2021 | 167 |
| 2013 | 163 | 2022 | 168 |
| 2014 | 164 | 2023 | 166 |
| 2015 | 163 | 2024 | 168 |
| 2016 | 169 | 2025 | 196 |
| 2017 | 169 | 2026 | 107 |
| 2018 | 168 |  |  |

Machine-readable non-BLS detail is in `DerivedData/Development/Phase6A/corpus-audit-summary.json`, `corpus-accepted-counts-by-year.tsv`, and `corpus-failures.tsv`.

## 10. Provenance/hash audit

All 2,007 non-BLS manifest entries passed artifact and source-artifact hash verification. Census and Beige Book text entries retain both derived-text and authoritative-PDF hashes plus extractor identity. HTML entries require source and parsed artifact provenance to match. A deliberately corrupted fixture was reported as provenance failure with parse/validation `not_run`; it was not silently dropped.

## 11. Timestamp-confidence audit

Final accepted non-BLS corpus:

- `exact`: 1,911
- `date_only`: 70
- `reconstructed`: 0

BLS contributes 813 `exact` rows. Combined supported inventory is therefore 2,724 exact and 70 date-only.

The 70 date-only rows are 21 Beige Books, 48 FOMC statements, and one Census release. Date-only rows retain no fabricated source time and use the established conservative next-New-York-midnight causal boundary. Explicit EST/EDT contradictions fail closed.

## 12. Coverage/gap audit

Bounds below are `[accepted lower bound, enumerated authoritative-artifact upper bound]`; `unknown` means the current source/path cannot establish an upper bound.

| Agency/family | Bound | Assessment |
|---|---:|---|
| BLS/CPI | [199, unknown] | Persisted through current boundary; no committed reproducible importer/corpus |
| BLS/EMPLOYMENT | [199, unknown] | Same blocker |
| BLS/EMPLOYMENT_ANNUAL | [17, unknown] | Same blocker |
| BLS/JOLTS | [199, unknown] | Same blocker |
| BLS/PPI | [199, unknown] | Same blocker |
| BEA/GDP families combined | [190, 190] | Archive enumeration and accepted corpus agree |
| BEA/PIO | [191, 193] | One timezone contradiction; one timestamp-less superseding data update |
| Census/CONSTRUCTION_SPENDING | [189, 192] | Three authoritative ID-collision participants excluded |
| Census/DURABLE_GOODS_ADVANCE | [196, 196] | Fully accepted |
| Census/MANUFACTURERS_ORDERS | [193, 196] | One unpublished shutdown report and two ID-collision participants |
| Census/NEW_RESIDENTIAL_CONSTRUCTION | [191, 195] | One identity contradiction and three collision/duplicate participants |
| Census/NEW_RESIDENTIAL_SALES | [191, 194] | One timezone contradiction and two duplicate participants |
| Census/RETAIL_SALES_ADVANCE | [198, 198] | Fully accepted |
| DOL_ETA/WEEKLY_CLAIMS | [46, unknown] | Official archive currently exposes only 2025; 2010-2024 and 2026 unresolved |
| Federal Reserve/BEIGE_BOOK | [124, 133] | Nine explicit timezone contradictions |
| Federal Reserve/FOMC_MINUTES | [134, 134] | Fully accepted |
| Federal Reserve/FOMC_STATEMENT | [138, 140] | Two distinct date-only statements collide at the conservative timestamp |

Suspicious historical gaps remain visible in family/year counts, particularly BEA 2014/2015/2019/2025, Census shutdown/delayed-release years, and the DOL/ETA 2010-2024 absence. They were not filled by assumptions.

## 13. Parse/validation failure inventory

Parse failures (14):

- BEA, 2019 PIO September: authoritative `EST` contradicts New York DST.
- BEA, 2025 PIO data update September: superseding data-update page has no authoritative release time.
- Census manufacturers orders August 2013: official page states the report was not released during shutdown and supplies no release time.
- Census new residential construction April 2015 artifact: May 2015 release carries a 2014 identity, failing identity-year validation.
- Census new residential sales January 2019 artifact: March 2019 header says `EST` during DST.
- Federal Reserve Beige Books on 2010-04-14, 2010-06-09, 2011-06-08, 2019-01-16, 2020-03-04, 2021-01-13, 2021-03-03, 2022-10-19, and 2023-04-19: explicit authoritative timezone abbreviations contradict New York offset.

Whole-batch validation failures (12):

- Five Census source IDs are duplicated: `cb11-131`, `cb13-194`, `cb13-204`, `cb15-126`, and `cb23-25` (10 candidates).
- `cb13-194` and `cb13-204` are duplicated archive artifacts for combined shutdown releases and also collide by agency/family/timestamp.
- Federal Reserve `monetary20140917a` and `monetary20140917c` are distinct FOMC statement pages with the same date-only causal timestamp.

All failures remain in JSON/TSV audits with diagnostics and are excluded from accepted manifests.

## 14. Federal Reserve scheduled/intermeeting coverage

The archive enumerated 140 FOMC statements, 134 FOMC minutes, and 133 Beige Books. The final accepted counts are 138, 134, and 124 respectively.

2010 includes 11 statements rather than only the eight scheduled meetings. The May 9, 2010 intermeeting occurrence `federal_reserve:monetary20100509a` is present, parsed, validated, and accepted. 2020 similarly contains 10 statement occurrences. Enumeration tests prove that statement/action links are retained while unrelated Board discount-rate minutes are not classified as FOMC minutes.

The two September 17, 2014 statement pages are both authoritative and distinct, but both are date-only and share the same family. They remain explicit validation failures rather than receiving a fabricated release time or invented family.

## 15. Disposable database creation and schema setup

- Database: `ea_economic_calendar_phase6a_20260824_001`
- Administrative creator/dropper: existing authorized local role `vjp` (`CREATEDB`/superuser).
- Database owner/application role: `pqxx`; no role privileges were altered.
- Schema: only `Database/migrations/072_economic_event.sql`.

An initial schema attempt made the table admin-owned and the application role correctly received `permission denied`; no rows were written. The database was immediately recreated, and migration 072 was applied as owner `pqxx`. Production LSTM was never used as staging.

## 16. First import results

Using the existing common CLI/service/repository path:

| Agency | Inserted | Unchanged | Updated | Rejected |
|---|---:|---:|---:|---:|
| BEA | 381 | 0 | 0 | 0 |
| DOL/ETA | 46 | 0 | 0 | 0 |
| Census | 1,158 | 0 | 0 | 0 |
| Federal Reserve | 396 | 0 | 0 | 0 |
| Total | 1,981 | 0 | 0 | 0 |

Final database identity check returned `1981|1981` for total rows versus distinct `(source_agency, source_event_id)` identities.

## 17. Idempotent second import results

| Agency | Inserted | Unchanged | Updated | Rejected |
|---|---:|---:|---:|---:|
| BEA | 0 | 381 | 0 | 0 |
| DOL/ETA | 0 | 46 | 0 | 0 |
| Census | 0 | 1,158 | 0 | 0 |
| Federal Reserve | 0 | 396 | 0 | 0 |
| Total | 0 | 1,981 | 0 | 0 |

The row count remained 1,981.

## 18. Conflict/atomicity results

The pre-batch raw-candidate trial proved that a 195-row Census batch containing an existing source-ID conflict produced zero inserts and 195 `batch_rolled_back_due_to_conflict` dispositions. Internal duplicate-ID Census batches and the same-family/timestamp Federal Reserve batch were rejected before repository writes.

The four repository integration suites passed and prove:

- timestamp-conflict rejection;
- source-ID metadata and family-conflict rejection;
- atomic rollback of mixed valid/conflicting batches;
- legitimate different-family/same-time insertion;
- cross-agency isolation.

The complete accepted database contained nine same-timestamp/different-family groups and zero cross-agency source-ID groups.

## 19. Regression results

Passed:

- `Tests/HistoricalFxTimestampTests.sh`
- `Tests/EconomicEventImportValidationTests.sh`
- `Tests/EconomicEventImportRepositoryTests.sh`
- `Tests/DolEtaWeeklyClaimsAdapterTests.sh`
- `Tests/BeaEconomicReleaseAdapterTests.sh`
- `Tests/BeaEconomicEventImportRepositoryTests.sh`
- `Tests/CensusEconomicReleaseAcquisitionTests.sh`
- `Tests/CensusEconomicReleaseAdapterTests.sh`
- `Tests/CensusEconomicEventImportRepositoryTests.sh`
- `Tests/FederalReserveEconomicReleaseAcquisitionTests.sh`
- `Tests/FederalReserveEconomicReleaseAdapterTests.sh`
- `Tests/FederalReserveEconomicEventImportRepositoryTests.sh`
- `Tests/EconomicEventBarAlignmentTests.sh`
- `Tests/EconomicEventRepositoryTests.sh`
- `Tests/EconomicEventRealBarAlignmentIntegrationTests.sh`
- `Tests/EconomicEventManifestAuditTests.sh`
- Manifest audit harness against BEA, Census, DOL/ETA, and Federal Reserve fixtures.

The DB integration suites used `LSTM_DB_HOST=/tmp LSTM_DB_USER=vjp` only to create/drop their own fixed disposable databases; their application connections used the existing test patterns. Every test database reported dropped.

## 20. Debug build result

Command:

```sh
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" \
  -configuration Release \
  -derivedDataPath "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development" \
  build
```

Result: `** BUILD SUCCEEDED **`.

Observed pre-existing configuration warnings: unavailable `LLVM22.xctoolchain` metadata and deprecated `-Ofast`. No new compiler diagnostic failed the build. The dirty-tree LSTM Release provenance build was intentionally not attempted and remains deferred until after human review/commit.

## 21. Production safety

Production LSTM was accessed only in explicit read-only transactions for the required BLS inventory and the two existing read-only BLS repository/alignment regressions. The inventory transaction executed only these query shapes against `economic_event`:

1. `SELECT count(*), min(source_release_date), max(source_release_date) ... WHERE source_agency='BLS'`;
2. grouped BLS counts/bounds by `event_family`;
3. grouped BLS counts by `historical_time_confidence`;
4. grouped BLS counts by release year.

The repository regressions used `pqxx::read_transaction`, `SELECT to_regclass('public.economic_event')`, and the parameterized half-open `SELECT ... FROM economic_event WHERE currency=$1 AND event_timestamp_utc >= $2 AND event_timestamp_utc < $3`. Their `SET LOCAL TIME ZONE` checks remained inside the read-only transaction. Transactions closed immediately.

```text
LSTM_DATABASE_READ_ONLY_ACCESSED=true
FOREX_DATABASE_ACCESSED=false
PRODUCTION_DATABASE_MUTATED=false

PRODUCTION_SCHEDULER_TOUCHED=false
PRODUCTION_EXPERIMENT_TOUCHED=false
PRODUCTION_WORKER_SIGNALED=false
PRODUCTION_DERIVEDDATA_TOUCHED=false
```

Active production scheduler/workers were observed only through `ps`; they were not signaled, paused, reconfigured, or otherwise touched.

## 22. Disposable DB cleanup

The Phase 6A database and all four regression databases were queried from `postgres` after cleanup; none remained.

```text
DISPOSABLE_DATABASE_USED=ea_economic_calendar_phase6a_20260824_001
DISPOSABLE_DATABASE_DROPPED=true
```

## 23. Known limitations

1. BLS is quantitatively present and read-verified but lacks a committed reproducible 2010-current acquisition/import corpus.
2. DOL/ETA's authoritative archive currently exposes only 2025 links; the missing 2010-2024 and 2026 occurrence inventory is unresolved.
3. Ten Census collision participants cannot all satisfy the existing immutable source-ID contract.
4. Two distinct September 17, 2014 FOMC statements cannot both satisfy the same-family/timestamp uniqueness contract without inventing a time or family.
5. Fourteen authoritative artifacts fail closed for missing/contradictory provenance.
6. Partial current-year counts are bounded by archive publication as of 2026-08-24.
7. The build retains pre-existing toolchain/optimization warnings.

## 24. File-level diff summary

Tracked modifications:

- `Scripts/fetch_bea_economic_releases.py`: archive enumeration cache reuse.
- `Scripts/fetch_census_economic_releases.py`: resume/retry/throttle and deterministic artifact/extraction/manifest reuse.
- `Scripts/fetch_dol_eta_weekly_claims.py`: one-fetch archive enumeration cache.
- `Scripts/fetch_federal_reserve_economic_releases.py`: annual press and Beige Book archive enumeration/classification.
- `Sources/BeaEconomicReleaseAdapter.cpp`: narrow historical BEA header/identity/family variants.
- `Sources/CensusEconomicReleaseAdapter.cpp`: narrow historical Census/shutdown/date-only variants.
- `Sources/FederalReserveEconomicReleaseAdapter.cpp`: historical URLs, minutes association, statement and Beige Book semantics.
- Adapter/acquisition tests for BEA, Census, and Federal Reserve.

New review files:

- `Scripts/audit_economic_event_manifest.py`
- `Tests/EconomicEventManifestAudit.cpp`
- `Tests/EconomicEventManifestAudit.sh`
- `Tests/EconomicEventManifestAuditTests.py`
- `Tests/EconomicEventManifestAuditTests.sh`
- This report.

No file was staged or committed. All acquisition/build/test artifacts remain under `DerivedData/Development`.

## 25. Final repository state

Before adding this report, tracked diff stat was 12 files changed, 693 insertions, and 63 deletions, plus five untracked audit files. `git diff --check` was clean. The exact final `git status --short`, `git diff --stat`, and `git diff --check` are captured again after report creation in the completion response.

## 26. Phase 6B readiness decision

`NOT_READY_FOR_PHASE_6B`

Exact blockers are the unreproducible BLS corpus, unresolved DOL/ETA 2010-current enumeration gap, Census identity collisions, same-family/date-only Federal Reserve collision, and remaining failed-closed authoritative artifacts. Although the accepted subset is provenance-valid and database-idempotent, it is not evidence of the complete intended 2010-current corpus.

## 27. Recommended Phase 6B procedure if ready

Not applicable while the readiness decision is `NOT_READY_FOR_PHASE_6B`. Before reconsidering Phase 6B:

1. Commit an authoritative, reproducible BLS acquisition/import source and re-audit all 813 rows.
2. Obtain or formally scope an authoritative DOL/ETA 2010-2024/2026 occurrence index without guessed URLs.
3. Make an explicit architecture decision for genuine Census publisher-ID collisions without weakening immutable identity or creating duplicate production rows.
4. Make an explicit contract decision for distinct same-family date-only FOMC publications.
5. Resolve or formally accept each failed-closed artifact as an unavailable/genuine non-publication exclusion.
6. Re-run Phase 6A acquisition audit, fresh disposable import, exact repeat, conflict suites, and build.
7. Only after a new `READY_FOR_PHASE_6B` decision, perform a production dry-run/read-only preflight, preserve accepted manifests and hashes as release artifacts, use the common CLI/service with explicit human authorization, reconcile counts immediately, and retain rollback/audit evidence.
