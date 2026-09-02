---
title: "LSTM Authoritative Economic Calendar Phase 6A Remediation"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_AuthoritativeEconomicCalendar_Phase6A_RemediationContinuation_Prompt.txt"
date: "2026-08-24"
---

# LSTM Authoritative Economic Calendar Phase 6A Remediation

## 1. Executive summary

Phase 6A remediation and the complete readiness rerun are finished. The
continuation found that the provisional DOL blocker was an implementation
source-discovery defect: the first-party archive requires a year-selecting
POST (`report=press`, `year=YYYY`), while the prior implementation used an
unparameterized GET and therefore saw only the server's default 2025 calendar.

The corrected implementation freshly enumerated and acquired 863 DOL/ETA
artifacts from 2010 through August 20, 2026. After narrow historical parser and
identity remediation, 815 unique DOL occurrences were accepted. Across all
five agencies, the fresh audit enumerated 3,647 occurrences, accepted 3,583,
recorded every nonaccepted row, found zero unresolved collisions, imported all
3,583 rows through the shared workflow, and repeated all 3,583 as unchanged.

The remaining omissions are documented first-party source defects or archive
limitations. They do not require fabricated timestamps, weaker validation, or
a different persistence architecture, and they do not invalidate the Phase 6B
foundation.

## 2. Starting worktree state

The continuation began on branch `lstm-feature-development` with substantial
uncommitted remediation already present. The starting status contained 39
modified tracked files plus untracked audit evidence, migration 080, BLS
acquisition/adapter code, shared acquisition support, tests, and this report.
Those changes were preserved and inspected rather than restarted or rewritten.

Active scheduler, training, and inference processes using
`DerivedData/ExpertAdvisor` were observed before test/build work. No
`LSTM_Release` command was launched, no scheduler command was issued, and no
production database was inspected or mutated. All continuation artifacts and
builds used `DerivedData/Development`.

## 3. Previously completed remediation verified

The existing remediation was retained and reverified:

- bounded reads, redirect/final-resource validation, acquisition failure
  categorization, requested/final URL provenance, and artifact hashing;
- audit v2 reporting for acquisition, provenance, parse, validation,
  collision, acceptance, and import states, including nonparsed rows;
- parser-version changes for material semantic changes;
- BLS annual-schedule acquisition and the 813-occurrence reproducible corpus;
- Census family-scoped occurrence identity and exact duplicate-alias handling;
- distinct September 17, 2014 FOMC document identities and migration 080's
  narrowly guarded same-boundary exception;
- repository comparison, conflict, transaction, and idempotency behavior.

Focused tests, the fresh full-corpus audit, repository suites, and the Debug
build all confirmed that these fixes remain compatible.

## 4. Additional changes made during this continuation

The continuation made only demonstrated Phase 6A changes:

1. DOL archive enumeration now POSTs the requested year and excludes links
   whose first-party path year does not match that selected archive response.
2. DOL parser version advanced to `dol_eta_weekly_claims_v3` for material
   historical semantics: legacy split embargo/date cells; abbreviated month
   forms and comma-less dates; attached `USDL12-...` identifiers; the evidenced
   `USDL 24-24-2399-NAT` form; and URL-qualified identity for the two distinct
   occurrences both numbered `USDL 16-567-NAT`.
3. Whole-corpus duplicate-evidence handling now recognizes two exact DOL
   wrong-directory-year aliases and retains the canonical matching-year URL.
4. Regression tests prove DOL's year POST, historical parser/identity variants,
   and deterministic DOL alias selection.
5. Economic-calendar shell harness products were mechanically moved under
   `DerivedData/Development/Tests` for this continuation's output constraint.
6. Retained-evidence row counts now derive from actual import results.

No Phase 6B, model, Tensor, training, inference, scheduler, campaign, or
commercial-consensus work was added.

## 5. DOL historical-coverage findings

The authoritative [DOL/ETA Weekly Claims archive](https://oui.doleta.gov/unemploy/archive.asp)
contains a POST form whose `report=press` and `year` selection returns that
year's calendar of immutable `/press/YYYY/MMDDYY.asp|pdf` links. A plain GET
returns only a default calendar and is not historical enumeration.

| Year | Enumerated | Parsed | Alias evidence | Accepted | Fail-closed |
|---:|---:|---:|---:|---:|---:|
| 2010 | 52 | 52 | 0 | 52 | 0 |
| 2011 | 52 | 44 | 0 | 44 | 8 |
| 2012 | 54 | 17 | 2 | 15 | 37 |
| 2013 | 52 | 52 | 0 | 52 | 0 |
| 2014 | 54 | 53 | 0 | 53 | 1 |
| 2015 | 52 | 52 | 0 | 52 | 0 |
| 2016 | 52 | 52 | 0 | 52 | 0 |
| 2017 | 52 | 52 | 0 | 52 | 0 |
| 2018 | 52 | 52 | 0 | 52 | 0 |
| 2019 | 51 | 51 | 0 | 51 | 0 |
| 2020 | 53 | 53 | 0 | 53 | 0 |
| 2021 | 52 | 52 | 0 | 52 | 0 |
| 2022 | 52 | 52 | 0 | 52 | 0 |
| 2023 | 52 | 52 | 0 | 52 | 0 |
| 2024 | 52 | 52 | 0 | 52 | 0 |
| 2025 | 46 | 46 | 0 | 46 | 0 |
| 2026 through Aug. 24 | 33 | 33 | 0 | 33 | 0 |
| **Total** | **863** | **817** | **2** | **815** | **46** |

The two 2012 aliases are byte-identical copies of 2013 artifacts exposed under
wrong directory years; the correct 2013 URLs are retained. The 2016 archive
contains two distinct publications both numbered `USDL 16-567-NAT`; their
immutable artifact URLs provide stable distinct occurrence identities.

Forty-five documents explicitly claim `EST` during daylight time or `EDT`
during standard time (8 in late 2011 and 37 in 2012). They remain rejected
rather than receiving an invented UTC instant. The other rejection is the
archive's own `Dummy file: March 15, 2014`, not an economic release. These are
authoritative-source limitations and are non-blocking: 815 accepted causal
occurrences span every year from 2010 through 2026, and every omission is
explicitly machine-audited.

## 6. BEA historical archive findings

The fresh [BEA archive](https://www.bea.gov/news/archive) product-filtered
snapshot contains 371 URLs: 174 GDP and 197 Personal Income and Outlays. The
retained first-party union contains 393 still-acquirable URLs; current
pagination omits 22 and adds none. All 393 were freshly downloaded and audited.

Strict retained-hash acquisition detected two same-URL byte changes. The 2025
data-update page remains rejected for lacking an authoritative release time.
The changed accepted May 2026 release retained identical parsed source
identity, timestamp, confidence, and reference period.

BEA pagination and same-URL bytes are therefore not stable complete historical
enumeration. This is an authoritative-source limitation, not an unresolved
adapter defect. The retained URL catalog, fresh hashes, and fresh semantic
audit reproduce 391 accepted occurrences without weakening validation.

## 7. BLS reproducibility result

Seventeen fresh official annual schedule resources for 2010-2026 produced
exactly 813 occurrences. All 813 were acquired, parsed, validated, collision
checked, accepted, imported, and repeated unchanged: CPI 199, Employment 199,
annual Employment 17, JOLTS 199, and PPI 199.

## 8. Census occurrence-identity result

Fresh Census acquisition enumerated 1,171 artifacts, parsed 1,168, classified
two exact aliases as duplicate evidence, and accepted 1,166. Family-scoped
identities prevent reused publisher numbers from colliding across supported
families. All six families cover 2010-2026; no identity collision remains.

## 9. Federal Reserve/FOMC occurrence-identity result

Fresh Federal Reserve acquisition enumerated 407 artifacts and accepted 398.
The [September 17, 2014 statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20140917a.htm)
and distinct [normalization-policy document](https://www.federalreserve.gov/newsevents/pressreleases/monetary20140917c.htm)
retain separate immutable identities at the same conservative date-only
boundary. Migration 080, validation, audit, and repository logic permit only
that evidenced pair. Nine explicit timezone contradictions remain fail-closed.

## 10. Fresh authoritative corpus results

Fresh artifacts were written under
`DerivedData/Development/Phase6A-continuation-20260824`; stale generated output
was not used as proof.

| Agency | Enumerated | Resources | Acquired | Parsed | Parse failures | Duplicate evidence | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| BEA | 393 | 393 | 393 | 391 | 2 | 0 | 391 |
| BLS | 813 | 17 | 813 | 813 | 0 | 0 | 813 |
| Census | 1,171 | 1,171 | 1,171 | 1,168 | 3 | 2 | 1,166 |
| DOL/ETA | 863 | 863 | 863 | 817 | 46 | 2 | 815 |
| Federal Reserve | 407 | 407 | 407 | 398 | 9 | 0 | 398 |
| **Total** | **3,647** | **2,851** | **3,647** | **3,587** | **60** | **4** | **3,583** |

All occurrence acquisitions succeeded. Validation failures were zero. The 64
nonaccepted evidence rows are 60 retained parse exclusions plus four aliases.

## 11. Collision/audit results

Whole-corpus audit covered all 3,647 rows, including acquisition/provenance and
parse failures; invalid rows were not silently removed before evidence
retention. Results: zero unresolved source-identity collisions, zero unresolved
same-agency/family/time collisions, four exact aliases, one permitted FOMC
same-time group, and zero accepted duplicate source identities.

Parser/binary/script/manifest hashes, requested/final URLs, extractor identity,
artifact hashes, and occurrence provenance are retained under
`AuditEvidence/AuthoritativeEconomicCalendar/Phase6A/2026-08-24-continuation`.

## 12. Disposable PostgreSQL import results

Migrations 072 and 080 were applied only in disposable databases. Import used
the shared CLI -> service -> repository workflow.

| Agency | Attempted | Inserted | Updated | Rejected |
|---|---:|---:|---:|---:|
| BEA | 391 | 391 | 0 | 0 |
| BLS | 813 | 813 | 0 | 0 |
| Census | 1,166 | 1,166 | 0 | 0 |
| DOL/ETA | 815 | 815 | 0 | 0 |
| Federal Reserve | 398 | 398 | 0 | 0 |
| **Total** | **3,583** | **3,583** | **0** | **0** |

Final verification returned 3,583 rows, zero duplicate source identities, one
same-time group, zero unpermitted same-time groups, and both named FOMC rows.

Used and dropped databases include the four repository-suite databases plus
`ea_economic_calendar_phase6a_continuation_20260824`,
`ea_economic_calendar_phase6a_readpath_20260824`, and
`ea_economic_calendar_phase6a_finalcheck_20260824`. A final query in the first
complete run used the wrong column name; its cleanup trap dropped the database.
The corrected final-check database reran the full import and assertions. No
`ea_economic_calendar_%` disposable database remains.

## 13. Repeat/idempotency results

| Agency | Inserted | Unchanged | Updated | Rejected |
|---|---:|---:|---:|---:|
| BEA | 0 | 391 | 0 | 0 |
| BLS | 0 | 813 | 0 | 0 |
| Census | 0 | 1,166 | 0 | 0 |
| DOL/ETA | 0 | 815 | 0 | 0 |
| Federal Reserve | 0 | 398 | 0 | 0 |
| **Total** | **0** | **3,583** | **0** | **0** |

Source identity remains unique; exact metadata is unchanged; incompatible
metadata is rejected; migration 080 permits only the named FOMC pair.

## 14. Conflict test results

The DOL, BEA, Census, and Federal Reserve import-repository suites passed in
disposable databases. They cover source-ID metadata/family conflicts, exact
timestamp conflicts, cross-agency isolation, arbitrary same-time rejection,
and the narrow named FOMC coexistence case.

## 15. Rollback test results

BEA, Census, and Federal Reserve suites each submit a mixed batch containing an
otherwise insertable candidate and a conflict. All assert rejection and an
unchanged pre-batch row count. No partial state survived.

## 16. Regression-test results

Passed:

- all five agency adapter suites and all agency acquisition suites;
- `AuthoritativeEconomicAcquisitionTests.sh` (7),
  `EconomicEventManifestAuditTests.sh` (5), import validation, and bar
  alignment;
- all four import-repository suites;
- repository read and real-bar alignment against an isolated BLS database
  (486 bars, 3 events);
- `git diff --check`, project `plutil`, Python compilation, and shell syntax.

An initial read-path invocation against the all-agency database failed the
test's intentional BLS-only count assertion (`january.size() == 4`). It was
investigated as fixture contamination and rerun against the required isolated
BLS corpus, where both suites passed. It is not hidden or counted as an
implementation regression.

## 17. Debug build result

```text
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Debug" -configuration Debug \
  -derivedDataPath DerivedData/Development build
```

Result: `** BUILD SUCCEEDED **`. It compiled/linked without launching the
executable. No clean was run.

## 18. Remaining limitations and classifications

| Item | Classification | Blocking? | Disposition |
|---|---|---:|---|
| 45 contradictory DOL EST/EDT documents | authoritative-source limitation | No | Retained fail-closed; 815 accepted occurrences span all years |
| DOL 2014 dummy link | authoritative-source limitation | No | Retained as nonrelease evidence; not imported |
| BEA pagination omits 22 live retained URLs | authoritative-source limitation | No | Retained catalog plus fresh union acquisition/audit |
| Two BEA same-URL hash changes | authoritative-source limitation | No | Detected; changed accepted semantics are identical |
| Other malformed/zone-contradictory artifacts | authoritative-source limitation | No | Retained with diagnostics and excluded |
| Commercial consensus/forecast data | intentionally deferred non-blocking work | No | Outside Phase 6A |

Implementation defects remaining: none demonstrated. Verification gaps:
none. Environmental/transient acquisition problems: none. No source limitation
is being used to conceal an implementation failure.

## 19. Git diff/status summary

The worktree remains intentionally uncommitted. Exact final status and diff
statistics are inserted after this report update.

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Scripts/audit_economic_event_manifest.py
 M Scripts/fetch_bea_economic_releases.py
 M Scripts/fetch_census_economic_releases.py
 M Scripts/fetch_dol_eta_weekly_claims.py
 M Scripts/fetch_federal_reserve_economic_releases.py
 M Sources/BeaEconomicReleaseAdapter.cpp
 M Sources/CensusEconomicReleaseAdapter.cpp
 M Sources/DolEtaWeeklyClaimsAdapter.cpp
 M Sources/EconomicEventImportRepository.cpp
 M Sources/EconomicEventImportService.cpp
 M Sources/EconomicEventImportValidation.cpp
 M Sources/EconomicEventImportValidation.hpp
 M Sources/FederalReserveEconomicReleaseAdapter.cpp
 M Tests/BeaEconomicEventImportRepositoryTests.sh
 M Tests/BeaEconomicReleaseAdapterTests.cpp
 M Tests/BeaEconomicReleaseAdapterTests.sh
 M Tests/CensusEconomicEventImportRepositoryTests.sh
 M Tests/CensusEconomicReleaseAcquisitionTests.py
 M Tests/CensusEconomicReleaseAdapterTests.cpp
 M Tests/CensusEconomicReleaseAdapterTests.sh
 M Tests/DolEtaWeeklyClaimsAdapterTests.cpp
 M Tests/DolEtaWeeklyClaimsAdapterTests.sh
 M Tests/EconomicEventBarAlignmentTests.sh
 M Tests/EconomicEventImportRepositoryTests.sh
 M Tests/EconomicEventImportValidationTests.cpp
 M Tests/EconomicEventImportValidationTests.sh
 M Tests/EconomicEventManifestAudit.cpp
 M Tests/EconomicEventManifestAudit.sh
 M Tests/EconomicEventManifestAuditTests.py
 M Tests/EconomicEventRealBarAlignmentIntegrationTests.sh
 M Tests/EconomicEventRepositoryTests.sh
 M Tests/FederalReserveEconomicEventImportRepositoryTests.cpp
 M Tests/FederalReserveEconomicEventImportRepositoryTests.sh
 M Tests/FederalReserveEconomicReleaseAcquisitionTests.py
 M Tests/FederalReserveEconomicReleaseAdapterTests.sh
 M Tests/fixtures/economic_calendar/bea/manifest.tsv
 M Tests/fixtures/economic_calendar/bea/manifest_bad_hash.tsv
 M Tests/fixtures/economic_calendar/bea/manifest_unsupported_type.tsv
 M Tests/fixtures/economic_calendar/census/manifest.tsv
 M Tests/fixtures/economic_calendar/census/manifest_bad_hash.tsv
 M Tests/fixtures/economic_calendar/census/manifest_unsupported_type.tsv
 M Tests/fixtures/economic_calendar/dol_eta/manifest.tsv
 M Tests/fixtures/economic_calendar/dol_eta/manifest_bad_hash.tsv
 M Tests/fixtures/economic_calendar/federal_reserve/manifest.tsv
 M Tests/fixtures/economic_calendar/federal_reserve/manifest_bad_hash.tsv
 M Tests/fixtures/economic_calendar/federal_reserve/manifest_missing_artifact.tsv
 M Tests/fixtures/economic_calendar/federal_reserve/manifest_unsupported_type.tsv
?? AuditEvidence/
?? Database/migrations/080_economic_event_distinct_same_time_identity.sql
?? LSTM_AuthoritativeEconomicCalendar_Phase6A_Remediation_Output.md
?? Scripts/authoritative_acquisition.py
?? Scripts/fetch_bls_economic_releases.py
?? Scripts/retain_phase6a_audit_evidence.py
?? Sources/BlsScheduleReleaseAdapter.cpp
?? Sources/BlsScheduleReleaseAdapter.hpp
?? Tests/AuthoritativeEconomicAcquisitionTests.py
?? Tests/AuthoritativeEconomicAcquisitionTests.sh
?? Tests/BlsEconomicReleaseAcquisitionTests.py
?? Tests/BlsEconomicReleaseAcquisitionTests.sh
?? Tests/BlsScheduleReleaseAdapterTests.cpp
?? Tests/BlsScheduleReleaseAdapterTests.sh
```

`git diff --stat` reports 48 tracked files changed, 1,037 insertions, and
190 deletions. It does not include 35 untracked files (principally the retained
evidence bundle, migration 080, BLS implementation/tests, and this report).

## 20. Final readiness verdict

Focused remediation, fresh first-party acquisition, full-corpus audit,
provenance, collision handling, disposable first/repeat import, identity
assertions, conflict and rollback suites, read-path regressions, static checks,
and the Debug build are complete. Remaining limitations are explicit
non-blocking authoritative-source limitations, not code or verification gaps.

# READY_FOR_PHASE_6B
