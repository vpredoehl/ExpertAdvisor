# Phase 17 authoritative BLS initial-actual readiness

This directory contains the deterministic Phase 17 review package generated
from the retained official BLS archive corpus and the production catalog under
`default_transaction_read_only=on`. No BLS actual was imported into production.

## Coverage decision

| Family | Retained releases | Certified initials | Selected consensus | Usable intersection | Exclusion or gap | Decision |
|---|---:|---:|---:|---:|---:|---|
| CPI | 198 | 198 | 179 | 177 | 1 unavailable release | PARTIAL |
| Employment | 199 | 190 | 180 | 171 | 9 reissues without initial provenance | PARTIAL |
| PPI | 199 | 199 | 179 | 179 | 0 | READY |
| JOLTS | 199 | 199 | 145 | 145 | 0 | READY |

The payload contains 786 certified initial revision-zero actuals and provides
672 new usable selected-consensus intersections. Together with the existing
PCE, GDP, and Retail Sales intersections, a future import would provide 1,076
usable intersections.

The sole unresolved canonical acquisition is CPI release 2016-06-16. BLS
returned HTTP 200 with `text/html` and a zero-length body. Historical failed
capture observations remain in `acquisition_failures.jsonl`; later validated
admissions are not recounted as current catalog gaps.

PPI retains the historical finished-goods/final-demand regime transition. An
independent comparison to retained OANDA observations matched 29 finished-goods
and 148 final-demand values, with zero mismatches.

## Deterministic payload

- `bls-initial-actual-manifest.jsonl`: 786 rows, SHA-256
  `cd4a124979cde17c65d3550d8b28293650aada458ae10f4a2fa288f547583420`.
- `bls-initial-actual-import.sql`: append-only transaction, SHA-256
  `4622f334bb2fd196287e768e1863745a70ed5fb4e09274cf594fdce85a7cc1a9`.
- `bls-candidate-classifications.jsonl`: all 796 target catalog releases,
  including the acquisition gap and nine initial-provenance exclusions.
- `bls-coverage-audit.json`: production reconciliation, family coverage,
  deterministic hashes, and PPI semantic cross-check.

Independent first and repeat regeneration was byte-identical. A disposable
database import inserted exactly 786 rows, reached 1,230 total actual rows,
exposed all 786 through the initial revision-zero feature view, found no causal
or observation-identity violations, regenerated an empty transaction after the
import, and rejected UPDATE and DELETE. The disposable database was dropped.

## Validation

Passed:

- `Tests/BlsEconomicEventReleaseActualTests.sh`
- `Tests/BlsProductionReadinessTests.sh`
- `Tests/EconomicEventReleaseActualImporterTests.sh`
- `Tests/EconomicEventReleaseActualHistoricalCorpusTests.sh`
- `Tests/EconomicEventActualPointInTimeTests.sh`
- `Tests/EconomicEventProductionActualCoverageTests.sh`
- `Tests/PceProductionReadinessTests.sh`
- `Tests/MultiFamilyInitialActualReadinessTests.sh`

No Xcode build was required because Phase 17 changes only Python, Node, shell,
documentation, and retained data.

## Safety boundary

- Production schema: 088.
- Production release actuals: 444.
- Production BLS release actuals: 0.
- Production BLS import performed: NO.
- Production release-actual writes: NO.
- Experiments 603-608 modified by this work: NO.
- Scheduler stopped or restarted: NO.
- Width-75 training started: NO.

Decision: `BLS_PRODUCTION_IMPORT_READY`

STOP: the next task is a separately controlled Phase 18 production BLS import.
Do not execute the SQL payload or begin width-75 training in Phase 17.
