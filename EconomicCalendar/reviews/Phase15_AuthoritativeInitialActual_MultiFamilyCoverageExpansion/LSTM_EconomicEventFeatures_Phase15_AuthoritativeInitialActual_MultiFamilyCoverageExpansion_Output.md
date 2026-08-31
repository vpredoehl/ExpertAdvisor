---
title: "LSTM Economic Event Features Phase 15 Authoritative Initial Actual Multi-Family Coverage Expansion"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_EconomicEventFeatures_Phase15_AuthoritativeInitialActual_MultiFamilyCoverageExpansion_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Economic Event Features Phase 15 Authoritative Initial Actual Multi-Family Coverage Expansion

Phase 15 is complete and committed locally. GDP and Retail Sales are production-ready; no production import or training was started.

## A. Starting state

- Starting HEAD: `ee1b43cf882adbe5f30e20ba34336a7532373035`
- Branch: `lstm-feature-development`
- Starting worktree: clean
- Production: `127.0.0.1 / LSTM`
- Migration: `088`
- Existing release actuals: 183, all `BEA / PCE`
- Existing PCE initial revision-0 rows: 183

Final commit: `0c4224553b349168fd8c28f0f8096998cb4a95dd`

## B. Family coverage matrix

| Family | Agency | Retained artifacts | Certified initial | Consensus | Usable intersection | Actual-only | Consensus-only | Certified span | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| CPI | BLS | 0 | 0 | 179 | 0 | 0 | 179 | — | NOT_READY |
| Employment | BLS | 0 | 0 | 180 | 0 | 0 | 180 | — | NOT_READY |
| GDP | BEA | 197 | 67 | 179 | 61 | 6 | 118 | 2010-01-29–2026-07-30 | READY |
| Retail Sales | Census | 200 | 194 | 181 | 175 | 19 | 6 | 2010-01-14–2026-08-14 | READY |
| PPI | BLS | 0 | 0 | 179 | 0 | 0 | 179 | — | NOT_READY |
| JOLTS | BLS | 0 | 0 | 145 | 0 | 0 | 145 | — | NOT_READY |
| FOMC | Federal Reserve | 136 | 0 | 120 | 0 | 0 | 120 | — | NOT_READY |

GDP excluded 130 later estimates as revisions. Retail excluded 190 revisions and six artifacts with unsupported headline semantics.

## C. Provenance findings

- GDP: first-release status is proven by exact BEA release titles and headline text identifying an advance or initial estimate.
- Retail Sales: proven by the Census `ADVANCE` marker, current reference month, headline total retail-and-food-services statistic, artifact SHA-256, and exact release timestamp.
- Six Retail PDFs fail closed because the supported headline m/m actual could not be independently proven.
- BLS provides historical archive indexes for [CPI](https://www.bls.gov/bls/news-release/cpi.htm), [Employment](https://www.bls.gov/bls/news-release/empsit.htm), [PPI](https://www.bls.gov/bls/news-release/ppi.htm), and [JOLTS](https://www.bls.gov/bls/news-release/jolts.htm). Direct repository acquisition received HTTP 403, so no release-specific evidence was retained and no current/revised tables were substituted.
- FOMC statements are retained, but narrative decisions were not manufactured into actuals. Most consensus rows are ranges, while runtime surprise channels require scalar-compatible pairs.

## D. Semantic contracts

Supported:

- GDP: real GDP, advance/initial estimate only; annualized q/q scalar percent; scale 1; `bea_real_gdp_annualized_quarterly_percent_v1`.
- Retail Sales: advance total retail and food-services sales; m/m scalar percent; scale 1; `census_advance_release_headline_mom_percent_v1`.

Not supported this phase:

- CPI headline/core and m/m/y/y variants
- NFP/unemployment/earnings variants
- PPI finished-goods/final-demand regimes
- JOLTS job-openings level
- FOMC narrative or range outcomes

## E. Determinism

Evidence: [Phase 15 README](</Volumes/Developer SSD/ExpertAdvisor/AuditEvidence/AuthoritativeEconomicCalendar/Phase15/2026-08-30/README.md>)

- [GDP manifest](</Volumes/Developer SSD/ExpertAdvisor/AuditEvidence/AuthoritativeEconomicCalendar/Phase15/2026-08-30/gdp-initial-actual-manifest.jsonl>): 67 rows, `d0ae2e798fe24d01fa0b3a1c802fb86b7479b66a5d39507f985a99ecdcec8770`
- [Retail manifest](</Volumes/Developer SSD/ExpertAdvisor/AuditEvidence/AuthoritativeEconomicCalendar/Phase15/2026-08-30/retail-sales-initial-actual-manifest.jsonl>): 194 rows, `fb971edf8483ed8975d22f9aecc1a401e1b429c0d1eabd8a57a5e7e0e1942d5e`
- [Coverage audit](</Volumes/Developer SSD/ExpertAdvisor/AuditEvidence/AuthoritativeEconomicCalendar/Phase15/2026-08-30/multi-family-coverage-audit.json>)
- [Candidate classifications](</Volumes/Developer SSD/ExpertAdvisor/AuditEvidence/AuthoritativeEconomicCalendar/Phase15/2026-08-30/multi-family-candidate-classifications.jsonl>)

Independent regeneration was byte-identical. Every manifest artifact hash was revalidated.

## F. Disposable validation

All passed:

- `Tests/MultiFamilyInitialActualReadinessTests.sh`
- `Tests/EconomicEventReleaseActualImporterTests.sh`
- `Tests/EconomicEventReleaseActualHistoricalCorpusTests.sh`
- `Tests/EconomicEventActualPointInTimeTests.sh`
- `Tests/EconomicEventProductionActualCoverageTests.sh`
- `Tests/PceProductionReadinessTests.sh`

The exact 67 GDP and 194 Retail rows were imported into a disposable database. It exposed exactly 261 initial revision-0 feature rows; repeat generation produced empty import transactions.

No Xcode build was required because compiled source did not change. The point-in-time test still compiled and validated the relevant C++ components.

## G. Production writes

`NO`

Production remains at 183 PCE rows. Target-family production rows remain zero.

## H. Experiment safety

- experiments 603–608 modified by this task: `NO`
- experiments queued: `NO`
- scheduler stopped/restarted: `NO`
- production binary replaced: `NO`
- width-75 training started: `NO`
- Campaign Manager state modified: `NO`

Scheduler PID 97217 and worker PIDs 43339/43344 remained unchanged. Experiments 605–606 progressed naturally from epoch 23 to 24.

## I. Recommended next production import

GDP plus Retail Sales as one reviewed group:

- 261 certified initials
- 236 usable consensus intersections
- Retail contributes 175 intersections
- GDP contributes 61 intersections
- Both use explicit, release-specific government provenance and clean scalar semantics

## J. Decision

`AUTHORITATIVE_MULTI_FAMILY_INITIAL_ACTUAL_COVERAGE_READY_FOR_NEXT_IMPORT`

## K. STOP

No production import, push, binary replacement, or width-75 training was performed.

Files changed: 10 files, 3,331 insertions. The implementation entry point is [prepare_multi_family_initial_actual_import.py](</Volumes/Developer SSD/ExpertAdvisor/EconomicCalendar/prepare_multi_family_initial_actual_import.py>).

- `git status --short`: clean
- `git diff --stat`: empty
- GitHub push: `NO`