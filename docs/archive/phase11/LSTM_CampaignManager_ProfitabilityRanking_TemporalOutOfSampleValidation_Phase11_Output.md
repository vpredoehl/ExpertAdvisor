---
title: "LSTM Campaign Manager Profitability Ranking Temporal Out-of-Sample Validation Phase 11"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_ProfitabilityRanking_TemporalOutOfSampleValidation_Phase11_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Profitability Ranking Temporal Out-of-Sample Validation Phase 11

## Outcome

Phase 11 is implemented as a deterministic, read-only feasibility audit plus a forward-validation precommit fallback.

Advisory assessment:

`HISTORICAL_HOLDOUT_UNAVAILABLE_FORWARD_VALIDATION_REQUIRED`

The historical evidence neither supports nor rejects weight `0.025`: no admissible subsequent-outcome cohort exists. Profitability-aware production ranking remains disabled, and the live profitability weight remains exactly `0`.

## Implementation

Added:

- `--validate-campaign-profitability-temporal`
- `--prepare-campaign-profitability-forward-validation=SNAPSHOT_ID,OUTCOME_START,OUTCOME_END`

Both commands:

- Use repeatable-read, read-only PostgreSQL transactions.
- Preserve exact FINAL-only Phase 9/10 evidence semantics.
- Reject checkpoint substitution, mismatches, overlap, and future leakage.
- Reconstruct weight-zero rankings exactly or fail closed.
- Fix the primary candidate at `0.025`; no holdout tuning or Phase 10 sweep occurs.
- Create no persistence, experiments, campaigns, rankings, or activation authority.

Files changed:

- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [ProfitabilityCalibration.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityCalibration.cpp>)
- [ProfitabilityVerification.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerification.hpp>)
- [ProfitabilityVerificationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerificationRepository.cpp>)
- [ProfitabilityVerificationRepository.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerificationRepository.hpp>)
- [ProfitabilityVerificationService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerificationService.cpp>)
- [ProfitabilityVerificationService.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerificationService.hpp>)
- [CampaignProfitabilityReadinessRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignProfitabilityReadinessRepositoryTests.cpp>)
- [CampaignProfitabilityReadinessRepositoryTests.sh](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignProfitabilityReadinessRepositoryTests.sh>)
- [ProfitabilityShadowRankingTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ProfitabilityShadowRankingTests.cpp>)
- [Phase 11 documentation](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignManagerPhase11TemporalProfitabilityValidation.rst>)

Migrations added: none.

## Temporal feasibility findings

| Snapshot | Run | As-of UTC | Candidates | Ranking evidence valid/unavailable | Subsequent outcomes | Classification |
|---:|---:|---|---:|---:|---:|---|
| 1 | 2 | 2026-08-12T03:08:52.477632Z | 1 | 0 / 1 | 0 | `insufficient_ranking_time_provenance` |
| 3 | 3 | 2026-08-13T04:13:46.667885Z | 1 | 0 / 1 | 0 | `insufficient_ranking_time_provenance` |
| 4 | 5 | 2026-08-15T13:01:03.113493Z | 100 | 0 / 100 | 0 | `insufficient_ranking_time_provenance` |
| 5 | 6 | 2026-08-28T16:30:10.130479Z | 79 | 28 / 51 | 0 | `insufficient_subsequent_outcome` |

Totals:

- Admissible: `0`
- Rejected: `4`
- Insufficient ranking-time provenance: `3`
- Insufficient subsequent outcome: `1`

Snapshots 1, 3, and 4 cannot reproduce the current authoritative historical control contract exactly. Snapshot 5 reproduces its zero-weight control exactly with no point-in-time provenance violations.

All persisted profitability windows end no later than `2026-01-01`, before the August 2026 ranking decisions. Snapshot 4 has four members associated with one later-arriving observation, but that observation covers `2025-01-01`–`2026-01-01`; it is rejected as temporal leakage, not treated as an outcome.

## Control versus precommitted 0.025

These are forward-precommit selection differences only; subsequent outcomes remain pending.

| Top N | Overlap | Candidate entrants | Control exits | Turnover |
|---:|---:|---|---|---:|
| 5 | 2 | 410, 417, 418 | 359, 360, 361 | 60% |
| 10 | 7 | 407, 411, 417 | 360, 361, 378 | 30% |
| 20 | 18 | 408, 409 | 369, 370 | 10% |

Ranking-time candidate coverage remains:

- Overall: `28/79 = 35.4430%`
- Top 5: `5/5 = 100%`
- Top 10: `9/10 = 90%`
- Top 20: `13/20 = 65%`

Subsequent-outcome coverage is `0%`. Therefore aggregate returns, averages, wins/losses, entrant-versus-exit profitability, per-cohort robustness, outlier sensitivity, bootstrap analysis, and statistical significance were not calculated.

The optional `0.0225` and `0.0275` values were not evaluated against outcomes. They remain explicitly non-selective sensitivity values.

## Forward precommit validation

An illustrative no-write precommit for snapshot 5 and `2026-08-29`–`2027-08-29` produced:

- Cohort identity: `fnv1a64:38ab27d06a6b2695`
- Control ranking: `fnv1a64:33527191afa4caec`
- Candidate ranking: `fnv1a64:e4478d9578b1e8c9`
- 79 deterministic member records
- Top-5/top-10/top-20 selection records
- Outcome identities marked `PENDING`

Determinism:

- Temporal audit SHA-256, both runs: `3feee54f44f60fcd496d5ee86470f3baffd389de6ab231232c8b5a8a81129909`
- Forward precommit SHA-256, both runs: `a30d6bab45699c7f85630c205f6101c6bdd8f364b8fb86a841c55c15e53e2aab`

A durable scientific precommit still requires saving the output in an external immutable/versioned artifact before observing the chosen outcome window.

## Validation

Passed:

```text
bash Tests/ProfitabilityShadowRankingTests.sh
bash Tests/CampaignProfitabilityReadinessRepositoryTests.sh
git diff --check
```

The repository test used and dropped a disposable PostgreSQL database. It covers Phase 9/10 regressions plus fixed-weight precommitment, deterministic replay, exact control reproduction, future-window rejection, no-write behavior, and forward-precommit identity.

CLI rejection checks:

```text
unrelated option exit: 1
overlapping window exit: 2
malformed specification exit: 1
```

Release build succeeded from isolated clean validation source:

```text
/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Release/ProfitabilityPhase11Validation
```

No build was placed under `DerivedData/Development`. No clean was run. Existing project warnings, primarily deprecated libpqxx calls, remain; the focused changed sources passed `-Wall -Wextra -Werror`.

## Scheduler and database safety

Scheduler PID `16569` and training worker PIDs `17115` and `17291` remained unchanged. The scheduler was not restarted, interrupted, or replaced. `--scheduler-status` was not invoked because it is write-capable.

Production counts before and after read-only validation were identical:

```text
recommendations | evaluation runs | evaluation results | snapshots | members | observations
187             | 5               | 181                | 4         | 181     | 43
```

Confirmed throughout:

```text
activation=false
live_profitability_weight=0
production_ranking_modified=false
recommendation_modified=false
ranking_snapshot_modified=false
experiment_created=false
experiment_queued=false
scheduler_modified=false
worker_modified=false
```

## Coverage-policy considerations

No threshold was adopted. A later human policy should consider:

- Overall and top-5/top-10/top-20 exact-FINAL coverage.
- Subsequent-outcome coverage.
- Missingness by era, symbol, and horizon.
- Multiple independent temporal cohorts.
- Whether selection changes depend on unavailable evidence.
- Robustness after removing dominant cohorts or outliers.

Current blockers are zero admissible historical cohorts, zero subsequent-outcome coverage, no completed forward validation, and no approved coverage policy.

## Next safe commands

```bash
PHASE11_BIN="/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Release/ProfitabilityPhase11Validation/Build/Products/Release/LSTM_Release"

PGOPTIONS='-c default_transaction_read_only=on' \
"$PHASE11_BIN" --validate-campaign-profitability-temporal
```

Prepare a genuinely future precommit without launching work:

```bash
PGOPTIONS='-c default_transaction_read_only=on' \
"$PHASE11_BIN" \
--prepare-campaign-profitability-forward-validation=5,2026-08-30,2027-08-30
```

Current `git status --short`:

```text
 M Sources/ExperimentScheduler.cpp
 M Sources/ProfitabilityCalibration.cpp
 M Sources/ProfitabilityVerification.hpp
 M Sources/ProfitabilityVerificationRepository.cpp
 M Sources/ProfitabilityVerificationRepository.hpp
 M Sources/ProfitabilityVerificationService.cpp
 M Sources/ProfitabilityVerificationService.hpp
 M Tests/CampaignProfitabilityReadinessRepositoryTests.cpp
 M Tests/CampaignProfitabilityReadinessRepositoryTests.sh
 M Tests/ProfitabilityShadowRankingTests.cpp
?? docs/CampaignManagerPhase11TemporalProfitabilityValidation.rst
```

Current tracked `git diff --stat`:

```text
10 files changed, 1258 insertions(+), 6 deletions(-)
```

The untracked Phase 11 documentation file is not included in that Git statistic.