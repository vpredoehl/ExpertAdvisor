---
title: "LSTM Campaign Manager Profitability Ranking Production Readiness Calibration Phase 10"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_CampaignManager_ProfitabilityRanking_ProductionReadiness_Calibration_Phase10_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Campaign Manager Profitability Ranking Production Readiness Calibration Phase 10

Implemented Phase 10 as a deterministic, advisory-only calibration workflow. Profitability-aware production ranking remains disabled; no experiments, recommendations, rankings, scheduler state, or worker state were modified.

## Implementation

The new `--calibrate-campaign-profitability=SNAPSHOT_ID` command:

- Runs inside a repeatable-read, PostgreSQL read-only transaction.
- Audits every ranking member and emits deterministic per-member provenance and recovery classification.
- Executes the complete 21-point sweep from `0` through `0.05`.
- Calculates movement, directionality, top-N, pairwise, stability, discontinuity, sensitivity, and readiness metrics.
- Reproduces snapshot 5 exactly at weight zero or fails closed.
- Rejects unrelated CLI options.
- Emits explicit safety fields including `activation=false`, `database_write=false`, `production_ranking_modified=false`, `experiment_created=false`, `experiment_queued=false`, and `scheduler_modified=false`.
- Does not provide any production ranking authority.

Core files:

- [ProfitabilityCalibration.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityCalibration.cpp>)
- [ProfitabilityVerification.hpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerification.hpp>)
- [ProfitabilityVerificationRepository.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerificationRepository.cpp>)
- [ProfitabilityVerificationService.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ProfitabilityVerificationService.cpp>)
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [Phase 10 documentation](</Volumes/Developer SSD/ExpertAdvisor/docs/CampaignManagerPhase10ProfitabilityCalibration.rst>)

Tests and integration:

- [ProfitabilityShadowRankingTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/ProfitabilityShadowRankingTests.cpp>)
- [CampaignProfitabilityReadinessRepositoryTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/CampaignProfitabilityReadinessRepositoryTests.cpp>)
- Associated test scripts and Xcode project membership were updated.

No migration was added. The result is a deterministic projection and does not require persistence.

## Coverage audit

Before and after remain identical: **28 / 79 = 35.4430% valid coverage**. No backfill was performed.

| Classification | Count | Evidence finding |
|---|---:|---|
| Valid exact-FINAL profitability | 28 | Exact final result and exact profitability observation exist |
| Recoverable historical absence | 44 | Exact final result exists; no exact or other persisted profitability observation |
| Final inference context mismatch | 4 | No admissible exact final result or profitability observation for the frozen source context |
| No exact final inference | 3 | No exact final result or profitability observation |
| Other unavailable | 0 | — |
| Invalid/incomplete frozen provenance | 0 | — |

The 44 historical absences are only theoretically recoverable through a separately authorized exact-FINAL replay under the same model, dates, scope, metric, and source semantics. The necessary persisted prediction material is not available for a safe local recomputation. The frozen recommendation/evaluation evidence cannot be rewritten, so Phase 10 deliberately did not introduce backfill tooling.

The command emits all 79 member records with recommendation, evaluation, ranking-member, experiment/model, symbol/horizon, frozen provenance, result/observation existence, reason, recovery class, and deterministic hash.

## Anchor comparison

Population at every weight: 79 total, 28 valid, 51 unavailable, 19 positive, 9 negative, 0 zero-profitability.

Top-N values below are `positive / negative / unavailable`; `R/E/X` means retained/entered/exited versus control.

| Weight | Movement: mean / median / p90; max up/down | Top 5 | Top 10 | Top 20 |
|---:|---|---|---|---|
| 0 | 0 / 0 / 0; 0/0 | 2/0/3; 5/0/0 | 6/0/4; 10/0/0 | 11/0/9; 20/0/0 |
| 0.01 | 2.7595 / 2 / 8; 9/16 | 4/0/1; 3/2/2 | 7/0/3; 9/1/1 | 11/0/9; 20/0/0 |
| 0.025 | 4.7595 / 3 / 14; 20/34 | 5/0/0; 2/3/3 | 9/0/1; 7/3/3 | 13/0/7; 18/2/2 |
| 0.05 | 7.3165 / 4 / 19; 41/38 | 5/0/0; 2/3/3 | 10/0/0; 6/4/4 | 15/0/5; 16/4/4 |

Directionality:

| Weight | Positive up/same/down | Negative up/same/down | Unavailable up/same/down |
|---:|---:|---:|---:|
| 0.01 | 14/5/0 | 0/2/7 | 26/5/20 |
| 0.025 | 16/3/0 | 0/2/7 | 24/10/17 |
| 0.05 | 16/0/3 | 0/2/7 | 4/10/37 |

Pairwise anchor results:

| Pair | Top 5/10/20 overlap | Ordinal changes | Mean / largest difference |
|---|---|---:|---|
| 0.01 vs 0.025 | 4 / 8 / 18 | 51 | 2.4051 / 18 |
| 0.025 vs 0.05 | 5 / 9 / 18 | 59 | 3.0633 / 24 |
| 0.01 vs 0.05 | 4 / 7 / 16 | 69 | 5.4684 / 34 |

Top-N difference recommendation IDs:

- `0.01 → 0.025`: top-5 `359,410`; top-10 `360,361,407,411`; top-20 `369,370,408,409`.
- `0.025 → 0.05`: top-5 none; top-10 `359,412`; top-20 `363,364,379,380`.
- `0.01 → 0.05`: top-5 `359,410`; top-10 `359,360,361,407,411,412`; top-20 `363,364,369,370,379,380,408,409`.

## Empirical response curve

Values are weight, mean absolute movement, and positive counts in top 5/10/20:

```text
0       0.0000   2  6 11
0.0025  0.5823   2  7 11
0.0050  1.9241   4  7 11
0.0075  2.5570   4  7 11
0.0100  2.7595   4  7 11
0.0125  3.3671   4  7 11
0.0150  3.7975   4  7 13
0.0175  4.2278   4  7 13
0.0200  4.3291   4  7 13
0.0225  4.7595   5  9 13
0.0250  4.7595   5  9 13
0.0275  5.0127   5  9 13
0.0300  5.4430   5  9 13
0.0325  5.7722   5  9 13
0.0350  6.1013   5  9 15
0.0375  6.1519   5 10 15
0.0400  6.6076   5 10 15
0.0425  6.8608   5 10 15
0.0450  7.0633   5 10 15
0.0475  7.1646   5 10 15
0.0500  7.3165   5 10 15
```

Response findings:

- First best observed top-5 composition: **0.0225**.
- First top-10 with at least nine positive candidates: **0.0225**.
- First best observed top-10 composition: **0.0375**.
- First top-20 improvement: **0.015**.
- Membership discontinuities: `0.0025, 0.005, 0.01, 0.015, 0.0225, 0.025, 0.035, 0.0375`.

Stable membership intervals:

- Top 5: `0–0.0025`, `0.005–0.0075`, `0.01–0.02`, `0.0225–0.05`.
- Top 10: `0`, `0.0025–0.02`, `0.0225`, `0.025–0.035`, `0.0375–0.05`.
- Top 20: `0–0.0125`, `0.015–0.0325`, `0.035–0.05`.

## Calibration assessment

The provisional minimum-effective region is **0.0225–0.0275**.

- `0.0225` is the smallest grid point achieving 5/5 positive top-5, 9/10 positive top-10, improved top-20, no positive member moving down, no negative member moving up, and materially less churn than `0.05`.
- `0.025` lies inside this region and remains a defensible anchor candidate. Exact top-10 membership is stable from `0.025–0.035`.
- At `0.03`, a positive-profitability member begins moving down, ending the strict minimum-effective interval.
- Weights below `0.015` do not improve top-20; weights through `0.02` still produce only 4/5 and 7/10 positive composition.
- `0.035–0.05` provides additional top-20/top-10 enrichment, but with substantially greater ordinal churn. At `0.05`, mean movement is 7.3165 and three positive members move down.

This supports `0.0225–0.0275` only as an **in-sample shadow calibration interval**, not a production decision.

## Production readiness

At the assessment weight `0.0225`:

- Overall valid coverage: **35.4430%**
- Top-5 valid coverage: **100%**
- Top-10 valid coverage: **90%**
- Top-20 valid coverage: **65%**

Production readiness is **false**. Blockers are:

- Low and uneven overall coverage.
- No repository-approved coverage policy or threshold.
- No temporal/out-of-sample validation.
- Historical missing evidence cannot safely be written into the frozen snapshot.
- Context-mismatched and no-exact-final cases cannot be substituted.

A later human-approved phase should define a justified coverage policy and perform temporal validation where the weight is selected using only information available at selection time, then assessed against subsequent realized profitability.

## Validation

Passed:

```text
bash Tests/ProfitabilityShadowRankingTests.sh
bash Tests/CampaignProfitabilityReadinessRepositoryTests.sh
git diff --check
```

The repository test used and dropped a disposable PostgreSQL database, verifies repeated deterministic output, invalid-provenance failure, exact-FINAL-only evidence, and unchanged experiment/recommendation/ranking/sentinel counts.

Production snapshot validation was run with PostgreSQL enforcing read-only mode. Two complete executions produced the identical SHA-256:

```text
f67d39baf4cb0113f027636e518ae214f1420e0fd79e56629b766ccde771c0ca
```

Release build succeeded:

```bash
xcodebuild \
  -project /tmp/ea_phase10_calibration.QuL9WL/ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/ProfitabilityPhase10ValidationRelease" \
  build
```

The validation clone was necessary because the Release provenance script correctly rejects a dirty source tree. No `clean` was run. Existing repository warnings, predominantly deprecated libpqxx calls, remain; the new calibration-focused compilation passes `-Wall -Wextra -Werror`.

The active scheduler remained PID `16569`; it was not interrupted or modified.

## Manual read-only validation

```bash
pgrep -fl LSTM_Release

PGOPTIONS='-c default_transaction_read_only=on' \
"/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/ProfitabilityPhase10ValidationRelease/Build/Products/Release/LSTM_Release" \
--calibrate-campaign-profitability=5
```

Determinism check:

```bash
for run in 1 2; do
  PGOPTIONS='-c default_transaction_read_only=on' \
  "/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/ProfitabilityPhase10ValidationRelease/Build/Products/Release/LSTM_Release" \
  --calibrate-campaign-profitability=5 | shasum -a 256
done
```

## Working tree

`lstm_watch.sql` was already modified before Phase 10 and was not touched.

```text
 M ExpertAdvisor.xcodeproj/project.pbxproj
 M Sources/ExperimentScheduler.cpp
 M Sources/ProfitabilityVerification.cpp
 M Sources/ProfitabilityVerification.hpp
 M Sources/ProfitabilityVerificationRepository.cpp
 M Sources/ProfitabilityVerificationRepository.hpp
 M Sources/ProfitabilityVerificationService.cpp
 M Sources/ProfitabilityVerificationService.hpp
 M Tests/CampaignProfitabilityReadinessRepositoryTests.cpp
 M Tests/CampaignProfitabilityReadinessRepositoryTests.sh
 M Tests/ProfitabilityShadowRankingTests.cpp
 M Tests/ProfitabilityShadowRankingTests.sh
 M lstm_watch.sql
?? Sources/ProfitabilityCalibration.cpp
?? docs/CampaignManagerPhase10ProfitabilityCalibration.rst
```

Exact tracked `git diff --stat`—which does not include the two untracked new files:

```text
13 files changed, 957 insertions(+), 2 deletions(-)
```

Profitability-aware Campaign Manager production ranking remains disabled.