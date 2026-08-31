---
title: "LSTM Economic Event Features Phase 19 Width-75 Production Readiness"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_EconomicEventFeatures_Phase19_Width75ProductionReadiness_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM Economic Event Features Phase 19 Width-75 Production Readiness

# WIDTH75_PRODUCTION_EXPERIMENT_NOT_READY

The repository implementation, schema, corpus, serialization, and offline execution path passed. The concrete blocker is operational: the active production executable predates the width-75 initial-actual implementation.

The first width-75 experiment was not queued or started.

## 1. Starting and final safety state

Repository:

- Commit: `63f936bc9ee4cdc301b6b4b3914ffa76ca1e84bf`
- Branch: `lstm-feature-development`
- Tracking: ahead of `github/lstm-feature-development` by 34 commits
- Starting and final worktree: clean
- Recent commits:
  - `63f936b Archive production BLS initial actual import`
  - `b6c282b Archive Phase 17 BLS acquisition review`
  - `52af13f Add authoritative BLS initial actual acquisition`
  - `081e2af Update production backup after GDP and retail actual import`
  - `5de0742 Archive Phase 15 multi-family coverage review`

Production database:

- Database/role: `LSTM` / `vjp`, local Unix socket
- PostgreSQL: 17.11 Homebrew
- Data directory: `/Volumes/Forex Data/forexdb`
- Recovery: false
- Latest migration: `088_economic_event_release_actual_provenance.sql`
- Migration checksum: `c78de60820f0dae0e6f20e1c5882232c7ec06cc957490c078cc4e3a2b7329b0e`

Scheduler:

- PID: `97217`
- State: active/running
- Fencing token: 81
- Production binary: `DerivedData/Production/SchedulerPauseResumePriorityQueueRelease/Build/Products/Release/LSTM_Release`
- Limits: two training workers, zero inference workers, two analysis workers
- Scheduler was not stopped, restarted, or reconfigured.

Final experiments 603–608 state:

| Experiment | State | Phase | Epoch | Worker | Model |
|---:|---|---|---:|---:|---:|
| 603 | pending | train | 60/80 | — | 1715 |
| 604 | pending | train | 60/80 | — | 1714 |
| 605 | running | train | 74/80 | 43339 | 1718 |
| 606 | running | train | 74/80 | 43344 | 1719 |
| 607 | pending | infer | 80/80 | — | 1727 |
| 608 | pending | infer | 80/80 | — | 1729 |

Their persisted models are width 71, semantic layout V4. No experiment, model, checkpoint, policy, or campaign row was changed.

## 2. Production corpus verification

Runtime-compatible counts exactly match Phase 19 expectations:

| Family | Actuals | Consensus | Usable |
|---|---:|---:|---:|
| PCE | 183 | 177 | 168 |
| GDP | 67 | 179 | 61 |
| RETAIL_SALES | 194 | 181 | 175 |
| CPI | 198 | 179 | 177 |
| EMPLOYMENT | 190 | 180 | 171 |
| PPI | 199 | 179 | 179 |
| JOLTS | 199 | 145 | 145 |
| **Total** | **1,230** | — | **1,076** |

CPI has 178 raw intersections but 177 usable intersections. Event 779 has incompatible semantics: year-over-year consensus versus month-over-month actual. It correctly fails closed.

All 1,230 production actuals are:

- `publication_state='initial'`
- `revision_sequence=0`
- Distinct by economic event
- Source-agency compatible
- Available no earlier than the event
- Retrieved no earlier than availability
- Backed by artifact path, SHA-256, semantic contract, and source provenance

There are currently no production revision rows. Synthetic point-in-time tests verify that a later revision cannot enter features.

The schema makes actuals immutable and exposes only initial revision zero through `economic_event_feature_release_actual`, as defined in [088_economic_event_release_actual_provenance.sql](/Volumes/Developer%20SSD/ExpertAdvisor/Database/Migrations/088_economic_event_release_actual_provenance.sql:140).

## 3. Exact width-75 contract

The width is proven by compile-time and runtime contracts:

```text
49 pre-economic Tensor features
+10 event/recency features
+ 8 consensus-era columns
+ 4 authoritative-initial surprise columns
=71 Tensor columns

71 Tensor columns
+4 historical return features
=75 model inputs
```

Relevant authorities are [FeatureLayout.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/FeatureLayout.hpp:63), [EconomicEventFeatureLayout.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/EconomicEventFeatureLayout.hpp:8), and [ModelInputContract.hpp](/Volumes/Developer%20SSD/ExpertAdvisor/Headers/ModelInputContract.hpp:49).

- Pre-economic model width: 53
- Event/recency width: 63
- Consensus-era width: 71
- Current width: 75
- Current semantic layout: V5
- V5 append-only predecessor: V4/width 71

Economic feature table:

| Index | Feature | Source/scope | Scale and default | Causal rule |
|---:|---|---|---|---|
| 49 | inflation_event | CPI, PPI, PCE | binary; 0 | Event occurred in completed 15-minute bar |
| 50 | employment_event | Employment, annual employment, JOLTS, claims | binary; 0 | Same |
| 51 | growth_event | GDP, durable goods | binary; 0 | Same |
| 52 | fed_policy_event | FOMC | binary; 0 | Same |
| 53 | consumer_demand_event | Retail sales | binary; 0 | Same |
| 54 | inflation_recency_decay | Most recent inflation event | `exp(-seconds/86400)`; 0 | Released events only |
| 55 | employment_recency_decay | Most recent employment event | same | Released events only |
| 56 | growth_recency_decay | Most recent growth event | same | Released events only |
| 57 | fed_policy_recency_decay | Most recent FOMC event | same | Released events only |
| 58 | consumer_demand_recency_decay | Most recent retail event | same | Released events only |
| 59 | relevant_event_has_consensus | Selected consensus | binary; 0 | Exact release boundary or most recent released event |
| 60 | relevant_event_consensus_low | Selected consensus low | family normalization; 0 | Same |
| 61 | relevant_event_consensus_high | Selected consensus high/scalar duplicate | family normalization; 0 | Same |
| 62 | relevant_event_consensus_is_range | Consensus shape | binary; 0 | Same |
| 63 | released_event_has_surprise | Closed provider-surprise channel | always 0 | Deliberately disabled |
| 64 | released_event_surprise | Closed provider-surprise channel | always 0 | Deliberately disabled |
| 65 | released_event_surprise_abs | Closed provider-surprise channel | always 0 | Deliberately disabled |
| 66 | released_event_surprise_direction | Closed provider-surprise channel | always 0 | Deliberately disabled |
| 67 | authoritative_initial_has_surprise | Revision-zero authoritative actual | binary; 0 | `available_at < completed-bar cutoff` |
| 68 | authoritative_initial_surprise | Actual minus consensus | family normalization; 0 | Same; compatible scalar semantics required |
| 69 | authoritative_initial_surprise_abs | Absolute surprise | family normalization; 0 | Same |
| 70 | authoritative_initial_surprise_direction | Surprise sign | -1/0/+1; 0 | Same |
| 71 | return_1 | Prior market closes | existing return scale | Completed history only |
| 72 | return_4 | Prior market closes | existing return scale | Completed history only |
| 73 | return_8 | Prior market closes | existing return scale | Completed history only |
| 74 | return_16 | Prior market closes | existing return scale | Completed history only |

## 4. Point-in-time and numerical semantics

The implementation in [EconomicEventFeatures.cpp](/Volumes/Developer%20SSD/ExpertAdvisor/Sources/EconomicEventFeatures.cpp:206) establishes:

- Future events require `event.timestamp < completed-bar cutoff`.
- At the exact release boundary, consensus can be exposed but the actual remains absent.
- The actual becomes visible only when `available_at < cutoff`; equality does not pass.
- Missing actual, missing consensus, and incompatible semantics produce deterministic zero surprise channels.
- Surprise is `(authoritative actual − selected consensus) / family scale`.
- Positive surprise means actual exceeded consensus.
- Scalar/range, unit, or qualifier mismatches fail closed.
- Later revisions are excluded at the database view.
- Same-time events preserve identity; selection uses importance, then the smallest positive economic-event ID.
- SQL orders by timestamp and economic-event ID, while the feature engine independently enforces deterministic relevance.
- Invalid shapes, nonfinite values, bad units, and float overflow fail before neural-network input.
- Training and inference both clamp finite model inputs to `[-10, 10]`.

Normalization:

- Percentage families: divide canonical percentage points by 10
- Employment counts: divide by 1,000,000
- JOLTS counts: divide by 10,000,000
- Recency: one-day exponential time constant
- Missing values: zero, accompanied by availability indicators

Production normalized surprise distributions:

| Family | n | Consensus range | Actual range | Surprise min / p05 / median / p95 / max |
|---|---:|---|---|---|
| PCE | 168 | -1.25…0.90 | -1.36…0.82 | -0.25 / -0.03 / 0 / 0.03 / 0.06 |
| GDP | 61 | -3.41…3.10 | -3.29…3.31 | -0.25 / -0.14 / 0 / 0.12 / 0.21 |
| Retail | 175 | -1.19…0.80 | -1.64…1.77 | -0.45 / -0.08 / 0 / 0.093 / 0.97 |
| CPI | 177 | -0.07…0.12 | -0.08…0.13 | -0.03 / -0.02 / 0 / 0.02 / 0.06 |
| Employment | 171 | 0.05…0.978 | -0.14…0.943 | -0.712 / -0.156 / 0.012 / 0.150 / 0.332 |
| PPI | 179 | -0.06…0.11 | -0.13…0.17 | -0.08 / -0.051 / 0 / 0.04 / 0.09 |
| JOLTS | 145 | 0.31…1.14 | 0.464…1.155 | -0.072 / -0.051 / 0.0046 / 0.076 / 0.309 |

All sampled production values were finite and comfortably inside the model clamp.

## 5. Historical coverage

The intended baseline metadata confirms:

- Train: `[2010-01-01, 2025-01-01)`
- Infer: `[2025-01-01, 2026-01-01)`

| Family | Train releases / consensus / actual / usable / defaults | Infer releases / consensus / actual / usable / defaults |
|---|---|
| PCE | 179 / 160 / 167 / 152 / 27 | 10 / 10 / 10 / 10 / 0 |
| GDP | 179 / 161 / 60 / 54 / 125 | 10 / 10 / 4 / 4 / 6 |
| Retail | 180 / 161 / 174 / 155 / 25 | 11 / 11 / 11 / 11 / 0 |
| CPI | 180 / 160 / 179 / 159 / 21 | 11 / 11 / 11 / 10 / 1 |
| Employment | 180 / 161 / 172 / 153 / 27 | 11 / 11 / 10 / 10 / 1 |
| PPI | 180 / 160 / 180 / 160 / 20 | 10 / 10 / 10 / 10 / 0 |
| JOLTS | 180 / 126 / 180 / 126 / 54 | 11 / 11 / 11 / 11 / 0 |

Usable coverage generally begins in July–August 2011; JOLTS begins in July 2014. The earlier deterministic-zero periods could encode a historical regime boundary, but they cannot leak future information. The inference window is within mature coverage. GDP remains intentionally sparse.

## 6. Serialization, resume, and parity

Verified:

- `model_meta` persists schema, input width, and hidden size.
- `model_input_semantics_meta` persists schema 1 and semantic layout version.
- Parameter dimensions are cross-checked against the persisted width.
- Width 75 is a registered width and cannot silently load as width 71.
- Explicit expansion validates append-only layout ancestry.
- Expansion moves recurrent/return rows correctly and zero-initializes new tensor columns.
- Expansion provenance persists source model, source width, target width, new column indices/names, layout version, and initialization policy.
- Resume expansion creates a new model row and does not mutate the source model.
- Scheduler commands propagate expansion only when `resume_expand_input_width=true`.
- Training, final inference, checkpoint inference, and regression/classification inference use the same `Tensor`, `ModelInputContract`, and feature-copy path.
- Profitability/analysis consumes authoritative inference results rather than rebuilding a different model input.
- No stale production hard-coded width or alternate experiment feature ordering was found.

Fresh width-75 creation is supported. Append-only expansion from width 71 is also supported and is recommended for the first controlled comparison.

## 7. Smoke and regression results

Passed:

- `EconomicEventFeaturesTests`: width 22
- `EconomicEventActualPointInTimeTests`
- `EconomicEventFeatureRangeRepositoryTests`
- `EconomicEventBarAlignmentTests`
- `EconomicEventProductionActualCoverageTests`: 5/5
- `EconomicEventTensorIntegrationTests`
- `EconomicEventFeaturesRealInputIntegrationTests`
- `LSTMModelInputCompatibilityTests`
- `LSTMInputWidthExpansionTests`
- `LSTMInputWidthExpansionPersistenceTests`
- `LSTMFeatureVectorParityTests`
- `LSTMInputWidthExpansionSchedulerIntegrationTests.sh DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`
- `SchedulerControlWorkerIdentityIntegrationTests.sh DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release`
- `git diff --check`

Mutating tests used disposable databases, which were dropped. No Phase 19 disposable database remains.

Release build passed:

```sh
xcodebuild -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Release \
  -derivedDataPath DerivedData/ExpertAdvisor \
  build
```

The build reported 420 warnings, dominated by existing libpqxx `exec_params` deprecations. There were no build errors. No production artifact was overwritten.

The deterministic smoke produced 75 finite, repeatable values for PCE, GDP, retail sales, CPI, employment, PPI, and JOLTS. Exact release cutoffs exposed consensus but not actual; the following cutoff exposed the expected authoritative surprise. A no-event case produced 75 deterministic zeros.

A production-read-only integration processed 486 bars and 2,601 events successfully.

## 8. Deferred first width-75 experiment design

Not queued.

| Field | Proposed value |
|---|---|
| Control experiment/model | Experiment 602 / model 1702 |
| Reason | Completed, unablated width-71/layout-V4 baseline with final inference and profitability evidence |
| Symbol | `audchfrmp` |
| Horizon | 4 |
| Source epoch | 80 |
| Target epoch | 100; 20 additional epochs |
| Train range | 2010-01-01 to 2025-01-01 |
| Infer range | 2025-01-01 to 2026-01-01 |
| Threshold | 0.0008 |
| Checkpoint interval | 20 |
| Objective | `legacy_first_hit_weighted_ce_v1` |
| Optimizer | SGD |
| Base LR | 1/3000 |
| Core multiplier | 119.75 |
| Head weight/bias multipliers | 25 / 2.5 |
| Donchian | enabled, lookback 20 |
| Warmup | `legacy_cold_boundary` |
| Feature mask | none |
| Input | width 75, semantic layout V5 |
| Initialization | explicit append-only expansion; columns 67–70 zero-initialized |
| Checkpoint inference | disabled for first controlled run |
| Checkpoint stop policy | disabled |
| Continuation policy | disabled |
| Profitability | normal final inference/analysis only; no checkpoint or continuation evaluation |

This is preferable to a fresh model because zero initialization preserves the width-71 model’s initial predictions and makes model 1702 the exact comparison control.

## 9. Runtime and capacity

Observed width-71 evidence:

- 80-epoch training: approximately 17.9–24.4 hours when paired
- Final inference: approximately 16.5–38.7 minutes
- Analysis: approximately 0.5–1.1 minutes
- Current workers: about 54–56% CPU and approximately 790 MiB RSS each

Width 71→75 adds only 1,024 LSTM input weights at hidden width 64—about 4 KiB of persistent float parameters. Input-side buffers grow about 5.6%; overall memory and runtime growth should be small.

Estimated proposed 20-epoch expansion:

- Training: approximately 4.5–6.2 hours
- Final inference: approximately 17–41 minutes
- Checkpoint inference: none under the proposed policy
- Persistent parameter memory increase: about 4 KiB
- Total worker RSS increase: expected to be minor relative to current ~790 MiB

The experiment should wait until 603–608 are complete. Current inference capacity is zero, so 607–608 cannot naturally complete inference under the present scheduler invocation.

## 10. Concrete blockers and narrowest remediation

1. The active production executable is not width-75 capable.

   - Production artifact mtime: `2026-08-30 06:56:52 -0500`
   - It predates commit `274edfd` (`2026-08-30 22:14:43`), which integrated initial actuals.
   - SHA-256: `8a252042fb4dcceb85cd313657edf4c961705ceaa0a2ec4aba838c5cf98a84e2`
   - It contains no `authoritative_initial_surprise` marker.
   - The audited Release artifact does contain that marker and passed the width-75 tests.

2. Protected workload/capacity has not drained.

   - Experiments 603–606 still require training.
   - Experiments 607–608 are waiting for inference while `max-infer-procs=0`.

Narrowest remediation: allow the protected training workload to finish, then perform a separately authorized controlled scheduler turnover that deploys an artifact built from audited commit `63f936b`, restores nonzero inference capacity, and lets 603–608 finish. Reverify the deployed hash and width-75 marker before queuing the deferred experiment.

## 11. Safety accounting

```text
production_rows_modified=0
experiments_603_608_modified=NO
experiments_queued=NO
scheduler_restarted=NO
production_binary_replaced=NO
campaign_manager_state_modified=NO
width75_training_started=NO
additional_economic_event_ingestion=NO
```

Files changed: none.
Behavior changed: none.

Final `git status --short`: empty.

Final `git diff --stat`: empty.

Final branch status:

```text
## lstm-feature-development...github/lstm-feature-development [ahead 34]
```