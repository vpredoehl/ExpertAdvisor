---
title: "LSTM EconomicEventFeatures Consensus Integration Phase 4"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_EconomicEventFeatures_ConsensusIntegration_Phase4_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM EconomicEventFeatures Consensus Integration Phase 4

Implemented consensus-aware `EconomicEventFeatures` with strict causal availability, append-only layout evolution, repository range loading, and model-width provenance.

1. Architecture found
   Existing columns 49–58 contained five same-bar family indicators and five 24-hour recency decays. Training and inference already share `EconomicEventFeatureEngine` through `Tensor`, with events becoming released only when `event_time < completed_bar_cutoff`.

2. Features implemented
   Appended columns 59–66:

   - `relevant_event_has_consensus`
   - `relevant_event_consensus_low`
   - `relevant_event_consensus_high`
   - `relevant_event_consensus_is_range`
   - `released_event_has_surprise`
   - `released_event_surprise`
   - `released_event_surprise_abs`
   - `released_event_surprise_direction`

3. Causal rules
   Because persisted forecasts have no historical `known_at` timestamp, final consensus is not projected into arbitrary earlier bars.

   - At an exact release cutoff: consensus is available; actual/surprise is withheld.
   - Once the release is strictly causal under existing alignment: consensus and compatible surprise are available.
   - Future events contribute nothing.
   - Same-timestamp selection uses importance, then lower positive event ID.
   - Only actual semantics from the same immutable selected observation are used.

4. Normalization
   Static, dataset-independent divisors:

   - CPI, PPI, PCE, GDP, durable goods, retail sales, FOMC: percentage points ÷ 10.
   - Employment: canonical count ÷ 1,000,000.
   - JOLTS: canonical count ÷ 10,000,000.

5. FOMC
   Range forecasts preserve low/high endpoints and set the range flag. No midpoint is invented, and range surprises remain unavailable. Compatible scalar FOMC observations may produce surprise.

6. Missing encoding
   Missing consensus encodes all eight fields as zero. `released_event_has_surprise` distinguishes unavailable/incompatible surprise from a genuine zero surprise.

7. Repository changes
   Both repository loaders now perform one `LEFT JOIN` against `economic_event_selected_consensus`; no N+1 or per-bar SQL was introduced. Consensus and actual fields are optional. Migration 083 appends selected-row actual semantics to the existing view without changing selection precedence or stored rows.

8. Layout
   Existing columns 0–58 retain their exact positions and semantics. New components occupy 59–66.

9. Widths

   - Economic-event block: 10 → 18
   - Physical Tensor: 59 → 67
   - Model input including four return channels: 63 → 71
   - Pre-event width 53 remains supported.

10. Input-width compatibility
    Semantic layout advanced v3 → v4 with v3 as its append-only predecessor. Width 63 remains registered and projects exactly the old 59-column Tensor prefix. Expansion to 71 continues through the existing explicit zero-initialized resume expansion workflow.

11. Files created

    - [083_economic_event_selected_consensus_release_semantics.sql](</Volumes/Developer SSD/ExpertAdvisor/Database/migrations/083_economic_event_selected_consensus_release_semantics.sql>)
    - [LSTM_EconomicEventConsensusFeatures.md](</Volumes/Developer SSD/ExpertAdvisor/EconomicCalendar/LSTM_EconomicEventConsensusFeatures.md>)

12. Files modified
    Core:

    - `Headers/FeatureLayout.hpp`
    - `Headers/ModelInputContract.hpp`
    - `Headers/ModelInputExpansion.hpp`
    - `Sources/EconomicEventFeatureLayout.hpp`
    - `Sources/EconomicEventFeatures.{hpp,cpp}`
    - `Sources/EconomicEventRepository.{hpp,cpp}`

    Tests:

    - `EconomicEventFeaturesTests.cpp`
    - `EconomicEventFeatureRangeRepositoryTests.{cpp,sh}`
    - `EconomicEventTensorIntegrationTests.cpp`
    - `EconomicEventFeaturesRealInputIntegrationTests.cpp`
    - `LSTMFeatureVectorParityTests.cpp`
    - `LSTMInputWidthExpansionTests.cpp`
    - `LSTMModelInputCompatibilityTests.cpp`
    - `DonchianTensorIntegrationTests.cpp`
    - 14 existing causal/feature-width integration test sources shown in `git status` below.

13. Tests added
    Coverage includes both providers, provider neutrality, missing/emergency FOMC behavior, exact-boundary causality, surprise timing/sign/zero validity, count and percent normalization, incompatible semantics, FOMC ranges, append-only positions, train/infer parity, repository joins, and old-width expansion ancestry.

14. Tests/build results

    Passed:

    - `EconomicEventFeaturesTests.sh`
    - `EconomicEventFeatureRangeRepositoryTests.sh`
    - `EconomicEventTensorIntegrationTests.sh`
    - `LSTMFeatureVectorParityTests.sh`
    - `LSTMInputWidthExpansionTests.sh`
    - `LSTMModelInputCompatibilityTests.sh`
    - 14 width-sensitive causal feature suites
    - Real-input integration source compiled with `-Wall -Wextra -Werror`
    - Debug Xcode build from `DerivedData/Development`
    - `git diff --check`

    The exact Release command reached the provenance phase but exited 65 because Release provenance requires a clean worktree. It was not bypassed. Debug compiled and linked successfully; an unrelated pre-existing unreachable-code warning remains in `CampaignOperationsProductionAdmissionService.cpp`.

15. Disposable databases
    Created and destroyed:

    - `ea_economic_event_feature_range_52844`
    - `ea_economic_event_feature_range_53996`
    - `ea_economic_event_feature_range_54607`

    A final database query confirmed none remain.

16. Production access
    Yes, for representative read-only SQL verification only.

17. Read-only confirmation
    Every production session used `default_transaction_read_only=on`, `BEGIN READ ONLY`, and reported `transaction_read_only = on`. No migration, ingestion, or data mutation was performed.

18. Real-data verification

    - Physical rows: 1,532
    - Selected rows: 1,521
    - OANDA: 1,405
    - Myfxbook: 116
    - JOLTS 2014-07-08: 4,530,000
    - JOLTS 2023-11-01: 9,250,000
    - PPI 2013-12-13: −0.1
    - Retail Sales 2022-09-15: 0.0
    - CPI 2023-12-12: 0.0
    - Emergency FOMC 2020-03-03 and 2020-03-15: zero selected rows

    Representative OANDA rows were checked for CPI, PPI, employment, JOLTS, GDP, PCE, retail sales, and FOMC. Myfxbook representatives were verified for CPI, PPI, JOLTS, and retail sales.

19. Performance
    One additional range-query join and optional field mapping. Alignment and feature calculation remain in-memory and linear; scheduler/training behavior outside feature width/content is unchanged.

20. Remaining issues/assumptions

    - Production must receive migration 083 in a separate deployment review before the new binary can query the appended view columns.
    - Historical forecasts lack observation-time history, so consensus availability is intentionally conservative.
    - Myfxbook selected rows contain no actual values, so they cannot produce surprise.
    - Runtime real-input execution against production migration 082 was intentionally deferred; equivalent data was verified through read-only SQL.
    - No commit or push was performed.

21. `git status --short`

```text
 M Headers/FeatureLayout.hpp
 M Headers/ModelInputContract.hpp
 M Headers/ModelInputExpansion.hpp
 M Sources/EconomicEventFeatureLayout.hpp
 M Sources/EconomicEventFeatures.cpp
 M Sources/EconomicEventFeatures.hpp
 M Sources/EconomicEventRepository.cpp
 M Sources/EconomicEventRepository.hpp
 M Tests/DonchianTensorIntegrationTests.cpp
 M Tests/EconomicEventFeatureRangeRepositoryTests.cpp
 M Tests/EconomicEventFeatureRangeRepositoryTests.sh
 M Tests/EconomicEventFeaturesRealInputIntegrationTests.cpp
 M Tests/EconomicEventFeaturesTests.cpp
 M Tests/EconomicEventTensorIntegrationTests.cpp
 M Tests/LSTMCausalCloseLocationTests.cpp
 M Tests/LSTMCausalDirectionalAdverseExcursionTests.cpp
 M Tests/LSTMCausalDirectionalEfficiencyTests.cpp
 M Tests/LSTMCausalDirectionalRangeTests.cpp
 M Tests/LSTMCausalHistoricalLevelProximityTests.cpp
 M Tests/LSTMCausalMultiBarRangePressureTests.cpp
 M Tests/LSTMCausalReturnAutocorrelationTests.cpp
 M Tests/LSTMCausalReturnDirectionImbalanceTests.cpp
 M Tests/LSTMCausalReturnSignPersistenceTests.cpp
 M Tests/LSTMCausalReturnSurpriseTests.cpp
 M Tests/LSTMCausalRollingRangeExpansionTests.cpp
 M Tests/LSTMCausalVolatilityRegimeTests.cpp
 M Tests/LSTMFeatureVectorParityTests.cpp
 M Tests/LSTMInputWidthExpansionTests.cpp
 M Tests/LSTMModelInputCompatibilityTests.cpp
 M Tests/LSTMRelativeTickVolumeTests.cpp
 M Tests/LSTMTrueSessionPhaseTests.cpp
?? CEE_Reviews/
?? Database/migrations/083_economic_event_selected_consensus_release_semantics.sql
?? EconomicCalendar/LSTM_EconomicEventConsensusFeatures.md
```

`CEE_Reviews/` was pre-existing and untouched.

22. `git diff --stat`

```text
31 files changed, 954 insertions(+), 161 deletions(-)
```

The two untracked files created by this task are not included in Git’s unstaged diff stat.