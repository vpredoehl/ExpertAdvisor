---
title: "LSTM EconomicEventFeatures Consensus Ablation Efficacy Phase 5"
document_type: "architecture review"
status: "final"
generated_from: "LSTM_EconomicEventFeatures_ConsensusAblation_Efficacy_Phase5_Prompt.txt"
reasoning_effort: "high"
model: "gpt-5.6-sol"
---

# LSTM EconomicEventFeatures Consensus Ablation Efficacy Phase 5

Implemented the deterministic, persisted, same-width consensus ablation. It is ready for operator review; no experiments were queued or executed.

## Architecture and implementation

The existing `feature_ablation_mask` architecture was reused without schema changes:

- Canonical comma-separated feature names are persisted on `experiment`.
- The mask participates in experiment identity and duplicate detection.
- Models inherit provenance through `model.experiment_id`.
- Scheduler workers reconstruct the mask from the persisted experiment.
- Inference reconstructs it from model lineage.
- Ordinary resume inherits the source mask and rejects mismatches.
- Explicit width expansion preserves append-only compatibility rules.

Added four individually addressable names in [FeatureAblation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/FeatureAblation.hpp>):

```text
relevant_event_has_consensus
relevant_event_consensus_low
relevant_event_consensus_high
relevant_event_consensus_is_range
```

Canonical control mask:

```text
relevant_event_has_consensus,relevant_event_consensus_low,relevant_event_consensus_high,relevant_event_consensus_is_range
```

Treatment persists an empty mask. Control persists the canonical four-name mask. Individual names remain available under existing semantics; no new alias/group parser or redundant persistence path was introduced.

Scheduler help now exposes `--ablate-features=NAME[,NAME...]`.

## Behavioral contract

- Tensor width: unchanged at 67.
- Model input width: unchanged at 71.
- Semantic layout: unchanged at v4.
- Treatment: columns 59–62 computed normally.
- Control: exactly columns 59–62 forced to zero after model-input projection.
- Columns 0–58: identical.
- Columns 63–66: zero in both arms.
- Return-feature suffix: unchanged.
- Tensor storage: not mutated.
- Width 63: cannot directly use the consensus mask because those columns are absent.
- Surprise: remains disabled.
- Weekly Claims employment behavior: unchanged.
- Missing versus genuine-zero consensus: preserved in treatment.
- FOMC low/high/range representation: preserved in treatment and fully zeroed in control.
- Unrelated ablation bits: unchanged.

No profitability formulas, semantics, persistence, or policy were changed.

## Files changed

- [FeatureAblation.hpp](</Volumes/Developer SSD/ExpertAdvisor/Headers/FeatureAblation.hpp>)
- [ExperimentScheduler.cpp](</Volumes/Developer SSD/ExpertAdvisor/Sources/ExperimentScheduler.cpp>)
- [EconomicEventTensorIntegrationTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/EconomicEventTensorIntegrationTests.cpp>)
- [LSTMFeatureVectorParityTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/LSTMFeatureVectorParityTests.cpp>)
- [LSTMInputWidthExpansionTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/LSTMInputWidthExpansionTests.cpp>)
- [LSTMModelInputCompatibilityTests.cpp](</Volumes/Developer SSD/ExpertAdvisor/Tests/LSTMModelInputCompatibilityTests.cpp>)

Files created: none. Migrations created: none.

## Validation

Passed:

```bash
Tests/LSTMModelInputCompatibilityTests.sh
Tests/LSTMFeatureVectorParityTests.sh
bash Tests/LSTMInputWidthExpansionTests.sh
Tests/EconomicEventFeaturesTests.sh
LSTM_TEST_PRODUCTS_DIR="$PWD/DerivedData/Development/ConsensusAblation/Build/Products/Debug" \
  Tests/EconomicEventTensorIntegrationTests.sh
git diff --check
```

All focused standalone C++ tests used `-Wall -Wextra -Werror`.

Isolated builds, both successful:

```bash
xcodebuild \
  -project ExpertAdvisor.xcodeproj \
  -scheme "LSTM Release" \
  -configuration Debug \
  -derivedDataPath DerivedData/Development/ConsensusAblation \
  ENABLE_USER_SCRIPT_SANDBOXING=NO \
  build
```

The Xcode build reports 398 existing libpqxx `exec_params` deprecation warnings. Directly affected strict tests were warning-clean.

No disposable databases were created, so none required destruction. Feature-specific database/worker integration was not rerun because the maintained integration runner invokes `--queue-experiment`, which this task explicitly prohibited. The unchanged generic persistence, reconstruction, and resume paths were verified by inspection and full compilation.

## Production inspection

Production `LSTM` was accessed only inside explicit read-only transactions followed by `ROLLBACK`.

Current relevant state during inspection:

- Running training: 2 experiments, IDs 565 and 578.
- Pending training: 13.
- Pending inference: 1.
- Scheduler remained on the Development Release binary.
- Two pre-cutover workers remained on the older executable.
- No scheduler or worker was modified.
- No training, inference, analysis, queue, pause, resume, or cancellation command was executed.

The existing profitability infrastructure automatically persists immutable final observations with:

- actionable count;
- aggregate terminal-horizon log-return sum;
- average terminal-horizon log-return per actionable prediction.

The current paired-objective evaluator already jointly interprets predictive and profitability evidence, but it intentionally requires a training-objective difference and rejects feature-mask differences. It was therefore not repurposed in this phase.

## Preferred initial pair

Configuration source: experiment 563, AUDCHF H4.

| Field | Value |
|---|---:|
| Symbol | `audchfrmp` |
| Horizon | 4 |
| Target epochs | 80 |
| Checkpoint interval | 20 |
| Threshold | 0.0008 |
| Core LR multiplier | 119.75 |
| Head LR multiplier | 25 |
| Training window | 2010-01-01 to 2025-01-01 |
| Inference window | 2025-01-01 to 2026-01-01 |
| Donchian mode/lookback | enabled / 20 |
| Warmup | `legacy_cold_boundary` |
| Objective | `legacy_first_hit_weighted_ce_v1` |
| Treatment mask | empty |
| Control mask | canonical four-feature mask |
| Expected input/layout | 71 / v4 |

Rationale:

- H4 and 80 epochs keep the first experiment bounded.
- No active AUDCHF H4 experiment duplicates this lineage.
- Both new arms are fresh width-71 models; experiment 563 is only the configuration source.
- Initialization is deterministic seed 42. Because both arms have identical width and shape, their initialization draws match.
- Existing final profitability observation 42 proves this symbol/window is evaluable: 11,768 actionable predictions, aggregate return `0.7311843944316075`, average `0.00006213327620934802`.
- The historical width-53 result is contextual evidence only and must not be used as the causal control.

A backup pair was deliberately deferred to avoid expanding the initial scope.

## Exact queue commands — not executed

These require the implementation to be deployed to the scheduler Release binary first. The currently live binary was not rebuilt.

Control:

```bash
"/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/Build/Products/Release/LSTM_Release" \
  --queue-experiment \
  --symbol=audchfrmp \
  --prediction-horizon=4 \
  --target-epochs=80 \
  --checkpoint-interval=20 \
  --threshold=0.0008 \
  --core-lr=119.75 \
  --head-lr=25 \
  --training-objective=legacy \
  --donchian20-mode=enabled \
  --feature-warmup-scope=legacy_cold_boundary \
  --donchian-lookback=20 \
  --train-start=2010-01-01 \
  --train-end=2025-01-01 \
  --infer-start=2025-01-01 \
  --infer-end=2026-01-01 \
  --ablate-features=relevant_event_has_consensus,relevant_event_consensus_low,relevant_event_consensus_high,relevant_event_consensus_is_range
```

Treatment:

```bash
"/Volumes/Developer SSD/ExpertAdvisor/DerivedData/Development/Build/Products/Release/LSTM_Release" \
  --queue-experiment \
  --symbol=audchfrmp \
  --prediction-horizon=4 \
  --target-epochs=80 \
  --checkpoint-interval=20 \
  --threshold=0.0008 \
  --core-lr=119.75 \
  --head-lr=25 \
  --training-objective=legacy \
  --donchian20-mode=enabled \
  --feature-warmup-scope=legacy_cold_boundary \
  --donchian-lookback=20 \
  --train-start=2010-01-01 \
  --train-end=2025-01-01 \
  --infer-start=2025-01-01 \
  --infer-end=2026-01-01 \
  --allow-duplicate-experiment
```

Only treatment requires `--allow-duplicate-experiment`, because the empty-mask source configuration already exists. Global duplicate protection was not weakened.

## Later comparison plan

Compare treatment minus control for:

- final inference accuracy;
- accept accuracy;
- leader score;
- accept rate;
- predicted neutral proportion;
- profitability actionable count;
- aggregate terminal-horizon log-return sum;
- average terminal-horizon log-return per actionable prediction.

Interpretation must be joint:

- Accuracy improvement with materially worse aggregate or average profitability is not an unqualified success.
- Flat predictive metrics with materially stronger, well-supported profitability can still demonstrate economic value.
- Disagreement between aggregate and average profitability should be reported as mixed.
- Actionable count supplies the support/context for both profitability measures.

A narrow follow-up can generalize the existing paired evaluator to a feature-ablation comparison mode that permits exactly this mask difference while reusing its profitability validation and disposition rules. Event-conditioned analysis was deliberately deferred.

## Repository state

`git status --short`:

```text
 M Headers/FeatureAblation.hpp
 M Sources/ExperimentScheduler.cpp
 M Tests/EconomicEventTensorIntegrationTests.cpp
 M Tests/LSTMFeatureVectorParityTests.cpp
 M Tests/LSTMInputWidthExpansionTests.cpp
 M Tests/LSTMModelInputCompatibilityTests.cpp
```

`git diff --stat`:

```text
 Headers/FeatureAblation.hpp                   |  13 +++-
 Sources/ExperimentScheduler.cpp               |   1 +
 Tests/EconomicEventTensorIntegrationTests.cpp | 106 ++++++++++++++++++++++++++
 Tests/LSTMFeatureVectorParityTests.cpp        |  38 +++++++--
 Tests/LSTMInputWidthExpansionTests.cpp        |   7 +-
 Tests/LSTMModelInputCompatibilityTests.cpp    |  49 ++++++++++++
 6 files changed, 203 insertions(+), 11 deletions(-)
```

No commit, push, production mutation, live-binary rebuild, scheduler restart, or experiment execution occurred.