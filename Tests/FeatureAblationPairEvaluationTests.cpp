#include "FeatureAblationPairEvaluation.hpp"

#include "FeatureAblation.hpp"
#include "InferenceProfitability.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace Feature = EA::FeatureAblationPairEvaluation;
namespace Pair = EA::PairedTrainingObjectiveEvaluation;
namespace Objective = EA::TrainingObjective;
namespace Profitability = EA::InferenceProfitability;

namespace
{

Pair::ObjectiveProvenance ObjectiveProvenance()
{
    const std::string canonical = Objective::CanonicalText(Objective::Legacy());
    return {canonical, Objective::DeterministicHash(canonical)};
}

Feature::ArmEvidence Arm(long long experimentId,
                         long long modelId,
                         long long inferenceId,
                         long long observationId,
                         bool ablation)
{
    Feature::ArmEvidence result;
    auto& arm = result.authoritative;
    auto& configuration = arm.configuration;
    configuration.experimentId = experimentId;
    configuration.symbol = "audchfrmp";
    configuration.predictionHorizon = 4;
    configuration.trainStart = "2010-01-01";
    configuration.trainEnd = "2025-01-01";
    configuration.inferenceStart = "2025-01-01";
    configuration.inferenceEnd = "2026-01-01";
    configuration.targetEpochs = 80;
    configuration.threshold = 0.0008;
    configuration.coreLearningRateMultiplier = 119.75;
    configuration.headLearningRateMultiplier = 25.0;
    configuration.checkpointInterval = 20;
    configuration.inputWidth = 71;
    configuration.hiddenSize = 64;
    configuration.layerCount = 1;
    configuration.windowSize = 64;
    configuration.modelMetadataSchemaVersion = 1;
    configuration.trainConfigurationSchemaVersion = 1;
    configuration.normalizationVersion = 1;
    configuration.classWeightDown = 1.0;
    configuration.classWeightNeutral = 1.0;
    configuration.classWeightUp = 1.0;
    configuration.featureWarmupScope = "legacy_cold_boundary";
    configuration.donchianMode = "enabled";
    configuration.donchianLookback = 20;
    configuration.featureAblationMask = ablation
        ? std::string(EA::kEconomicEventConsensusAblationMaskText)
        : "";
    configuration.optimizerMetadataSchemaVersion = 1;
    configuration.optimizerType = 1;
    configuration.optimizerUpdateCount = 100;
    configuration.persistedCoreLearningRateMultiplier = 119.75;
    configuration.persistedHeadWeightLearningRateMultiplier = 25.0;
    configuration.persistedHeadBiasLearningRateMultiplier = 25.0;
    configuration.labelRuleId = 1;
    configuration.targetType = 0;
    configuration.targetScale = 1.0;
    configuration.targetStandardDeviation = 1.0;
    configuration.modelInputMetadataSchemaVersion = 1;
    configuration.modelInputLayoutVersion = 4;
    configuration.persistedTrainingSymbol = configuration.symbol;
    configuration.persistedTrainingStart = configuration.trainStart;
    configuration.persistedTrainingEnd = configuration.trainEnd;
    configuration.experimentObjective = ObjectiveProvenance();
    configuration.runProvenance = {
        "b5b925234367ede13528f6bdbfa0ae328ecc5103", "phase6", false,
        "Release", "AppleClang-18", "079", "scheduler-v1",
        "LSTM_Release"};

    result.extended.trainingObjectiveVersion = 1;
    result.extended.lossDefinitionVersion = 1;
    result.extended.auxiliaryLossMode = "disabled";
    result.extended.targetClippingDefinition = "none";
    result.extended.objectiveNormalizationIdentity =
        "weighted_loss_sum_by_weight_sum_gradients__calculate_batch_return_by_example_count_v1";
    result.extended.checkpointPolicyScope = "symbol_horizon";
    result.extended.checkpointPolicyStopMode = "next_checkpoint";
    result.extended.checkpointPolicyGraceEvaluations = 1;
    result.extended.checkpointPolicyRevision = 1;
    result.extended.configuredModelInputWidth = 71;
    result.extended.configuredModelInputLayoutVersion = 4;

    arm.experimentStatus = "completed";
    arm.experimentPhase = "done";
    arm.finalModelId = modelId;
    result.exactFinalInferenceResultId = inferenceId;

    Pair::ClassificationEvidence classification;
    classification.inferenceResultId = inferenceId;
    classification.modelId = modelId;
    classification.inferenceScope = "final";
    classification.status = "completed";
    classification.symbol = configuration.symbol;
    classification.predictionHorizon = configuration.predictionHorizon;
    classification.threshold = configuration.threshold;
    classification.windowSize = configuration.windowSize;
    classification.labelRuleId = configuration.labelRuleId;
    classification.targetType = configuration.targetType;
    classification.inferenceStart = configuration.inferenceStart;
    classification.inferenceEnd = configuration.inferenceEnd;
    classification.completedEpochs = configuration.targetEpochs;
    classification.accuracy = ablation ? 0.61 : 0.64;
    classification.predictedDownProportion = 0.30;
    classification.predictedNeutralProportion = ablation ? 0.40 : 0.35;
    classification.predictedUpProportion = ablation ? 0.30 : 0.35;
    classification.analysisId = inferenceId + 1000;
    classification.analysisExperimentId = experimentId;
    classification.analysisModelId = modelId;
    classification.analysisScope = "final";
    classification.analysisStatus = "completed";
    classification.inferenceAccuracy = classification.accuracy;
    classification.acceptAccuracy = ablation ? 0.66 : 0.70;
    classification.acceptRate = ablation ? 0.60 : 0.65;
    classification.leaderScore = ablation ? 0.55 : 0.60;
    classification.predictedDownCount = ablation ? 30 : 35;
    classification.predictedNeutralCount = ablation ? 40 : 35;
    classification.predictedUpCount = 30;
    classification.acceptedPredictionCount = ablation ? 60 : 65;
    arm.classification = classification;

    Pair::ProfitabilityEvidence profitability;
    profitability.observationId = observationId;
    profitability.experimentId = experimentId;
    profitability.modelId = modelId;
    profitability.inferenceResultId = inferenceId;
    profitability.inferenceScope = "final";
    profitability.inferenceStart = configuration.inferenceStart;
    profitability.inferenceEnd = configuration.inferenceEnd;
    profitability.predictionCount = 100;
    profitability.actionableCount = ablation ? 60 : 70;
    profitability.aggregateTerminalHorizonLogReturnSum =
        ablation ? 0.12 : 0.21;
    profitability.averageTerminalHorizonLogReturnPerActionablePrediction =
        profitability.aggregateTerminalHorizonLogReturnSum /
        static_cast<double>(profitability.actionableCount);
    profitability.metricDefinitionCanonical =
        Profitability::kMetricDefinitionCanonical;
    profitability.metricDefinitionHash = Profitability::MetricDefinitionHash();
    profitability.sourceContentHash = "fnv1a64:0000000000000001";
    profitability.observationIdentityCanonical =
        "fixture=" + std::to_string(observationId);
    profitability.observationIdentityHash = Objective::DeterministicHash(
        profitability.observationIdentityCanonical);
    arm.profitability = profitability;
    return result;
}

bool Has(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

Feature::ComparisonResult EvaluatePair(
    const Feature::ArmEvidence& control,
    const Feature::ArmEvidence& ablation,
    std::string_view expected = EA::kEconomicEventConsensusAblationMaskText)
{
    return Feature::Compare(control, ablation, expected);
}

} // namespace

int main()
{
    const Feature::ArmEvidence control = Arm(601, 1601, 2601, 3601, false);
    const Feature::ArmEvidence ablation = Arm(602, 1602, 2602, 3602, true);

    const auto complete = EvaluatePair(control, ablation);
    assert(complete.disposition == Feature::Disposition::ComparableComplete);
    assert(complete.invalidReasons.empty());
    assert(complete.canonicalAblatedFeatureSet ==
           EA::FeatureAblationMask::Parse(
               std::string(EA::kEconomicEventConsensusAblationMaskText))
               .CanonicalText());
    assert(!complete.ablationIdentityHash.empty());
    assert(Feature::EvaluationIdentityCanonical(
               control, ablation, complete)
               .starts_with("feature_ablation_pair_evaluation_v3;"));
    assert(complete.predictionCount.controlMinusAblation == 0.0);
    assert(complete.actionableCount.controlMinusAblation == 10.0);
    assert(std::fabs(*complete.aggregateProfitability.controlMinusAblation -
                     0.09) < 1.0e-15);
    assert(std::fabs(*complete.inferenceAccuracy.controlMinusAblation -
                     0.03) < 1.0e-15);
    assert(Feature::ExitCode(complete.disposition) == 0);

    const auto legacy = Feature::CompareLegacyConsensusPair(
        ablation, control);
    assert(legacy.disposition == Feature::Disposition::ComparableComplete);
    assert(legacy.aggregateProfitability.controlMinusAblation ==
           complete.aggregateProfitability.controlMinusAblation);
    assert(legacy.inferenceAccuracy.controlMinusAblation ==
           complete.inferenceAccuracy.controlMinusAblation);

    auto arbitraryControl = control;
    auto arbitraryAblation = ablation;
    arbitraryAblation.authoritative.configuration.featureAblationMask =
        "relative_tick_volume";
    assert(EvaluatePair(arbitraryControl, arbitraryAblation,
                   "relative_tick_volume")
               .disposition == Feature::Disposition::ComparableComplete);

    auto historicalControl = control;
    auto historicalAblation = ablation;
    for (auto* arm : {&historicalControl, &historicalAblation})
    {
        arm->extended.configuredModelInputWidth.reset();
        arm->extended.configuredModelInputLayoutVersion.reset();
        arm->authoritative.configuration.inputWidth = 75;
        arm->authoritative.configuration.modelInputLayoutVersion = 5;
    }
    assert(EvaluatePair(historicalControl, historicalAblation).disposition ==
           Feature::Disposition::ComparableComplete);

    // optimizerUpdateCount is post-ablation optimizer behavior, not
    // pre-ablation scientific configuration. A feature ablation can change
    // gradient finiteness and therefore the number of successful SGD updates.
    auto differentOptimizerUpdates = ablation;
    differentOptimizerUpdates.authoritative.configuration.optimizerUpdateCount =
        control.authoritative.configuration.optimizerUpdateCount - 3;
    const auto updateCountDifference =
        EvaluatePair(control, differentOptimizerUpdates);
    assert(updateCountDifference.disposition ==
           Feature::Disposition::ComparableComplete);
    assert(!Has(updateCountDifference.invalidReasons,
                "optimizer_update_count_mismatch"));
    assert(Feature::ExitCode(updateCountDifference.disposition) == 0);

    // Analysis persistence rounds inference accuracy to six decimal places.
    // A discrepancy within half of one unit in the sixth decimal remains the
    // same underlying inference result.
    auto roundedAnalysisAccuracy = ablation;
    roundedAnalysisAccuracy.authoritative.classification->accuracy =
        *roundedAnalysisAccuracy.authoritative.classification->inferenceAccuracy +
        3.7e-7;
    const auto roundedAccuracyResult =
        EvaluatePair(control, roundedAnalysisAccuracy);
    assert(roundedAccuracyResult.disposition ==
           Feature::Disposition::ComparableComplete);
    assert(!Has(roundedAccuracyResult.invalidReasons,
                "ablation_inference_analysis_accuracy_mismatch"));

    // A discrepancy beyond the six-decimal rounding boundary is a genuine
    // consistency failure and must continue to fail closed.
    auto inconsistentAnalysisAccuracy = ablation;
    inconsistentAnalysisAccuracy.authoritative.classification->accuracy =
        *inconsistentAnalysisAccuracy.authoritative.classification->inferenceAccuracy +
        6.0e-7;
    const auto inconsistentAccuracyResult =
        EvaluatePair(control, inconsistentAnalysisAccuracy);
    assert(Has(inconsistentAccuracyResult.invalidReasons,
               "ablation_inference_analysis_accuracy_mismatch"));

    auto mismatch = ablation;
    mismatch.authoritative.configuration.symbol = "eurusdrmp";
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "symbol_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.predictionHorizon = 6;
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "prediction_horizon_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.threshold = 0.0009;
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "threshold_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.coreLearningRateMultiplier = 120.0;
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "core_lr_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.trainStart = "2011-01-01";
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "train_start_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.inferenceEnd = "2026-02-01";
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "inference_end_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.featureWarmupScope =
        "full_history_warmup";
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "feature_warmup_scope_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.donchianMode = "disabled";
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "donchian_mode_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.donchianLookback = 40;
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "donchian_lookback_mismatch"));
    mismatch = ablation;
    mismatch.extended.trainingObjectiveVersion = 2;
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "training_objective_version_mismatch"));

    mismatch = ablation;
    mismatch.authoritative.configuration.featureAblationMask =
        "relative_tick_volume";
    const auto unrelatedAblation = EvaluatePair(control, mismatch);
    assert(unrelatedAblation.disposition ==
           Feature::Disposition::InvalidAblationPair);
    assert(Has(unrelatedAblation.invalidReasons,
               "ablation_mask_does_not_match_expected"));

    auto extraControl = control;
    extraControl.authoritative.configuration.featureAblationMask =
        "relative_tick_volume";
    assert(Has(EvaluatePair(extraControl, ablation).invalidReasons,
               "control_feature_ablation_mask_not_empty"));

    const auto reversed = EvaluatePair(ablation, control);
    assert(reversed.disposition == Feature::Disposition::InvalidAblationPair);
    assert(Has(reversed.invalidReasons,
               "control_feature_ablation_mask_not_empty"));

    auto pendingControl = control;
    auto pendingAblation = ablation;
    pendingControl.authoritative.experimentStatus = "running";
    pendingControl.authoritative.experimentPhase = "train";
    pendingControl.authoritative.finalModelId.reset();
    pendingControl.exactFinalInferenceResultId.reset();
    pendingControl.authoritative.classification.reset();
    pendingControl.authoritative.profitability.reset();
    pendingAblation.authoritative.experimentStatus = "pending";
    pendingAblation.authoritative.experimentPhase = "pending";
    pendingAblation.authoritative.finalModelId.reset();
    pendingAblation.exactFinalInferenceResultId.reset();
    pendingAblation.authoritative.classification.reset();
    pendingAblation.authoritative.profitability.reset();
    const auto pending = EvaluatePair(pendingControl, pendingAblation);
    assert(pending.disposition ==
           Feature::Disposition::ComparableIncomplete);
    assert(Feature::ExitCode(pending.disposition) == 4);

    auto noProfitability = ablation;
    noProfitability.authoritative.profitability.reset();
    const auto unavailable = EvaluatePair(control, noProfitability);
    assert(unavailable.disposition ==
           Feature::Disposition::ProfitabilityEvidenceUnavailable);
    assert(Has(unavailable.incompleteReasons,
               "ablation_final_profitability_evidence_unavailable"));
    assert(unavailable.inferenceAccuracy.controlMinusAblation.has_value());

    auto zeroActionable = ablation;
    zeroActionable.authoritative.profitability->actionableCount = 0;
    zeroActionable.authoritative.profitability
        ->aggregateTerminalHorizonLogReturnSum = 0.0;
    zeroActionable.authoritative.profitability
        ->averageTerminalHorizonLogReturnPerActionablePrediction.reset();
    const auto zeroActionableResult = EvaluatePair(control, zeroActionable);
    assert(zeroActionableResult.disposition ==
           Feature::Disposition::ComparableComplete);
    assert(zeroActionableResult.averageProfitability.ablation == std::nullopt);
    assert(zeroActionableResult.averageProfitability.controlMinusAblation ==
           std::nullopt);

    auto noInference = ablation;
    noInference.authoritative.classification.reset();
    noInference.authoritative.profitability.reset();
    noInference.exactFinalInferenceResultId.reset();
    assert(EvaluatePair(control, noInference).disposition ==
           Feature::Disposition::MissingFinalInference);

    auto checkpointOnly = ablation;
    checkpointOnly.authoritative.classification->inferenceScope = "checkpoint";
    checkpointOnly.authoritative.classification->checkpointEvalId = 77;
    assert(Has(EvaluatePair(control, checkpointOnly).invalidReasons,
               "ablation_classification_not_exact_final_scope"));

    // Layout 6 is same-width historical evidence, never corrected evidence.
    auto surpriseControl = control;
    auto surpriseAblation = ablation;
    surpriseControl.authoritative.configuration.experimentId = 619;
    surpriseAblation.authoritative.configuration.experimentId = 620;
    surpriseControl.authoritative.configuration.symbol = "eurusdrmp";
    surpriseAblation.authoritative.configuration.symbol = "eurusdrmp";
    surpriseControl.authoritative.configuration.featureAblationMask.clear();
    surpriseAblation.authoritative.configuration.featureAblationMask =
        std::string(EA::kCausalEconomicEventSurpriseAblationMaskText);
    for (auto* arm : {&surpriseControl, &surpriseAblation})
    {
        arm->extended.configuredModelInputWidth = 77;
        arm->extended.configuredModelInputLayoutVersion = 6;
        arm->extended.economicCalendarSnapshotId = 1;
        arm->extended.economicCalendarSnapshotHash =
            "fnv1a64:67610f94f5c8e7cc";
        arm->authoritative.configuration.inputWidth = 77;
        arm->authoritative.configuration.modelInputLayoutVersion = 6;
        arm->authoritative.classification->symbol = "eurusdrmp";
        arm->authoritative.classification->analysisExperimentId =
            arm->authoritative.configuration.experimentId;
        arm->authoritative.profitability->experimentId =
            arm->authoritative.configuration.experimentId;
    }
    const auto surprise = EvaluatePair(
        surpriseControl, surpriseAblation,
        "causal_first_release_surprise,"
        "causal_first_release_surprise_available,"
        "causal_first_release_surprise");
    assert(surprise.disposition == Feature::Disposition::ComparableComplete);
    assert(surprise.evidenceClassification ==
           Feature::EvidenceClassification::PreFixCausalSurpriseEvidence);
    assert(surprise.canonicalAblatedFeatureSet ==
           EA::kCausalEconomicEventSurpriseAblationMaskText);

    auto correctedControl = surpriseControl;
    auto correctedAblation = surpriseAblation;
    correctedControl.authoritative.configuration.experimentId = 624;
    correctedAblation.authoritative.configuration.experimentId = 625;
    for (auto* arm : {&correctedControl, &correctedAblation})
    {
        arm->extended.configuredModelInputLayoutVersion = 7;
        arm->authoritative.configuration.modelInputLayoutVersion = 7;
        arm->authoritative.classification->analysisExperimentId =
            arm->authoritative.configuration.experimentId;
        arm->authoritative.profitability->experimentId =
            arm->authoritative.configuration.experimentId;
    }
    const auto corrected = EvaluatePair(
        correctedControl, correctedAblation,
        EA::kCausalEconomicEventSurpriseAblationMaskText);
    assert(corrected.disposition == Feature::Disposition::ComparableComplete);
    assert(corrected.evidenceClassification == Feature::EvidenceClassification::
               CorrectedCausalSurprisePairEvidence);

    auto missingInputIdentityControl = correctedControl;
    auto missingInputIdentityAblation = correctedAblation;
    missingInputIdentityControl.extended.configuredModelInputWidth.reset();
    missingInputIdentityControl.extended.configuredModelInputLayoutVersion.reset();
    missingInputIdentityAblation.extended.configuredModelInputWidth.reset();
    missingInputIdentityAblation.extended.configuredModelInputLayoutVersion.reset();
    assert(Has(EvaluatePair(
                   missingInputIdentityControl, missingInputIdentityAblation,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "control_causal_surprise_model_input_identity_missing"));

    auto wrongWidthControl = correctedControl;
    auto wrongWidthAblation = correctedAblation;
    for (auto* arm : {&wrongWidthControl, &wrongWidthAblation})
    {
        arm->extended.configuredModelInputWidth = 76;
        arm->authoritative.configuration.inputWidth = 76;
    }
    assert(Has(EvaluatePair(
                   wrongWidthControl, wrongWidthAblation,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "control_causal_surprise_model_input_width_not_77"));

    auto correctedControlWithAblation = correctedControl;
    correctedControlWithAblation.authoritative.configuration
        .featureAblationMask = "relative_tick_volume";
    assert(Has(EvaluatePair(
                   correctedControlWithAblation, correctedAblation,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "control_feature_ablation_mask_not_empty"));

    auto correctedScientificMismatch = correctedAblation;
    correctedScientificMismatch.authoritative.configuration.threshold = 0.0009;
    assert(Has(EvaluatePair(
                   correctedControl, correctedScientificMismatch,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "threshold_mismatch"));
    correctedScientificMismatch = correctedAblation;
    correctedScientificMismatch.authoritative.configuration
        .headLearningRateMultiplier = 24.0;
    assert(Has(EvaluatePair(
                   correctedControl, correctedScientificMismatch,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "head_lr_mismatch"));
    correctedScientificMismatch = correctedAblation;
    correctedScientificMismatch.authoritative.configuration.trainEnd =
        "2024-12-31";
    assert(Has(EvaluatePair(
                   correctedControl, correctedScientificMismatch,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "train_end_mismatch"));
    correctedScientificMismatch = correctedAblation;
    correctedScientificMismatch.authoritative.configuration
        .experimentObjective.canonical += "different=true;";
    correctedScientificMismatch.authoritative.configuration
        .experimentObjective.hash = Objective::DeterministicHash(
            correctedScientificMismatch.authoritative.configuration
                .experimentObjective.canonical);
    assert(Has(EvaluatePair(
                   correctedControl, correctedScientificMismatch,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "training_objective_mismatch"));

    auto noncanonicalSurpriseAblation = correctedAblation;
    noncanonicalSurpriseAblation.authoritative.configuration
        .featureAblationMask =
            " causal_first_release_surprise,"
            "causal_first_release_surprise_available,"
            "causal_first_release_surprise ";
    const auto canonicalizedPersistedMask = EvaluatePair(
        correctedControl, noncanonicalSurpriseAblation,
        EA::kCausalEconomicEventSurpriseAblationMaskText);
    assert(canonicalizedPersistedMask.disposition ==
           Feature::Disposition::ComparableComplete);
    assert(canonicalizedPersistedMask.canonicalAblatedFeatureSet ==
           EA::kCausalEconomicEventSurpriseAblationMaskText);

    auto oneChannel = correctedAblation;
    oneChannel.authoritative.configuration.featureAblationMask =
        "causal_first_release_surprise";
    assert(Has(EvaluatePair(correctedControl, oneChannel,
                       EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "ablation_mask_does_not_match_expected"));

    auto extraChannel = correctedAblation;
    extraChannel.authoritative.configuration.featureAblationMask +=
        ",relative_tick_volume";
    assert(Has(EvaluatePair(correctedControl, extraChannel,
                       EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "ablation_mask_does_not_match_expected"));

    auto widthMismatch = correctedAblation;
    widthMismatch.extended.configuredModelInputWidth = 75;
    widthMismatch.authoritative.configuration.inputWidth = 75;
    assert(Has(EvaluatePair(correctedControl, widthMismatch,
                       EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "configured_model_input_width_mismatch"));

    auto layoutMismatch = correctedAblation;
    layoutMismatch.extended.configuredModelInputLayoutVersion = 6;
    layoutMismatch.authoritative.configuration.modelInputLayoutVersion = 6;
    const auto mixedLayouts = EvaluatePair(
        correctedControl, layoutMismatch,
        EA::kCausalEconomicEventSurpriseAblationMaskText);
    assert(Has(mixedLayouts.invalidReasons,
               "configured_model_input_layout_mismatch"));
    assert(mixedLayouts.evidenceClassification ==
           Feature::EvidenceClassification::IncompatibleOrInvalidEvidence);

    // Snapshot identity is a matched within-pair reproducibility boundary,
    // never the feature treatment. Corrected layout-7 evidence requires a
    // complete immutable identity and fails closed on any difference.
    assert(EvaluatePair(
               correctedControl, correctedAblation,
               EA::kCausalEconomicEventSurpriseAblationMaskText)
               .disposition == Feature::Disposition::ComparableComplete);
    auto differentSnapshotId = correctedAblation;
    differentSnapshotId.extended.economicCalendarSnapshotId = 2;
    assert(Has(EvaluatePair(
                   correctedControl, differentSnapshotId,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "economic_calendar_snapshot_id_mismatch"));
    auto differentSnapshotHash = correctedAblation;
    differentSnapshotHash.extended.economicCalendarSnapshotHash =
        "fnv1a64:0000000000000001";
    assert(Has(EvaluatePair(
                   correctedControl, differentSnapshotHash,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "economic_calendar_snapshot_hash_mismatch"));
    auto mixedCorpus = correctedAblation;
    mixedCorpus.extended.economicCalendarSnapshotId.reset();
    mixedCorpus.extended.economicCalendarSnapshotHash.reset();
    assert(Has(EvaluatePair(
                   surpriseControl, mixedCorpus,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "economic_calendar_snapshot_id_mismatch"));
    auto missingSnapshotsControl = correctedControl;
    auto missingSnapshotsAblation = correctedAblation;
    missingSnapshotsControl.extended.economicCalendarSnapshotId.reset();
    missingSnapshotsControl.extended.economicCalendarSnapshotHash.reset();
    missingSnapshotsAblation.extended.economicCalendarSnapshotId.reset();
    missingSnapshotsAblation.extended.economicCalendarSnapshotHash.reset();
    assert(Has(EvaluatePair(
                   missingSnapshotsControl, missingSnapshotsAblation,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "control_corrected_causal_surprise_snapshot_identity_missing"));
    auto partialSnapshot = correctedAblation;
    partialSnapshot.extended.economicCalendarSnapshotHash.reset();
    assert(Has(EvaluatePair(
                   correctedControl, partialSnapshot,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "ablation_economic_calendar_snapshot_identity_invalid"));

    auto configuredFinalWidthMismatch = correctedAblation;
    configuredFinalWidthMismatch.authoritative.configuration.inputWidth = 75;
    assert(Has(EvaluatePair(
                   correctedControl, configuredFinalWidthMismatch,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "ablation_configured_final_model_input_width_mismatch"));

    auto configuredFinalLayoutMismatch = correctedAblation;
    configuredFinalLayoutMismatch.authoritative.configuration
        .modelInputLayoutVersion = 5;
    assert(Has(EvaluatePair(
                   correctedControl, configuredFinalLayoutMismatch,
                   EA::kCausalEconomicEventSurpriseAblationMaskText)
                   .invalidReasons,
               "ablation_configured_final_model_input_layout_mismatch"));

    mismatch = ablation;
    mismatch.authoritative.configuration.hiddenSize = 32;
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "hidden_size_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.experimentObjective.canonical +=
        "changed=true;";
    mismatch.authoritative.configuration.experimentObjective.hash =
        Objective::DeterministicHash(
            mismatch.authoritative.configuration.experimentObjective.canonical);
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "training_objective_mismatch"));
    mismatch = ablation;
    mismatch.extended.freshInitializationSeed = 43U;
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "fresh_initialization_seed_mismatch"));
    mismatch = ablation;
    mismatch.authoritative.configuration.resumeModelId = 42;
    mismatch.extended.freshInitializationSeed.reset();
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "resume_model_id_mismatch"));
    mismatch = ablation;
    mismatch.extended.continuationPolicyEnabled = true;
    mismatch.extended.continuationPolicyScientificIdentity = "policy-a";
    assert(Has(EvaluatePair(control, mismatch).invalidReasons,
               "continuation_policy_enabled_mismatch"));

    bool unknownRejected = false;
    try
    {
        const auto unknown = EvaluatePair(control, ablation, "unknown_feature");
        unknownRejected = unknown.disposition ==
            Feature::Disposition::InvalidAblationPair &&
            Has(unknown.invalidReasons, "feature_ablation_mask_invalid");
    }
    catch (...) {}
    assert(unknownRejected);

    const auto emptyExpectedMask = EvaluatePair(control, ablation, "");
    assert(emptyExpectedMask.disposition ==
           Feature::Disposition::InvalidAblationPair);
    assert(Has(emptyExpectedMask.invalidReasons,
               "expected_ablation_mask_empty"));

    auto operationallyDifferent = ablation;
    operationallyDifferent.operational.schedulerPriority = "high";
    operationallyDifferent.operational.workerPid = 12345;
    assert(EvaluatePair(control, operationallyDifferent).disposition ==
           Feature::Disposition::ComparableComplete);

    assert(Feature::ParseExperimentIdPair("601:602") ==
           std::make_pair(601LL, 602LL));
    for (const std::string invalid :
         {"", "601", ":602", "601:", "0:602", "601:0", "601:602:603",
          "601x:602"})
    {
        bool rejected = false;
        try
        {
            (void)Feature::ParseExperimentIdPair(invalid);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        assert(rejected);
    }

    std::cout << "FeatureAblationPairEvaluationTests passed\n";
    return 0;
}
