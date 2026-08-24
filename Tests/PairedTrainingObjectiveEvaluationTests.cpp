#include "PairedTrainingObjectiveEvaluation.hpp"

#include "InferenceProfitability.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace Pair = EA::PairedTrainingObjectiveEvaluation;
namespace Objective = EA::TrainingObjective;
namespace Profitability = EA::InferenceProfitability;

namespace
{

Pair::ObjectiveProvenance Provenance(
    const Objective::Configuration& objective)
{
    const std::string canonical = Objective::CanonicalText(objective);
    return {canonical, Objective::DeterministicHash(canonical)};
}

Pair::ArmEvidence Arm(long long experimentId,
                      long long modelId,
                      long long inferenceId,
                      long long observationId,
                      const Objective::Configuration& objective,
                      double aggregate,
                      std::uint64_t actionable,
                      double accuracy = 0.70,
                      double neutral = 0.30)
{
    Pair::ArmEvidence arm;
    auto& c = arm.configuration;
    c.experimentId = experimentId;
    c.symbol = "usdcadrmp";
    c.predictionHorizon = 6;
    c.trainStart = "2020-01-01";
    c.trainEnd = "2025-01-01";
    c.inferenceStart = "2025-01-02";
    c.inferenceEnd = "2026-01-01";
    c.targetEpochs = 80;
    c.threshold = 0.0008;
    c.coreLearningRateMultiplier = 1.0;
    c.headLearningRateMultiplier = 1.0;
    c.checkpointInterval = 20;
    c.inputWidth = 50;
    c.hiddenSize = 64;
    c.layerCount = 1;
    c.windowSize = 64;
    c.modelMetadataSchemaVersion = 1;
    c.trainConfigurationSchemaVersion = 1;
    c.normalizationVersion = 1;
    c.classWeightDown = 1.0;
    c.classWeightNeutral = 1.0;
    c.classWeightUp = 1.0;
    c.featureWarmupScope = "full_history_warmup";
    c.donchianMode = "enabled";
    c.donchianLookback = 20;
    c.featureAblationMask = "";
    c.optimizerMetadataSchemaVersion = 1;
    c.optimizerType = 1;
    c.optimizerUpdateCount = 100;
    c.persistedCoreLearningRateMultiplier = 1.0;
    c.persistedHeadWeightLearningRateMultiplier = 1.0;
    c.persistedHeadBiasLearningRateMultiplier = 1.0;
    c.labelRuleId = 1;
    c.targetType = 1;
    c.targetScale = 1.0;
    c.targetStandardDeviation = 1.0;
    c.modelInputMetadataSchemaVersion = 1;
    c.modelInputLayoutVersion = 1;
    c.persistedTrainingSymbol = c.symbol;
    c.persistedTrainingStart = c.trainStart;
    c.persistedTrainingEnd = c.trainEnd;
    c.experimentObjective = Provenance(objective);
    c.runProvenance = {
        "abc123", "phase6", false, "Release", "AppleClang-18",
        "079", "scheduler-v1", "LSTM_Release"};

    arm.experimentStatus = "completed";
    arm.experimentPhase = "done";
    arm.finalModelId = modelId;
    arm.materializedModelObjectives = {
        {modelId - 1, false, Provenance(objective)},
        {modelId, true, Provenance(objective)}};
    arm.runtimeObjective = {
        "TRAINING_OBJECTIVE_ACTIVE", objective.objectiveIdentifier,
        Objective::Identity(objective)};

    Pair::ClassificationEvidence classification;
    classification.inferenceResultId = inferenceId;
    classification.modelId = modelId;
    classification.inferenceScope = "final";
    classification.status = "completed";
    classification.symbol = c.symbol;
    classification.predictionHorizon = c.predictionHorizon;
    classification.threshold = c.threshold;
    classification.windowSize = c.windowSize;
    classification.labelRuleId = c.labelRuleId;
    classification.targetType = c.targetType;
    classification.inferenceStart = c.inferenceStart;
    classification.inferenceEnd = c.inferenceEnd;
    classification.completedEpochs = c.targetEpochs;
    classification.accuracy = accuracy;
    classification.predictedDownProportion = (1.0 - neutral) / 2.0;
    classification.predictedNeutralProportion = neutral;
    classification.predictedUpProportion = (1.0 - neutral) / 2.0;
    classification.analysisId = inferenceId + 1000;
    classification.analysisExperimentId = experimentId;
    classification.analysisModelId = modelId;
    classification.analysisScope = "final";
    classification.analysisStatus = "completed";
    classification.inferenceAccuracy = accuracy;
    classification.acceptAccuracy = accuracy + 0.05;
    classification.acceptRate = 1.0 - neutral;
    classification.leaderScore = accuracy * 0.9;
    arm.classification = classification;

    Pair::ProfitabilityEvidence profitability;
    profitability.observationId = observationId;
    profitability.experimentId = experimentId;
    profitability.modelId = modelId;
    profitability.inferenceResultId = inferenceId;
    profitability.inferenceScope = "final";
    profitability.inferenceStart = c.inferenceStart;
    profitability.inferenceEnd = c.inferenceEnd;
    profitability.predictionCount = 100;
    profitability.actionableCount = actionable;
    profitability.aggregateTerminalHorizonLogReturnSum = aggregate;
    if (actionable != 0)
        profitability.averageTerminalHorizonLogReturnPerActionablePrediction =
            aggregate / static_cast<double>(actionable);
    profitability.metricDefinitionCanonical =
        Profitability::kMetricDefinitionCanonical;
    profitability.metricDefinitionHash =
        Profitability::MetricDefinitionHash();
    profitability.sourceContentHash = "fnv1a64:0000000000000001";
    profitability.observationIdentityCanonical =
        "test_observation_identity=" + std::to_string(observationId);
    profitability.observationIdentityHash = Objective::DeterministicHash(
        profitability.observationIdentityCanonical);
    arm.profitability = profitability;
    return arm;
}

Pair::MaterialityPolicy Policy()
{
    Pair::MaterialityPolicy policy;
    policy.minimumProfitabilityImprovement = 0.01;
    policy.maximumProfitabilityWorsening = 0.01;
    policy.classification = Pair::ClassificationDegradationPolicy{
        0.02, 0.02, 0.05, 0.02, 0.05};
    return policy;
}

Pair::ComparisonResult Compare(const Pair::ArmEvidence& control,
                               const Pair::ArmEvidence& treatment)
{
    return Pair::Compare(control, treatment, Policy());
}

bool Has(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

} // namespace

int main()
{
    const auto baseControl = Arm(
        990001, 1990001, 2990001, 3990001,
        Objective::Legacy(), 0.10, 50);
    const auto baseTreatment = Arm(
        990002, 1990002, 2990002, 3990002,
        Objective::ProfitabilityAuxiliary(), 0.20, 50);

    // Identical scientific configuration with the recognized objective-only
    // difference is valid and promising under the explicit fixture policy.
    const auto promising = Compare(baseControl, baseTreatment);
    assert(promising.disposition == Pair::Disposition::Promising);
    assert(promising.invalidReasons.empty());
    assert(promising.aggregateProfitability.treatmentMinusControl == 0.10);
    assert(Pair::DispositionText(promising.disposition) == "PROMISING");

    auto sameObjective = baseTreatment;
    sameObjective.configuration.experimentObjective =
        baseControl.configuration.experimentObjective;
    sameObjective.materializedModelObjectives = {
        {1990000, false, Provenance(Objective::Legacy())},
        {1990002, true, Provenance(Objective::Legacy())}};
    sameObjective.runtimeObjective = {
        "TRAINING_OBJECTIVE_ACTIVE",
        Objective::Legacy().objectiveIdentifier,
        Objective::Identity(Objective::Legacy())};
    assert(Compare(baseControl, sameObjective).disposition ==
           Pair::Disposition::InvalidComparison);

    auto mismatch = baseTreatment;
    mismatch.configuration.symbol = "eurusdrmp";
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "symbol_mismatch"));
    mismatch = baseTreatment;
    mismatch.configuration.predictionHorizon = 12;
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "prediction_horizon_mismatch"));
    mismatch = baseTreatment;
    mismatch.configuration.trainEnd = "2024-12-31";
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "train_end_mismatch"));
    mismatch = baseTreatment;
    mismatch.configuration.threshold = 0.0008000000000001;
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "threshold_mismatch"));
    mismatch = baseTreatment;
    mismatch.configuration.coreLearningRateMultiplier = 1.0001;
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "core_lr_mismatch"));
    mismatch = baseTreatment;
    mismatch.configuration.persistedHeadBiasLearningRateMultiplier = 1.0001;
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "persisted_head_bias_lr_mismatch"));
    mismatch = baseTreatment;
    mismatch.configuration.inputWidth = 49;
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "input_width_mismatch"));
    mismatch = baseTreatment;
    mismatch.configuration.featureAblationMask = "return_surprise";
    assert(Has(Compare(baseControl, mismatch).invalidReasons,
               "feature_ablation_mask_mismatch"));

    // experiment.c_next_threshold is double precision, while the model and
    // exact inference context persist the original float train metadata.  The
    // repository's established 1e-7 boundary accepts this ordinary 0.0008
    // round trip, but not a value outside that boundary.
    auto floatRoundTrip = baseTreatment;
    floatRoundTrip.classification->threshold =
        static_cast<double>(static_cast<float>(0.0008));
    assert(floatRoundTrip.classification->threshold !=
           floatRoundTrip.configuration.threshold);
    assert(Compare(baseControl, floatRoundTrip).disposition ==
           Pair::Disposition::Promising);
    floatRoundTrip.classification->threshold =
        floatRoundTrip.configuration.threshold + 1.0000001e-7;
    const auto outsideThresholdTolerance =
        Compare(baseControl, floatRoundTrip);
    assert(outsideThresholdTolerance.disposition ==
           Pair::Disposition::InvalidComparison);
    assert(Has(outsideThresholdTolerance.invalidReasons,
               "treatment_classification_context_mismatch"));

    auto objectiveMismatch = baseTreatment;
    objectiveMismatch.materializedModelObjectives[1].objective =
        Provenance(Objective::Legacy());
    assert(Has(Compare(baseControl, objectiveMismatch).invalidReasons,
               "treatment_experiment_model_objective_mismatch"));

    auto canonicalObjectiveMismatch = baseTreatment;
    canonicalObjectiveMismatch.configuration.experimentObjective.canonical =
        Objective::CanonicalText(Objective::ProfitabilityAuxiliary()) + " ";
    canonicalObjectiveMismatch.configuration.experimentObjective.hash =
        Objective::DeterministicHash(
            canonicalObjectiveMismatch.configuration.experimentObjective.canonical);
    assert(Has(Compare(baseControl, canonicalObjectiveMismatch).invalidReasons,
               "treatment_experiment_objective_invalid"));

    auto incomplete = baseTreatment;
    incomplete.experimentStatus = "running";
    incomplete.experimentPhase = "train";
    assert(Compare(baseControl, incomplete).disposition ==
           Pair::Disposition::Incomplete);

    auto missingProfitability = baseTreatment;
    missingProfitability.profitability.reset();
    const auto missingProfitabilityResult =
        Compare(baseControl, missingProfitability);
    assert(missingProfitabilityResult.disposition ==
           Pair::Disposition::Incomplete);
    assert(Has(missingProfitabilityResult.incompleteReasons,
               "treatment_final_profitability_missing"));

    auto checkpointOnly = baseTreatment;
    checkpointOnly.profitability->inferenceScope = "checkpoint";
    checkpointOnly.profitability->checkpointEvalId = 42;
    const auto checkpointOnlyResult = Compare(baseControl, checkpointOnly);
    assert(checkpointOnlyResult.disposition ==
           Pair::Disposition::InvalidComparison);
    assert(Has(checkpointOnlyResult.invalidReasons,
               "treatment_profitability_not_exact_final_scope"));

    const auto worseTreatment = Arm(
        990002, 1990002, 2990002, 3990002,
        Objective::ProfitabilityAuxiliary(), -0.20, 50);
    const auto worse = Compare(baseControl, worseTreatment);
    assert(std::abs(*worse.aggregateProfitability.treatmentMinusControl + 0.30) <
           1.0e-12);
    assert(worse.disposition == Pair::Disposition::NotPromising);

    const auto zeroControl = Arm(
        990001, 1990001, 2990001, 3990001,
        Objective::Legacy(), 0.0, 0);
    const auto zeroTreatment = Arm(
        990002, 1990002, 2990002, 3990002,
        Objective::ProfitabilityAuxiliary(), 0.0, 0);
    const auto zero = Compare(zeroControl, zeroTreatment);
    assert(zero.disposition == Pair::Disposition::Mixed);
    assert(!zero.averageProfitability.treatmentMinusControl);

    auto degradedTreatment = baseTreatment;
    degradedTreatment.classification->accuracy = 0.60;
    degradedTreatment.classification->inferenceAccuracy = 0.60;
    assert(Compare(baseControl, degradedTreatment).disposition ==
           Pair::Disposition::NotPromising);

    auto mixedTreatment = baseTreatment;
    mixedTreatment.profitability->predictionCount = 250;
    mixedTreatment.profitability->actionableCount = 200;
    mixedTreatment.profitability->averageTerminalHorizonLogReturnPerActionablePrediction =
        0.001;
    // Aggregate improves, average worsens: deterministic MIXED.
    assert(Compare(baseControl, mixedTreatment).disposition ==
           Pair::Disposition::Mixed);

    auto noRuntime = baseTreatment;
    noRuntime.runtimeObjective.reset();
    assert(Compare(baseControl, noRuntime).disposition ==
           Pair::Disposition::Promising);

    std::cout << "paired_training_objective_evaluation_tests_passed\n";
    return 0;
}
