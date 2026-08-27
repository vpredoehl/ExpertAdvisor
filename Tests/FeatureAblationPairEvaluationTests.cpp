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
                         bool control)
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
    configuration.featureAblationMask = control
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
    classification.accuracy = control ? 0.61 : 0.64;
    classification.predictedDownProportion = 0.30;
    classification.predictedNeutralProportion = control ? 0.40 : 0.35;
    classification.predictedUpProportion = control ? 0.30 : 0.35;
    classification.analysisId = inferenceId + 1000;
    classification.analysisExperimentId = experimentId;
    classification.analysisModelId = modelId;
    classification.analysisScope = "final";
    classification.analysisStatus = "completed";
    classification.inferenceAccuracy = classification.accuracy;
    classification.acceptAccuracy = control ? 0.66 : 0.70;
    classification.acceptRate = control ? 0.60 : 0.65;
    classification.leaderScore = control ? 0.55 : 0.60;
    arm.classification = classification;

    Pair::ProfitabilityEvidence profitability;
    profitability.observationId = observationId;
    profitability.experimentId = experimentId;
    profitability.modelId = modelId;
    profitability.inferenceResultId = inferenceId;
    profitability.inferenceScope = "final";
    profitability.inferenceStart = configuration.inferenceStart;
    profitability.inferenceEnd = configuration.inferenceEnd;
    profitability.predictionCount = control ? 100 : 110;
    profitability.actionableCount = control ? 60 : 70;
    profitability.aggregateTerminalHorizonLogReturnSum =
        control ? 0.12 : 0.21;
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

} // namespace

int main()
{
    const Feature::ArmEvidence control = Arm(601, 1601, 2601, 3601, true);
    const Feature::ArmEvidence treatment = Arm(602, 1602, 2602, 3602, false);

    const auto complete = Feature::Compare(control, treatment);
    assert(complete.disposition == Feature::Disposition::ComparableComplete);
    assert(complete.invalidReasons.empty());
    assert(complete.canonicalAblatedFeatureSet ==
           EA::FeatureAblationMask::Parse(
               std::string(EA::kEconomicEventConsensusAblationMaskText))
               .CanonicalText());
    assert(!complete.ablationIdentityHash.empty());
    assert(complete.predictionCount.treatmentMinusControl == 10.0);
    assert(complete.actionableCount.treatmentMinusControl == 10.0);
    assert(std::fabs(*complete.aggregateProfitability.treatmentMinusControl -
                     0.09) < 1.0e-15);
    assert(std::fabs(*complete.inferenceAccuracy.treatmentMinusControl -
                     0.03) < 1.0e-15);
    assert(Feature::ExitCode(complete.disposition) == 0);

    auto mismatch = treatment;
    mismatch.authoritative.configuration.symbol = "eurusdrmp";
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "symbol_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.predictionHorizon = 6;
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "prediction_horizon_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.threshold = 0.0009;
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "threshold_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.coreLearningRateMultiplier = 120.0;
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "core_lr_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.trainStart = "2011-01-01";
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "train_start_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.inferenceEnd = "2026-02-01";
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "inference_end_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.featureWarmupScope =
        "full_history_warmup";
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "feature_warmup_scope_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.donchianMode = "disabled";
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "donchian_mode_mismatch"));
    mismatch = treatment;
    mismatch.authoritative.configuration.donchianLookback = 40;
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "donchian_lookback_mismatch"));
    mismatch = treatment;
    mismatch.extended.trainingObjectiveVersion = 2;
    assert(Has(Feature::Compare(control, mismatch).invalidReasons,
               "training_objective_version_mismatch"));

    mismatch = treatment;
    mismatch.authoritative.configuration.featureAblationMask =
        "relative_tick_volume";
    const auto unrelatedTreatment = Feature::Compare(control, mismatch);
    assert(unrelatedTreatment.disposition ==
           Feature::Disposition::InvalidAblationPair);
    assert(Has(unrelatedTreatment.invalidReasons,
               "treatment_contains_feature_ablations"));

    auto extraControl = control;
    extraControl.authoritative.configuration.featureAblationMask +=
        ",relative_tick_volume";
    assert(Has(Feature::Compare(extraControl, treatment).invalidReasons,
               "control_ablation_not_consensus_feature_family"));

    const auto reversed = Feature::Compare(treatment, control);
    assert(reversed.disposition == Feature::Disposition::InvalidAblationPair);
    assert(Has(reversed.invalidReasons, "reversed_control_treatment_order"));

    auto pendingControl = control;
    auto pendingTreatment = treatment;
    pendingControl.authoritative.experimentStatus = "running";
    pendingControl.authoritative.experimentPhase = "train";
    pendingControl.authoritative.finalModelId.reset();
    pendingControl.exactFinalInferenceResultId.reset();
    pendingControl.authoritative.classification.reset();
    pendingControl.authoritative.profitability.reset();
    pendingTreatment.authoritative.experimentStatus = "pending";
    pendingTreatment.authoritative.experimentPhase = "pending";
    pendingTreatment.authoritative.finalModelId.reset();
    pendingTreatment.exactFinalInferenceResultId.reset();
    pendingTreatment.authoritative.classification.reset();
    pendingTreatment.authoritative.profitability.reset();
    const auto pending = Feature::Compare(pendingControl, pendingTreatment);
    assert(pending.disposition ==
           Feature::Disposition::ComparableIncomplete);
    assert(Feature::ExitCode(pending.disposition) == 4);

    auto noProfitability = treatment;
    noProfitability.authoritative.profitability.reset();
    const auto unavailable = Feature::Compare(control, noProfitability);
    assert(unavailable.disposition ==
           Feature::Disposition::ProfitabilityEvidenceUnavailable);
    assert(Has(unavailable.incompleteReasons,
               "treatment_final_profitability_evidence_unavailable"));

    auto noInference = treatment;
    noInference.authoritative.classification.reset();
    noInference.authoritative.profitability.reset();
    noInference.exactFinalInferenceResultId.reset();
    assert(Feature::Compare(control, noInference).disposition ==
           Feature::Disposition::MissingFinalInference);

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
