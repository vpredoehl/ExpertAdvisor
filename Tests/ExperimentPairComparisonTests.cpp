#include "ExperimentPairComparison.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace Comparison = EA::ExperimentPairComparison;
namespace Feature = EA::FeatureAblationPairEvaluation;
namespace Shared = EA::PairedTrainingObjectiveEvaluation;

namespace
{

Feature::ArmEvidence Arm(long long experimentId,
                         std::string featureAblationMask,
                         double inferenceAccuracy,
                         double acceptRate,
                         double aggregateReturn,
                         std::uint64_t actionableCount,
                         std::uint64_t winningCount)
{
    Feature::ArmEvidence result;
    auto& shared = result.authoritative;
    auto& configuration = shared.configuration;
    configuration.experimentId = experimentId;
    configuration.symbol = "cadchfrmp";
    configuration.predictionHorizon = 4;
    configuration.trainStart = "2010-01-01";
    configuration.trainEnd = "2025-01-01";
    configuration.inferenceStart = "2025-01-01";
    configuration.inferenceEnd = "2026-01-01";
    configuration.targetEpochs = 20;
    configuration.threshold = 0.0008;
    configuration.coreLearningRateMultiplier = 120.0;
    configuration.headLearningRateMultiplier = 25.0;
    configuration.checkpointInterval = 20;
    configuration.inputWidth = 80;
    configuration.hiddenSize = 64;
    configuration.layerCount = 1;
    configuration.windowSize = 64;
    configuration.modelMetadataSchemaVersion = 1;
    configuration.trainConfigurationSchemaVersion = 1;
    configuration.normalizationVersion = 1;
    configuration.classWeightDown = 1.0;
    configuration.classWeightNeutral = 1.0;
    configuration.classWeightUp = 1.0;
    configuration.featureWarmupScope = "full_history_warmup";
    configuration.donchianMode = "enabled";
    configuration.donchianLookback = 20;
    configuration.featureAblationMask = std::move(featureAblationMask);
    configuration.optimizerMetadataSchemaVersion = 1;
    configuration.optimizerType = 1;
    configuration.optimizerFirstMomentBufferCount = 4;
    configuration.optimizerSecondMomentBufferCount = 4;
    configuration.persistedCoreLearningRateMultiplier = 120.0;
    configuration.persistedHeadWeightLearningRateMultiplier = 25.0;
    configuration.persistedHeadBiasLearningRateMultiplier = 25.0;
    configuration.labelRuleId = 1;
    configuration.targetType = 0;
    configuration.targetScale = 1.0;
    configuration.targetStandardDeviation = 1.0;
    configuration.modelInputMetadataSchemaVersion = 1;
    configuration.modelInputLayoutVersion = 8;
    configuration.persistedTrainingSymbol = configuration.symbol;
    configuration.persistedTrainingStart = configuration.trainStart;
    configuration.persistedTrainingEnd = configuration.trainEnd;
    configuration.experimentObjective = {
        "objective_id=legacy_first_hit_weighted_ce_v1;",
        "fnv1a64:0000000000000001"};

    result.extended.configuredModelInputWidth = 80;
    result.extended.configuredModelInputLayoutVersion = 8;
    result.extended.economicCalendarSnapshotId = 1;
    result.extended.economicCalendarSnapshotHash =
        "fnv1a64:0000000000000002";
    result.extended.freshInitializationSeed = 43U;
    result.extended.trainingObjectiveVersion = 1;
    result.extended.lossDefinitionVersion = 1;
    result.extended.auxiliaryLossMode = "disabled";
    result.extended.targetClippingDefinition = "none";
    result.extended.objectiveNormalizationIdentity = "fixture_v1";
    result.extended.checkpointPolicyScope = "symbol_horizon";
    result.extended.checkpointPolicyStopMode = "next_checkpoint";
    result.extended.checkpointPolicyGraceEvaluations = 1;
    result.extended.checkpointPolicyRevision = 1;

    shared.experimentStatus = "completed";
    shared.experimentPhase = "done";
    shared.finalModelId = experimentId + 1000;
    shared.trainingExecution = {
        experimentId * 10 + 1, "train", 8, 80, "train", "source_commit",
        "train_sha256", "LSTM_Release", "train_runtime_identity", ""};
    shared.inferenceExecution = {
        experimentId * 10 + 2, "infer", 8, 80, "infer", "source_commit",
        "infer_sha256", "lstm-infer-worker", "infer_runtime_identity", ""};
    result.exactFinalInferenceResultId = experimentId + 2000;

    Shared::ClassificationEvidence classification;
    classification.inferenceAccuracy = inferenceAccuracy;
    classification.acceptRate = acceptRate;
    classification.acceptAccuracy = 0.75;
    classification.leaderScore = 0.60;
    shared.classification = classification;

    Shared::ProfitabilityEvidence profitability;
    profitability.predictionCount = 100;
    profitability.actionableCount = actionableCount;
    profitability.winningActionableCount = winningCount;
    profitability.losingActionableCount = actionableCount - winningCount;
    profitability.grossPositiveTerminalHorizonLogReturnSum =
        aggregateReturn + 0.20;
    profitability.grossNegativeTerminalHorizonLogReturnSum = -0.20;
    profitability.aggregateTerminalHorizonLogReturnSum = aggregateReturn;
    if (actionableCount != 0)
        profitability.averageTerminalHorizonLogReturnPerActionablePrediction =
            aggregateReturn / static_cast<double>(actionableCount);
    shared.profitability = profitability;
    return result;
}

bool Has(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

Comparison::Request FeatureAblationRequest()
{
    Comparison::Request request;
    request.armALabel = "control";
    request.armBLabel = "ablation";
    request.intentionalDifferenceFields = {"feature_ablation_mask"};
    return request;
}

const Comparison::IdentityField& Identity(
    const Comparison::ArmResultSet& arm,
    const std::string& name)
{
    const auto found = std::find_if(
        arm.scientificIdentity.begin(), arm.scientificIdentity.end(),
        [&name](const Comparison::IdentityField& field)
        {
            return field.name == name;
        });
    assert(found != arm.scientificIdentity.end());
    return *found;
}

} // namespace

int main()
{
    const std::string ablationMask =
        "tg4_inner_break_any:tg4_source_tg3_confluent:"
        "tg4_source_tg3_structurally_eligible";
    const auto controlEvidence = Arm(658, "", 0.70, 0.55, 0.10, 40, 25);
    const auto ablationEvidence = Arm(
        659, ablationMask, 0.72, 0.45, -0.05, 30, 12);
    const auto control = Comparison::MakeArmResultSet(controlEvidence);
    const auto ablation = Comparison::MakeArmResultSet(ablationEvidence);
    const auto request = FeatureAblationRequest();

    const auto complete = Comparison::Compare(control, ablation, request);
    assert(complete.status == Comparison::Status::ComparableComplete);
    assert(complete.invalidReasons.empty());
    assert(complete.unexpectedDifferences.empty());
    assert(complete.intentionalDifferences.size() == 1);
    assert(complete.intentionalDifferences[0].field ==
           "feature_ablation_mask");
    assert(complete.intentionalDifferences[0].armA.empty());
    assert(complete.intentionalDifferences[0].armB == ablationMask);

    // Every delta is B-A: positive, negative, and equal values are preserved.
    assert(std::fabs(*complete.inferenceAccuracy.armBMinusArmA - 0.02) <
           1.0e-15);
    assert(std::fabs(*complete.acceptRate.armBMinusArmA + 0.10) < 1.0e-15);
    assert(*complete.acceptAccuracy.armBMinusArmA == 0.0);
    assert(*complete.predictionCount.armBMinusArmA == 0.0);
    assert(*complete.actionableCount.armBMinusArmA == -10.0);
    assert(*complete.winningActionableCount.armBMinusArmA == -13.0);
    assert(std::fabs(*complete.aggregateReturn.armBMinusArmA + 0.15) <
           1.0e-15);

    // A persisted zero/false remains present; it is not confused with absent
    // final-model evidence.
    assert(Identity(control, "target_bias").value == "0");
    assert(Identity(control, "target_use_zscore").value == "false");

    auto mismatch = ablation;
    for (auto& field : mismatch.scientificIdentity)
        if (field.name == "prediction_horizon") field.value = "5";
    const auto incompatible = Comparison::Compare(control, mismatch, request);
    assert(incompatible.status ==
           Comparison::Status::IncompatibleScientificIdentity);
    assert(incompatible.unexpectedDifferences.size() == 1);
    assert(incompatible.unexpectedDifferences[0].field ==
           "prediction_horizon");
    assert(!incompatible.inferenceAccuracy.armBMinusArmA);
    assert(!incompatible.aggregateReturn.armBMinusArmA);

    auto missingAnalysis = ablation;
    missingAnalysis.finalAnalysisAvailable = false;
    missingAnalysis.metrics.inferenceAccuracy.reset();
    missingAnalysis.metrics.acceptRate.reset();
    missingAnalysis.metrics.acceptAccuracy.reset();
    missingAnalysis.metrics.leaderScore.reset();
    const auto analysisIncomplete = Comparison::Compare(
        control, missingAnalysis, request);
    assert(analysisIncomplete.status ==
           Comparison::Status::ComparableIncomplete);
    assert(Has(analysisIncomplete.incompleteReasons,
               "arm_b_final_analysis_unavailable"));
    assert(!analysisIncomplete.inferenceAccuracy.armBMinusArmA);
    assert(analysisIncomplete.aggregateReturn.armBMinusArmA);

    auto missingInference = missingAnalysis;
    missingInference.finalInferenceAvailable = false;
    const auto inferenceIncomplete = Comparison::Compare(
        control, missingInference, request);
    assert(Has(inferenceIncomplete.incompleteReasons,
               "arm_b_final_inference_unavailable"));

    auto missingProfitability = ablation;
    missingProfitability.profitabilityObservationAvailable = false;
    missingProfitability.metrics.predictionCount.reset();
    missingProfitability.metrics.actionableCount.reset();
    missingProfitability.metrics.winningActionableCount.reset();
    missingProfitability.metrics.losingActionableCount.reset();
    missingProfitability.metrics.actionablePercentage.reset();
    missingProfitability.metrics.winPercentage.reset();
    missingProfitability.metrics.grossPositiveReturn.reset();
    missingProfitability.metrics.grossNegativeReturn.reset();
    missingProfitability.metrics.aggregateReturn.reset();
    missingProfitability.metrics.averageReturnPerAction.reset();
    const auto profitabilityIncomplete = Comparison::Compare(
        control, missingProfitability, request);
    assert(Has(profitabilityIncomplete.incompleteReasons,
               "arm_b_profitability_observation_unavailable"));
    assert(!profitabilityIncomplete.aggregateReturn.armBMinusArmA);

    auto zeroActionEvidence = Arm(659, ablationMask, 0.72, 0.45, 0.0, 0, 0);
    const auto zeroAction = Comparison::MakeArmResultSet(zeroActionEvidence);
    assert(zeroAction.metrics.actionablePercentage == 0.0);
    assert(!zeroAction.metrics.winPercentage);
    assert(!zeroAction.metrics.averageReturnPerAction);
    const auto zeroActionComparison = Comparison::Compare(
        control, zeroAction, request);
    assert(zeroActionComparison.status ==
           Comparison::Status::ComparableIncomplete);
    assert(!zeroActionComparison.winPercentage.armBMinusArmA);
    assert(!zeroActionComparison.averageReturnPerAction.armBMinusArmA);

    const std::string rendered = Comparison::Render(complete);
    assert(rendered == Comparison::Render(complete));
    assert(rendered.find("delta_sign_convention=arm_b_minus_arm_a") !=
           std::string::npos);
    assert(rendered.find(
               "EXPERIMENT_PAIR_ARM_IDENTITY,role=arm_a,field=symbol,"
               "value=\"cadchfrmp\"") != std::string::npos);
    assert(rendered.find(
               "EXPERIMENT_PAIR_ARM_IDENTITY,role=arm_b,field=symbol,"
               "value=\"cadchfrmp\"") != std::string::npos);
    assert(rendered.find("subjective_winner=NONE") != std::string::npos);
    const auto inferencePosition = rendered.find("metric=inference_accuracy");
    const auto acceptRatePosition = rendered.find("metric=accept_rate");
    const auto predictionPosition = rendered.find("metric=prediction_count");
    const auto aggregatePosition = rendered.find("metric=aggregate_return");
    assert(inferencePosition < acceptRatePosition);
    assert(acceptRatePosition < predictionPosition);
    assert(predictionPosition < aggregatePosition);

    auto noModelControlEvidence = controlEvidence;
    auto noModelAblationEvidence = ablationEvidence;
    for (auto* evidence : {&noModelControlEvidence, &noModelAblationEvidence})
    {
        auto& shared = evidence->authoritative;
        shared.experimentStatus = "running";
        shared.experimentPhase = "train";
        shared.finalModelId.reset();
        shared.trainingExecution.reset();
        shared.inferenceExecution.reset();
        shared.classification.reset();
        shared.profitability.reset();
        evidence->exactFinalInferenceResultId.reset();
        // These are the default-looking values historically exposed by the
        // loader when no final model had populated the metadata members.
        shared.configuration.inputWidth = 0;
        shared.configuration.modelInputLayoutVersion = 0;
        shared.configuration.persistedTrainingSymbol.clear();
        shared.configuration.persistedTrainingStart.clear();
        shared.configuration.persistedTrainingEnd.clear();
    }
    const auto noModelControl =
        Comparison::MakeArmResultSet(noModelControlEvidence);
    const auto noModelAblation =
        Comparison::MakeArmResultSet(noModelAblationEvidence);
    assert(!Identity(noModelControl, "model_input_width").value);
    assert(!Identity(noModelControl,
                     "model_input_semantic_layout_version").value);
    assert(!Identity(noModelControl, "persisted_training_symbol").value);
    const auto noModelComparison = Comparison::Compare(
        noModelControl, noModelAblation, request);
    assert(noModelComparison.status ==
           Comparison::Status::ComparableIncomplete);
    assert(noModelComparison.unexpectedDifferences.empty());
    const std::string noModelRendered =
        Comparison::Render(noModelComparison);
    assert(noModelRendered.find(
               "field=model_input_width,value=NULL") != std::string::npos);
    assert(noModelRendered.find(
               "field=model_input_width,value=\"0\"") == std::string::npos);
    assert(noModelRendered.find(
               "field=persisted_training_symbol,value=NULL") !=
           std::string::npos);

    const std::string summary = Comparison::RenderSummary(complete);
    assert(summary == Comparison::RenderSummary(complete));
    assert(summary.find("experiment_a_id=658,experiment_b_id=659") !=
           std::string::npos);
    assert(summary.find("arm_order=argument_order") != std::string::npos);
    assert(summary.find("status=comparable_complete") != std::string::npos);
    assert(summary.find("scientific_comparability=comparable") !=
           std::string::npos);
    assert(summary.find(
               "kind=intentional,field=feature_ablation_mask") !=
           std::string::npos);
    assert(summary.find("kind=unexpected,field=NONE") != std::string::npos);
    assert(summary.find(
               "role=arm_a,final_model_available=true,"
               "training_provenance_available=true") != std::string::npos);
    assert(summary.find(
               "metric=aggregate_return,arm_a=0.1,arm_b=-0.05,"
               "arm_b_minus_arm_a=-0.15000000000000002") !=
           std::string::npos);
    assert(summary.find("subjective_winner") == std::string::npos);
    assert(summary.find("recommendation") == std::string::npos);
    assert(summary.find("ranking") == std::string::npos);
    const auto summaryEvidenceA = summary.find("role=arm_a");
    const auto summaryEvidenceB = summary.find("role=arm_b");
    const auto summaryIntentional = summary.find("kind=intentional");
    const auto summaryUnexpected = summary.find("kind=unexpected");
    const auto summaryReasons = summary.find("EXPERIMENT_PAIR_SUMMARY_REASONS");
    const auto summaryInference = summary.find("metric=inference_accuracy");
    const auto summaryAggregate = summary.find("metric=aggregate_return");
    assert(summaryEvidenceA < summaryEvidenceB);
    assert(summaryEvidenceB < summaryIntentional);
    assert(summaryIntentional < summaryUnexpected);
    assert(summaryUnexpected < summaryReasons);
    assert(summaryReasons < summaryInference);
    assert(summaryInference < summaryAggregate);

    std::cout << "Experiment pair comparison tests passed\n";
    return 0;
}
