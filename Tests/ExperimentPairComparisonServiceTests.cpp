#include "ExperimentPairComparisonService.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Comparison = EA::ExperimentPairComparison;
namespace Feature = EA::FeatureAblationPairEvaluation;
namespace Shared = EA::PairedTrainingObjectiveEvaluation;

namespace
{

Feature::ArmEvidence Arm(long long experimentId,
                         std::string featureAblationMask,
                         double inferenceAccuracy,
                         double aggregateReturn,
                         std::uint64_t actionableCount = 40)
{
    Feature::ArmEvidence result;
    auto& shared = result.authoritative;
    auto& configuration = shared.configuration;
    configuration.experimentId = experimentId;
    configuration.symbol = "audchfrmp";
    configuration.predictionHorizon = 4;
    configuration.trainStart = "2010-01-01";
    configuration.trainEnd = "2025-01-01";
    configuration.inferenceStart = "2025-01-01";
    configuration.inferenceEnd = "2026-01-01";
    configuration.targetEpochs = 80;
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
    result.extended.economicCalendarSnapshotId = 50;
    result.extended.economicCalendarSnapshotHash =
        "fnv1a64:0000000000000002";
    result.extended.freshInitializationSeed = 42U;
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
    classification.inferenceResultId = experimentId + 2000;
    classification.modelId = experimentId + 1000;
    classification.inferenceScope = "final";
    classification.status = "completed";
    classification.inferenceAccuracy = inferenceAccuracy;
    classification.acceptRate = 0.55;
    classification.acceptAccuracy = 0.75;
    classification.leaderScore = 0.60;
    shared.classification = classification;

    Shared::ProfitabilityEvidence profitability;
    profitability.predictionCount = 100;
    profitability.actionableCount = actionableCount;
    profitability.winningActionableCount = actionableCount / 2;
    profitability.losingActionableCount = actionableCount / 2;
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

class FixtureSource final : public Comparison::EvidenceSource
{
public:
    std::map<long long, Feature::ArmEvidence> evidence;
    mutable std::vector<long long> loads;
    mutable int writeOperations = 0;

    Feature::ArmEvidence Load(long long experimentId) const override
    {
        loads.push_back(experimentId);
        const auto found = evidence.find(experimentId);
        if (found == evidence.end())
            throw Comparison::EvidenceUnavailableError(
                "experiment_" + std::to_string(experimentId) + "_not_found");
        return found->second;
    }
};

void ExpectParseFailure(std::string_view value)
{
    try
    {
        (void)Comparison::ParseExperimentIdPair(value);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
    }
}

Comparison::ComparisonResult Run(FixtureSource& source,
                                 long long first,
                                 long long second,
                                 std::string* rendered = nullptr)
{
    std::ostringstream output;
    std::ostringstream errors;
    const int exitCode = Comparison::RunComparisonCommand(
        {{first, second}}, source, output, errors);
    assert(exitCode == 0);
    assert(errors.str().empty());
    if (rendered) *rendered = output.str();

    const auto& evidenceA = source.evidence.at(first);
    const auto& evidenceB = source.evidence.at(second);
    return Comparison::Compare(
        Comparison::MakeArmResultSet(evidenceA),
        Comparison::MakeArmResultSet(evidenceB),
        Comparison::MakeComparisonRequest(evidenceA, evidenceB));
}

bool Has(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

} // namespace

int main()
{
    assert(Comparison::ParseExperimentIdPair("17:29") ==
           std::pair(17LL, 29LL));
    for (std::string_view malformed : {
             "", ":", "1:", ":2", "a:2", "1:b", "0:2", "-1:2",
             "1:0", "1:-2", "1:2:3", "1:2x", "1:1", " 1:2"})
        ExpectParseFailure(malformed);

    FixtureSource source;
    source.evidence.emplace(17, Arm(17, "", 0.70, 0.10, 40));
    source.evidence.emplace(29, Arm(29, "", 0.72, -0.05, 30));
    std::string rendered;
    const auto complete = Run(source, 17, 29, &rendered);
    assert(complete.status == Comparison::Status::ComparableComplete);
    assert(source.loads == std::vector<long long>({17, 29}));
    assert(source.writeOperations == 0);
    assert(complete.experimentAId == 17);
    assert(complete.experimentBId == 29);
    assert(std::fabs(*complete.inferenceAccuracy.armBMinusArmA - 0.02) <
           1.0e-15);
    assert(std::fabs(*complete.aggregateReturn.armBMinusArmA + 0.15) <
           1.0e-15);
    assert(rendered == Comparison::Render(complete));
    assert(rendered.find("delta_sign_convention=arm_b_minus_arm_a") !=
           std::string::npos);
    assert(rendered.find("subjective_winner=NONE") != std::string::npos);
    assert(rendered.find("recommendation=") == std::string::npos);

    source.loads.clear();
    std::ostringstream summaryOutput;
    std::ostringstream summaryErrors;
    assert(Comparison::RunComparisonCommand(
               {{17, 29}, true}, source, summaryOutput, summaryErrors) == 0);
    assert(summaryErrors.str().empty());
    assert(source.loads == std::vector<long long>({17, 29}));
    assert(source.writeOperations == 0);
    assert(summaryOutput.str() == Comparison::RenderSummary(complete));
    assert(summaryOutput.str().find("EXPERIMENT_PAIR_ARM_IDENTITY") ==
           std::string::npos);
    assert(summaryOutput.str().find("subjective_winner") ==
           std::string::npos);
    assert(summaryOutput.str().find("recommendation") == std::string::npos);
    assert(summaryOutput.str().find("ranking") == std::string::npos);

    // Arm order is never normalized by the generic service.
    source.loads.clear();
    const auto reversed = Run(source, 29, 17);
    assert(source.loads == std::vector<long long>({29, 17}));
    assert(reversed.experimentAId == 29);
    assert(reversed.experimentBId == 17);
    assert(std::fabs(*reversed.aggregateReturn.armBMinusArmA - 0.15) <
           1.0e-15);

    const std::string mask = "tg4_inner_break_any";
    source.evidence[29] = Arm(29, mask, 0.72, -0.05, 30);
    const auto ablation = Run(source, 17, 29);
    assert(ablation.status == Comparison::Status::ComparableComplete);
    assert(ablation.intentionalDifferences.size() == 1);
    assert(ablation.intentionalDifferences[0].field ==
           "feature_ablation_mask");

    // Two different non-empty masks are ambiguous and are not whitelisted.
    source.evidence[17] = Arm(17, "economic_event", 0.70, 0.10, 40);
    const auto ambiguousTreatment = Run(source, 17, 29);
    assert(ambiguousTreatment.status ==
           Comparison::Status::IncompatibleScientificIdentity);
    assert(ambiguousTreatment.intentionalDifferences.empty());
    assert(ambiguousTreatment.unexpectedDifferences.size() == 1);
    assert(ambiguousTreatment.unexpectedDifferences[0].field ==
           "feature_ablation_mask");

    source.evidence[17] = Arm(17, "", 0.70, 0.10, 40);
    source.evidence[29] = Arm(29, "", 0.72, -0.05, 30);
    source.evidence[29].authoritative.configuration.predictionHorizon = 8;
    const auto mismatch = Run(source, 17, 29);
    assert(mismatch.status ==
           Comparison::Status::IncompatibleScientificIdentity);
    assert(mismatch.unexpectedDifferences.size() == 1);
    assert(mismatch.unexpectedDifferences[0].field == "prediction_horizon");
    assert(!mismatch.inferenceAccuracy.armBMinusArmA);

    source.evidence[29] = Arm(29, "", 0.72, -0.05, 30);
    source.evidence[29].exactFinalInferenceResultId.reset();
    const auto missingFinalInference = Run(source, 17, 29);
    assert(missingFinalInference.status ==
           Comparison::Status::ComparableIncomplete);
    assert(Has(missingFinalInference.incompleteReasons,
               "arm_b_final_inference_unavailable"));

    source.evidence[29] = Arm(29, "", 0.72, -0.05, 30);
    source.evidence[29].authoritative.classification.reset();
    const auto missingAnalysis = Run(source, 17, 29);
    assert(Has(missingAnalysis.incompleteReasons,
               "arm_b_final_analysis_unavailable"));
    assert(!missingAnalysis.inferenceAccuracy.armBMinusArmA);

    source.evidence[29] = Arm(29, "", 0.72, -0.05, 30);
    source.evidence[29].authoritative.profitability.reset();
    const auto missingProfitability = Run(source, 17, 29);
    assert(Has(missingProfitability.incompleteReasons,
               "arm_b_profitability_observation_unavailable"));
    assert(!missingProfitability.aggregateReturn.armBMinusArmA);

    source.evidence[29] = Arm(29, "", 0.72, 0.0, 0);
    const auto zeroAction = Run(source, 17, 29);
    assert(zeroAction.status == Comparison::Status::ComparableIncomplete);
    assert(zeroAction.actionableCount.armB == 0.0);
    assert(!zeroAction.winPercentage.armB);
    assert(!zeroAction.averageReturnPerAction.armB);

    FixtureSource missingSource;
    missingSource.evidence.emplace(17, Arm(17, "", 0.70, 0.10));
    std::ostringstream missingOutput;
    std::ostringstream missingErrors;
    assert(Comparison::RunComparisonCommand(
               {{17, 404}}, missingSource, missingOutput, missingErrors) == 3);
    assert(missingOutput.str().empty());
    assert(missingErrors.str().find("experiment_404_not_found") !=
           std::string::npos);
    assert(missingSource.loads == std::vector<long long>({17, 404}));
    assert(missingSource.writeOperations == 0);

    FixtureSource invalidCommandSource;
    std::ostringstream invalidOutput;
    std::ostringstream invalidErrors;
    assert(Comparison::RunComparisonCommand(
               {{17, 17}}, invalidCommandSource, invalidOutput,
               invalidErrors) == 3);
    assert(invalidCommandSource.loads.empty());
    assert(invalidOutput.str().empty());
    assert(invalidErrors.str().find("distinct_experiment_IDs") !=
           std::string::npos);

    std::cout << "Experiment pair comparison service tests passed\n";
    return 0;
}
