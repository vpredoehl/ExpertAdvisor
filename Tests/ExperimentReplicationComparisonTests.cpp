#include "ExperimentReplicationComparisonService.hpp"

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

namespace Replication = EA::ExperimentReplicationComparison;
namespace Pair = EA::ExperimentPairComparison;
namespace Feature = EA::FeatureAblationPairEvaluation;
namespace Shared = EA::PairedTrainingObjectiveEvaluation;

namespace
{

constexpr std::string_view kMask =
    "tg4_inner_break_any,tg4_source_tg3_structurally_eligible,"
    "tg4_source_tg3_confluent";

Feature::ArmEvidence Arm(long long experimentId,
                         std::string mask,
                         unsigned int seed,
                         double inferenceAccuracy,
                         double aggregateReturn)
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
    configuration.featureAblationMask = std::move(mask);
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
    result.extended.freshInitializationSeed = seed;
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
    classification.acceptRate = 0.55 + aggregateReturn;
    classification.acceptAccuracy = 0.75;
    classification.leaderScore = 0.60;
    shared.classification = classification;

    Shared::ProfitabilityEvidence profitability;
    profitability.predictionCount = 100;
    profitability.actionableCount = 40;
    profitability.winningActionableCount = 20;
    profitability.losingActionableCount = 20;
    profitability.grossPositiveTerminalHorizonLogReturnSum =
        aggregateReturn + 0.20;
    profitability.grossNegativeTerminalHorizonLogReturnSum = -0.20;
    profitability.aggregateTerminalHorizonLogReturnSum = aggregateReturn;
    profitability.averageTerminalHorizonLogReturnPerActionablePrediction =
        aggregateReturn / 40.0;
    shared.profitability = profitability;
    return result;
}

void ClearFinalEvidence(Feature::ArmEvidence& arm)
{
    arm.authoritative.experimentStatus = "running";
    arm.authoritative.experimentPhase = "train";
    arm.authoritative.finalModelId.reset();
    arm.authoritative.trainingExecution.reset();
    arm.authoritative.inferenceExecution.reset();
    arm.authoritative.classification.reset();
    arm.authoritative.profitability.reset();
    arm.exactFinalInferenceResultId.reset();
}

Pair::ComparisonResult PairResult(const Feature::ArmEvidence& armA,
                                  const Feature::ArmEvidence& armB)
{
    return Pair::Compare(Pair::MakeArmResultSet(armA),
                         Pair::MakeArmResultSet(armB),
                         Pair::MakeComparisonRequest(armA, armB));
}

const Replication::MetricAggregate& Metric(
    const Replication::Result& result,
    std::string_view name)
{
    const auto found = std::find_if(
        result.metrics.begin(), result.metrics.end(),
        [name](const auto& metric) { return metric.name == name; });
    assert(found != result.metrics.end());
    return *found;
}

class FixtureSource final : public Pair::EvidenceSource
{
public:
    std::map<long long, Feature::ArmEvidence> evidence;
    mutable std::vector<long long> loads;
    mutable int writes = 0;

    Feature::ArmEvidence Load(long long experimentId) const override
    {
        loads.push_back(experimentId);
        const auto found = evidence.find(experimentId);
        if (found == evidence.end())
            throw Pair::EvidenceUnavailableError(
                "experiment_" + std::to_string(experimentId) + "_not_found");
        return found->second;
    }
};

void ExpectParseFailure(std::string_view text)
{
    try
    {
        (void)Replication::ParseExperimentIdPairs(text);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
    }
}

Replication::Result CompatibleTwoPairResult(unsigned int secondSeed = 43)
{
    auto a1 = Arm(101, "", 42, 0.60, 0.10);
    auto b1 = Arm(102, std::string{kMask}, 42, 0.70, 0.20);
    auto a2 = Arm(103, "", secondSeed, 0.65, 0.30);
    auto b2 = Arm(104, std::string{kMask}, secondSeed, 0.60, 0.20);
    return Replication::Compare(
        {PairResult(a1, b1), PairResult(a2, b2)});
}

} // namespace

int main()
{
    const std::vector<std::pair<long long, long long>> parsedPairs {
        {101, 102}, {103, 104}};
    assert(Replication::ParseExperimentIdPairs("101:102,103:104") ==
           parsedPairs);
    for (std::string_view malformed : {
             "", "1:2", "1:2,", ",1:2", "1:2,,3:4", "a:2,3:4",
             "0:2,3:4", "-1:2,3:4", "1:1,3:4", "1:2,1:2",
             "1:2,2:3", "1:2,3:1", "1:2,3:3"})
        ExpectParseFailure(malformed);

    // Complete compatible replications preserve pair order, arm order, B-A,
    // and the declared different-seed dimension.
    const auto complete = CompatibleTwoPairResult();
    assert(complete.compatibility == Replication::Compatibility::Compatible);
    assert(complete.pairs[0].experimentAId == 101);
    assert(complete.pairs[0].experimentBId == 102);
    assert(complete.pairs[1].experimentAId == 103);
    assert(complete.pairs[1].experimentBId == 104);
    assert(complete.seedReplicationMode ==
           Replication::SeedReplicationMode::DifferentSeeds);
    const auto& accuracy = Metric(complete, "inference_accuracy");
    assert(accuracy.pairCount == 2);
    assert(accuracy.availableCount == 2);
    assert(std::fabs(*accuracy.pairDeltas[0] - 0.10) < 1.0e-15);
    assert(std::fabs(*accuracy.pairDeltas[1] + 0.05) < 1.0e-15);
    assert(std::fabs(*accuracy.descriptiveMean - 0.025) < 1.0e-15);
    assert(accuracy.positiveCount == 1);
    assert(accuracy.zeroCount == 0);
    assert(accuracy.negativeCount == 1);

    const auto sameSeed = CompatibleTwoPairResult(42);
    assert(sameSeed.compatibility == Replication::Compatibility::Compatible);
    assert(sameSeed.seedReplicationMode ==
           Replication::SeedReplicationMode::SameSeedRepeatedPairs);

    auto a1 = Arm(201, "", 42, 0.60, 0.10);
    auto b1 = Arm(202, std::string{kMask}, 42, 0.70, 0.20);
    auto a2 = Arm(203, "", 43, 0.65, 0.30);
    auto b2 = Arm(204, std::string{kMask}, 43, 0.60, 0.20);

    // An unfinished but authoritatively configured pair contributes NULL,
    // never zero, while the complete pair remains descriptively available.
    ClearFinalEvidence(a2);
    ClearFinalEvidence(b2);
    const auto incomplete = Replication::Compare(
        {PairResult(a1, b1), PairResult(a2, b2)});
    assert(incomplete.compatibility == Replication::Compatibility::Compatible);
    assert(incomplete.pairs[1].status == Pair::Status::ComparableIncomplete);
    const auto& incompleteAccuracy = Metric(incomplete, "inference_accuracy");
    assert(incompleteAccuracy.availableCount == 1);
    assert(incompleteAccuracy.pairDeltas[0]);
    assert(!incompleteAccuracy.pairDeltas[1]);
    assert(std::fabs(*incompleteAccuracy.descriptiveMean - 0.10) < 1.0e-15);

    // One absent metric has the same missing-data behavior even when the rest
    // of that pair's evidence exists.
    a2 = Arm(203, "", 43, 0.65, 0.30);
    b2 = Arm(204, std::string{kMask}, 43, 0.60, 0.20);
    b2.authoritative.classification->acceptAccuracy.reset();
    const auto missingMetric = Replication::Compare(
        {PairResult(a1, b1), PairResult(a2, b2)});
    const auto& acceptAccuracy = Metric(missingMetric, "accept_accuracy");
    assert(acceptAccuracy.availableCount == 1);
    assert(!acceptAccuracy.pairDeltas[1]);
    assert(acceptAccuracy.descriptiveMean == 0.0);

    // Positive, zero, and negative counts are exact and use unrounded deltas.
    auto a3 = Arm(205, "", 44, 0.50, 0.40);
    auto b3 = Arm(206, std::string{kMask}, 44, 0.50, 0.40);
    const auto three = Replication::Compare(
        {PairResult(a1, b1), PairResult(a2, b2), PairResult(a3, b3)});
    const auto& threeAccuracy = Metric(three, "inference_accuracy");
    assert(threeAccuracy.positiveCount == 1);
    assert(threeAccuracy.zeroCount == 1);
    assert(threeAccuracy.negativeCount == 1);

    // Same feature-ablation intervention is accepted; known changes to its
    // exact mask are incompatible even when each pair is internally valid.
    b2 = Arm(204, "different_feature", 43, 0.60, 0.20);
    auto mismatch = Replication::Compare(
        {PairResult(a1, b1), PairResult(a2, b2)});
    assert(mismatch.compatibility ==
           Replication::Compatibility::Incompatible);
    assert(mismatch.metrics.empty());

    const auto expectCrossMismatch = [&](auto mutate)
    {
        auto left = Arm(203, "", 43, 0.65, 0.30);
        auto right = Arm(204, std::string{kMask}, 43, 0.60, 0.20);
        mutate(left);
        mutate(right);
        const auto result = Replication::Compare(
            {PairResult(a1, b1), PairResult(left, right)});
        assert(result.compatibility ==
               Replication::Compatibility::Incompatible);
        assert(result.metrics.empty());
    };
    expectCrossMismatch([](auto& arm) {
        arm.authoritative.configuration.symbol = "cadchfrmp";
        arm.authoritative.configuration.persistedTrainingSymbol = "cadchfrmp";
    });
    expectCrossMismatch([](auto& arm) {
        arm.authoritative.configuration.predictionHorizon = 8;
    });
    expectCrossMismatch([](auto& arm) {
        arm.authoritative.configuration.trainStart = "2011-01-01";
        arm.authoritative.configuration.persistedTrainingStart = "2011-01-01";
        arm.authoritative.configuration.inferenceEnd = "2027-01-01";
    });
    expectCrossMismatch([](auto& arm) {
        arm.extended.configuredModelInputWidth = 77;
        arm.extended.configuredModelInputLayoutVersion = 7;
        arm.authoritative.configuration.inputWidth = 77;
        arm.authoritative.configuration.modelInputLayoutVersion = 7;
        arm.authoritative.trainingExecution->modelInputWidth = 77;
        arm.authoritative.trainingExecution->semanticLayoutVersion = 7;
        arm.authoritative.inferenceExecution->modelInputWidth = 77;
        arm.authoritative.inferenceExecution->semanticLayoutVersion = 7;
    });
    expectCrossMismatch([](auto& arm) {
        arm.extended.economicCalendarSnapshotId = 51;
        arm.extended.economicCalendarSnapshotHash =
            "fnv1a64:0000000000000003";
    });
    expectCrossMismatch([](auto& arm) {
        arm.authoritative.configuration.experimentObjective = {
            "objective_id=another_objective_v1;",
            "fnv1a64:0000000000000004"};
    });

    // A/B seed mismatch is an ordinary unexpected pair difference, never an
    // alternative intervention or an accepted replication dimension.
    auto badSeedB = b1;
    badSeedB.extended.freshInitializationSeed = 99;
    const auto badSeedPair = PairResult(a1, badSeedB);
    assert(badSeedPair.status == Pair::Status::IncompatibleScientificIdentity);
    mismatch = Replication::Compare(
        {badSeedPair, PairResult(a2, Arm(204, std::string{kMask}, 43,
                                       0.60, 0.20))});
    assert(mismatch.compatibility == Replication::Compatibility::Incompatible);

    // If immutable configured input identity and final-model fallback are both
    // absent, absence is undetermined—not a fabricated mismatch.
    a2 = Arm(203, "", 43, 0.65, 0.30);
    b2 = Arm(204, std::string{kMask}, 43, 0.60, 0.20);
    a2.extended.configuredModelInputWidth.reset();
    a2.extended.configuredModelInputLayoutVersion.reset();
    b2.extended.configuredModelInputWidth.reset();
    b2.extended.configuredModelInputLayoutVersion.reset();
    ClearFinalEvidence(a2);
    ClearFinalEvidence(b2);
    const auto undetermined = Replication::Compare(
        {PairResult(a1, b1), PairResult(a2, b2)});
    assert(undetermined.compatibility == Replication::Compatibility::
               UndeterminedDueToMissingEvidence);
    assert(undetermined.metrics.empty());

    const std::string rendered = Replication::Render(complete);
    assert(rendered == Replication::Render(complete));
    assert(rendered.find("subjective_winner=NONE") != std::string::npos);
    assert(rendered.find("recommendation=") == std::string::npos);
    assert(rendered.find("ranking=") == std::string::npos);
    assert(rendered.find("composite_score=") == std::string::npos);
    assert(rendered.find("statistical_independence=not_inferred") !=
           std::string::npos);
    assert(rendered.find("pair_deltas=0.09999999999999998|-0.050000000000000044") !=
           std::string::npos);

    FixtureSource source;
    source.evidence.emplace(101, Arm(101, "", 42, 0.60, 0.10));
    source.evidence.emplace(102, Arm(102, std::string{kMask}, 42, 0.70, 0.20));
    source.evidence.emplace(103, Arm(103, "", 43, 0.65, 0.30));
    source.evidence.emplace(104, Arm(104, std::string{kMask}, 43, 0.60, 0.20));
    std::ostringstream output;
    std::ostringstream errors;
    assert(Replication::RunComparisonCommand(
               {{{101, 102}, {103, 104}}}, source, output, errors) == 0);
    assert(errors.str().empty());
    assert(source.loads == std::vector<long long>({101, 102, 103, 104}));
    assert(source.writes == 0);
    assert(output.str() == rendered);

    FixtureSource missingSource;
    missingSource.evidence = source.evidence;
    missingSource.evidence.erase(104);
    std::ostringstream missingOutput;
    std::ostringstream missingErrors;
    assert(Replication::RunComparisonCommand(
               {{{101, 102}, {103, 104}}}, missingSource,
               missingOutput, missingErrors) == 3);
    assert(missingOutput.str().empty());
    assert(missingErrors.str().find("experiment_104_not_found") !=
           std::string::npos);
    assert(missingSource.writes == 0);

    std::cout << "Experiment replication comparison tests passed\n";
    return 0;
}
