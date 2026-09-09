#include "../Sources/StrategyEvaluationCore/StrategyEvaluation.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace Profitability = EA::InferenceProfitability;
namespace Strategy = EA::StrategyEvaluation;

namespace
{

Strategy::MarketPathProvenance Provenance()
{
    Strategy::MarketPathProvenance value;
    value.adapterFamily = "phase18b_controlled_fixture";
    value.adapterVersion = 1;
    value.modelId = 18004;
    value.inferenceScientificIdentityCanonical =
        "phase18b_inference_v1;model=18004;fixture=controlled;";
    value.inferenceScientificIdentityHash =
        Profitability::DeterministicHash(
            value.inferenceScientificIdentityCanonical);
    value.symbol = "eurusdrmp";
    value.inferenceWindowSize = 3;
    value.predictionHorizon = 2;
    value.evaluationStart = "2025-01-01";
    value.evaluationEnd = "2025-02-01";
    value.barIntervalSeconds = 900;
    value.timestampSemantics = "fixture_utc_v1";
    value.ohlcIntervalSemantics = "fixture_ask_ohlc_v1";
    value.priceDomain = "ask";
    value.marketDataSource = "immutable_fixture_v1";
    value.marketDataSourceRelation = "fixture";
    value.pathOrdering = "source_row_ascending_v1";
    value.metricDefinitionCanonical =
        Strategy::kFixedStopMetricDefinitionCanonical;
    value.metricDefinitionHash = Strategy::FixedStopMetricDefinitionHash();
    return value;
}

Strategy::PredictionProbabilities Probabilities(int predictedClass,
                                                float directional)
{
    if (predictedClass == Profitability::kNeutralClass)
        return {{0.2f, 0.6f, 0.2f}};
    const float other = (1.0f - directional) / 2.0f;
    if (predictedClass == Profitability::kDownClass)
        return {{directional, other, other}};
    return {{other, other, directional}};
}

Strategy::StrategyEvaluationObservation Observation(
    std::uint64_t ordinal,
    int predictedClass,
    float directional,
    Strategy::MarketPathPoint first,
    Strategy::MarketPathPoint terminal)
{
    const std::uint64_t start = ordinal * 10;
    const std::int64_t timeBase =
        static_cast<std::int64_t>(ordinal) * 10000;
    first.sourceRow = start + 3;
    first.timestampUnixSeconds = timeBase + 1900;
    terminal.sourceRow = start + 4;
    terminal.timestampUnixSeconds = timeBase + 2800;
    Strategy::StrategyEvaluationObservation value;
    value.observationOrdinal = ordinal;
    value.inferenceWindowStartRow = start;
    value.decisionRow = start + 2;
    value.terminalRow = start + 4;
    value.predictedClass = predictedClass;
    value.probabilities = Probabilities(predictedClass, directional);
    value.decisionClose = 100.0f;
    value.terminalClose = terminal.close;
    value.decisionTimestampUnixSeconds = timeBase + 1000;
    value.terminalTimestampUnixSeconds = terminal.timestampUnixSeconds;
    value.marketPath = {first, terminal};
    return value;
}

Strategy::AuthoritativeMarketPath Fixture()
{
    std::vector<Strategy::StrategyEvaluationObservation> observations;
    observations.push_back(Observation(
        0, Profitability::kUpClass, 0.34f,
        {0, {}, 100, 103, 95, 102},
        {0, {}, 102, 112, 101, 110}));
    observations.push_back(Observation(
        1, Profitability::kDownClass, 0.50f,
        {0, {}, 100, 105, 95, 100},
        {0, {}, 100, 112, 98, 105}));
    observations.push_back(Observation(
        2, Profitability::kUpClass, 0.70f,
        {0, {}, 85, 90, 80, 88},
        {0, {}, 88, 95, 82, 92}));
    observations.push_back(Observation(
        3, Profitability::kDownClass, 1.00f,
        {0, {}, 100, 105, 92, 95},
        {0, {}, 95, 100, 88, 90}));
    observations.push_back(Observation(
        4, Profitability::kNeutralClass, 0.60f,
        {0, {}, 100, 130, 70, 105},
        {0, {}, 105, 140, 60, 110}));
    return Strategy::BuildAuthoritativeMarketPath(
        Provenance(), std::move(observations));
}

std::string Error(const auto& operation)
{
    try { operation(); }
    catch (const std::exception& error) { return error.what(); }
    return {};
}

const Strategy::Phase18BStrategyVariantResult& Variant(
    const Strategy::ControlledOneSidedStopExtensionExperimentResult& result,
    const std::string& identity)
{
    for (const auto& variant : result.variants)
    {
        if (variant.strategyOutputIdentity == identity)
            return variant;
    }
    throw std::runtime_error("variant_not_found");
}

double Value(const std::optional<double>& value)
{
    return value.value_or(0.0);
}

void ReconcilePairwise(
    const Strategy::ControlledOneSidedStopExtensionExperimentResult& result,
    const Strategy::Phase18BPairwiseDelta& pairwise)
{
    const auto& candidate = Variant(
        result, pairwise.values.candidateStrategyOutputIdentity);
    const auto& reference = Variant(
        result, pairwise.values.referenceStrategyOutputIdentity);
    assert(pairwise.values.aggregateDirectionalLogReturnDelta ==
           candidate.metrics.aggregateDirectionalLogReturn -
               reference.metrics.aggregateDirectionalLogReturn);
    assert(pairwise.values.averageDirectionalLogReturnDelta ==
           Value(candidate.metrics.averageDirectionalLogReturn) -
               Value(reference.metrics.averageDirectionalLogReturn));
    assert(pairwise.values.winningRateDelta ==
           Value(candidate.metrics.winningRate) -
               Value(reference.metrics.winningRate));
    assert(pairwise.values.losingRateDelta ==
           Value(candidate.metrics.losingRate) -
               Value(reference.metrics.losingRate));
    assert(pairwise.values.stopHitRateDelta ==
           Value(candidate.metrics.stopHitRate) -
               Value(reference.metrics.stopHitRate));
    assert(pairwise.values.terminalExitRateDelta ==
           Value(candidate.metrics.terminalExitRate) -
               Value(reference.metrics.terminalExitRate));
    assert(pairwise.values.averageHoldingDurationSecondsDelta ==
           Value(candidate.metrics.averageHoldingDurationSeconds) -
               Value(reference.metrics.averageHoldingDurationSeconds));
    assert(pairwise.values.maximumDrawdownDelta ==
           candidate.metrics.maximumDrawdown -
               reference.metrics.maximumDrawdown);
}

} // namespace

int main()
{
    const auto path = Fixture();
    const auto originalPathCanonical = path.Canonical();
    const auto originalPathHash = path.Hash();
    const auto phase18A =
        Strategy::EvaluateControlledProbabilityConditionedStopExperiment(path);
    const auto first =
        Strategy::EvaluateControlledOneSidedStopExtensionExperiment(path);
    const auto repeated =
        Strategy::EvaluateControlledOneSidedStopExtensionExperiment(path);

    assert(path.Canonical() == originalPathCanonical);
    assert(path.Hash() == originalPathHash);
    assert(first.requiresReadOnlyTransaction);
    assert(!first.productionRowsModified);
    assert(first.experimentIdentityCanonical ==
           repeated.experimentIdentityCanonical);
    assert(first.experimentIdentityHash == repeated.experimentIdentityHash);
    assert(first.canonicalLines == repeated.canonicalLines);
    assert(first.resultHash == repeated.resultHash);
    assert(first.variants.size() == 4);
    assert(first.variants[0].strategyOutputIdentity ==
           Strategy::kBaselineTerminalOutputIdentity);
    assert(first.variants[1].strategyOutputIdentity ==
           Strategy::kFixedStopLossOutputIdentity);
    assert(first.variants[2].strategyOutputIdentity ==
           Strategy::kProbabilityConditionedStopLossOutputIdentity);
    assert(first.variants[3].strategyOutputIdentity ==
           Strategy::kProbabilityConditionedStopExtensionOutputIdentity);

    // Adding Phase 18B does not change the existing Phase 18A strategy result.
    assert(first.variants[2].evaluation.strategyIdentity ==
           phase18A.variants[2].evaluation.strategyIdentity);
    assert(first.variants[2].evaluation.executionResults ==
           phase18A.variants[2].evaluation.executionResults);
    assert(first.variants[2].evaluation.resultCanonical ==
           phase18A.variants[2].evaluation.resultCanonical);
    assert(first.variants[2].evaluation.resultHash ==
           phase18A.variants[2].evaluation.resultHash);

    for (const auto& variant : first.variants)
    {
        assert(variant.evaluation.sourceContentHash == path.Hash());
        assert(variant.metrics.observationCount == 5);
        assert(variant.metrics.actionableCount == 4);
        assert(variant.metrics.nonActionableCount == 1);
        assert(variant.metrics.stopHitCount +
                   variant.metrics.terminalExitCount == 4);
        assert(variant.metrics.winningCount + variant.metrics.losingCount +
                   variant.metrics.zeroOutcomeCount == 4);
        assert(variant.metrics.averageHoldingDurationSeconds);
        assert(variant.metrics.medianHoldingDurationSeconds);
        assert(variant.metrics.maximumAdverseExcursion);
        assert(variant.metrics.maximumFavorableExcursion);
        std::uint64_t bucketTotal = 0;
        for (const auto& bucket : variant.confidenceBuckets)
            bucketTotal += bucket.actionableCount;
        assert(bucketTotal == variant.metrics.actionableCount);
        for (const auto& bucket : variant.confidenceBuckets)
            assert(bucket.actionableCount == 1);
    }

    const auto& extension = first.variants[3];
    assert(extension.mechanismMetrics.fixedFloorMultiplierCount == 2);
    assert(extension.mechanismMetrics.extendedStopMultiplierCount == 2);
    assert(extension.mechanismMetrics.fixedFloorMultiplierRate == 0.5);
    assert(extension.mechanismMetrics.extendedStopMultiplierRate == 0.5);
    assert(extension.mechanismMetrics.averageExtendedStopMultiplier);
    assert(*extension.mechanismMetrics.averageExtendedStopMultiplier > 1.0);

    for (std::size_t index = 0;
         index < first.extensionDeltasVersusFixedByConfidenceBucket.size();
         ++index)
    {
        const auto& delta =
            first.extensionDeltasVersusFixedByConfidenceBucket[index];
        const auto& candidate = first.variants[3].confidenceBuckets[index];
        const auto& reference = first.variants[1].confidenceBuckets[index];
        assert(delta.actionableCount == candidate.actionableCount);
        assert(delta.stopHitCountDifferenceVersusFixed ==
               static_cast<std::int64_t>(candidate.stopHitCount) -
                   static_cast<std::int64_t>(reference.stopHitCount));
        assert(delta.aggregateDirectionalLogReturnDifferenceVersusFixed ==
               candidate.aggregateDirectionalLogReturn -
                   reference.aggregateDirectionalLogReturn);
    }
    assert(first.extensionDeltasVersusFixedByConfidenceBucket[0].
               stopHitCountDifferenceVersusFixed == 0);
    assert(first.extensionDeltasVersusFixedByConfidenceBucket[1].
               stopHitCountDifferenceVersusFixed == 0);
    assert(first.extensionDeltasVersusFixedByConfidenceBucket[0].
               aggregateDirectionalLogReturnDifferenceVersusFixed == 0.0);
    assert(first.extensionDeltasVersusFixedByConfidenceBucket[1].
               aggregateDirectionalLogReturnDifferenceVersusFixed == 0.0);

    assert(first.pairwiseDeltas.size() == 4);
    assert(first.pairwiseDeltas[0].comparisonRole == "PRIMARY");
    assert(first.pairwiseDeltas[0].values.candidateStrategyOutputIdentity ==
           Strategy::kProbabilityConditionedStopExtensionOutputIdentity);
    assert(first.pairwiseDeltas[0].values.referenceStrategyOutputIdentity ==
           Strategy::kFixedStopLossOutputIdentity);
    assert(first.pairwiseDeltas[1].values.referenceStrategyOutputIdentity ==
           Strategy::kBaselineTerminalOutputIdentity);
    assert(first.pairwiseDeltas[2].values.referenceStrategyOutputIdentity ==
           Strategy::kProbabilityConditionedStopLossOutputIdentity);
    assert(first.pairwiseDeltas[3].values.candidateStrategyOutputIdentity ==
           Strategy::kProbabilityConditionedStopLossOutputIdentity);
    assert(first.pairwiseDeltas[3].values.referenceStrategyOutputIdentity ==
           Strategy::kFixedStopLossOutputIdentity);
    for (const auto& pairwise : first.pairwiseDeltas)
        ReconcilePairwise(first, pairwise);

    assert(first.canonicalLines.find(
        "phase18b_mapping_identity=directional_probability_one_sided_stop_extension_v1") !=
        std::string::npos);
    assert(first.canonicalLines.find("comparison_role=PRIMARY") !=
           std::string::npos);
    assert(first.canonicalLines.find("production_rows_modified=false") !=
           std::string::npos);
    assert(first.canonicalLines.find("PHASE18B_EXTENSION_MECHANISM") !=
           std::string::npos);
    assert(first.canonicalLines.find("stop_hit_rate_difference_vs_fixed=") !=
           std::string::npos);

    const auto identity =
        Strategy::EvaluateControlledOneSidedStopExtensionExperiment(
            path, {1, 0.001, 0.50, 1.0, 1.0});
    assert(identity.variants[1].evaluation.executionResults ==
           identity.variants[3].evaluation.executionResults);
    assert(identity.pairwiseDeltas[0].values.
               aggregateDirectionalLogReturnDelta == 0.0);
    assert(identity.pairwiseDeltas[0].values.
               averageDirectionalLogReturnDelta == 0.0);
    assert(identity.pairwiseDeltas[0].values.stopHitRateDelta == 0.0);
    assert(identity.pairwiseDeltas[0].values.
               averageHoldingDurationSecondsDelta == 0.0);

    Strategy::ValidateControlledOneSidedStopExtensionInvocation(
        {true, true, false, false, false, false, false});
    assert(Error([] {
        Strategy::ValidateControlledOneSidedStopExtensionInvocation(
            {true, true, true, false, false, false, false});
    }).find("standalone --infer") != std::string::npos);
    assert(Error([] {
        Strategy::ValidateControlledOneSidedStopExtensionInvocation(
            {true, true, false, true, false, false, false});
    }).find("no scheduler context") != std::string::npos);
    assert(Error([] {
        Strategy::ValidateControlledOneSidedStopExtensionInvocation(
            {false, true, false, false, false, false, false});
    }).find("standalone --infer") != std::string::npos);
    assert(Error([] {
        Strategy::ValidateControlledOneSidedStopExtensionInvocation(
            {true, false, false, false, false, false, false});
    }).find("explicit --model") != std::string::npos);
    return 0;
}
