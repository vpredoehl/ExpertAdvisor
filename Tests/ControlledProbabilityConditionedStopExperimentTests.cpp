#include "../Sources/StrategyEvaluationCore/StrategyEvaluation.hpp"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

namespace Profitability = EA::InferenceProfitability;
namespace Strategy = EA::StrategyEvaluation;

namespace
{

Strategy::MarketPathProvenance Provenance()
{
    Strategy::MarketPathProvenance value;
    value.adapterFamily = "phase18a_controlled_fixture";
    value.adapterVersion = 1;
    value.modelId = 18002;
    value.inferenceScientificIdentityCanonical =
        "phase18a_inference_v1;model=18002;fixture=controlled;";
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

} // namespace

int main()
{
    const auto path = Fixture();
    const auto first =
        Strategy::EvaluateControlledProbabilityConditionedStopExperiment(path);
    const auto repeated =
        Strategy::EvaluateControlledProbabilityConditionedStopExperiment(path);
    assert(first.experimentIdentityCanonical ==
           repeated.experimentIdentityCanonical);
    assert(first.experimentIdentityHash == repeated.experimentIdentityHash);
    assert(first.canonicalLines == repeated.canonicalLines);
    assert(first.resultHash == repeated.resultHash);
    assert(first.variants.size() == 3);
    assert(first.variants[0].strategyOutputIdentity ==
           Strategy::kBaselineTerminalOutputIdentity);
    assert(first.variants[1].strategyOutputIdentity ==
           Strategy::kFixedStopLossOutputIdentity);
    assert(first.variants[2].strategyOutputIdentity ==
           Strategy::kProbabilityConditionedStopLossOutputIdentity);
    assert(first.pairwiseDeltas.size() == 3);
    assert(first.pairwiseDeltas[0].candidateStrategyOutputIdentity ==
           Strategy::kProbabilityConditionedStopLossOutputIdentity);
    assert(first.pairwiseDeltas[0].referenceStrategyOutputIdentity ==
           Strategy::kBaselineTerminalOutputIdentity);
    assert(first.pairwiseDeltas[1].referenceStrategyOutputIdentity ==
           Strategy::kFixedStopLossOutputIdentity);
    assert(first.pairwiseDeltas[2].candidateStrategyOutputIdentity ==
           Strategy::kFixedStopLossOutputIdentity);
    assert(first.pairwiseDeltas[2].referenceStrategyOutputIdentity ==
           Strategy::kBaselineTerminalOutputIdentity);

    for (const auto& variant : first.variants)
    {
        assert(variant.evaluation.sourceContentHash == path.Hash());
        assert(variant.metrics.observationCount == 5);
        assert(variant.metrics.actionableCount == 4);
        assert(variant.metrics.nonActionableCount == 1);
        assert(variant.metrics.stopHitCount +
                   variant.metrics.terminalExitCount == 4);
        assert(variant.metrics.averageHoldingDurationSeconds);
        assert(variant.metrics.medianHoldingDurationSeconds);
        assert(variant.metrics.maximumAdverseExcursion);
        assert(variant.metrics.maximumFavorableExcursion);
        std::uint64_t bucketCount = 0;
        for (const auto& bucket : variant.confidenceBuckets)
            bucketCount += bucket.actionableCount;
        assert(bucketCount == 4);
        assert(variant.confidenceBuckets[0].actionableCount == 1);
        assert(variant.confidenceBuckets[1].actionableCount == 1);
        assert(variant.confidenceBuckets[2].actionableCount == 1);
        assert(variant.confidenceBuckets[3].actionableCount == 1);
    }
    assert(!first.variants[0].metrics.averageEffectiveStopLogarithmicDistance);
    assert(first.variants[1].metrics.averageStopMultiplier == 1.0);
    assert(first.variants[2].metrics.averageStopMultiplier);
    assert(first.observationEvidence.size() == 15);
    for (const auto& item : first.observationEvidence)
    {
        const auto& observation = path.Observations()[item.observationOrdinal];
        assert(item.predictedClass == observation.predictedClass);
        assert(item.probabilities == *observation.probabilities);
        assert(item.decisionTimestampUnixSeconds ==
               *observation.decisionTimestampUnixSeconds);
        assert(item.decisionPrice == observation.decisionClose);
        if (item.actionable)
        {
            assert(item.execution.entryTimestampUnixSeconds ==
                   observation.decisionTimestampUnixSeconds);
            assert(item.execution.entryPrice == observation.decisionClose);
            assert(item.normalizedDirectionalConfidence);
            assert(item.holdingDurationSeconds);
        }
        else
        {
            assert(item.execution.reason ==
                   Strategy::StrategyExitReason::noAction);
            assert(!item.execution.entryPrice);
            assert(!item.execution.exitPrice);
            assert(!item.directionalProbability);
        }
    }
    assert(first.canonicalLines.find(
        "mapping_identity=directional_probability_linear_bounded_v1") !=
        std::string::npos);
    assert(first.canonicalLines.find("PHASE18A_PAIRWISE_DELTA") !=
           std::string::npos);
    assert(first.canonicalLines.find("PHASE18A_OBSERVATION_EVIDENCE") !=
           std::string::npos);

    const double logTwo = std::log(2.0);
    const auto identity =
        Strategy::EvaluateControlledProbabilityConditionedStopExperiment(
            path, {1, logTwo, 1.0, 1.0});
    assert(identity.variants[1].evaluation.executionResults ==
           identity.variants[2].evaluation.executionResults);
    assert(identity.pairwiseDeltas[1].aggregateDirectionalLogReturnDelta ==
           0.0);
    assert(identity.pairwiseDeltas[1].averageDirectionalLogReturnDelta == 0.0);
    assert(identity.pairwiseDeltas[1].stopHitRateDelta == 0.0);
    assert(identity.pairwiseDeltas[1].averageHoldingDurationSecondsDelta ==
           0.0);

    auto missing = path.Observations();
    missing[0].probabilities.reset();
    const auto missingPath = Strategy::BuildAuthoritativeMarketPath(
        Provenance(), std::move(missing));
    assert(Error([&] {
        Strategy::EvaluateControlledProbabilityConditionedStopExperiment(
            missingPath);
    }) == "probability_conditioned_stop_probabilities_missing");
    return 0;
}
