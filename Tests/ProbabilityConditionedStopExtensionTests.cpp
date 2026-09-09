#include "../Sources/StrategyEvaluationCore/StrategyEvaluation.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Profitability = EA::InferenceProfitability;
namespace Strategy = EA::StrategyEvaluation;

namespace
{

std::string Error(const auto& operation)
{
    try { operation(); }
    catch (const std::exception& error) { return error.what(); }
    return {};
}

Strategy::MarketPathProvenance Provenance(std::size_t horizon)
{
    Strategy::MarketPathProvenance value;
    value.adapterFamily = "phase18b_strategy_fixture";
    value.adapterVersion = 1;
    value.modelId = 18003;
    value.inferenceScientificIdentityCanonical =
        "phase18b_inference_v1;model=18003;fixture=strategy;";
    value.inferenceScientificIdentityHash =
        Profitability::DeterministicHash(
            value.inferenceScientificIdentityCanonical);
    value.symbol = "eurusdrmp";
    value.inferenceWindowSize = 3;
    value.predictionHorizon = horizon;
    value.evaluationStart = "2025-01-01";
    value.evaluationEnd = "2025-01-02";
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
    const float other = (1.0f - directional) / 2.0f;
    if (predictedClass == Profitability::kDownClass)
        return {{directional, other, other}};
    return {{other, other, directional}};
}

Strategy::StrategyEvaluationObservation Observation(
    std::uint64_t ordinal,
    int predictedClass,
    float directional,
    std::vector<Strategy::MarketPathPoint> points)
{
    Strategy::StrategyEvaluationObservation observation;
    observation.observationOrdinal = ordinal;
    observation.inferenceWindowStartRow = ordinal * 10;
    observation.decisionRow = observation.inferenceWindowStartRow + 2;
    observation.terminalRow = observation.decisionRow + points.size();
    observation.predictedClass = predictedClass;
    observation.probabilities = Probabilities(predictedClass, directional);
    observation.decisionClose = 100.0f;
    observation.terminalClose = points.back().close;
    observation.decisionTimestampUnixSeconds =
        1000 + static_cast<std::int64_t>(ordinal) * 10000;
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        points[index].sourceRow = observation.decisionRow + index + 1;
        points[index].timestampUnixSeconds =
            *observation.decisionTimestampUnixSeconds +
            static_cast<std::int64_t>((index + 1) * 900);
    }
    observation.terminalTimestampUnixSeconds =
        points.back().timestampUnixSeconds;
    observation.marketPath = std::move(points);
    return observation;
}

void EqualStatistics(const Profitability::Statistics& left,
                     const Profitability::Statistics& right)
{
    assert(left.predictionCount == right.predictionCount);
    assert(left.actionableCount == right.actionableCount);
    assert(left.winningActionableCount == right.winningActionableCount);
    assert(left.losingActionableCount == right.losingActionableCount);
    assert(left.grossPositiveTerminalHorizonLogReturnSum ==
           right.grossPositiveTerminalHorizonLogReturnSum);
    assert(left.grossNegativeTerminalHorizonLogReturnSum ==
           right.grossNegativeTerminalHorizonLogReturnSum);
    assert(left.aggregateTerminalHorizonLogReturnSum ==
           right.aggregateTerminalHorizonLogReturnSum);
    assert(left.upActionableCount == right.upActionableCount);
    assert(left.downActionableCount == right.downActionableCount);
}

} // namespace

int main()
{
    const Strategy::ProbabilityConditionedStopExtensionStrategy strategy;
    assert(strategy.Identity().family ==
           Strategy::kProbabilityConditionedStopExtensionStrategyFamily);
    assert(strategy.Identity().version == 1);
    assert(strategy.Configuration().baseStopLogarithmicDistance == 0.001);
    assert(strategy.Configuration().activationConfidence == 0.50);
    assert(strategy.Configuration().minimumStopMultiplier == 1.0);
    assert(strategy.Configuration().maximumStopMultiplier == 1.25);

    assert(strategy.StopMultiplierForNormalizedConfidence(0.0) == 1.0);
    assert(strategy.StopMultiplierForNormalizedConfidence(0.25) == 1.0);
    assert(strategy.StopMultiplierForNormalizedConfidence(0.50) == 1.0);
    assert(strategy.StopMultiplierForNormalizedConfidence(0.500001) > 1.0);
    assert(strategy.StopMultiplierForNormalizedConfidence(0.75) == 1.125);
    assert(strategy.StopMultiplierForNormalizedConfidence(1.0) == 1.25);

    assert(strategy.NormalizedDirectionalConfidence(0.0) == 0.0);
    assert(strategy.NormalizedDirectionalConfidence(0.2) == 0.0);
    assert(strategy.NormalizedDirectionalConfidence(1.0 / 3.0) == 0.0);
    assert(strategy.NormalizedDirectionalConfidence(1.0) == 1.0);
    double previous = 1.0;
    for (int ordinal = 0; ordinal <= 1000; ++ordinal)
    {
        const double confidence = static_cast<double>(ordinal) / 1000.0;
        const double multiplier =
            strategy.StopMultiplierForNormalizedConfidence(confidence);
        assert(multiplier >= 1.0);
        assert(multiplier >= previous);
        previous = multiplier;
    }

    assert(Error([&] {
        strategy.NormalizedDirectionalConfidence(-0.01);
    }) == "invalid_probability_conditioned_directional_probability");
    assert(Error([&] {
        strategy.NormalizedDirectionalConfidence(1.01);
    }) == "invalid_probability_conditioned_directional_probability");
    assert(Error([&] {
        strategy.StopMultiplierForNormalizedConfidence(
            std::numeric_limits<double>::quiet_NaN());
    }) == "invalid_probability_conditioned_stop_extension_normalized_confidence");
    assert(Error([] {
        Strategy::ProbabilityConditionedStopExtensionStrategy invalid{{2}};
    }) == "unsupported_probability_conditioned_stop_extension_configuration_schema_version");
    assert(Error([] {
        Strategy::ProbabilityConditionedStopExtensionStrategy invalid{{
            1, 0.001, 0.50, 0.99, 1.25}};
    }) == "invalid_probability_conditioned_stop_extension_configuration");
    assert(Error([] {
        Strategy::ProbabilityConditionedStopExtensionStrategy invalid{{
            1, 0.001, 1.0, 1.0, 1.25}};
    }) == "invalid_probability_conditioned_stop_extension_configuration");

    std::vector<Strategy::StrategyEvaluationObservation> observations;
    observations.push_back(Observation(
        0, Profitability::kUpClass, 0.50f,
        {{0, {}, 100.0f, 100.2f, 99.85f, 100.1f},
         {0, {}, 100.1f, 100.3f, 99.7f, 100.2f}}));
    observations.push_back(Observation(
        1, Profitability::kDownClass, 0.60f,
        {{0, {}, 100.0f, 100.15f, 99.8f, 99.9f},
         {0, {}, 99.9f, 100.3f, 99.6f, 99.7f}}));
    observations.push_back(Observation(
        2, Profitability::kUpClass, 0.80f,
        {{0, {}, 100.0f, 100.4f, 99.8f, 100.2f},
         {0, {}, 100.2f, 100.5f, 99.6f, 100.3f}}));
    const auto path = Strategy::BuildAuthoritativeMarketPath(
        Provenance(2), observations);
    Strategy::StrategyEvaluationInput input{path};

    const Strategy::ProbabilityConditionedStopExtensionStrategy identity{{
        1, 0.001, 0.50, 1.0, 1.0}};
    const auto identityResult = Strategy::EvaluateStrategy(identity, input);
    const auto fixedResult = Strategy::EvaluateStrategy(
        Strategy::FixedStopLossStrategy{{1, 0.001}}, input);
    assert(identityResult.executionResults == fixedResult.executionResults);
    EqualStatistics(identityResult.statistics, fixedResult.statistics);

    const auto extensionResult = Strategy::EvaluateStrategy(strategy, input);
    for (std::size_t index = 0; index < 2; ++index)
    {
        const auto decision = strategy.StopDecision(path.Observations()[index]);
        assert(decision.normalizedDirectionalConfidence <= 0.50);
        assert(decision.stopMultiplier == 1.0);
        assert(extensionResult.executionResults[index] ==
               fixedResult.executionResults[index]);
    }

    const Strategy::ProbabilityConditionedStopExtensionStrategy changed{{
        1, 0.001, 0.60, 1.0, 1.25}};
    assert(changed.Identity().hash != strategy.Identity().hash);
    const Strategy::ProbabilityConditionedStopExtensionStrategy changedMax{{
        1, 0.001, 0.50, 1.0, 1.20}};
    assert(changedMax.Identity().hash != strategy.Identity().hash);

    const auto repeated = Strategy::EvaluateStrategy(strategy, input);
    assert(extensionResult.executionResults == repeated.executionResults);
    assert(extensionResult.resultCanonical == repeated.resultCanonical);
    assert(extensionResult.resultHash == repeated.resultHash);
    return 0;
}
