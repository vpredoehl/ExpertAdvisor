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
    value.adapterFamily = "phase18a_test_adapter";
    value.adapterVersion = 1;
    value.modelId = 18001;
    value.inferenceScientificIdentityCanonical =
        "phase18a_inference_v1;model=18001;fixture=strategy;";
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
                                                float directional = 0.8f)
{
    const float remainder = (1.0f - directional) / 2.0f;
    if (predictedClass == Profitability::kDownClass)
        return {{directional, remainder, remainder}};
    if (predictedClass == Profitability::kUpClass)
        return {{remainder, remainder, directional}};
    return {{0.2f, 0.6f, 0.2f}};
}

Strategy::StrategyEvaluationObservation Observation(
    int predictedClass,
    Strategy::PredictionProbabilities probabilities,
    std::vector<Strategy::MarketPathPoint> points,
    std::uint64_t ordinal = 0,
    std::uint64_t windowStart = 10)
{
    Strategy::StrategyEvaluationObservation observation;
    observation.observationOrdinal = ordinal;
    observation.inferenceWindowStartRow = windowStart;
    observation.decisionRow = windowStart + 2;
    observation.terminalRow = observation.decisionRow + points.size();
    observation.predictedClass = predictedClass;
    observation.probabilities = probabilities;
    observation.decisionClose = 100.0f;
    observation.terminalClose = points.back().close;
    observation.decisionTimestampUnixSeconds =
        1000 + static_cast<std::int64_t>(ordinal) * 10000;
    observation.terminalTimestampUnixSeconds =
        points.back().timestampUnixSeconds;
    observation.marketPath = std::move(points);
    return observation;
}

Strategy::AuthoritativeMarketPath Path(
    int predictedClass,
    const std::vector<Strategy::MarketPathPoint>& points,
    float directional = 0.8f)
{
    return Strategy::BuildAuthoritativeMarketPath(
        Provenance(points.size()),
        {Observation(predictedClass,
                     Probabilities(predictedClass, directional), points)});
}

Strategy::StrategyEvaluationResult Evaluate(
    const Strategy::ProbabilityConditionedStopLossStrategy& strategy,
    int predictedClass,
    const std::vector<Strategy::MarketPathPoint>& points,
    float directional = 0.8f)
{
    return Strategy::EvaluateStrategy(
        strategy, Strategy::StrategyEvaluationInput(
            Path(predictedClass, points, directional)));
}

void Near(double actual, double expected)
{
    assert(std::fabs(actual - expected) < 1.0e-12);
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
    assert(left.upWinningActionableCount == right.upWinningActionableCount);
    assert(left.upLosingActionableCount == right.upLosingActionableCount);
    assert(left.downWinningActionableCount == right.downWinningActionableCount);
    assert(left.downLosingActionableCount == right.downLosingActionableCount);
    assert(left.upTerminalHorizonLogReturnSum ==
           right.upTerminalHorizonLogReturnSum);
    assert(left.downTerminalHorizonLogReturnSum ==
           right.downTerminalHorizonLogReturnSum);
    assert(left.upGrossPositiveTerminalHorizonLogReturnSum ==
           right.upGrossPositiveTerminalHorizonLogReturnSum);
    assert(left.upGrossNegativeTerminalHorizonLogReturnSum ==
           right.upGrossNegativeTerminalHorizonLogReturnSum);
    assert(left.downGrossPositiveTerminalHorizonLogReturnSum ==
           right.downGrossPositiveTerminalHorizonLogReturnSum);
    assert(left.downGrossNegativeTerminalHorizonLogReturnSum ==
           right.downGrossNegativeTerminalHorizonLogReturnSum);
}

} // namespace

int main()
{
    const Strategy::ProbabilityConditionedStopLossStrategy defaultStrategy;
    assert(defaultStrategy.Identity().family ==
           Strategy::kProbabilityConditionedStopLossStrategyFamily);
    assert(defaultStrategy.Identity().version == 1);
    assert(defaultStrategy.Configuration().baseStopLogarithmicDistance ==
           0.0010);

    // The predeclared v1 mapping is locked independently of any outcome.
    assert(defaultStrategy.NormalizedDirectionalConfidence(1.0 / 3.0) == 0.0);
    assert(defaultStrategy.NormalizedDirectionalConfidence(2.0 / 3.0) == 0.5);
    assert(defaultStrategy.NormalizedDirectionalConfidence(1.0) == 1.0);
    assert(defaultStrategy.StopMultiplierForNormalizedConfidence(0.0) == 0.75);
    assert(defaultStrategy.StopMultiplierForNormalizedConfidence(0.5) == 1.0);
    assert(defaultStrategy.StopMultiplierForNormalizedConfidence(1.0) == 1.25);
    assert(defaultStrategy.NormalizedDirectionalConfidence(0.0) == 0.0);
    assert(Error([&] {
        defaultStrategy.NormalizedDirectionalConfidence(
            std::numeric_limits<double>::quiet_NaN());
    }) == "invalid_probability_conditioned_directional_probability");
    assert(Error([&] {
        defaultStrategy.NormalizedDirectionalConfidence(1.01);
    }) == "invalid_probability_conditioned_directional_probability");

    assert(Error([] {
        Strategy::ProbabilityConditionedStopLossStrategy invalid{{2}};
    }) == "unsupported_probability_conditioned_stop_configuration_schema_version");
    assert(Error([] {
        Strategy::ProbabilityConditionedStopLossStrategy invalid{{1, 0.0}};
    }) == "invalid_probability_conditioned_stop_configuration");
    assert(Error([] {
        Strategy::ProbabilityConditionedStopLossStrategy invalid{{
            1, 0.001, 1.25, 0.75}};
    }) == "invalid_probability_conditioned_stop_configuration");
    assert(Error([] {
        Strategy::ProbabilityConditionedStopLossStrategy invalid{{
            1, 0.001, std::numeric_limits<double>::infinity(), 1.25}};
    }) == "invalid_probability_conditioned_stop_configuration");

    const std::vector<Strategy::MarketPathPoint> longTerminalPoints = {
        {13, 1900, 100.0f, 100.05f, 99.95f, 100.02f},
        {14, 2800, 100.02f, 100.08f, 99.94f, 100.06f}};
    const auto longTerminal = Evaluate(
        defaultStrategy, Profitability::kUpClass, longTerminalPoints);
    assert(longTerminal.executionResults[0].reason ==
           Strategy::StrategyExitReason::terminalExit);
    const auto longDecision = defaultStrategy.StopDecision(
        Path(Profitability::kUpClass, longTerminalPoints).
            Observations().front());
    const double expectedConfidence =
        (static_cast<double>(0.8f) - 1.0 / 3.0) / (2.0 / 3.0);
    const double expectedMultiplier = 0.75 + 0.50 * expectedConfidence;
    Near(longDecision.normalizedDirectionalConfidence, expectedConfidence);
    Near(longDecision.stopMultiplier, expectedMultiplier);
    Near(longDecision.effectiveStopLogarithmicDistance,
         0.001 * expectedMultiplier);
    Near(*longTerminal.executionResults[0].initialStopPrice,
         longDecision.initialStopPrice);

    const std::vector<Strategy::MarketPathPoint> shortTerminalPoints = {
        {13, 1900, 100.0f, 100.05f, 99.95f, 99.98f},
        {14, 2800, 99.98f, 100.06f, 99.92f, 99.94f}};
    const auto shortTerminal = Evaluate(
        defaultStrategy, Profitability::kDownClass, shortTerminalPoints);
    assert(shortTerminal.executionResults[0].reason ==
           Strategy::StrategyExitReason::terminalExit);

    const double logTwo = std::log(2.0);
    const Strategy::ProbabilityConditionedStopLossStrategy identity{{
        1, logTwo, 1.0, 1.0}};

    // Exact touch and pass-through fill at the stop; gaps fill at the open.
    const auto longTouch = Evaluate(identity, Profitability::kUpClass, {
        {13, 1900, 60.0f, 70.0f, 50.0f, 55.0f}});
    assert(longTouch.executionResults[0].reason ==
           Strategy::StrategyExitReason::fixedStopExit);
    Near(*longTouch.executionResults[0].exitPrice, 50.0);
    const auto longGap = Evaluate(identity, Profitability::kUpClass, {
        {13, 1900, 40.0f, 60.0f, 35.0f, 55.0f}});
    Near(*longGap.executionResults[0].exitPrice, 40.0);
    const auto shortTouch = Evaluate(identity, Profitability::kDownClass, {
        {13, 1900, 150.0f, 200.0f, 140.0f, 160.0f}});
    Near(*shortTouch.executionResults[0].exitPrice, 200.0);
    const auto shortGap = Evaluate(identity, Profitability::kDownClass, {
        {13, 1900, 220.0f, 230.0f, 180.0f, 190.0f}});
    Near(*shortGap.executionResults[0].exitPrice, 220.0);
    Near(*longTouch.executionResults[0].directionalLogReturn,
         *shortTouch.executionResults[0].directionalLogReturn);

    const auto neutral = Evaluate(identity, Profitability::kNeutralClass, {
        {13, 1900, 100.0f, 300.0f, 10.0f, 250.0f}});
    assert(neutral.executionResults[0].reason ==
           Strategy::StrategyExitReason::noAction);
    assert(!neutral.executionResults[0].entryPrice);
    assert(!neutral.executionResults[0].initialStopPrice);
    assert(neutral.statistics.actionableCount == 0);

    // Identity conditioning is bit-for-bit equal to FixedStopLossStrategy v1
    // for every execution field and profitability statistic used here.
    std::vector<Strategy::StrategyEvaluationObservation> observations;
    observations.push_back(Observation(
        Profitability::kUpClass, Probabilities(Profitability::kUpClass),
        {{13, 1900, 60, 70, 50, 55}}, 0, 10));
    observations.push_back(Observation(
        Profitability::kNeutralClass,
        Probabilities(Profitability::kNeutralClass),
        {{23, 11900, 100, 110, 90, 100}}, 1, 20));
    observations.push_back(Observation(
        Profitability::kDownClass, Probabilities(Profitability::kDownClass),
        {{33, 21900, 100, 150, 80, 90}}, 2, 30));
    const auto parityPath = Strategy::BuildAuthoritativeMarketPath(
        Provenance(1), observations);
    Strategy::StrategyEvaluationInput parityInput{parityPath};
    const auto conditionedParity = Strategy::EvaluateStrategy(
        identity, parityInput);
    const auto fixedParity = Strategy::EvaluateStrategy(
        Strategy::FixedStopLossStrategy{{1, logTwo}}, parityInput);
    assert(conditionedParity.executionResults == fixedParity.executionResults);
    EqualStatistics(conditionedParity.statistics, fixedParity.statistics);

    const auto repeated = Strategy::EvaluateStrategy(identity, parityInput);
    assert(conditionedParity.executionResults == repeated.executionResults);
    assert(conditionedParity.resultCanonical == repeated.resultCanonical);
    assert(conditionedParity.resultHash == repeated.resultHash);

    auto missingObservation = observations.front();
    missingObservation.observationOrdinal = 0;
    missingObservation.probabilities.reset();
    const auto missingPath = Strategy::BuildAuthoritativeMarketPath(
        Provenance(1), {missingObservation});
    assert(Error([&] {
        Strategy::EvaluateStrategy(
            defaultStrategy,
            Strategy::StrategyEvaluationInput(missingPath));
    }) == "probability_conditioned_stop_probabilities_missing");

    auto invalidObservation = observations.front();
    invalidObservation.observationOrdinal = 0;
    invalidObservation.probabilities = Strategy::PredictionProbabilities{{
        std::numeric_limits<float>::quiet_NaN(), 0.1f, 0.9f}};
    assert(Error([&] {
        Strategy::BuildAuthoritativeMarketPath(
            Provenance(1), {invalidObservation});
    }) == "invalid_market_path_prediction_probability");
    invalidObservation.probabilities = Strategy::PredictionProbabilities{{
        -0.1f, 0.1f, 1.0f}};
    assert(Error([&] {
        Strategy::BuildAuthoritativeMarketPath(
            Provenance(1), {invalidObservation});
    }) == "invalid_market_path_prediction_probability");
    return 0;
}
