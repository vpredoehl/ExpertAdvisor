#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Sources/InferenceProfitability.hpp"
#include "../Sources/StrategyEvaluationCore/StrategyEvaluation.hpp"

namespace Profitability = EA::InferenceProfitability;
namespace Strategy = EA::StrategyEvaluation;

namespace
{

std::string ErrorFrom(const auto& operation)
{
    try
    {
        operation();
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    return {};
}

Strategy::PredictionProbabilities ProbabilitiesFor(int predictedClass)
{
    if (predictedClass == Profitability::kDownClass)
        return {{0.8f, 0.1f, 0.1f}};
    if (predictedClass == Profitability::kUpClass)
        return {{0.1f, 0.1f, 0.8f}};
    return {{0.2f, 0.6f, 0.2f}};
}

Strategy::MarketPathProvenance Provenance(std::size_t horizon)
{
    Strategy::MarketPathProvenance value;
    value.adapterFamily = "test_authoritative_tensor_adapter";
    value.adapterVersion = 1;
    value.modelId = 42;
    value.inferenceScientificIdentityCanonical =
        "inference_scientific_identity_v1;model=42;fixture=fixed_stop;";
    value.inferenceScientificIdentityHash = Profitability::DeterministicHash(
        value.inferenceScientificIdentityCanonical);
    value.symbol = "eurusdrmp";
    value.inferenceWindowSize = 4;
    value.predictionHorizon = horizon;
    value.evaluationStart = "2025-01-01";
    value.evaluationEnd = "2025-02-01";
    value.barIntervalSeconds = 900;
    value.timestampSemantics = "utc_unix_seconds_bar_start_v1";
    value.ohlcIntervalSemantics = "ask_ohlc_15_minute_v1";
    value.priceDomain = "ask";
    value.marketDataSource = "fixture_using_production_contract_v1";
    value.marketDataSourceRelation = "eurusdrmp";
    value.pathOrdering = "strict_timestamp_ascending_source_row_v1";
    value.metricDefinitionCanonical =
        Strategy::kFixedStopMetricDefinitionCanonical;
    value.metricDefinitionHash = Strategy::FixedStopMetricDefinitionHash();
    return value;
}

Strategy::MarketPathPoint Point(std::uint64_t row,
                                std::int64_t timestamp,
                                float open,
                                float high,
                                float low,
                                float close)
{
    return {row, timestamp, open, high, low, close};
}

Strategy::StrategyEvaluationObservation Observation(
    int predictedClass,
    float entry,
    std::vector<Strategy::MarketPathPoint> path,
    std::uint64_t ordinal = 0,
    std::uint64_t windowStartRow = 7)
{
    Strategy::StrategyEvaluationObservation value;
    value.observationOrdinal = ordinal;
    value.inferenceWindowStartRow = windowStartRow;
    value.decisionRow = windowStartRow + 3;
    value.terminalRow = value.decisionRow + path.size();
    value.predictedClass = predictedClass;
    value.probabilities = ProbabilitiesFor(predictedClass);
    value.decisionClose = entry;
    value.terminalClose = path.back().close;
    value.decisionTimestampUnixSeconds = 1000 +
        static_cast<std::int64_t>(ordinal) * 10000;
    value.terminalTimestampUnixSeconds =
        path.back().timestampUnixSeconds;
    value.marketPath = std::move(path);
    return value;
}

Strategy::AuthoritativeMarketPath PathFor(
    int predictedClass,
    const std::vector<Strategy::MarketPathPoint>& points,
    float entry = 100.0f)
{
    return Strategy::BuildAuthoritativeMarketPath(
        Provenance(points.size()),
        {Observation(predictedClass, entry, points)});
}

Strategy::StrategyEvaluationResult Evaluate(
    const Strategy::FixedStopLossStrategy& strategy,
    int predictedClass,
    const std::vector<Strategy::MarketPathPoint>& points,
    float entry = 100.0f)
{
    return Strategy::EvaluateStrategy(
        strategy,
        Strategy::StrategyEvaluationInput(
            PathFor(predictedClass, points, entry)));
}

void AssertNear(double actual, double expected)
{
    assert(std::fabs(actual - expected) < 1.0e-12);
}

void AssertStop(const Strategy::StrategyEvaluationResult& result,
                double exitPrice,
                std::uint64_t pathOrdinal)
{
    assert(result.executionResults.size() == 1);
    const auto& execution = result.executionResults.front();
    assert(execution.reason == Strategy::StrategyExitReason::fixedStopExit);
    assert(execution.exitPrice);
    AssertNear(*execution.exitPrice, exitPrice);
    assert(execution.triggeringPathPointOrdinal == pathOrdinal);
    assert(result.statistics.actionableCount == 1);
    assert(result.statistics.losingActionableCount == 1);
}

void AssertTerminal(const Strategy::StrategyEvaluationResult& result,
                    double exitPrice,
                    bool winning)
{
    assert(result.executionResults.size() == 1);
    const auto& execution = result.executionResults.front();
    assert(execution.reason == Strategy::StrategyExitReason::terminalExit);
    assert(execution.exitPrice);
    AssertNear(*execution.exitPrice, exitPrice);
    assert(!execution.triggeringPathPointOrdinal);
    assert(result.statistics.actionableCount == 1);
    assert(result.statistics.winningActionableCount == (winning ? 1U : 0U));
    assert(result.statistics.losingActionableCount == (winning ? 0U : 1U));
}

} // namespace

int main()
{
    const std::vector<Strategy::MarketPathPoint> basePoints = {
        Point(11, 1900, 100.0f, 110.0f, 90.0f, 105.0f),
        Point(12, 2800, 105.0f, 115.0f, 80.0f, 110.0f),
        Point(13, 3700, 110.0f, 120.0f, 70.0f, 115.0f)};
    const auto base = PathFor(Profitability::kUpClass, basePoints);
    const auto repeated = PathFor(Profitability::kUpClass, basePoints);
    assert(base.Canonical() == repeated.Canonical());
    assert(base.Hash() == repeated.Hash());

    auto orderingChanged = basePoints;
    std::swap(orderingChanged[0].open, orderingChanged[1].open);
    std::swap(orderingChanged[0].high, orderingChanged[1].high);
    std::swap(orderingChanged[0].low, orderingChanged[1].low);
    std::swap(orderingChanged[0].close, orderingChanged[1].close);
    assert(PathFor(Profitability::kUpClass, orderingChanged).Hash() !=
           base.Hash());
    auto timestampChanged = basePoints;
    timestampChanged[1].timestampUnixSeconds = 2801;
    assert(PathFor(Profitability::kUpClass, timestampChanged).Hash() !=
           base.Hash());
    auto ohlcChanged = basePoints;
    ohlcChanged[1].high = 116.0f;
    assert(PathFor(Profitability::kUpClass, ohlcChanged).Hash() !=
           base.Hash());
    auto provenanceChanged = Provenance(basePoints.size());
    provenanceChanged.priceDomain = "midpoint";
    const auto changedProvenancePath = Strategy::BuildAuthoritativeMarketPath(
        provenanceChanged,
        {Observation(Profitability::kUpClass, 100.0f, basePoints)});
    assert(changedProvenancePath.Hash() != base.Hash());

    auto unordered = basePoints;
    unordered[1].timestampUnixSeconds = 1900;
    assert(ErrorFrom([&] { PathFor(Profitability::kUpClass, unordered); }) ==
           "unordered_market_path_points");
    auto missingTimestamp = basePoints;
    missingTimestamp[0].timestampUnixSeconds.reset();
    assert(ErrorFrom([&] {
        PathFor(Profitability::kUpClass, missingTimestamp);
    }) == "missing_market_path_timestamp");
    auto nonfinite = basePoints;
    nonfinite[0].high = std::numeric_limits<float>::infinity();
    assert(ErrorFrom([&] { PathFor(Profitability::kUpClass, nonfinite); }) ==
           "invalid_market_path_ohlc_price");
    auto impossible = basePoints;
    impossible[0].high = 99.0f;
    assert(ErrorFrom([&] { PathFor(Profitability::kUpClass, impossible); }) ==
           "impossible_market_path_ohlc");
    auto unsupportedProvenance = Provenance(basePoints.size());
    unsupportedProvenance.canonicalizationVersion = 2;
    assert(ErrorFrom([&] {
        Strategy::BuildAuthoritativeMarketPath(
            unsupportedProvenance,
            {Observation(Profitability::kUpClass, 100.0f, basePoints)});
    }) == "unsupported_market_path_canonicalization_version");

    const double distance = std::log(2.0);
    const Strategy::FixedStopLossStrategy fixed{{1, distance}};
    assert(ErrorFrom([&] {
        Strategy::FixedStopLossStrategy invalid{{2, distance}};
    }) == "unsupported_fixed_stop_configuration_schema_version");
    assert(ErrorFrom([&] {
        Strategy::FixedStopLossStrategy invalid{{1, 0.0}};
    }) == "invalid_fixed_stop_logarithmic_distance");
    assert(ErrorFrom([&] {
        Strategy::FixedStopLossStrategy invalid{{1, -0.1}};
    }) == "invalid_fixed_stop_logarithmic_distance");
    assert(ErrorFrom([&] {
        Strategy::FixedStopLossStrategy invalid{{
            1, std::numeric_limits<double>::quiet_NaN()}};
    }) == "invalid_fixed_stop_logarithmic_distance");

    // Long: terminal exits, exact touch, pass-through, gap, and a favorable
    // move followed by a hit all use the one original stop at 50.
    AssertTerminal(Evaluate(fixed, Profitability::kUpClass, {
        Point(11, 1900, 100, 120, 60, 110)}), 110.0, true);
    AssertTerminal(Evaluate(fixed, Profitability::kUpClass, {
        Point(11, 1900, 100, 110, 60, 90)}), 90.0, false);
    AssertStop(Evaluate(fixed, Profitability::kUpClass, {
        Point(11, 1900, 60, 70, 50, 55)}), 50.0, 0);
    AssertStop(Evaluate(fixed, Profitability::kUpClass, {
        Point(11, 1900, 60, 70, 40, 55)}), 50.0, 0);
    AssertStop(Evaluate(fixed, Profitability::kUpClass, {
        Point(11, 1900, 40, 60, 35, 55)}), 40.0, 0);
    const auto longFavorableThenStop = Evaluate(
        fixed, Profitability::kUpClass, {
            Point(11, 1900, 100, 200, 70, 180),
            Point(12, 2800, 180, 185, 40, 45)});
    AssertStop(longFavorableThenStop, 50.0, 1);
    AssertNear(*longFavorableThenStop.executionResults[0].initialStopPrice,
               50.0);

    // Short is mathematically mirrored: the immutable stop is 200.
    AssertTerminal(Evaluate(fixed, Profitability::kDownClass, {
        Point(11, 1900, 100, 150, 80, 90)}), 90.0, true);
    AssertTerminal(Evaluate(fixed, Profitability::kDownClass, {
        Point(11, 1900, 100, 150, 80, 110)}), 110.0, false);
    AssertStop(Evaluate(fixed, Profitability::kDownClass, {
        Point(11, 1900, 150, 200, 140, 160)}), 200.0, 0);
    AssertStop(Evaluate(fixed, Profitability::kDownClass, {
        Point(11, 1900, 150, 210, 140, 160)}), 200.0, 0);
    AssertStop(Evaluate(fixed, Profitability::kDownClass, {
        Point(11, 1900, 220, 230, 180, 190)}), 220.0, 0);
    const auto shortFavorableThenStop = Evaluate(
        fixed, Profitability::kDownClass, {
            Point(11, 1900, 100, 150, 50, 60),
            Point(12, 2800, 60, 210, 55, 205)});
    AssertStop(shortFavorableThenStop, 200.0, 1);
    AssertNear(*shortFavorableThenStop.executionResults[0].initialStopPrice,
               200.0);

    const auto neutral = Evaluate(fixed, Profitability::kNeutralClass, {
        Point(11, 1900, 100, 300, 10, 250)});
    assert(neutral.executionResults[0].reason ==
           Strategy::StrategyExitReason::noAction);
    assert(!neutral.executionResults[0].initialStopPrice);
    assert(neutral.statistics.predictionCount == 1);
    assert(neutral.statistics.actionableCount == 0);

    // Future path changes alter provenance/result identity but never the
    // decision class or initial stop.
    const auto futureA = Evaluate(fixed, Profitability::kUpClass, {
        Point(11, 1900, 100, 120, 60, 110)});
    const auto futureB = Evaluate(fixed, Profitability::kUpClass, {
        Point(11, 1900, 100, 140, 55, 130)});
    assert(futureA.sourceContentHash != futureB.sourceContentHash);
    assert(futureA.evaluationIdentity->hash != futureB.evaluationIdentity->hash);
    assert(*futureA.executionResults[0].initialStopPrice ==
           *futureB.executionResults[0].initialStopPrice);
    assert(futureA.executionResults[0].direction ==
           futureB.executionResults[0].direction);

    const auto deterministicFirst = Evaluate(
        fixed, Profitability::kUpClass, basePoints);
    const auto deterministicSecond = Evaluate(
        fixed, Profitability::kUpClass, basePoints);
    assert(deterministicFirst.resultCanonical ==
           deterministicSecond.resultCanonical);
    assert(deterministicFirst.resultHash == deterministicSecond.resultHash);
    assert(deterministicFirst.executionResults ==
           deterministicSecond.executionResults);

    Strategy::StrategyConfiguration ruleV1;
    ruleV1.entries = {{"execution_rule", Strategy::kFixedStopExecutionRule},
                      {"execution_rule_version", "1"}};
    Strategy::StrategyConfiguration ruleV2 = ruleV1;
    ruleV2.entries[1].value = "2";
    assert(Strategy::BuildStrategyIdentity(
               Strategy::kFixedStopLossStrategyFamily, 1, ruleV1).hash !=
           Strategy::BuildStrategyIdentity(
               Strategy::kFixedStopLossStrategyFamily, 1, ruleV2).hash);

    Strategy::StrategyEvaluationInput baselineOnly({
        {Profitability::kUpClass, 100.0f, 110.0f}});
    assert(ErrorFrom([&] { Strategy::EvaluateStrategy(fixed, baselineOnly); }) ==
           "fixed_stop_requires_authoritative_market_path");

    return 0;
}
