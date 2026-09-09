#include "../Sources/StrategyEvaluationCore/StrategyEvaluation.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace Strategy = EA::StrategyEvaluation;
namespace Profitability = EA::InferenceProfitability;

namespace
{

Strategy::PredictionProbabilities Probabilities(int predictedClass)
{
    if (predictedClass == Profitability::kDownClass)
        return {{0.8f, 0.1f, 0.1f}};
    if (predictedClass == Profitability::kUpClass)
        return {{0.1f, 0.1f, 0.8f}};
    return {{0.1f, 0.8f, 0.1f}};
}

Strategy::AuthoritativeMarketPath Fixture()
{
    Strategy::MarketPathProvenance provenance;
    provenance.adapterFamily = "phase17d_fixture";
    provenance.adapterVersion = 1;
    provenance.modelId = 17004;
    provenance.inferenceScientificIdentityCanonical =
        "phase17d_inference_v1;model=17004;period=fixed;";
    provenance.inferenceScientificIdentityHash =
        Profitability::DeterministicHash(
            provenance.inferenceScientificIdentityCanonical);
    provenance.symbol = "eurusdrmp";
    provenance.inferenceWindowSize = 3;
    provenance.predictionHorizon = 2;
    provenance.evaluationStart = "2025-01-01";
    provenance.evaluationEnd = "2025-02-01";
    provenance.barIntervalSeconds = 900;
    provenance.timestampSemantics = "fixture_utc";
    provenance.ohlcIntervalSemantics = "fixture_ohlc";
    provenance.priceDomain = "ask";
    provenance.marketDataSource = "immutable_fixture";
    provenance.marketDataSourceRelation = "fixture";
    provenance.pathOrdering = "source_row_ascending";
    provenance.metricDefinitionCanonical =
        Strategy::kFixedStopMetricDefinitionCanonical;
    provenance.metricDefinitionHash = Strategy::FixedStopMetricDefinitionHash();

    std::vector<Strategy::StrategyEvaluationObservation> observations;
    const int classes[] = {Profitability::kUpClass,
                           Profitability::kNeutralClass,
                           Profitability::kDownClass};
    for (std::uint64_t ordinal = 0; ordinal < 3; ++ordinal)
    {
        Strategy::StrategyEvaluationObservation observation;
        observation.observationOrdinal = ordinal;
        observation.inferenceWindowStartRow = ordinal * 10;
        observation.decisionRow = ordinal * 10 + 2;
        observation.terminalRow = ordinal * 10 + 4;
        observation.predictedClass = classes[ordinal];
        observation.probabilities = Probabilities(classes[ordinal]);
        observation.decisionClose = 100.0f;
        observation.terminalClose = ordinal == 2 ? 99.0f : 101.0f;
        observation.decisionTimestampUnixSeconds = 1000 + ordinal * 10000;
        observation.terminalTimestampUnixSeconds = 2800 + ordinal * 10000;
        observation.marketPath = {
            {ordinal * 10 + 3, 1900 + static_cast<std::int64_t>(ordinal * 10000),
             100.0f, 100.1f, 99.9f, 100.0f},
            {ordinal * 10 + 4, 2800 + static_cast<std::int64_t>(ordinal * 10000),
             100.0f, 101.2f, 98.8f, observation.terminalClose}};
        observations.push_back(std::move(observation));
    }
    return Strategy::BuildAuthoritativeMarketPath(
        std::move(provenance), std::move(observations));
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
    const auto first = Strategy::EvaluateControlledFixedStopExperiment(path);
    const auto repeated = Strategy::EvaluateControlledFixedStopExperiment(path);
    assert(first.canonicalCsv == repeated.canonicalCsv);
    assert(first.resultHash == repeated.resultHash);
    assert(first.experimentIdentityHash == repeated.experimentIdentityHash);
    assert(first.variants.size() == 4);
    assert(first.comparisonMetricDefinitionHash ==
           Profitability::MetricDefinitionHash());
    assert(first.comparisonMetricDefinitionCanonical ==
           Profitability::kMetricDefinitionCanonical);

    assert(Strategy::kControlledFixedStopDistanceGrid[0] == 0.0005);
    assert(Strategy::kControlledFixedStopDistanceGrid[1] == 0.0010);
    assert(Strategy::kControlledFixedStopDistanceGrid[2] == 0.0020);
    for (std::size_t index = 0; index < first.variants.size(); ++index)
    {
        const auto& variant = first.variants[index];
        assert(variant.evaluation.sourceContentHash == path.Hash());
        assert(variant.evaluation.statistics.predictionCount == 3);
        assert(variant.noActionCount == 1);
        assert(variant.noActionCount +
                   variant.evaluation.statistics.actionableCount == 3);
        assert(variant.fixedStopExitCount + variant.terminalExitCount ==
               variant.evaluation.statistics.actionableCount);
        assert(variant.evaluation.executionResults[1].reason ==
               Strategy::StrategyExitReason::noAction);
        assert(variant.evaluation.executionResults[0].observationOrdinal == 0);
        assert(variant.evaluation.executionResults[2].observationOrdinal == 2);
    }
    assert(first.variants[0].fixedStopExitCount == 0);
    assert(first.variants[0].terminalExitCount == 2);
    assert(first.variants[0].evaluation.metricDefinitionHash ==
           Profitability::MetricDefinitionHash());
    assert(first.variants[1].evaluation.metricDefinitionHash ==
           Strategy::FixedStopMetricDefinitionHash());
    assert(first.variants[1].evaluation.strategyIdentity.hash !=
           first.variants[2].evaluation.strategyIdentity.hash);
    assert(first.variants[2].evaluation.strategyIdentity.hash !=
           first.variants[3].evaluation.strategyIdentity.hash);

    Strategy::StrategyEvaluationInput baselineInput{path};
    const auto baseline = Strategy::EvaluateStrategy(
        Strategy::BaselineTerminalStrategy{}, baselineInput);
    Profitability::Accumulator historical;
    for (const auto& observation : path.Observations())
        historical.Observe(observation.predictedClass,
                           observation.decisionClose,
                           observation.terminalClose);
    assert(baseline.statistics.aggregateTerminalHorizonLogReturnSum ==
           historical.statistics().aggregateTerminalHorizonLogReturnSum);
    assert(baseline.statistics.actionableCount ==
           historical.statistics().actionableCount);

    const auto changedFuture = [&] {
        auto provenance = path.Provenance();
        auto observations = path.Observations();
        observations[0].marketPath[0].high = 100.2f;
        return Strategy::BuildAuthoritativeMarketPath(
            std::move(provenance), std::move(observations));
    }();
    const auto changed = Strategy::EvaluateControlledFixedStopExperiment(
        changedFuture);
    assert(changed.variants[1].evaluation.executionResults[0].direction ==
           first.variants[1].evaluation.executionResults[0].direction);
    assert(changed.variants[1].evaluation.executionResults[0].initialStopPrice ==
           first.variants[1].evaluation.executionResults[0].initialStopPrice);

    assert(Error([&] {
        Strategy::EvaluateControlledFixedStopExperiment(path, {0.001, 0.001});
    }) == "invalid_or_duplicate_fixed_stop_grid_distance");
    assert(Error([&] {
        Strategy::EvaluateControlledFixedStopExperiment(path, {0.0});
    }) == "invalid_or_duplicate_fixed_stop_grid_distance");
    assert(Error([&] {
        Strategy::EvaluateControlledFixedStopExperiment(
            path, {std::numeric_limits<double>::infinity()});
    }) == "invalid_or_duplicate_fixed_stop_grid_distance");
    assert(Error([&] {
        Strategy::EvaluateControlledFixedStopExperiment(path, {});
    }) == "empty_fixed_stop_grid");
    assert(Error([&] {
        Strategy::EvaluateControlledFixedStopExperiment(
            path, {0.0005, 0.0020, 0.0010});
    }) == "controlled_fixed_stop_grid_mismatch");
    assert(Error([&] {
        Strategy::EvaluateControlledFixedStopExperiment(
            path, {0.0005, 0.0010, 0.0030});
    }) == "controlled_fixed_stop_grid_mismatch");
    return 0;
}
