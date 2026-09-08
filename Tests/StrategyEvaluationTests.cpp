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

void AssertStatisticsEqual(const Profitability::Statistics& left,
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
    assert(left.AverageTerminalHorizonLogReturnPerActionablePrediction() ==
           right.AverageTerminalHorizonLogReturnPerActionablePrediction());
}

void AssertBaselineParity(
    const std::vector<Strategy::StrategyEvaluationObservation>& observations)
{
    Profitability::Accumulator historical;
    for (const auto& observation : observations)
    {
        historical.Observe(observation.predictedClass,
                           observation.decisionClose,
                           observation.terminalClose);
    }

    Strategy::StrategyEvaluationInput input{observations};
    const Strategy::BaselineTerminalStrategy baseline;
    const auto first = Strategy::EvaluateStrategy(baseline, input);
    const auto repeated = Strategy::EvaluateStrategy(baseline, input);

    AssertStatisticsEqual(first.statistics, historical.statistics());
    AssertStatisticsEqual(repeated.statistics, historical.statistics());
    assert(first.sourceContentHash == historical.SourceContentHash());
    assert(repeated.sourceContentHash == historical.SourceContentHash());
    assert(first.metricDefinitionCanonical ==
           Profitability::kMetricDefinitionCanonical);
    assert(first.metricDefinitionHash == Profitability::MetricDefinitionHash());
    assert(first.strategyIdentity == repeated.strategyIdentity);
}

} // namespace

int main()
{
    Strategy::StrategyConfiguration ordered;
    ordered.entries = {{"threshold", "0.25"}, {"mode", "terminal"}};
    Strategy::StrategyConfiguration reversed;
    reversed.entries = {{"mode", "terminal"}, {"threshold", "0.25"}};
    assert(Strategy::CanonicalConfiguration(ordered) ==
           Strategy::CanonicalConfiguration(reversed));
    assert(Strategy::ConfigurationHash(ordered) ==
           Strategy::ConfigurationHash(reversed));
    assert(Strategy::CanonicalConfiguration(ordered) ==
           "strategy_configuration_v1;schema_version=1:1;entry_count=1:2;"
           "entry[0].key=4:mode;entry[0].value=8:terminal;"
           "entry[1].key=9:threshold;entry[1].value=4:0.25;");
    assert(Strategy::ConfigurationHash(ordered) ==
           "fnv1a64:9c517a4ae4833029");

    const auto identity =
        Strategy::BuildStrategyIdentity("example", 1, ordered);
    assert(identity.configurationHash == Strategy::ConfigurationHash(ordered));
    assert(identity.hash.starts_with("fnv1a64:"));
    assert(Strategy::BuildStrategyIdentity("other", 1, ordered).hash !=
           identity.hash);
    assert(Strategy::BuildStrategyIdentity("example", 2, ordered).hash !=
           identity.hash);

    Strategy::StrategyConfiguration duplicate;
    duplicate.entries = {{"same", "first"}, {"same", "second"}};
    assert(ErrorFrom([&] { Strategy::CanonicalConfiguration(duplicate); }) ==
           "duplicate_strategy_configuration_key");
    Strategy::StrategyConfiguration unsupported;
    unsupported.schemaVersion = 2;
    assert(ErrorFrom([&] { Strategy::CanonicalConfiguration(unsupported); }) ==
           "unsupported_strategy_configuration_schema_version");
    assert(ErrorFrom([&] { Strategy::BaselineTerminalStrategy value{ordered}; }) ==
           "baseline_terminal_strategy_unsupported_configuration");
    assert(ErrorFrom([&] {
        Strategy::BaselineTerminalStrategy value{unsupported};
    }) == "baseline_terminal_strategy_unsupported_configuration");

    assert(Strategy::DirectionForPredictedClass(Profitability::kNeutralClass) ==
           Strategy::PositionDirection::flat);
    assert(Strategy::DirectionForPredictedClass(Profitability::kDownClass) ==
           Strategy::PositionDirection::shortPosition);
    assert(Strategy::DirectionForPredictedClass(Profitability::kUpClass) ==
           Strategy::PositionDirection::longPosition);
    assert(ErrorFrom([&] { Strategy::DirectionForPredictedClass(9); }) ==
           "invalid_strategy_predicted_class");

    const Strategy::BaselineTerminalStrategy baseline;
    AssertBaselineParity({});
    AssertBaselineParity({
        {Profitability::kNeutralClass, 100.0f, 150.0f},
        {Profitability::kUpClass,
         std::numeric_limits<float>::quiet_NaN(), 110.0f}});
    AssertBaselineParity({
        {Profitability::kUpClass, 100.0f, 110.0f},
        {Profitability::kUpClass, 100.0f, 90.0f},
        {Profitability::kUpClass, 100.0f, 100.0f}});
    AssertBaselineParity({
        {Profitability::kDownClass, 100.0f, 90.0f},
        {Profitability::kDownClass, 100.0f, 110.0f},
        {Profitability::kDownClass, 100.0f, 100.0f}});
    AssertBaselineParity({
        {Profitability::kNeutralClass, 100.0f, 150.0f},
        {Profitability::kUpClass, 100.0f, 120.0f},
        {Profitability::kDownClass, 100.0f, 120.0f},
        {Profitability::kUpClass, 0.0f, 100.0f},
        {Profitability::kDownClass, 100.0f,
         std::numeric_limits<float>::infinity()},
        {Profitability::kUpClass, 100.0f, 100.0f}});

    Strategy::StrategyEvaluationObservation enriched{
        Profitability::kUpClass, 100.0f, 105.0f};
    enriched.probabilities = Strategy::PredictionProbabilities{
        {0.1f, 0.2f, 0.7f}};
    enriched.decisionTimestampUnixSeconds = 1735689600;
    enriched.terminalTimestampUnixSeconds = 1735693200;
    enriched.marketPath.push_back(
        {1, 1735689900, 100.0f, 101.0f, 99.0f, 100.5f});
    AssertBaselineParity({enriched});

    Strategy::StrategyEvaluationInput invalidClassInput{{
        {9, 100.0f, 110.0f}}};
    assert(ErrorFrom([&] {
        Strategy::EvaluateStrategy(baseline, invalidClassInput);
    }) == "inference_profitability_invalid_predicted_class");

    Strategy::StrategyEvaluationProvenance provenance;
    provenance.inferenceScientificIdentityCanonical =
        "inference_scientific_identity_v1;model=42;configuration=stable;";
    provenance.inferenceScientificIdentityHash =
        Profitability::DeterministicHash(
            provenance.inferenceScientificIdentityCanonical);
    provenance.symbol = "EUR_USD";
    provenance.predictionHorizon = 12;
    provenance.inferenceStart = "2025-01-01";
    provenance.inferenceEnd = "2025-12-31";
    provenance.metricDefinitionHash = Profitability::MetricDefinitionHash();
    provenance.marketDataProvenanceHash =
        Profitability::DeterministicHash("ordered_market_data_v1");

    const auto evaluationIdentity = Strategy::BuildStrategyEvaluationIdentity(
        baseline.Identity(), provenance);
    assert(evaluationIdentity == Strategy::BuildStrategyEvaluationIdentity(
        baseline.Identity(), provenance));
    assert(evaluationIdentity.hash.starts_with("fnv1a64:"));
    Strategy::StrategyEvaluationProvenance changedModel = provenance;
    changedModel.inferenceScientificIdentityCanonical += "changed=true;";
    changedModel.inferenceScientificIdentityHash =
        Profitability::DeterministicHash(
            changedModel.inferenceScientificIdentityCanonical);
    assert(Strategy::BuildStrategyEvaluationIdentity(
               baseline.Identity(), changedModel).hash !=
           evaluationIdentity.hash);
    Strategy::StrategyEvaluationProvenance invalid = provenance;
    invalid.marketDataProvenanceHash.clear();
    assert(ErrorFrom([&] {
        Strategy::BuildStrategyEvaluationIdentity(baseline.Identity(), invalid);
    }) == "invalid_strategy_evaluation_provenance");

    return 0;
}
