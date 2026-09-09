#include "StrategyEvaluation.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::StrategyEvaluation
{
namespace
{

std::string Hex64(std::uint64_t value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        const unsigned shift = static_cast<unsigned>((15 - index) * 4);
        result[index] = digits[(value >> shift) & 0x0fU];
    }
    return result;
}

std::string CanonicalDouble(double value)
{
    return "f64:" + Hex64(std::bit_cast<std::uint64_t>(value));
}

std::string CsvDouble(double value)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

void RequirePairedExecution(
    const AuthoritativeMarketPath& path,
    const StrategyEvaluationResult& baseline,
    const StrategyEvaluationResult& candidate)
{
    if (baseline.sourceContentHash != path.Hash() ||
        candidate.sourceContentHash != path.Hash() ||
        !baseline.evaluationIdentity || !candidate.evaluationIdentity ||
        baseline.executionResults.size() != path.Observations().size() ||
        candidate.executionResults.size() != path.Observations().size())
        throw std::runtime_error("strategy_comparison_path_mismatch");

    for (std::size_t index = 0; index < path.Observations().size(); ++index)
    {
        const auto& observation = path.Observations()[index];
        const auto& left = baseline.executionResults[index];
        const auto& right = candidate.executionResults[index];
        const PositionDirection expected =
            DirectionForPredictedClass(observation.predictedClass);
        const std::optional<std::int64_t> expectedEntryTimestamp =
            expected == PositionDirection::flat
                ? std::nullopt
                : std::optional<std::int64_t>{
                      observation.decisionTimestampUnixSeconds};
        const std::optional<double> expectedEntry =
            expected == PositionDirection::flat
                ? std::nullopt
                : std::optional<double>{
                      static_cast<double>(observation.decisionClose)};
        if (left.observationOrdinal != observation.observationOrdinal ||
            right.observationOrdinal != observation.observationOrdinal ||
            left.direction != expected || right.direction != expected ||
            left.entryTimestampUnixSeconds !=
                right.entryTimestampUnixSeconds ||
            left.entryTimestampUnixSeconds != expectedEntryTimestamp ||
            left.entryPrice != right.entryPrice ||
            left.entryPrice != expectedEntry)
            throw std::runtime_error(
                "strategy_comparison_observation_pairing_mismatch");
        if (expected == PositionDirection::flat)
        {
            if (left.reason != StrategyExitReason::noAction ||
                right.reason != StrategyExitReason::noAction)
                throw std::runtime_error(
                    "strategy_comparison_neutral_action_mismatch");
            continue;
        }
        if (left.exitTimestampUnixSeconds !=
                observation.terminalTimestampUnixSeconds ||
            left.exitPrice != std::optional<double>{
                static_cast<double>(observation.terminalClose)})
            throw std::runtime_error(
                "strategy_comparison_baseline_terminal_mismatch");
    }
}

void ReconcileVariant(ControlledStrategyVariantResult& variant)
{
    for (const auto& execution : variant.evaluation.executionResults)
    {
        switch (execution.reason)
        {
            case StrategyExitReason::noAction:
                ++variant.noActionCount;
                break;
            case StrategyExitReason::fixedStopExit:
                ++variant.fixedStopExitCount;
                break;
            case StrategyExitReason::terminalExit:
                ++variant.terminalExitCount;
                break;
        }
        if (execution.directionalLogReturn &&
            *execution.directionalLogReturn == 0.0)
            ++variant.zeroOutcomeCount;
    }
    const auto& statistics = variant.evaluation.statistics;
    if (variant.noActionCount + statistics.actionableCount !=
            statistics.predictionCount ||
        variant.fixedStopExitCount + variant.terminalExitCount !=
            statistics.actionableCount)
        throw std::runtime_error("strategy_comparison_accounting_mismatch");
}

std::string Comparison(double delta)
{
    if (delta > 0.0) return "better";
    if (delta < 0.0) return "worse";
    return "equal";
}

} // namespace

ControlledFixedStopExperimentResult EvaluateControlledFixedStopExperiment(
    const AuthoritativeMarketPath& marketPath,
    const std::vector<double>& fixedStopLogarithmicDistances)
{
    std::set<std::uint64_t> distanceBits;
    for (const double distance : fixedStopLogarithmicDistances)
    {
        if (!std::isfinite(distance) || distance <= 0.0 ||
            !distanceBits.insert(std::bit_cast<std::uint64_t>(distance)).second)
            throw std::invalid_argument(
                "invalid_or_duplicate_fixed_stop_grid_distance");
    }
    if (fixedStopLogarithmicDistances.empty())
        throw std::invalid_argument("empty_fixed_stop_grid");
    if (fixedStopLogarithmicDistances.size() !=
            kControlledFixedStopDistanceGrid.size() ||
        !std::equal(fixedStopLogarithmicDistances.begin(),
                    fixedStopLogarithmicDistances.end(),
                    kControlledFixedStopDistanceGrid.begin()))
    {
        throw std::invalid_argument("controlled_fixed_stop_grid_mismatch");
    }

    StrategyEvaluationInput input{marketPath};
    ControlledFixedStopExperimentResult result;
    result.comparisonMetricDefinitionCanonical =
        InferenceProfitability::kMetricDefinitionCanonical;
    result.comparisonMetricDefinitionHash =
        InferenceProfitability::MetricDefinitionHash();
    result.experimentIdentityCanonical =
        "controlled_fixed_stop_experiment_v1;market_path_hash=" +
        marketPath.Hash() + ";comparison_metric_hash=" +
        result.comparisonMetricDefinitionHash + ";grid_count=" +
        std::to_string(fixedStopLogarithmicDistances.size()) + ";";
    for (std::size_t index = 0;
         index < fixedStopLogarithmicDistances.size(); ++index)
        result.experimentIdentityCanonical +=
            "grid[" + std::to_string(index) + "]=" +
            CanonicalDouble(fixedStopLogarithmicDistances[index]) + ";";
    result.experimentIdentityHash = InferenceProfitability::DeterministicHash(
        result.experimentIdentityCanonical);
    result.variants.reserve(fixedStopLogarithmicDistances.size() + 1);

    ControlledStrategyVariantResult baseline;
    baseline.evaluation = EvaluateStrategy(BaselineTerminalStrategy{}, input);
    ReconcileVariant(baseline);
    if (baseline.fixedStopExitCount != 0)
        throw std::runtime_error("baseline_fixed_stop_exit_detected");
    baseline.profitabilityComparisonVersusBaseline = "baseline";
    result.variants.push_back(std::move(baseline));

    const auto& baselineResult = result.variants.front().evaluation;
    const double baselineAverage = baselineResult.statistics.
        AverageTerminalHorizonLogReturnPerActionablePrediction().value_or(0.0);
    for (const double distance : fixedStopLogarithmicDistances)
    {
        ControlledStrategyVariantResult variant;
        variant.fixedStopLogarithmicDistance = distance;
        variant.evaluation = EvaluateStrategy(
            FixedStopLossStrategy{{1, distance}}, input);
        RequirePairedExecution(marketPath, baselineResult, variant.evaluation);
        ReconcileVariant(variant);
        variant.aggregateReturnDeltaVersusBaseline =
            variant.evaluation.statistics.
                aggregateTerminalHorizonLogReturnSum -
            baselineResult.statistics.aggregateTerminalHorizonLogReturnSum;
        variant.averageActionableReturnDeltaVersusBaseline =
            variant.evaluation.statistics.
                AverageTerminalHorizonLogReturnPerActionablePrediction().
                    value_or(0.0) - baselineAverage;
        variant.profitabilityComparisonVersusBaseline = Comparison(
            variant.aggregateReturnDeltaVersusBaseline);
        result.variants.push_back(std::move(variant));
    }

    const auto& provenance = marketPath.Provenance();
    std::ostringstream csv;
    csv << "experiment_identity_hash,variant_ordinal,strategy_family,"
           "strategy_identity_hash,fixed_stop_log_distance,model_id,"
           "inference_identity_hash,evaluation_start,evaluation_end,"
           "observation_count,actionable_count,no_action_count,"
           "fixed_stop_exit_count,terminal_exit_count,positive_count,"
           "zero_count,negative_count,aggregate_log_return,"
           "average_actionable_log_return,aggregate_delta_vs_baseline,"
           "average_delta_vs_baseline,comparison_vs_baseline,"
           "market_path_hash,comparison_metric_definition_hash,"
           "strategy_result_metric_definition_hash,result_hash\n";
    for (std::size_t index = 0; index < result.variants.size(); ++index)
    {
        const auto& variant = result.variants[index];
        const auto& evaluation = variant.evaluation;
        const auto& statistics = evaluation.statistics;
        csv << result.experimentIdentityHash << ',' << index << ','
            << evaluation.strategyIdentity.family << ','
            << evaluation.strategyIdentity.hash << ','
            << (variant.fixedStopLogarithmicDistance
                    ? CsvDouble(*variant.fixedStopLogarithmicDistance)
                    : "baseline") << ','
            << provenance.modelId << ','
            << provenance.inferenceScientificIdentityHash << ','
            << provenance.evaluationStart << ',' << provenance.evaluationEnd
            << ',' << statistics.predictionCount << ','
            << statistics.actionableCount << ',' << variant.noActionCount
            << ',' << variant.fixedStopExitCount << ','
            << variant.terminalExitCount << ','
            << statistics.winningActionableCount << ','
            << variant.zeroOutcomeCount << ','
            << statistics.losingActionableCount << ','
            << CsvDouble(statistics.aggregateTerminalHorizonLogReturnSum)
            << ','
            << CsvDouble(statistics.
                   AverageTerminalHorizonLogReturnPerActionablePrediction().
                       value_or(0.0)) << ','
            << CsvDouble(variant.aggregateReturnDeltaVersusBaseline) << ','
            << CsvDouble(variant.averageActionableReturnDeltaVersusBaseline)
            << ',' << variant.profitabilityComparisonVersusBaseline << ','
            << marketPath.Hash() << ','
            << result.comparisonMetricDefinitionHash << ','
            << evaluation.metricDefinitionHash << ','
            << evaluation.resultHash << '\n';
    }
    result.canonicalCsv = csv.str();
    result.resultHash = InferenceProfitability::DeterministicHash(
        result.canonicalCsv);
    return result;
}

} // namespace EA::StrategyEvaluation
