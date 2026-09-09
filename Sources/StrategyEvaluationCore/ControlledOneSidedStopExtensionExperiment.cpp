#include "StrategyEvaluation.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
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

std::string DecimalDouble(double value)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

std::string OptionalDouble(const std::optional<double>& value)
{
    return value ? DecimalDouble(*value) : "NULL";
}

std::string OptionalTimestamp(const std::optional<std::int64_t>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalOrdinal(const std::optional<std::uint64_t>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

const char* DirectionName(PositionDirection direction)
{
    switch (direction)
    {
        case PositionDirection::flat: return "flat";
        case PositionDirection::shortPosition: return "short";
        case PositionDirection::longPosition: return "long";
    }
    return "invalid";
}

double DirectionalMove(PositionDirection direction,
                       double entryPrice,
                       double price)
{
    const double value = std::log(price / entryPrice);
    return direction == PositionDirection::longPosition ? value : -value;
}

struct Excursions
{
    double adverseMagnitude = 0.0;
    double favorable = 0.0;
};

Excursions ComputeExcursions(
    const StrategyEvaluationObservation& observation,
    const StrategyExecutionResult& execution)
{
    if (execution.direction == PositionDirection::flat ||
        !execution.entryPrice || !execution.exitPrice)
        return {};

    double minimum = 0.0;
    double maximum = 0.0;
    for (std::size_t pathOrdinal = 0;
         pathOrdinal < observation.marketPath.size(); ++pathOrdinal)
    {
        if (execution.triggeringPathPointOrdinal == pathOrdinal)
        {
            const double exitMove = DirectionalMove(
                execution.direction, *execution.entryPrice,
                *execution.exitPrice);
            minimum = std::min(minimum, exitMove);
            maximum = std::max(maximum, exitMove);
            break;
        }

        const auto& point = observation.marketPath[pathOrdinal];
        const double adversePrice =
            execution.direction == PositionDirection::longPosition
                ? static_cast<double>(point.low)
                : static_cast<double>(point.high);
        const double favorablePrice =
            execution.direction == PositionDirection::longPosition
                ? static_cast<double>(point.high)
                : static_cast<double>(point.low);
        minimum = std::min(minimum, DirectionalMove(
            execution.direction, *execution.entryPrice, adversePrice));
        maximum = std::max(maximum, DirectionalMove(
            execution.direction, *execution.entryPrice, favorablePrice));
    }
    return {-minimum, maximum};
}

std::size_t ConfidenceBucketIndex(double confidence)
{
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0)
        throw std::runtime_error("phase18b_invalid_normalized_confidence");
    if (confidence < 0.25) return 0;
    if (confidence < 0.50) return 1;
    if (confidence < 0.75) return 2;
    return 3;
}

void RequireFourWayPairing(
    const AuthoritativeMarketPath& path,
    const std::vector<Phase18BStrategyVariantResult>& variants,
    const ProbabilityConditionedStopExtensionStrategy& extension)
{
    if (variants.size() != 4)
        throw std::runtime_error("phase18b_strategy_count_mismatch");
    for (const auto& variant : variants)
    {
        if (variant.evaluation.sourceContentHash != path.Hash() ||
            !variant.evaluation.evaluationIdentity ||
            variant.evaluation.executionResults.size() !=
                path.Observations().size())
            throw std::runtime_error("phase18b_strategy_path_mismatch");
    }

    for (std::size_t index = 0; index < path.Observations().size(); ++index)
    {
        const auto& observation = path.Observations()[index];
        if (!observation.probabilities)
            throw std::runtime_error(
                "phase18b_observation_probabilities_missing");
        const PositionDirection expected =
            DirectionForPredictedClass(observation.predictedClass);
        const std::optional<std::int64_t> expectedTimestamp =
            expected == PositionDirection::flat
                ? std::nullopt : observation.decisionTimestampUnixSeconds;
        const std::optional<double> expectedPrice =
            expected == PositionDirection::flat
                ? std::nullopt
                : std::optional<double>{observation.decisionClose};
        for (const auto& variant : variants)
        {
            const auto& execution =
                variant.evaluation.executionResults[index];
            if (execution.observationOrdinal !=
                    observation.observationOrdinal ||
                execution.direction != expected ||
                execution.entryTimestampUnixSeconds != expectedTimestamp ||
                execution.entryPrice != expectedPrice)
            {
                throw std::runtime_error(
                    "phase18b_observation_pairing_mismatch");
            }
            if (expected == PositionDirection::flat &&
                (execution.reason != StrategyExitReason::noAction ||
                 execution.initialStopPrice || execution.exitPrice ||
                 execution.directionalLogReturn))
            {
                throw std::runtime_error(
                    "phase18b_neutral_action_pairing_mismatch");
            }
        }

        const auto& baseline = variants[0].evaluation.executionResults[index];
        if (expected != PositionDirection::flat &&
            (baseline.reason != StrategyExitReason::terminalExit ||
             baseline.exitTimestampUnixSeconds !=
                observation.terminalTimestampUnixSeconds ||
             baseline.exitPrice != std::optional<double>{
                observation.terminalClose}))
            throw std::runtime_error("phase18b_baseline_pairing_mismatch");

        if (expected != PositionDirection::flat)
        {
            const auto decision = extension.StopDecision(observation);
            if (decision.stopMultiplier < 1.0)
                throw std::runtime_error("phase18b_stop_floor_violated");
            if (decision.normalizedDirectionalConfidence <=
                    extension.Configuration().activationConfidence &&
                variants[1].evaluation.executionResults[index] !=
                    variants[3].evaluation.executionResults[index])
            {
                throw std::runtime_error(
                    "phase18b_fixed_floor_execution_mismatch");
            }
        }
    }
}

Phase18AStrategyMetrics BuildMetrics(
    const Phase18BStrategyVariantResult& variant,
    const std::vector<Phase18AObservationEvidence>& evidence)
{
    Phase18AStrategyMetrics metrics;
    const auto& statistics = variant.evaluation.statistics;
    metrics.observationCount = statistics.predictionCount;
    metrics.actionableCount = statistics.actionableCount;
    metrics.nonActionableCount = metrics.observationCount -
        metrics.actionableCount;
    metrics.winningCount = statistics.winningActionableCount;
    metrics.losingCount = statistics.losingActionableCount;
    metrics.aggregateDirectionalLogReturn =
        statistics.aggregateTerminalHorizonLogReturnSum;
    if (metrics.actionableCount != 0)
    {
        const double denominator =
            static_cast<double>(metrics.actionableCount);
        metrics.averageDirectionalLogReturn =
            metrics.aggregateDirectionalLogReturn / denominator;
        metrics.winningRate =
            static_cast<double>(metrics.winningCount) / denominator;
        metrics.losingRate =
            static_cast<double>(metrics.losingCount) / denominator;
    }

    std::vector<double> holdingDurations;
    double stopDistanceSum = 0.0;
    double stopMultiplierSum = 0.0;
    std::uint64_t stopEvidenceCount = 0;
    double equity = 0.0;
    double peak = 0.0;
    for (const auto& item : evidence)
    {
        if (item.strategyOutputIdentity != variant.strategyOutputIdentity ||
            !item.actionable)
            continue;
        if (item.execution.reason == StrategyExitReason::fixedStopExit)
            ++metrics.stopHitCount;
        else if (item.execution.reason == StrategyExitReason::terminalExit)
            ++metrics.terminalExitCount;
        if (item.execution.directionalLogReturn &&
            *item.execution.directionalLogReturn == 0.0)
            ++metrics.zeroOutcomeCount;
        if (item.holdingDurationSeconds)
            holdingDurations.push_back(*item.holdingDurationSeconds);
        if (item.maximumAdverseExcursion)
        {
            metrics.maximumAdverseExcursion = std::max(
                metrics.maximumAdverseExcursion.value_or(0.0),
                *item.maximumAdverseExcursion);
        }
        if (item.maximumFavorableExcursion)
        {
            metrics.maximumFavorableExcursion = std::max(
                metrics.maximumFavorableExcursion.value_or(0.0),
                *item.maximumFavorableExcursion);
        }
        if (item.effectiveStopLogarithmicDistance &&
            item.appliedStopMultiplier)
        {
            stopDistanceSum += *item.effectiveStopLogarithmicDistance;
            stopMultiplierSum += *item.appliedStopMultiplier;
            ++stopEvidenceCount;
        }
        equity += *item.execution.directionalLogReturn;
        peak = std::max(peak, equity);
        metrics.maximumDrawdown = std::max(
            metrics.maximumDrawdown, peak - equity);
    }
    if (metrics.actionableCount != 0)
    {
        const double denominator =
            static_cast<double>(metrics.actionableCount);
        metrics.stopHitRate =
            static_cast<double>(metrics.stopHitCount) / denominator;
        metrics.zeroOutcomeRate =
            static_cast<double>(metrics.zeroOutcomeCount) / denominator;
        metrics.terminalExitRate =
            static_cast<double>(metrics.terminalExitCount) / denominator;
    }
    if (!holdingDurations.empty())
    {
        metrics.averageHoldingDurationSeconds = std::accumulate(
            holdingDurations.begin(), holdingDurations.end(), 0.0) /
            static_cast<double>(holdingDurations.size());
        std::sort(holdingDurations.begin(), holdingDurations.end());
        const std::size_t middle = holdingDurations.size() / 2;
        metrics.medianHoldingDurationSeconds =
            holdingDurations.size() % 2 != 0
                ? holdingDurations[middle]
                : (holdingDurations[middle - 1] +
                   holdingDurations[middle]) / 2.0;
    }
    if (stopEvidenceCount != 0)
    {
        const double denominator =
            static_cast<double>(stopEvidenceCount);
        metrics.averageEffectiveStopLogarithmicDistance =
            stopDistanceSum / denominator;
        metrics.averageStopMultiplier = stopMultiplierSum / denominator;
    }
    if (metrics.nonActionableCount + metrics.actionableCount !=
            metrics.observationCount ||
        metrics.stopHitCount + metrics.terminalExitCount !=
            metrics.actionableCount ||
        metrics.winningCount + metrics.losingCount +
            metrics.zeroOutcomeCount != metrics.actionableCount)
        throw std::runtime_error("phase18b_strategy_accounting_mismatch");
    return metrics;
}

std::array<Phase18AConfidenceBucketMetrics, 4> BuildBuckets(
    const Phase18BStrategyVariantResult& variant,
    const std::vector<Phase18AObservationEvidence>& evidence)
{
    std::array<Phase18AConfidenceBucketMetrics, 4> buckets{{
        {0.00, 0.25, false, 0, 0, 0, 0, 0, 0, 0.0, std::nullopt},
        {0.25, 0.50, false, 0, 0, 0, 0, 0, 0, 0.0, std::nullopt},
        {0.50, 0.75, false, 0, 0, 0, 0, 0, 0, 0.0, std::nullopt},
        {0.75, 1.00, true, 0, 0, 0, 0, 0, 0, 0.0, std::nullopt}}};
    for (const auto& item : evidence)
    {
        if (item.strategyOutputIdentity != variant.strategyOutputIdentity ||
            !item.actionable || !item.normalizedDirectionalConfidence ||
            !item.execution.directionalLogReturn)
            continue;
        auto& bucket = buckets[ConfidenceBucketIndex(
            *item.normalizedDirectionalConfidence)];
        ++bucket.actionableCount;
        if (*item.execution.directionalLogReturn > 0.0)
            ++bucket.winningCount;
        else if (*item.execution.directionalLogReturn < 0.0)
            ++bucket.losingCount;
        else
            ++bucket.zeroOutcomeCount;
        if (item.execution.reason == StrategyExitReason::fixedStopExit)
            ++bucket.stopHitCount;
        else
            ++bucket.terminalExitCount;
        bucket.aggregateDirectionalLogReturn +=
            *item.execution.directionalLogReturn;
    }
    for (auto& bucket : buckets)
    {
        if (bucket.actionableCount != 0)
            bucket.averageDirectionalLogReturn =
                bucket.aggregateDirectionalLogReturn /
                static_cast<double>(bucket.actionableCount);
    }
    return buckets;
}

Phase18BMechanismMetrics BuildMechanismMetrics(
    const Phase18BStrategyVariantResult& variant,
    const std::vector<Phase18AObservationEvidence>& evidence)
{
    Phase18BMechanismMetrics metrics;
    double extendedMultiplierSum = 0.0;
    for (const auto& item : evidence)
    {
        if (item.strategyOutputIdentity != variant.strategyOutputIdentity ||
            !item.actionable)
            continue;
        if (!item.appliedStopMultiplier || *item.appliedStopMultiplier < 1.0)
            throw std::runtime_error("phase18b_stop_floor_evidence_mismatch");
        if (*item.appliedStopMultiplier == 1.0)
            ++metrics.fixedFloorMultiplierCount;
        else
        {
            ++metrics.extendedStopMultiplierCount;
            extendedMultiplierSum += *item.appliedStopMultiplier;
        }
    }
    if (metrics.fixedFloorMultiplierCount +
            metrics.extendedStopMultiplierCount !=
        variant.metrics.actionableCount)
        throw std::runtime_error("phase18b_mechanism_accounting_mismatch");
    if (variant.metrics.actionableCount != 0)
    {
        const double denominator =
            static_cast<double>(variant.metrics.actionableCount);
        metrics.fixedFloorMultiplierRate =
            static_cast<double>(metrics.fixedFloorMultiplierCount) /
            denominator;
        metrics.extendedStopMultiplierRate =
            static_cast<double>(metrics.extendedStopMultiplierCount) /
            denominator;
    }
    if (metrics.extendedStopMultiplierCount != 0)
    {
        metrics.averageExtendedStopMultiplier =
            extendedMultiplierSum /
            static_cast<double>(metrics.extendedStopMultiplierCount);
    }
    return metrics;
}

double Value(const std::optional<double>& value)
{
    return value.value_or(0.0);
}

Phase18APairwiseDelta Pairwise(
    const Phase18BStrategyVariantResult& candidate,
    const Phase18BStrategyVariantResult& reference)
{
    const auto& left = candidate.metrics;
    const auto& right = reference.metrics;
    if (left.observationCount != right.observationCount ||
        left.actionableCount != right.actionableCount ||
        left.nonActionableCount != right.nonActionableCount)
        throw std::runtime_error("phase18b_pairwise_population_mismatch");
    return {
        candidate.strategyOutputIdentity,
        reference.strategyOutputIdentity,
        left.aggregateDirectionalLogReturn -
            right.aggregateDirectionalLogReturn,
        Value(left.averageDirectionalLogReturn) -
            Value(right.averageDirectionalLogReturn),
        Value(left.winningRate) - Value(right.winningRate),
        Value(left.losingRate) - Value(right.losingRate),
        Value(left.stopHitRate) - Value(right.stopHitRate),
        Value(left.terminalExitRate) - Value(right.terminalExitRate),
        Value(left.averageHoldingDurationSeconds) -
            Value(right.averageHoldingDurationSeconds),
        left.maximumDrawdown - right.maximumDrawdown};
}

void AppendStrategyLine(
    std::ostringstream& output,
    const ControlledOneSidedStopExtensionExperimentResult& result,
    const Phase18BStrategyVariantResult& variant)
{
    const auto& metrics = variant.metrics;
    output << "PHASE18B_STRATEGY_RESULT"
           << ",experiment_identity_hash=" << result.experimentIdentityHash
           << ",strategy=" << variant.strategyOutputIdentity
           << ",strategy_family=" << variant.evaluation.strategyIdentity.family
           << ",strategy_version=" << variant.evaluation.strategyIdentity.version
           << ",strategy_identity_hash=" << variant.evaluation.strategyIdentity.hash
           << ",strategy_configuration_hash="
           << variant.evaluation.strategyIdentity.configurationHash
           << ",observation_count=" << metrics.observationCount
           << ",actionable_count=" << metrics.actionableCount
           << ",non_actionable_count=" << metrics.nonActionableCount
           << ",aggregate_directional_log_return="
           << DecimalDouble(metrics.aggregateDirectionalLogReturn)
           << ",average_actionable_log_return="
           << OptionalDouble(metrics.averageDirectionalLogReturn)
           << ",winning_count=" << metrics.winningCount
           << ",winning_rate=" << OptionalDouble(metrics.winningRate)
           << ",losing_count=" << metrics.losingCount
           << ",losing_rate=" << OptionalDouble(metrics.losingRate)
           << ",zero_count=" << metrics.zeroOutcomeCount
           << ",zero_rate=" << OptionalDouble(metrics.zeroOutcomeRate)
           << ",stop_hit_count=" << metrics.stopHitCount
           << ",stop_hit_rate=" << OptionalDouble(metrics.stopHitRate)
           << ",terminal_exit_count=" << metrics.terminalExitCount
           << ",terminal_exit_rate=" << OptionalDouble(metrics.terminalExitRate)
           << ",average_holding_seconds="
           << OptionalDouble(metrics.averageHoldingDurationSeconds)
           << ",median_holding_seconds="
           << OptionalDouble(metrics.medianHoldingDurationSeconds)
           << ",maximum_adverse_excursion="
           << OptionalDouble(metrics.maximumAdverseExcursion)
           << ",maximum_favorable_excursion="
           << OptionalDouble(metrics.maximumFavorableExcursion)
           << ",maximum_drawdown=" << DecimalDouble(metrics.maximumDrawdown)
           << ",average_effective_stop_log_distance="
           << OptionalDouble(metrics.averageEffectiveStopLogarithmicDistance)
           << ",average_stop_multiplier="
           << OptionalDouble(metrics.averageStopMultiplier)
           << ",result_hash=" << variant.evaluation.resultHash << '\n';
}

} // namespace

void ValidateControlledOneSidedStopExtensionInvocation(
    const ControlledOneSidedStopExtensionInvocationContext& context)
{
    if (!context.inferenceMode || !context.hasExplicitModel ||
        context.inferAll || context.schedulerExperiment ||
        context.schedulerCheckpointEvaluation || context.schedulerWorkerAttempt ||
        context.frozenOutcome)
    {
        throw std::invalid_argument(
            "--probability-conditioned-stop-extension-evaluation requires "
            "standalone --infer with one explicit --model and no scheduler context");
    }
}

ControlledOneSidedStopExtensionExperimentResult
EvaluateControlledOneSidedStopExtensionExperiment(
    const AuthoritativeMarketPath& marketPath,
    ProbabilityConditionedStopExtensionConfiguration configuration)
{
    const ProbabilityConditionedStopLossConfiguration phase18AConfiguration{
        kProbabilityConditionedStopLossConfigurationSchemaVersion,
        configuration.baseStopLogarithmicDistance,
        kPhase18ADefaultMinimumStopMultiplier,
        kPhase18ADefaultMaximumStopMultiplier};
    const ProbabilityConditionedStopLossStrategy symmetric{
        phase18AConfiguration};
    const ProbabilityConditionedStopExtensionStrategy extension{
        configuration};
    const FixedStopLossStrategy fixed{{
        kFixedStopLossConfigurationSchemaVersion,
        configuration.baseStopLogarithmicDistance}};
    const BaselineTerminalStrategy baseline;
    StrategyEvaluationInput input{marketPath};

    ControlledOneSidedStopExtensionExperimentResult result;
    result.phase18AConfiguration = phase18AConfiguration;
    result.phase18BConfiguration = configuration;
    result.experimentIdentityCanonical =
        "controlled_one_sided_stop_extension_experiment_v1;"
        "phase18a_mapping_identity=" +
        std::string{kDirectionalProbabilityMappingIdentity} +
        ";phase18b_mapping_identity=" +
        std::string{kOneSidedStopExtensionMappingIdentity} +
        ";market_path_hash=" + marketPath.Hash() +
        ";base_stop_logarithmic_distance=" +
        CanonicalDouble(configuration.baseStopLogarithmicDistance) +
        ";activation_confidence=" +
        CanonicalDouble(configuration.activationConfidence) +
        ";minimum_stop_multiplier=" +
        CanonicalDouble(configuration.minimumStopMultiplier) +
        ";maximum_stop_multiplier=" +
        CanonicalDouble(configuration.maximumStopMultiplier) +
        ";baseline_strategy_hash=" + baseline.Identity().hash +
        ";fixed_strategy_hash=" + fixed.Identity().hash +
        ";phase18a_strategy_hash=" + symmetric.Identity().hash +
        ";phase18b_strategy_hash=" + extension.Identity().hash + ";";
    result.experimentIdentityHash = InferenceProfitability::DeterministicHash(
        result.experimentIdentityCanonical);

    result.variants = {
        {kBaselineTerminalOutputIdentity,
         EvaluateStrategy(baseline, input), {}, {}, {}},
        {kFixedStopLossOutputIdentity,
         EvaluateStrategy(fixed, input), {}, {}, {}},
        {kProbabilityConditionedStopLossOutputIdentity,
         EvaluateStrategy(symmetric, input), {}, {}, {}},
        {kProbabilityConditionedStopExtensionOutputIdentity,
         EvaluateStrategy(extension, input), {}, {}, {}}};
    RequireFourWayPairing(marketPath, result.variants, extension);

    result.observationEvidence.reserve(
        marketPath.Observations().size() * result.variants.size());
    for (const auto& variant : result.variants)
    {
        for (std::size_t index = 0;
             index < marketPath.Observations().size(); ++index)
        {
            const auto& observation = marketPath.Observations()[index];
            const auto& execution = variant.evaluation.executionResults[index];
            Phase18AObservationEvidence item;
            item.strategyOutputIdentity = variant.strategyOutputIdentity;
            item.observationOrdinal = observation.observationOrdinal;
            item.predictedClass = observation.predictedClass;
            item.direction = execution.direction;
            item.actionable = execution.direction != PositionDirection::flat;
            item.decisionTimestampUnixSeconds =
                *observation.decisionTimestampUnixSeconds;
            item.decisionPrice = observation.decisionClose;
            item.probabilities = *observation.probabilities;
            item.execution = execution;
            if (item.actionable)
            {
                const auto symmetricDecision =
                    symmetric.StopDecision(observation);
                const auto extensionDecision =
                    extension.StopDecision(observation);
                item.directionalProbability =
                    extensionDecision.directionalProbability;
                item.normalizedDirectionalConfidence =
                    extensionDecision.normalizedDirectionalConfidence;
                if (variant.strategyOutputIdentity ==
                    kFixedStopLossOutputIdentity)
                {
                    item.baseStopLogarithmicDistance =
                        configuration.baseStopLogarithmicDistance;
                    item.appliedStopMultiplier = 1.0;
                    item.effectiveStopLogarithmicDistance =
                        configuration.baseStopLogarithmicDistance;
                }
                else if (variant.strategyOutputIdentity ==
                         kProbabilityConditionedStopLossOutputIdentity)
                {
                    item.baseStopLogarithmicDistance =
                        configuration.baseStopLogarithmicDistance;
                    item.appliedStopMultiplier =
                        symmetricDecision.stopMultiplier;
                    item.effectiveStopLogarithmicDistance =
                        symmetricDecision.effectiveStopLogarithmicDistance;
                }
                else if (variant.strategyOutputIdentity ==
                         kProbabilityConditionedStopExtensionOutputIdentity)
                {
                    item.baseStopLogarithmicDistance =
                        configuration.baseStopLogarithmicDistance;
                    item.appliedStopMultiplier =
                        extensionDecision.stopMultiplier;
                    item.effectiveStopLogarithmicDistance =
                        extensionDecision.effectiveStopLogarithmicDistance;
                }
                if (!execution.entryTimestampUnixSeconds ||
                    !execution.exitTimestampUnixSeconds ||
                    *execution.exitTimestampUnixSeconds <
                        *execution.entryTimestampUnixSeconds)
                    throw std::runtime_error(
                        "phase18b_invalid_holding_duration");
                item.holdingDurationSeconds = static_cast<double>(
                    *execution.exitTimestampUnixSeconds -
                    *execution.entryTimestampUnixSeconds);
                const Excursions excursions = ComputeExcursions(
                    observation, execution);
                item.maximumAdverseExcursion = excursions.adverseMagnitude;
                item.maximumFavorableExcursion = excursions.favorable;
            }
            result.observationEvidence.push_back(std::move(item));
        }
    }

    for (auto& variant : result.variants)
    {
        variant.metrics = BuildMetrics(variant, result.observationEvidence);
        variant.confidenceBuckets = BuildBuckets(
            variant, result.observationEvidence);
    }
    result.variants[3].mechanismMetrics = BuildMechanismMetrics(
        result.variants[3], result.observationEvidence);

    for (std::size_t index = 0;
         index < result.extensionDeltasVersusFixedByConfidenceBucket.size();
         ++index)
    {
        const auto& candidate = result.variants[3].confidenceBuckets[index];
        const auto& reference = result.variants[1].confidenceBuckets[index];
        if (candidate.actionableCount != reference.actionableCount)
            throw std::runtime_error(
                "phase18b_bucket_population_mismatch");
        auto& delta =
            result.extensionDeltasVersusFixedByConfidenceBucket[index];
        delta.actionableCount = candidate.actionableCount;
        delta.stopHitCountDifferenceVersusFixed =
            static_cast<std::int64_t>(candidate.stopHitCount) -
            static_cast<std::int64_t>(reference.stopHitCount);
        if (candidate.actionableCount != 0)
        {
            delta.stopHitRateDifferenceVersusFixed =
                static_cast<double>(delta.stopHitCountDifferenceVersusFixed) /
                static_cast<double>(candidate.actionableCount);
        }
        delta.aggregateDirectionalLogReturnDifferenceVersusFixed =
            candidate.aggregateDirectionalLogReturn -
            reference.aggregateDirectionalLogReturn;
    }

    result.pairwiseDeltas = {
        {"PRIMARY", Pairwise(result.variants[3], result.variants[1])},
        {"secondary", Pairwise(result.variants[3], result.variants[0])},
        {"diagnostic", Pairwise(result.variants[3], result.variants[2])},
        {"phase18a_retained", Pairwise(result.variants[2], result.variants[1])}};

    std::ostringstream lines;
    lines << "PHASE18B_EXPERIMENT"
          << ",experiment_identity_hash=" << result.experimentIdentityHash
          << ",experiment_version=1"
          << ",phase18a_mapping_identity="
          << kDirectionalProbabilityMappingIdentity
          << ",phase18b_mapping_identity="
          << kOneSidedStopExtensionMappingIdentity
          << ",market_path_hash=" << marketPath.Hash()
          << ",model_id=" << marketPath.Provenance().modelId
          << ",prediction_horizon="
          << marketPath.Provenance().predictionHorizon
          << ",base_stop_log_distance="
          << DecimalDouble(configuration.baseStopLogarithmicDistance)
          << ",activation_confidence="
          << DecimalDouble(configuration.activationConfidence)
          << ",minimum_stop_multiplier="
          << DecimalDouble(configuration.minimumStopMultiplier)
          << ",maximum_stop_multiplier="
          << DecimalDouble(configuration.maximumStopMultiplier)
          << ",requires_read_only_transaction=true"
          << ",production_rows_modified=false\n";
    for (const auto& variant : result.variants)
    {
        AppendStrategyLine(lines, result, variant);
        for (std::size_t index = 0;
             index < variant.confidenceBuckets.size(); ++index)
        {
            const auto& bucket = variant.confidenceBuckets[index];
            lines << "PHASE18B_CONFIDENCE_BUCKET"
                  << ",experiment_identity_hash="
                  << result.experimentIdentityHash
                  << ",strategy=" << variant.strategyOutputIdentity
                  << ",bucket_ordinal=" << index
                  << ",lower_inclusive="
                  << DecimalDouble(bucket.lowerInclusive)
                  << ",upper=" << DecimalDouble(bucket.upperExclusive)
                  << ",includes_upper="
                  << (bucket.includesUpperBound ? 1 : 0)
                  << ",actionable_count=" << bucket.actionableCount
                  << ",winning_count=" << bucket.winningCount
                  << ",losing_count=" << bucket.losingCount
                  << ",zero_count=" << bucket.zeroOutcomeCount
                  << ",stop_hit_count=" << bucket.stopHitCount
                  << ",terminal_exit_count=" << bucket.terminalExitCount
                  << ",aggregate_directional_log_return="
                  << DecimalDouble(bucket.aggregateDirectionalLogReturn)
                  << ",average_directional_log_return="
                  << OptionalDouble(bucket.averageDirectionalLogReturn);
            if (variant.strategyOutputIdentity ==
                kProbabilityConditionedStopExtensionOutputIdentity)
            {
                const auto& delta =
                    result.extensionDeltasVersusFixedByConfidenceBucket[index];
                lines << ",stop_hit_count_difference_vs_fixed="
                      << delta.stopHitCountDifferenceVersusFixed
                      << ",stop_hit_rate_difference_vs_fixed="
                      << DecimalDouble(
                          delta.stopHitRateDifferenceVersusFixed)
                      << ",aggregate_return_difference_vs_fixed="
                      << DecimalDouble(
                          delta.aggregateDirectionalLogReturnDifferenceVersusFixed);
            }
            lines << '\n';
        }
    }

    const auto& mechanism = result.variants[3].mechanismMetrics;
    lines << "PHASE18B_EXTENSION_MECHANISM"
          << ",experiment_identity_hash=" << result.experimentIdentityHash
          << ",strategy="
          << kProbabilityConditionedStopExtensionOutputIdentity
          << ",multiplier_exactly_one_count="
          << mechanism.fixedFloorMultiplierCount
          << ",multiplier_exactly_one_rate="
          << OptionalDouble(mechanism.fixedFloorMultiplierRate)
          << ",multiplier_greater_than_one_count="
          << mechanism.extendedStopMultiplierCount
          << ",multiplier_greater_than_one_rate="
          << OptionalDouble(mechanism.extendedStopMultiplierRate)
          << ",average_multiplier_among_extended="
          << OptionalDouble(mechanism.averageExtendedStopMultiplier) << '\n';

    for (const auto& pairwise : result.pairwiseDeltas)
    {
        const auto& delta = pairwise.values;
        lines << "PHASE18B_PAIRWISE_DELTA"
              << ",experiment_identity_hash="
              << result.experimentIdentityHash
              << ",comparison_role=" << pairwise.comparisonRole
              << ",candidate=" << delta.candidateStrategyOutputIdentity
              << ",reference=" << delta.referenceStrategyOutputIdentity
              << ",aggregate_directional_log_return_delta="
              << DecimalDouble(delta.aggregateDirectionalLogReturnDelta)
              << ",average_directional_log_return_delta="
              << DecimalDouble(delta.averageDirectionalLogReturnDelta)
              << ",winning_rate_delta="
              << DecimalDouble(delta.winningRateDelta)
              << ",losing_rate_delta="
              << DecimalDouble(delta.losingRateDelta)
              << ",stop_hit_rate_delta="
              << DecimalDouble(delta.stopHitRateDelta)
              << ",terminal_exit_rate_delta="
              << DecimalDouble(delta.terminalExitRateDelta)
              << ",average_holding_seconds_delta="
              << DecimalDouble(delta.averageHoldingDurationSecondsDelta)
              << ",maximum_drawdown_delta="
              << DecimalDouble(delta.maximumDrawdownDelta) << '\n';
    }
    for (const auto& item : result.observationEvidence)
    {
        lines << "PHASE18B_OBSERVATION_EVIDENCE"
              << ",experiment_identity_hash="
              << result.experimentIdentityHash
              << ",strategy=" << item.strategyOutputIdentity
              << ",observation_ordinal=" << item.observationOrdinal
              << ",predicted_class=" << item.predictedClass
              << ",direction=" << DirectionName(item.direction)
              << ",actionable=" << (item.actionable ? 1 : 0)
              << ",decision_timestamp_unix_seconds="
              << item.decisionTimestampUnixSeconds
              << ",decision_price=" << DecimalDouble(item.decisionPrice)
              << ",probability_down="
              << DecimalDouble(item.probabilities.downNeutralUp[0])
              << ",probability_neutral="
              << DecimalDouble(item.probabilities.downNeutralUp[1])
              << ",probability_up="
              << DecimalDouble(item.probabilities.downNeutralUp[2])
              << ",directional_probability="
              << OptionalDouble(item.directionalProbability)
              << ",normalized_directional_confidence="
              << OptionalDouble(item.normalizedDirectionalConfidence)
              << ",base_stop_log_distance="
              << OptionalDouble(item.baseStopLogarithmicDistance)
              << ",applied_stop_multiplier="
              << OptionalDouble(item.appliedStopMultiplier)
              << ",effective_stop_log_distance="
              << OptionalDouble(item.effectiveStopLogarithmicDistance)
              << ",entry_timestamp_unix_seconds="
              << OptionalTimestamp(item.execution.entryTimestampUnixSeconds)
              << ",entry_price=" << OptionalDouble(item.execution.entryPrice)
              << ",stop_price="
              << OptionalDouble(item.execution.initialStopPrice)
              << ",exit_timestamp_unix_seconds="
              << OptionalTimestamp(item.execution.exitTimestampUnixSeconds)
              << ",exit_price=" << OptionalDouble(item.execution.exitPrice)
              << ",exit_reason="
              << StrategyExitReasonText(item.execution.reason)
              << ",directional_log_return="
              << OptionalDouble(item.execution.directionalLogReturn)
              << ",triggering_path_point_ordinal="
              << OptionalOrdinal(item.execution.triggeringPathPointOrdinal)
              << ",triggering_source_row="
              << OptionalOrdinal(item.execution.triggeringSourceRow)
              << ",holding_duration_seconds="
              << OptionalDouble(item.holdingDurationSeconds)
              << ",maximum_adverse_excursion="
              << OptionalDouble(item.maximumAdverseExcursion)
              << ",maximum_favorable_excursion="
              << OptionalDouble(item.maximumFavorableExcursion) << '\n';
    }
    result.canonicalLines = lines.str();
    result.resultHash = InferenceProfitability::DeterministicHash(
        result.canonicalLines);
    return result;
}

} // namespace EA::StrategyEvaluation
