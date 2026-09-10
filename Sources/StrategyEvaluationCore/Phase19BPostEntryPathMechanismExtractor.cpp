#include "Phase19BPostEntryPathMechanismExtractor.hpp"

#include "../InferenceProfitability.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::StrategyEvaluation
{
namespace
{

struct FrozenModel
{
    long long modelId;
    std::string_view symbol;
    std::uint64_t predictionHorizon;
    std::string_view cohortRole;
};

constexpr std::array<FrozenModel, 7> kFrozenModels{{
    {1745, "AUDCHF", 4, "primary"},
    {1743, "EURCHF", 12, "primary"},
    {1729, "USDCAD", 6, "primary"},
    {1735, "GBPUSD", 6, "primary"},
    {1662, "GBPJPY", 8, "primary"},
    {1692, "AUDCAD", 14, "primary"},
    {1805, "EURUSD", 4, "diagnostic"},
}};

std::string Decimal(double value)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

template <typename T>
std::string OptionalInteger(const std::optional<T>& value)
{
    return value ? std::to_string(*value) : std::string{};
}

std::string OptionalDecimal(const std::optional<double>& value)
{
    return value ? Decimal(*value) : std::string{};
}

std::string DirectionText(PositionDirection value)
{
    switch (value)
    {
        case PositionDirection::flat: return "flat";
        case PositionDirection::shortPosition: return "short";
        case PositionDirection::longPosition: return "long";
    }
    throw std::runtime_error("phase19b_invalid_direction");
}

std::string NormalizeSymbol(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) {
                       if (character >= 'a' && character <= 'z')
                           return static_cast<char>(character - 'a' + 'A');
                       return static_cast<char>(character);
                   });
    if (value.ends_with("RMP")) value.resize(value.size() - 3);
    return value;
}

const FrozenModel& RequireFrozenIdentity(
    const MarketPathProvenance& provenance)
{
    const auto model = std::find_if(
        kFrozenModels.begin(), kFrozenModels.end(),
        [&provenance](const FrozenModel& item) {
            return item.modelId == provenance.modelId;
        });
    if (model == kFrozenModels.end())
        throw std::invalid_argument("phase19b_model_not_in_frozen_cohort");
    if (NormalizeSymbol(provenance.symbol) != model->symbol)
        throw std::invalid_argument("phase19b_frozen_model_symbol_mismatch");
    if (provenance.predictionHorizon != model->predictionHorizon)
        throw std::invalid_argument("phase19b_frozen_model_horizon_mismatch");
    const bool frozenWindow =
        (provenance.evaluationStart == "2025-01-01" &&
         provenance.evaluationEnd == "2026-01-01") ||
        (provenance.evaluationStart == "2026-01-01" &&
         provenance.evaluationEnd == "2026-09-01");
    if (!frozenWindow)
        throw std::invalid_argument("phase19b_window_not_frozen");
    return *model;
}

double DirectionalMove(PositionDirection direction,
                       double entryPrice,
                       double price)
{
    const double raw = std::log(price / entryPrice);
    return direction == PositionDirection::longPosition ? raw : -raw;
}

double DirectionalAdverseExtreme(
    PositionDirection direction,
    double entryPrice,
    const MarketPathPoint& point)
{
    return DirectionalMove(
        direction, entryPrice,
        direction == PositionDirection::longPosition
            ? static_cast<double>(point.low)
            : static_cast<double>(point.high));
}

double DirectionalFavorableExtreme(
    PositionDirection direction,
    double entryPrice,
    const MarketPathPoint& point)
{
    return DirectionalMove(
        direction, entryPrice,
        direction == PositionDirection::longPosition
            ? static_cast<double>(point.high)
            : static_cast<double>(point.low));
}

double HorizonFraction(std::uint64_t pathPointOrdinal,
                       std::uint64_t horizon)
{
    return static_cast<double>(pathPointOrdinal + 1) /
        static_cast<double>(horizon);
}

std::string ObservationIdentity(
    const AuthoritativeMarketPath& marketPath,
    const StrategyEvaluationObservation& observation)
{
    std::ostringstream canonical;
    canonical << "phase19b_observation_identity_v1;market_path_hash="
              << marketPath.Hash() << ";model_id="
              << marketPath.Provenance().modelId
              << ";observation_ordinal=" << observation.observationOrdinal
              << ";decision_row=" << observation.decisionRow
              << ";decision_timestamp_unix_seconds="
              << *observation.decisionTimestampUnixSeconds << ';';
    return InferenceProfitability::DeterministicHash(canonical.str());
}

const Phase18BStrategyVariantResult& RequireVariant(
    const ControlledOneSidedStopExtensionExperimentResult& experiment,
    std::string_view identity)
{
    const auto result = std::find_if(
        experiment.variants.begin(), experiment.variants.end(),
        [identity](const Phase18BStrategyVariantResult& variant) {
            return variant.strategyOutputIdentity == identity;
        });
    if (result == experiment.variants.end())
        throw std::runtime_error("phase19b_phase18b_variant_missing");
    return *result;
}

double RequireFrozenAggregateDelta(
    const ControlledOneSidedStopExtensionExperimentResult& experiment)
{
    for (const auto& pairwise : experiment.pairwiseDeltas)
    {
        if (pairwise.values.candidateStrategyOutputIdentity ==
                kProbabilityConditionedStopExtensionOutputIdentity &&
            pairwise.values.referenceStrategyOutputIdentity ==
                kFixedStopLossOutputIdentity)
        {
            return pairwise.values.aggregateDirectionalLogReturnDelta;
        }
    }
    throw std::runtime_error("phase19b_phase18b_pairwise_delta_missing");
}

void RequireTsvSafe(std::string_view value)
{
    if (value.find_first_of("\t\r\n") != std::string_view::npos)
        throw std::runtime_error("phase19b_tsv_unsafe_text");
}

std::string BuildObservationsTsv(
    const Phase19BPathMechanismResult& result)
{
    std::ostringstream output;
    output
        << "schema_identity\tschema_version\tmodel_id\texperiment_id\tsymbol"
        << "\tcohort_role\tprediction_horizon\twindow_start\twindow_end"
        << "\tobservation_id\tobservation_ordinal\tinference_window_start_row"
        << "\tentry_source_row\tterminal_source_row\tentry_timestamp_unix_seconds"
        << "\tterminal_timestamp_unix_seconds\tpredicted_class\tdirection"
        << "\tdirectional_probability\tnormalized_directional_confidence"
        << "\tactivation_status\tfixed_base_stop_log_distance"
        << "\textension_multiplier\textension_stop_log_distance\tmapping_identity"
        << "\tfixed_strategy_return\textension_strategy_return"
        << "\textension_minus_fixed_return\toutcome_class\tfixed_stop_hit"
        << "\textension_stop_hit\tfixed_exit_reason\textension_exit_reason"
        << "\tmaximum_adverse_excursion\tmaximum_favorable_excursion"
        << "\tmae_path_point_ordinal\tmae_source_row\tmae_timestamp_unix_seconds"
        << "\tmfe_path_point_ordinal\tmfe_source_row\tmfe_timestamp_unix_seconds"
        << "\tfirst_fixed_stop_path_point_ordinal\tfirst_fixed_stop_source_row"
        << "\tfirst_fixed_stop_timestamp_unix_seconds"
        << "\tfirst_extension_stop_path_point_ordinal\tfirst_extension_stop_source_row"
        << "\tfirst_extension_stop_timestamp_unix_seconds"
        << "\tbars_from_entry_to_fixed_stop\tbars_from_entry_to_extension_stop"
        << "\tfixed_stop_horizon_fraction\textension_stop_horizon_fraction"
        << "\tbars_remaining_after_fixed_stop\trecovered_to_entry"
        << "\tfirst_recovery_to_entry_path_point_ordinal"
        << "\tfirst_recovery_to_entry_source_row"
        << "\tfirst_recovery_to_entry_timestamp_unix_seconds"
        << "\tbars_from_fixed_stop_to_recovery"
        << "\trecovery_to_entry_horizon_fraction"
        << "\tachieved_favorable_base_stop_after_fixed_breach"
        << "\tfirst_favorable_base_stop_path_point_ordinal"
        << "\tfirst_favorable_base_stop_source_row"
        << "\tfirst_favorable_base_stop_timestamp_unix_seconds"
        << "\tterminal_directional_return\tterminal_directional_class"
        << "\tbars_in_authoritative_path\tadverse_excursion_beyond_fixed_stop"
        << "\trequired_extra_room_before_recovery"
        << "\tmae_horizon_fraction\tmfe_horizon_fraction"
        << "\tcomparison_semantics\n";

    for (const auto& item : result.observations)
    {
        output
            << result.schemaIdentity << '\t' << result.schemaVersion << '\t'
            << result.modelId << '\t'
            << OptionalInteger(result.experimentId) << '\t'
            << result.symbol << '\t' << result.cohortRole << '\t'
            << result.predictionHorizon << '\t' << result.windowStart << '\t'
            << result.windowEnd << '\t' << item.observationId << '\t'
            << item.observationOrdinal << '\t'
            << item.inferenceWindowStartRow << '\t' << item.entrySourceRow
            << '\t' << item.terminalSourceRow << '\t'
            << item.entryTimestampUnixSeconds << '\t'
            << item.terminalTimestampUnixSeconds << '\t'
            << item.predictedClass << '\t' << DirectionText(item.direction)
            << '\t' << Decimal(item.directionalProbability) << '\t'
            << Decimal(item.normalizedDirectionalConfidence) << '\t'
            << (item.activated ? "true" : "false") << '\t'
            << Decimal(item.baseStopLogarithmicDistance) << '\t'
            << Decimal(item.extensionMultiplier) << '\t'
            << Decimal(item.extensionStopLogarithmicDistance) << '\t'
            << kOneSidedStopExtensionMappingIdentity << '\t'
            << Decimal(item.fixedStrategyReturn) << '\t'
            << Decimal(item.extensionStrategyReturn) << '\t'
            << Decimal(item.extensionMinusFixedReturn) << '\t'
            << Phase19BOutcomeClassText(item.outcomeClass) << '\t'
            << (item.fixedStopHit ? "true" : "false") << '\t'
            << (item.extensionStopHit ? "true" : "false") << '\t'
            << StrategyExitReasonText(item.fixedExitReason) << '\t'
            << StrategyExitReasonText(item.extensionExitReason) << '\t'
            << Decimal(item.maximumAdverseExcursion) << '\t'
            << Decimal(item.maximumFavorableExcursion) << '\t'
            << OptionalInteger(item.maximumAdversePathPointOrdinal) << '\t'
            << OptionalInteger(item.maximumAdverseSourceRow) << '\t'
            << OptionalInteger(item.maximumAdverseTimestampUnixSeconds) << '\t'
            << OptionalInteger(item.maximumFavorablePathPointOrdinal) << '\t'
            << OptionalInteger(item.maximumFavorableSourceRow) << '\t'
            << OptionalInteger(item.maximumFavorableTimestampUnixSeconds) << '\t'
            << OptionalInteger(item.fixedStopPathPointOrdinal) << '\t'
            << OptionalInteger(item.fixedStopSourceRow) << '\t'
            << OptionalInteger(item.fixedStopTimestampUnixSeconds) << '\t'
            << OptionalInteger(item.extensionStopPathPointOrdinal) << '\t'
            << OptionalInteger(item.extensionStopSourceRow) << '\t'
            << OptionalInteger(item.extensionStopTimestampUnixSeconds) << '\t'
            << OptionalInteger(item.barsFromEntryToFixedStop) << '\t'
            << OptionalInteger(item.barsFromEntryToExtensionStop) << '\t'
            << OptionalDecimal(item.fixedStopHorizonFraction) << '\t'
            << OptionalDecimal(item.extensionStopHorizonFraction) << '\t'
            << OptionalInteger(item.barsRemainingAfterFixedStop) << '\t'
            << (item.recoveredToEntry ? "true" : "false") << '\t'
            << OptionalInteger(item.recoveryToEntryPathPointOrdinal) << '\t'
            << OptionalInteger(item.recoveryToEntrySourceRow) << '\t'
            << OptionalInteger(item.recoveryToEntryTimestampUnixSeconds) << '\t'
            << OptionalInteger(item.barsFromFixedStopToRecovery) << '\t'
            << OptionalDecimal(item.recoveryToEntryHorizonFraction) << '\t'
            << (item.achievedFavorableBaseStopAfterFixedBreach
                    ? "true" : "false") << '\t'
            << OptionalInteger(item.favorableBaseStopPathPointOrdinal) << '\t'
            << OptionalInteger(item.favorableBaseStopSourceRow) << '\t'
            << OptionalInteger(
                   item.favorableBaseStopTimestampUnixSeconds) << '\t'
            << Decimal(item.terminalDirectionalReturn) << '\t'
            << item.terminalDirectionalClass << '\t'
            << item.barsInAuthoritativePath << '\t'
            << Decimal(item.adverseExcursionBeyondFixedStop) << '\t'
            << OptionalDecimal(item.requiredExtraRoomBeforeRecovery) << '\t'
            << OptionalDecimal(item.maximumAdverseHorizonFraction) << '\t'
            << OptionalDecimal(item.maximumFavorableHorizonFraction) << '\t'
            << kPhase19BComparisonSemantics << '\n';
    }
    return output.str();
}

std::string BuildPathTraceTsv(const Phase19BPathMechanismResult& result)
{
    std::ostringstream output;
    output
        << "schema_identity\tschema_version\tmodel_id\tobservation_id"
        << "\tobservation_ordinal\tpath_point_ordinal\tsource_row"
        << "\ttimestamp_unix_seconds\thorizon_fraction"
        << "\tdirectional_open_log_return"
        << "\tdirectional_adverse_extreme_log_return"
        << "\tdirectional_favorable_extreme_log_return"
        << "\tdirectional_close_log_return\n";
    for (const auto& item : result.pathTrace)
    {
        output << result.schemaIdentity << '\t' << result.schemaVersion << '\t'
               << result.modelId << '\t' << item.observationId << '\t'
               << item.observationOrdinal << '\t' << item.pathPointOrdinal
               << '\t' << item.sourceRow << '\t'
               << item.timestampUnixSeconds << '\t'
               << Decimal(item.horizonFraction) << '\t'
               << Decimal(item.directionalOpenLogReturn) << '\t'
               << Decimal(item.directionalAdverseExtremeLogReturn) << '\t'
               << Decimal(item.directionalFavorableExtremeLogReturn) << '\t'
               << Decimal(item.directionalCloseLogReturn) << '\n';
    }
    return output.str();
}

std::string BuildMetadataTsv(
    const Phase19BPathMechanismResult& result,
    const std::string& observationsHash,
    const std::string& pathTraceHash)
{
    std::ostringstream output;
    output << "schema_identity\tschema_version\tkey\tvalue\n";
    const auto append = [&output, &result](std::string_view key,
                                           const std::string& value) {
        output << result.schemaIdentity << '\t' << result.schemaVersion << '\t'
               << key << '\t' << value << '\n';
    };
    append("strategy_identity",
           kProbabilityConditionedStopExtensionOutputIdentity);
    append("mapping_identity", kOneSidedStopExtensionMappingIdentity);
    append("comparison_semantics", kPhase19BComparisonSemantics);
    append("base_stop_logarithmic_distance",
           Decimal(kPhase18BDefaultBaseStopLogarithmicDistance));
    append("activation_confidence",
           Decimal(kPhase18BDefaultActivationConfidence));
    append("minimum_stop_multiplier",
           Decimal(kPhase18BMinimumStopMultiplier));
    append("maximum_stop_multiplier",
           Decimal(kPhase18BDefaultMaximumStopMultiplier));
    append("model_id", std::to_string(result.modelId));
    append("experiment_id", result.experimentId
               ? std::to_string(*result.experimentId) : "NULL");
    append("symbol", result.symbol);
    append("cohort_role", result.cohortRole);
    append("diagnostic_model_pooled", "false");
    append("prediction_horizon", std::to_string(result.predictionHorizon));
    append("window_start", result.windowStart);
    append("window_end", result.windowEnd);
    append("total_observation_count",
           std::to_string(result.totalObservationCount));
    append("total_actionable_count",
           std::to_string(result.totalActionableCount));
    append("activated_count", std::to_string(result.activatedCount));
    append("saved_count", std::to_string(result.savedCount));
    append("harmed_count", std::to_string(result.harmedCount));
    append("unchanged_count", std::to_string(result.unchangedCount));
    append("activated_fixed_return_sum",
           Decimal(result.activatedFixedReturnSum));
    append("activated_extension_return_sum",
           Decimal(result.activatedExtensionReturnSum));
    append("per_observation_delta_sum",
           Decimal(result.perObservationDeltaSum));
    append("frozen_phase18b_aggregate_delta",
           Decimal(result.frozenPhase18BAggregateDelta));
    append("aggregate_reconciliation_residual",
           Decimal(result.aggregateReconciliationResidual));
    append("market_path_hash", result.marketPathHash);
    append("phase18b_experiment_identity_hash",
           result.phase18BExperimentIdentityHash);
    append("observations_content_hash", observationsHash);
    append("path_trace_content_hash", pathTraceHash);
    append("result_hash", result.resultHash);
    append("requires_read_only_transaction",
           result.requiresReadOnlyTransaction ? "true" : "false");
    append("production_rows_modified",
           result.productionRowsModified ? "true" : "false");
    append("path_point_ordinal_semantics", "zero_based_after_entry");
    append("normalized_timing_semantics",
           "(path_point_ordinal_plus_one)/prediction_horizon");
    append("recovery_scan_semantics",
           "whole_bars_strictly_after_fixed_stop_crossing_bar");
    append("required_extra_room_semantics",
           "max_adverse_through_recovery_bar_minus_base_stop_clamped_at_zero");
    append("undefined_value_representation", "empty_tsv_field");
    return output.str();
}

} // namespace

const char* Phase19BOutcomeClassText(Phase19BOutcomeClass value) noexcept
{
    switch (value)
    {
        case Phase19BOutcomeClass::saved: return "SAVED";
        case Phase19BOutcomeClass::harmed: return "HARMED";
        case Phase19BOutcomeClass::unchanged: return "UNCHANGED";
    }
    return "INVALID";
}

void ValidatePhase19BInvocation(const Phase19BInvocationContext& context)
{
    if (!context.inferenceMode || !context.hasExplicitModel ||
        context.inferAll || context.schedulerExperiment ||
        context.schedulerCheckpointEvaluation ||
        context.schedulerWorkerAttempt || context.frozenOutcome)
    {
        throw std::invalid_argument(
            "phase19b_path_mechanism_requires_standalone_single_model_inference");
    }
}

Phase19BPathMechanismResult ExtractPhase19BPostEntryPathMechanism(
    const AuthoritativeMarketPath& marketPath,
    Phase19BExtractionContext context)
{
    if (!context.readOnlyTransactionEnforced)
        throw std::invalid_argument(
            "phase19b_read_only_transaction_not_enforced");
    const auto& provenance = marketPath.Provenance();
    const FrozenModel& frozenModel = RequireFrozenIdentity(provenance);
    if (provenance.metricDefinitionCanonical !=
            kFixedStopMetricDefinitionCanonical ||
        provenance.metricDefinitionHash != FixedStopMetricDefinitionHash())
    {
        throw std::invalid_argument("phase19b_market_path_semantics_mismatch");
    }

    ProbabilityConditionedStopExtensionConfiguration frozenConfiguration;
    if (frozenConfiguration.baseStopLogarithmicDistance != 0.001 ||
        frozenConfiguration.activationConfidence != 0.50 ||
        frozenConfiguration.minimumStopMultiplier != 1.00 ||
        frozenConfiguration.maximumStopMultiplier != 1.25)
    {
        throw std::runtime_error("phase19b_frozen_configuration_mismatch");
    }
    ProbabilityConditionedStopExtensionStrategy extensionStrategy{
        frozenConfiguration};
    if (extensionStrategy.Identity().family !=
            kProbabilityConditionedStopExtensionStrategyFamily)
        throw std::runtime_error("phase19b_frozen_strategy_identity_mismatch");

    const auto phase18B =
        EvaluateControlledOneSidedStopExtensionExperiment(
            marketPath, frozenConfiguration);
    const auto& fixed = RequireVariant(
        phase18B, kFixedStopLossOutputIdentity);
    const auto& extension = RequireVariant(
        phase18B, kProbabilityConditionedStopExtensionOutputIdentity);
    if (fixed.evaluation.executionResults.size() !=
            marketPath.Observations().size() ||
        extension.evaluation.executionResults.size() !=
            marketPath.Observations().size())
    {
        throw std::runtime_error("phase19b_observation_pairing_mismatch");
    }

    Phase19BPathMechanismResult result;
    result.schemaIdentity = kPhase19BPathMechanismSchemaIdentity;
    result.experimentId = context.experimentId;
    result.modelId = provenance.modelId;
    result.symbol = std::string{frozenModel.symbol};
    result.cohortRole = std::string{frozenModel.cohortRole};
    result.predictionHorizon = provenance.predictionHorizon;
    result.windowStart = provenance.evaluationStart;
    result.windowEnd = provenance.evaluationEnd;
    result.totalObservationCount = marketPath.Observations().size();
    result.totalActionableCount = fixed.metrics.actionableCount;
    result.marketPathHash = marketPath.Hash();
    result.phase18BExperimentIdentityHash = phase18B.experimentIdentityHash;
    result.frozenPhase18BAggregateDelta =
        RequireFrozenAggregateDelta(phase18B);

    for (std::size_t index = 0;
         index < marketPath.Observations().size(); ++index)
    {
        const auto& observation = marketPath.Observations()[index];
        const auto& fixedExecution =
            fixed.evaluation.executionResults[index];
        const auto& extensionExecution =
            extension.evaluation.executionResults[index];
        if (fixedExecution.observationOrdinal !=
                observation.observationOrdinal ||
            extensionExecution.observationOrdinal !=
                observation.observationOrdinal ||
            fixedExecution.direction != extensionExecution.direction)
        {
            throw std::runtime_error("phase19b_observation_pairing_mismatch");
        }
        if (fixedExecution.direction == PositionDirection::flat)
            continue;

        const auto decision = extensionStrategy.StopDecision(observation);
        if (decision.normalizedDirectionalConfidence <=
            frozenConfiguration.activationConfidence)
        {
            if (fixedExecution != extensionExecution)
                throw std::runtime_error(
                    "phase19b_nonactivated_execution_mismatch");
            continue;
        }
        if (!fixedExecution.directionalLogReturn ||
            !extensionExecution.directionalLogReturn ||
            !observation.decisionTimestampUnixSeconds ||
            !observation.terminalTimestampUnixSeconds)
        {
            throw std::runtime_error("phase19b_activated_evidence_missing");
        }

        Phase19BPathMechanismObservation item;
        item.observationId = ObservationIdentity(marketPath, observation);
        item.observationOrdinal = observation.observationOrdinal;
        item.inferenceWindowStartRow = observation.inferenceWindowStartRow;
        item.entrySourceRow = observation.decisionRow;
        item.terminalSourceRow = observation.terminalRow;
        item.entryTimestampUnixSeconds =
            *observation.decisionTimestampUnixSeconds;
        item.terminalTimestampUnixSeconds =
            *observation.terminalTimestampUnixSeconds;
        item.predictedClass = observation.predictedClass;
        item.direction = fixedExecution.direction;
        item.directionalProbability = decision.directionalProbability;
        item.normalizedDirectionalConfidence =
            decision.normalizedDirectionalConfidence;
        item.activated = true;
        item.baseStopLogarithmicDistance =
            frozenConfiguration.baseStopLogarithmicDistance;
        item.extensionMultiplier = decision.stopMultiplier;
        item.extensionStopLogarithmicDistance =
            decision.effectiveStopLogarithmicDistance;
        item.fixedStrategyReturn = *fixedExecution.directionalLogReturn;
        item.extensionStrategyReturn =
            *extensionExecution.directionalLogReturn;
        item.extensionMinusFixedReturn =
            item.extensionStrategyReturn - item.fixedStrategyReturn;
        if (item.extensionMinusFixedReturn > 0.0)
        {
            item.outcomeClass = Phase19BOutcomeClass::saved;
            ++result.savedCount;
        }
        else if (item.extensionMinusFixedReturn < 0.0)
        {
            item.outcomeClass = Phase19BOutcomeClass::harmed;
            ++result.harmedCount;
        }
        else
        {
            item.outcomeClass = Phase19BOutcomeClass::unchanged;
            ++result.unchangedCount;
        }
        item.fixedExitReason = fixedExecution.reason;
        item.extensionExitReason = extensionExecution.reason;
        item.fixedStopHit =
            fixedExecution.reason == StrategyExitReason::fixedStopExit;
        item.extensionStopHit =
            extensionExecution.reason == StrategyExitReason::fixedStopExit;
        item.fixedStopPathPointOrdinal =
            fixedExecution.triggeringPathPointOrdinal;
        item.extensionStopPathPointOrdinal =
            extensionExecution.triggeringPathPointOrdinal;
        item.fixedStopSourceRow = fixedExecution.triggeringSourceRow;
        item.extensionStopSourceRow = extensionExecution.triggeringSourceRow;
        if (item.fixedStopPathPointOrdinal)
        {
            const auto& point = observation.marketPath[
                *item.fixedStopPathPointOrdinal];
            item.fixedStopTimestampUnixSeconds = point.timestampUnixSeconds;
            item.barsFromEntryToFixedStop =
                *item.fixedStopPathPointOrdinal + 1;
            item.fixedStopHorizonFraction = HorizonFraction(
                *item.fixedStopPathPointOrdinal, provenance.predictionHorizon);
            item.barsRemainingAfterFixedStop =
                provenance.predictionHorizon -
                *item.barsFromEntryToFixedStop;
        }
        if (item.extensionStopPathPointOrdinal)
        {
            const auto& point = observation.marketPath[
                *item.extensionStopPathPointOrdinal];
            item.extensionStopTimestampUnixSeconds =
                point.timestampUnixSeconds;
            item.barsFromEntryToExtensionStop =
                *item.extensionStopPathPointOrdinal + 1;
            item.extensionStopHorizonFraction = HorizonFraction(
                *item.extensionStopPathPointOrdinal,
                provenance.predictionHorizon);
        }

        const double entryPrice =
            static_cast<double>(observation.decisionClose);
        double minimumDirectionalExcursion = 0.0;
        double maximumDirectionalExcursion = 0.0;
        for (std::size_t pathOrdinal = 0;
             pathOrdinal < observation.marketPath.size(); ++pathOrdinal)
        {
            const auto& point = observation.marketPath[pathOrdinal];
            const double adverse = DirectionalAdverseExtreme(
                item.direction, entryPrice, point);
            const double favorable = DirectionalFavorableExtreme(
                item.direction, entryPrice, point);
            if (adverse < minimumDirectionalExcursion)
            {
                minimumDirectionalExcursion = adverse;
                item.maximumAdversePathPointOrdinal = pathOrdinal;
                item.maximumAdverseSourceRow = point.sourceRow;
                item.maximumAdverseTimestampUnixSeconds =
                    point.timestampUnixSeconds;
            }
            if (favorable > maximumDirectionalExcursion)
            {
                maximumDirectionalExcursion = favorable;
                item.maximumFavorablePathPointOrdinal = pathOrdinal;
                item.maximumFavorableSourceRow = point.sourceRow;
                item.maximumFavorableTimestampUnixSeconds =
                    point.timestampUnixSeconds;
            }

            Phase19BPathTracePoint trace;
            trace.observationId = item.observationId;
            trace.observationOrdinal = item.observationOrdinal;
            trace.pathPointOrdinal = pathOrdinal;
            trace.sourceRow = point.sourceRow;
            trace.timestampUnixSeconds = *point.timestampUnixSeconds;
            trace.horizonFraction = HorizonFraction(
                pathOrdinal, provenance.predictionHorizon);
            trace.directionalOpenLogReturn = DirectionalMove(
                item.direction, entryPrice, point.open);
            trace.directionalAdverseExtremeLogReturn = adverse;
            trace.directionalFavorableExtremeLogReturn = favorable;
            trace.directionalCloseLogReturn = DirectionalMove(
                item.direction, entryPrice, point.close);
            result.pathTrace.push_back(std::move(trace));
        }
        item.maximumAdverseExcursion = -minimumDirectionalExcursion;
        item.maximumFavorableExcursion = maximumDirectionalExcursion;
        if (item.maximumAdversePathPointOrdinal)
            item.maximumAdverseHorizonFraction = HorizonFraction(
                *item.maximumAdversePathPointOrdinal,
                provenance.predictionHorizon);
        if (item.maximumFavorablePathPointOrdinal)
            item.maximumFavorableHorizonFraction = HorizonFraction(
                *item.maximumFavorablePathPointOrdinal,
                provenance.predictionHorizon);
        item.adverseExcursionBeyondFixedStop = std::max(
            0.0, item.maximumAdverseExcursion -
                frozenConfiguration.baseStopLogarithmicDistance);

        if (item.fixedStopPathPointOrdinal)
        {
            const std::size_t firstSubsequent =
                static_cast<std::size_t>(*item.fixedStopPathPointOrdinal + 1);
            for (std::size_t pathOrdinal = firstSubsequent;
                 pathOrdinal < observation.marketPath.size(); ++pathOrdinal)
            {
                const auto& point = observation.marketPath[pathOrdinal];
                const double favorable = DirectionalFavorableExtreme(
                    item.direction, entryPrice, point);
                if (!item.recoveredToEntry && favorable >= 0.0)
                {
                    item.recoveredToEntry = true;
                    item.recoveryToEntryPathPointOrdinal = pathOrdinal;
                    item.recoveryToEntrySourceRow = point.sourceRow;
                    item.recoveryToEntryTimestampUnixSeconds =
                        point.timestampUnixSeconds;
                    item.barsFromFixedStopToRecovery = pathOrdinal -
                        *item.fixedStopPathPointOrdinal;
                    item.recoveryToEntryHorizonFraction = HorizonFraction(
                        pathOrdinal, provenance.predictionHorizon);
                }
                if (!item.achievedFavorableBaseStopAfterFixedBreach &&
                    favorable >=
                        frozenConfiguration.baseStopLogarithmicDistance)
                {
                    item.achievedFavorableBaseStopAfterFixedBreach = true;
                    item.favorableBaseStopPathPointOrdinal = pathOrdinal;
                    item.favorableBaseStopSourceRow = point.sourceRow;
                    item.favorableBaseStopTimestampUnixSeconds =
                        point.timestampUnixSeconds;
                }
            }
        }
        if (item.recoveryToEntryPathPointOrdinal)
        {
            double adverseThroughRecovery = 0.0;
            for (std::size_t pathOrdinal = 0;
                 pathOrdinal <= *item.recoveryToEntryPathPointOrdinal;
                 ++pathOrdinal)
            {
                adverseThroughRecovery = std::max(
                    adverseThroughRecovery,
                    -DirectionalAdverseExtreme(
                        item.direction, entryPrice,
                        observation.marketPath[pathOrdinal]));
            }
            item.requiredExtraRoomBeforeRecovery = std::max(
                0.0, adverseThroughRecovery -
                    frozenConfiguration.baseStopLogarithmicDistance);
        }

        item.terminalDirectionalReturn = DirectionalMove(
            item.direction, entryPrice, observation.terminalClose);
        item.terminalDirectionalClass =
            item.terminalDirectionalReturn > 0.0 ? "positive" :
            (item.terminalDirectionalReturn < 0.0 ? "negative" : "neutral");
        item.barsInAuthoritativePath = observation.marketPath.size();

        result.activatedFixedReturnSum += item.fixedStrategyReturn;
        result.activatedExtensionReturnSum += item.extensionStrategyReturn;
        result.perObservationDeltaSum += item.extensionMinusFixedReturn;
        result.observations.push_back(std::move(item));
    }

    result.activatedCount = result.observations.size();
    if (result.savedCount + result.harmedCount + result.unchangedCount !=
            result.activatedCount)
        throw std::runtime_error("phase19b_outcome_accounting_mismatch");
    if (phase18B.requiresReadOnlyTransaction != true ||
        phase18B.productionRowsModified != false)
        throw std::runtime_error("phase19b_phase18b_read_only_contract_mismatch");
    result.aggregateReconciliationResidual =
        result.perObservationDeltaSum - result.frozenPhase18BAggregateDelta;

    RequireTsvSafe(result.schemaIdentity);
    RequireTsvSafe(result.symbol);
    RequireTsvSafe(result.cohortRole);
    result.observationsTsv = BuildObservationsTsv(result);
    result.pathTraceTsv = BuildPathTraceTsv(result);
    const std::string observationsHash =
        InferenceProfitability::DeterministicHash(result.observationsTsv);
    const std::string pathTraceHash =
        InferenceProfitability::DeterministicHash(result.pathTraceTsv);
    std::ostringstream semantic;
    semantic << "phase19b_path_mechanism_result_v1;"
             << "schema=" << result.schemaIdentity << ';'
             << "model=" << result.modelId << ';'
             << "experiment="
             << (result.experimentId
                     ? std::to_string(*result.experimentId) : "NULL") << ';'
             << "market_path_hash=" << result.marketPathHash << ';'
             << "phase18b_experiment_identity_hash="
             << result.phase18BExperimentIdentityHash << ';'
             << "observations_hash=" << observationsHash << ';'
             << "path_trace_hash=" << pathTraceHash << ';';
    result.resultHash =
        InferenceProfitability::DeterministicHash(semantic.str());
    result.metadataTsv = BuildMetadataTsv(
        result, observationsHash, pathTraceHash);
    return result;
}

} // namespace EA::StrategyEvaluation
