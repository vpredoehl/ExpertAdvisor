#include "Phase19StateInteractionAnalysis.hpp"

#include "../../Headers/FeatureLayout.hpp"

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

struct CohortSpec
{
    long long modelId;
    const char* symbol;
    std::uint64_t horizon;
    Phase19CohortRole role;
};

constexpr std::array<CohortSpec, 7> kCohort{{
    {1745, "audchfrmp", 4, Phase19CohortRole::primary},
    {1743, "eurchfrmp", 12, Phase19CohortRole::primary},
    {1729, "usdcadrmp", 6, Phase19CohortRole::primary},
    {1735, "gbpusdrmp", 6, Phase19CohortRole::primary},
    {1662, "gbpjpyrmp", 8, Phase19CohortRole::primary},
    {1692, "audcadrmp", 14, Phase19CohortRole::primary},
    {1805, "eurusdrmp", 4, Phase19CohortRole::diagnostic}}};

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

std::string Decimal(double value)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

std::string OptionalDecimal(const std::optional<double>& value)
{
    return value ? Decimal(*value) : "NULL";
}

double Value(const std::optional<double>& value)
{
    return value.value_or(0.0);
}

const char* RoleText(Phase19CohortRole value)
{
    return value == Phase19CohortRole::primary ? "primary" : "diagnostic";
}

const char* WindowText(Phase19WindowLabel value)
{
    return value == Phase19WindowLabel::discoveryHistory2025
        ? "discovery_history_2025"
        : "temporal_validation_history_2026";
}

std::pair<Phase19CohortRole, Phase19WindowLabel> ValidateScope(
    const MarketPathProvenance& provenance)
{
    const auto model = std::find_if(kCohort.begin(), kCohort.end(),
        [&](const CohortSpec& item) { return item.modelId == provenance.modelId; });
    if (model == kCohort.end())
        throw std::invalid_argument("phase19_model_not_in_predeclared_cohort");
    if (provenance.symbol != model->symbol ||
        provenance.predictionHorizon != model->horizon)
        throw std::invalid_argument("phase19_persisted_model_identity_mismatch");
    if (provenance.evaluationStart == "2025-01-01" &&
        provenance.evaluationEnd == "2026-01-01")
        return {model->role, Phase19WindowLabel::discoveryHistory2025};
    if (provenance.evaluationStart == "2026-01-01" &&
        provenance.evaluationEnd == "2026-09-01")
        return {model->role,
                Phase19WindowLabel::temporalValidationHistory2026};
    throw std::invalid_argument("phase19_date_range_not_predeclared");
}

const Phase18AObservationEvidence& Evidence(
    const ControlledOneSidedStopExtensionExperimentResult& phase18B,
    const std::string& strategy,
    std::uint64_t ordinal)
{
    for (const auto& item : phase18B.observationEvidence)
        if (item.strategyOutputIdentity == strategy &&
            item.observationOrdinal == ordinal)
            return item;
    throw std::runtime_error("phase19_paired_evidence_missing");
}

Phase19StrategyMetrics BuildMetrics(
    const ControlledOneSidedStopExtensionExperimentResult& phase18B,
    const std::string& strategy,
    const std::vector<std::uint64_t>& ordinals,
    std::uint64_t activatedCount)
{
    Phase19StrategyMetrics metrics;
    metrics.observationCount = ordinals.size();
    metrics.actionableCount = ordinals.size();
    metrics.activatedCount = activatedCount;
    std::vector<double> holding;
    double stopDistanceSum = 0.0;
    double multiplierSum = 0.0;
    double equity = 0.0;
    double peak = 0.0;
    for (const auto ordinal : ordinals)
    {
        const auto& item = Evidence(phase18B, strategy, ordinal);
        if (!item.actionable || !item.execution.directionalLogReturn ||
            !item.holdingDurationSeconds ||
            !item.maximumAdverseExcursion ||
            !item.maximumFavorableExcursion ||
            !item.effectiveStopLogarithmicDistance ||
            !item.appliedStopMultiplier)
            throw std::runtime_error("phase19_actionable_evidence_incomplete");
        const double outcome = *item.execution.directionalLogReturn;
        metrics.aggregateDirectionalLogReturn += outcome;
        if (outcome > 0.0) ++metrics.winningCount;
        else if (outcome < 0.0) ++metrics.losingCount;
        else ++metrics.zeroCount;
        if (item.execution.reason == StrategyExitReason::fixedStopExit)
            ++metrics.stopHitCount;
        else if (item.execution.reason == StrategyExitReason::terminalExit)
            ++metrics.terminalExitCount;
        else
            throw std::runtime_error("phase19_actionable_exit_reason_invalid");
        holding.push_back(*item.holdingDurationSeconds);
        metrics.maximumAdverseExcursion = std::max(
            metrics.maximumAdverseExcursion.value_or(0.0),
            *item.maximumAdverseExcursion);
        metrics.maximumFavorableExcursion = std::max(
            metrics.maximumFavorableExcursion.value_or(0.0),
            *item.maximumFavorableExcursion);
        stopDistanceSum += *item.effectiveStopLogarithmicDistance;
        multiplierSum += *item.appliedStopMultiplier;
        equity += outcome;
        peak = std::max(peak, equity);
        metrics.maximumDrawdown = std::max(
            metrics.maximumDrawdown.value_or(0.0), peak - equity);
    }
    if (!ordinals.empty())
    {
        const double count = static_cast<double>(ordinals.size());
        metrics.averageDirectionalLogReturn =
            metrics.aggregateDirectionalLogReturn / count;
        metrics.winningRate = metrics.winningCount / count;
        metrics.losingRate = metrics.losingCount / count;
        metrics.zeroRate = metrics.zeroCount / count;
        metrics.stopHitRate = metrics.stopHitCount / count;
        metrics.terminalExitRate = metrics.terminalExitCount / count;
        metrics.averageHoldingDurationSeconds =
            std::accumulate(holding.begin(), holding.end(), 0.0) / count;
        std::sort(holding.begin(), holding.end());
        const std::size_t middle = holding.size() / 2;
        metrics.medianHoldingDurationSeconds = holding.size() % 2
            ? holding[middle]
            : (holding[middle - 1] + holding[middle]) / 2.0;
        metrics.averageEffectiveStopLogarithmicDistance =
            stopDistanceSum / count;
        metrics.averageStopMultiplier = multiplierSum / count;
    }
    if (metrics.winningCount + metrics.losingCount + metrics.zeroCount !=
            metrics.actionableCount ||
        metrics.stopHitCount + metrics.terminalExitCount !=
            metrics.actionableCount)
        throw std::runtime_error("phase19_metric_accounting_mismatch");
    return metrics;
}

Phase19PairwiseDelta Delta(const Phase19StrategyMetrics& extension,
                           const Phase19StrategyMetrics& fixed)
{
    if (extension.observationCount != fixed.observationCount ||
        extension.actionableCount != fixed.actionableCount ||
        extension.activatedCount != fixed.activatedCount)
        throw std::runtime_error("phase19_pairwise_population_mismatch");
    return {
        extension.aggregateDirectionalLogReturn -
            fixed.aggregateDirectionalLogReturn,
        Value(extension.averageDirectionalLogReturn) -
            Value(fixed.averageDirectionalLogReturn),
        Value(extension.winningRate) - Value(fixed.winningRate),
        Value(extension.losingRate) - Value(fixed.losingRate),
        Value(extension.stopHitRate) - Value(fixed.stopHitRate),
        Value(extension.terminalExitRate) - Value(fixed.terminalExitRate),
        Value(extension.averageHoldingDurationSeconds) -
            Value(fixed.averageHoldingDurationSeconds),
        Value(extension.maximumDrawdown) - Value(fixed.maximumDrawdown)};
}

void AppendMetrics(std::ostringstream& lines,
                   const char* record,
                   const std::string& analysisHash,
                   const std::string& scopeName,
                   const std::string& strategy,
                   const Phase19StrategyMetrics& metrics)
{
    lines << record << ",analysis_configuration_hash=" << analysisHash
          << ",scope=" << scopeName
          << ",strategy=" << strategy
          << ",observation_count=" << metrics.observationCount
          << ",actionable_count=" << metrics.actionableCount
          << ",activated_count=" << metrics.activatedCount
          << ",aggregate_directional_log_return="
          << Decimal(metrics.aggregateDirectionalLogReturn)
          << ",average_actionable_log_return="
          << OptionalDecimal(metrics.averageDirectionalLogReturn)
          << ",winning_count=" << metrics.winningCount
          << ",winning_rate=" << OptionalDecimal(metrics.winningRate)
          << ",losing_count=" << metrics.losingCount
          << ",losing_rate=" << OptionalDecimal(metrics.losingRate)
          << ",zero_count=" << metrics.zeroCount
          << ",zero_rate=" << OptionalDecimal(metrics.zeroRate)
          << ",stop_hit_count=" << metrics.stopHitCount
          << ",stop_hit_rate=" << OptionalDecimal(metrics.stopHitRate)
          << ",terminal_exit_count=" << metrics.terminalExitCount
          << ",terminal_exit_rate=" << OptionalDecimal(metrics.terminalExitRate)
          << ",average_holding_seconds="
          << OptionalDecimal(metrics.averageHoldingDurationSeconds)
          << ",median_holding_seconds="
          << OptionalDecimal(metrics.medianHoldingDurationSeconds)
          << ",maximum_adverse_excursion_outcome="
          << OptionalDecimal(metrics.maximumAdverseExcursion)
          << ",maximum_favorable_excursion_outcome="
          << OptionalDecimal(metrics.maximumFavorableExcursion)
          << ",maximum_drawdown=" << OptionalDecimal(metrics.maximumDrawdown)
          << ",average_effective_stop_log_distance="
          << OptionalDecimal(metrics.averageEffectiveStopLogarithmicDistance)
          << ",average_stop_multiplier="
          << OptionalDecimal(metrics.averageStopMultiplier) << '\n';
}

void AppendDelta(std::ostringstream& lines,
                 const char* record,
                 const std::string& analysisHash,
                 const std::string& scopeName,
                 const Phase19PairwiseDelta& value)
{
    lines << record << ",analysis_configuration_hash=" << analysisHash
          << ",scope=" << scopeName
          << ",candidate="
          << kProbabilityConditionedStopExtensionOutputIdentity
          << ",reference=" << kFixedStopLossOutputIdentity
          << ",aggregate_return_delta=" << Decimal(value.aggregateReturnDelta)
          << ",average_return_delta=" << Decimal(value.averageReturnDelta)
          << ",winning_rate_delta=" << Decimal(value.winningRateDelta)
          << ",losing_rate_delta=" << Decimal(value.losingRateDelta)
          << ",stop_hit_rate_delta=" << Decimal(value.stopHitRateDelta)
          << ",terminal_exit_rate_delta="
          << Decimal(value.terminalExitRateDelta)
          << ",average_holding_seconds_delta="
          << Decimal(value.averageHoldingDurationSecondsDelta)
          << ",maximum_drawdown_delta="
          << Decimal(value.maximumDrawdownDelta) << '\n';
}

} // namespace

const std::array<Phase19FeatureIdentity, kPhase19FeatureCount>&
Phase19FeatureIdentities()
{
    static const std::array<Phase19FeatureIdentity, kPhase19FeatureCount>
        identities{{
        {Phase19Feature::volatilityRegime, "volatility_state",
         "volatility_regime", causalVolatilityRegimeCol,
         "tensor_feature:volatility_regime:column=38",
         "decision_bar_completed_close_causal"},
        {Phase19Feature::rollingRangeExpansion, "volatility_state",
         "rolling_range_expansion", causalRollingRangeExpansionCol,
         "tensor_feature:rolling_range_expansion:column=46",
         "decision_bar_completed_range_causal"},
        {Phase19Feature::directionalRange,
         "directional_movement_efficiency_state", "directional_range",
         causalDirectionalRangeCol,
         "tensor_feature:directional_range:column=39",
         "predecessor_completed_bar_only"},
        {Phase19Feature::directionalEfficiency,
         "directional_movement_efficiency_state", "directional_efficiency",
         causalDirectionalPersistenceCol,
         "tensor_feature:directional_efficiency:column=41",
         "predecessor_completed_closes_only"},
        {Phase19Feature::returnSignPersistence,
         "directional_movement_efficiency_state", "return_sign_persistence",
         causalReturnSignPersistenceCol,
         "tensor_feature:return_sign_persistence:column=42",
         "predecessor_completed_closes_only"},
        {Phase19Feature::returnDirectionImbalance,
         "directional_movement_efficiency_state",
         "return_direction_imbalance", causalReturnDirectionImbalanceCol,
         "tensor_feature:return_direction_imbalance:column=43",
         "predecessor_completed_closes_only"},
        {Phase19Feature::historicalLevelProximity,
         "structural_proximity_state", "historical_level_proximity",
         historicalLevelProximityCol,
         "tensor_feature:historical_level_proximity:column=47",
         "decision_bar_completed_historical_levels_causal"}}};
    static_assert(causalVolatilityRegimeCol == 38);
    static_assert(causalDirectionalRangeCol == 39);
    static_assert(causalDirectionalPersistenceCol == 41);
    static_assert(causalReturnSignPersistenceCol == 42);
    static_assert(causalReturnDirectionImbalanceCol == 43);
    static_assert(causalRollingRangeExpansionCol == 46);
    static_assert(historicalLevelProximityCol == 47);
    return identities;
}

Phase19QuartileBoundaries ComputePhase19QuartileBoundaries(
    std::vector<double> values)
{
    if (values.empty())
        throw std::invalid_argument("phase19_quartiles_require_values");
    for (double value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument("phase19_nonfinite_state_value");
    std::sort(values.begin(), values.end());
    const auto boundary = [&](std::size_t numerator) {
        // Deterministic inverse empirical CDF. Values equal to a boundary are
        // assigned to the upper stratum, so ties are never split by ordinal.
        const std::size_t index = std::min(
            values.size() - 1, (values.size() * numerator) / 4);
        return values[index];
    };
    return {boundary(1), boundary(2), boundary(3)};
}

std::size_t Phase19QuartileStratum(
    double value, const Phase19QuartileBoundaries& boundaries)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("phase19_nonfinite_state_value");
    if (value < boundaries.q1) return 0;
    if (value < boundaries.q2) return 1;
    if (value < boundaries.q3) return 2;
    return 3;
}

const char* Phase19SupportLabel(std::uint64_t activatedCount) noexcept
{
    if (activatedCount < 30) return "sparse";
    if (activatedCount < 100) return "limited";
    if (activatedCount < 300) return "moderate";
    return "substantial";
}

std::vector<std::string> ComputePhase19CategoricalGroups(
    std::vector<std::string> categories)
{
    for (const auto& value : categories)
        if (value.empty() || value.find_first_of(",\n\r") != std::string::npos)
            throw std::invalid_argument("phase19_invalid_category_identity");
    std::sort(categories.begin(), categories.end());
    categories.erase(std::unique(categories.begin(), categories.end()),
                     categories.end());
    return categories;
}

std::size_t Phase19CategoricalStratum(
    const std::string& category,
    const std::vector<std::string>& sortedGroups)
{
    if (!std::is_sorted(sortedGroups.begin(), sortedGroups.end()) ||
        std::adjacent_find(sortedGroups.begin(), sortedGroups.end()) !=
            sortedGroups.end())
        throw std::invalid_argument("phase19_categories_not_canonical");
    const auto found = std::lower_bound(
        sortedGroups.begin(), sortedGroups.end(), category);
    if (found == sortedGroups.end() || *found != category)
        throw std::invalid_argument("phase19_category_unavailable");
    return static_cast<std::size_t>(found - sortedGroups.begin());
}

void ValidatePhase19Invocation(const Phase19InvocationContext& context)
{
    if (!context.inferenceMode || !context.hasExplicitModel ||
        context.inferAll || context.schedulerExperiment ||
        context.schedulerCheckpointEvaluation || context.schedulerWorkerAttempt ||
        context.frozenOutcome)
        throw std::invalid_argument(
            "--probability-stop-extension-state-analysis requires standalone "
            "--infer with one explicit --model and no scheduler context");
}

Phase19StateInteractionAnalysisResult EvaluatePhase19StateInteractionAnalysis(
    const AuthoritativeMarketPath& marketPath,
    const std::vector<Phase19EntryState>& entryStates,
    Phase19AnalysisContext context)
{
    const auto [role, window] = ValidateScope(marketPath.Provenance());
    if (entryStates.size() != marketPath.Observations().size())
        throw std::invalid_argument("phase19_entry_state_population_mismatch");
    for (std::size_t index = 0; index < entryStates.size(); ++index)
    {
        if (entryStates[index].observationOrdinal !=
            marketPath.Observations()[index].observationOrdinal)
            throw std::invalid_argument("phase19_entry_state_ordinal_mismatch");
        for (double value : entryStates[index].values)
            if (!std::isfinite(value))
                throw std::invalid_argument("phase19_nonfinite_state_value");
    }

    const auto phase18B =
        EvaluateControlledOneSidedStopExtensionExperiment(marketPath);
    const auto extensionIdentity = std::find_if(
        phase18B.variants.begin(), phase18B.variants.end(),
        [](const Phase18BStrategyVariantResult& item) {
            return item.strategyOutputIdentity ==
                kProbabilityConditionedStopExtensionOutputIdentity;
        });
    if (extensionIdentity == phase18B.variants.end())
        throw std::runtime_error("phase19_phase18b_extension_missing");

    Phase19StateInteractionAnalysisResult result;
    result.cohortRole = role;
    result.windowLabel = window;
    result.modelAcceptanceQualified = context.modelAcceptanceQualified;
    result.phase18BStrategyConfigurationHash =
        extensionIdentity->evaluation.strategyIdentity.configurationHash;

    std::ostringstream configuration;
    configuration
        << "phase19_analysis_configuration_v1;analysis_identity="
        << kPhase19AnalysisIdentity
        << ";activation_confidence="
        << CanonicalDouble(kPhase18BDefaultActivationConfidence)
        << ";minimum_stop_multiplier="
        << CanonicalDouble(kPhase18BMinimumStopMultiplier)
        << ";maximum_stop_multiplier="
        << CanonicalDouble(kPhase18BDefaultMaximumStopMultiplier)
        << ";base_stop_log_distance="
        << CanonicalDouble(kPhase18BDefaultBaseStopLogarithmicDistance)
        << ";reference=" << kFixedStopLossOutputIdentity
        << ";candidate="
        << kProbabilityConditionedStopExtensionOutputIdentity
        << ";quartiles=inverse_empirical_cdf_floor_index_ties_to_upper_v1;";
    for (const auto& identity : Phase19FeatureIdentities())
        configuration << "feature=" << identity.runtimeIdentity
                      << ":timing=" << identity.entryTiming << ';';
    result.analysisConfigurationCanonical = configuration.str();
    result.analysisConfigurationHash =
        InferenceProfitability::DeterministicHash(
            result.analysisConfigurationCanonical);

    std::vector<std::uint64_t> allActionable;
    std::vector<std::uint64_t> activated;
    std::vector<std::uint64_t> confidenceLow;
    std::vector<std::uint64_t> confidenceHigh;
    std::uint64_t confidenceLowActivatedCount = 0;
    for (const auto& observation : marketPath.Observations())
    {
        const auto& item = Evidence(
            phase18B, kProbabilityConditionedStopExtensionOutputIdentity,
            observation.observationOrdinal);
        if (!item.actionable) continue;
        allActionable.push_back(observation.observationOrdinal);
        if (!item.normalizedDirectionalConfidence)
            throw std::runtime_error("phase19_confidence_missing");
        if (*item.normalizedDirectionalConfidence >= 0.50 &&
            *item.normalizedDirectionalConfidence < 0.75)
            confidenceLow.push_back(observation.observationOrdinal);
        else if (*item.normalizedDirectionalConfidence >= 0.75)
            confidenceHigh.push_back(observation.observationOrdinal);
        if (*item.normalizedDirectionalConfidence >
            kPhase18BDefaultActivationConfidence)
        {
            activated.push_back(observation.observationOrdinal);
            if (*item.normalizedDirectionalConfidence < 0.75)
                ++confidenceLowActivatedCount;
        }
    }

    const auto addPopulation = [&](const std::string& name,
                                   const std::vector<std::uint64_t>& ordinals,
                                   std::uint64_t activatedInPopulation) {
        Phase19PopulationResult population;
        population.populationIdentity = name;
        population.fixed = BuildMetrics(
            phase18B, kFixedStopLossOutputIdentity, ordinals,
            activatedInPopulation);
        population.extension = BuildMetrics(
            phase18B, kProbabilityConditionedStopExtensionOutputIdentity,
            ordinals, activatedInPopulation);
        population.extensionMinusFixed = Delta(
            population.extension, population.fixed);
        result.populations.push_back(std::move(population));
    };
    addPopulation("all_actionable", allActionable, activated.size());
    addPopulation("activated_confidence_gt_0.50", activated, activated.size());
    addPopulation("confidence_0.50_to_lt_0.75", confidenceLow,
                  confidenceLowActivatedCount);
    addPopulation("confidence_0.75_to_1.00_inclusive", confidenceHigh,
                  confidenceHigh.size());

    for (std::size_t featureIndex = 0;
         featureIndex < kPhase19FeatureCount; ++featureIndex)
    {
        std::vector<double> values;
        values.reserve(activated.size());
        for (const auto ordinal : activated)
            values.push_back(entryStates[ordinal].values[featureIndex]);
        if (values.empty())
            continue;
        const auto boundaries = ComputePhase19QuartileBoundaries(values);
        std::array<std::vector<std::uint64_t>, 4> strata;
        for (const auto ordinal : activated)
        {
            const double value = entryStates[ordinal].values[featureIndex];
            strata[Phase19QuartileStratum(value, boundaries)].push_back(ordinal);
        }
        std::uint64_t reconciled = 0;
        for (std::size_t stratum = 0; stratum < strata.size(); ++stratum)
        {
            Phase19StateStratumResult item;
            item.featureIdentity = Phase19FeatureIdentities()[featureIndex];
            item.boundaries = boundaries;
            item.stratumOrdinal = stratum;
            item.activatedCount = strata[stratum].size();
            item.shareOfActivated = static_cast<double>(item.activatedCount) /
                static_cast<double>(activated.size());
            item.supportLabel = Phase19SupportLabel(item.activatedCount);
            item.fixed = BuildMetrics(
                phase18B, kFixedStopLossOutputIdentity, strata[stratum],
                item.activatedCount);
            item.extension = BuildMetrics(
                phase18B,
                kProbabilityConditionedStopExtensionOutputIdentity,
                strata[stratum], item.activatedCount);
            item.extensionMinusFixed = Delta(item.extension, item.fixed);
            reconciled += item.activatedCount;
            result.stateStrata.push_back(std::move(item));
        }
        if (reconciled != activated.size())
            throw std::runtime_error("phase19_strata_population_mismatch");
    }

    std::ostringstream lines;
    lines << "PHASE19_ANALYSIS"
          << ",analysis_version=" << kPhase19AnalysisConfigurationVersion
          << ",analysis_identity=" << kPhase19AnalysisIdentity
          << ",analysis_configuration_hash="
          << result.analysisConfigurationHash
          << ",model_id=" << marketPath.Provenance().modelId
          << ",symbol=" << marketPath.Provenance().symbol
          << ",prediction_horizon="
          << marketPath.Provenance().predictionHorizon
          << ",evaluation_start="
          << marketPath.Provenance().evaluationStart
          << ",evaluation_end=" << marketPath.Provenance().evaluationEnd
          << ",window_label=" << WindowText(window)
          << ",cohort_role=" << RoleText(role)
          << ",model_acceptance_qualified="
          << (context.modelAcceptanceQualified ? "true" : "false")
          << ",market_path_hash=" << marketPath.Hash()
          << ",inference_identity_hash="
          << marketPath.Provenance().inferenceScientificIdentityHash
          << ",phase18b_strategy_configuration_hash="
          << result.phase18BStrategyConfigurationHash
          << ",base_stop_log_distance="
          << Decimal(kPhase18BDefaultBaseStopLogarithmicDistance)
          << ",activation_confidence="
          << Decimal(kPhase18BDefaultActivationConfidence)
          << ",minimum_stop_multiplier="
          << Decimal(kPhase18BMinimumStopMultiplier)
          << ",maximum_stop_multiplier="
          << Decimal(kPhase18BDefaultMaximumStopMultiplier)
          << ",requires_read_only_transaction=true"
          << ",production_rows_modified=false\n";
    for (const auto& identity : Phase19FeatureIdentities())
        lines << "PHASE19_FEATURE_IDENTITY"
              << ",analysis_configuration_hash="
              << result.analysisConfigurationHash
              << ",family=" << identity.family
              << ",semantic_name=" << identity.semanticName
              << ",tensor_column=" << identity.tensorColumn
              << ",runtime_identity=" << identity.runtimeIdentity
              << ",entry_timing=" << identity.entryTiming
              << ",state_role=gating_allowed_entry_time_causal\n";
    lines << "PHASE19_OUTCOME_IDENTITY"
          << ",analysis_configuration_hash="
          << result.analysisConfigurationHash
          << ",fields=directional_log_return|stop_hit|terminal_exit|holding_time|maximum_adverse_excursion|maximum_favorable_excursion|maximum_drawdown"
          << ",state_role=outcomes_only_gating_prohibited\n";
    for (const auto& population : result.populations)
    {
        AppendMetrics(lines, "PHASE19_POPULATION_STRATEGY",
                      result.analysisConfigurationHash,
                      population.populationIdentity,
                      kFixedStopLossOutputIdentity, population.fixed);
        AppendMetrics(lines, "PHASE19_POPULATION_STRATEGY",
                      result.analysisConfigurationHash,
                      population.populationIdentity,
                      kProbabilityConditionedStopExtensionOutputIdentity,
                      population.extension);
        AppendDelta(lines, "PHASE19_POPULATION_DELTA",
                    result.analysisConfigurationHash,
                    population.populationIdentity,
                    population.extensionMinusFixed);
    }
    for (const auto& item : result.stateStrata)
    {
        const std::string scope = item.featureIdentity.semanticName +
            ":quartile_" + std::to_string(item.stratumOrdinal + 1);
        lines << "PHASE19_STATE_STRATUM"
              << ",analysis_configuration_hash="
              << result.analysisConfigurationHash
              << ",scope=" << scope
              << ",family=" << item.featureIdentity.family
              << ",feature=" << item.featureIdentity.semanticName
              << ",stratum_ordinal=" << item.stratumOrdinal
              << ",q1=" << Decimal(item.boundaries.q1)
              << ",q2=" << Decimal(item.boundaries.q2)
              << ",q3=" << Decimal(item.boundaries.q3)
              << ",boundary_rule=lower_inclusive_ties_to_upper"
              << ",activated_count=" << item.activatedCount
              << ",share_of_all_activated=" << Decimal(item.shareOfActivated)
              << ",support_label=" << item.supportLabel << '\n';
        AppendMetrics(lines, "PHASE19_STATE_STRATUM_STRATEGY",
                      result.analysisConfigurationHash, scope,
                      kFixedStopLossOutputIdentity, item.fixed);
        AppendMetrics(lines, "PHASE19_STATE_STRATUM_STRATEGY",
                      result.analysisConfigurationHash, scope,
                      kProbabilityConditionedStopExtensionOutputIdentity,
                      item.extension);
        AppendDelta(lines, "PHASE19_STATE_STRATUM_DELTA",
                    result.analysisConfigurationHash, scope,
                    item.extensionMinusFixed);
    }
    result.canonicalLines = lines.str();
    result.resultHash = InferenceProfitability::DeterministicHash(
        result.canonicalLines);
    result.canonicalLines += "PHASE19_RESULT_HASH,result_hash=" +
        result.resultHash + '\n';
    return result;
}

} // namespace EA::StrategyEvaluation
