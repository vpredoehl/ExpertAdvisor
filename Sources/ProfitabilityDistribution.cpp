#include "ProfitabilityDistribution.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string OptionalDoubleText(const std::optional<double>& value)
{
    return value ? CanonicalRecommendationDouble(*value) : "NULL";
}

bool TaggedHash(const std::string& value)
{
    if (value.size() != 24 || !value.starts_with("fnv1a64:")) return false;
    return std::all_of(value.begin() + 8, value.end(), [](char byte) {
        return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
    });
}

bool IsoDate(const std::string& value)
{
    if (value.size() != 10 || value[4] != '-' || value[7] != '-') return false;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        if (index == 4 || index == 7) continue;
        if (value[index] < '0' || value[index] > '9') return false;
    }

    const int year = (value[0] - '0') * 1000 + (value[1] - '0') * 100 +
        (value[2] - '0') * 10 + (value[3] - '0');
    const int month = (value[5] - '0') * 10 + (value[6] - '0');
    const int day = (value[8] - '0') * 10 + (value[9] - '0');
    if (year < 1 || month < 1 || month > 12 || day < 1) return false;

    constexpr int daysByMonth[] = {
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int daysInMonth = daysByMonth[month - 1];
    const bool leapYear = year % 4 == 0 &&
        (year % 100 != 0 || year % 400 == 0);
    if (month == 2 && leapYear) ++daysInMonth;
    return day <= daysInMonth;
}

std::optional<std::string> ValidateSemanticIdentity(
    const ProfitabilitySemanticIdentity& identity,
    const std::string& field)
{
    if (identity.version <= 0 || identity.canonical.empty() ||
        identity.hash.empty())
        return "empty_" + field;
    if (!TaggedHash(identity.hash) ||
        RecommendationCanonicalHash(identity.canonical) != identity.hash)
        return "invalid_" + field + "_hash";
    return std::nullopt;
}

bool NearlyEqual(double left, double right)
{
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <= 1e-12 * scale;
}

bool NearlyEqualAccumulated(double left,
                            double right,
                            std::uint64_t termCount)
{
    const long double scale = std::max({
        1.0L, std::abs(static_cast<long double>(left)),
        std::abs(static_cast<long double>(right))});
    const long double operations = std::max(
        64.0L, 8.0L * static_cast<long double>(termCount));
    const long double tolerance =
        static_cast<long double>(std::numeric_limits<double>::epsilon()) *
        operations * scale;
    return std::abs(static_cast<long double>(left) -
                    static_cast<long double>(right)) <= tolerance;
}

ProfitabilityPopulationIdentity IdentityFor(
    const ProfitabilityObservation& observation)
{
    return {
        observation.metricDefinitionCanonical,
        observation.metricDefinitionHash,
        observation.inferenceScope,
        observation.inferenceStart,
        observation.inferenceEnd,
        observation.symbol,
        observation.predictionHorizon,
        observation.modelInputWidth,
        observation.featureClassIdentity,
        observation.inferenceEvaluationSemanticIdentity,
        observation.scoringSemanticIdentity,
        observation.evaluationSemanticIdentity};
}

void AppendIdentity(std::string& canonical,
                    const std::string& name,
                    const ProfitabilitySemanticIdentity& identity)
{
    canonical += ";" + name + "_version=" + std::to_string(identity.version);
    canonical += ";" + name + "_canonical=" + LengthText(identity.canonical);
    canonical += ";" + name + "_hash=" + identity.hash;
}

std::string MembershipCanonical(
    const std::vector<ProfitabilityObservation>& observations)
{
    std::string canonical =
        "profitability_distribution_membership_v1;count=" +
        std::to_string(observations.size());
    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        canonical += ";member[" + std::to_string(index) + "]=" +
            LengthText(observations[index].observationIdentityCanonical);
    }
    return canonical;
}

double Quantile(const std::vector<double>& sortedValues, double probability)
{
    if (sortedValues.size() == 1) return sortedValues.front();
    const long double position = static_cast<long double>(probability) *
        static_cast<long double>(sortedValues.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const long double fraction = position - static_cast<long double>(lower);
    const long double value = static_cast<long double>(sortedValues[lower]) +
        (static_cast<long double>(sortedValues[upper]) -
         static_cast<long double>(sortedValues[lower])) * fraction;
    return static_cast<double>(value);
}

std::string SummaryCanonical(const ProfitabilityDistributionSummary& summary)
{
    std::string canonical = "profitability_distribution_summary_v1";
    canonical += ";state=" + ProfitabilityPopulationStateText(summary.state);
    canonical += ";reason=" + LengthText(summary.reason);
    canonical += ";population_identity=" +
        LengthText(summary.populationIdentityCanonical);
    canonical += ";population_identity_hash=" +
        (summary.populationIdentityHash.empty() ? "NULL" :
         summary.populationIdentityHash);
    canonical += ";membership=" + LengthText(summary.membershipCanonical);
    canonical += ";membership_hash=" +
        (summary.membershipHash.empty() ? "NULL" : summary.membershipHash);
    canonical += ";normalization_policy=" +
        LengthText(summary.normalizationPolicyCanonical);
    canonical += ";normalization_policy_hash=" +
        summary.normalizationPolicyHash;
    canonical += ";population_count=" +
        std::to_string(summary.populationCount);
    canonical += ";analyzable_population_count=" +
        std::to_string(summary.analyzablePopulationCount);
    canonical += ";zero_actionable_count=" +
        std::to_string(summary.zeroActionableCount);
    canonical += ";total_prediction_count=" +
        std::to_string(summary.totalPredictionCount);
    canonical += ";total_actionable_count=" +
        std::to_string(summary.totalActionableCount);
    canonical += ";negative_count=" + std::to_string(summary.negativeCount);
    canonical += ";zero_count=" + std::to_string(summary.zeroCount);
    canonical += ";positive_count=" + std::to_string(summary.positiveCount);
    canonical += ";minimum=" + OptionalDoubleText(summary.minimum);
    canonical += ";maximum=" + OptionalDoubleText(summary.maximum);
    canonical += ";mean=" + OptionalDoubleText(summary.mean);
    canonical += ";median=" + OptionalDoubleText(summary.median);
    canonical += ";population_standard_deviation=" +
        OptionalDoubleText(summary.populationStandardDeviation);
    canonical += ";median_absolute_deviation=" +
        OptionalDoubleText(summary.medianAbsoluteDeviation);
    canonical += ";quantile_count=" + std::to_string(summary.quantiles.size());
    for (std::size_t index = 0; index < summary.quantiles.size(); ++index)
    {
        canonical += ";quantile_probability[" + std::to_string(index) + "]=" +
            CanonicalRecommendationDouble(summary.quantiles[index].probability);
        canonical += ";quantile_value[" + std::to_string(index) + "]=" +
            CanonicalRecommendationDouble(summary.quantiles[index].value);
    }
    canonical += ";profitability_weight=0;profitability_score_contribution=0";
    return canonical;
}

std::string NormalizationCanonical(
    const ProfitabilityNormalizationResult& result)
{
    std::string canonical = "profitability_normalization_result_v1";
    canonical += ";profitability_observation_id=" +
        std::to_string(result.profitabilityObservationId);
    canonical += ";state=" + ProfitabilityNormalizationStateText(result.state);
    canonical += ";reason=" + LengthText(result.reason);
    canonical += ";raw_profitability_metric=" +
        OptionalDoubleText(result.rawProfitabilityMetric);
    canonical += ";population_identity_hash=" +
        (result.populationIdentityHash.empty() ? "NULL" :
         result.populationIdentityHash);
    canonical += ";membership_hash=" +
        (result.membershipHash.empty() ? "NULL" : result.membershipHash);
    canonical += ";population_size=" + std::to_string(result.populationSize);
    canonical += ";analyzable_population_size=" +
        std::to_string(result.analyzablePopulationSize);
    canonical += ";actionable_count=" + std::to_string(result.actionableCount);
    canonical += ";empirical_midrank_percentile=" +
        OptionalDoubleText(result.empiricalMidrankPercentile);
    canonical += ";bounded_candidate_metric=" +
        OptionalDoubleText(result.boundedCandidateMetric);
    canonical += ";support_reliability=" +
        CanonicalRecommendationDouble(result.supportReliability);
    canonical += ";profitability_weight=0;profitability_score_contribution=0";
    return canonical;
}

bool AddWithoutOverflow(std::uint64_t value, std::uint64_t& total)
{
    if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
    total += value;
    return true;
}

} // namespace

std::optional<std::string> ValidateProfitabilityObservation(
    const ProfitabilityObservation& observation)
{
    if (observation.profitabilityObservationId <= 0 ||
        observation.experimentId <= 0 || observation.modelId <= 0 ||
        observation.inferenceEvalResultId <= 0)
        return "invalid_profitability_observation_provenance_id";
    if (observation.metricDefinitionCanonical.empty() ||
        observation.metricDefinitionHash.empty() ||
        observation.sourceContentHash.empty() ||
        observation.observationIdentityCanonical.empty() ||
        observation.observationIdentityHash.empty())
        return "empty_profitability_observation_provenance";
    if (!TaggedHash(observation.metricDefinitionHash) ||
        RecommendationCanonicalHash(observation.metricDefinitionCanonical) !=
            observation.metricDefinitionHash)
        return "invalid_metric_definition_hash";
    if (!TaggedHash(observation.sourceContentHash))
        return "invalid_source_content_hash";
    if (!TaggedHash(observation.observationIdentityHash) ||
        RecommendationCanonicalHash(observation.observationIdentityCanonical) !=
            observation.observationIdentityHash)
        return "invalid_observation_identity_hash";
    if (observation.inferenceScope != "final" &&
        observation.inferenceScope != "checkpoint")
        return "invalid_inference_scope";
    if (!IsoDate(observation.inferenceStart) ||
        !IsoDate(observation.inferenceEnd) ||
        observation.inferenceStart >= observation.inferenceEnd)
        return "invalid_inference_window";
    if (observation.symbol.empty() || observation.predictionHorizon <= 0 ||
        observation.modelInputWidth <= 0)
        return "empty_profitability_comparability_provenance";
    if (const auto error = ValidateSemanticIdentity(
            observation.featureClassIdentity, "feature_class_identity"))
        return error;
    if (const auto error = ValidateSemanticIdentity(
            observation.inferenceEvaluationSemanticIdentity,
            "inference_evaluation_semantic_identity"))
        return error;
    if (const auto error = ValidateSemanticIdentity(
            observation.scoringSemanticIdentity, "scoring_semantic_identity"))
        return error;
    if (const auto error = ValidateSemanticIdentity(
            observation.evaluationSemanticIdentity,
            "evaluation_semantic_identity"))
        return error;
    if (observation.actionableCount > observation.predictionCount ||
        observation.winningActionableCount > observation.actionableCount ||
        observation.losingActionableCount >
            observation.actionableCount - observation.winningActionableCount)
        return "invalid_profitability_counts";
    if (!std::isfinite(
            observation.grossPositiveTerminalHorizonLogReturnSum) ||
        !std::isfinite(
            observation.grossNegativeTerminalHorizonLogReturnSum) ||
        !std::isfinite(observation.aggregateTerminalHorizonLogReturnSum) ||
        observation.grossPositiveTerminalHorizonLogReturnSum < 0.0 ||
        observation.grossNegativeTerminalHorizonLogReturnSum > 0.0 ||
        !NearlyEqualAccumulated(
            observation.grossPositiveTerminalHorizonLogReturnSum +
                observation.grossNegativeTerminalHorizonLogReturnSum,
            observation.aggregateTerminalHorizonLogReturnSum,
            observation.actionableCount))
        return "invalid_profitability_sums";
    if (observation.averageTerminalHorizonLogReturnPerActionablePrediction &&
        !std::isfinite(
            *observation.averageTerminalHorizonLogReturnPerActionablePrediction))
        return "nonfinite_profitability_average";
    if (observation.actionableCount == 0)
    {
        if (observation.winningActionableCount != 0 ||
            observation.losingActionableCount != 0 ||
            observation.grossPositiveTerminalHorizonLogReturnSum != 0.0 ||
            observation.grossNegativeTerminalHorizonLogReturnSum != 0.0 ||
            observation.aggregateTerminalHorizonLogReturnSum != 0.0 ||
            observation.averageTerminalHorizonLogReturnPerActionablePrediction)
            return "invalid_zero_actionable_profitability_shape";
    }
    else
    {
        if (!observation.averageTerminalHorizonLogReturnPerActionablePrediction)
            return "missing_profitability_average";
        const double expected =
            observation.aggregateTerminalHorizonLogReturnSum /
            static_cast<double>(observation.actionableCount);
        if (!NearlyEqual(
                expected,
                *observation.averageTerminalHorizonLogReturnPerActionablePrediction))
            return "inconsistent_profitability_average";
    }
    return std::nullopt;
}

std::string ProfitabilityPopulationIdentityCanonicalText(
    const ProfitabilityPopulationIdentity& identity)
{
    if (identity.metricDefinitionCanonical.empty() ||
        identity.metricDefinitionHash.empty() || identity.inferenceScope.empty() ||
        identity.inferenceStart.empty() || identity.inferenceEnd.empty() ||
        identity.symbol.empty() || identity.predictionHorizon <= 0 ||
        identity.modelInputWidth <= 0)
        throw std::invalid_argument("invalid_profitability_population_identity");
    if (!TaggedHash(identity.metricDefinitionHash) ||
        RecommendationCanonicalHash(identity.metricDefinitionCanonical) !=
            identity.metricDefinitionHash ||
        (identity.inferenceScope != "final" &&
         identity.inferenceScope != "checkpoint") ||
        !IsoDate(identity.inferenceStart) || !IsoDate(identity.inferenceEnd) ||
        identity.inferenceStart >= identity.inferenceEnd)
        throw std::invalid_argument("invalid_profitability_population_identity");
    if (ValidateSemanticIdentity(identity.featureClassIdentity,
                                 "feature_class_identity") ||
        ValidateSemanticIdentity(identity.inferenceEvaluationSemanticIdentity,
                                 "inference_evaluation_semantic_identity") ||
        ValidateSemanticIdentity(identity.scoringSemanticIdentity,
                                 "scoring_semantic_identity") ||
        ValidateSemanticIdentity(identity.evaluationSemanticIdentity,
                                 "evaluation_semantic_identity"))
        throw std::invalid_argument("invalid_profitability_population_identity");
    std::string canonical = "profitability_distribution_population_v1";
    canonical += ";metric_definition_canonical=" +
        LengthText(identity.metricDefinitionCanonical);
    canonical += ";metric_definition_hash=" + identity.metricDefinitionHash;
    canonical += ";inference_scope=" + LengthText(identity.inferenceScope);
    canonical += ";inference_start=" + identity.inferenceStart;
    canonical += ";inference_end=" + identity.inferenceEnd;
    canonical += ";symbol=" + LengthText(identity.symbol);
    canonical += ";prediction_horizon=" +
        std::to_string(identity.predictionHorizon);
    canonical += ";model_input_width=" +
        std::to_string(identity.modelInputWidth);
    AppendIdentity(canonical, "feature_class", identity.featureClassIdentity);
    AppendIdentity(canonical, "inference_evaluation_semantic",
                   identity.inferenceEvaluationSemanticIdentity);
    AppendIdentity(canonical, "scoring_semantic",
                   identity.scoringSemanticIdentity);
    AppendIdentity(canonical, "evaluation_semantic",
                   identity.evaluationSemanticIdentity);
    return canonical;
}

std::string ProfitabilityPopulationIdentityHash(
    const ProfitabilityPopulationIdentity& identity)
{
    return RecommendationCanonicalHash(
        ProfitabilityPopulationIdentityCanonicalText(identity));
}

std::optional<std::string> ValidateProfitabilityNormalizationPolicy(
    const ProfitabilityNormalizationPolicy& policy)
{
    if (policy.version != 1) return "unsupported_profitability_policy_version";
    if (policy.minimumAnalyzablePopulationSize == 0)
        return "invalid_minimum_analyzable_population_size";
    if (policy.supportHalfSaturationActionableCount == 0)
        return "invalid_support_half_saturation_count";
    std::set<double> distinct;
    for (double probability : policy.quantiles)
    {
        if (!std::isfinite(probability) || probability < 0.0 ||
            probability > 1.0)
            return "invalid_profitability_quantile";
        if (!distinct.insert(probability).second)
            return "duplicate_profitability_quantile";
    }
    return std::nullopt;
}

std::string ProfitabilityNormalizationPolicyCanonicalText(
    const ProfitabilityNormalizationPolicy& policy)
{
    if (const auto error = ValidateProfitabilityNormalizationPolicy(policy))
        throw std::invalid_argument(*error);
    std::vector<double> quantiles = policy.quantiles;
    std::sort(quantiles.begin(), quantiles.end());
    std::string canonical = "profitability_normalization_policy_v1";
    canonical += ";version=" + std::to_string(policy.version);
    canonical += ";primary_metric=average_terminal_horizon_log_return_per_actionable_prediction";
    canonical += ";transform=signed_empirical_midrank_percentile_v1";
    canonical += ";minimum_analyzable_population_size=" +
        std::to_string(policy.minimumAnalyzablePopulationSize);
    canonical += ";support=actionable_count_over_actionable_count_plus_half_saturation_v1";
    canonical += ";support_half_saturation_actionable_count=" +
        std::to_string(policy.supportHalfSaturationActionableCount);
    canonical += ";quantile_method=linear_interpolation_n_minus_1_v1";
    canonical += ";quantile_count=" + std::to_string(quantiles.size());
    for (std::size_t index = 0; index < quantiles.size(); ++index)
        canonical += ";quantile[" + std::to_string(index) + "]=" +
            CanonicalRecommendationDouble(quantiles[index]);
    canonical += ";profitability_weight=0;profitability_score_contribution=0";
    return canonical;
}

std::string ProfitabilityNormalizationPolicyHash(
    const ProfitabilityNormalizationPolicy& policy)
{
    return RecommendationCanonicalHash(
        ProfitabilityNormalizationPolicyCanonicalText(policy));
}

std::string ProfitabilityPopulationStateText(
    ProfitabilityPopulationState state)
{
    switch (state)
    {
        case ProfitabilityPopulationState::valid: return "valid";
        case ProfitabilityPopulationState::insufficientPopulation:
            return "insufficient_population";
        case ProfitabilityPopulationState::noAnalyzableObservations:
            return "no_analyzable_observations";
        case ProfitabilityPopulationState::invalid: return "invalid";
    }
    throw std::invalid_argument("invalid_profitability_population_state");
}

std::string ProfitabilityNormalizationStateText(
    ProfitabilityNormalizationState state)
{
    switch (state)
    {
        case ProfitabilityNormalizationState::available: return "available";
        case ProfitabilityNormalizationState::insufficientPopulation:
            return "insufficient_population";
        case ProfitabilityNormalizationState::zeroActionable:
            return "zero_actionable";
        case ProfitabilityNormalizationState::invalidPopulation:
            return "invalid_population";
    }
    throw std::invalid_argument("invalid_profitability_normalization_state");
}

ProfitabilityDistributionAnalysis AnalyzeProfitabilityDistribution(
    std::vector<ProfitabilityObservation> observations,
    const ProfitabilityNormalizationPolicy& policy)
{
    ProfitabilityDistributionAnalysis analysis;
    auto& summary = analysis.summary;
    if (const auto error = ValidateProfitabilityNormalizationPolicy(policy))
        throw std::invalid_argument(*error);
    summary.normalizationPolicyCanonical =
        ProfitabilityNormalizationPolicyCanonicalText(policy);
    summary.normalizationPolicyHash =
        ProfitabilityNormalizationPolicyHash(policy);
    summary.populationCount = observations.size();

    std::sort(observations.begin(), observations.end(),
        [](const auto& left, const auto& right) {
            return std::tie(left.observationIdentityCanonical,
                            left.profitabilityObservationId) <
                   std::tie(right.observationIdentityCanonical,
                            right.profitabilityObservationId);
        });
    summary.membershipCanonical = MembershipCanonical(observations);
    summary.membershipHash =
        RecommendationCanonicalHash(summary.membershipCanonical);

    std::string invalidReason;
    std::set<long long> observationIds;
    std::set<std::string> observationIdentities;
    std::optional<ProfitabilityPopulationIdentity> identity;
    for (const auto& observation : observations)
    {
        if (const auto error = ValidateProfitabilityObservation(observation))
        {
            invalidReason = *error;
            break;
        }
        if (!observationIds.insert(observation.profitabilityObservationId).second ||
            !observationIdentities.insert(
                observation.observationIdentityCanonical).second)
        {
            invalidReason = "duplicate_profitability_observation";
            break;
        }
        const auto memberIdentity = IdentityFor(observation);
        if (!identity) identity = memberIdentity;
        else if (*identity != memberIdentity)
        {
            invalidReason = "incompatible_profitability_population_identity";
            break;
        }
        if (!AddWithoutOverflow(observation.predictionCount,
                                summary.totalPredictionCount) ||
            !AddWithoutOverflow(observation.actionableCount,
                                summary.totalActionableCount))
        {
            invalidReason = "profitability_population_count_overflow";
            break;
        }
    }

    if (observations.empty()) invalidReason = "empty_profitability_population";
    if (!invalidReason.empty())
    {
        summary.state = ProfitabilityPopulationState::invalid;
        summary.reason = invalidReason;
        summary.totalPredictionCount = 0;
        summary.totalActionableCount = 0;
        for (const auto& observation : observations)
        {
            ProfitabilityNormalizationResult result;
            result.profitabilityObservationId =
                observation.profitabilityObservationId;
            result.state = ProfitabilityNormalizationState::invalidPopulation;
            result.reason = invalidReason;
            if (observation.averageTerminalHorizonLogReturnPerActionablePrediction &&
                std::isfinite(*observation.
                    averageTerminalHorizonLogReturnPerActionablePrediction))
                result.rawProfitabilityMetric = observation.
                    averageTerminalHorizonLogReturnPerActionablePrediction;
            result.membershipHash = summary.membershipHash;
            result.populationSize = observations.size();
            result.actionableCount = observation.actionableCount;
            result.canonical = NormalizationCanonical(result);
            result.hash = RecommendationCanonicalHash(result.canonical);
            analysis.normalizationResults.push_back(std::move(result));
        }
        summary.canonical = SummaryCanonical(summary);
        summary.hash = RecommendationCanonicalHash(summary.canonical);
    }
    else
    {
        summary.populationIdentity = identity;
        summary.populationIdentityCanonical =
            ProfitabilityPopulationIdentityCanonicalText(*identity);
        summary.populationIdentityHash =
            ProfitabilityPopulationIdentityHash(*identity);
        std::vector<double> values;
        values.reserve(observations.size());
        for (const auto& observation : observations)
        {
            if (observation.actionableCount == 0)
                ++summary.zeroActionableCount;
            else
            {
                const double value = *observation.
                    averageTerminalHorizonLogReturnPerActionablePrediction;
                values.push_back(value == 0.0 ? 0.0 : value);
                if (value < 0.0) ++summary.negativeCount;
                else if (value > 0.0) ++summary.positiveCount;
                else ++summary.zeroCount;
            }
        }
        std::sort(values.begin(), values.end());
        summary.analyzablePopulationCount = values.size();
        if (values.empty())
        {
            summary.state =
                ProfitabilityPopulationState::noAnalyzableObservations;
            summary.reason = "all_observations_zero_actionable";
        }
        else
        {
            summary.state = values.size() < policy.minimumAnalyzablePopulationSize
                ? ProfitabilityPopulationState::insufficientPopulation
                : ProfitabilityPopulationState::valid;
            summary.reason = summary.state == ProfitabilityPopulationState::valid
                ? "available" : "insufficient_analyzable_population";
            summary.minimum = values.front();
            summary.maximum = values.back();
            long double sum = 0.0L;
            for (double value : values) sum += static_cast<long double>(value);
            const long double mean = sum /
                static_cast<long double>(values.size());
            summary.mean = static_cast<double>(mean);
            summary.median = Quantile(values, 0.5);
            long double squared = 0.0L;
            for (double value : values)
            {
                const long double delta =
                    static_cast<long double>(value) - mean;
                squared += delta * delta;
            }
            summary.populationStandardDeviation =
                values.front() == values.back() ? 0.0 : static_cast<double>(
                    std::sqrt(squared /
                              static_cast<long double>(values.size())));
            std::vector<double> absoluteDeviations;
            absoluteDeviations.reserve(values.size());
            for (double value : values)
                absoluteDeviations.push_back(
                    std::abs(value - *summary.median));
            std::sort(absoluteDeviations.begin(), absoluteDeviations.end());
            summary.medianAbsoluteDeviation =
                values.front() == values.back() ? 0.0 :
                Quantile(absoluteDeviations, 0.5);
            std::vector<double> requestedQuantiles = policy.quantiles;
            std::sort(requestedQuantiles.begin(), requestedQuantiles.end());
            for (double probability : requestedQuantiles)
                summary.quantiles.push_back(
                    {probability, Quantile(values, probability)});
        }

        for (const auto& observation : observations)
        {
            ProfitabilityNormalizationResult result;
            result.profitabilityObservationId =
                observation.profitabilityObservationId;
            result.populationIdentityHash = summary.populationIdentityHash;
            result.membershipHash = summary.membershipHash;
            result.populationSize = observations.size();
            result.analyzablePopulationSize = values.size();
            result.actionableCount = observation.actionableCount;
            const long double support =
                static_cast<long double>(observation.actionableCount);
            const long double half = static_cast<long double>(
                policy.supportHalfSaturationActionableCount);
            result.supportReliability = static_cast<double>(
                support / (support + half));
            if (observation.actionableCount == 0)
            {
                result.state = ProfitabilityNormalizationState::zeroActionable;
                result.reason = "average_unavailable_zero_actionable";
            }
            else
            {
                const double raw = *observation.
                    averageTerminalHorizonLogReturnPerActionablePrediction;
                result.rawProfitabilityMetric = raw == 0.0 ? 0.0 : raw;
                const auto lower = std::lower_bound(values.begin(), values.end(),
                                                    raw);
                const auto upper = std::upper_bound(values.begin(), values.end(),
                                                    raw);
                const long double less = static_cast<long double>(
                    std::distance(values.begin(), lower));
                const long double equal = static_cast<long double>(
                    std::distance(lower, upper));
                const double percentile = static_cast<double>(
                    (less + 0.5L * equal) /
                    static_cast<long double>(values.size()));
                result.empiricalMidrankPercentile = percentile;
                if (raw < 0.0)
                    result.boundedCandidateMetric = 0.5 * percentile;
                else if (raw > 0.0)
                    result.boundedCandidateMetric = 0.5 + 0.5 * percentile;
                else
                    result.boundedCandidateMetric = 0.5;
                result.state =
                    summary.state == ProfitabilityPopulationState::valid
                    ? ProfitabilityNormalizationState::available
                    : ProfitabilityNormalizationState::insufficientPopulation;
                result.reason =
                    result.state == ProfitabilityNormalizationState::available
                    ? "available" : "insufficient_analyzable_population";
            }
            result.canonical = NormalizationCanonical(result);
            result.hash = RecommendationCanonicalHash(result.canonical);
            analysis.normalizationResults.push_back(std::move(result));
        }
        summary.canonical = SummaryCanonical(summary);
        summary.hash = RecommendationCanonicalHash(summary.canonical);
    }

    analysis.canonical = "profitability_distribution_analysis_v1;summary=" +
        LengthText(summary.canonical) + ";normalization_count=" +
        std::to_string(analysis.normalizationResults.size());
    for (std::size_t index = 0;
         index < analysis.normalizationResults.size(); ++index)
        analysis.canonical += ";normalization[" + std::to_string(index) +
            "]=" + LengthText(analysis.normalizationResults[index].canonical);
    analysis.canonical +=
        ";profitability_weight=0;profitability_score_contribution=0";
    analysis.hash = RecommendationCanonicalHash(analysis.canonical);
    return analysis;
}

std::optional<std::string> ValidateProfitabilityShadowNormalizationPolicy(
    const ProfitabilityShadowNormalizationPolicy& policy)
{
    if (policy.version != kProfitabilityShadowNormalizationPolicyVersion)
        return "unsupported_profitability_shadow_normalization_policy_version";
    if (policy.minimumAnalyzablePopulationSize < 2 ||
        policy.minimumAnalyzablePopulationSize > 10000)
        return "invalid_profitability_shadow_minimum_population";
    if (policy.supportHalfSaturationActionableCount == 0)
        return "invalid_profitability_shadow_support_half_saturation";
    return std::nullopt;
}

std::string ProfitabilityShadowNormalizationPolicyCanonicalText(
    const ProfitabilityShadowNormalizationPolicy& policy)
{
    if (const auto error =
            ValidateProfitabilityShadowNormalizationPolicy(policy))
        throw std::invalid_argument(*error);
    return "campaign_profitability_shadow_normalization_policy_v1;version=" +
        std::to_string(policy.version) +
        ";scope=source_ranking_snapshot_frozen_populated_observations;"
        "cross_context_population=explicit;"
        "raw_metric=average_terminal_horizon_log_return_per_actionable_prediction;"
        "empirical_percentile=midrank;"
        "bounded_metric=negative:0.5*percentile,zero:0.5,positive:0.5+0.5*percentile;"
        "support_reliability=actionable_count/(actionable_count+half_saturation);"
        "normalized_value=support_reliability*(2*bounded_metric-1);"
        "unavailable=excluded_from_population_and_no_contribution;"
        "zero_actionable=valid_zero_aggregate_average_undefined_no_contribution;"
        "minimum_analyzable_population_size=" +
        std::to_string(policy.minimumAnalyzablePopulationSize) +
        ";support_half_saturation_actionable_count=" +
        std::to_string(policy.supportHalfSaturationActionableCount) +
        ";base_phase3c_normalization_policy_hash=" +
        ProfitabilityNormalizationPolicyHash({});
}

std::string ProfitabilityShadowNormalizationPolicyHash(
    const ProfitabilityShadowNormalizationPolicy& policy)
{
    return RecommendationCanonicalHash(
        ProfitabilityShadowNormalizationPolicyCanonicalText(policy));
}

std::string ProfitabilityShadowNormalizationStateText(
    ProfitabilityShadowNormalizationState state)
{
    switch (state)
    {
        case ProfitabilityShadowNormalizationState::available:
            return "available";
        case ProfitabilityShadowNormalizationState::insufficientPopulation:
            return "insufficient_population";
        case ProfitabilityShadowNormalizationState::zeroActionable:
            return "zero_actionable";
    }
    throw std::logic_error("unknown_profitability_shadow_normalization_state");
}

ProfitabilityShadowNormalizationAnalysis
AnalyzeProfitabilityShadowNormalization(
    std::vector<ProfitabilityShadowNormalizationInput> inputs,
    const ProfitabilityShadowNormalizationPolicy& policy)
{
    if (const auto error =
            ValidateProfitabilityShadowNormalizationPolicy(policy))
        throw std::invalid_argument(*error);

    std::sort(inputs.begin(), inputs.end(), [](const auto& left,
                                               const auto& right) {
        return std::tie(left.profitabilityObservationId,
                        left.evidenceIdentityHash) <
            std::tie(right.profitabilityObservationId,
                     right.evidenceIdentityHash);
    });
    std::set<long long> observationIds;
    std::set<std::string> evidenceHashes;
    std::vector<double> values;
    values.reserve(inputs.size());
    for (const auto& input : inputs)
    {
        if (input.profitabilityObservationId <= 0 ||
            input.evidenceIdentityHash.empty() ||
            !observationIds.insert(input.profitabilityObservationId).second ||
            !evidenceHashes.insert(input.evidenceIdentityHash).second)
            throw std::invalid_argument(
                "invalid_profitability_shadow_normalization_input_identity");
        if (input.actionableCount == 0)
        {
            if (input.averageTerminalHorizonLogReturnPerActionablePrediction)
                throw std::invalid_argument(
                    "invalid_profitability_shadow_zero_actionable_average");
            continue;
        }
        if (!input.averageTerminalHorizonLogReturnPerActionablePrediction ||
            !std::isfinite(*input.
                averageTerminalHorizonLogReturnPerActionablePrediction))
            throw std::invalid_argument(
                "invalid_profitability_shadow_average_metric");
        const double value = *input.
            averageTerminalHorizonLogReturnPerActionablePrediction;
        values.push_back(value == 0.0 ? 0.0 : value);
    }
    std::sort(values.begin(), values.end());

    ProfitabilityShadowNormalizationAnalysis analysis;
    analysis.policyCanonical =
        ProfitabilityShadowNormalizationPolicyCanonicalText(policy);
    analysis.policyHash =
        ProfitabilityShadowNormalizationPolicyHash(policy);
    analysis.populatedEvidenceCount = inputs.size();
    analysis.analyzableEvidenceCount = values.size();
    analysis.zeroActionableCount = inputs.size() - values.size();
    analysis.membershipCanonical =
        "campaign_profitability_shadow_normalization_membership_v1;count=" +
        std::to_string(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index)
    {
        const auto& input = inputs[index];
        analysis.membershipCanonical += ";member[" +
            std::to_string(index) + "].observation_id=" +
            std::to_string(input.profitabilityObservationId) +
            ";member[" + std::to_string(index) + "].actionable_count=" +
            std::to_string(input.actionableCount) +
            ";member[" + std::to_string(index) + "].average=" +
            OptionalDoubleText(input.
                averageTerminalHorizonLogReturnPerActionablePrediction) +
            ";member[" + std::to_string(index) + "].evidence_hash=" +
            input.evidenceIdentityHash;
    }
    analysis.membershipHash =
        RecommendationCanonicalHash(analysis.membershipCanonical);

    const bool sufficient = values.size() >=
        policy.minimumAnalyzablePopulationSize;
    for (const auto& input : inputs)
    {
        ProfitabilityShadowNormalizationResult result;
        result.profitabilityObservationId = input.profitabilityObservationId;
        const long double support =
            static_cast<long double>(input.actionableCount);
        const long double half = static_cast<long double>(
            policy.supportHalfSaturationActionableCount);
        result.supportReliability = static_cast<double>(
            support / (support + half));
        if (input.actionableCount == 0)
        {
            result.state =
                ProfitabilityShadowNormalizationState::zeroActionable;
            result.reason =
                "average_unavailable_zero_actionable_no_contribution";
        }
        else
        {
            const double raw = *input.
                averageTerminalHorizonLogReturnPerActionablePrediction;
            result.rawProfitabilityMetric = raw == 0.0 ? 0.0 : raw;
            const auto lower = std::lower_bound(values.begin(), values.end(), raw);
            const auto upper = std::upper_bound(values.begin(), values.end(), raw);
            const long double less = static_cast<long double>(
                std::distance(values.begin(), lower));
            const long double equal = static_cast<long double>(
                std::distance(lower, upper));
            const double percentile = static_cast<double>(
                (less + 0.5L * equal) /
                static_cast<long double>(values.size()));
            result.empiricalMidrankPercentile = percentile;
            const double bounded = raw < 0.0 ? 0.5 * percentile
                : raw > 0.0 ? 0.5 + 0.5 * percentile : 0.5;
            result.boundedCandidateMetric = bounded;
            result.state = sufficient
                ? ProfitabilityShadowNormalizationState::available
                : ProfitabilityShadowNormalizationState::insufficientPopulation;
            result.reason = sufficient ? "available"
                : "insufficient_analyzable_population_no_contribution";
            if (sufficient)
                result.normalizedProfitabilityValue =
                    result.supportReliability * (2.0 * bounded - 1.0);
        }
        result.canonical =
            "campaign_profitability_shadow_normalization_result_v1;"
            "observation_id=" + std::to_string(result.profitabilityObservationId) +
            ";state=" + ProfitabilityShadowNormalizationStateText(result.state) +
            ";reason=" + LengthText(result.reason) +
            ";raw_metric=" + OptionalDoubleText(result.rawProfitabilityMetric) +
            ";empirical_midrank_percentile=" +
            OptionalDoubleText(result.empiricalMidrankPercentile) +
            ";bounded_candidate_metric=" +
            OptionalDoubleText(result.boundedCandidateMetric) +
            ";support_reliability=" +
            CanonicalRecommendationDouble(result.supportReliability) +
            ";normalized_profitability_value=" +
            OptionalDoubleText(result.normalizedProfitabilityValue) +
            ";policy_hash=" + analysis.policyHash +
            ";membership_hash=" + analysis.membershipHash;
        result.hash = RecommendationCanonicalHash(result.canonical);
        analysis.results.push_back(std::move(result));
    }
    analysis.canonical =
        "campaign_profitability_shadow_normalization_analysis_v1;policy=" +
        LengthText(analysis.policyCanonical) +
        ";membership=" + LengthText(analysis.membershipCanonical) +
        ";populated_evidence_count=" +
        std::to_string(analysis.populatedEvidenceCount) +
        ";analyzable_evidence_count=" +
        std::to_string(analysis.analyzableEvidenceCount) +
        ";zero_actionable_count=" +
        std::to_string(analysis.zeroActionableCount);
    for (std::size_t index = 0; index < analysis.results.size(); ++index)
        analysis.canonical += ";result[" + std::to_string(index) + "]=" +
            LengthText(analysis.results[index].canonical);
    analysis.hash = RecommendationCanonicalHash(analysis.canonical);
    return analysis;
}

} // namespace EA::ExperimentRecommendation
