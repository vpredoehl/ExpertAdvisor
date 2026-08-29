#include "ProfitabilityVerification.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace EA::ProfitabilityVerification
{

std::string OutcomeJobReadinessText(OutcomeJobReadiness value)
{
    switch (value)
    {
        case OutcomeJobReadiness::waitingForOutcomeData:
            return "waiting_for_outcome_data";
        case OutcomeJobReadiness::partiallyAvailable:
            return "partially_available";
        case OutcomeJobReadiness::readyToExecute:
            return "ready_to_execute";
        case OutcomeJobReadiness::incompatibleSource:
            return "incompatible_source";
    }
    throw std::logic_error("unknown_campaign_profitability_outcome_readiness");
}
namespace
{

void AppendField(std::string& output,
                 std::string_view name,
                 const std::string& value)
{
    output.append(name);
    output.push_back('=');
    output.append(std::to_string(value.size()));
    output.push_back(':');
    output.append(value);
    output.push_back(';');
}

std::string OptionalId(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalDouble(const std::optional<double>& value)
{
    if (!value) return "NULL";
    if (!std::isfinite(*value)) return "NONFINITE";
    std::array<char, 128> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(),
        *value == 0.0 ? 0.0 : *value, std::chars_format::general);
    if (converted.ec != std::errc{})
        throw std::runtime_error("profitability_number_format_failed");
    return std::string(buffer.data(), converted.ptr);
}

bool TaggedHash(const std::string& value)
{
    if (value.size() != 24 || !value.starts_with("fnv1a64:")) return false;
    return std::all_of(value.begin() + 8, value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

bool NearlyEqual(double left, double right)
{
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <=
        32.0 * std::numeric_limits<double>::epsilon() * scale;
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

EvidenceResult Result(const ExpectedFinalEvidence& expected,
                      EvidenceState state,
                      std::string reason,
                      const std::optional<InferenceProfitability::Observation>&
                          observation = std::nullopt)
{
    EvidenceResult result;
    result.experimentId = expected.experimentId;
    if (expected.modelId > 0) result.finalModelId = expected.modelId;
    if (expected.inferenceEvalResultId > 0)
        result.finalInferenceEvalResultId = expected.inferenceEvalResultId;
    result.observation = observation;
    result.state = state;
    result.reason = std::move(reason);

    std::string canonical = "profitability_exact_final_evidence_v1;";
    AppendField(canonical, "contract_version",
                std::to_string(kEvidenceContractVersion));
    AppendField(canonical, "experiment_id",
                std::to_string(expected.experimentId));
    AppendField(canonical, "model_id",
                expected.modelId > 0 ? std::to_string(expected.modelId) : "NULL");
    AppendField(canonical, "inference_eval_result_id",
                expected.inferenceEvalResultId > 0
                    ? std::to_string(expected.inferenceEvalResultId) : "NULL");
    AppendField(canonical, "inference_start",
                expected.inferenceStart.empty() ? "NULL" : expected.inferenceStart);
    AppendField(canonical, "inference_end",
                expected.inferenceEnd.empty() ? "NULL" : expected.inferenceEnd);
    AppendField(canonical, "state", EvidenceStateText(state));
    AppendField(canonical, "reason", result.reason);
    AppendField(canonical, "observation_identity",
                observation ? observation->observationIdentityCanonical : "NULL");
    result.evidenceIdentityCanonical = std::move(canonical);
    result.evidenceIdentityHash = InferenceProfitability::DeterministicHash(
        result.evidenceIdentityCanonical);
    return result;
}

std::optional<std::string> ValidateValues(
    const InferenceProfitability::Observation& observation)
{
    const auto& statistics = observation.statistics;
    const auto average =
        observation.averageTerminalHorizonLogReturnPerActionablePrediction;
    if (statistics.actionableCount > statistics.predictionCount ||
        statistics.winningActionableCount > statistics.actionableCount ||
        statistics.losingActionableCount > statistics.actionableCount ||
        statistics.winningActionableCount >
            statistics.actionableCount - statistics.losingActionableCount)
        return "invalid_profitability_counts";
    if (!std::isfinite(statistics.grossPositiveTerminalHorizonLogReturnSum) ||
        !std::isfinite(statistics.grossNegativeTerminalHorizonLogReturnSum) ||
        !std::isfinite(statistics.aggregateTerminalHorizonLogReturnSum) ||
        (average && !std::isfinite(*average)))
        return "nonfinite_profitability_value";
    if (statistics.grossPositiveTerminalHorizonLogReturnSum < 0.0 ||
        statistics.grossNegativeTerminalHorizonLogReturnSum > 0.0 ||
        !NearlyEqualAccumulated(
            statistics.aggregateTerminalHorizonLogReturnSum,
            statistics.grossPositiveTerminalHorizonLogReturnSum +
                statistics.grossNegativeTerminalHorizonLogReturnSum,
            statistics.actionableCount))
        return "inconsistent_profitability_returns";
    if (statistics.actionableCount == 0)
    {
        if (average || statistics.winningActionableCount != 0 ||
            statistics.losingActionableCount != 0 ||
            statistics.grossPositiveTerminalHorizonLogReturnSum != 0.0 ||
            statistics.grossNegativeTerminalHorizonLogReturnSum != 0.0 ||
            statistics.aggregateTerminalHorizonLogReturnSum != 0.0)
            return "invalid_zero_actionable_profitability_shape";
    }
    else
    {
        if (!average) return "missing_profitability_average";
        const double expected = statistics.aggregateTerminalHorizonLogReturnSum /
            static_cast<double>(statistics.actionableCount);
        if (!NearlyEqual(expected, *average))
            return "inconsistent_profitability_average";
    }
    return std::nullopt;
}

ProfitabilitySign Sign(const EvidenceResult& result)
{
    if (EvidenceStateIsInvalid(result.state)) return ProfitabilitySign::invalid;
    if (result.state != EvidenceState::valid || !result.observation)
        return ProfitabilitySign::unavailable;
    const auto& observation = *result.observation;
    if (observation.statistics.actionableCount == 0)
        return ProfitabilitySign::zeroActionable;
    const double value =
        *observation.averageTerminalHorizonLogReturnPerActionablePrediction;
    return value > 0.0 ? ProfitabilitySign::positive
         : value < 0.0 ? ProfitabilitySign::negative
                       : ProfitabilitySign::zero;
}

int StatePriority(const ShadowCandidate& candidate)
{
    switch (candidate.profitabilitySign)
    {
        case ProfitabilitySign::positive: return 0;
        case ProfitabilitySign::zero: return 1;
        case ProfitabilitySign::negative: return 2;
        case ProfitabilitySign::zeroActionable: return 3;
        case ProfitabilitySign::unavailable: return 4;
        case ProfitabilitySign::invalid: return 5;
    }
    throw std::logic_error("unknown_profitability_sign");
}

double AverageOrNegativeInfinity(const ShadowCandidate& candidate)
{
    if (!candidate.profitability.observation ||
        !candidate.profitability.observation
             ->averageTerminalHorizonLogReturnPerActionablePrediction)
        return -std::numeric_limits<double>::infinity();
    return *candidate.profitability.observation
        ->averageTerminalHorizonLogReturnPerActionablePrediction;
}

double AggregateOrNegativeInfinity(const ShadowCandidate& candidate)
{
    if (!candidate.profitability.observation)
        return -std::numeric_limits<double>::infinity();
    return candidate.profitability.observation->statistics
        .aggregateTerminalHorizonLogReturnSum;
}

std::uint64_t Actionable(const ShadowCandidate& candidate)
{
    return candidate.profitability.observation
        ? candidate.profitability.observation->statistics.actionableCount : 0;
}

EvidenceResult ContractFailure(EvidenceResult result, std::string reason)
{
    result.state = EvidenceState::invalidProvenance;
    result.reason = std::move(reason);
    AppendField(result.evidenceIdentityCanonical,
                "campaign_contract_failure", result.reason);
    result.evidenceIdentityHash = InferenceProfitability::DeterministicHash(
        result.evidenceIdentityCanonical);
    return result;
}

bool FrozenEvidenceMatches(
    const ExperimentRecommendation::RecommendationSource::
        FinalProfitabilityEvidence& frozen,
    const EvidenceResult& current)
{
    if (current.state == EvidenceState::valid && current.observation)
    {
        const auto& observation = *current.observation;
        return frozen.Available() &&
            frozen.finalInferenceEvalResultId ==
                current.finalInferenceEvalResultId &&
            frozen.profitabilityObservationId == observation.observationId &&
            frozen.inferenceScope == "final" &&
            frozen.inferenceStart == observation.provenance.inferenceStart &&
            frozen.inferenceEnd == observation.provenance.inferenceEnd &&
            frozen.actionablePredictionCount ==
                static_cast<long long>(observation.statistics.actionableCount) &&
            frozen.aggregateTerminalHorizonLogReturnSum ==
                observation.statistics.aggregateTerminalHorizonLogReturnSum &&
            frozen.averageTerminalHorizonLogReturnPerActionablePrediction ==
                observation
                    .averageTerminalHorizonLogReturnPerActionablePrediction &&
            frozen.metricDefinitionHash == observation.metricDefinitionHash &&
            frozen.sourceContentHash == observation.sourceContentHash &&
            frozen.observationIdentityHash ==
                observation.observationIdentityHash &&
            frozen.unavailableReason.empty();
    }
    if (current.state == EvidenceState::unavailable ||
        current.state == EvidenceState::incomplete)
    {
        return !frozen.Available() && frozen.inferenceScope == "final" &&
            frozen.finalInferenceEvalResultId ==
                current.finalInferenceEvalResultId &&
            frozen.unavailableReason == current.reason;
    }
    return false;
}

} // namespace

std::string EvidenceStateText(EvidenceState state)
{
    switch (state)
    {
        case EvidenceState::valid: return "valid";
        case EvidenceState::unavailable: return "unavailable";
        case EvidenceState::incomplete: return "incomplete";
        case EvidenceState::ambiguous: return "ambiguous";
        case EvidenceState::invalidProvenance: return "invalid_provenance";
        case EvidenceState::invalidMetricDefinition:
            return "invalid_metric_definition";
        case EvidenceState::invalidValues: return "invalid_values";
    }
    throw std::logic_error("unknown_profitability_evidence_state");
}

bool EvidenceStateIsInvalid(EvidenceState state)
{
    return state == EvidenceState::ambiguous ||
        state == EvidenceState::invalidProvenance ||
        state == EvidenceState::invalidMetricDefinition ||
        state == EvidenceState::invalidValues;
}

EvidenceResult ValidateExactFinalObservation(
    const ExpectedFinalEvidence& expected,
    const std::optional<InferenceProfitability::Observation>& observation)
{
    if (expected.experimentId <= 0 || expected.modelId <= 0 ||
        expected.inferenceEvalResultId <= 0 || expected.inferenceStart.empty() ||
        expected.inferenceEnd.empty())
        return Result(expected, EvidenceState::incomplete,
                      "incomplete_exact_final_expectation");
    if (!observation)
        return Result(expected, EvidenceState::unavailable,
                      "no_profitability_observation");
    const auto& value = *observation;
    if (value.observationId <= 0 ||
        value.provenance.experimentId !=
            std::optional<long long>{expected.experimentId} ||
        value.provenance.modelId != expected.modelId ||
        value.provenance.inferenceEvalResultId !=
            expected.inferenceEvalResultId ||
        value.provenance.scope !=
            InferenceProfitability::Scope::finalInference ||
        value.provenance.checkpointEvalId.has_value() ||
        value.provenance.inferenceStart != expected.inferenceStart ||
        value.provenance.inferenceEnd != expected.inferenceEnd)
        return Result(expected, EvidenceState::invalidProvenance,
                      "exact_final_profitability_provenance_mismatch", value);
    if (value.metricDefinitionCanonical !=
            InferenceProfitability::kMetricDefinitionCanonical ||
        value.metricDefinitionHash !=
            InferenceProfitability::MetricDefinitionHash() ||
        InferenceProfitability::DeterministicHash(
            value.metricDefinitionCanonical) != value.metricDefinitionHash)
        return Result(expected, EvidenceState::invalidMetricDefinition,
                      "profitability_metric_definition_mismatch", value);
    if (const auto invalid = ValidateValues(value))
        return Result(expected, EvidenceState::invalidValues, *invalid, value);
    if (!TaggedHash(value.sourceContentHash) ||
        !TaggedHash(value.observationIdentityHash))
        return Result(expected, EvidenceState::invalidProvenance,
                      "profitability_observation_identity_invalid", value);
    InferenceProfitability::ObservationRequest identityRequest;
    identityRequest.provenance = value.provenance;
    identityRequest.statistics = value.statistics;
    identityRequest.sourceContentHash = value.sourceContentHash;
    identityRequest.metricDefinitionCanonical =
        value.metricDefinitionCanonical;
    const std::string expectedObservationIdentity =
        InferenceProfitability::BuildObservationIdentityCanonical(
            identityRequest);
    if (value.observationIdentityCanonical != expectedObservationIdentity ||
        InferenceProfitability::DeterministicHash(
            value.observationIdentityCanonical) != value.observationIdentityHash)
        return Result(expected, EvidenceState::invalidProvenance,
                      "profitability_observation_identity_invalid", value);
    return Result(expected, EvidenceState::valid,
                  value.statistics.actionableCount == 0
                      ? "valid_zero_actionable" : "valid", value);
}

EvidenceResult EnforceFrozenCampaignEvidence(
    const std::optional<long long>& candidateSourceModelId,
    const std::optional<ExperimentRecommendation::RecommendationSource::
        FinalProfitabilityEvidence>& frozen,
    EvidenceResult current)
{
    if (!candidateSourceModelId ||
        current.finalModelId != candidateSourceModelId)
        return ContractFailure(
            std::move(current), "candidate_source_model_mismatch");
    if (!frozen)
        return ContractFailure(
            std::move(current),
            "candidate_profitability_contract_unavailable");
    if (!FrozenEvidenceMatches(*frozen, current))
        return ContractFailure(
            std::move(current),
            "candidate_frozen_profitability_evidence_mismatch");
    return current;
}

EvidenceResult ValidateFrozenCampaignEvidence(
    long long sourceExperimentId,
    const std::optional<long long>& sourceModelId,
    const ExperimentRecommendation::RecommendationSource::
        FinalProfitabilityEvidence& frozen,
    const std::optional<InferenceProfitability::Observation>& observation)
{
    if (sourceExperimentId <= 0 ||
        ExperimentRecommendation::ValidateRecommendationFinalProfitabilityEvidence(
            frozen))
        throw std::invalid_argument("invalid_frozen_campaign_profitability_evidence");
    if (frozen.Available())
    {
        if (!sourceModelId || !observation ||
            !frozen.finalInferenceEvalResultId || !frozen.inferenceStart ||
            !frozen.inferenceEnd)
            return ContractFailure(EvidenceResult{},
                "populated_frozen_profitability_reference_incomplete");
        ExpectedFinalEvidence expected{
            sourceExperimentId, *sourceModelId,
            *frozen.finalInferenceEvalResultId,
            *frozen.inferenceStart, *frozen.inferenceEnd};
        return EnforceFrozenCampaignEvidence(
            sourceModelId, frozen,
            ValidateExactFinalObservation(expected, observation));
    }
    if (observation)
        return ContractFailure(EvidenceResult{},
            "unavailable_frozen_profitability_has_observation");
    EvidenceResult result;
    result.experimentId = sourceExperimentId;
    result.finalModelId = sourceModelId;
    result.finalInferenceEvalResultId = frozen.finalInferenceEvalResultId;
    result.state = EvidenceState::unavailable;
    result.reason = frozen.unavailableReason;
    result.evidenceIdentityCanonical =
        "campaign_frozen_profitability_evidence_v1;";
    AppendField(result.evidenceIdentityCanonical, "experiment_id",
                std::to_string(sourceExperimentId));
    AppendField(result.evidenceIdentityCanonical, "model_id",
                OptionalId(sourceModelId));
    AppendField(result.evidenceIdentityCanonical, "inference_result_id",
                OptionalId(frozen.finalInferenceEvalResultId));
    AppendField(result.evidenceIdentityCanonical, "state", "unavailable");
    AppendField(result.evidenceIdentityCanonical, "reason",
                frozen.unavailableReason);
    result.evidenceIdentityHash = InferenceProfitability::DeterministicHash(
        result.evidenceIdentityCanonical);
    return result;
}

std::vector<long long> ParseDeclaredExperimentIds(const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument("profitability_experiment_set_empty");
    std::vector<long long> result;
    std::set<long long> seen;
    std::size_t begin = 0;
    while (begin <= value.size())
    {
        const std::size_t end = value.find(',', begin);
        const std::string_view token{
            value.data() + begin,
            (end == std::string::npos ? value.size() : end) - begin};
        long long experimentId = 0;
        const auto parsed = std::from_chars(
            token.data(), token.data() + token.size(), experimentId);
        if (token.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != token.data() + token.size() || experimentId <= 0)
            throw std::invalid_argument(
                "invalid_profitability_experiment_id_set");
        if (!seen.insert(experimentId).second)
            throw std::invalid_argument(
                "duplicate_profitability_experiment_id");
        result.push_back(experimentId);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

int ExitCode(const std::vector<EvidenceResult>& results)
{
    if (std::any_of(results.begin(), results.end(), [](const auto& value) {
            return EvidenceStateIsInvalid(value.state); }))
        return 3;
    if (std::any_of(results.begin(), results.end(), [](const auto& value) {
            return value.state != EvidenceState::valid; }))
        return 4;
    return 0;
}

std::string ProfitabilitySignText(ProfitabilitySign sign)
{
    switch (sign)
    {
        case ProfitabilitySign::positive: return "positive";
        case ProfitabilitySign::zero: return "zero";
        case ProfitabilitySign::negative: return "negative";
        case ProfitabilitySign::zeroActionable: return "zero_actionable";
        case ProfitabilitySign::unavailable: return "unavailable";
        case ProfitabilitySign::invalid: return "invalid";
    }
    throw std::logic_error("unknown_profitability_sign");
}

std::string ShadowRankingPolicyCanonicalText()
{
    return "campaign_profitability_shadow_ranking_policy_v1;version=1;"
        "evidence=exact_final_only;checkpoint_substitution=forbidden;"
        "order=valid_positive,valid_zero,valid_negative,zero_actionable,"
        "unavailable,invalid;within_valid=average_return_desc,"
        "aggregate_return_desc,actionable_count_desc;"
        "tie_break=current_rank_asc,ranking_member_id_asc;"
        "missing_is_not_zero=true;invalid_is_not_negative=true;"
        "live_profitability_weight=0;live_profitability_score_contribution=0;"
        "activation=disabled";
}

std::string ShadowRankingPolicyHash()
{
    return InferenceProfitability::DeterministicHash(
        ShadowRankingPolicyCanonicalText());
}

ShadowRanking BuildShadowRanking(std::vector<ShadowCandidate> candidates)
{
    for (auto& candidate : candidates)
    {
        if (candidate.rankingMemberId <= 0 || candidate.recommendationId <= 0 ||
            candidate.sourceExperimentId <= 0 || candidate.currentRank <= 0 ||
            !std::isfinite(candidate.leaderScore) ||
            !std::isfinite(candidate.inferenceAccuracy) ||
            (candidate.currentScore && !std::isfinite(*candidate.currentScore)) ||
            (candidate.predictedNeutralProportion &&
             !std::isfinite(*candidate.predictedNeutralProportion)))
            throw std::invalid_argument("invalid_profitability_shadow_candidate");
        candidate.profitabilitySign = Sign(candidate.profitability);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const ShadowCandidate& left, const ShadowCandidate& right) {
        const int leftState = StatePriority(left);
        const int rightState = StatePriority(right);
        if (leftState != rightState) return leftState < rightState;
        if (leftState <= 2)
        {
            const double leftAverage = AverageOrNegativeInfinity(left);
            const double rightAverage = AverageOrNegativeInfinity(right);
            if (leftAverage != rightAverage) return leftAverage > rightAverage;
            const double leftAggregate = AggregateOrNegativeInfinity(left);
            const double rightAggregate = AggregateOrNegativeInfinity(right);
            if (leftAggregate != rightAggregate)
                return leftAggregate > rightAggregate;
            if (Actionable(left) != Actionable(right))
                return Actionable(left) > Actionable(right);
        }
        return std::tie(left.currentRank, left.rankingMemberId) <
            std::tie(right.currentRank, right.rankingMemberId);
    });

    ShadowRanking ranking;
    ranking.policyCanonical = ShadowRankingPolicyCanonicalText();
    ranking.policyHash = ShadowRankingPolicyHash();
    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        auto& candidate = candidates[index];
        candidate.profitabilityShadowRank = static_cast<int>(index + 1);
        candidate.rankDelta =
            candidate.currentRank - candidate.profitabilityShadowRank;
        std::string canonical = "campaign_profitability_shadow_candidate_v1;";
        AppendField(canonical, "ranking_member_id",
                    std::to_string(candidate.rankingMemberId));
        AppendField(canonical, "recommendation_id",
                    std::to_string(candidate.recommendationId));
        AppendField(canonical, "source_experiment_id",
                    std::to_string(candidate.sourceExperimentId));
        AppendField(canonical, "source_model_id",
                    OptionalId(candidate.sourceModelId));
        AppendField(canonical, "current_rank",
                    std::to_string(candidate.currentRank));
        AppendField(canonical, "current_score",
                    OptionalDouble(candidate.currentScore));
        AppendField(canonical, "evidence_identity_hash",
                    candidate.profitability.evidenceIdentityHash);
        AppendField(canonical, "profitability_sign",
                    ProfitabilitySignText(candidate.profitabilitySign));
        AppendField(canonical, "shadow_rank",
                    std::to_string(candidate.profitabilityShadowRank));
        AppendField(canonical, "rank_delta",
                    std::to_string(candidate.rankDelta));
        candidate.canonical = std::move(canonical);
        candidate.hash = InferenceProfitability::DeterministicHash(
            candidate.canonical);
    }
    ranking.candidates = std::move(candidates);
    ranking.canonical = "campaign_profitability_shadow_ranking_v1;";
    AppendField(ranking.canonical, "policy", ranking.policyCanonical);
    AppendField(ranking.canonical, "candidate_count",
                std::to_string(ranking.candidates.size()));
    for (std::size_t index = 0; index < ranking.candidates.size(); ++index)
        AppendField(ranking.canonical,
                    "candidate[" + std::to_string(index) + "]",
                    ranking.candidates[index].canonical);
    ranking.hash = InferenceProfitability::DeterministicHash(ranking.canonical);
    return ranking;
}

std::vector<double> ParseProfitabilityShadowWeights(const std::string& value)
{
    if (value.empty())
        throw std::invalid_argument("profitability_shadow_weights_empty");
    std::vector<double> weights;
    std::set<double> unique;
    std::size_t begin = 0;
    while (begin <= value.size())
    {
        const std::size_t comma = value.find(',', begin);
        const std::string_view token{value.data() + begin,
            (comma == std::string::npos ? value.size() : comma) - begin};
        if (token.empty())
            throw std::invalid_argument("malformed_profitability_shadow_weight");
        double weight = 0.0;
        const auto parsed = std::from_chars(
            token.data(), token.data() + token.size(), weight,
            std::chars_format::general);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() ||
            !std::isfinite(weight))
            throw std::invalid_argument("malformed_profitability_shadow_weight");
        if (weight < 0.0)
            throw std::invalid_argument("negative_profitability_shadow_weight");
        if (weight > kMaximumPhase9ProfitabilityShadowWeight)
            throw std::invalid_argument(
                "profitability_shadow_weight_exceeds_phase9_upper_bound");
        if (!unique.insert(weight == 0.0 ? 0.0 : weight).second)
            throw std::invalid_argument("duplicate_profitability_shadow_weight");
        weights.push_back(weight == 0.0 ? 0.0 : weight);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    std::sort(weights.begin(), weights.end());
    return weights;
}

WeightedShadowRanking BuildWeightedShadowRanking(
    std::vector<ShadowCandidate> candidates,
    long long controlSnapshotId,
    long long sourceEvaluationRunId,
    const std::string& controlSnapshotIdentityHash,
    double shadowWeight,
    const ExperimentRecommendation::ProfitabilityShadowNormalizationPolicy&
        normalizationPolicy)
{
    namespace Recommendation = EA::ExperimentRecommendation;
    if (controlSnapshotId <= 0 || sourceEvaluationRunId <= 0 ||
        !TaggedHash(controlSnapshotIdentityHash) || !std::isfinite(shadowWeight) ||
        shadowWeight < 0.0 ||
        shadowWeight > kMaximumPhase9ProfitabilityShadowWeight)
        throw std::invalid_argument("invalid_weighted_profitability_shadow_policy");

    std::set<long long> memberIds;
    std::set<long long> recommendationIds;
    std::set<long long> evaluationIds;
    std::set<int> ranks;
    std::map<long long,
             Recommendation::ProfitabilityShadowNormalizationInput>
        normalizationInputByObservation;
    std::map<std::string, long long> observationByEvidenceIdentity;
    std::vector<Recommendation::ProfitabilityShadowNormalizationInput>
        normalizationInputs;
    for (auto& candidate : candidates)
    {
        if (candidate.rankingMemberId <= 0 || candidate.recommendationId <= 0 ||
            candidate.recommendationEvaluationResultId <= 0 ||
            candidate.recommendationEvaluationRunId != sourceEvaluationRunId ||
            candidate.sourceExperimentId <= 0 || candidate.currentRank <= 0 ||
            candidate.symbol.empty() || candidate.horizon <= 0 ||
            !candidate.currentScore || !std::isfinite(*candidate.currentScore) ||
            !memberIds.insert(candidate.rankingMemberId).second ||
            !recommendationIds.insert(candidate.recommendationId).second ||
            !evaluationIds.insert(candidate.recommendationEvaluationResultId).second ||
            !ranks.insert(candidate.currentRank).second)
            throw std::invalid_argument(
                "invalid_weighted_profitability_shadow_candidate");
        candidate.profitabilitySign = Sign(candidate.profitability);
        if (EvidenceStateIsInvalid(candidate.profitability.state))
            throw std::invalid_argument(
                "invalid_profitability_evidence_blocks_weighted_shadow");
        if (candidate.profitability.state == EvidenceState::valid &&
            candidate.profitability.observation)
        {
            const auto& observation = *candidate.profitability.observation;
            Recommendation::ProfitabilityShadowNormalizationInput input{
                observation.observationId,
                observation.statistics.actionableCount,
                observation.averageTerminalHorizonLogReturnPerActionablePrediction,
                candidate.profitability.evidenceIdentityHash};
            const auto [identity, identityInserted] =
                observationByEvidenceIdentity.emplace(
                    input.evidenceIdentityHash, input.profitabilityObservationId);
            const auto [existing, observationInserted] =
                normalizationInputByObservation.emplace(
                    input.profitabilityObservationId, input);
            if ((!identityInserted && identity->second !=
                    input.profitabilityObservationId) ||
                (!observationInserted &&
                    (existing->second.actionableCount != input.actionableCount ||
                     existing->second.
                         averageTerminalHorizonLogReturnPerActionablePrediction !=
                         input.
                         averageTerminalHorizonLogReturnPerActionablePrediction ||
                     existing->second.evidenceIdentityHash !=
                         input.evidenceIdentityHash)))
                throw std::invalid_argument(
                    "conflicting_repeated_profitability_shadow_evidence");
        }
    }
    if (ranks.size() != candidates.size() ||
        (!ranks.empty() && (*ranks.begin() != 1 ||
         *ranks.rbegin() != static_cast<int>(ranks.size()))))
        throw std::invalid_argument(
            "noncontiguous_weighted_profitability_control_ranks");
    normalizationInputs.reserve(normalizationInputByObservation.size());
    for (auto& [observationId, input] : normalizationInputByObservation)
    {
        (void)observationId;
        normalizationInputs.push_back(std::move(input));
    }

    WeightedShadowRanking ranking;
    ranking.controlSnapshotId = controlSnapshotId;
    ranking.sourceEvaluationRunId = sourceEvaluationRunId;
    ranking.shadowWeight = shadowWeight == 0.0 ? 0.0 : shadowWeight;
    ranking.normalization = Recommendation::AnalyzeProfitabilityShadowNormalization(
        std::move(normalizationInputs), normalizationPolicy);
    ranking.policyCanonical =
        "campaign_profitability_weighted_shadow_ranking_policy_v1;version=1;"
        "shadow_only=true;activation=disabled;database_write=false;"
        "control_snapshot_id=" + std::to_string(controlSnapshotId) +
        ";control_snapshot_identity_hash=" + controlSnapshotIdentityHash +
        ";source_evaluation_run_id=" + std::to_string(sourceEvaluationRunId) +
        ";control_score=ranking_member_final_score;"
        "shadow_score=control_score+profitability_contribution;"
        "raw_metric=average_terminal_horizon_log_return_per_actionable_prediction;"
        "profitability_contribution=shadow_weight*normalized_profitability_value;"
        "shadow_weight=" + OptionalDouble(ranking.shadowWeight) +
        ";phase9_shadow_weight_upper_bound=" +
        OptionalDouble(kMaximumPhase9ProfitabilityShadowWeight) +
        ";normalization_policy_hash=" + ranking.normalization.policyHash +
        ";normalization_membership_hash=" + ranking.normalization.membershipHash +
        ";repeated_candidate_evidence=deduplicated_by_observation_identity;"
        "unavailable=explicit_no_contribution_control_score_retained;"
        "zero_actionable=explicit_no_contribution_control_score_retained;"
        "order=shadow_score_desc,control_rank_asc,ranking_member_id_asc;"
        "rank_delta=control_rank-shadow_rank;live_rank_authoritative=true;"
        "live_profitability_weight=0;live_profitability_score_contribution=0";
    ranking.policyHash = InferenceProfitability::DeterministicHash(
        ranking.policyCanonical);

    std::map<long long, Recommendation::ProfitabilityShadowNormalizationResult>
        normalizedByObservation;
    for (const auto& result : ranking.normalization.results)
        normalizedByObservation.emplace(result.profitabilityObservationId, result);
    ranking.candidates.reserve(candidates.size());
    for (auto& source : candidates)
    {
        WeightedShadowCandidate candidate;
        candidate.source = std::move(source);
        candidate.shadowFinalScore = *candidate.source.currentScore;
        if (candidate.source.profitability.state == EvidenceState::valid &&
            candidate.source.profitability.observation)
        {
            const auto found = normalizedByObservation.find(
                candidate.source.profitability.observation->observationId);
            if (found == normalizedByObservation.end())
                throw std::logic_error("profitability_shadow_normalization_missing");
            const auto& normalized = found->second;
            candidate.normalizationState =
                Recommendation::ProfitabilityShadowNormalizationStateText(
                    normalized.state);
            candidate.normalizationReason = normalized.reason;
            candidate.empiricalMidrankPercentile =
                normalized.empiricalMidrankPercentile;
            candidate.boundedCandidateMetric = normalized.boundedCandidateMetric;
            candidate.supportReliability = normalized.supportReliability;
            candidate.normalizedProfitabilityValue =
                normalized.normalizedProfitabilityValue;
            if (candidate.normalizedProfitabilityValue)
            {
                candidate.profitabilityContribution = ranking.shadowWeight *
                    *candidate.normalizedProfitabilityValue;
                candidate.shadowFinalScore +=
                    *candidate.profitabilityContribution;
            }
        }
        else
        {
            candidate.normalizationState = "explicitly_unavailable";
            candidate.normalizationReason =
                candidate.source.profitability.reason;
        }
        ranking.candidates.push_back(std::move(candidate));
    }
    std::sort(ranking.candidates.begin(), ranking.candidates.end(),
        [](const auto& left, const auto& right) {
            if (left.shadowFinalScore != right.shadowFinalScore)
                return left.shadowFinalScore > right.shadowFinalScore;
            return std::tie(left.source.currentRank,
                            left.source.rankingMemberId) <
                std::tie(right.source.currentRank,
                         right.source.rankingMemberId);
        });
    for (std::size_t index = 0; index < ranking.candidates.size(); ++index)
    {
        auto& candidate = ranking.candidates[index];
        candidate.shadowRank = static_cast<int>(index) + 1;
        candidate.rankDelta =
            candidate.source.currentRank - candidate.shadowRank;
        candidate.canonical =
            "campaign_profitability_weighted_shadow_candidate_v1;";
        AppendField(candidate.canonical, "ranking_member_id",
                    std::to_string(candidate.source.rankingMemberId));
        AppendField(candidate.canonical, "recommendation_id",
                    std::to_string(candidate.source.recommendationId));
        AppendField(candidate.canonical, "evaluation_result_id",
                    std::to_string(
                        candidate.source.recommendationEvaluationResultId));
        AppendField(candidate.canonical, "control_rank",
                    std::to_string(candidate.source.currentRank));
        AppendField(candidate.canonical, "normalization_state",
                    candidate.normalizationState);
        AppendField(candidate.canonical, "normalized_profitability_value",
                    OptionalDouble(candidate.normalizedProfitabilityValue));
        AppendField(candidate.canonical, "profitability_contribution",
                    OptionalDouble(candidate.profitabilityContribution));
        AppendField(candidate.canonical, "shadow_final_score",
                    OptionalDouble(candidate.shadowFinalScore));
        AppendField(candidate.canonical, "shadow_rank",
                    std::to_string(candidate.shadowRank));
        AppendField(candidate.canonical, "rank_delta",
                    std::to_string(candidate.rankDelta));
        candidate.hash = InferenceProfitability::DeterministicHash(
            candidate.canonical);
    }
    ranking.canonical =
        "campaign_profitability_weighted_shadow_ranking_v1;";
    AppendField(ranking.canonical, "policy", ranking.policyCanonical);
    AppendField(ranking.canonical, "normalization",
                ranking.normalization.canonical);
    AppendField(ranking.canonical, "candidate_count",
                std::to_string(ranking.candidates.size()));
    for (std::size_t index = 0; index < ranking.candidates.size(); ++index)
        AppendField(ranking.canonical, "candidate[" + std::to_string(index) + "]",
                    ranking.candidates[index].canonical);
    ranking.hash = InferenceProfitability::DeterministicHash(ranking.canonical);
    return ranking;
}

std::string ReadinessActionText(ReadinessAction action)
{
    switch (action)
    {
        case ReadinessAction::readyForShadowValidation:
            return "ready_for_shadow_validation";
        case ReadinessAction::blockedEvidenceContract:
            return "blocked_evidence_contract";
        case ReadinessAction::blockedSoftwareReadiness:
            return "blocked_software_readiness";
        case ReadinessAction::needsPolicyDecision:
            return "needs_policy_decision";
        case ReadinessAction::eligibleForActivationReview:
            return "eligible_for_activation_review";
    }
    throw std::logic_error("unknown_profitability_readiness_action");
}

ReadinessGate EvaluateReadinessGate(bool softwareReady,
                                    bool integrationLoaded,
                                    const ShadowRanking& shadow)
{
    ReadinessGate gate;
    gate.profitabilitySoftwareReady = softwareReady;
    gate.activationPerformed = false;
    if (!softwareReady)
    {
        gate.action = ReadinessAction::blockedSoftwareReadiness;
        return gate;
    }
    const bool invalid = std::any_of(
        shadow.candidates.begin(), shadow.candidates.end(),
        [](const auto& candidate) {
            return EvidenceStateIsInvalid(candidate.profitability.state);
        });
    gate.campaignProfitabilityContractReady = integrationLoaded && !invalid;
    if (!gate.campaignProfitabilityContractReady)
    {
        gate.action = ReadinessAction::blockedEvidenceContract;
        return gate;
    }
    gate.profitabilityShadowRankingReady = true;
    gate.action = shadow.candidates.empty()
        ? ReadinessAction::needsPolicyDecision
        : ReadinessAction::readyForShadowValidation;
    return gate;
}

} // namespace EA::ProfitabilityVerification
