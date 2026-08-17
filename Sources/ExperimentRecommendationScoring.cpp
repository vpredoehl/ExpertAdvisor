#include "ExperimentRecommendationScoring.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <locale>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string TrimAscii(const std::string& value)
{
    const auto whitespace = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
               c == '\f' || c == '\v';
    };
    std::size_t first = 0;
    while (first < value.size() && whitespace(value[first])) ++first;
    std::size_t last = value.size();
    while (last > first && whitespace(value[last - 1])) --last;
    return value.substr(first, last - first);
}

double ParseFiniteDouble(const std::string& key, const std::string& text)
{
    if (text.empty())
        throw std::invalid_argument("invalid_scoring_policy_number:" + key);

    double value = 0.0;
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    stream >> std::noskipws >> value;

    if (!stream || !stream.eof() || !std::isfinite(value))
        throw std::invalid_argument("invalid_scoring_policy_number:" + key);

    return value == 0.0 ? 0.0 : value;
}

long long ParseInteger(const std::string& key, const std::string& text)
{
    long long value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument("invalid_scoring_policy_integer:" + key);
    return value;
}

bool ParseBoolean(const std::string& key, const std::string& text)
{
    if (text == "true" || text == "1") return true;
    if (text == "false" || text == "0") return false;
    throw std::invalid_argument("invalid_scoring_policy_boolean:" + key);
}

std::string StableHash(const std::string& canonical)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char byte : canonical)
    {
        value ^= static_cast<std::uint64_t>(byte);
        value *= 1099511628211ULL;
    }
    constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (int index = 15; index >= 0; --index)
    {
        result[static_cast<std::size_t>(index)] = digits[value & 0x0fU];
        value >>= 4U;
    }
    return "fnv1a64:" + result;
}

double ClampUnit(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

void AddComponent(
    RecommendationScoreResult& result,
    const std::string& name,
    const std::string& reason,
    const std::string& input,
    double normalized,
    double weight,
    bool penalty,
    const std::string& explanation)
{
    if (!std::isfinite(normalized) || !std::isfinite(weight))
        throw std::invalid_argument("recommendation_score_component_nonfinite");
    const double bounded = ClampUnit(normalized);
    result.components.push_back(RecommendationScoreComponent{
        name, reason, input, bounded, weight, bounded * weight, penalty,
        explanation});
}

double ParameterPreference(
    const RecommendationScoringPolicy& policy,
    const std::string& parameter)
{
    if (parameter == kCoreLrMult) return policy.coreLrPreference;
    if (parameter == kHeadLrMult) return policy.headLrPreference;
    if (parameter == kLabelThreshold) return policy.labelThresholdPreference;
    if (parameter == kPredictionHorizon)
        return policy.predictionHorizonPreference;
    return -1.0;
}

RecommendationScoreResult InvalidResult(
    const RecommendationScoringPolicy& policy,
    const std::string& reason)
{
    RecommendationScoreResult result;
    result.reasonCode = reason;
    result.explanationSummary = reason;
    if (!ValidateRecommendationScoringPolicy(policy))
    {
        result.scoringPolicyCanonical =
            RecommendationScoringPolicyCanonicalText(policy);
        result.scoringPolicyHash = RecommendationScoringPolicyHash(policy);
        result.scoringVersion = policy.scoringVersion;
    }
    return result;
}

} // namespace

RecommendationScoringPolicy ParseRecommendationScoringPolicy(
    const std::string& text)
{
    RecommendationScoringPolicy policy;
    if (TrimAscii(text).empty()) return policy;

    std::set<std::string> seen;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t comma = text.find(',', begin);
        const std::string assignment = TrimAscii(text.substr(
            begin, comma == std::string::npos ? std::string::npos
                                              : comma - begin));
        if (assignment.empty())
            throw std::invalid_argument("malformed_scoring_policy_assignment");
        const std::size_t equals = assignment.find('=');
        if (equals == std::string::npos ||
            assignment.find('=', equals + 1) != std::string::npos)
            throw std::invalid_argument("malformed_scoring_policy_assignment");
        const std::string key = TrimAscii(assignment.substr(0, equals));
        const std::string value = TrimAscii(assignment.substr(equals + 1));
        if (key.empty() || value.empty())
            throw std::invalid_argument("malformed_scoring_policy_assignment");
        if (!seen.insert(key).second)
            throw std::invalid_argument("duplicate_scoring_policy_key:" + key);

        if (key == "scoring_version")
        {
            const long long parsed = ParseInteger(key, value);
            if (parsed < std::numeric_limits<int>::min() ||
                parsed > std::numeric_limits<int>::max())
                throw std::invalid_argument(
                    "invalid_scoring_policy_integer:" + key);
            policy.scoringVersion = static_cast<int>(parsed);
        }
        else if (key == "leader_score_weight")
            policy.leaderScoreWeight = ParseFiniteDouble(key, value);
        else if (key == "inference_accuracy_weight")
            policy.inferenceAccuracyWeight = ParseFiniteDouble(key, value);
        else if (key == "evidence_strength_weight")
            policy.evidenceStrengthWeight = ParseFiniteDouble(key, value);
        else if (key == "neutral_balance_weight")
            policy.neutralBalanceWeight = ParseFiniteDouble(key, value);
        else if (key == "structural_distance_weight")
            policy.structuralDistanceWeight = ParseFiniteDouble(key, value);
        else if (key == "parameter_preference_weight")
            policy.parameterPreferenceWeight = ParseFiniteDouble(key, value);
        else if (key == "source_rank_weight")
            policy.sourceRankWeight = ParseFiniteDouble(key, value);
        else if (key == "horizon_change_penalty_weight")
            policy.horizonChangePenaltyWeight = ParseFiniteDouble(key, value);
        else if (key == "relative_mutation_penalty_weight")
            policy.relativeMutationPenaltyWeight = ParseFiniteDouble(key, value);
        else if (key == "minimum_evidence_count")
            policy.minimumEvidenceCount = ParseInteger(key, value);
        else if (key == "evidence_saturation_count")
            policy.evidenceSaturationCount = ParseInteger(key, value);
        else if (key == "preferred_neutral_proportion")
            policy.preferredNeutralProportion = ParseFiniteDouble(key, value);
        else if (key == "maximum_neutral_proportion")
            policy.maximumNeutralProportion = ParseFiniteDouble(key, value);
        else if (key == "maximum_relative_mutation")
            policy.maximumRelativeMutation = ParseFiniteDouble(key, value);
        else if (key == "maximum_absolute_structural_distance")
            policy.maximumAbsoluteStructuralDistance = ParseFiniteDouble(key, value);
        else if (key == "allow_missing_neutral_proportion")
            policy.allowMissingNeutralProportion = ParseBoolean(key, value);
        else if (key == "score_floor")
            policy.scoreFloor = ParseFiniteDouble(key, value);
        else if (key == "score_ceiling")
            policy.scoreCeiling = ParseFiniteDouble(key, value);
        else if (key == "core_lr_preference")
            policy.coreLrPreference = ParseFiniteDouble(key, value);
        else if (key == "head_lr_preference")
            policy.headLrPreference = ParseFiniteDouble(key, value);
        else if (key == "label_threshold_preference")
            policy.labelThresholdPreference = ParseFiniteDouble(key, value);
        else if (key == "prediction_horizon_preference")
            policy.predictionHorizonPreference = ParseFiniteDouble(key, value);
        else
            throw std::invalid_argument("unknown_scoring_policy_key:" + key);

        if (comma == std::string::npos) break;
        begin = comma + 1;
        if (begin == text.size())
            throw std::invalid_argument("malformed_scoring_policy_assignment");
    }

    if (const auto error = ValidateRecommendationScoringPolicy(policy))
        throw std::invalid_argument(*error);
    return policy;
}

std::optional<std::string> ValidateRecommendationScoringPolicy(
    const RecommendationScoringPolicy& policy)
{
    if (policy.scoringVersion != 1) return "unsupported_scoring_version";
    const double weights[] = {
        policy.leaderScoreWeight, policy.inferenceAccuracyWeight,
        policy.evidenceStrengthWeight, policy.neutralBalanceWeight,
        policy.structuralDistanceWeight, policy.parameterPreferenceWeight,
        policy.sourceRankWeight, policy.horizonChangePenaltyWeight,
        policy.relativeMutationPenaltyWeight};
    double positiveWeight = 0.0;
    for (std::size_t index = 0; index < std::size(weights); ++index)
    {
        if (!std::isfinite(weights[index])) return "nonfinite_scoring_weight";
        if (weights[index] < 0.0) return "negative_scoring_weight";
        if (index < 7) positiveWeight += weights[index];
    }
    if (!(positiveWeight > 0.0)) return "positive_scoring_weight_required";
    if (policy.minimumEvidenceCount <= 0)
        return "minimum_evidence_count_must_be_positive";
    if (policy.evidenceSaturationCount < policy.minimumEvidenceCount)
        return "evidence_saturation_below_minimum";
    if (!std::isfinite(policy.preferredNeutralProportion) ||
        policy.preferredNeutralProportion < 0.0 ||
        policy.preferredNeutralProportion > 1.0)
        return "preferred_neutral_proportion_out_of_range";
    if (!std::isfinite(policy.maximumNeutralProportion) ||
        policy.maximumNeutralProportion < policy.preferredNeutralProportion ||
        policy.maximumNeutralProportion > 1.0)
        return "maximum_neutral_proportion_out_of_range";
    if (!std::isfinite(policy.maximumRelativeMutation) ||
        policy.maximumRelativeMutation <= 0.0)
        return "maximum_relative_mutation_must_be_positive";
    if (!std::isfinite(policy.maximumAbsoluteStructuralDistance) ||
        policy.maximumAbsoluteStructuralDistance <= 0.0)
        return "maximum_absolute_structural_distance_must_be_positive";
    if (!std::isfinite(policy.scoreFloor) ||
        !std::isfinite(policy.scoreCeiling) || policy.scoreFloor < 0.0 ||
        policy.scoreCeiling > 1.0 || policy.scoreFloor >= policy.scoreCeiling)
        return "invalid_scoring_bounds";
    const double preferences[] = {
        policy.coreLrPreference, policy.headLrPreference,
        policy.labelThresholdPreference, policy.predictionHorizonPreference};
    for (const double preference : preferences)
        if (!std::isfinite(preference) || preference < 0.0 || preference > 1.0)
            return "parameter_preference_out_of_range";
    return std::nullopt;
}

std::string RecommendationScoringPolicyCanonicalText(
    const RecommendationScoringPolicy& policy)
{
    if (const auto error = ValidateRecommendationScoringPolicy(policy))
        throw std::invalid_argument(*error);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_scoring_policy_v1"
        << ";scoring_version=" << policy.scoringVersion
        << ";leader_score_weight=" << CanonicalRecommendationDouble(policy.leaderScoreWeight)
        << ";inference_accuracy_weight=" << CanonicalRecommendationDouble(policy.inferenceAccuracyWeight)
        << ";evidence_strength_weight=" << CanonicalRecommendationDouble(policy.evidenceStrengthWeight)
        << ";neutral_balance_weight=" << CanonicalRecommendationDouble(policy.neutralBalanceWeight)
        << ";structural_distance_weight=" << CanonicalRecommendationDouble(policy.structuralDistanceWeight)
        << ";parameter_preference_weight=" << CanonicalRecommendationDouble(policy.parameterPreferenceWeight)
        << ";source_rank_weight=" << CanonicalRecommendationDouble(policy.sourceRankWeight)
        << ";horizon_change_penalty_weight=" << CanonicalRecommendationDouble(policy.horizonChangePenaltyWeight)
        << ";relative_mutation_penalty_weight=" << CanonicalRecommendationDouble(policy.relativeMutationPenaltyWeight)
        << ";minimum_evidence_count=" << policy.minimumEvidenceCount
        << ";evidence_saturation_count=" << policy.evidenceSaturationCount
        << ";preferred_neutral_proportion=" << CanonicalRecommendationDouble(policy.preferredNeutralProportion)
        << ";maximum_neutral_proportion=" << CanonicalRecommendationDouble(policy.maximumNeutralProportion)
        << ";maximum_relative_mutation=" << CanonicalRecommendationDouble(policy.maximumRelativeMutation)
        << ";maximum_absolute_structural_distance=" << CanonicalRecommendationDouble(policy.maximumAbsoluteStructuralDistance)
        << ";allow_missing_neutral_proportion=" << (policy.allowMissingNeutralProportion ? 1 : 0)
        << ";score_floor=" << CanonicalRecommendationDouble(policy.scoreFloor)
        << ";score_ceiling=" << CanonicalRecommendationDouble(policy.scoreCeiling)
        << ";core_lr_preference=" << CanonicalRecommendationDouble(policy.coreLrPreference)
        << ";head_lr_preference=" << CanonicalRecommendationDouble(policy.headLrPreference)
        << ";label_threshold_preference=" << CanonicalRecommendationDouble(policy.labelThresholdPreference)
        << ";prediction_horizon_preference=" << CanonicalRecommendationDouble(policy.predictionHorizonPreference);
    return out.str();
}

std::string RecommendationScoringPolicyHash(
    const RecommendationScoringPolicy& policy)
{
    return StableHash(RecommendationScoringPolicyCanonicalText(policy));
}

RecommendationScoreResult ScoreExperimentRecommendation(
    const RecommendationScoringPolicy& policy,
    const RecommendationScoringInput& input)
{
    if (ValidateRecommendationScoringPolicy(policy))
        return InvalidResult(policy, "invalid_scoring_policy");
    if (input.recommendationStatus != "proposed")
        return InvalidResult(policy, "unsupported_recommendation_status");
    if (input.recommendationId <= 0 || input.sourceExperimentId <= 0 ||
        input.sourcePredictionHorizon <= 0 || input.sourceRankWithinGroup <= 0 ||
        input.generationOrdinal <= 0 || input.structuralRank <= 0 ||
        input.semanticCanonicalText.empty() ||
        input.invocationCanonicalText.empty() ||
        input.recommendationPolicyCanonicalText.empty())
        return InvalidResult(policy, "incomplete_scoring_identity");
    if (!std::isfinite(input.sourceLeaderScore) ||
        !std::isfinite(input.sourceInferenceAccuracy))
        return InvalidResult(policy, "nonfinite_source_metric");
    if (input.sourceInferenceAccuracy < 0.0 ||
        input.sourceInferenceAccuracy > 1.0)
        return InvalidResult(policy, "source_inference_accuracy_out_of_range");
    if (input.sourceEvidenceCount < policy.minimumEvidenceCount)
        return InvalidResult(policy, "source_evidence_below_scoring_minimum");
    if (!std::isfinite(input.absoluteDelta) || input.absoluteDelta < 0.0 ||
        (input.relativeDelta &&
         (!std::isfinite(*input.relativeDelta) || *input.relativeDelta < 0.0)))
        return InvalidResult(policy, "invalid_structural_metadata");
    if (input.sourcePredictedNeutralProportion &&
        (!std::isfinite(*input.sourcePredictedNeutralProportion) ||
         *input.sourcePredictedNeutralProportion < 0.0 ||
         *input.sourcePredictedNeutralProportion > 1.0))
        return InvalidResult(policy, "invalid_neutral_proportion");
    if (!input.sourcePredictedNeutralProportion &&
        !policy.allowMissingNeutralProportion)
        return InvalidResult(policy, "missing_neutral_proportion");
    const double parameterPreference = ParameterPreference(
        policy, input.changedParameter);
    if (parameterPreference < 0.0)
        return InvalidResult(policy, "unsupported_changed_parameter");

    RecommendationScoreResult result;
    result.valid = true;
    result.reasonCode = "scored";
    result.explanationSummary = "deterministic_advisory_prioritization";
    result.scoringPolicyCanonical =
        RecommendationScoringPolicyCanonicalText(policy);
    result.scoringPolicyHash = RecommendationScoringPolicyHash(policy);
    result.scoringVersion = policy.scoringVersion;

    AddComponent(result, "leader_quality", "leader_score_clamped_unit",
                 CanonicalRecommendationDouble(input.sourceLeaderScore),
                 ClampUnit(input.sourceLeaderScore), policy.leaderScoreWeight,
                 false, "Source leader score, clamped to the supported unit range.");
    AddComponent(result, "inference_accuracy", "inference_accuracy_unit",
                 CanonicalRecommendationDouble(input.sourceInferenceAccuracy),
                 input.sourceInferenceAccuracy,
                 policy.inferenceAccuracyWeight, false,
                 "Persisted final inference accuracy.");

    double evidence = 1.0;
    if (policy.evidenceSaturationCount > policy.minimumEvidenceCount)
    {
        evidence = ClampUnit(static_cast<double>(
            input.sourceEvidenceCount - policy.minimumEvidenceCount) /
            static_cast<double>(policy.evidenceSaturationCount -
                                policy.minimumEvidenceCount));
    }
    AddComponent(result, "evidence_strength", "linear_evidence_saturation",
                 std::to_string(input.sourceEvidenceCount), evidence,
                 policy.evidenceStrengthWeight, false,
                 "Evidence count linearly saturates at the configured count.");

    double neutral = 0.5;
    std::string neutralReason = "neutral_proportion_missing_allowed";
    std::string neutralInput = "NULL";
    if (input.sourcePredictedNeutralProportion)
    {
        const double value = *input.sourcePredictedNeutralProportion;
        neutralInput = CanonicalRecommendationDouble(value);
        neutralReason = "neutral_proportion_preferred_band";
        if (value <= policy.preferredNeutralProportion)
            neutral = 1.0;
        else if (policy.maximumNeutralProportion ==
                 policy.preferredNeutralProportion)
            neutral = 0.0;
        else
            neutral = ClampUnit((policy.maximumNeutralProportion - value) /
                (policy.maximumNeutralProportion -
                 policy.preferredNeutralProportion));
    }
    AddComponent(result, "neutral_balance", neutralReason, neutralInput,
                 neutral, policy.neutralBalanceWeight, false,
                 "Lower neutral dominance receives a bounded preference.");

    result.structuralDistance = DeriveRecommendationScoringDistance(input);
    const double structuralMaximum = input.relativeDelta
        ? policy.maximumRelativeMutation
        : policy.maximumAbsoluteStructuralDistance;
    const double proximity = ClampUnit(
        1.0 - result.structuralDistance / structuralMaximum);
    AddComponent(result, "structural_proximity",
                 input.relativeDelta ? "relative_delta_distance"
                                     : "absolute_zero_source_fallback",
                 CanonicalRecommendationDouble(result.structuralDistance),
                 proximity, policy.structuralDistanceWeight, false,
                 "Smaller distance derived from persisted Step 2 deltas is preferred.");

    AddComponent(result, "parameter_preference",
                 "operator_parameter_preference", input.changedParameter,
                 parameterPreference, policy.parameterPreferenceWeight, false,
                 "Explicit operator preference; it is not predicted performance.");
    AddComponent(result, "source_rank", "step3_source_rank_reciprocal",
                 std::to_string(input.sourceRankWithinGroup),
                 1.0 / static_cast<double>(input.sourceRankWithinGroup),
                 policy.sourceRankWeight, false,
                 "Small preference for the deterministic Step 3 source rank.");

    double horizonPenalty = 0.0;
    if (input.changedParameter == kPredictionHorizon)
    {
        if (!input.horizonDelta)
            return InvalidResult(policy, "missing_horizon_delta");
        horizonPenalty = ClampUnit(
            static_cast<double>(std::abs(
                static_cast<long long>(*input.horizonDelta))) /
            static_cast<double>(input.sourcePredictionHorizon));
    }
    AddComponent(result, "horizon_change_penalty",
                 input.changedParameter == kPredictionHorizon
                    ? "normalized_horizon_delta" : "not_horizon_mutation",
                 input.horizonDelta ? std::to_string(*input.horizonDelta) : "NULL",
                 horizonPenalty, policy.horizonChangePenaltyWeight, true,
                 "Horizon changes receive a bounded research-scope penalty.");

    const double mutationPenalty = input.relativeDelta
        ? ClampUnit(*input.relativeDelta / policy.maximumRelativeMutation)
        : ClampUnit(input.absoluteDelta /
                    policy.maximumAbsoluteStructuralDistance);
    AddComponent(result, "relative_mutation_penalty",
                 input.relativeDelta ? "relative_step2_delta"
                                     : "absolute_zero_source_fallback",
                 input.relativeDelta
                    ? CanonicalRecommendationDouble(*input.relativeDelta)
                    : CanonicalRecommendationDouble(input.absoluteDelta),
                 mutationPenalty, policy.relativeMutationPenaltyWeight, true,
                 "Larger persisted Step 2 mutations receive a bounded penalty.");

    double positiveWeight = 0.0;
    double positiveContribution = 0.0;
    double penaltyContribution = 0.0;
    for (const RecommendationScoreComponent& component : result.components)
    {
        if (component.penalty)
            penaltyContribution += component.weightedContribution;
        else
        {
            positiveWeight += component.weight;
            positiveContribution += component.weightedContribution;
        }
    }
    result.rawPositiveScore = positiveContribution / positiveWeight;
    result.rawPenaltyScore = penaltyContribution / positiveWeight;
    result.rawTotalScore =
        result.rawPositiveScore - result.rawPenaltyScore;
    result.finalScore = std::clamp(result.rawTotalScore,
                                   policy.scoreFloor, policy.scoreCeiling);
    if (!std::isfinite(result.finalScore))
        return InvalidResult(policy, "nonfinite_total_score");
    return result;
}

double DeriveRecommendationScoringDistance(
    const RecommendationScoringInput& input)
{
    return input.relativeDelta.value_or(input.absoluteDelta);
}

std::vector<RankedRecommendationScore> RankRecommendationScores(
    std::vector<RankedRecommendationScore> scores)
{
    for (const RankedRecommendationScore& item : scores)
        if (!item.score.valid)
            throw std::invalid_argument("cannot_rank_invalid_recommendation_score");
    std::sort(scores.begin(), scores.end(),
        [](const RankedRecommendationScore& lhs,
           const RankedRecommendationScore& rhs) {
            return std::tuple{
                -lhs.score.finalScore, -lhs.score.rawPositiveScore,
                lhs.score.rawPenaltyScore, -lhs.input.sourceLeaderScore,
                -lhs.input.sourceInferenceAccuracy,
                -lhs.input.sourceEvidenceCount, lhs.score.structuralDistance,
                lhs.input.semanticCanonicalText,
                lhs.input.recommendationPolicyCanonicalText,
                lhs.input.recommendationId} <
            std::tuple{
                -rhs.score.finalScore, -rhs.score.rawPositiveScore,
                rhs.score.rawPenaltyScore, -rhs.input.sourceLeaderScore,
                -rhs.input.sourceInferenceAccuracy,
                -rhs.input.sourceEvidenceCount, rhs.score.structuralDistance,
                rhs.input.semanticCanonicalText,
                rhs.input.recommendationPolicyCanonicalText,
                rhs.input.recommendationId};
        });

    int tieGroup = 0;
    int currentRank = 0;
    std::optional<double> previousScore;
    for (std::size_t index = 0; index < scores.size(); ++index)
    {
        if (!previousScore || scores[index].score.finalScore != *previousScore)
        {
            ++tieGroup;
            currentRank = static_cast<int>(index + 1);
            previousScore = scores[index].score.finalScore;
        }
        scores[index].rankingOrdinal = static_cast<int>(index + 1);
        scores[index].scoreRank = currentRank;
        scores[index].tieGroup = tieGroup;
    }
    return scores;
}

} // namespace EA::ExperimentRecommendation
