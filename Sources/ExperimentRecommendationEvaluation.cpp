#include "ExperimentRecommendationEvaluation.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <locale>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string StableHash(const std::string& canonical)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char byte : canonical)
    {
        value ^= static_cast<std::uint64_t>(byte);
        value *= 1099511628211ULL;
    }
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 16> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index)
    {
        const unsigned shift = static_cast<unsigned>(
            (encoded.size() - index - 1U) * 4U);
        encoded[index] = digits[(value >> shift) & 0x0fU];
    }
    return "fnv1a64:" + std::string(encoded.begin(), encoded.end());
}

template <typename Value>
std::string OptionalInteger(const std::optional<Value>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalDouble(const std::optional<double>& value)
{
    if (!value) return "NULL";
    if (std::isnan(*value)) return "NaN_REJECTED";
    if (*value == std::numeric_limits<double>::infinity())
        return "POSITIVE_INFINITY_REJECTED";
    if (*value == -std::numeric_limits<double>::infinity())
        return "NEGATIVE_INFINITY_REJECTED";
    return CanonicalRecommendationDouble(*value);
}

std::string EvidenceDouble(double value)
{
    return OptionalDouble(std::optional<double>{value});
}

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

bool IsSupportedParameter(const std::string& value)
{
    return value == kCoreLrMult || value == kHeadLrMult ||
           value == kLabelThreshold || value == kPredictionHorizon;
}

bool HasInvalidPersistedScoringEvidence(
    const RecommendationEvaluationInput& input)
{
    const RecommendationScoringInput& score = input.scoringInput;
    return score.recommendationId <= 0 || input.recommendationScanId <= 0 ||
        score.sourceExperimentId <= 0 || score.recommendationStatus != "proposed" ||
        score.sourcePredictionHorizon <= 0 || score.sourceRankWithinGroup <= 0 ||
        score.generationOrdinal <= 0 || score.structuralRank <= 0 ||
        score.sourceEvidenceCount < 0 || input.sourceSymbol.empty() ||
        score.semanticCanonicalText.empty() ||
        input.recommendationSemanticHash.empty() ||
        score.invocationCanonicalText.empty() ||
        score.recommendationPolicyCanonicalText.empty() ||
        !std::isfinite(score.sourceLeaderScore) ||
        !std::isfinite(score.sourceInferenceAccuracy) ||
        score.sourceInferenceAccuracy < 0.0 ||
        score.sourceInferenceAccuracy > 1.0 ||
        !std::isfinite(score.absoluteDelta) || score.absoluteDelta < 0.0 ||
        (score.relativeDelta &&
         (!std::isfinite(*score.relativeDelta) || *score.relativeDelta < 0.0)) ||
        (score.sourcePredictedNeutralProportion &&
         (!std::isfinite(*score.sourcePredictedNeutralProportion) ||
          *score.sourcePredictedNeutralProportion < 0.0 ||
          *score.sourcePredictedNeutralProportion > 1.0)) ||
        (score.changedParameter == kPredictionHorizon && !score.horizonDelta);
}

RecommendationEvaluationResult BaseResult(
    const RecommendationEvaluationPolicy& policy,
    const RecommendationEvaluationInput& input)
{
    RecommendationEvaluationResult result;
    result.recommendationId = input.scoringInput.recommendationId;
    result.evaluationPolicyCanonical =
        RecommendationEvaluationPolicyCanonicalText(policy);
    result.evaluationPolicyHash = RecommendationEvaluationPolicyHash(policy);
    result.evaluationVersion = policy.evaluationVersion;
    result.evaluatorVersion = policy.evaluatorVersion;
    result.scoringPolicyHash = RecommendationScoringPolicyHash(
        policy.scoringPolicy);
    result.scoringVersion = policy.scoringPolicy.scoringVersion;
    result.evidenceCanonical =
        RecommendationEvaluationEvidenceCanonicalText(input);
    result.evidenceHash = StableHash(result.evidenceCanonical);
    std::ostringstream identity;
    identity.imbue(std::locale::classic());
    identity << "experiment_recommendation_evaluation_identity_v1"
             << ";evaluation_policy="
             << LengthText(result.evaluationPolicyCanonical)
             << ";recommendation_semantic="
             << LengthText(input.scoringInput.semanticCanonicalText)
             << ";recommendation_policy="
             << LengthText(input.scoringInput.recommendationPolicyCanonicalText)
             << ";evidence=" << LengthText(result.evidenceCanonical);
    result.evaluationIdentityCanonical = identity.str();
    result.evaluationIdentityHash = StableHash(result.evaluationIdentityCanonical);
    result.recommendationSemanticCanonical =
        input.scoringInput.semanticCanonicalText;
    result.recommendationSemanticHash = input.recommendationSemanticHash;
    return result;
}

RecommendationEvaluationResult Blocked(
    RecommendationEvaluationResult result,
    RecommendationEvaluationDisposition disposition,
    const std::string& reason,
    const std::string& explanation,
    int missingEvidenceCount = 0)
{
    result.eligibility = RecommendationEligibility::ineligible;
    result.disposition = disposition;
    result.reasonCode = reason;
    result.explanation = explanation;
    result.missingEvidenceCount = missingEvidenceCount;
    return result;
}

int ConflictOrder(const std::string& status)
{
    if (status == "pending") return 0;
    if (status == "running") return 1;
    if (status == "paused") return 1;
    if (status == "completed") return 2;
    return 3;
}

} // namespace

std::optional<std::string> ValidateRecommendationEvaluationPolicy(
    const RecommendationEvaluationPolicy& policy)
{
    if (policy.evaluationVersion != 1)
        return "unsupported_evaluation_version";
    if (policy.evaluatorVersion != 1)
        return "unsupported_evaluator_version";
    if (const auto error = ValidateRecommendationScoringPolicy(
            policy.scoringPolicy))
        return "invalid_embedded_scoring_policy:" + *error;
    return std::nullopt;
}

std::string RecommendationEvaluationPolicyCanonicalText(
    const RecommendationEvaluationPolicy& policy)
{
    if (const auto error = ValidateRecommendationEvaluationPolicy(policy))
        throw std::invalid_argument(*error);
    const std::string scoring = RecommendationScoringPolicyCanonicalText(
        policy.scoringPolicy);
    return "experiment_recommendation_evaluation_policy_v1;evaluation_version=" +
        std::to_string(policy.evaluationVersion) + ";evaluator_version=" +
        std::to_string(policy.evaluatorVersion) + ";scoring_policy=" +
        LengthText(scoring);
}

std::string RecommendationEvaluationPolicyHash(
    const RecommendationEvaluationPolicy& policy)
{
    return StableHash(RecommendationEvaluationPolicyCanonicalText(policy));
}

std::string RecommendationEvaluationCanonicalHash(const std::string& canonical)
{
    return StableHash(canonical);
}

std::string RecommendationEligibilityText(RecommendationEligibility value)
{
    switch (value)
    {
        case RecommendationEligibility::eligible: return "eligible";
        case RecommendationEligibility::ineligible: return "ineligible";
    }
    throw std::invalid_argument("invalid_recommendation_eligibility");
}

std::string RecommendationEvaluationDispositionText(
    RecommendationEvaluationDisposition value)
{
    switch (value)
    {
        case RecommendationEvaluationDisposition::advisoryReady:
            return "advisory_ready";
        case RecommendationEvaluationDisposition::insufficientEvidence:
            return "insufficient_evidence";
        case RecommendationEvaluationDisposition::blockedPendingDuplicate:
            return "blocked_pending_duplicate";
        case RecommendationEvaluationDisposition::blockedActiveDuplicate:
            return "blocked_active_duplicate";
        case RecommendationEvaluationDisposition::completedDuplicate:
            return "completed_duplicate";
        case RecommendationEvaluationDisposition::staleSourceEvidence:
            return "stale_source_evidence";
        case RecommendationEvaluationDisposition::unsupportedRecommendationFamily:
            return "unsupported_recommendation_family";
        case RecommendationEvaluationDisposition::invalidPersistedEvidence:
            return "invalid_persisted_evidence";
    }
    throw std::invalid_argument("invalid_recommendation_evaluation_disposition");
}

std::optional<RecommendationEvaluationDisposition>
ParseRecommendationEvaluationDisposition(const std::string& value)
{
    if (value == "advisory_ready")
        return RecommendationEvaluationDisposition::advisoryReady;
    if (value == "insufficient_evidence")
        return RecommendationEvaluationDisposition::insufficientEvidence;
    if (value == "blocked_pending_duplicate")
        return RecommendationEvaluationDisposition::blockedPendingDuplicate;
    if (value == "blocked_active_duplicate")
        return RecommendationEvaluationDisposition::blockedActiveDuplicate;
    if (value == "completed_duplicate")
        return RecommendationEvaluationDisposition::completedDuplicate;
    if (value == "stale_source_evidence")
        return RecommendationEvaluationDisposition::staleSourceEvidence;
    if (value == "unsupported_recommendation_family")
        return RecommendationEvaluationDisposition::unsupportedRecommendationFamily;
    if (value == "invalid_persisted_evidence")
        return RecommendationEvaluationDisposition::invalidPersistedEvidence;
    return std::nullopt;
}

std::string RecommendationEvaluationEvidenceCanonicalText(
    const RecommendationEvaluationInput& input)
{
    std::vector<RecommendationEvaluationExperimentConflict> conflicts =
        input.exactExperimentConflicts;
    std::sort(conflicts.begin(), conflicts.end(), [](const auto& left,
                                                      const auto& right) {
        return std::tie(left.status, left.experimentId) <
               std::tie(right.status, right.experimentId);
    });
    const RecommendationScoringInput& score = input.scoringInput;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_evaluation_evidence_v1"
        << ";recommendation_id=" << score.recommendationId
        << ";recommendation_scan_id=" << input.recommendationScanId
        << ";recommendation_status=" << LengthText(score.recommendationStatus)
        << ";source_experiment_id=" << score.sourceExperimentId
        << ";source_model_id=" << OptionalInteger(input.sourceModelId)
        << ";source_analysis_id=" << OptionalInteger(input.sourceAnalysisId)
        << ";source_symbol=" << LengthText(input.sourceSymbol)
        << ";source_horizon=" << score.sourcePredictionHorizon
        << ";source_rank=" << score.sourceRankWithinGroup
        << ";leader_score=" << EvidenceDouble(score.sourceLeaderScore)
        << ";inference_accuracy="
        << EvidenceDouble(score.sourceInferenceAccuracy)
        << ";neutral_proportion="
        << OptionalDouble(score.sourcePredictedNeutralProportion)
        << ";evidence_count=" << score.sourceEvidenceCount
        << ";changed_parameter=" << LengthText(score.changedParameter)
        << ";source_value=" << LengthText(score.sourceValueCanonical)
        << ";proposed_value=" << LengthText(score.proposedValueCanonical)
        << ";absolute_delta=" << EvidenceDouble(score.absoluteDelta)
        << ";relative_delta=" << OptionalDouble(score.relativeDelta)
        << ";horizon_delta=" << OptionalInteger(score.horizonDelta)
        << ";generation_ordinal=" << score.generationOrdinal
        << ";structural_rank=" << score.structuralRank
        << ";semantic=" << LengthText(score.semanticCanonicalText)
        << ";semantic_hash=" << LengthText(input.recommendationSemanticHash)
        << ";invocation=" << LengthText(score.invocationCanonicalText)
        << ";recommendation_policy="
        << LengthText(score.recommendationPolicyCanonicalText)
        << ";duplicate_type=" << LengthText(score.duplicateType)
        << ";scan_status=" << LengthText(input.scanStatus)
        << ";source_experiment_status="
        << LengthText(input.sourceExperimentStatus)
        << ";source_experiment_phase="
        << LengthText(input.sourceExperimentPhase)
        << ";current_source_model_id="
        << OptionalInteger(input.currentSourceModelId)
        << ";current_source_analysis_id="
        << OptionalInteger(input.currentSourceAnalysisId)
        << ";current_source_analysis_status="
        << LengthText(input.currentSourceAnalysisStatus)
        << ";current_source_analysis_scope="
        << LengthText(input.currentSourceAnalysisScope)
        << ";conflict_count=" << conflicts.size();
    for (std::size_t index = 0; index < conflicts.size(); ++index)
    {
        out << ";conflict[" << index << "]=" << conflicts[index].experimentId
            << ":" << LengthText(conflicts[index].status);
    }
    return out.str();
}

std::string RecommendationEvaluationEvidenceHash(
    const RecommendationEvaluationInput& input)
{
    return StableHash(RecommendationEvaluationEvidenceCanonicalText(input));
}

RecommendationEvaluationResult EvaluateExperimentRecommendation(
    const RecommendationEvaluationPolicy& policy,
    const RecommendationEvaluationInput& input)
{
    if (const auto error = ValidateRecommendationEvaluationPolicy(policy))
        throw std::invalid_argument(*error);
    RecommendationEvaluationResult result = BaseResult(policy, input);
    const RecommendationScoringInput& score = input.scoringInput;

    if (HasInvalidPersistedScoringEvidence(input))
        return Blocked(std::move(result),
            RecommendationEvaluationDisposition::invalidPersistedEvidence,
            "invalid_persisted_evidence",
            "Persisted recommendation identity or numeric evidence is invalid.");

    if (!IsSupportedParameter(score.changedParameter))
        return Blocked(std::move(result),
            RecommendationEvaluationDisposition::unsupportedRecommendationFamily,
            "unsupported_recommendation_family",
            "The recommendation family is not supported by evaluation policy version 1.");

    int missing = 0;
    if (!input.sourceModelId) ++missing;
    if (!input.sourceAnalysisId) ++missing;
    if (!input.currentSourceModelId) ++missing;
    if (!input.currentSourceAnalysisId) ++missing;
    if (score.sourceEvidenceCount < policy.scoringPolicy.minimumEvidenceCount)
        ++missing;
    if (missing != 0)
        return Blocked(std::move(result),
            RecommendationEvaluationDisposition::insufficientEvidence,
            "insufficient_persisted_source_evidence",
            "Required persisted model, analysis, or sample evidence is missing.",
            missing);

    if (input.scanStatus != "completed" ||
        input.sourceExperimentStatus != "completed" ||
        input.sourceExperimentPhase != "done" ||
        input.currentSourceAnalysisStatus != "completed" ||
        input.currentSourceAnalysisScope != "final" ||
        input.sourceModelId != input.currentSourceModelId ||
        input.sourceAnalysisId != input.currentSourceAnalysisId)
        return Blocked(std::move(result),
            RecommendationEvaluationDisposition::staleSourceEvidence,
            "stale_source_provenance",
            "Stored recommendation provenance no longer matches the completed source evidence.");

    std::vector<RecommendationEvaluationExperimentConflict> conflicts =
        input.exactExperimentConflicts;
    std::sort(conflicts.begin(), conflicts.end(), [](const auto& left,
                                                      const auto& right) {
        return std::tuple{ConflictOrder(left.status), left.experimentId} <
               std::tuple{ConflictOrder(right.status), right.experimentId};
    });
    for (const auto& conflict : conflicts)
    {
        if (conflict.status == "pending")
            return Blocked(std::move(result),
                RecommendationEvaluationDisposition::blockedPendingDuplicate,
                "exact_pending_experiment_duplicate",
                "An exact pending experiment already represents this semantic configuration.");
        if (conflict.status == "running" || conflict.status == "paused")
            return Blocked(std::move(result),
                RecommendationEvaluationDisposition::blockedActiveDuplicate,
                "exact_active_experiment_duplicate",
                "An exact active experiment already represents this semantic configuration.");
        if (conflict.status == "completed")
            return Blocked(std::move(result),
                RecommendationEvaluationDisposition::completedDuplicate,
                "exact_completed_experiment_duplicate",
                "An exact completed experiment already represents this semantic configuration.");
    }

    RecommendationScoreResult scoreResult = ScoreExperimentRecommendation(
        policy.scoringPolicy, score);
    if (!scoreResult.valid)
    {
        const bool evidence = scoreResult.reasonCode ==
            "source_evidence_below_scoring_minimum" ||
            scoreResult.reasonCode == "missing_neutral_proportion";
        return Blocked(std::move(result), evidence
                ? RecommendationEvaluationDisposition::insufficientEvidence
                : RecommendationEvaluationDisposition::invalidPersistedEvidence,
            scoreResult.reasonCode, scoreResult.explanationSummary,
            evidence ? 1 : 0);
    }

    result.eligibility = RecommendationEligibility::eligible;
    result.disposition = RecommendationEvaluationDisposition::advisoryReady;
    result.reasonCode = "advisory_ready";
    result.explanation =
        "Persisted provenance is complete and the unchanged Step 4 scoring policy produced an advisory score.";
    result.finalScore = scoreResult.finalScore;
    result.rawPositiveScore = scoreResult.rawPositiveScore;
    result.rawPenaltyScore = scoreResult.rawPenaltyScore;
    result.rawTotalScore = scoreResult.rawTotalScore;
    result.components = std::move(scoreResult.components);
    return result;
}

std::vector<RecommendationEvaluationResult> RankRecommendationEvaluations(
    std::vector<RecommendationEvaluationResult> results)
{
    std::sort(results.begin(), results.end(), [](const auto& left,
                                                 const auto& right) {
        if (left.finalScore.has_value() != right.finalScore.has_value())
            return left.finalScore.has_value();
        if (left.finalScore && *left.finalScore != *right.finalScore)
            return *left.finalScore > *right.finalScore;
        if (left.disposition != right.disposition)
            return RecommendationEvaluationDispositionText(left.disposition) <
                   RecommendationEvaluationDispositionText(right.disposition);
        if (left.evaluationIdentityCanonical != right.evaluationIdentityCanonical)
            return left.evaluationIdentityCanonical <
                   right.evaluationIdentityCanonical;
        return left.recommendationId < right.recommendationId;
    });
    for (std::size_t index = 0; index < results.size(); ++index)
        results[index].rankingOrdinal = static_cast<int>(index + 1U);
    return results;
}

} // namespace EA::ExperimentRecommendation
