#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

#include "../Sources/ExperimentRecommendationEvaluation.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationEvaluationInput Input(long long id = 1)
{
    RecommendationEvaluationInput input;
    auto& score = input.scoringInput;
    score.recommendationId = id;
    score.sourceExperimentId = 10;
    score.sourcePredictionHorizon = 12;
    score.sourceRankWithinGroup = 1;
    score.sourceLeaderScore = 0.75;
    score.sourceInferenceAccuracy = 0.70;
    score.sourcePredictedNeutralProportion = 0.30;
    score.sourceEvidenceCount = 100;
    score.changedParameter = "core_lr_mult";
    score.sourceValueCanonical = "1";
    score.proposedValueCanonical = "1.25";
    score.absoluteDelta = 0.25;
    score.relativeDelta = 0.25;
    score.generationOrdinal = 1;
    score.structuralRank = 1;
    score.semanticCanonicalText = "semantic_" + std::to_string(id);
    input.recommendationSemanticHash = "semantic_hash_" + std::to_string(id);
    score.invocationCanonicalText = "invocation_" + std::to_string(id);
    score.recommendationPolicyCanonicalText = "policy_v1";
    score.duplicateType = "no_duplicate";
    score.recommendationStatus = "proposed";
    input.recommendationScanId = 20;
    input.sourceModelId = 30;
    input.sourceAnalysisId = 40;
    input.sourceSymbol = "eurusd";
    input.scanStatus = "completed";
    input.sourceExperimentStatus = "completed";
    input.sourceExperimentPhase = "done";
    input.currentSourceModelId = 30;
    input.currentSourceAnalysisId = 40;
    input.currentSourceAnalysisStatus = "completed";
    input.currentSourceAnalysisScope = "final";
    return input;
}

void RequireDisposition(const RecommendationEvaluationInput& input,
                        RecommendationEvaluationDisposition disposition,
                        const std::string& reason)
{
    const auto result = EvaluateExperimentRecommendation({}, input);
    assert(result.eligibility == RecommendationEligibility::ineligible);
    assert(result.disposition == disposition);
    assert(result.reasonCode == reason);
    assert(!result.finalScore);
    assert(result.components.empty());
}

} // namespace

int main()
{
    const RecommendationEvaluationPolicy policy;
    assert(!ValidateRecommendationEvaluationPolicy(policy));
    assert(RecommendationEvaluationPolicyCanonicalText(policy).find(
        "experiment_recommendation_scoring_policy_v1") != std::string::npos);
    assert(RecommendationEvaluationPolicyHash(policy).rfind("fnv1a64:", 0) == 0);

    const auto ready = EvaluateExperimentRecommendation(policy, Input());
    assert(ready.eligibility == RecommendationEligibility::eligible);
    assert(ready.disposition == RecommendationEvaluationDisposition::advisoryReady);
    assert(ready.reasonCode == "advisory_ready");
    assert(ready.finalScore && *ready.finalScore >= 0.0 && *ready.finalScore <= 1.0);
    assert(ready.components.size() == 9);
    assert(ready.scoringVersion == policy.scoringPolicy.scoringVersion);

    const auto repeat = EvaluateExperimentRecommendation(policy, Input());
    assert(repeat.evaluationIdentityCanonical == ready.evaluationIdentityCanonical);
    assert(repeat.evaluationIdentityHash == ready.evaluationIdentityHash);
    assert(repeat.finalScore == ready.finalScore);
    assert(repeat.explanation == ready.explanation);

    auto reordered = Input();
    reordered.exactExperimentConflicts = {{80, "completed"}, {70, "failed"}};
    auto alternateOrder = reordered;
    std::reverse(alternateOrder.exactExperimentConflicts.begin(),
                 alternateOrder.exactExperimentConflicts.end());
    assert(RecommendationEvaluationEvidenceCanonicalText(reordered) ==
           RecommendationEvaluationEvidenceCanonicalText(alternateOrder));
    auto changedSemanticHash = Input();
    changedSemanticHash.recommendationSemanticHash = "different_step1_hash";
    assert(RecommendationEvaluationEvidenceCanonicalText(changedSemanticHash) !=
           RecommendationEvaluationEvidenceCanonicalText(Input()));

    auto missing = Input();
    missing.sourceModelId.reset();
    missing.currentSourceAnalysisId.reset();
    const auto missingResult = EvaluateExperimentRecommendation(policy, missing);
    assert(missingResult.disposition ==
           RecommendationEvaluationDisposition::insufficientEvidence);
    assert(missingResult.missingEvidenceCount == 2);

    auto stale = Input();
    stale.currentSourceModelId = 31;
    RequireDisposition(stale,
        RecommendationEvaluationDisposition::staleSourceEvidence,
        "stale_source_provenance");

    auto unsupported = Input();
    unsupported.scoringInput.changedParameter = "opaque_family";
    RequireDisposition(unsupported,
        RecommendationEvaluationDisposition::unsupportedRecommendationFamily,
        "unsupported_recommendation_family");

    auto pending = Input();
    pending.exactExperimentConflicts = {{91, "pending"}};
    RequireDisposition(pending,
        RecommendationEvaluationDisposition::blockedPendingDuplicate,
        "exact_pending_experiment_duplicate");
    auto active = Input();
    active.exactExperimentConflicts = {{92, "running"}};
    RequireDisposition(active,
        RecommendationEvaluationDisposition::blockedActiveDuplicate,
        "exact_active_experiment_duplicate");
    auto paused = Input();
    paused.exactExperimentConflicts = {{94, "paused"}};
    RequireDisposition(paused,
        RecommendationEvaluationDisposition::blockedActiveDuplicate,
        "exact_active_experiment_duplicate");
    auto completed = Input();
    completed.exactExperimentConflicts = {{93, "completed"}};
    RequireDisposition(completed,
        RecommendationEvaluationDisposition::completedDuplicate,
        "exact_completed_experiment_duplicate");

    auto invalid = Input();
    invalid.scoringInput.sourceInferenceAccuracy =
        std::numeric_limits<double>::quiet_NaN();
    const auto invalidResult = EvaluateExperimentRecommendation(policy, invalid);
    assert(invalidResult.disposition ==
           RecommendationEvaluationDisposition::invalidPersistedEvidence);
    assert(invalidResult.evidenceCanonical.find("NaN_REJECTED") !=
           std::string::npos);

    auto infinite = Input();
    infinite.scoringInput.sourceLeaderScore =
        std::numeric_limits<double>::infinity();
    const auto infiniteResult = EvaluateExperimentRecommendation(policy, infinite);
    assert(infiniteResult.disposition ==
           RecommendationEvaluationDisposition::invalidPersistedEvidence);
    assert(infiniteResult.evidenceCanonical.find("POSITIVE_INFINITY_REJECTED") !=
           std::string::npos);

    auto negativeInfinite = Input();
    negativeInfinite.scoringInput.sourceInferenceAccuracy =
        -std::numeric_limits<double>::infinity();
    const auto negativeInfiniteResult = EvaluateExperimentRecommendation(
        policy, negativeInfinite);
    assert(negativeInfiniteResult.disposition ==
           RecommendationEvaluationDisposition::invalidPersistedEvidence);
    assert(negativeInfiniteResult.evidenceCanonical.find(
        "NEGATIVE_INFINITY_REJECTED") != std::string::npos);

    // Classification precedence is deliberately fail-closed:
    // invalid > unsupported > missing > stale > duplicate > ready.
    auto invalidPrecedence = Input();
    invalidPrecedence.scoringInput.sourceLeaderScore =
        std::numeric_limits<double>::quiet_NaN();
    invalidPrecedence.scoringInput.changedParameter = "opaque_family";
    invalidPrecedence.sourceModelId.reset();
    invalidPrecedence.scanStatus = "running";
    invalidPrecedence.exactExperimentConflicts = {{95, "pending"}};
    assert(EvaluateExperimentRecommendation(policy, invalidPrecedence).disposition ==
           RecommendationEvaluationDisposition::invalidPersistedEvidence);

    auto invalidStructuralPrecedence = Input();
    invalidStructuralPrecedence.scoringInput.relativeDelta =
        std::numeric_limits<double>::quiet_NaN();
    invalidStructuralPrecedence.exactExperimentConflicts = {{95, "pending"}};
    assert(EvaluateExperimentRecommendation(
        policy, invalidStructuralPrecedence).disposition ==
        RecommendationEvaluationDisposition::invalidPersistedEvidence);

    auto unsupportedPrecedence = Input();
    unsupportedPrecedence.scoringInput.changedParameter = "opaque_family";
    unsupportedPrecedence.sourceModelId.reset();
    unsupportedPrecedence.scanStatus = "running";
    unsupportedPrecedence.exactExperimentConflicts = {{95, "pending"}};
    assert(EvaluateExperimentRecommendation(policy, unsupportedPrecedence).disposition ==
           RecommendationEvaluationDisposition::unsupportedRecommendationFamily);

    auto missingPrecedence = Input();
    missingPrecedence.sourceModelId.reset();
    missingPrecedence.scanStatus = "running";
    missingPrecedence.exactExperimentConflicts = {{95, "pending"}};
    assert(EvaluateExperimentRecommendation(policy, missingPrecedence).disposition ==
           RecommendationEvaluationDisposition::insufficientEvidence);

    auto stalePrecedence = Input();
    stalePrecedence.scanStatus = "running";
    stalePrecedence.exactExperimentConflicts = {{95, "pending"}};
    assert(EvaluateExperimentRecommendation(policy, stalePrecedence).disposition ==
           RecommendationEvaluationDisposition::staleSourceEvidence);

    auto duplicatePrecedence = Input();
    duplicatePrecedence.exactExperimentConflicts = {
        {97, "completed"}, {96, "running"}, {98, "pending"}};
    assert(EvaluateExperimentRecommendation(policy, duplicatePrecedence).disposition ==
           RecommendationEvaluationDisposition::blockedPendingDuplicate);

    RecommendationEvaluationPolicy changed = policy;
    changed.evaluationVersion = 2;
    assert(ValidateRecommendationEvaluationPolicy(changed) ==
           std::optional<std::string>{"unsupported_evaluation_version"});
    changed = policy;
    changed.evaluatorVersion = 2;
    assert(ValidateRecommendationEvaluationPolicy(changed) ==
           std::optional<std::string>{"unsupported_evaluator_version"});
    changed = policy;
    changed.scoringPolicy.leaderScoreWeight = 0.30;
    assert(RecommendationEvaluationPolicyCanonicalText(changed) !=
           RecommendationEvaluationPolicyCanonicalText(policy));

    auto low = Input(2);
    low.scoringInput.sourceLeaderScore = 0.1;
    auto high = Input(3);
    high.scoringInput.sourceLeaderScore = 0.9;
    auto blocked = Input(4);
    blocked.exactExperimentConflicts = {{100, "pending"}};
    auto ranked = RankRecommendationEvaluations({
        EvaluateExperimentRecommendation(policy, low),
        EvaluateExperimentRecommendation(policy, blocked),
        EvaluateExperimentRecommendation(policy, high)});
    assert(ranked.size() == 3);
    assert(ranked[0].finalScore && ranked[1].finalScore && !ranked[2].finalScore);
    assert(ranked[0].rankingOrdinal == 1 && ranked[2].rankingOrdinal == 3);

    for (const auto disposition : {
        RecommendationEvaluationDisposition::advisoryReady,
        RecommendationEvaluationDisposition::insufficientEvidence,
        RecommendationEvaluationDisposition::blockedPendingDuplicate,
        RecommendationEvaluationDisposition::blockedActiveDuplicate,
        RecommendationEvaluationDisposition::completedDuplicate,
        RecommendationEvaluationDisposition::staleSourceEvidence,
        RecommendationEvaluationDisposition::unsupportedRecommendationFamily,
        RecommendationEvaluationDisposition::invalidPersistedEvidence})
    {
        const std::string text = RecommendationEvaluationDispositionText(disposition);
        assert(ParseRecommendationEvaluationDisposition(text) == disposition);
    }
    assert(!ParseRecommendationEvaluationDisposition("unknown"));
    return 0;
}
