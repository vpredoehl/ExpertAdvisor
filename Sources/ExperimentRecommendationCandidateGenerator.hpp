#pragma once

#include "ExperimentRecommendation.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

// Pure Phase 4A Step 2 results. This interface performs no persistence,
// scheduler orchestration, scoring, approval, or queueing. Candidate canonical
// text is authoritative identity; hashes are accelerators and collisions are
// retained and reported rather than merged.

enum class RecommendationSourceEligibilityReason
{
    eligible,
    policyDisabled,
    invalidPolicy,
    invalidSourceExperimentId,
    invalidSemanticConfiguration,
    invalidInvocationConfiguration,
    missingLeaderScore,
    missingInferenceAccuracy,
    nonfiniteLeaderScore,
    nonfiniteInferenceAccuracy,
    insufficientEvidence,
    leaderScoreBelowMinimum,
    inferenceAccuracyBelowMinimum,
    missingPredictedNeutralProportion,
    nonfinitePredictedNeutralProportion,
    predictedNeutralProportionOutOfRange,
    predictedNeutralProportionAboveMaximum
};

struct RecommendationSourceEligibilityResult
{
    bool eligible = false;
    RecommendationSourceEligibilityReason reason =
        RecommendationSourceEligibilityReason::invalidPolicy;
    std::string detail;
};

enum class RecommendationMutationParameter
{
    coreLrMult,
    headLrMult,
    labelThreshold,
    predictionHorizon
};

enum class RecommendationCandidateRejectionReason
{
    missingSourceValue,
    invalidCandidateValue,
    unchangedCandidate,
    singleParameterRuleViolation,
    duplicateCanonicalCandidate,
    perSourceLimit
};

struct GeneratedRecommendationCandidate
{
    RecommendationMutationParameter parameter =
        RecommendationMutationParameter::coreLrMult;
    double sourceValue = 0.0;
    double proposedValue = 0.0;
    double absoluteDelta = 0.0;
    std::optional<double> relativeDelta;
    std::optional<int> horizonDelta;
    RecommendationCandidateIdentity semanticIdentity;
    RecommendationInvocationIdentity invocationIdentity;
};

struct RejectedRecommendationCandidate
{
    RecommendationMutationParameter parameter =
        RecommendationMutationParameter::coreLrMult;
    std::string proposedValue;
    RecommendationCandidateRejectionReason reason =
        RecommendationCandidateRejectionReason::invalidCandidateValue;
    std::string semanticCanonicalText;
    std::string semanticHash;
};

struct RecommendationCandidateHashCollision
{
    std::string hash;
    std::string firstCanonicalText;
    std::string secondCanonicalText;
};

struct RecommendationCandidateGenerationCounters
{
    std::size_t attempted = 0;
    std::size_t validBeforeDeduplication = 0;
    std::size_t rejectedMissingSourceValue = 0;
    std::size_t rejectedInvalidValue = 0;
    std::size_t rejectedUnchanged = 0;
    std::size_t rejectedSingleParameterRule = 0;
    std::size_t rejectedDuplicate = 0;
    std::size_t hashCollisions = 0;
    std::size_t rejectedByPerSourceLimit = 0;
    std::size_t emitted = 0;
};

struct RecommendationCandidateDeduplicationResult
{
    std::vector<GeneratedRecommendationCandidate> candidates;
    std::vector<RejectedRecommendationCandidate> rejected;
    std::vector<RecommendationCandidateHashCollision> collisions;
};

struct RecommendationCandidateGenerationResult
{
    RecommendationSourceEligibilityResult eligibility;
    std::vector<GeneratedRecommendationCandidate> candidates;
    std::vector<RejectedRecommendationCandidate> rejected;
    std::vector<RecommendationCandidateHashCollision> collisions;
    RecommendationCandidateGenerationCounters counters;
};

RecommendationSourceEligibilityResult EvaluateRecommendationSource(
    const RecommendationPolicy& policy,
    const RecommendationSource& source);

RecommendationCandidateDeduplicationResult
DeduplicateRecommendationCandidatesCanonical(
    std::vector<GeneratedRecommendationCandidate> candidates);

RecommendationCandidateGenerationResult GenerateRecommendationCandidates(
    const RecommendationPolicy& policy,
    const RecommendationSource& source);

std::string RecommendationSourceEligibilityReasonText(
    RecommendationSourceEligibilityReason reason);
std::string RecommendationMutationParameterText(
    RecommendationMutationParameter parameter);
std::string RecommendationCandidateRejectionReasonText(
    RecommendationCandidateRejectionReason reason);

} // namespace EA::ExperimentRecommendation
