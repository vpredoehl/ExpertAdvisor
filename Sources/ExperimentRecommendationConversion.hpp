#pragma once

#include "ExperimentRecommendationCandidateGenerator.hpp"
#include "ExperimentRecommendationRanking.hpp"
#include "ExperimentRecommendationReview.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationConversionContractVersion = 1;

enum class RecommendationConversionEvidenceState
{
    missing,
    pending,
    completed,
    failed
};

enum class RecommendationConversionReason
{
    eligible,
    recommendationMissing,
    recommendationLifecycleNotConvertible,
    manualAuthorizationMissing,
    manualAuthorizationSupersededOrIneffective,
    evaluationMissingOrIncomplete,
    evaluationInvalid,
    scoreMissingOrIncomplete,
    scoreInvalid,
    recommendationBlocked,
    mutationMissing,
    unsupportedMutationFamily,
    malformedProposedValue,
    proposedValueOutsideAllowedRange,
    sourceConfigurationIncomplete,
    sourceValueMismatch,
    proposedValueEqualsSourceValue,
    multipleMutationsDetected,
    duplicateConversionIdentity,
    inconsistentProvenance,
    inconsistentSourceExperimentIdentity
};

struct RecommendationConversionReviewAuthorization
{
    bool present = false;
    long long recommendationId = -1;
    RecommendationReviewAction latestAction = RecommendationReviewAction::approve;
    RecommendationStatus resultingStatus = RecommendationStatus::approved;
    bool latestActionEffective = false;
    bool superseded = false;
    std::string authorizationCanonical;
    std::string authorizationHash;
};

struct RecommendationConversionEvaluationEvidence
{
    RecommendationConversionEvidenceState state =
        RecommendationConversionEvidenceState::missing;
    bool valid = false;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    RecommendationEligibility eligibility = RecommendationEligibility::ineligible;
    RecommendationEvaluationDisposition disposition =
        RecommendationEvaluationDisposition::invalidPersistedEvidence;
    std::string evaluationIdentityCanonical;
    std::string evaluationIdentityHash;
    std::string evaluationPolicyCanonical;
    std::string evaluationPolicyHash;
    std::string scoringPolicyHash;
};

struct RecommendationConversionScoreEvidence
{
    RecommendationConversionEvidenceState state =
        RecommendationConversionEvidenceState::missing;
    bool valid = false;
    long long recommendationId = -1;
    std::optional<double> finalScore;
    std::string scoringPolicyCanonical;
    std::string scoringPolicyHash;
};

// Ranking provenance is presentation-only evidence. It is validated when
// supplied but deliberately excluded from authorization and identity.
struct RecommendationConversionRankingProvenance
{
    std::string snapshotIdentityCanonical;
    std::string snapshotIdentityHash;
    std::string memberIdentityCanonical;
    std::string memberIdentityHash;
    RecommendationRankingBucket bucket =
        RecommendationRankingBucket::nonActionable;
    int bucketRank = 0;
};

struct RecommendationConversionMutation
{
    std::string family;
    std::string sourceValueCanonical;
    std::string proposedValueCanonical;
};

struct RecommendationConversionRequest
{
    bool recommendationExists = false;
    long long recommendationId = -1;
    RecommendationStatus recommendationStatus = RecommendationStatus::proposed;
    long long sourceExperimentId = -1;
    long long recommendationSourceExperimentId = -1;
    std::string recommendationSemanticCanonical;
    std::string recommendationSemanticHash;
    std::string recommendationInvocationCanonical;
    std::string recommendationInvocationHash;
    RecommendationConversionReviewAuthorization reviewAuthorization;
    RecommendationConversionEvaluationEvidence evaluation;
    RecommendationConversionScoreEvidence score;
    std::optional<RecommendationConversionRankingProvenance> ranking;
    ExperimentInvocationConfiguration sourceInvocation;
    std::vector<RecommendationConversionMutation> mutations;
    // Canonical text is authoritative; hashes alone cannot prove a duplicate.
    std::vector<std::string> existingConversionCanonicals;
};

struct RecommendationConversionEligibilityResult
{
    bool eligible = false;
    RecommendationConversionReason reason =
        RecommendationConversionReason::recommendationMissing;
    std::string explanation;
};

struct ProposedExperimentSpecification
{
    long long sourceExperimentId = -1;
    long long recommendationId = -1;
    ExperimentInvocationConfiguration proposedInvocation;
    // These canonical snapshots make a durable proposal independently
    // auditable without reloading a mutable source experiment row.
    std::string sourceInvocationCanonical;
    std::string proposedInvocationCanonical;
    RecommendationMutationParameter changedParameter =
        RecommendationMutationParameter::coreLrMult;
    std::string sourceValueCanonical;
    std::string proposedValueCanonical;
    std::string recommendationSemanticHash;
    std::string evaluationIdentityHash;
    std::string evaluationPolicyHash;
    std::string scoringPolicyHash;
    std::string reviewAuthorizationHash;
    std::optional<std::string> rankingSnapshotIdentityHash;
    std::string conversionIdentityCanonical;
    std::string conversionIdentityHash;
};

struct RecommendationConversionResult
{
    RecommendationConversionEligibilityResult eligibility;
    std::optional<ProposedExperimentSpecification> proposal;
};

std::string RecommendationConversionEvidenceStateText(
    RecommendationConversionEvidenceState value);
std::string RecommendationConversionReasonText(RecommendationConversionReason value);
std::string RecommendationConversionReasonExplanation(
    RecommendationConversionReason value);

RecommendationConversionResult BuildProposedExperimentSpecification(
    const RecommendationConversionRequest& request);

} // namespace EA::ExperimentRecommendation
