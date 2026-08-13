#pragma once

#include "Donchian20Mode.hpp"
#include "ExperimentRecommendationConversionWorkflow.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignPlanContractVersion = 1;
inline constexpr int kRecommendationCampaignPlanningPolicyCanonicalVersion = 2;
inline constexpr int kDefaultRecommendationCampaignMaximumSelected = 10;
inline constexpr int kDefaultRecommendationCampaignMaximumCandidates = 100;
inline constexpr int kMaximumRecommendationCampaignCandidates = 1000;

struct RecommendationCampaignPlanningPolicy
{
    bool enabled = false;
    int maximumSelectedRecommendations =
        kDefaultRecommendationCampaignMaximumSelected;
    int maximumCandidatesConsidered =
        kDefaultRecommendationCampaignMaximumCandidates;
    double minimumLeaderScore = 0.0;
    double minimumInferenceAccuracy = 0.0;
    double maximumPredictedNeutralProportion = 0.80;
    std::optional<double> minimumProfitability;
    std::optional<int> maximumPerSymbol;
    std::optional<int> maximumPerHorizon;
    std::optional<int> maximumPerSourceExperiment = 1;
    bool reconsiderRejectedWorkflows = false;
    bool reconsiderFailedWorkflows = false;
    bool reconsiderCancelledWorkflows = false;
    bool completedWorkflowsExcludeSelection = true;
    bool inconsistentWorkflowsAlwaysExclude = true;
    std::string tieBreaking =
        "ranking_global_ordinal_then_recommendation_id_then_ranking_member_id_"
        "then_canonical_evidence";
    // Empty preserves the source recommendation's mode. A non-empty value
    // explicitly requests one arm or the deterministic enabled/zero-ablation
    // pair.
    std::vector<Donchian20Mode> donchian20Arms;
    // Version 1 is retained only when reconstructing an older persisted
    // campaign policy that predates the arm field.
    int canonicalVersion = kRecommendationCampaignPlanningPolicyCanonicalVersion;
    int contractVersion = kRecommendationCampaignPlanContractVersion;
};

std::optional<std::string> ValidateRecommendationCampaignPlanningPolicy(
    const RecommendationCampaignPlanningPolicy& policy);
std::string RecommendationCampaignPlanningPolicyCanonicalText(
    const RecommendationCampaignPlanningPolicy& policy);
std::string RecommendationCampaignPlanningPolicyHash(
    const RecommendationCampaignPlanningPolicy& policy);
std::vector<Donchian20Mode> ParseRecommendationCampaignDonchian20Arms(
    const std::string& value);
std::string RecommendationCampaignDonchian20ArmsCanonicalText(
    const std::vector<Donchian20Mode>& arms);
RecommendationCampaignPlanningPolicy
ParseRecommendationCampaignPlanningPolicyCanonicalText(
    const std::string& canonical);

struct RecommendationCampaignPlanningScope
{
    long long rankingSnapshotId = -1;
    std::optional<std::string> symbol;
    std::optional<int> horizon;
    std::optional<long long> recommendationId;
};

std::optional<std::string> ValidateRecommendationCampaignPlanningScope(
    const RecommendationCampaignPlanningScope& scope);
std::string RecommendationCampaignPlanningScopeCanonicalText(
    const RecommendationCampaignPlanningScope& scope);
RecommendationCampaignPlanningScope
ParseRecommendationCampaignPlanningScopeCanonicalText(
    const std::string& canonical);

struct RecommendationCampaignWorkflowEvidence
{
    long long proposalId = -1;
    std::string proposalIdentityCanonical;
    std::string proposalIdentityHash;
    std::optional<long long> latestReviewDecisionId;
    std::optional<std::string> latestReviewDisposition;
    std::optional<long long> executionId;
    std::optional<std::string> executionIdentityCanonical;
    std::optional<std::string> executionIdentityHash;
    std::optional<long long> activationId;
    std::optional<std::string> activationIdentityCanonical;
    std::optional<std::string> activationIdentityHash;
    std::optional<long long> convertedExperimentId;
    RecommendationConversionWorkflowState state =
        RecommendationConversionWorkflowState::inconsistent;
    RecommendationConversionWorkflowIntegrity integrity =
        RecommendationConversionWorkflowIntegrity::inconsistent;
    std::vector<std::string> diagnosticCodes;
};

struct RecommendationCampaignCandidateInput
{
    long long rankingSnapshotId = -1;
    std::string rankingSnapshotIdentityCanonical;
    std::string rankingSnapshotIdentityHash;
    std::string rankingPolicyCanonical;
    std::string rankingPolicyHash;
    int rankingVersion = 0;
    long long rankingMemberId = -1;
    int rankingPosition = 0;
    std::string rankingBucket;
    std::optional<double> rankingScore;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    std::string family;
    std::optional<int> targetEpochs;
    double leaderScore = 0.0;
    double inferenceAccuracy = 0.0;
    std::optional<double> predictedNeutralProportion;
    std::optional<double> profitabilityMetric;
    std::optional<std::string> profitabilityMetricIdentity;
    std::string recommendationSemanticCanonical;
    std::string recommendationSemanticHash;
    std::string recommendationInvocationCanonical;
    std::string recommendationInvocationHash;
    std::optional<Donchian20Mode> campaignDonchian20Mode;
    bool persistedProvenanceValid = true;
    std::vector<RecommendationCampaignWorkflowEvidence> workflows;
};

enum class RecommendationCampaignDecision
{
    include,
    exclude
};

enum class RecommendationCampaignReason
{
    selected,
    planningDisabled,
    recommendationNotRanked,
    rankingNotAdvisoryReady,
    belowMinimumLeaderScore,
    belowMinimumInferenceAccuracy,
    predictedNeutralMetricUnavailable,
    predictedNeutralAboveMaximum,
    profitabilityMetricUnavailable,
    belowMinimumProfitability,
    proposalAlreadyExists,
    workflowPendingReview,
    workflowRejectedNotReconsiderable,
    workflowApprovedNotExecuted,
    workflowExecutedPaused,
    workflowActivatedOrSchedulerOwned,
    workflowCompleted,
    workflowFailedNotReconsiderable,
    workflowCancelledNotReconsiderable,
    workflowInconsistent,
    duplicateConversionIdentity,
    sourceExperimentConflict,
    symbolLimitReached,
    horizonLimitReached,
    sourceExperimentLimitReached,
    campaignLimitReached,
    candidateLimitReached,
    unsupportedContractVersion,
    identityValidationFailed
};

std::string RecommendationCampaignDecisionText(
    RecommendationCampaignDecision decision);
std::string RecommendationCampaignReasonText(
    RecommendationCampaignReason reason);
std::string RecommendationCampaignReasonExplanation(
    RecommendationCampaignReason reason);

struct RecommendationCampaignPlanCandidate
{
    int ordinal = 0;
    RecommendationCampaignCandidateInput input;
    RecommendationCampaignDecision decision =
        RecommendationCampaignDecision::exclude;
    std::vector<RecommendationCampaignReason> reasons;
};

struct RecommendationCampaignPlanSummary
{
    int candidateCount = 0;
    int selectedCount = 0;
    int excludedCount = 0;
};

struct RecommendationCampaignPlanInput
{
    RecommendationCampaignPlanningPolicy policy;
    RecommendationCampaignPlanningScope scope;
    std::string rankingSnapshotIdentityCanonical;
    std::string rankingSnapshotIdentityHash;
    std::string generatedAt;
    std::vector<RecommendationCampaignCandidateInput> candidates;
};

struct RecommendationCampaignPlan
{
    int contractVersion = kRecommendationCampaignPlanContractVersion;
    RecommendationCampaignPlanningPolicy policy;
    RecommendationCampaignPlanningScope scope;
    std::string policyCanonical;
    std::string policyHash;
    std::string scopeCanonical;
    std::string generatedAt;
    std::vector<RecommendationCampaignPlanCandidate> candidates;
    RecommendationCampaignPlanSummary summary;
    std::string identityCanonical;
    std::string identityHash;
};

RecommendationCampaignPlan PlanRecommendationCampaign(
    const RecommendationCampaignPlanInput& input);

bool RecommendationCampaignPlanOrderingIsDeterministic(
    const RecommendationCampaignPlan& plan);

} // namespace EA::ExperimentRecommendation
