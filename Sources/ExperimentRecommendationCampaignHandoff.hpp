#pragma once

#include "ExperimentRecommendationConversionWorkflow.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kDefaultRecommendationCampaignHandoffListLimit = 100;
inline constexpr int kMaximumRecommendationCampaignHandoffListLimit = 1000;

enum class RecommendationCampaignHandoffState
{
    awaitingPhase4cReview,
    partiallyReviewed,
    reviewRejected,
    readyForPhase4cExecution,
    partiallyExecuted,
    readyForPhase4cActivation,
    partiallyActivated,
    fullyActivated,
    inconsistent
};

enum class RecommendationCampaignHandoffReviewStatus
{
    awaitingReview,
    approved,
    rejected
};

enum class RecommendationCampaignHandoffExecutionStatus
{
    notExecuted,
    executed
};

enum class RecommendationCampaignHandoffActivationStatus
{
    notActivated,
    activated
};

enum class RecommendationCampaignHandoffIntegrity
{
    consistent,
    inconsistent
};

struct RecommendationCampaignHandoffProposalFacts
{
    long long proposalId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    std::string identityCanonical;
    std::string identityHash;
    std::optional<RecommendationConversionWorkflowReviewFact> latestReview;
    std::optional<long long> executionId;
    std::optional<long long> activationId;
    RecommendationConversionWorkflowDerivation workflow;
};

struct RecommendationCampaignHandoffMemberInput
{
    int memberOrdinal = 0;
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    int rankingPosition = 0;
    std::string selectedMemberIdentityCanonical;
    std::string selectedMemberIdentityHash;
    long long conversionProposalId = -1;
    std::string proposalIdentityCanonical;
    std::string proposalIdentityHash;
    std::optional<RecommendationCampaignHandoffProposalFacts> proposal;
};

struct RecommendationCampaignHandoffInput
{
    long long materializationId = -1;
    long long campaignApprovalId = -1;
    int materializationContractVersion = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;
    int selectedMemberCount = 0;
    std::vector<RecommendationCampaignHandoffMemberInput> members;
};

struct RecommendationCampaignHandoffMember
{
    int memberOrdinal = 0;
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    int rankingPosition = 0;
    long long conversionProposalId = -1;
    std::string proposalIdentityHash;
    RecommendationCampaignHandoffReviewStatus reviewStatus =
        RecommendationCampaignHandoffReviewStatus::awaitingReview;
    std::optional<long long> reviewDecisionId;
    RecommendationCampaignHandoffExecutionStatus executionStatus =
        RecommendationCampaignHandoffExecutionStatus::notExecuted;
    std::optional<long long> executionId;
    RecommendationCampaignHandoffActivationStatus activationStatus =
        RecommendationCampaignHandoffActivationStatus::notActivated;
    std::optional<long long> activationId;
    RecommendationConversionWorkflowState workflowState =
        RecommendationConversionWorkflowState::inconsistent;
    RecommendationCampaignHandoffIntegrity integrity =
        RecommendationCampaignHandoffIntegrity::consistent;
    std::vector<std::string> diagnosticCodes;
};

struct RecommendationCampaignHandoffSummary
{
    int totalMembers = 0;
    int proposalsPresent = 0;
    int awaitingReview = 0;
    int approved = 0;
    int rejected = 0;
    int noAuthoritativeReview = 0;
    int executionsPresent = 0;
    int activationsPresent = 0;
    int integrityFailures = 0;
};

struct RecommendationCampaignHandoff
{
    long long materializationId = -1;
    long long campaignApprovalId = -1;
    std::string materializationIdentityHash;
    bool materializationComplete = false;
    RecommendationCampaignHandoffState state =
        RecommendationCampaignHandoffState::inconsistent;
    RecommendationCampaignHandoffIntegrity integrity =
        RecommendationCampaignHandoffIntegrity::inconsistent;
    RecommendationCampaignHandoffSummary summary;
    std::vector<RecommendationCampaignHandoffMember> members;
};

std::string RecommendationCampaignHandoffStateText(
    RecommendationCampaignHandoffState state);
std::string RecommendationCampaignHandoffReviewStatusText(
    RecommendationCampaignHandoffReviewStatus status);
std::string RecommendationCampaignHandoffExecutionStatusText(
    RecommendationCampaignHandoffExecutionStatus status);
std::string RecommendationCampaignHandoffActivationStatusText(
    RecommendationCampaignHandoffActivationStatus status);
std::string RecommendationCampaignHandoffIntegrityText(
    RecommendationCampaignHandoffIntegrity integrity);

RecommendationCampaignHandoff BuildRecommendationCampaignHandoff(
    const RecommendationCampaignHandoffInput& input);

} // namespace EA::ExperimentRecommendation
