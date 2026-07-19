#pragma once

#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignExecutionContractVersion = 1;
inline constexpr int kMaximumRecommendationCampaignExecutionMembers = 1000;

struct RecommendationCampaignExecutionRequest
{
    long long materializationId = -1;
    bool dryRun = false;
};

struct RecommendationCampaignExecutionMemberInput
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long proposalId = -1;
    std::optional<long long> authorizationReviewDecisionId;
    bool authoritativeReviewApproved = false;
    std::optional<long long> executionId;
    std::optional<long long> experimentId;
};

struct RecommendationCampaignExecutionInput
{
    long long materializationId = -1;
    int materializationContractVersion = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;
    int selectedMemberCount = 0;
    bool materializationComplete = false;
    bool workflowEvidenceConsistent = false;
    std::vector<RecommendationCampaignExecutionMemberInput> members;
};

enum class RecommendationCampaignExecutionMemberAction
{
    create,
    alreadySatisfied
};

enum class RecommendationCampaignExecutionPlanState
{
    ready,
    alreadySatisfied
};

struct RecommendationCampaignExecutionMemberPlan
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long proposalId = -1;
    std::optional<long long> authorizationReviewDecisionId;
    std::optional<long long> previousExecutionId;
    std::optional<long long> previousExperimentId;
    RecommendationCampaignExecutionMemberAction action =
        RecommendationCampaignExecutionMemberAction::create;
    std::vector<std::string> diagnosticCodes;
};

struct RecommendationCampaignExecutionPlan
{
    RecommendationCampaignExecutionRequest request;
    std::string materializationIdentityHash;
    std::string operationIdentityCanonical;
    std::string operationIdentityHash;
    RecommendationCampaignExecutionPlanState state =
        RecommendationCampaignExecutionPlanState::ready;
    int proposalsValidated = 0;
    int alreadySatisfiedCount = 0;
    std::vector<RecommendationCampaignExecutionMemberPlan> members;
    std::vector<std::string> diagnosticCodes;
};

std::string RecommendationCampaignExecutionPlanStateText(
    RecommendationCampaignExecutionPlanState state);
std::string RecommendationCampaignExecutionMemberActionText(
    RecommendationCampaignExecutionMemberAction action);

RecommendationCampaignExecutionRequest NormalizeRecommendationCampaignExecutionRequest(
    const RecommendationCampaignExecutionRequest& request);

RecommendationCampaignExecutionPlan BuildRecommendationCampaignExecutionPlan(
    const RecommendationCampaignExecutionRequest& request,
    const RecommendationCampaignExecutionInput& input);

} // namespace EA::ExperimentRecommendation
