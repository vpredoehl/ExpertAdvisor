#pragma once

#include "ExperimentRecommendationCampaignActivation.hpp"
#include "ExperimentRecommendationCampaignExecution.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignLaunchContractVersion = 1;
inline constexpr int kMaximumRecommendationCampaignLaunchMembers = 1000;

struct RecommendationCampaignLaunchRequest
{
    long long materializationId = -1;
    bool dryRun = false;
};

struct RecommendationCampaignLaunchMemberIdentity
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    long long proposalId = -1;
};

struct RecommendationCampaignLaunchInput
{
    long long materializationId = -1;
    int materializationContractVersion = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;
    int selectedMemberCount = 0;
    std::vector<RecommendationCampaignLaunchMemberIdentity> members;
    RecommendationCampaignExecutionPlan executionPlan;
    std::optional<RecommendationCampaignActivationPlan> activationPlan;
};

enum class RecommendationCampaignLaunchPlanState
{
    readyToLaunch,
    readyToActivateExistingExecutions,
    alreadySatisfied
};

enum class RecommendationCampaignLaunchExecutionDisposition
{
    create,
    reuse
};

enum class RecommendationCampaignLaunchActivationDisposition
{
    createAfterExecution,
    create,
    reuse
};

struct RecommendationCampaignLaunchMemberPlan
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    long long proposalId = -1;
    std::optional<long long> authorizationReviewDecisionId;
    std::optional<long long> executionId;
    std::optional<long long> experimentId;
    std::optional<long long> activationId;
    RecommendationCampaignLaunchExecutionDisposition executionDisposition =
        RecommendationCampaignLaunchExecutionDisposition::create;
    RecommendationCampaignLaunchActivationDisposition activationDisposition =
        RecommendationCampaignLaunchActivationDisposition::createAfterExecution;
    std::optional<std::string> preStatus;
    std::optional<std::string> prePhase;
    std::optional<std::string> postStatus;
    std::optional<std::string> postPhase;
    std::vector<std::string> diagnosticCodes;
};

struct RecommendationCampaignLaunchPlan
{
    RecommendationCampaignLaunchRequest request;
    int materializationContractVersion = 0;
    std::string materializationIdentityHash;
    std::string operationIdentityCanonical;
    std::string operationIdentityHash;
    RecommendationCampaignLaunchPlanState state =
        RecommendationCampaignLaunchPlanState::readyToLaunch;
    int membersValidated = 0;
    int executionCreateCount = 0;
    int executionReuseCount = 0;
    int activationCreateCount = 0;
    int activationReuseCount = 0;
    int alreadySatisfiedCount = 0;
    std::vector<RecommendationCampaignLaunchMemberPlan> members;
    std::vector<std::string> diagnosticCodes;
};

RecommendationCampaignLaunchRequest NormalizeRecommendationCampaignLaunchRequest(
    const RecommendationCampaignLaunchRequest& request);

RecommendationCampaignLaunchPlan BuildRecommendationCampaignLaunchPlan(
    const RecommendationCampaignLaunchRequest& request,
    const RecommendationCampaignLaunchInput& input);

std::string RecommendationCampaignLaunchPlanStateText(
    RecommendationCampaignLaunchPlanState state);
std::string RecommendationCampaignLaunchExecutionDispositionText(
    RecommendationCampaignLaunchExecutionDisposition disposition);
std::string RecommendationCampaignLaunchActivationDispositionText(
    RecommendationCampaignLaunchActivationDisposition disposition);

} // namespace EA::ExperimentRecommendation
