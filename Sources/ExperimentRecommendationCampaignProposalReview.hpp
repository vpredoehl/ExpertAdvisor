#pragma once

#include "ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignProposalReviewContractVersion = 1;
inline constexpr int kMaximumRecommendationCampaignProposalReviewMembers = 1000;

struct RecommendationCampaignProposalReviewRequest
{
    long long materializationId = -1;
    RecommendationConversionProposalReviewDecision decision =
        RecommendationConversionProposalReviewDecision::approve;
    std::string operatorIdentity;
    std::string reasonText;
    bool dryRun = false;
};

struct RecommendationCampaignProposalReviewMemberInput
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long proposalId = -1;
    std::optional<PersistedRecommendationConversionProposalReviewDecision>
        authoritativeReview;
};

struct RecommendationCampaignProposalReviewInput
{
    long long materializationId = -1;
    int materializationContractVersion = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;
    int selectedMemberCount = 0;
    bool materializationComplete = false;
    bool workflowEvidenceConsistent = false;
    std::vector<RecommendationCampaignProposalReviewMemberInput> members;
};

enum class RecommendationCampaignProposalReviewMemberAction
{
    insert,
    alreadySatisfied
};

enum class RecommendationCampaignProposalReviewPlanState
{
    ready,
    alreadySatisfied
};

struct RecommendationCampaignProposalReviewMemberPlan
{
    int memberOrdinal = 0;
    long long materializationMemberId = -1;
    long long proposalId = -1;
    std::optional<long long> previousReviewDecisionId;
    std::optional<RecommendationConversionProposalReviewDisposition>
        previousDisposition;
    RecommendationCampaignProposalReviewMemberAction action =
        RecommendationCampaignProposalReviewMemberAction::insert;
    std::vector<std::string> diagnosticCodes;
};

struct RecommendationCampaignProposalReviewPlan
{
    RecommendationCampaignProposalReviewRequest request;
    std::string materializationIdentityHash;
    std::string operationIdentityCanonical;
    std::string operationIdentityHash;
    std::string phase4cRequestId;
    RecommendationCampaignProposalReviewPlanState state =
        RecommendationCampaignProposalReviewPlanState::ready;
    std::vector<RecommendationCampaignProposalReviewMemberPlan> members;
    int proposalsValidated = 0;
    int alreadySatisfiedCount = 0;
    int conflictCount = 0;
    std::vector<std::string> diagnosticCodes;
};

std::string RecommendationCampaignProposalReviewPlanStateText(
    RecommendationCampaignProposalReviewPlanState state);
std::string RecommendationCampaignProposalReviewMemberActionText(
    RecommendationCampaignProposalReviewMemberAction action);

RecommendationCampaignProposalReviewRequest
NormalizeRecommendationCampaignProposalReviewRequest(
    const RecommendationCampaignProposalReviewRequest& request);

RecommendationCampaignProposalReviewPlan
BuildRecommendationCampaignProposalReviewPlan(
    const RecommendationCampaignProposalReviewRequest& request,
    const RecommendationCampaignProposalReviewInput& input);

} // namespace EA::ExperimentRecommendation
