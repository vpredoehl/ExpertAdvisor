#pragma once

#include "ExperimentRecommendationCampaignApproval.hpp"
#include "ExperimentRecommendationConversion.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignMaterializationContractVersion = 1;
inline constexpr std::size_t
    kRecommendationCampaignMaterializationOperatorMaximum = 200;
inline constexpr std::size_t
    kRecommendationCampaignMaterializationReasonMaximum = 2000;
inline constexpr std::size_t
    kRecommendationCampaignMaterializationIdentityCanonicalMaximum =
        64 * 1024 * 1024;

struct RecommendationCampaignMaterializationRequest
{
    long long campaignApprovalId = -1;
    std::string operatorIdentity;
    std::string reasonText;
};

struct RecommendationCampaignMaterializationSelectedMember
{
    int memberOrdinal = 0;
    long long rankingMemberId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    int rankingPosition = 0;
    std::string selectedMemberIdentityCanonical;
    std::string selectedMemberIdentityHash;
    ProposedExperimentSpecification proposal;
};

struct RecommendationCampaignMaterializationEvidence
{
    int materializationContractVersion =
        kRecommendationCampaignMaterializationContractVersion;
    long long campaignApprovalId = -1;
    RecommendationCampaignApprovalEvidence approval;
    std::string operatorIdentity;
    std::string reasonText;
    int selectedMemberCount = 0;
    std::vector<RecommendationCampaignMaterializationSelectedMember> members;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;
};

RecommendationCampaignMaterializationRequest
NormalizeRecommendationCampaignMaterializationRequest(
    const RecommendationCampaignMaterializationRequest& request);

RecommendationCampaignMaterializationEvidence
BuildRecommendationCampaignMaterializationEvidence(
    const RecommendationCampaignMaterializationRequest& request,
    const RecommendationCampaignApprovalEvidence& approval,
    const RecommendationCampaignPlan& plan,
    const std::vector<ProposedExperimentSpecification>& proposals);

void ValidateRecommendationCampaignMaterializationEvidence(
    const RecommendationCampaignMaterializationEvidence& evidence);

} // namespace EA::ExperimentRecommendation
