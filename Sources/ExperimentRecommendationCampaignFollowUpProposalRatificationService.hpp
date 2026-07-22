#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp"

#include <string>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

struct RecommendationCampaignFollowUpProposalRatificationRequest
{
    long long reviewEventId = 0;
    std::string expectedReviewIdentityHash;
    std::string ratifierIdentity;
    std::string ratificationBasis;
};

RecommendationCampaignFollowUpProposalRatificationRequest
ValidateRecommendationCampaignFollowUpProposalRatificationRequest(
    const RecommendationCampaignFollowUpProposalRatificationRequest& request);

RecommendationCampaignFollowUpProposalRatificationPersistResult
RatifyRecommendationCampaignFollowUpProposal(
    pqxx::connection& connection,
    const RecommendationCampaignFollowUpProposalRatificationRequest& request);

} // namespace EA::ExperimentRecommendation
