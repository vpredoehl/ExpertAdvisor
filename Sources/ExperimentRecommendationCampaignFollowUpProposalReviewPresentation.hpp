#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

void WriteRecommendationCampaignFollowUpProposalReview(
    std::ostream& output,
    const PersistedRecommendationCampaignFollowUpProposalReview& persisted);

int RunShowRecommendationCampaignFollowUpProposalReview(
    const std::string& connectionString,
    long long reviewEventId,
    std::ostream& output,
    std::ostream& errors);

int RunListRecommendationCampaignFollowUpProposalReviews(
    const std::string& connectionString,
    int limit,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
