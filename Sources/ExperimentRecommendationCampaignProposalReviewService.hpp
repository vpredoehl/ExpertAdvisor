#pragma once

#include "ExperimentRecommendationCampaignProposalReview.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationCampaignProposalReviewCommand(
    const std::string& connectionString,
    const RecommendationCampaignProposalReviewRequest& request,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
