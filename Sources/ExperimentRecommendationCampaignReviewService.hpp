#pragma once

#include "ExperimentRecommendationCampaignPlanning.hpp"

#include <ostream>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationCampaignReviewCommand(
    const std::string& connectionString,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
