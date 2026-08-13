#pragma once

#include "ExperimentRecommendationCampaignPlanning.hpp"

#include <ostream>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationCampaignPlanningCommand(
    const std::string& connectionString,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
