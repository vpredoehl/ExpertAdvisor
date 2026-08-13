#pragma once

#include "ExperimentRecommendationCampaignExecution.hpp"

#include <ostream>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationCampaignExecutionCommand(
    const std::string& connectionString,
    const RecommendationCampaignExecutionRequest& request,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
