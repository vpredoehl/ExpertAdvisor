#pragma once

#include "ExperimentRecommendationCampaignLaunch.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationCampaignLaunchCommand(
    const std::string& connectionString,
    const RecommendationCampaignLaunchRequest& request,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
