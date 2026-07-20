#pragma once

#include "ExperimentRecommendationCampaignStatus.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationCampaignStatusCommand(
    const std::string& connectionString,
    const RecommendationCampaignStatusRequest& request,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
