#pragma once

#include "ExperimentRecommendationCampaignActivation.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecommendationCampaignActivationCommand(
    const std::string& connectionString,
    const RecommendationCampaignActivationRequest& request,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
