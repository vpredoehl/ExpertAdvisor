#pragma once

#include "ExperimentRecommendationCampaignHandoff.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunShowRecommendationCampaignHandoffCommand(
    const std::string& connectionString,
    long long materializationId,
    std::ostream& output,
    std::ostream& errors);

int RunListRecommendationCampaignHandoffsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
