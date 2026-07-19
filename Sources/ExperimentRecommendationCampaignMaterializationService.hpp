#pragma once

#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"

#include <optional>
#include <ostream>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunMaterializeRecommendationCampaignCommand(
    const std::string& connectionString,
    const RecommendationCampaignMaterializationRequest& request,
    std::ostream& output,
    std::ostream& errors);

int RunShowRecommendationCampaignMaterializationCommand(
    const std::string& connectionString,
    long long materializationId,
    std::ostream& output,
    std::ostream& errors);

int RunListRecommendationCampaignMaterializationsCommand(
    const std::string& connectionString,
    std::optional<long long> campaignApprovalId,
    int limit,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
