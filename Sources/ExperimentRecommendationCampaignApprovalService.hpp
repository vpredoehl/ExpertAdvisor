#pragma once

#include "ExperimentRecommendationCampaignApproval.hpp"
#include "ExperimentRecommendationCampaignApprovalRepository.hpp"
#include "ExperimentRecommendationCampaignPlanning.hpp"

#include <optional>
#include <ostream>
#include <string>

namespace EA::ExperimentRecommendation
{

int RunRecordRecommendationCampaignApprovalCommand(
    const std::string& connectionString,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope,
    const RecommendationCampaignApprovalRequest& request,
    std::ostream& output,
    std::ostream& errors);

int RunShowRecommendationCampaignApprovalCommand(
    const std::string& connectionString,
    long long campaignApprovalId,
    std::ostream& output,
    std::ostream& errors);

int RunListRecommendationCampaignApprovalsCommand(
    const std::string& connectionString,
    std::optional<RecommendationCampaignApprovalDecision> decision,
    int limit,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
