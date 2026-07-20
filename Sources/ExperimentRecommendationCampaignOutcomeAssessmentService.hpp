#pragma once

#include "ExperimentRecommendationCampaignOutcomeAssessmentRepository.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

RecommendationCampaignOutcomeAssessmentLifecycleState
RecommendationCampaignOutcomeAssessmentLifecycleFromCampaignStatus(
    RecommendationCampaignStatusTerminalResult terminalResult);

RecommendationCampaignOutcomeAssessment
BuildRecommendationCampaignOutcomeAssessmentFromEvidence(
    const RecommendationCampaignOutcomeAssessmentEvidenceSnapshot& evidence);

void WriteRecommendationCampaignOutcomeAssessment(
    std::ostream& output,
    const RecommendationCampaignOutcomeAssessment& assessment);

int RunRecommendationCampaignOutcomeAssessmentCommand(
    const std::string& connectionString,
    const RecommendationCampaignOutcomeAssessmentRequest& request,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
