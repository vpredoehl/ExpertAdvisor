#pragma once

#include "ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentRecommendation
{

void WriteRecommendationCampaignFollowUpProposalPreview(
    std::ostream& output,
    const PersistedRecommendationCampaignFollowUpProposal& persisted);

int RunPreviewRecommendationCampaignFollowUpProposalCommand(
    const std::string& connectionString,
    long long followUpProposalId,
    std::ostream& output,
    std::ostream& errors);

} // namespace EA::ExperimentRecommendation
