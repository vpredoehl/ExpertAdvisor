#pragma once

#include "ExperimentRecommendationCampaignOutcomeAssessment.hpp"
#include "ExperimentRecommendationCampaignStatus.hpp"

#include <pqxx/pqxx>

#include <optional>
#include <vector>

namespace EA::ExperimentRecommendation
{

struct RecommendationCampaignOutcomeAssessmentRequest
{
    long long materializationId = -1;
};

struct RecommendationCampaignOutcomeAssessmentScientificMemberEvidence
{
    int memberOrdinal = 0;
    RecommendationCampaignOutcomeAssessmentConsistencyState inputConsistency =
        RecommendationCampaignOutcomeAssessmentConsistencyState::Consistent;
    std::optional<RecommendationCampaignOutcomeAssessmentSourceEvidence>
        sourceEvidence;
    std::optional<RecommendationCampaignOutcomeAssessmentResultEvidence>
        resultEvidence;
};

struct RecommendationCampaignOutcomeAssessmentEvidenceSnapshot
{
    RecommendationCampaignOutcomeAssessmentCampaignIdentity campaignIdentity;
    RecommendationCampaignOutcomeAssessmentMaterializationIdentity
        materializationIdentity;
    RecommendationCampaignStatusSnapshot statusSnapshot;
    std::vector<
        RecommendationCampaignOutcomeAssessmentScientificMemberEvidence>
        scientificMembers;
};

RecommendationCampaignOutcomeAssessmentRequest
NormalizeRecommendationCampaignOutcomeAssessmentRequest(
    const RecommendationCampaignOutcomeAssessmentRequest& request);

bool RecommendationCampaignOutcomeAssessmentSchemasExist(
    pqxx::transaction_base& transaction);

RecommendationCampaignOutcomeAssessmentEvidenceSnapshot
LoadRecommendationCampaignOutcomeAssessmentEvidence(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignOutcomeAssessmentRequest& request);

RecommendationCampaignOutcomeAssessmentEvidenceSnapshot
ReadRecommendationCampaignOutcomeAssessmentEvidence(
    pqxx::connection& connection,
    const RecommendationCampaignOutcomeAssessmentRequest& request);

} // namespace EA::ExperimentRecommendation
