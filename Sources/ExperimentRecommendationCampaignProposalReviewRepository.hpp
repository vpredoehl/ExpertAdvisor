#pragma once

#include "ExperimentRecommendationCampaignProposalReview.hpp"
#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"

#include <optional>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignProposalReviewSchemasExist(
    pqxx::transaction_base& transaction);

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignProposalReviewMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId);

void LockRecommendationCampaignProposalReviews(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization);

RecommendationCampaignProposalReviewInput
LoadRecommendationCampaignProposalReviewInput(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization);

std::vector<PersistedRecommendationConversionProposalReviewDecision>
PersistRecommendationCampaignProposalReviews(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignProposalReviewPlan& plan);

} // namespace EA::ExperimentRecommendation
