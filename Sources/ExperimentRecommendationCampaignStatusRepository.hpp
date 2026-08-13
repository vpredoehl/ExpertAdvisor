#pragma once

#include "ExperimentRecommendationCampaignStatus.hpp"

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignStatusSchemasExist(
    pqxx::transaction_base& transaction);

RecommendationCampaignStatusSnapshot LoadRecommendationCampaignStatusSnapshot(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignStatusRequest& request);

RecommendationCampaignStatusSnapshot ReadRecommendationCampaignStatusSnapshot(
    pqxx::connection& connection,
    const RecommendationCampaignStatusRequest& request);

} // namespace EA::ExperimentRecommendation
