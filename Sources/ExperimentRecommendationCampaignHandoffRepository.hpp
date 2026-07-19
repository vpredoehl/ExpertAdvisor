#pragma once

#include "ExperimentRecommendationCampaignHandoff.hpp"

#include <optional>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignHandoffSchemasExist(
    pqxx::transaction_base& transaction);

std::optional<RecommendationCampaignHandoff>
FindRecommendationCampaignHandoff(
    pqxx::transaction_base& transaction,
    long long materializationId);

std::vector<RecommendationCampaignHandoff>
ListRecommendationCampaignHandoffs(
    pqxx::transaction_base& transaction,
    int limit = kDefaultRecommendationCampaignHandoffListLimit);

} // namespace EA::ExperimentRecommendation
