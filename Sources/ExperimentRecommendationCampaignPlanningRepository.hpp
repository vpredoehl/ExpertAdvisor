#pragma once

#include "ExperimentRecommendationCampaignPlanning.hpp"

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignPlanningSchemasExist(pqxx::connection& connection);
bool RecommendationCampaignPlanningSchemasExist(
    pqxx::transaction_base& transaction);

RecommendationCampaignPlanInput LoadRecommendationCampaignPlanInput(
    pqxx::connection& connection,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope);
RecommendationCampaignPlanInput LoadRecommendationCampaignPlanInput(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope);

} // namespace EA::ExperimentRecommendation
