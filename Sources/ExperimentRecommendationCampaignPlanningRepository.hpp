#pragma once

#include "ExperimentRecommendationCampaignPlanning.hpp"

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignPlanningSchemasExist(pqxx::connection& connection);

RecommendationCampaignPlanInput LoadRecommendationCampaignPlanInput(
    pqxx::connection& connection,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope);

} // namespace EA::ExperimentRecommendation
