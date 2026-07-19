#pragma once

#include "ExperimentRecommendationCampaignExecution.hpp"
#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"
#include "ExperimentRecommendationConversionExecutionRepository.hpp"

#include <optional>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignExecutionSchemasExist(
    pqxx::transaction_base& transaction);

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignExecutionMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId);

void LockRecommendationCampaignExecutions(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization);

RecommendationCampaignExecutionInput LoadRecommendationCampaignExecutionInput(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization);

std::vector<PersistedRecommendationConversionExecution>
PersistRecommendationCampaignExecutions(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignExecutionPlan& plan);

} // namespace EA::ExperimentRecommendation
