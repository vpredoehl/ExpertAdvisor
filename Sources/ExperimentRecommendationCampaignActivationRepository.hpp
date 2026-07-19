#pragma once

#include "ExperimentRecommendationCampaignActivation.hpp"
#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"
#include "ExperimentRecommendationConversionActivationRepository.hpp"

#include <optional>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignActivationSchemasExist(
    pqxx::transaction_base& transaction);

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignActivationMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId);

void LockRecommendationCampaignActivations(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization);

RecommendationCampaignActivationInput LoadRecommendationCampaignActivationInput(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization);

std::vector<PersistedRecommendationConversionActivation>
PersistRecommendationCampaignActivations(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignActivationPlan& plan);

} // namespace EA::ExperimentRecommendation
