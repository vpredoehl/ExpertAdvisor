#pragma once

#include "ExperimentRecommendationCampaignLaunch.hpp"
#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"
#include "ExperimentRecommendationConversionActivationRepository.hpp"
#include "ExperimentRecommendationConversionExecutionRepository.hpp"

#include <functional>
#include <optional>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

struct RecommendationCampaignLaunchPersistResult
{
    RecommendationCampaignLaunchPlan plan;
    std::vector<PersistedRecommendationConversionExecution> createdExecutions;
    std::vector<PersistedRecommendationConversionActivation> createdActivations;
};

enum class RecommendationCampaignLaunchTestPoint
{
    afterExecutionMutation,
    afterExperimentCreationOrReuseMutation,
    afterActivationMutation
};

using RecommendationCampaignLaunchTestHook =
    std::function<void(RecommendationCampaignLaunchTestPoint)>;

bool RecommendationCampaignLaunchSchemasExist(
    pqxx::transaction_base& transaction);

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignLaunchMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId);

RecommendationCampaignLaunchPlan ValidateRecommendationCampaignLaunchDryRun(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization);

// The caller owns the single write transaction. This function acquires the
// existing Phase 4C/Phase 5 lock domains, starts no transaction, commits no
// transaction, and propagates every failure to the outer owner.
RecommendationCampaignLaunchPersistResult
LaunchRecommendationCampaignInTransaction(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization);

// Verification-only overload.  Ordinary Phase 5 and production callers use
// the overload above, which supplies no hook.
RecommendationCampaignLaunchPersistResult
LaunchRecommendationCampaignInTransaction(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization,
    RecommendationCampaignLaunchTestHook testHook);

} // namespace EA::ExperimentRecommendation
