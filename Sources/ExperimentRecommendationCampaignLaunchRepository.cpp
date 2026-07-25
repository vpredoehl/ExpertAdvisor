#include "ExperimentRecommendationCampaignLaunchRepository.hpp"

#include "ExperimentRecommendationCampaignActivationRepository.hpp"
#include "ExperimentRecommendationCampaignExecutionRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

RecommendationCampaignLaunchInput BuildInput(
    const PersistedRecommendationCampaignMaterialization& materialization,
    RecommendationCampaignExecutionPlan executionPlan,
    std::optional<RecommendationCampaignActivationPlan> activationPlan)
{
    RecommendationCampaignLaunchInput input;
    input.materializationId = materialization.materializationId;
    input.materializationContractVersion = materialization.contractVersion;
    input.materializationIdentityCanonical = materialization.identityCanonical;
    input.materializationIdentityHash = materialization.identityHash;
    input.selectedMemberCount = materialization.selectedMemberCount;
    input.executionPlan = std::move(executionPlan);
    input.activationPlan = std::move(activationPlan);
    input.members.reserve(materialization.members.size());
    for (const auto& member : materialization.members)
        input.members.push_back({
            member.memberOrdinal,
            member.materializationMemberId,
            member.rankingMemberId,
            member.recommendationId,
            member.sourceExperimentId,
            member.conversionProposalId});
    return input;
}

RecommendationCampaignExecutionPlan LoadExecutionPlan(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    const auto input = LoadRecommendationCampaignExecutionInput(
        transaction, materialization);
    return BuildRecommendationCampaignExecutionPlan(
        {request.materializationId, request.dryRun}, input);
}

RecommendationCampaignActivationPlan LoadActivationPlan(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    const auto input = LoadRecommendationCampaignActivationInput(
        transaction, materialization);
    return BuildRecommendationCampaignActivationPlan(
        {request.materializationId, request.dryRun}, input);
}

} // namespace

bool RecommendationCampaignLaunchSchemasExist(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(R"SQL(
SELECT to_regclass('experiment_recommendation_campaign_materialization') IS NOT NULL
   AND to_regclass('experiment_recommendation_campaign_materialization_member') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_proposal') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_review_decision') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_execution') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_activation') IS NOT NULL
   AND to_regclass('experiment') IS NOT NULL
   AND pg_get_serial_sequence('experiment','experiment_id') IS NOT NULL
   AND pg_get_serial_sequence(
       'experiment_recommendation_conversion_execution',
       'recommendation_conversion_execution_id') IS NOT NULL
   AND pg_get_serial_sequence(
       'experiment_recommendation_conversion_activation',
       'recommendation_conversion_activation_id') IS NOT NULL;
)SQL").one_row()[0].as<bool>();
}

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignLaunchMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId)
{
    if (materializationId <= 0)
        throw std::invalid_argument("campaign_launch_materialization_id_invalid");
    return FindRecommendationCampaignMaterialization(
        transaction, materializationId);
}

RecommendationCampaignLaunchPlan ValidateRecommendationCampaignLaunchDryRun(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    if (!request.dryRun)
        throw std::invalid_argument("campaign_launch_dry_run_required");
    const auto executionPlan = LoadExecutionPlan(
        transaction, request, materialization);
    if (executionPlan.state == RecommendationCampaignExecutionPlanState::ready)
        return BuildRecommendationCampaignLaunchPlan(
            request, BuildInput(materialization, executionPlan, std::nullopt));
    const auto activationPlan = LoadActivationPlan(
        transaction, request, materialization);
    return BuildRecommendationCampaignLaunchPlan(
        request, BuildInput(materialization, executionPlan, activationPlan));
}

RecommendationCampaignLaunchPersistResult
LaunchRecommendationCampaignInTransaction(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    return LaunchRecommendationCampaignInTransaction(
        transaction, request, materialization, {});
}

RecommendationCampaignLaunchPersistResult
LaunchRecommendationCampaignInTransaction(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignLaunchRequest& request,
    const PersistedRecommendationCampaignMaterialization& materialization,
    RecommendationCampaignLaunchTestHook testHook)
{
    if (request.dryRun)
        throw std::invalid_argument("campaign_launch_write_request_required");

    // Global lock hierarchy: all proposal review/execution advisory keys in
    // ascending proposal ID, followed by all activation advisory keys in
    // ascending execution ID, followed by experiment rows in ascending ID.
    LockRecommendationCampaignExecutions(transaction, materialization);
    const auto executionPlan = LoadExecutionPlan(
        transaction, request, materialization);

    RecommendationCampaignLaunchPersistResult result;
    if (executionPlan.state == RecommendationCampaignExecutionPlanState::ready)
    {
        // Validate the complete immutable membership/execution phase before
        // the first experiment or execution insert.
        (void)BuildRecommendationCampaignLaunchPlan(
            request, BuildInput(materialization, executionPlan, std::nullopt));
        result.createdExecutions = PersistRecommendationCampaignExecutions(
            transaction, executionPlan);
        if (result.createdExecutions.size() != executionPlan.members.size())
            throw std::runtime_error(
                "campaign_launch_execution_count_mismatch");
        if (testHook)
        {
            testHook(
                RecommendationCampaignLaunchTestPoint::
                    afterExecutionMutation);
            testHook(
                RecommendationCampaignLaunchTestPoint::
                    afterExperimentCreationOrReuseMutation);
        }
    }
    else if (testHook)
        testHook(
            RecommendationCampaignLaunchTestPoint::
                afterExperimentCreationOrReuseMutation);

    // Newly created executions are visible in this transaction. The existing
    // Step 2 lock helper resolves the complete set, sorts activation advisory
    // keys and experiment row IDs, and acquires both established domains.
    LockRecommendationCampaignActivations(transaction, materialization);
    const auto activationPlan = LoadActivationPlan(
        transaction, request, materialization);
    result.plan = BuildRecommendationCampaignLaunchPlan(
        request, BuildInput(materialization, executionPlan, activationPlan));
    if (activationPlan.state == RecommendationCampaignActivationPlanState::ready)
    {
        result.createdActivations = PersistRecommendationCampaignActivations(
            transaction, activationPlan);
        if (result.createdActivations.size() != activationPlan.members.size())
            throw std::runtime_error(
                "campaign_launch_activation_count_mismatch");
        if (testHook)
            testHook(
                RecommendationCampaignLaunchTestPoint::
                    afterActivationMutation);
    }
    return result;
}

} // namespace EA::ExperimentRecommendation
