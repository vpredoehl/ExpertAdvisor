#include "ExperimentRecommendationCampaignLaunchService.hpp"

#include "ExperimentRecommendationCampaignLaunchRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <map>
#include <ostream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void PrintOptionalId(std::ostream& output, const std::optional<long long>& value)
{
    if (value) output << *value;
    else output << "null";
}

void PrintOptionalText(
    std::ostream& output,
    const std::optional<std::string>& value)
{
    if (value) output << RecommendationMachineText(*value);
    else output << "null";
}

void PrintDiagnostics(
    std::ostream& output,
    const std::vector<std::string>& diagnostics)
{
    if (diagnostics.empty())
    {
        output << "none";
        return;
    }
    for (std::size_t index = 0; index < diagnostics.size(); ++index)
    {
        if (index != 0) output << '|';
        output << RecommendationMachineText(diagnostics[index]);
    }
}

void PrintSafety(
    std::ostream& output,
    bool dryRun,
    int executionCreated,
    int activationCreated,
    bool failed)
{
    const int experimentCreated = executionCreated;
    const int experimentUpdated = activationCreated;
    output << "dry_run=" << (dryRun ? "true" : "false")
           << ",write_confirmed=" << (!dryRun ? "true" : "false")
           << ",all_or_nothing=true,transaction_count="
           << (dryRun ? 0 : 1)
           << ",scheduler_started=false,scheduler_signaled=false,"
              "scheduler_polled=false,workers_launched=false,"
              "direct_training_started=false,direct_inference_started=false,"
              "direct_analysis_started=false,automatic_follow_up=false,"
              "campaign_state_row_created=false,experiment_rows_created="
           << experimentCreated
           << ",execution_rows_created=" << executionCreated
           << ",activation_rows_created=" << activationCreated
           << ",experiment_rows_updated=" << experimentUpdated
           << ",sequences_advanced=";
    if (failed && !dryRun) output << "unknown";
    else output << (experimentCreated + executionCreated + activationCreated);
}

std::string SuccessStatus(
    const RecommendationCampaignLaunchPlan& plan,
    int executionsCreated,
    int activationsCreated)
{
    if (plan.request.dryRun)
    {
        switch (plan.state)
        {
            case RecommendationCampaignLaunchPlanState::readyToLaunch:
                return "dry_run_ready_to_launch";
            case RecommendationCampaignLaunchPlanState::
                    readyToActivateExistingExecutions:
                return "dry_run_ready_to_activate_existing_executions";
            case RecommendationCampaignLaunchPlanState::alreadySatisfied:
                return "already_satisfied";
        }
    }
    if (executionsCreated > 0) return "newly_launched";
    if (activationsCreated > 0) return "activated_existing_executions";
    return "already_satisfied";
}

std::string DecisionText(const RecommendationCampaignLaunchPlan& plan)
{
    switch (plan.state)
    {
        case RecommendationCampaignLaunchPlanState::readyToLaunch:
            return "execute_and_activate";
        case RecommendationCampaignLaunchPlanState::
                readyToActivateExistingExecutions:
            return "activate_existing_executions";
        case RecommendationCampaignLaunchPlanState::alreadySatisfied:
            return "no_change";
    }
    throw std::invalid_argument("campaign_launch_plan_state_invalid");
}

void PrintResult(
    std::ostream& output,
    const RecommendationCampaignLaunchPlan& plan,
    const std::vector<PersistedRecommendationConversionExecution>& executions,
    const std::vector<PersistedRecommendationConversionActivation>& activations)
{
    std::map<long long, const PersistedRecommendationConversionExecution*>
        executionByProposal;
    for (const auto& execution : executions)
        executionByProposal.emplace(execution.proposalId, &execution);
    std::map<long long, const PersistedRecommendationConversionActivation*>
        activationByExecution;
    for (const auto& activation : activations)
        activationByExecution.emplace(activation.executionId, &activation);

    const int executionCreated = static_cast<int>(executions.size());
    const int activationCreated = static_cast<int>(activations.size());
    const int executionReused = plan.executionReuseCount;
    const int activationReused = plan.activationReuseCount;
    const std::string status = SuccessStatus(
        plan, executionCreated, activationCreated);
    output << "RECOMMENDATION_CAMPAIGN_LAUNCH"
           << ",materialization_id=" << plan.request.materializationId
           << ",materialization_contract_version="
           << plan.materializationContractVersion
           << ",materialization_identity_hash="
           << RecommendationMachineText(plan.materializationIdentityHash)
           << ",launch_contract_version="
           << kRecommendationCampaignLaunchContractVersion
           << ",launch_operation_identity_hash="
           << RecommendationMachineText(plan.operationIdentityHash)
           << ",status=" << status
           << ",decision=" << DecisionText(plan)
           << ",diagnostic=";
    PrintDiagnostics(output, plan.diagnosticCodes);
    output << ",member_count=" << plan.members.size()
           << ",execution_created_count=" << executionCreated
           << ",execution_reused_count=" << executionReused
           << ",activation_created_count=" << activationCreated
           << ",activation_reused_count=" << activationReused
           << ",already_satisfied_count=" << plan.alreadySatisfiedCount
           << ",blocked_count=0,";
    PrintSafety(
        output, plan.request.dryRun, executionCreated, activationCreated, false);
    output << '\n';

    for (const auto& member : plan.members)
    {
        std::optional<long long> authorization =
            member.authorizationReviewDecisionId;
        std::optional<long long> executionId = member.executionId;
        std::optional<long long> experimentId = member.experimentId;
        std::optional<long long> activationId = member.activationId;
        bool executionWasCreated = false;
        bool activationWasCreated = false;
        const auto createdExecution = executionByProposal.find(member.proposalId);
        if (createdExecution != executionByProposal.end())
        {
            executionWasCreated = true;
            authorization = createdExecution->second->reviewDecisionId;
            executionId = createdExecution->second->executionId;
            experimentId = createdExecution->second->experimentId;
        }
        if (executionId)
        {
            const auto createdActivation = activationByExecution.find(*executionId);
            if (createdActivation != activationByExecution.end())
            {
                activationWasCreated = true;
                activationId = createdActivation->second->activationId;
            }
        }
        output << "RECOMMENDATION_CAMPAIGN_LAUNCH_MEMBER"
               << ",materialization_id=" << plan.request.materializationId
               << ",member_ordinal=" << member.memberOrdinal
               << ",materialization_member_id="
               << member.materializationMemberId
               << ",recommendation_id=" << member.recommendationId
               << ",source_experiment_id=" << member.sourceExperimentId
               << ",ranking_member_id=" << member.rankingMemberId
               << ",proposal_id=" << member.proposalId
               << ",original_authorizing_review_id=";
        PrintOptionalId(output, authorization);
        output << ",execution_id=";
        PrintOptionalId(output, executionId);
        output << ",experiment_id=";
        PrintOptionalId(output, experimentId);
        output << ",activation_id=";
        PrintOptionalId(output, activationId);
        output << ",execution_disposition="
               << RecommendationCampaignLaunchExecutionDispositionText(
                      member.executionDisposition)
               << ",activation_disposition="
               << RecommendationCampaignLaunchActivationDispositionText(
                      member.activationDisposition)
               << ",execution_created="
               << (executionWasCreated ? "true" : "false")
               << ",activation_created="
               << (activationWasCreated ? "true" : "false")
               << ",pre_status=";
        PrintOptionalText(output, member.preStatus);
        output << ",pre_phase=";
        PrintOptionalText(output, member.prePhase);
        output << ",post_status=";
        PrintOptionalText(output, member.postStatus);
        output << ",post_phase=";
        PrintOptionalText(output, member.postPhase);
        output << ",diagnostic=";
        PrintDiagnostics(output, member.diagnosticCodes);
        output << ',';
        PrintSafety(
            output, plan.request.dryRun,
            executionWasCreated ? 1 : 0,
            activationWasCreated ? 1 : 0,
            false);
        output << '\n';
    }
}

void PrintFailure(
    std::ostream& errors,
    const RecommendationCampaignLaunchRequest& request,
    const std::string& materializationIdentityHash,
    int memberCount,
    const std::string& code,
    bool conflict)
{
    errors << (conflict ? "RECOMMENDATION_CAMPAIGN_LAUNCH_CONFLICT"
                        : "RECOMMENDATION_CAMPAIGN_LAUNCH_FAILED")
           << ",materialization_id=" << request.materializationId
           << ",materialization_contract_version=null"
           << ",materialization_identity_hash="
           << (materializationIdentityHash.empty()
                   ? "null"
                   : RecommendationMachineText(materializationIdentityHash))
           << ",launch_contract_version="
           << kRecommendationCampaignLaunchContractVersion
           << ",launch_operation_identity_hash=null,status="
           << (conflict ? "conflict" : "failed")
           << ",decision=no_change,diagnostic="
           << RecommendationMachineText(code)
           << ",member_count=" << memberCount
           << ",execution_created_count=0,execution_reused_count=0,"
              "activation_created_count=0,activation_reused_count=0,"
              "already_satisfied_count=0,blocked_count="
           << (memberCount > 0 ? memberCount : 0) << ',';
    PrintSafety(errors, request.dryRun, 0, 0, true);
    errors << '\n';
}

} // namespace

int RunRecommendationCampaignLaunchCommand(
    const std::string& connectionString,
    const RecommendationCampaignLaunchRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    int memberCount = 0;
    std::string materializationIdentityHash;
    try
    {
        const auto normalized = NormalizeRecommendationCampaignLaunchRequest(
            request);
        pqxx::connection connection{connectionString};
        if (normalized.dryRun)
        {
            pqxx::read_transaction transaction{connection};
            transaction.exec(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
            if (!RecommendationCampaignLaunchSchemasExist(transaction))
                throw std::runtime_error("campaign_launch_schemas_required");
            const auto materialization =
                LoadRecommendationCampaignLaunchMaterialization(
                    transaction, normalized.materializationId);
            if (!materialization)
                throw std::runtime_error(
                    "campaign_launch_materialization_not_found");
            memberCount = materialization->selectedMemberCount;
            materializationIdentityHash = materialization->identityHash;
            const auto plan = ValidateRecommendationCampaignLaunchDryRun(
                transaction, normalized, *materialization);
            PrintResult(output, plan, {}, {});
            return 0;
        }

        pqxx::work transaction{connection};
        if (!RecommendationCampaignLaunchSchemasExist(transaction))
            throw std::runtime_error("campaign_launch_schemas_required");
        const auto materialization =
            LoadRecommendationCampaignLaunchMaterialization(
                transaction, normalized.materializationId);
        if (!materialization)
            throw std::runtime_error(
                "campaign_launch_materialization_not_found");
        memberCount = materialization->selectedMemberCount;
        materializationIdentityHash = materialization->identityHash;
        if (materialization->members.size() > static_cast<std::size_t>(
                kMaximumRecommendationCampaignLaunchMembers))
            throw std::runtime_error("campaign_launch_member_limit_exceeded");
        auto result = LaunchRecommendationCampaignInTransaction(
            transaction, normalized, *materialization);
        transaction.commit();
        PrintResult(
            output, result.plan, result.createdExecutions,
            result.createdActivations);
        return 0;
    }
    catch (const std::exception& error)
    {
        const std::string code = error.what();
        const bool conflict = code.find("conflict") != std::string::npos;
        PrintFailure(
            errors, request, materializationIdentityHash, memberCount, code,
            conflict);
        return conflict ? 2 : 1;
    }
}

} // namespace EA::ExperimentRecommendation
