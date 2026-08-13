#include "ExperimentRecommendationCampaignExecutionService.hpp"

#include "ExperimentRecommendationCampaignExecutionRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <map>
#include <ostream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void PrintSafety(std::ostream& output, bool dryRun, int created)
{
    output << "read_only=" << (dryRun ? "true" : "false")
           << ",dry_run=" << (dryRun ? "true" : "false")
           << ",campaign_execution_applied=" << (created > 0 ? "true" : "false")
           << ",phase4c_execution_rows_created=" << created
           << ",proposals_executed=" << (created > 0 ? "true" : "false")
           << ",proposals_activated=false,experiments_created=" << created
           << ",experiments_queued=false,experiment_status_changed=false,"
              "scheduler_started=false,workers_started=false,"
              "automatic_progression=false";
}

void PrintOptionalId(std::ostream& output, const std::optional<long long>& value)
{
    if (value) output << *value;
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
    for (std::size_t i = 0; i < diagnostics.size(); ++i)
    {
        if (i != 0) output << '|';
        output << RecommendationMachineText(diagnostics[i]);
    }
}

void PrintResult(
    std::ostream& output,
    const RecommendationCampaignExecutionPlan& plan,
    const std::vector<PersistedRecommendationConversionExecution>& created)
{
    std::map<long long, const PersistedRecommendationConversionExecution*> byProposal;
    for (const auto& execution : created)
        byProposal.emplace(execution.proposalId, &execution);
    const std::string result = plan.request.dryRun
        ? (plan.state == RecommendationCampaignExecutionPlanState::alreadySatisfied
               ? "already_satisfied" : "validated")
        : (plan.state == RecommendationCampaignExecutionPlanState::alreadySatisfied
               ? "already_satisfied" : "executed");
    output << "RECOMMENDATION_CAMPAIGN_EXECUTION"
           << ",materialization_id=" << plan.request.materializationId
           << ",materialization_identity_hash="
           << RecommendationMachineText(plan.materializationIdentityHash)
           << ",selected_member_count=" << plan.members.size()
           << ",proposals_validated=" << plan.proposalsValidated
           << ",executions_inserted=" << created.size()
           << ",already_satisfied_count=" << plan.alreadySatisfiedCount
           << ",campaign_operation_identity_hash="
           << RecommendationMachineText(plan.operationIdentityHash)
           << ",result=" << result << ",diagnostic_codes=";
    PrintDiagnostics(output, plan.diagnosticCodes);
    output << ',';
    PrintSafety(output, plan.request.dryRun, static_cast<int>(created.size()));
    output << '\n';

    for (const auto& member : plan.members)
    {
        const auto found = byProposal.find(member.proposalId);
        std::optional<long long> executionId = member.previousExecutionId;
        std::optional<long long> experimentId = member.previousExperimentId;
        if (found != byProposal.end())
        {
            executionId = found->second->executionId;
            experimentId = found->second->experimentId;
        }
        output << "RECOMMENDATION_CAMPAIGN_EXECUTION_MEMBER"
               << ",materialization_id=" << plan.request.materializationId
               << ",member_ordinal=" << member.memberOrdinal
               << ",materialization_member_id=" << member.materializationMemberId
               << ",proposal_id=" << member.proposalId
               << ",authorization_review_decision_id=";
        PrintOptionalId(output, member.authorizationReviewDecisionId);
        output << ",previous_execution_id=";
        PrintOptionalId(output, member.previousExecutionId);
        output << ",resulting_execution_id=";
        PrintOptionalId(output, executionId);
        output << ",resulting_experiment_id=";
        PrintOptionalId(output, experimentId);
        output << ",created=" << (found != byProposal.end() ? "true" : "false")
               << ",created_experiment_status="
               << (found != byProposal.end() ? "paused" : "null")
               << ",diagnostic_codes=";
        PrintDiagnostics(output, member.diagnosticCodes);
        output << ',';
        PrintSafety(output, plan.request.dryRun,
                    found != byProposal.end() ? 1 : 0);
        output << '\n';
    }
}

RecommendationCampaignExecutionPlan ValidateDryRun(
    pqxx::connection& connection,
    const RecommendationCampaignExecutionRequest& request,
    int& selectedMemberCount,
    int& proposalsValidated,
    std::string& materializationIdentityHash)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    if (!RecommendationCampaignExecutionSchemasExist(transaction))
        throw std::runtime_error("campaign_execution_schemas_required");
    const auto materialization = LoadRecommendationCampaignExecutionMaterialization(
        transaction, request.materializationId);
    if (!materialization)
        throw std::runtime_error("campaign_execution_materialization_not_found");
    selectedMemberCount = materialization->selectedMemberCount;
    materializationIdentityHash = materialization->identityHash;
    const auto input = LoadRecommendationCampaignExecutionInput(
        transaction, *materialization);
    proposalsValidated = static_cast<int>(input.members.size());
    return BuildRecommendationCampaignExecutionPlan(request, input);
}

} // namespace

int RunRecommendationCampaignExecutionCommand(
    const std::string& connectionString,
    const RecommendationCampaignExecutionRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    int selectedMemberCount = 0;
    int proposalsValidated = 0;
    std::string materializationIdentityHash;
    try
    {
        const auto normalized = NormalizeRecommendationCampaignExecutionRequest(request);
        pqxx::connection connection{connectionString};
        if (normalized.dryRun)
        {
            const auto plan = ValidateDryRun(
                connection, normalized, selectedMemberCount,
                proposalsValidated, materializationIdentityHash);
            PrintResult(output, plan, {});
            return 0;
        }

        pqxx::work transaction{connection};
        if (!RecommendationCampaignExecutionSchemasExist(transaction))
            throw std::runtime_error("campaign_execution_schemas_required");
        const auto materialization = LoadRecommendationCampaignExecutionMaterialization(
            transaction, normalized.materializationId);
        if (!materialization)
            throw std::runtime_error("campaign_execution_materialization_not_found");
        selectedMemberCount = materialization->selectedMemberCount;
        materializationIdentityHash = materialization->identityHash;
        if (materialization->members.size() > static_cast<std::size_t>(
                kMaximumRecommendationCampaignExecutionMembers))
            throw std::runtime_error("campaign_execution_member_limit_exceeded");
        LockRecommendationCampaignExecutions(transaction, *materialization);
        const auto input = LoadRecommendationCampaignExecutionInput(
            transaction, *materialization);
        proposalsValidated = static_cast<int>(input.members.size());
        const auto plan = BuildRecommendationCampaignExecutionPlan(
            normalized, input);
        std::vector<PersistedRecommendationConversionExecution> created;
        if (plan.state == RecommendationCampaignExecutionPlanState::ready)
            created = PersistRecommendationCampaignExecutions(transaction, plan);
        transaction.commit();
        PrintResult(output, plan, created);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string code = error.what();
        const bool conflict = code.find("conflict") != std::string::npos;
        errors << (conflict ? "RECOMMENDATION_CAMPAIGN_EXECUTION_CONFLICT"
                            : "RECOMMENDATION_CAMPAIGN_EXECUTION_INVALID")
               << ",materialization_id=" << request.materializationId
               << ",materialization_identity_hash="
               << (materializationIdentityHash.empty() ? "null"
                   : RecommendationMachineText(materializationIdentityHash))
               << ",selected_member_count=" << selectedMemberCount
               << ",proposals_validated=" << proposalsValidated
               << ",executions_inserted=0,"
                  "already_satisfied_count=0,campaign_operation_identity_hash=null,"
               << "result=" << (conflict ? "conflict" : "invalid")
               << ",diagnostic_codes=" << RecommendationMachineText(code) << ',';
        PrintSafety(errors, request.dryRun, 0);
        errors << '\n';
        return conflict ? 2 : 1;
    }
    catch (const std::exception& error)
    {
        const std::string code = error.what();
        const bool conflict = code.find("conflict") != std::string::npos;
        errors << (conflict ? "RECOMMENDATION_CAMPAIGN_EXECUTION_CONFLICT"
                            : "RECOMMENDATION_CAMPAIGN_EXECUTION_FAILED")
               << ",materialization_id=" << request.materializationId
               << ",materialization_identity_hash="
               << (materializationIdentityHash.empty() ? "null"
                   : RecommendationMachineText(materializationIdentityHash))
               << ",selected_member_count=" << selectedMemberCount
               << ",proposals_validated=" << proposalsValidated
               << ",executions_inserted=0,"
                  "already_satisfied_count=0,campaign_operation_identity_hash=null,"
               << "result=" << (conflict ? "conflict" : "failed")
               << ",diagnostic_codes=" << RecommendationMachineText(code) << ',';
        PrintSafety(errors, request.dryRun, 0);
        errors << '\n';
        return conflict ? 2 : 1;
    }
}

} // namespace EA::ExperimentRecommendation
