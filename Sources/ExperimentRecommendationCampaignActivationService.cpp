#include "ExperimentRecommendationCampaignActivationService.hpp"

#include "ExperimentRecommendationCampaignActivationRepository.hpp"
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
           << ",all_or_nothing=true"
           << ",campaign_activation_applied="
           << (created > 0 ? "true" : "false")
           << ",phase4c_activation_rows_created=" << created
           << ",experiments_transitioned_count=" << created
           << ",proposals_created=false,proposal_reviews_created=false,"
              "executions_created=false,experiments_created=false,"
              "scheduler_started=false,workers_launched=false,"
              "direct_process_launch=false,automatic_follow_up=false";
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
    for (std::size_t index = 0; index < diagnostics.size(); ++index)
    {
        if (index != 0) output << '|';
        output << RecommendationMachineText(diagnostics[index]);
    }
}

void PrintResult(
    std::ostream& output,
    const RecommendationCampaignActivationPlan& plan,
    const std::vector<PersistedRecommendationConversionActivation>& created)
{
    std::map<long long, const PersistedRecommendationConversionActivation*>
        byExecution;
    for (const auto& activation : created)
        byExecution.emplace(activation.executionId, &activation);
    const std::string result = plan.request.dryRun
        ? (plan.state == RecommendationCampaignActivationPlanState::
               alreadySatisfied
               ? "already_satisfied" : "dry_run_ready")
        : (plan.state == RecommendationCampaignActivationPlanState::
               alreadySatisfied
               ? "already_satisfied" : "activated");
    output << "RECOMMENDATION_CAMPAIGN_ACTIVATION"
           << ",materialization_id=" << plan.request.materializationId
           << ",materialization_identity_hash="
           << RecommendationMachineText(plan.materializationIdentityHash)
           << ",operation_contract_version="
           << kRecommendationCampaignActivationContractVersion
           << ",campaign_operation_identity_hash="
           << RecommendationMachineText(plan.operationIdentityHash)
           << ",member_count=" << plan.members.size()
           << ",members_validated=" << plan.membersValidated
           << ",activated_count=" << created.size()
           << ",already_activated_count=" << plan.alreadyActivatedCount
           << ",result=" << result << ",diagnostic_codes=";
    PrintDiagnostics(output, plan.diagnosticCodes);
    output << ',';
    PrintSafety(output, plan.request.dryRun, static_cast<int>(created.size()));
    output << '\n';

    for (const auto& member : plan.members)
    {
        const auto found = byExecution.find(member.executionId);
        std::optional<long long> activationId = member.previousActivationId;
        if (found != byExecution.end())
            activationId = found->second->activationId;
        output << "RECOMMENDATION_CAMPAIGN_ACTIVATION_MEMBER"
               << ",materialization_id=" << plan.request.materializationId
               << ",member_ordinal=" << member.memberOrdinal
               << ",materialization_member_id="
               << member.materializationMemberId
               << ",recommendation_id=" << member.recommendationId
               << ",proposal_id=" << member.proposalId
               << ",authorization_review_decision_id="
               << member.authorizationReviewDecisionId
               << ",execution_id=" << member.executionId
               << ",experiment_id=" << member.experimentId
               << ",activation_id=";
        PrintOptionalId(output, activationId);
        output << ",pre_activation_status="
               << RecommendationMachineText(member.preActivationStatus)
               << ",pre_activation_phase="
               << RecommendationMachineText(member.preActivationPhase)
               << ",post_activation_status="
               << RecommendationMachineText(member.postActivationStatus)
               << ",post_activation_phase="
               << RecommendationMachineText(member.postActivationPhase)
               << ",activation_created="
               << (found != byExecution.end() ? "true" : "false")
               << ",already_satisfied="
               << (member.action ==
                           RecommendationCampaignActivationMemberAction::
                               alreadySatisfied
                       ? "true" : "false")
               << ",diagnostic_codes=";
        PrintDiagnostics(output, member.diagnosticCodes);
        output << ',';
        PrintSafety(
            output, plan.request.dryRun,
            found != byExecution.end() ? 1 : 0);
        output << '\n';
    }
}

RecommendationCampaignActivationPlan ValidateDryRun(
    pqxx::connection& connection,
    const RecommendationCampaignActivationRequest& request,
    int& selectedMemberCount,
    int& membersValidated,
    std::string& materializationIdentityHash)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    if (!RecommendationCampaignActivationSchemasExist(transaction))
        throw std::runtime_error("campaign_activation_schemas_required");
    const auto materialization =
        LoadRecommendationCampaignActivationMaterialization(
            transaction, request.materializationId);
    if (!materialization)
        throw std::runtime_error(
            "campaign_activation_materialization_not_found");
    selectedMemberCount = materialization->selectedMemberCount;
    materializationIdentityHash = materialization->identityHash;
    const auto input = LoadRecommendationCampaignActivationInput(
        transaction, *materialization);
    membersValidated = static_cast<int>(input.members.size());
    return BuildRecommendationCampaignActivationPlan(request, input);
}

} // namespace

int RunRecommendationCampaignActivationCommand(
    const std::string& connectionString,
    const RecommendationCampaignActivationRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    int selectedMemberCount = 0;
    int membersValidated = 0;
    std::string materializationIdentityHash;
    try
    {
        const auto normalized =
            NormalizeRecommendationCampaignActivationRequest(request);
        pqxx::connection connection{connectionString};
        if (normalized.dryRun)
        {
            const auto plan = ValidateDryRun(
                connection, normalized, selectedMemberCount, membersValidated,
                materializationIdentityHash);
            PrintResult(output, plan, {});
            return 0;
        }

        pqxx::work transaction{connection};
        if (!RecommendationCampaignActivationSchemasExist(transaction))
            throw std::runtime_error("campaign_activation_schemas_required");
        const auto materialization =
            LoadRecommendationCampaignActivationMaterialization(
                transaction, normalized.materializationId);
        if (!materialization)
            throw std::runtime_error(
                "campaign_activation_materialization_not_found");
        selectedMemberCount = materialization->selectedMemberCount;
        materializationIdentityHash = materialization->identityHash;
        if (materialization->members.size() > static_cast<std::size_t>(
                kMaximumRecommendationCampaignActivationMembers))
            throw std::runtime_error(
                "campaign_activation_member_limit_exceeded");
        LockRecommendationCampaignActivations(transaction, *materialization);
        const auto input = LoadRecommendationCampaignActivationInput(
            transaction, *materialization);
        membersValidated = static_cast<int>(input.members.size());
        const auto plan = BuildRecommendationCampaignActivationPlan(
            normalized, input);
        std::vector<PersistedRecommendationConversionActivation> created;
        if (plan.state == RecommendationCampaignActivationPlanState::ready)
            created = PersistRecommendationCampaignActivations(
                transaction, plan);
        transaction.commit();
        PrintResult(output, plan, created);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string code = error.what();
        const bool conflict = code.find("conflict") != std::string::npos;
        errors << (conflict ? "RECOMMENDATION_CAMPAIGN_ACTIVATION_CONFLICT"
                            : "RECOMMENDATION_CAMPAIGN_ACTIVATION_INVALID")
               << ",materialization_id=" << request.materializationId
               << ",materialization_identity_hash="
               << (materializationIdentityHash.empty()
                       ? "null"
                       : RecommendationMachineText(
                             materializationIdentityHash))
               << ",operation_contract_version="
               << kRecommendationCampaignActivationContractVersion
               << ",member_count=" << selectedMemberCount
               << ",members_validated=" << membersValidated
               << ",activated_count=0,already_activated_count=0,"
                  "campaign_operation_identity_hash=null,result="
               << (conflict ? "conflict" : "invalid")
               << ",diagnostic_codes=" << RecommendationMachineText(code)
               << ',';
        PrintSafety(errors, request.dryRun, 0);
        errors << '\n';
        return conflict ? 2 : 1;
    }
    catch (const std::exception& error)
    {
        const std::string code = error.what();
        const bool conflict = code.find("conflict") != std::string::npos;
        errors << (conflict ? "RECOMMENDATION_CAMPAIGN_ACTIVATION_CONFLICT"
                            : "RECOMMENDATION_CAMPAIGN_ACTIVATION_FAILED")
               << ",materialization_id=" << request.materializationId
               << ",materialization_identity_hash="
               << (materializationIdentityHash.empty()
                       ? "null"
                       : RecommendationMachineText(
                             materializationIdentityHash))
               << ",operation_contract_version="
               << kRecommendationCampaignActivationContractVersion
               << ",member_count=" << selectedMemberCount
               << ",members_validated=" << membersValidated
               << ",activated_count=0,already_activated_count=0,"
                  "campaign_operation_identity_hash=null,result="
               << (conflict ? "conflict" : "failed")
               << ",diagnostic_codes=" << RecommendationMachineText(code)
               << ',';
        PrintSafety(errors, request.dryRun, 0);
        errors << '\n';
        return conflict ? 2 : 1;
    }
}

} // namespace EA::ExperimentRecommendation
