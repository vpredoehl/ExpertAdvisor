#include "ExperimentRecommendationCampaignProposalReviewService.hpp"

#include "ExperimentRecommendationCampaignProposalReviewRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <map>
#include <ostream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void PrintSafety(
    std::ostream& output,
    bool dryRun,
    int inserted)
{
    output << "read_only=" << (dryRun ? "true" : "false")
           << ",dry_run=" << (dryRun ? "true" : "false")
           << ",campaign_reviews_created="
           << (inserted > 0 ? "true" : "false")
           << ",phase4c_review_decisions_created=" << inserted
           << ",proposals_executed=false,proposals_activated=false,"
              "experiments_created=false,experiments_queued=false,"
              "experiment_status_changed=false,scheduler_started=false,"
              "workers_started=false,automatic_progression=false";
}

void PrintOptionalId(
    std::ostream& output,
    const std::optional<long long>& value)
{
    if (value) output << *value;
    else output << "null";
}

std::string DispositionText(
    const std::optional<RecommendationConversionProposalReviewDisposition>& value)
{
    return value
        ? RecommendationConversionProposalReviewDispositionText(*value)
        : "pending_review";
}

std::string DecisionTextOrInvalid(
    RecommendationConversionProposalReviewDecision decision)
{
    try
    {
        return RecommendationConversionProposalReviewDecisionText(decision);
    }
    catch (const std::invalid_argument&)
    {
        return "invalid";
    }
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
    const RecommendationCampaignProposalReviewPlan& plan,
    const std::vector<PersistedRecommendationConversionProposalReviewDecision>&
        inserted)
{
    std::map<long long, long long> insertedIds;
    for (const auto& decision : inserted)
        insertedIds.emplace(decision.proposalId, decision.reviewDecisionId);
    const bool dryRun = plan.request.dryRun;
    const std::string result = dryRun
        ? (plan.state ==
                   RecommendationCampaignProposalReviewPlanState::alreadySatisfied
               ? "already_satisfied"
               : "validated")
        : (plan.state ==
                   RecommendationCampaignProposalReviewPlanState::alreadySatisfied
               ? "already_satisfied"
               : "reviewed");
    output << "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW"
           << ",materialization_id=" << plan.request.materializationId
           << ",materialization_identity_hash="
           << RecommendationMachineText(plan.materializationIdentityHash)
           << ",campaign_decision="
           << RecommendationConversionProposalReviewDecisionText(
                  plan.request.decision)
           << ",operator="
           << RecommendationMachineText(plan.request.operatorIdentity)
           << ",reason=" << RecommendationMachineText(plan.request.reasonText)
           << ",selected_member_count=" << plan.members.size()
           << ",proposals_validated=" << plan.proposalsValidated
           << ",reviews_inserted=" << inserted.size()
           << ",already_satisfied_count=" << plan.alreadySatisfiedCount
           << ",conflict_count=" << plan.conflictCount
           << ",campaign_operation_identity_hash="
           << RecommendationMachineText(plan.operationIdentityHash)
           << ",phase4c_request_id="
           << RecommendationMachineText(plan.phase4cRequestId)
           << ",result=" << result << ",diagnostic_codes=";
    PrintDiagnostics(output, plan.diagnosticCodes);
    output << ',';
    PrintSafety(output, dryRun, static_cast<int>(inserted.size()));
    output << '\n';

    const auto resultingDisposition =
        plan.request.decision ==
                RecommendationConversionProposalReviewDecision::approve
            ? RecommendationConversionProposalReviewDisposition::approved
            : RecommendationConversionProposalReviewDisposition::rejected;
    for (const auto& member : plan.members)
    {
        const auto insertedId = insertedIds.find(member.proposalId);
        const std::optional<long long> resultingId =
            insertedId != insertedIds.end()
            ? std::optional<long long>{insertedId->second}
            : member.previousReviewDecisionId;
        output << "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW_MEMBER"
               << ",materialization_id=" << plan.request.materializationId
               << ",member_ordinal=" << member.memberOrdinal
               << ",materialization_member_id="
               << member.materializationMemberId
               << ",proposal_id=" << member.proposalId
               << ",previous_authoritative_review_status="
               << DispositionText(member.previousDisposition)
               << ",previous_authoritative_review_id=";
        PrintOptionalId(output, member.previousReviewDecisionId);
        output << ",resulting_review_status="
               << RecommendationConversionProposalReviewDispositionText(
                      resultingDisposition)
               << ",resulting_review_id=";
        PrintOptionalId(output, resultingId);
        output << ",inserted="
               << (insertedId != insertedIds.end() ? "true" : "false")
               << ",diagnostic_codes=";
        PrintDiagnostics(output, member.diagnosticCodes);
        output << ',';
        PrintSafety(
            output, dryRun,
            insertedId != insertedIds.end() ? 1 : 0);
        output << '\n';
    }
}

void UseConsistentSnapshot(pqxx::read_transaction& transaction)
{
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
}

RecommendationCampaignProposalReviewPlan ValidateDryRun(
    pqxx::connection& connection,
    const RecommendationCampaignProposalReviewRequest& request,
    int& selectedMemberCount,
    int& proposalsValidated,
    std::string& materializationIdentityHash)
{
    pqxx::read_transaction transaction{connection};
    UseConsistentSnapshot(transaction);
    if (!RecommendationCampaignProposalReviewSchemasExist(transaction))
        throw std::runtime_error(
            "campaign_proposal_review_schemas_required");
    const auto materialization =
        LoadRecommendationCampaignProposalReviewMaterialization(
            transaction, request.materializationId);
    if (!materialization)
        throw std::runtime_error(
            "campaign_proposal_review_materialization_not_found");
    selectedMemberCount = materialization->selectedMemberCount;
    materializationIdentityHash = materialization->identityHash;
    const auto input = LoadRecommendationCampaignProposalReviewInput(
        transaction, *materialization);
    proposalsValidated = static_cast<int>(input.members.size());
    return BuildRecommendationCampaignProposalReviewPlan(request, input);
}

} // namespace

int RunRecommendationCampaignProposalReviewCommand(
    const std::string& connectionString,
    const RecommendationCampaignProposalReviewRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    int selectedMemberCount = 0;
    int proposalsValidated = 0;
    std::string materializationIdentityHash;
    try
    {
        const auto normalized =
            NormalizeRecommendationCampaignProposalReviewRequest(request);
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
        if (!RecommendationCampaignProposalReviewSchemasExist(transaction))
            throw std::runtime_error(
                "campaign_proposal_review_schemas_required");
        const auto materialization =
            LoadRecommendationCampaignProposalReviewMaterialization(
                transaction, normalized.materializationId);
        if (!materialization)
            throw std::runtime_error(
                "campaign_proposal_review_materialization_not_found");
        selectedMemberCount = materialization->selectedMemberCount;
        materializationIdentityHash = materialization->identityHash;
        if (materialization->members.size() > static_cast<std::size_t>(
                kMaximumRecommendationCampaignProposalReviewMembers))
            throw std::runtime_error(
                "campaign_proposal_review_member_limit_exceeded");
        LockRecommendationCampaignProposalReviews(
            transaction, *materialization);
        const auto input = LoadRecommendationCampaignProposalReviewInput(
            transaction, *materialization);
        proposalsValidated = static_cast<int>(input.members.size());
        const auto plan = BuildRecommendationCampaignProposalReviewPlan(
            normalized, input);
        std::vector<PersistedRecommendationConversionProposalReviewDecision>
            inserted;
        if (plan.state == RecommendationCampaignProposalReviewPlanState::ready)
            inserted = PersistRecommendationCampaignProposalReviews(
                transaction, plan);
        transaction.commit();
        PrintResult(output, plan, inserted);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string code = error.what();
        const bool conflict = code.find("conflict") != std::string::npos;
        errors << (conflict
                       ? "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW_CONFLICT"
                       : "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW_INVALID")
               << ",materialization_id=" << request.materializationId
               << ",campaign_decision="
               << DecisionTextOrInvalid(request.decision)
               << ",operator="
               << RecommendationMachineText(request.operatorIdentity)
               << ",reason=" << RecommendationMachineText(request.reasonText)
               << ",materialization_identity_hash="
               << (materializationIdentityHash.empty()
                       ? "null"
                       : RecommendationMachineText(materializationIdentityHash))
               << ",selected_member_count=" << selectedMemberCount
               << ",proposals_validated=" << proposalsValidated
               << ",reviews_inserted=0,already_satisfied_count=0,"
               << "conflict_count=" << (conflict ? 1 : 0)
               << ",campaign_operation_identity_hash=null,result="
               << (conflict ? "conflict" : "invalid")
               << ",diagnostic_codes="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors, request.dryRun, 0);
        errors << '\n';
        return conflict ? 2 : 1;
    }
    catch (const std::exception& error)
    {
        const std::string code = error.what();
        const bool conflict = code.find("conflict") != std::string::npos;
        errors << (conflict
                       ? "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW_CONFLICT"
                       : "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW_FAILED")
               << ",materialization_id=" << request.materializationId
               << ",campaign_decision="
               << DecisionTextOrInvalid(request.decision)
               << ",operator="
               << RecommendationMachineText(request.operatorIdentity)
               << ",reason=" << RecommendationMachineText(request.reasonText)
               << ",materialization_identity_hash="
               << (materializationIdentityHash.empty()
                       ? "null"
                       : RecommendationMachineText(materializationIdentityHash))
               << ",selected_member_count=" << selectedMemberCount
               << ",proposals_validated=" << proposalsValidated
               << ",reviews_inserted=0,already_satisfied_count=0,"
               << "conflict_count=" << (conflict ? 1 : 0)
               << ",campaign_operation_identity_hash=null,result="
               << (conflict ? "conflict" : "failed")
               << ",diagnostic_codes="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors, request.dryRun, 0);
        errors << '\n';
        return conflict ? 2 : 1;
    }
}

} // namespace EA::ExperimentRecommendation
