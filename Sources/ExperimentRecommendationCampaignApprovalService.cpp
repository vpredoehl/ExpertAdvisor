#include "ExperimentRecommendationCampaignApprovalService.hpp"

#include "ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "ExperimentRecommendationCampaignReview.hpp"
#include "ExperimentRecommendationService.hpp"

#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void PrintMutationSafety(std::ostream& output, bool recorded)
{
    output << "campaign_approval_recorded=" << (recorded ? "true" : "false")
           << ",campaign_execution_started=false,proposal_created=false,"
              "experiment_created=false,experiment_modified=false,"
              "activation_created=false,scheduler_started=false,"
              "worker_started=false";
}

void PrintReadSafety(std::ostream& output)
{
    output << "read_only=true,campaign_execution_started=false,"
              "proposal_created=false,experiment_created=false,"
              "experiment_modified=false,activation_created=false,"
              "scheduler_started=false,worker_started=false";
}

void PrintApproval(
    std::ostream& output,
    const char* event,
    const PersistedRecommendationCampaignApproval& persisted,
    bool readOnly,
    std::optional<bool> recorded = std::nullopt)
{
    const auto& value = persisted.evidence;
    output << event
           << ",campaign_approval_id=" << persisted.campaignApprovalId
           << ",approval_contract_version="
           << value.approvalContractVersion
           << ",ranking_snapshot_id=" << value.rankingSnapshotId
           << ",ranking_snapshot_identity_hash="
           << RecommendationMachineText(value.rankingSnapshotIdentityHash)
           << ",planning_policy_hash="
           << RecommendationMachineText(value.planningPolicyHash)
           << ",planning_scope_canonical="
           << RecommendationMachineText(value.planningScopeCanonical)
           << ",campaign_plan_identity_hash="
           << RecommendationMachineText(value.campaignPlanIdentityHash)
           << ",review_contract_version=" << value.reviewContractVersion
           << ",campaign_review_identity_hash="
           << RecommendationMachineText(value.campaignReviewIdentityHash)
           << ",candidate_count=" << value.summary.candidateCount
           << ",selected_count=" << value.summary.selectedCount
           << ",excluded_count=" << value.summary.excludedCount
           << ",duplicate_group_count="
           << value.summary.duplicateGroupCount
           << ",duplicate_candidate_count="
           << value.summary.duplicateCandidateCount
           << ",considered_family_count="
           << value.summary.consideredFamilyCount
           << ",selected_family_count="
           << value.summary.selectedFamilyCount
           << ",considered_symbol_count="
           << value.summary.consideredSymbolCount
           << ",selected_symbol_count="
           << value.summary.selectedSymbolCount
           << ",considered_horizon_count="
           << value.summary.consideredHorizonCount
           << ",selected_horizon_count="
           << value.summary.selectedHorizonCount
           << ",deterministic_ordering_verified="
           << (value.summary.deterministicOrderingVerified ? "true" : "false")
           << ",decision="
           << RecommendationCampaignApprovalDecisionText(value.decision)
           << ",reviewer="
           << RecommendationMachineText(value.reviewerIdentity)
           << ",reason=" << RecommendationMachineText(value.reasonText)
           << ",approval_identity_hash="
           << RecommendationMachineText(value.approvalIdentityHash)
           << ",created_at=" << RecommendationMachineText(persisted.createdAt)
           << ',';
    if (readOnly) PrintReadSafety(output);
    else PrintMutationSafety(output, recorded.value_or(false));
    output << '\n';
}

} // namespace

int RunRecordRecommendationCampaignApprovalCommand(
    const std::string& connectionString,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope,
    const RecommendationCampaignApprovalRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        const RecommendationCampaignApprovalRequest normalized =
            NormalizeRecommendationCampaignApprovalRequest(request);
        if (const auto error = ValidateRecommendationCampaignPlanningPolicy(
                policy))
            throw std::invalid_argument(*error);
        if (const auto error = ValidateRecommendationCampaignPlanningScope(
                scope))
            throw std::invalid_argument(*error);

        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        if (!RecommendationCampaignPlanningSchemasExist(transaction) ||
            !RecommendationCampaignApprovalSchemaExists(transaction))
            throw std::runtime_error(
                "recommendation_campaign_approval_schemas_required");
        const RecommendationCampaignPlanInput input =
            LoadRecommendationCampaignPlanInput(transaction, policy, scope);
        const RecommendationCampaignPlan plan =
            PlanRecommendationCampaign(input);
        const RecommendationCampaignReview review =
            ReviewRecommendationCampaignPlan(plan);
        const RecommendationCampaignApprovalEvidence evidence =
            BuildRecommendationCampaignApprovalEvidence(
                scope.rankingSnapshotId,
                input.rankingSnapshotIdentityCanonical,
                input.rankingSnapshotIdentityHash,
                plan,
                review,
                normalized);
        RecommendationCampaignApprovalPersistResult result =
            PersistRecommendationCampaignApproval(transaction, evidence);
        transaction.commit();

        const bool recorded = result.outcome ==
            RecommendationCampaignApprovalPersistOutcome::recorded;
        PrintApproval(
            output,
            recorded ? "RECOMMENDATION_CAMPAIGN_APPROVAL_RECORDED"
                     : "RECOMMENDATION_CAMPAIGN_APPROVAL_ALREADY_RECORDED",
            result.approval,
            false,
            recorded);
        output << "Campaign approval records one explicit operator decision; "
                  "it does not execute the campaign or change any experiment.\n";
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        errors << (message ==
                           "recommendation_campaign_approval_review_identity_mismatch"
                       ? "RECOMMENDATION_CAMPAIGN_APPROVAL_STALE"
                       : "RECOMMENDATION_CAMPAIGN_APPROVAL_INVALID")
               << ",error=" << RecommendationMachineText(message) << ',';
        PrintMutationSafety(errors, false);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        const std::string message = error.what();
        if (message == "recommendation_campaign_approval_review_conflict")
        {
            errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_CONFLICT,error="
                   << RecommendationMachineText(message) << ',';
            PrintMutationSafety(errors, false);
            errors << '\n';
            return 3;
        }
        if (message == "recommendation_campaign_ranking_snapshot_not_found")
        {
            errors << "RECOMMENDATION_CAMPAIGN_RANKING_SNAPSHOT_NOT_FOUND"
                   << ",ranking_snapshot_id=" << scope.rankingSnapshotId << ',';
            PrintMutationSafety(errors, false);
            errors << '\n';
            return 1;
        }
        errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_FAILED,error="
               << RecommendationMachineText(message) << ',';
        PrintMutationSafety(errors, false);
        errors << '\n';
        return 2;
    }
}

int RunShowRecommendationCampaignApprovalCommand(
    const std::string& connectionString,
    long long campaignApprovalId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        if (!RecommendationCampaignApprovalSchemaExists(connection))
            throw std::runtime_error(
                "recommendation_campaign_approval_schema_required");
        const auto approval = FindRecommendationCampaignApproval(
            connection, campaignApprovalId);
        if (!approval)
        {
            errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_NOT_FOUND"
                   << ",campaign_approval_id=" << campaignApprovalId << ',';
            PrintReadSafety(errors);
            errors << '\n';
            return 1;
        }
        PrintApproval(
            output, "RECOMMENDATION_CAMPAIGN_APPROVAL", *approval, true);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintReadSafety(errors);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_FAILED,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintReadSafety(errors);
        errors << '\n';
        return 2;
    }
}

int RunListRecommendationCampaignApprovalsCommand(
    const std::string& connectionString,
    std::optional<RecommendationCampaignApprovalDecision> decision,
    int limit,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        if (!RecommendationCampaignApprovalSchemaExists(connection))
            throw std::runtime_error(
                "recommendation_campaign_approval_schema_required");
        const auto approvals = ListRecommendationCampaignApprovals(
            connection, decision, limit);
        for (const auto& approval : approvals)
            PrintApproval(
                output, "RECOMMENDATION_CAMPAIGN_APPROVAL", approval, true);
        output << "RECOMMENDATION_CAMPAIGN_APPROVAL_LIST_COMPLETE,count="
               << approvals.size() << ',';
        PrintReadSafety(output);
        output << '\n';
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintReadSafety(errors);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_FAILED,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintReadSafety(errors);
        errors << '\n';
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
