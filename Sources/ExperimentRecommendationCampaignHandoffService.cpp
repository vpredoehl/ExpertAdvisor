#include "ExperimentRecommendationCampaignHandoffService.hpp"

#include "ExperimentRecommendationCampaignHandoffRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <pqxx/pqxx>

#include <optional>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void PrintSafety(std::ostream& output)
{
    output << "read_only=true,campaign_materializations_created=false,"
              "conversion_proposals_created=false,"
              "conversion_reviews_created=false,"
              "conversion_executions_created=false,"
              "conversion_activations_created=false,"
              "experiments_created=false,scheduler_started=false,"
              "workers_started=false";
}

void PrintOptionalId(
    std::ostream& output,
    const std::optional<long long>& value)
{
    if (value) output << *value;
    else output << "null";
}

void UseConsistentSnapshot(pqxx::read_transaction& transaction)
{
    // libpqxx read_transaction is read-only but defaults to READ COMMITTED,
    // which may use a different snapshot for each statement.  Handoff reads
    // span the manifest, members, and current Phase 4C evidence, so pin one
    // snapshot before the first query.
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
}

void PrintHandoff(
    std::ostream& output,
    const RecommendationCampaignHandoff& handoff,
    bool includeMembers)
{
    const auto& summary = handoff.summary;
    output << "RECOMMENDATION_CAMPAIGN_HANDOFF"
           << ",materialization_id=" << handoff.materializationId
           << ",campaign_approval_id=" << handoff.campaignApprovalId
           << ",materialization_identity_hash="
           << RecommendationMachineText(handoff.materializationIdentityHash)
           << ",materialization_complete="
           << (handoff.materializationComplete ? "true" : "false")
           << ",handoff_state="
           << RecommendationCampaignHandoffStateText(handoff.state)
           << ",integrity="
           << RecommendationCampaignHandoffIntegrityText(handoff.integrity)
           << ",total_members=" << summary.totalMembers
           << ",proposals_present=" << summary.proposalsPresent
           << ",awaiting_review=" << summary.awaitingReview
           << ",approved=" << summary.approved
           << ",rejected=" << summary.rejected
           << ",no_authoritative_review=" << summary.noAuthoritativeReview
           << ",executions_present=" << summary.executionsPresent
           << ",activations_present=" << summary.activationsPresent
           << ",integrity_failures=" << summary.integrityFailures << ',';
    PrintSafety(output);
    output << '\n';

    if (!includeMembers) return;
    for (const auto& member : handoff.members)
    {
        output << "RECOMMENDATION_CAMPAIGN_HANDOFF_MEMBER"
               << ",materialization_id=" << handoff.materializationId
               << ",campaign_approval_id=" << handoff.campaignApprovalId
               << ",materialization_identity_hash="
               << RecommendationMachineText(handoff.materializationIdentityHash)
               << ",member_ordinal=" << member.memberOrdinal
               << ",ranking_member_id=" << member.rankingMemberId
               << ",recommendation_id=" << member.recommendationId
               << ",source_experiment_id=" << member.sourceExperimentId
               << ",ranking_position=" << member.rankingPosition
               << ",conversion_proposal_id=" << member.conversionProposalId
               << ",proposal_identity_hash="
               << RecommendationMachineText(member.proposalIdentityHash)
               << ",proposal_review_status="
               << RecommendationCampaignHandoffReviewStatusText(
                      member.reviewStatus)
               << ",proposal_review_decision_id=";
        PrintOptionalId(output, member.reviewDecisionId);
        output << ",execution_status="
               << RecommendationCampaignHandoffExecutionStatusText(
                      member.executionStatus)
               << ",execution_id=";
        PrintOptionalId(output, member.executionId);
        output << ",activation_status="
               << RecommendationCampaignHandoffActivationStatusText(
                      member.activationStatus)
               << ",activation_id=";
        PrintOptionalId(output, member.activationId);
        output << ",phase4c_workflow_state="
               << RecommendationConversionWorkflowStateText(
                      member.workflowState);
        output << ",integrity="
               << RecommendationCampaignHandoffIntegrityText(member.integrity)
               << ",diagnostic_count=" << member.diagnosticCodes.size()
               << ",diagnostic_codes=";
        for (std::size_t i = 0; i < member.diagnosticCodes.size(); ++i)
        {
            if (i != 0) output << '|';
            output << RecommendationMachineText(member.diagnosticCodes[i]);
        }
        output << ",read_only=true\n";
    }
}

} // namespace

int RunShowRecommendationCampaignHandoffCommand(
    const std::string& connectionString,
    long long materializationId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        UseConsistentSnapshot(transaction);
        if (!RecommendationCampaignHandoffSchemasExist(transaction))
            throw std::runtime_error(
                "recommendation_campaign_handoff_schemas_required");
        const auto handoff = FindRecommendationCampaignHandoff(
            transaction, materializationId);
        if (!handoff)
        {
            errors << "RECOMMENDATION_CAMPAIGN_HANDOFF_NOT_FOUND"
                   << ",materialization_id=" << materializationId << ',';
            PrintSafety(errors);
            errors << '\n';
            return 1;
        }
        PrintHandoff(output, *handoff, true);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_HANDOFF_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_HANDOFF_FAILED,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 2;
    }
}

int RunListRecommendationCampaignHandoffsCommand(
    const std::string& connectionString,
    int limit,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        UseConsistentSnapshot(transaction);
        if (!RecommendationCampaignHandoffSchemasExist(transaction))
            throw std::runtime_error(
                "recommendation_campaign_handoff_schemas_required");
        const auto handoffs = ListRecommendationCampaignHandoffs(
            transaction, limit);
        for (const auto& handoff : handoffs)
            PrintHandoff(output, handoff, false);
        output << "RECOMMENDATION_CAMPAIGN_HANDOFF_LIST_COMPLETE,count="
               << handoffs.size() << ',';
        PrintSafety(output);
        output << '\n';
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_HANDOFF_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_HANDOFF_FAILED,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
