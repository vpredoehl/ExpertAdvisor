#include "ExperimentRecommendationCampaignMaterializationService.hpp"

#include "ExperimentRecommendationCampaignApprovalRepository.hpp"
#include "ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "ExperimentRecommendationCampaignReview.hpp"
#include "ExperimentRecommendationService.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

void ValidateExistingMaterializationReplay(
    const PersistedRecommendationCampaignMaterialization& persisted,
    const RecommendationCampaignApprovalEvidence& approval,
    const RecommendationCampaignMaterializationRequest& request)
{
    if (persisted.campaignApprovalId != request.campaignApprovalId ||
        persisted.materializedBy != request.operatorIdentity ||
        persisted.reasonText != request.reasonText)
        throw std::runtime_error(
            "recommendation_campaign_materialization_conflict");

    RecommendationCampaignMaterializationEvidence evidence;
    evidence.materializationContractVersion = persisted.contractVersion;
    evidence.campaignApprovalId = persisted.campaignApprovalId;
    evidence.approval = approval;
    evidence.operatorIdentity = persisted.materializedBy;
    evidence.reasonText = persisted.reasonText;
    evidence.selectedMemberCount = persisted.selectedMemberCount;
    evidence.materializationIdentityCanonical = persisted.identityCanonical;
    evidence.materializationIdentityHash = persisted.identityHash;
    evidence.members.reserve(persisted.members.size());
    for (const auto& stored : persisted.members)
    {
        RecommendationCampaignMaterializationSelectedMember member;
        member.memberOrdinal = stored.memberOrdinal;
        member.rankingMemberId = stored.rankingMemberId;
        member.recommendationId = stored.recommendationId;
        member.sourceExperimentId = stored.sourceExperimentId;
        member.rankingPosition = stored.rankingPosition;
        member.selectedMemberIdentityCanonical =
            stored.selectedMemberIdentityCanonical;
        member.selectedMemberIdentityHash = stored.selectedMemberIdentityHash;
        member.proposal.recommendationId = stored.recommendationId;
        member.proposal.sourceExperimentId = stored.sourceExperimentId;
        member.proposal.conversionIdentityCanonical =
            stored.proposalIdentityCanonical;
        member.proposal.conversionIdentityHash = stored.proposalIdentityHash;
        evidence.members.push_back(std::move(member));
    }
    try
    {
        ValidateRecommendationCampaignMaterializationEvidence(evidence);
    }
    catch (const std::invalid_argument&)
    {
        throw std::runtime_error(
            "invalid_persisted_recommendation_campaign_materialization");
    }
}

void PrintSafety(std::ostream& output, bool readOnly, bool recorded,
                 bool existingIdentical,
                 int createdProposals)
{
    output << "read_only=" << (readOnly ? "true" : "false")
           << ",campaign_materialization_recorded="
           << (recorded ? "true" : "false")
           << ",campaign_materialization_existing_identical="
           << (existingIdentical ? "true" : "false")
           << ",conversion_proposals_created="
           << (createdProposals > 0 ? "true" : "false")
           << ",conversion_proposals_created_count=" << createdProposals
           << ",conversion_reviews_created=false,"
              "conversion_executions_created=false,"
              "conversion_activations_created=false,experiments_created=false,"
              "experiments_modified=false,scheduler_started=false,"
              "workers_started=false,campaign_executed=false";
}

void PrintMaterialization(
    std::ostream& output,
    const char* event,
    const PersistedRecommendationCampaignMaterialization& value,
    bool readOnly,
    bool recorded,
    int newlyCreated,
    int reused)
{
    output << event
           << ",campaign_materialization_id=" << value.materializationId
           << ",campaign_approval_id=" << value.campaignApprovalId
           << ",materialization_contract_version=" << value.contractVersion
           << ",ranking_snapshot_id=" << value.rankingSnapshotId
           << ",approval_identity_hash="
           << RecommendationMachineText(value.approvalIdentityHash)
           << ",campaign_plan_identity_hash="
           << RecommendationMachineText(value.campaignPlanIdentityHash)
           << ",campaign_review_identity_hash="
           << RecommendationMachineText(value.campaignReviewIdentityHash)
           << ",materialization_identity_hash="
           << RecommendationMachineText(value.identityHash)
           << ",selected_member_count=" << value.selectedMemberCount
           << ",linked_member_count=" << value.members.size()
           << ",initially_created_proposal_count="
           << value.initiallyCreatedProposalCount
           << ",initially_reused_proposal_count="
           << value.initiallyReusedProposalCount
           << ",newly_created_proposal_count=" << newlyCreated
           << ",reused_existing_proposal_count=" << reused
           << ",materialized_by="
           << RecommendationMachineText(value.materializedBy)
           << ",reason=" << RecommendationMachineText(value.reasonText)
           << ",created_at=" << RecommendationMachineText(value.createdAt)
           << ',';
    PrintSafety(output, readOnly, recorded, !readOnly && !recorded, newlyCreated);
    output << '\n';
    for (const auto& member : value.members)
        output << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_MEMBER"
               << ",campaign_materialization_id=" << value.materializationId
               << ",member_ordinal=" << member.memberOrdinal
               << ",ranking_member_id=" << member.rankingMemberId
               << ",recommendation_id=" << member.recommendationId
               << ",source_experiment_id=" << member.sourceExperimentId
               << ",ranking_position=" << member.rankingPosition
               << ",conversion_proposal_id=" << member.conversionProposalId
               << ",selected_member_identity_hash="
               << RecommendationMachineText(member.selectedMemberIdentityHash)
               << ",proposal_identity_hash="
               << RecommendationMachineText(member.proposalIdentityHash)
               << ",read_only=" << (readOnly ? "true" : "false")
               << ",experiments_created=false,"
                  "scheduler_started=false,workers_started=false\n";
}

} // namespace

int RunMaterializeRecommendationCampaignCommand(
    const std::string& connectionString,
    const RecommendationCampaignMaterializationRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        const auto normalized =
            NormalizeRecommendationCampaignMaterializationRequest(request);
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        if (!RecommendationCampaignPlanningSchemasExist(transaction) ||
            !RecommendationCampaignApprovalSchemaExists(transaction) ||
            !RecommendationCampaignMaterializationSchemaExists(transaction))
            throw std::runtime_error(
                "recommendation_campaign_materialization_schemas_required");
        const auto approval = FindRecommendationCampaignApproval(
            transaction, normalized.campaignApprovalId);
        if (!approval)
        {
            errors << "RECOMMENDATION_CAMPAIGN_APPROVAL_NOT_FOUND"
                   << ",campaign_approval_id=" << normalized.campaignApprovalId
                   << ',';
            PrintSafety(errors, false, false, false, 0);
            errors << '\n';
            return 1;
        }
        if (approval->evidence.decision !=
            RecommendationCampaignApprovalDecision::approved)
            throw std::invalid_argument(
                "recommendation_campaign_materialization_approval_rejected");

        if (auto existing = FindRecommendationCampaignMaterializationByApproval(
                transaction, normalized.campaignApprovalId))
        {
            ValidateExistingMaterializationReplay(
                *existing, approval->evidence, normalized);
            transaction.commit();
            PrintMaterialization(
                output,
                "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_ALREADY_RECORDED",
                *existing, false, false, 0, existing->selectedMemberCount);
            return 0;
        }

        const auto policy = ParseRecommendationCampaignPlanningPolicyCanonicalText(
            approval->evidence.planningPolicyCanonical);
        const auto scope = ParseRecommendationCampaignPlanningScopeCanonicalText(
            approval->evidence.planningScopeCanonical);
        const auto input = LoadRecommendationCampaignPlanInput(
            transaction, policy, scope);
        const auto plan = PlanRecommendationCampaign(input);
        const auto review = ReviewRecommendationCampaignPlan(plan);
        RecommendationCampaignApprovalRequest approvalRequest;
        approvalRequest.decision = approval->evidence.decision;
        approvalRequest.expectedCampaignReviewIdentityHash =
            approval->evidence.campaignReviewIdentityHash;
        approvalRequest.reviewerIdentity = approval->evidence.reviewerIdentity;
        approvalRequest.reasonText = approval->evidence.reasonText;
        const auto reconstructedApproval =
            BuildRecommendationCampaignApprovalEvidence(
                scope.rankingSnapshotId,
                input.rankingSnapshotIdentityCanonical,
                input.rankingSnapshotIdentityHash,
                plan, review, approvalRequest);
        if (reconstructedApproval.approvalIdentityCanonical !=
                approval->evidence.approvalIdentityCanonical ||
            reconstructedApproval.approvalIdentityHash !=
                approval->evidence.approvalIdentityHash ||
            reconstructedApproval.campaignReviewIdentityCanonical !=
                approval->evidence.campaignReviewIdentityCanonical ||
            reconstructedApproval.campaignReviewIdentityHash !=
                approval->evidence.campaignReviewIdentityHash)
            throw std::invalid_argument(
                "recommendation_campaign_materialization_approval_stale");

        std::vector<ProposedExperimentSpecification> proposals;
        proposals.reserve(static_cast<std::size_t>(plan.summary.selectedCount));
        for (const auto& candidate : plan.candidates)
        {
            if (candidate.decision != RecommendationCampaignDecision::include)
                continue;
            auto conversionRequest = LoadRecommendationCampaignConversionRequest(
                transaction, scope.rankingSnapshotId,
                candidate.input.rankingMemberId,
                candidate.input.recommendationId,
                candidate.input.campaignDonchian20Mode);
            const auto conversion = BuildProposedExperimentSpecification(
                conversionRequest);
            if (!conversion.eligibility.eligible || !conversion.proposal)
                throw std::invalid_argument(
                    "recommendation_campaign_materialization_member_ineligible:" +
                    RecommendationConversionReasonText(
                        conversion.eligibility.reason));
            proposals.push_back(*conversion.proposal);
        }
        const auto evidence = BuildRecommendationCampaignMaterializationEvidence(
            normalized, reconstructedApproval, plan, proposals);
        auto result = PersistRecommendationCampaignMaterialization(
            transaction, evidence);
        transaction.commit();
        const bool recorded = result.outcome ==
            RecommendationCampaignMaterializationPersistOutcome::recorded;
        PrintMaterialization(
            output,
            recorded ? "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_RECORDED"
                     : "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_ALREADY_RECORDED",
            result.materialization, false, recorded,
            result.newlyCreatedProposalCount, result.reusedProposalCount);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors, false, false, false, 0);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        const std::string message = error.what();
        errors << (message == "recommendation_campaign_materialization_conflict"
                       ? "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_CONFLICT"
                       : "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_FAILED")
               << ",error=" << RecommendationMachineText(message) << ',';
        PrintSafety(errors, false, false, false, 0);
        errors << '\n';
        return message == "recommendation_campaign_materialization_conflict"
            ? 3 : 2;
    }
}

int RunShowRecommendationCampaignMaterializationCommand(
    const std::string& connectionString,
    long long materializationId,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        if (!RecommendationCampaignMaterializationSchemaExists(connection))
            throw std::runtime_error(
                "recommendation_campaign_materialization_schema_required");
        const auto value = FindRecommendationCampaignMaterialization(
            connection, materializationId);
        if (!value)
        {
            errors << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_NOT_FOUND"
                   << ",campaign_materialization_id=" << materializationId
                   << ',';
            PrintSafety(errors, true, false, false, 0);
            errors << '\n';
            return 1;
        }
        PrintMaterialization(
            output, "RECOMMENDATION_CAMPAIGN_MATERIALIZATION", *value,
            true, false, 0, 0);
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors, true, false, false, 0);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_FAILED,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors, true, false, false, 0);
        errors << '\n';
        return 2;
    }
}

int RunListRecommendationCampaignMaterializationsCommand(
    const std::string& connectionString,
    std::optional<long long> campaignApprovalId,
    int limit,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        if (!RecommendationCampaignMaterializationSchemaExists(connection))
            throw std::runtime_error(
                "recommendation_campaign_materialization_schema_required");
        const auto values = ListRecommendationCampaignMaterializations(
            connection, campaignApprovalId, limit);
        for (const auto& value : values)
            PrintMaterialization(output,
                "RECOMMENDATION_CAMPAIGN_MATERIALIZATION", value,
                true, false, 0, 0);
        output << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_LIST_COMPLETE,count="
               << values.size() << ',';
        PrintSafety(output, true, false, false, 0);
        output << '\n';
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors, true, false, false, 0);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_MATERIALIZATION_FAILED,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors, true, false, false, 0);
        errors << '\n';
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
