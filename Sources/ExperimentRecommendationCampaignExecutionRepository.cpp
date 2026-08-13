#include "ExperimentRecommendationCampaignExecutionRepository.hpp"

#include "ExperimentRecommendationCampaignHandoffRepository.hpp"
#include "ExperimentRecommendationConversionRepository.hpp"
#include "ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignExecutionSchemasExist(
    pqxx::transaction_base& transaction)
{
    return RecommendationCampaignHandoffSchemasExist(transaction);
}

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignExecutionMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId)
{
    if (materializationId <= 0)
        throw std::invalid_argument("campaign_execution_materialization_id_invalid");
    return FindRecommendationCampaignMaterialization(
        transaction, materializationId);
}

void LockRecommendationCampaignExecutions(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    std::vector<long long> proposalIds;
    proposalIds.reserve(materialization.members.size());
    for (const auto& member : materialization.members)
        proposalIds.push_back(member.conversionProposalId);
    std::sort(proposalIds.begin(), proposalIds.end());
    if (std::adjacent_find(proposalIds.begin(), proposalIds.end()) !=
        proposalIds.end())
        throw std::runtime_error("campaign_execution_duplicate_proposal_link");
    for (const auto proposalId : proposalIds)
        LockRecommendationConversionProposalReviewSequence(
            transaction, proposalId);
}

RecommendationCampaignExecutionInput LoadRecommendationCampaignExecutionInput(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    const auto handoff = FindRecommendationCampaignHandoff(
        transaction, materialization.materializationId);
    if (!handoff)
        throw std::runtime_error("campaign_execution_materialization_disappeared");

    std::map<int, const RecommendationCampaignHandoffMember*> byOrdinal;
    for (const auto& member : handoff->members)
    {
        if (!byOrdinal.emplace(member.memberOrdinal, &member).second)
            throw std::runtime_error("campaign_execution_duplicate_handoff_ordinal");
    }

    RecommendationCampaignExecutionInput input;
    input.materializationId = materialization.materializationId;
    input.materializationContractVersion = materialization.contractVersion;
    input.materializationIdentityCanonical = materialization.identityCanonical;
    input.materializationIdentityHash = materialization.identityHash;
    input.selectedMemberCount = materialization.selectedMemberCount;
    input.materializationComplete = handoff->materializationComplete;
    input.workflowEvidenceConsistent =
        handoff->integrity == RecommendationCampaignHandoffIntegrity::consistent;
    input.members.reserve(materialization.members.size());

    for (const auto& stored : materialization.members)
    {
        RecommendationCampaignExecutionMemberInput member;
        member.memberOrdinal = stored.memberOrdinal;
        member.materializationMemberId = stored.materializationMemberId;
        member.proposalId = stored.conversionProposalId;
        const auto found = byOrdinal.find(stored.memberOrdinal);
        if (found == byOrdinal.end() ||
            found->second->conversionProposalId != stored.conversionProposalId)
        {
            input.workflowEvidenceConsistent = false;
            input.members.push_back(std::move(member));
            continue;
        }
        const auto& observed = *found->second;
        const auto proposal = FindRecommendationConversionProposal(
            transaction, stored.conversionProposalId);
        if (!proposal ||
            proposal->proposal.recommendationId != stored.recommendationId ||
            proposal->proposal.sourceExperimentId != stored.sourceExperimentId ||
            proposal->proposal.conversionIdentityCanonical !=
                stored.proposalIdentityCanonical ||
            proposal->proposal.conversionIdentityHash !=
                stored.proposalIdentityHash)
            input.workflowEvidenceConsistent = false;
        member.authorizationReviewDecisionId = observed.reviewDecisionId;
        member.authoritativeReviewApproved =
            observed.reviewStatus == RecommendationCampaignHandoffReviewStatus::approved;
        member.executionId = observed.executionId;
        if (observed.executionId)
        {
            const auto execution = FindRecommendationConversionExecutionByProposal(
                transaction, stored.conversionProposalId);
            if (!execution || execution->executionId != *observed.executionId)
            {
                input.workflowEvidenceConsistent = false;
            }
            else
            {
                ValidatePersistedRecommendationConversionExecution(
                    transaction, *execution);
                // An immutable execution remains authorized by its original
                // approving review even if a later ordinary review reverses
                // the proposal. Preserve that durable linkage in retry output.
                member.authorizationReviewDecisionId =
                    execution->reviewDecisionId;
                member.experimentId = execution->experimentId;
            }
        }
        input.members.push_back(std::move(member));
    }
    return input;
}

std::vector<PersistedRecommendationConversionExecution>
PersistRecommendationCampaignExecutions(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignExecutionPlan& plan)
{
    if (plan.state != RecommendationCampaignExecutionPlanState::ready)
        throw std::invalid_argument("campaign_execution_plan_not_writable");
    std::vector<PersistedRecommendationConversionExecution> persisted;
    persisted.reserve(plan.members.size());
    for (const auto& member : plan.members)
    {
        if (member.action != RecommendationCampaignExecutionMemberAction::create)
            throw std::invalid_argument("campaign_execution_partial_write_forbidden");
        RecommendationConversionExecutionResult result;
        try
        {
            result = ExecuteApprovedRecommendationConversionProposalInTransaction(
                transaction, member.proposalId);
        }
        catch (const pqxx::unique_violation&)
        {
            throw std::runtime_error(
                "campaign_execution_experiment_identity_conflict");
        }
        if (result.outcome != RecommendationConversionExecutionOutcome::created ||
            !result.execution || !member.authorizationReviewDecisionId ||
            result.execution->reviewDecisionId !=
                *member.authorizationReviewDecisionId)
            throw std::runtime_error("campaign_execution_insert_did_not_create");
        persisted.push_back(std::move(*result.execution));
    }
    return persisted;
}

} // namespace EA::ExperimentRecommendation
