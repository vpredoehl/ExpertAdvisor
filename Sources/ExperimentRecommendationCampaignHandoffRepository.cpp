#include "ExperimentRecommendationCampaignHandoffRepository.hpp"

#include "ExperimentRecommendationCampaignMaterializationRepository.hpp"
#include "ExperimentRecommendationConversionWorkflowRepository.hpp"

#include <map>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

RecommendationCampaignHandoff BuildFromMaterialization(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    RecommendationCampaignHandoffInput input;
    input.materializationId = materialization.materializationId;
    input.campaignApprovalId = materialization.campaignApprovalId;
    input.materializationContractVersion = materialization.contractVersion;
    input.materializationIdentityCanonical = materialization.identityCanonical;
    input.materializationIdentityHash = materialization.identityHash;
    input.selectedMemberCount = materialization.selectedMemberCount;

    std::vector<long long> proposalIds;
    proposalIds.reserve(materialization.members.size());
    for (const auto& member : materialization.members)
        proposalIds.push_back(member.conversionProposalId);
    const auto workflows = ListRecommendationConversionWorkflowsForProposals(
        transaction, proposalIds);
    std::map<long long, const RecommendationConversionWorkflowView*> byProposal;
    for (const auto& workflow : workflows)
    {
        if (!byProposal.emplace(workflow.proposalId, &workflow).second)
            throw std::runtime_error(
                "recommendation_campaign_handoff_duplicate_proposal_projection");
    }

    input.members.reserve(materialization.members.size());
    for (const auto& stored : materialization.members)
    {
        RecommendationCampaignHandoffMemberInput member;
        member.memberOrdinal = stored.memberOrdinal;
        member.rankingMemberId = stored.rankingMemberId;
        member.recommendationId = stored.recommendationId;
        member.sourceExperimentId = stored.sourceExperimentId;
        member.rankingPosition = stored.rankingPosition;
        member.selectedMemberIdentityCanonical =
            stored.selectedMemberIdentityCanonical;
        member.selectedMemberIdentityHash = stored.selectedMemberIdentityHash;
        member.conversionProposalId = stored.conversionProposalId;
        member.proposalIdentityCanonical = stored.proposalIdentityCanonical;
        member.proposalIdentityHash = stored.proposalIdentityHash;
        const auto found = byProposal.find(stored.conversionProposalId);
        if (found != byProposal.end())
        {
            const auto& workflow = *found->second;
            RecommendationCampaignHandoffProposalFacts proposal;
            proposal.proposalId = workflow.proposalId;
            proposal.recommendationId = workflow.recommendationId;
            proposal.sourceExperimentId = workflow.sourceExperimentId;
            proposal.identityCanonical = workflow.proposalIdentityCanonical;
            proposal.identityHash = workflow.proposalIdentityHash;
            proposal.latestReview = workflow.latestReview;
            if (workflow.execution)
                proposal.executionId = workflow.execution->executionId;
            if (workflow.activation)
                proposal.activationId = workflow.activation->activationId;
            proposal.workflow = workflow.derivation;
            member.proposal = std::move(proposal);
        }
        input.members.push_back(std::move(member));
    }
    return BuildRecommendationCampaignHandoff(input);
}

void ValidateLimit(int limit)
{
    if (limit <= 0 || limit > kMaximumRecommendationCampaignHandoffListLimit)
        throw std::invalid_argument(
            "recommendation_campaign_handoff_limit_invalid");
}

} // namespace

bool RecommendationCampaignHandoffSchemasExist(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(R"SQL(
SELECT to_regclass('experiment_recommendation_campaign_materialization') IS NOT NULL
   AND to_regclass('experiment_recommendation_campaign_materialization_member') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_proposal') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_review_decision') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_execution') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_activation') IS NOT NULL
   AND to_regclass('experiment') IS NOT NULL;
)SQL").one_row()[0].as<bool>();
}

std::optional<RecommendationCampaignHandoff>
FindRecommendationCampaignHandoff(
    pqxx::transaction_base& transaction,
    long long materializationId)
{
    if (materializationId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_handoff_materialization_id_invalid");
    const auto materialization = FindRecommendationCampaignMaterialization(
        transaction, materializationId);
    if (!materialization) return std::nullopt;
    return BuildFromMaterialization(transaction, *materialization);
}

std::vector<RecommendationCampaignHandoff>
ListRecommendationCampaignHandoffs(
    pqxx::transaction_base& transaction,
    int limit)
{
    ValidateLimit(limit);
    const auto materializations = ListRecommendationCampaignMaterializations(
        transaction, std::nullopt, limit);
    std::vector<RecommendationCampaignHandoff> handoffs;
    handoffs.reserve(materializations.size());
    for (const auto& materialization : materializations)
        handoffs.push_back(BuildFromMaterialization(
            transaction, materialization));
    return handoffs;
}

} // namespace EA::ExperimentRecommendation
