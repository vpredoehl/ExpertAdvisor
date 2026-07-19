#include "ExperimentRecommendationCampaignProposalReviewRepository.hpp"

#include "ExperimentRecommendationCampaignHandoffRepository.hpp"
#include "ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{

bool RecommendationCampaignProposalReviewSchemasExist(
    pqxx::transaction_base& transaction)
{
    return RecommendationCampaignHandoffSchemasExist(transaction);
}

std::optional<PersistedRecommendationCampaignMaterialization>
LoadRecommendationCampaignProposalReviewMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId)
{
    if (materializationId <= 0)
        throw std::invalid_argument(
            "campaign_proposal_review_materialization_id_invalid");
    return FindRecommendationCampaignMaterialization(
        transaction, materializationId);
}

void LockRecommendationCampaignProposalReviews(
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
        throw std::runtime_error(
            "campaign_proposal_review_duplicate_proposal_link");
    for (const auto proposalId : proposalIds)
        LockRecommendationConversionProposalReviewSequence(
            transaction, proposalId);
}

RecommendationCampaignProposalReviewInput
LoadRecommendationCampaignProposalReviewInput(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationCampaignMaterialization& materialization)
{
    const auto handoff = FindRecommendationCampaignHandoff(
        transaction, materialization.materializationId);
    if (!handoff)
        throw std::runtime_error(
            "campaign_proposal_review_materialization_disappeared");
    std::vector<long long> proposalIds;
    proposalIds.reserve(materialization.members.size());
    for (const auto& member : materialization.members)
        proposalIds.push_back(member.conversionProposalId);
    const auto currentReviews =
        ListRecommendationConversionProposalCurrentReviews(
            transaction, proposalIds);
    std::map<long long, RecommendationConversionProposalCurrentReview> byId;
    for (const auto& review : currentReviews)
    {
        if (!byId.emplace(review.proposalId, review.currentReview).second)
            throw std::runtime_error(
                "campaign_proposal_review_duplicate_review_projection");
    }

    RecommendationCampaignProposalReviewInput input;
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
        RecommendationCampaignProposalReviewMemberInput member;
        member.memberOrdinal = stored.memberOrdinal;
        member.materializationMemberId = stored.materializationMemberId;
        member.proposalId = stored.conversionProposalId;
        const auto found = byId.find(stored.conversionProposalId);
        if (found == byId.end())
        {
            input.workflowEvidenceConsistent = false;
        }
        else
        {
            member.authoritativeReview = found->second.latestDecision;
        }
        input.members.push_back(std::move(member));
    }
    return input;
}

std::vector<PersistedRecommendationConversionProposalReviewDecision>
PersistRecommendationCampaignProposalReviews(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignProposalReviewPlan& plan)
{
    if (plan.state != RecommendationCampaignProposalReviewPlanState::ready)
        throw std::invalid_argument(
            "campaign_proposal_review_plan_not_writable");
    std::vector<PersistedRecommendationConversionProposalReviewDecision>
        persisted;
    persisted.reserve(plan.members.size());
    for (const auto& member : plan.members)
    {
        if (member.action !=
            RecommendationCampaignProposalReviewMemberAction::insert)
            throw std::invalid_argument(
                "campaign_proposal_review_partial_write_forbidden");
        RecommendationConversionProposalReviewRequest request;
        request.proposalId = member.proposalId;
        request.decision = plan.request.decision;
        request.requestId = plan.phase4cRequestId;
        request.operatorIdentity = plan.request.operatorIdentity;
        request.reasonText = plan.request.reasonText;
        auto result = PersistRecommendationConversionProposalReviewDecision(
            transaction, request);
        if (result.outcome !=
                RecommendationConversionProposalReviewPersistOutcome::recorded ||
            !result.decision)
            throw std::runtime_error(
                "campaign_proposal_review_insert_did_not_record");
        persisted.push_back(std::move(*result.decision));
    }
    return persisted;
}

} // namespace EA::ExperimentRecommendation
