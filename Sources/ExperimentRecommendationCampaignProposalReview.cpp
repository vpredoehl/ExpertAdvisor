#include "ExperimentRecommendationCampaignProposalReview.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

RecommendationConversionProposalReviewDisposition DispositionFor(
    RecommendationConversionProposalReviewDecision decision)
{
    return decision == RecommendationConversionProposalReviewDecision::approve
        ? RecommendationConversionProposalReviewDisposition::approved
        : RecommendationConversionProposalReviewDisposition::rejected;
}

std::string OperationCanonical(
    const RecommendationCampaignProposalReviewRequest& request,
    const RecommendationCampaignProposalReviewInput& input)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_proposal_review_v1"
        << ";operation_contract_version="
        << kRecommendationCampaignProposalReviewContractVersion
        << ";materialization_id=" << input.materializationId
        << ";materialization_contract_version="
        << input.materializationContractVersion
        << ";materialization_identity_canonical="
        << LengthText(input.materializationIdentityCanonical)
        << ";materialization_identity_hash="
        << LengthText(input.materializationIdentityHash)
        << ";decision="
        << LengthText(RecommendationConversionProposalReviewDecisionText(
               request.decision))
        << ";operator=" << LengthText(request.operatorIdentity)
        << ";reason=" << LengthText(request.reasonText);
    return out.str();
}

void ValidatePersistedReview(
    const PersistedRecommendationConversionProposalReviewDecision& review,
    long long proposalId)
{
    if (review.reviewDecisionId <= 0 || review.proposalId != proposalId)
        throw std::invalid_argument(
            "campaign_proposal_review_authoritative_review_invalid");
    RecommendationConversionProposalReviewRequest request;
    request.proposalId = review.proposalId;
    request.decision = review.decision;
    request.requestId = review.requestId;
    request.operatorIdentity = review.operatorIdentity;
    request.reasonText = review.reasonText;
    try
    {
        const auto normalized =
            NormalizeRecommendationConversionProposalReviewRequest(request);
        if (normalized.requestId != review.requestId ||
            normalized.operatorIdentity != review.operatorIdentity ||
            normalized.reasonText != review.reasonText)
            throw std::invalid_argument("noncanonical_review");
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(
            "campaign_proposal_review_authoritative_review_invalid");
    }
}

} // namespace

std::string RecommendationCampaignProposalReviewPlanStateText(
    RecommendationCampaignProposalReviewPlanState state)
{
    switch (state)
    {
        case RecommendationCampaignProposalReviewPlanState::ready:
            return "ready";
        case RecommendationCampaignProposalReviewPlanState::alreadySatisfied:
            return "already_satisfied";
    }
    throw std::invalid_argument("campaign_proposal_review_plan_state_invalid");
}

std::string RecommendationCampaignProposalReviewMemberActionText(
    RecommendationCampaignProposalReviewMemberAction action)
{
    switch (action)
    {
        case RecommendationCampaignProposalReviewMemberAction::insert:
            return "insert";
        case RecommendationCampaignProposalReviewMemberAction::alreadySatisfied:
            return "already_satisfied";
    }
    throw std::invalid_argument("campaign_proposal_review_member_action_invalid");
}

RecommendationCampaignProposalReviewRequest
NormalizeRecommendationCampaignProposalReviewRequest(
    const RecommendationCampaignProposalReviewRequest& request)
{
    if (request.materializationId <= 0)
        throw std::invalid_argument(
            "campaign_proposal_review_materialization_id_invalid");
    RecommendationConversionProposalReviewRequest phase4c;
    phase4c.proposalId = 1;
    phase4c.decision = request.decision;
    phase4c.requestId = "campaign-review-validation";
    phase4c.operatorIdentity = request.operatorIdentity;
    phase4c.reasonText = request.reasonText;
    const auto normalizedPhase4c =
        NormalizeRecommendationConversionProposalReviewRequest(phase4c);
    if (!normalizedPhase4c.operatorIdentity)
        throw std::invalid_argument(
            "campaign_proposal_review_operator_required");
    if (!normalizedPhase4c.reasonText)
        throw std::invalid_argument("campaign_proposal_review_reason_required");
    RecommendationCampaignProposalReviewRequest normalized = request;
    normalized.operatorIdentity = *normalizedPhase4c.operatorIdentity;
    normalized.reasonText = *normalizedPhase4c.reasonText;
    return normalized;
}

RecommendationCampaignProposalReviewPlan
BuildRecommendationCampaignProposalReviewPlan(
    const RecommendationCampaignProposalReviewRequest& request,
    const RecommendationCampaignProposalReviewInput& input)
{
    const auto normalized =
        NormalizeRecommendationCampaignProposalReviewRequest(request);
    if (input.materializationId != normalized.materializationId ||
        input.materializationContractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        input.materializationIdentityCanonical.empty() ||
        input.materializationIdentityHash != RecommendationCanonicalHash(
            input.materializationIdentityCanonical))
        throw std::invalid_argument(
            "campaign_proposal_review_materialization_identity_invalid");
    if (!input.materializationComplete || !input.workflowEvidenceConsistent)
        throw std::invalid_argument(
            "campaign_proposal_review_workflow_evidence_invalid");
    if (input.selectedMemberCount <= 0 || input.members.empty())
        throw std::invalid_argument("campaign_proposal_review_zero_members");
    if (input.selectedMemberCount != static_cast<int>(input.members.size()))
        throw std::invalid_argument(
            "campaign_proposal_review_member_count_mismatch");
    if (input.members.size() > static_cast<std::size_t>(
            kMaximumRecommendationCampaignProposalReviewMembers))
        throw std::invalid_argument(
            "campaign_proposal_review_member_limit_exceeded");

    RecommendationCampaignProposalReviewPlan plan;
    plan.request = normalized;
    plan.materializationIdentityHash = input.materializationIdentityHash;
    plan.operationIdentityCanonical = OperationCanonical(normalized, input);
    plan.operationIdentityHash = RecommendationCanonicalHash(
        plan.operationIdentityCanonical);
    plan.phase4cRequestId = "campaign-review-v1-m" +
        std::to_string(input.materializationId) + "-" +
        plan.operationIdentityHash;
    if (plan.phase4cRequestId.size() >
        kRecommendationConversionProposalReviewRequestIdMaximum)
        throw std::invalid_argument(
            "campaign_proposal_review_request_identity_too_long");

    std::set<long long> memberIds;
    std::set<long long> proposalIds;
    plan.members.reserve(input.members.size());
    bool sawInsert = false;
    bool sawSatisfied = false;
    for (std::size_t index = 0; index < input.members.size(); ++index)
    {
        const auto& source = input.members[index];
        if (source.memberOrdinal != static_cast<int>(index) + 1 ||
            source.materializationMemberId <= 0 || source.proposalId <= 0)
            throw std::invalid_argument(
                "campaign_proposal_review_member_identity_invalid");
        if (!memberIds.insert(source.materializationMemberId).second ||
            !proposalIds.insert(source.proposalId).second)
            throw std::invalid_argument(
                "campaign_proposal_review_duplicate_member_or_proposal");
        RecommendationCampaignProposalReviewMemberPlan member;
        member.memberOrdinal = source.memberOrdinal;
        member.materializationMemberId = source.materializationMemberId;
        member.proposalId = source.proposalId;
        if (!source.authoritativeReview)
        {
            member.action = RecommendationCampaignProposalReviewMemberAction::insert;
            member.diagnosticCodes.push_back("review_required");
            sawInsert = true;
        }
        else
        {
            const auto& review = *source.authoritativeReview;
            ValidatePersistedReview(review, source.proposalId);
            member.previousReviewDecisionId = review.reviewDecisionId;
            member.previousDisposition = DispositionFor(review.decision);
            if (review.decision != normalized.decision)
                throw std::invalid_argument(
                    "campaign_proposal_review_opposite_decision_conflict");
            if (review.requestId != plan.phase4cRequestId ||
                review.operatorIdentity !=
                    std::optional<std::string>{normalized.operatorIdentity} ||
                review.reasonText !=
                    std::optional<std::string>{normalized.reasonText})
                throw std::invalid_argument(
                    "campaign_proposal_review_existing_decision_conflict");
            member.action =
                RecommendationCampaignProposalReviewMemberAction::alreadySatisfied;
            member.diagnosticCodes.push_back("campaign_review_already_satisfied");
            ++plan.alreadySatisfiedCount;
            sawSatisfied = true;
        }
        plan.members.push_back(std::move(member));
    }
    plan.proposalsValidated = static_cast<int>(plan.members.size());
    if (sawInsert && sawSatisfied)
        throw std::invalid_argument(
            "campaign_proposal_review_partial_operation_conflict");
    if (sawSatisfied)
    {
        plan.state = RecommendationCampaignProposalReviewPlanState::alreadySatisfied;
        plan.diagnosticCodes.push_back("campaign_review_already_satisfied");
    }
    else
    {
        plan.state = RecommendationCampaignProposalReviewPlanState::ready;
        plan.diagnosticCodes.push_back("campaign_review_ready");
    }
    return plan;
}

} // namespace EA::ExperimentRecommendation
