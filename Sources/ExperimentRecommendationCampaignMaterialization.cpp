#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include "ExperimentRecommendation.hpp"

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

bool HasControlOrNul(const std::string& value)
{
    return std::any_of(value.begin(), value.end(), [](unsigned char value) {
        return value == 0 || value < 0x20 || value == 0x7f;
    });
}

std::string TrimAscii(const std::string& value)
{
    const auto blank = [](unsigned char character) {
        return character == ' ' || character == '\t' || character == '\n' ||
               character == '\r' || character == '\f' || character == '\v';
    };
    std::size_t begin = 0;
    while (begin < value.size() && blank(value[begin])) ++begin;
    std::size_t end = value.size();
    while (end > begin && blank(value[end - 1])) --end;
    return value.substr(begin, end - begin);
}

std::string SelectedMemberCanonical(
    int ordinal,
    const RecommendationCampaignPlanCandidate& candidate,
    const ProposedExperimentSpecification& proposal)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_materialization_member_v1"
        << ";ordinal=" << ordinal
        << ";ranking_member_id=" << candidate.input.rankingMemberId
        << ";ranking_position=" << candidate.input.rankingPosition
        << ";recommendation_id=" << candidate.input.recommendationId
        << ";source_experiment_id=" << candidate.input.sourceExperimentId
        << ";recommendation_semantic="
        << LengthText(candidate.input.recommendationSemanticCanonical)
        << ";recommendation_semantic_hash="
        << LengthText(candidate.input.recommendationSemanticHash)
        << ";recommendation_invocation="
        << LengthText(candidate.input.recommendationInvocationCanonical)
        << ";recommendation_invocation_hash="
        << LengthText(candidate.input.recommendationInvocationHash)
        << ";proposal_identity="
        << LengthText(proposal.conversionIdentityCanonical)
        << ";proposal_identity_hash="
        << LengthText(proposal.conversionIdentityHash);
    return out.str();
}

std::string MaterializationCanonical(
    const RecommendationCampaignMaterializationEvidence& evidence)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_materialization_v1"
        << ";contract_version=" << evidence.materializationContractVersion
        << ";campaign_approval_id=" << evidence.campaignApprovalId
        << ";campaign_approval_identity="
        << LengthText(evidence.approval.approvalIdentityCanonical)
        << ";campaign_approval_identity_hash="
        << LengthText(evidence.approval.approvalIdentityHash)
        << ";ranking_snapshot_id=" << evidence.approval.rankingSnapshotId
        << ";ranking_snapshot_identity="
        << LengthText(evidence.approval.rankingSnapshotIdentityCanonical)
        << ";ranking_snapshot_identity_hash="
        << LengthText(evidence.approval.rankingSnapshotIdentityHash)
        << ";planning_policy="
        << LengthText(evidence.approval.planningPolicyCanonical)
        << ";planning_policy_hash="
        << LengthText(evidence.approval.planningPolicyHash)
        << ";planning_scope="
        << LengthText(evidence.approval.planningScopeCanonical)
        << ";campaign_plan_identity="
        << LengthText(evidence.approval.campaignPlanIdentityCanonical)
        << ";campaign_plan_identity_hash="
        << LengthText(evidence.approval.campaignPlanIdentityHash)
        << ";campaign_review_identity="
        << LengthText(evidence.approval.campaignReviewIdentityCanonical)
        << ";campaign_review_identity_hash="
        << LengthText(evidence.approval.campaignReviewIdentityHash)
        << ";approval_decision="
        << RecommendationCampaignApprovalDecisionText(
               evidence.approval.decision)
        << ";approval_reviewer="
        << LengthText(evidence.approval.reviewerIdentity)
        << ";approval_reason=" << LengthText(evidence.approval.reasonText)
        << ";materialized_by=" << LengthText(evidence.operatorIdentity)
        << ";materialization_reason=" << LengthText(evidence.reasonText)
        << ";selected_member_count=" << evidence.members.size();
    for (std::size_t i = 0; i < evidence.members.size(); ++i)
    {
        const auto& member = evidence.members[i];
        out << ";member[" << i << "].identity="
            << LengthText(member.selectedMemberIdentityCanonical)
            << ";member[" << i << "].identity_hash="
            << LengthText(member.selectedMemberIdentityHash)
            << ";member[" << i << "].proposal_identity="
            << LengthText(member.proposal.conversionIdentityCanonical)
            << ";member[" << i << "].proposal_identity_hash="
            << LengthText(member.proposal.conversionIdentityHash);
    }
    return out.str();
}

} // namespace

RecommendationCampaignMaterializationRequest
NormalizeRecommendationCampaignMaterializationRequest(
    const RecommendationCampaignMaterializationRequest& request)
{
    RecommendationCampaignMaterializationRequest normalized = request;
    normalized.operatorIdentity = TrimAscii(request.operatorIdentity);
    normalized.reasonText = TrimAscii(request.reasonText);
    if (normalized.campaignApprovalId <= 0 ||
        normalized.operatorIdentity.empty() || normalized.reasonText.empty() ||
        normalized.operatorIdentity.size() >
            kRecommendationCampaignMaterializationOperatorMaximum ||
        normalized.reasonText.size() >
            kRecommendationCampaignMaterializationReasonMaximum ||
        HasControlOrNul(normalized.operatorIdentity) ||
        HasControlOrNul(normalized.reasonText))
        throw std::invalid_argument(
            "recommendation_campaign_materialization_request_invalid");
    return normalized;
}

RecommendationCampaignMaterializationEvidence
BuildRecommendationCampaignMaterializationEvidence(
    const RecommendationCampaignMaterializationRequest& request,
    const RecommendationCampaignApprovalEvidence& approval,
    const RecommendationCampaignPlan& plan,
    const std::vector<ProposedExperimentSpecification>& proposals)
{
    const auto normalized =
        NormalizeRecommendationCampaignMaterializationRequest(request);
    ValidateRecommendationCampaignApprovalEvidence(approval);
    if (approval.decision != RecommendationCampaignApprovalDecision::approved)
        throw std::invalid_argument(
            "recommendation_campaign_materialization_approval_rejected");
    if (approval.campaignPlanIdentityCanonical != plan.identityCanonical ||
        approval.campaignPlanIdentityHash != plan.identityHash ||
        approval.summary.selectedCount <= 0 ||
        approval.summary.selectedCount != plan.summary.selectedCount)
        throw std::invalid_argument(
            "recommendation_campaign_materialization_approval_stale");

    std::vector<const RecommendationCampaignPlanCandidate*> selected;
    for (const auto& candidate : plan.candidates)
        if (candidate.decision == RecommendationCampaignDecision::include)
            selected.push_back(&candidate);
    if (selected.size() != proposals.size() ||
        selected.size() != static_cast<std::size_t>(approval.summary.selectedCount))
        throw std::invalid_argument(
            "recommendation_campaign_materialization_member_count_mismatch");

    RecommendationCampaignMaterializationEvidence evidence;
    evidence.campaignApprovalId = normalized.campaignApprovalId;
    evidence.approval = approval;
    evidence.operatorIdentity = normalized.operatorIdentity;
    evidence.reasonText = normalized.reasonText;
    evidence.selectedMemberCount = static_cast<int>(selected.size());
    std::set<std::string> proposalCanonicals;
    for (std::size_t index = 0; index < selected.size(); ++index)
    {
        const auto& candidate = *selected[index];
        const auto& proposal = proposals[index];
        if (candidate.input.recommendationId != proposal.recommendationId ||
            candidate.input.sourceExperimentId != proposal.sourceExperimentId ||
            !proposalCanonicals.insert(
                proposal.conversionIdentityCanonical).second)
            throw std::invalid_argument(
                "recommendation_campaign_materialization_member_invalid");
        if (candidate.input.campaignDonchian20Mode &&
            proposal.proposedInvocation.configuration.donchian20Mode !=
                *candidate.input.campaignDonchian20Mode)
            throw std::invalid_argument(
                "recommendation_campaign_materialization_donchian20_arm_mismatch");
        RecommendationCampaignMaterializationSelectedMember member;
        member.memberOrdinal = static_cast<int>(index) + 1;
        member.rankingMemberId = candidate.input.rankingMemberId;
        member.recommendationId = candidate.input.recommendationId;
        member.sourceExperimentId = candidate.input.sourceExperimentId;
        member.rankingPosition = candidate.input.rankingPosition;
        member.proposal = proposal;
        member.selectedMemberIdentityCanonical = SelectedMemberCanonical(
            member.memberOrdinal, candidate, proposal);
        member.selectedMemberIdentityHash = RecommendationCanonicalHash(
            member.selectedMemberIdentityCanonical);
        evidence.members.push_back(std::move(member));
    }
    evidence.materializationIdentityCanonical =
        MaterializationCanonical(evidence);
    evidence.materializationIdentityHash = RecommendationCanonicalHash(
        evidence.materializationIdentityCanonical);
    ValidateRecommendationCampaignMaterializationEvidence(evidence);
    return evidence;
}

void ValidateRecommendationCampaignMaterializationEvidence(
    const RecommendationCampaignMaterializationEvidence& evidence)
{
    if (evidence.materializationContractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        evidence.campaignApprovalId <= 0 ||
        evidence.approval.decision !=
            RecommendationCampaignApprovalDecision::approved ||
        evidence.selectedMemberCount <= 0 ||
        evidence.selectedMemberCount !=
            static_cast<int>(evidence.members.size()) ||
        evidence.operatorIdentity.empty() || evidence.reasonText.empty() ||
        evidence.operatorIdentity.size() >
            kRecommendationCampaignMaterializationOperatorMaximum ||
        evidence.reasonText.size() >
            kRecommendationCampaignMaterializationReasonMaximum ||
        HasControlOrNul(evidence.operatorIdentity) ||
        HasControlOrNul(evidence.reasonText) ||
        evidence.materializationIdentityCanonical.empty() ||
        evidence.materializationIdentityCanonical.size() >
            kRecommendationCampaignMaterializationIdentityCanonicalMaximum ||
        evidence.materializationIdentityHash != RecommendationCanonicalHash(
            evidence.materializationIdentityCanonical) ||
        MaterializationCanonical(evidence) !=
            evidence.materializationIdentityCanonical)
        throw std::invalid_argument(
            "recommendation_campaign_materialization_evidence_invalid");
    ValidateRecommendationCampaignApprovalEvidence(evidence.approval);
    std::set<int> ordinals;
    std::set<std::string> proposalCanonicals;
    for (const auto& member : evidence.members)
    {
        if (member.memberOrdinal <= 0 ||
            member.memberOrdinal > evidence.selectedMemberCount ||
            member.rankingMemberId <= 0 || member.recommendationId <= 0 ||
            member.sourceExperimentId <= 0 || member.rankingPosition <= 0 ||
            member.proposal.recommendationId != member.recommendationId ||
            member.proposal.sourceExperimentId != member.sourceExperimentId ||
            member.selectedMemberIdentityCanonical.empty() ||
            member.selectedMemberIdentityHash != RecommendationCanonicalHash(
                member.selectedMemberIdentityCanonical) ||
            member.proposal.conversionIdentityHash != RecommendationCanonicalHash(
                member.proposal.conversionIdentityCanonical) ||
            !ordinals.insert(member.memberOrdinal).second ||
            !proposalCanonicals.insert(
                member.proposal.conversionIdentityCanonical).second)
            throw std::invalid_argument(
                "recommendation_campaign_materialization_member_invalid");
    }
    for (int ordinal = 1; ordinal <= evidence.selectedMemberCount; ++ordinal)
        if (!ordinals.contains(ordinal))
            throw std::invalid_argument(
                "recommendation_campaign_materialization_member_ordinal_invalid");
}

} // namespace EA::ExperimentRecommendation
