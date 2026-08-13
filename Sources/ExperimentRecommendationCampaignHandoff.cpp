#include "ExperimentRecommendationCampaignHandoff.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void AddDiagnostic(
    RecommendationCampaignHandoffMember& member,
    const std::string& code)
{
    if (std::find(member.diagnosticCodes.begin(),
                  member.diagnosticCodes.end(), code) ==
        member.diagnosticCodes.end())
        member.diagnosticCodes.push_back(code);
}

void ValidateImmutableInput(const RecommendationCampaignHandoffInput& input)
{
    if (input.materializationId <= 0 || input.campaignApprovalId <= 0 ||
        input.materializationContractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        input.materializationIdentityCanonical.empty() ||
        input.materializationIdentityHash != RecommendationCanonicalHash(
            input.materializationIdentityCanonical) ||
        input.selectedMemberCount <= 0 ||
        input.selectedMemberCount != static_cast<int>(input.members.size()))
        throw std::invalid_argument(
            "recommendation_campaign_handoff_materialization_invalid");

    std::set<long long> proposals;
    for (std::size_t index = 0; index < input.members.size(); ++index)
    {
        const auto& member = input.members[index];
        if (member.memberOrdinal != static_cast<int>(index) + 1 ||
            member.rankingMemberId <= 0 || member.recommendationId <= 0 ||
            member.sourceExperimentId <= 0 || member.rankingPosition <= 0 ||
            member.conversionProposalId <= 0 ||
            member.selectedMemberIdentityCanonical.empty() ||
            member.selectedMemberIdentityHash != RecommendationCanonicalHash(
                member.selectedMemberIdentityCanonical) ||
            member.proposalIdentityCanonical.empty() ||
            member.proposalIdentityHash != RecommendationCanonicalHash(
                member.proposalIdentityCanonical) ||
            !proposals.insert(member.conversionProposalId).second)
            throw std::invalid_argument(
                "recommendation_campaign_handoff_materialization_member_invalid");
    }
}

RecommendationCampaignHandoffState DeriveState(
    const RecommendationCampaignHandoffSummary& summary)
{
    if (summary.integrityFailures > 0)
        return RecommendationCampaignHandoffState::inconsistent;
    if (summary.rejected > 0)
        return RecommendationCampaignHandoffState::reviewRejected;
    if (summary.approved == 0)
        return RecommendationCampaignHandoffState::awaitingPhase4cReview;
    if (summary.awaitingReview > 0)
        return RecommendationCampaignHandoffState::partiallyReviewed;
    if (summary.executionsPresent == 0)
        return RecommendationCampaignHandoffState::readyForPhase4cExecution;
    if (summary.executionsPresent < summary.totalMembers)
        return RecommendationCampaignHandoffState::partiallyExecuted;
    if (summary.activationsPresent == 0)
        return RecommendationCampaignHandoffState::readyForPhase4cActivation;
    if (summary.activationsPresent < summary.totalMembers)
        return RecommendationCampaignHandoffState::partiallyActivated;
    return RecommendationCampaignHandoffState::fullyActivated;
}

} // namespace

std::string RecommendationCampaignHandoffStateText(
    RecommendationCampaignHandoffState state)
{
    switch (state)
    {
        case RecommendationCampaignHandoffState::awaitingPhase4cReview:
            return "awaiting_phase4c_review";
        case RecommendationCampaignHandoffState::partiallyReviewed:
            return "partially_reviewed";
        case RecommendationCampaignHandoffState::reviewRejected:
            return "review_rejected";
        case RecommendationCampaignHandoffState::readyForPhase4cExecution:
            return "ready_for_phase4c_execution";
        case RecommendationCampaignHandoffState::partiallyExecuted:
            return "partially_executed";
        case RecommendationCampaignHandoffState::readyForPhase4cActivation:
            return "ready_for_phase4c_activation";
        case RecommendationCampaignHandoffState::partiallyActivated:
            return "partially_activated";
        case RecommendationCampaignHandoffState::fullyActivated:
            return "fully_activated";
        case RecommendationCampaignHandoffState::inconsistent:
            return "inconsistent";
    }
    throw std::invalid_argument("invalid_recommendation_campaign_handoff_state");
}

std::string RecommendationCampaignHandoffReviewStatusText(
    RecommendationCampaignHandoffReviewStatus status)
{
    switch (status)
    {
        case RecommendationCampaignHandoffReviewStatus::awaitingReview:
            return "awaiting_review";
        case RecommendationCampaignHandoffReviewStatus::approved:
            return "approved";
        case RecommendationCampaignHandoffReviewStatus::rejected:
            return "rejected";
    }
    throw std::invalid_argument(
        "invalid_recommendation_campaign_handoff_review_status");
}

std::string RecommendationCampaignHandoffExecutionStatusText(
    RecommendationCampaignHandoffExecutionStatus status)
{
    switch (status)
    {
        case RecommendationCampaignHandoffExecutionStatus::notExecuted:
            return "not_executed";
        case RecommendationCampaignHandoffExecutionStatus::executed:
            return "executed";
    }
    throw std::invalid_argument(
        "invalid_recommendation_campaign_handoff_execution_status");
}

std::string RecommendationCampaignHandoffActivationStatusText(
    RecommendationCampaignHandoffActivationStatus status)
{
    switch (status)
    {
        case RecommendationCampaignHandoffActivationStatus::notActivated:
            return "not_activated";
        case RecommendationCampaignHandoffActivationStatus::activated:
            return "activated";
    }
    throw std::invalid_argument(
        "invalid_recommendation_campaign_handoff_activation_status");
}

std::string RecommendationCampaignHandoffIntegrityText(
    RecommendationCampaignHandoffIntegrity integrity)
{
    switch (integrity)
    {
        case RecommendationCampaignHandoffIntegrity::consistent:
            return "consistent";
        case RecommendationCampaignHandoffIntegrity::inconsistent:
            return "inconsistent";
    }
    throw std::invalid_argument(
        "invalid_recommendation_campaign_handoff_integrity");
}

RecommendationCampaignHandoff BuildRecommendationCampaignHandoff(
    const RecommendationCampaignHandoffInput& input)
{
    ValidateImmutableInput(input);
    RecommendationCampaignHandoff result;
    result.materializationId = input.materializationId;
    result.campaignApprovalId = input.campaignApprovalId;
    result.materializationIdentityHash = input.materializationIdentityHash;
    result.materializationComplete = true;
    result.summary.totalMembers = input.selectedMemberCount;
    result.members.reserve(input.members.size());

    for (const auto& source : input.members)
    {
        RecommendationCampaignHandoffMember member;
        member.memberOrdinal = source.memberOrdinal;
        member.rankingMemberId = source.rankingMemberId;
        member.recommendationId = source.recommendationId;
        member.sourceExperimentId = source.sourceExperimentId;
        member.rankingPosition = source.rankingPosition;
        member.conversionProposalId = source.conversionProposalId;
        member.proposalIdentityHash = source.proposalIdentityHash;

        if (!source.proposal)
        {
            AddDiagnostic(member, "linked_proposal_missing");
        }
        else
        {
            const auto& proposal = *source.proposal;
            ++result.summary.proposalsPresent;
            if (proposal.proposalId != source.conversionProposalId)
                AddDiagnostic(member, "proposal_id_mismatch");
            if (proposal.recommendationId != source.recommendationId)
                AddDiagnostic(member, "proposal_recommendation_mismatch");
            if (proposal.sourceExperimentId != source.sourceExperimentId)
                AddDiagnostic(member, "proposal_source_experiment_mismatch");
            if (proposal.identityCanonical != source.proposalIdentityCanonical)
                AddDiagnostic(member, "proposal_identity_canonical_mismatch");
            if (proposal.identityHash != source.proposalIdentityHash)
                AddDiagnostic(member, "proposal_identity_hash_mismatch");

            if (proposal.latestReview)
            {
                member.reviewDecisionId =
                    proposal.latestReview->reviewDecisionId;
                if (!IsValidRecommendationConversionWorkflowReviewFact(
                        *proposal.latestReview, proposal.proposalId))
                    AddDiagnostic(member, "latest_review_invalid");
                else if (proposal.latestReview->decision == "approve")
                    member.reviewStatus =
                        RecommendationCampaignHandoffReviewStatus::approved;
                else
                    member.reviewStatus =
                        RecommendationCampaignHandoffReviewStatus::rejected;
            }
            member.executionId = proposal.executionId;
            if (member.executionId)
                member.executionStatus =
                    RecommendationCampaignHandoffExecutionStatus::executed;
            member.activationId = proposal.activationId;
            if (member.activationId)
                member.activationStatus =
                    RecommendationCampaignHandoffActivationStatus::activated;
            member.workflowState = proposal.workflow.state;
            for (const auto& code : proposal.workflow.diagnosticCodes)
                AddDiagnostic(member, code);
            if (proposal.workflow.integrity ==
                    RecommendationConversionWorkflowIntegrity::inconsistent &&
                proposal.workflow.diagnosticCodes.empty())
                AddDiagnostic(member, "downstream_workflow_inconsistent");
        }

        switch (member.reviewStatus)
        {
            case RecommendationCampaignHandoffReviewStatus::awaitingReview:
                ++result.summary.awaitingReview;
                ++result.summary.noAuthoritativeReview;
                break;
            case RecommendationCampaignHandoffReviewStatus::approved:
                ++result.summary.approved;
                break;
            case RecommendationCampaignHandoffReviewStatus::rejected:
                ++result.summary.rejected;
                break;
        }
        if (member.executionId) ++result.summary.executionsPresent;
        if (member.activationId) ++result.summary.activationsPresent;
        if (!member.diagnosticCodes.empty())
        {
            member.integrity = RecommendationCampaignHandoffIntegrity::inconsistent;
            ++result.summary.integrityFailures;
        }
        result.members.push_back(std::move(member));
    }

    result.state = DeriveState(result.summary);
    result.integrity = result.summary.integrityFailures == 0
        ? RecommendationCampaignHandoffIntegrity::consistent
        : RecommendationCampaignHandoffIntegrity::inconsistent;
    return result;
}

} // namespace EA::ExperimentRecommendation
