#include "ExperimentRecommendationCampaignActivation.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationConversionActivation.hpp"

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

std::string OperationCanonical(
    const RecommendationCampaignActivationInput& input)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_activation_v1"
        << ";operation_contract_version="
        << kRecommendationCampaignActivationContractVersion
        << ";materialization_id=" << input.materializationId
        << ";materialization_contract_version="
        << input.materializationContractVersion
        << ";materialization_identity_canonical="
        << LengthText(input.materializationIdentityCanonical)
        << ";materialization_identity_hash="
        << LengthText(input.materializationIdentityHash);
    return out.str();
}

} // namespace

RecommendationCampaignActivationRequest NormalizeRecommendationCampaignActivationRequest(
    const RecommendationCampaignActivationRequest& request)
{
    if (request.materializationId <= 0)
        throw std::invalid_argument(
            "campaign_activation_materialization_id_invalid");
    return request;
}

RecommendationCampaignActivationPlan BuildRecommendationCampaignActivationPlan(
    const RecommendationCampaignActivationRequest& request,
    const RecommendationCampaignActivationInput& input)
{
    const auto normalized = NormalizeRecommendationCampaignActivationRequest(
        request);
    if (input.materializationId != normalized.materializationId ||
        input.materializationContractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        input.materializationIdentityCanonical.empty() ||
        input.materializationIdentityHash != RecommendationCanonicalHash(
            input.materializationIdentityCanonical))
        throw std::invalid_argument(
            "campaign_activation_materialization_identity_invalid");
    if (!input.materializationComplete || !input.workflowEvidenceConsistent)
        throw std::invalid_argument(
            "campaign_activation_workflow_evidence_invalid");
    if (input.selectedMemberCount <= 0 || input.members.empty())
        throw std::invalid_argument("campaign_activation_zero_members");
    if (input.selectedMemberCount != static_cast<int>(input.members.size()))
        throw std::invalid_argument("campaign_activation_member_count_mismatch");
    if (input.members.size() > static_cast<std::size_t>(
            kMaximumRecommendationCampaignActivationMembers))
        throw std::invalid_argument("campaign_activation_member_limit_exceeded");

    RecommendationCampaignActivationPlan plan;
    plan.request = normalized;
    plan.materializationIdentityHash = input.materializationIdentityHash;
    plan.operationIdentityCanonical = OperationCanonical(input);
    plan.operationIdentityHash = RecommendationCanonicalHash(
        plan.operationIdentityCanonical);
    plan.members.reserve(input.members.size());

    std::set<long long> memberIds;
    std::set<long long> proposalIds;
    std::set<long long> executionIds;
    std::set<long long> experimentIds;
    bool sawActivate = false;
    bool sawExisting = false;
    for (std::size_t index = 0; index < input.members.size(); ++index)
    {
        const auto& source = input.members[index];
        if (source.memberOrdinal != static_cast<int>(index) + 1 ||
            source.materializationMemberId <= 0 ||
            source.recommendationId <= 0 || source.proposalId <= 0 ||
            !source.authorizationReviewDecisionId ||
            *source.authorizationReviewDecisionId <= 0 ||
            !source.executionId || *source.executionId <= 0 ||
            !source.experimentId || *source.experimentId <= 0 ||
            !memberIds.insert(source.materializationMemberId).second ||
            !proposalIds.insert(source.proposalId).second ||
            !executionIds.insert(*source.executionId).second ||
            !experimentIds.insert(*source.experimentId).second)
            throw std::invalid_argument(
                "campaign_activation_member_identity_invalid");
        if (source.evidenceState ==
            RecommendationCampaignActivationEvidenceState::invalid)
            throw std::invalid_argument(
                source.evidenceDiagnostic.empty()
                    ? "campaign_activation_member_evidence_invalid"
                    : source.evidenceDiagnostic);

        RecommendationCampaignActivationMemberPlan member;
        member.memberOrdinal = source.memberOrdinal;
        member.materializationMemberId = source.materializationMemberId;
        member.recommendationId = source.recommendationId;
        member.proposalId = source.proposalId;
        member.authorizationReviewDecisionId =
            *source.authorizationReviewDecisionId;
        member.executionId = *source.executionId;
        member.experimentId = *source.experimentId;
        member.previousActivationId = source.activationId;
        member.preActivationStatus = source.experimentStatus;
        member.preActivationPhase = source.experimentPhase;
        member.postActivationStatus =
            kRecommendationConversionActivationResultingStatus;
        member.postActivationPhase =
            kRecommendationConversionActivationResultingPhase;

        if (source.evidenceState ==
            RecommendationCampaignActivationEvidenceState::eligible)
        {
            if (source.activationId ||
                source.experimentStatus !=
                    kRecommendationConversionActivationPreviousStatus ||
                source.experimentPhase !=
                    kRecommendationConversionActivationPreviousPhase)
                throw std::invalid_argument(
                    "campaign_activation_eligible_state_invalid");
            member.action =
                RecommendationCampaignActivationMemberAction::activate;
            member.diagnosticCodes.push_back("campaign_activation_required");
            sawActivate = true;
        }
        else
        {
            if (!source.activationId || *source.activationId <= 0 ||
                source.experimentStatus !=
                    kRecommendationConversionActivationResultingStatus ||
                source.experimentPhase !=
                    kRecommendationConversionActivationResultingPhase)
                throw std::invalid_argument(
                    "campaign_activation_existing_state_invalid");
            member.action = RecommendationCampaignActivationMemberAction::
                alreadySatisfied;
            member.diagnosticCodes.push_back(
                "campaign_activation_already_satisfied");
            ++plan.alreadyActivatedCount;
            sawExisting = true;
        }
        plan.members.push_back(std::move(member));
    }

    plan.membersValidated = static_cast<int>(plan.members.size());
    if (sawActivate && sawExisting)
        throw std::invalid_argument(
            "campaign_activation_partial_operation_conflict");
    if (sawExisting)
    {
        plan.state = RecommendationCampaignActivationPlanState::alreadySatisfied;
        plan.diagnosticCodes.push_back(
            "campaign_activation_already_satisfied");
    }
    else
    {
        plan.state = RecommendationCampaignActivationPlanState::ready;
        plan.diagnosticCodes.push_back("campaign_activation_ready");
    }
    return plan;
}

} // namespace EA::ExperimentRecommendation
