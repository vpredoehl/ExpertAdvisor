#include "ExperimentRecommendationCampaignExecution.hpp"

#include "ExperimentRecommendation.hpp"

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
    const RecommendationCampaignExecutionInput& input)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_execution_v1"
        << ";operation_contract_version="
        << kRecommendationCampaignExecutionContractVersion
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

std::string RecommendationCampaignExecutionPlanStateText(
    RecommendationCampaignExecutionPlanState state)
{
    switch (state)
    {
        case RecommendationCampaignExecutionPlanState::ready:
            return "ready";
        case RecommendationCampaignExecutionPlanState::alreadySatisfied:
            return "already_satisfied";
    }
    throw std::invalid_argument("campaign_execution_plan_state_invalid");
}

std::string RecommendationCampaignExecutionMemberActionText(
    RecommendationCampaignExecutionMemberAction action)
{
    switch (action)
    {
        case RecommendationCampaignExecutionMemberAction::create:
            return "create";
        case RecommendationCampaignExecutionMemberAction::alreadySatisfied:
            return "already_satisfied";
    }
    throw std::invalid_argument("campaign_execution_member_action_invalid");
}

RecommendationCampaignExecutionRequest NormalizeRecommendationCampaignExecutionRequest(
    const RecommendationCampaignExecutionRequest& request)
{
    if (request.materializationId <= 0)
        throw std::invalid_argument("campaign_execution_materialization_id_invalid");
    return request;
}

RecommendationCampaignExecutionPlan BuildRecommendationCampaignExecutionPlan(
    const RecommendationCampaignExecutionRequest& request,
    const RecommendationCampaignExecutionInput& input)
{
    const auto normalized = NormalizeRecommendationCampaignExecutionRequest(request);
    if (input.materializationId != normalized.materializationId ||
        input.materializationContractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        input.materializationIdentityCanonical.empty() ||
        input.materializationIdentityHash != RecommendationCanonicalHash(
            input.materializationIdentityCanonical))
        throw std::invalid_argument("campaign_execution_materialization_identity_invalid");
    if (!input.materializationComplete || !input.workflowEvidenceConsistent)
        throw std::invalid_argument("campaign_execution_workflow_evidence_invalid");
    if (input.selectedMemberCount <= 0 || input.members.empty())
        throw std::invalid_argument("campaign_execution_zero_members");
    if (input.selectedMemberCount != static_cast<int>(input.members.size()))
        throw std::invalid_argument("campaign_execution_member_count_mismatch");
    if (input.members.size() > static_cast<std::size_t>(
            kMaximumRecommendationCampaignExecutionMembers))
        throw std::invalid_argument("campaign_execution_member_limit_exceeded");

    RecommendationCampaignExecutionPlan plan;
    plan.request = normalized;
    plan.materializationIdentityHash = input.materializationIdentityHash;
    plan.operationIdentityCanonical = OperationCanonical(input);
    plan.operationIdentityHash = RecommendationCanonicalHash(
        plan.operationIdentityCanonical);
    plan.members.reserve(input.members.size());

    std::set<long long> memberIds;
    std::set<long long> proposalIds;
    bool sawCreate = false;
    bool sawExisting = false;
    for (std::size_t index = 0; index < input.members.size(); ++index)
    {
        const auto& source = input.members[index];
        if (source.memberOrdinal != static_cast<int>(index) + 1 ||
            source.materializationMemberId <= 0 || source.proposalId <= 0 ||
            !memberIds.insert(source.materializationMemberId).second ||
            !proposalIds.insert(source.proposalId).second)
            throw std::invalid_argument(
                "campaign_execution_member_identity_invalid");

        RecommendationCampaignExecutionMemberPlan member;
        member.memberOrdinal = source.memberOrdinal;
        member.materializationMemberId = source.materializationMemberId;
        member.proposalId = source.proposalId;
        member.authorizationReviewDecisionId =
            source.authorizationReviewDecisionId;
        member.previousExecutionId = source.executionId;
        member.previousExperimentId = source.experimentId;
        if (source.executionId)
        {
            if (*source.executionId <= 0 || !source.experimentId ||
                *source.experimentId <= 0)
                throw std::invalid_argument(
                    "campaign_execution_existing_execution_invalid");
            member.action =
                RecommendationCampaignExecutionMemberAction::alreadySatisfied;
            member.diagnosticCodes.push_back("campaign_execution_already_satisfied");
            ++plan.alreadySatisfiedCount;
            sawExisting = true;
        }
        else
        {
            if (source.experimentId)
                throw std::invalid_argument(
                    "campaign_execution_orphan_experiment_invalid");
            if (!source.authoritativeReviewApproved ||
                !source.authorizationReviewDecisionId ||
                *source.authorizationReviewDecisionId <= 0)
                throw std::invalid_argument(
                    "campaign_execution_proposal_not_approved");
            member.action = RecommendationCampaignExecutionMemberAction::create;
            member.diagnosticCodes.push_back("campaign_execution_required");
            sawCreate = true;
        }
        plan.members.push_back(std::move(member));
    }

    plan.proposalsValidated = static_cast<int>(plan.members.size());
    if (sawCreate && sawExisting)
        throw std::invalid_argument("campaign_execution_partial_operation_conflict");
    if (sawExisting)
    {
        plan.state = RecommendationCampaignExecutionPlanState::alreadySatisfied;
        plan.diagnosticCodes.push_back("campaign_execution_already_satisfied");
    }
    else
    {
        plan.state = RecommendationCampaignExecutionPlanState::ready;
        plan.diagnosticCodes.push_back("campaign_execution_ready");
    }
    return plan;
}

} // namespace EA::ExperimentRecommendation
