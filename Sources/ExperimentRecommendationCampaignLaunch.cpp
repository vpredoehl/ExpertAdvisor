#include "ExperimentRecommendationCampaignLaunch.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationConversionActivation.hpp"

#include <locale>
#include <map>
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

std::string OperationCanonical(const RecommendationCampaignLaunchInput& input)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_launch_v1"
        << ";operation_contract_version="
        << kRecommendationCampaignLaunchContractVersion
        << ";operation_type=atomic_campaign_conversion_launch"
        << ";materialization_id=" << input.materializationId
        << ";materialization_contract_version="
        << input.materializationContractVersion
        << ";materialization_identity_canonical="
        << LengthText(input.materializationIdentityCanonical)
        << ";materialization_identity_hash="
        << LengthText(input.materializationIdentityHash)
        << ";ordered_member_count=" << input.members.size()
        << ";launch_semantics=phase4c_execution_then_phase4c_activation_v1";
    return out.str();
}

std::map<int, const RecommendationCampaignActivationMemberPlan*>
ActivationMembersByOrdinal(const RecommendationCampaignActivationPlan& plan)
{
    std::map<int, const RecommendationCampaignActivationMemberPlan*> result;
    for (const auto& member : plan.members)
        if (!result.emplace(member.memberOrdinal, &member).second)
            throw std::invalid_argument(
                "campaign_launch_duplicate_activation_ordinal");
    return result;
}

} // namespace

RecommendationCampaignLaunchRequest NormalizeRecommendationCampaignLaunchRequest(
    const RecommendationCampaignLaunchRequest& request)
{
    if (request.materializationId <= 0)
        throw std::invalid_argument("campaign_launch_materialization_id_invalid");
    return request;
}

RecommendationCampaignLaunchPlan BuildRecommendationCampaignLaunchPlan(
    const RecommendationCampaignLaunchRequest& request,
    const RecommendationCampaignLaunchInput& input)
{
    const auto normalized = NormalizeRecommendationCampaignLaunchRequest(request);
    if (input.materializationId != normalized.materializationId ||
        input.materializationContractVersion !=
            kRecommendationCampaignMaterializationContractVersion ||
        input.materializationIdentityCanonical.empty() ||
        input.materializationIdentityHash != RecommendationCanonicalHash(
            input.materializationIdentityCanonical))
        throw std::invalid_argument(
            "campaign_launch_materialization_identity_invalid");
    if (input.selectedMemberCount <= 0 || input.members.empty())
        throw std::invalid_argument("campaign_launch_zero_members");
    if (input.selectedMemberCount != static_cast<int>(input.members.size()))
        throw std::invalid_argument("campaign_launch_member_count_mismatch");
    if (input.members.size() > static_cast<std::size_t>(
            kMaximumRecommendationCampaignLaunchMembers))
        throw std::invalid_argument("campaign_launch_member_limit_exceeded");
    if (input.executionPlan.request.materializationId != input.materializationId ||
        input.executionPlan.members.size() != input.members.size() ||
        input.executionPlan.materializationIdentityHash !=
            input.materializationIdentityHash)
        throw std::invalid_argument("campaign_launch_execution_plan_invalid");
    if (input.activationPlan &&
        (input.activationPlan->request.materializationId !=
             input.materializationId ||
         input.activationPlan->members.size() != input.members.size() ||
         input.activationPlan->materializationIdentityHash !=
             input.materializationIdentityHash))
        throw std::invalid_argument("campaign_launch_activation_plan_invalid");
    if (input.activationPlan)
    {
        for (const auto& member : input.activationPlan->members)
        {
            const bool create = member.action ==
                RecommendationCampaignActivationMemberAction::activate;
            if ((input.activationPlan->state ==
                     RecommendationCampaignActivationPlanState::ready) !=
                    create ||
                (create && member.previousActivationId) ||
                (!create && !member.previousActivationId) ||
                member.preActivationStatus !=
                    (create
                         ? kRecommendationConversionActivationPreviousStatus
                         : kRecommendationConversionActivationResultingStatus) ||
                member.preActivationPhase !=
                    (create
                         ? kRecommendationConversionActivationPreviousPhase
                         : kRecommendationConversionActivationResultingPhase) ||
                member.postActivationStatus !=
                    kRecommendationConversionActivationResultingStatus ||
                member.postActivationPhase !=
                    kRecommendationConversionActivationResultingPhase)
                throw std::invalid_argument(
                    "campaign_launch_activation_state_invalid");
        }
    }

    RecommendationCampaignLaunchPlan plan;
    plan.request = normalized;
    plan.materializationContractVersion = input.materializationContractVersion;
    plan.materializationIdentityHash = input.materializationIdentityHash;
    plan.operationIdentityCanonical = OperationCanonical(input);
    plan.operationIdentityHash = RecommendationCanonicalHash(
        plan.operationIdentityCanonical);
    plan.members.reserve(input.members.size());

    const bool executionsCreated = input.executionPlan.state ==
        RecommendationCampaignExecutionPlanState::ready;
    if (!executionsCreated && !input.activationPlan)
        throw std::invalid_argument(
            "campaign_launch_existing_executions_require_activation_plan");
    if (executionsCreated && input.activationPlan &&
        input.activationPlan->state ==
            RecommendationCampaignActivationPlanState::alreadySatisfied)
        throw std::invalid_argument(
            "campaign_launch_new_execution_activation_state_invalid");

    std::map<int, const RecommendationCampaignActivationMemberPlan*>
        activationMembers;
    if (input.activationPlan)
        activationMembers = ActivationMembersByOrdinal(*input.activationPlan);

    std::set<long long> materializationMemberIds;
    std::set<long long> rankingMemberIds;
    std::set<long long> proposalIds;
    for (std::size_t index = 0; index < input.members.size(); ++index)
    {
        const auto& identity = input.members[index];
        const auto& execution = input.executionPlan.members[index];
        if (identity.memberOrdinal != static_cast<int>(index) + 1 ||
            identity.materializationMemberId <= 0 ||
            identity.rankingMemberId <= 0 || identity.recommendationId <= 0 ||
            identity.sourceExperimentId <= 0 || identity.proposalId <= 0 ||
            execution.memberOrdinal != identity.memberOrdinal ||
            execution.materializationMemberId !=
                identity.materializationMemberId ||
            execution.proposalId != identity.proposalId ||
            !materializationMemberIds.insert(
                identity.materializationMemberId).second ||
            !rankingMemberIds.insert(identity.rankingMemberId).second ||
            !proposalIds.insert(identity.proposalId).second)
            throw std::invalid_argument("campaign_launch_member_identity_invalid");

        RecommendationCampaignLaunchMemberPlan member;
        member.memberOrdinal = identity.memberOrdinal;
        member.materializationMemberId = identity.materializationMemberId;
        member.rankingMemberId = identity.rankingMemberId;
        member.recommendationId = identity.recommendationId;
        member.sourceExperimentId = identity.sourceExperimentId;
        member.proposalId = identity.proposalId;
        member.authorizationReviewDecisionId =
            execution.authorizationReviewDecisionId;

        if (executionsCreated)
        {
            if (execution.action !=
                    RecommendationCampaignExecutionMemberAction::create ||
                execution.previousExecutionId || execution.previousExperimentId)
                throw std::invalid_argument(
                    "campaign_launch_new_execution_plan_invalid");
            member.executionDisposition =
                RecommendationCampaignLaunchExecutionDisposition::create;
            ++plan.executionCreateCount;
            member.activationDisposition = input.activationPlan
                ? RecommendationCampaignLaunchActivationDisposition::create
                : RecommendationCampaignLaunchActivationDisposition::
                      createAfterExecution;
            ++plan.activationCreateCount;
            member.preStatus =
                kRecommendationConversionActivationPreviousStatus;
            member.prePhase = kRecommendationConversionActivationPreviousPhase;
            member.postStatus =
                kRecommendationConversionActivationResultingStatus;
            member.postPhase =
                kRecommendationConversionActivationResultingPhase;
            member.diagnosticCodes.push_back(
                "campaign_launch_execution_and_activation_required");
        }
        else
        {
            if (execution.action != RecommendationCampaignExecutionMemberAction::
                    alreadySatisfied ||
                !execution.previousExecutionId ||
                !execution.previousExperimentId)
                throw std::invalid_argument(
                    "campaign_launch_existing_execution_plan_invalid");
            member.executionDisposition =
                RecommendationCampaignLaunchExecutionDisposition::reuse;
            member.executionId = execution.previousExecutionId;
            member.experimentId = execution.previousExperimentId;
            ++plan.executionReuseCount;
        }

        if (input.activationPlan)
        {
            const auto found = activationMembers.find(identity.memberOrdinal);
            if (found == activationMembers.end())
                throw std::invalid_argument(
                    "campaign_launch_activation_member_missing");
            const auto& activation = *found->second;
            if (activation.materializationMemberId !=
                    identity.materializationMemberId ||
                activation.recommendationId != identity.recommendationId ||
                activation.proposalId != identity.proposalId ||
                (!executionsCreated &&
                 (activation.executionId != *member.executionId ||
                  activation.experimentId != *member.experimentId)))
                throw std::invalid_argument(
                    "campaign_launch_activation_member_invalid");
            member.authorizationReviewDecisionId =
                activation.authorizationReviewDecisionId;
            member.executionId = activation.executionId;
            member.experimentId = activation.experimentId;
            member.activationId = activation.previousActivationId;
            member.preStatus = activation.preActivationStatus;
            member.prePhase = activation.preActivationPhase;
            member.postStatus = activation.postActivationStatus;
            member.postPhase = activation.postActivationPhase;
            if (activation.action ==
                RecommendationCampaignActivationMemberAction::activate)
            {
                member.activationDisposition =
                    RecommendationCampaignLaunchActivationDisposition::create;
                if (!executionsCreated) ++plan.activationCreateCount;
                member.diagnosticCodes.push_back(
                    "campaign_launch_activation_required");
            }
            else
            {
                if (executionsCreated)
                    throw std::invalid_argument(
                        "campaign_launch_new_execution_already_activated");
                member.activationDisposition =
                    RecommendationCampaignLaunchActivationDisposition::reuse;
                ++plan.activationReuseCount;
                ++plan.alreadySatisfiedCount;
                member.diagnosticCodes.push_back(
                    "campaign_launch_already_satisfied");
            }
        }
        plan.members.push_back(std::move(member));
    }

    plan.membersValidated = static_cast<int>(plan.members.size());
    if (executionsCreated)
    {
        plan.state = RecommendationCampaignLaunchPlanState::readyToLaunch;
        plan.diagnosticCodes.push_back("campaign_launch_ready");
    }
    else if (input.activationPlan->state ==
             RecommendationCampaignActivationPlanState::ready)
    {
        plan.state = RecommendationCampaignLaunchPlanState::
            readyToActivateExistingExecutions;
        plan.diagnosticCodes.push_back(
            "campaign_launch_existing_executions_ready");
    }
    else
    {
        plan.state = RecommendationCampaignLaunchPlanState::alreadySatisfied;
        plan.diagnosticCodes.push_back("campaign_launch_already_satisfied");
    }
    return plan;
}

std::string RecommendationCampaignLaunchPlanStateText(
    RecommendationCampaignLaunchPlanState state)
{
    switch (state)
    {
        case RecommendationCampaignLaunchPlanState::readyToLaunch:
            return "ready_to_launch";
        case RecommendationCampaignLaunchPlanState::
                readyToActivateExistingExecutions:
            return "ready_to_activate_existing_executions";
        case RecommendationCampaignLaunchPlanState::alreadySatisfied:
            return "already_satisfied";
    }
    throw std::invalid_argument("campaign_launch_plan_state_invalid");
}

std::string RecommendationCampaignLaunchExecutionDispositionText(
    RecommendationCampaignLaunchExecutionDisposition disposition)
{
    switch (disposition)
    {
        case RecommendationCampaignLaunchExecutionDisposition::create:
            return "create";
        case RecommendationCampaignLaunchExecutionDisposition::reuse:
            return "reuse";
    }
    throw std::invalid_argument(
        "campaign_launch_execution_disposition_invalid");
}

std::string RecommendationCampaignLaunchActivationDispositionText(
    RecommendationCampaignLaunchActivationDisposition disposition)
{
    switch (disposition)
    {
        case RecommendationCampaignLaunchActivationDisposition::
                createAfterExecution:
            return "create_after_execution";
        case RecommendationCampaignLaunchActivationDisposition::create:
            return "create";
        case RecommendationCampaignLaunchActivationDisposition::reuse:
            return "reuse";
    }
    throw std::invalid_argument(
        "campaign_launch_activation_disposition_invalid");
}

} // namespace EA::ExperimentRecommendation
