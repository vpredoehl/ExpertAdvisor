#include "../Sources/ExperimentRecommendationCampaignLaunch.hpp"
#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationConversionActivation.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignLaunchInput BaseInput(bool existingExecutions)
{
    RecommendationCampaignLaunchInput input;
    input.materializationId = 71;
    input.materializationContractVersion =
        kRecommendationCampaignMaterializationContractVersion;
    input.materializationIdentityCanonical = "launch-materialization";
    input.materializationIdentityHash = RecommendationCanonicalHash(
        input.materializationIdentityCanonical);
    input.selectedMemberCount = 2;
    input.members = {
        {1, 101, 201, 301, 401, 501},
        {2, 102, 202, 302, 402, 502}};

    auto& execution = input.executionPlan;
    execution.request = {71, false};
    execution.materializationIdentityHash = input.materializationIdentityHash;
    execution.state = existingExecutions
        ? RecommendationCampaignExecutionPlanState::alreadySatisfied
        : RecommendationCampaignExecutionPlanState::ready;
    for (int index = 0; index < 2; ++index)
    {
        RecommendationCampaignExecutionMemberPlan member;
        member.memberOrdinal = index + 1;
        member.materializationMemberId = 101 + index;
        member.proposalId = 501 + index;
        member.authorizationReviewDecisionId = 601 + index;
        if (existingExecutions)
        {
            member.action = RecommendationCampaignExecutionMemberAction::
                alreadySatisfied;
            member.previousExecutionId = 701 + index;
            member.previousExperimentId = 801 + index;
        }
        else
        {
            member.action = RecommendationCampaignExecutionMemberAction::create;
        }
        execution.members.push_back(std::move(member));
    }
    return input;
}

RecommendationCampaignActivationPlan ActivationPlan(bool alreadySatisfied)
{
    RecommendationCampaignActivationPlan plan;
    plan.request = {71, false};
    plan.materializationIdentityHash = RecommendationCanonicalHash(
        "launch-materialization");
    plan.state = alreadySatisfied
        ? RecommendationCampaignActivationPlanState::alreadySatisfied
        : RecommendationCampaignActivationPlanState::ready;
    for (int index = 0; index < 2; ++index)
    {
        RecommendationCampaignActivationMemberPlan member;
        member.memberOrdinal = index + 1;
        member.materializationMemberId = 101 + index;
        member.recommendationId = 301 + index;
        member.proposalId = 501 + index;
        member.authorizationReviewDecisionId = 601 + index;
        member.executionId = 701 + index;
        member.experimentId = 801 + index;
        member.postActivationStatus =
            kRecommendationConversionActivationResultingStatus;
        member.postActivationPhase =
            kRecommendationConversionActivationResultingPhase;
        if (alreadySatisfied)
        {
            member.previousActivationId = 901 + index;
            member.preActivationStatus =
                kRecommendationConversionActivationResultingStatus;
            member.preActivationPhase =
                kRecommendationConversionActivationResultingPhase;
            member.action = RecommendationCampaignActivationMemberAction::
                alreadySatisfied;
        }
        else
        {
            member.preActivationStatus =
                kRecommendationConversionActivationPreviousStatus;
            member.preActivationPhase =
                kRecommendationConversionActivationPreviousPhase;
            member.action = RecommendationCampaignActivationMemberAction::activate;
        }
        plan.members.push_back(std::move(member));
    }
    return plan;
}

void ExpectInvalid(const std::function<void()>& operation)
{
    bool rejected = false;
    try
    {
        operation();
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main()
{
    const auto allNew = BuildRecommendationCampaignLaunchPlan(
        {71, false}, BaseInput(false));
    assert(allNew.state ==
           RecommendationCampaignLaunchPlanState::readyToLaunch);
    assert(allNew.executionCreateCount == 2);
    assert(allNew.activationCreateCount == 2);
    assert(!allNew.members[0].executionId);
    assert(!allNew.members[0].experimentId);
    assert(!allNew.members[0].activationId);
    assert(allNew.members[0].activationDisposition ==
           RecommendationCampaignLaunchActivationDisposition::
               createAfterExecution);

    const auto deterministic = BuildRecommendationCampaignLaunchPlan(
        {71, true}, BaseInput(false));
    assert(deterministic.operationIdentityCanonical ==
           allNew.operationIdentityCanonical);
    assert(deterministic.operationIdentityHash == allNew.operationIdentityHash);
    assert(deterministic.members[0].memberOrdinal == 1);
    assert(deterministic.members[1].memberOrdinal == 2);

    auto changedIdentity = BaseInput(false);
    changedIdentity.materializationIdentityCanonical += "-changed";
    changedIdentity.materializationIdentityHash = RecommendationCanonicalHash(
        changedIdentity.materializationIdentityCanonical);
    changedIdentity.executionPlan.materializationIdentityHash =
        changedIdentity.materializationIdentityHash;
    const auto changed = BuildRecommendationCampaignLaunchPlan(
        {71, false}, changedIdentity);
    assert(changed.operationIdentityHash != allNew.operationIdentityHash);

    auto existing = BaseInput(true);
    existing.activationPlan = ActivationPlan(false);
    const auto activateExisting = BuildRecommendationCampaignLaunchPlan(
        {71, false}, existing);
    assert(activateExisting.state == RecommendationCampaignLaunchPlanState::
           readyToActivateExistingExecutions);
    assert(activateExisting.executionReuseCount == 2);
    assert(activateExisting.activationCreateCount == 2);
    assert(activateExisting.members[0].executionId == 701);
    assert(activateExisting.members[0].preStatus == "paused");

    auto satisfiedInput = BaseInput(true);
    satisfiedInput.activationPlan = ActivationPlan(true);
    const auto satisfied = BuildRecommendationCampaignLaunchPlan(
        {71, false}, satisfiedInput);
    assert(satisfied.state ==
           RecommendationCampaignLaunchPlanState::alreadySatisfied);
    assert(satisfied.activationReuseCount == 2);
    assert(satisfied.alreadySatisfiedCount == 2);
    assert(satisfied.members[0].activationId == 901);
    assert(satisfied.members[0].authorizationReviewDecisionId == 601);

    ExpectInvalid([] {
        auto input = BaseInput(false);
        input.members.clear();
        input.selectedMemberCount = 0;
        input.executionPlan.members.clear();
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(false);
        input.members[1].memberOrdinal = 3;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(false);
        input.executionPlan.members[1].action =
            RecommendationCampaignExecutionMemberAction::alreadySatisfied;
        input.executionPlan.members[1].previousExecutionId = 702;
        input.executionPlan.members[1].previousExperimentId = 802;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(true);
        auto activation = ActivationPlan(false);
        activation.members[1].action =
            RecommendationCampaignActivationMemberAction::alreadySatisfied;
        activation.members[1].previousActivationId = 902;
        activation.members[1].preActivationStatus = "pending";
        input.activationPlan = activation;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(true);
        input.executionPlan.members[1].previousExperimentId.reset();
        input.activationPlan = ActivationPlan(false);
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(true);
        auto activation = ActivationPlan(false);
        activation.members[1].proposalId = 999;
        input.activationPlan = activation;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(true);
        auto activation = ActivationPlan(false);
        activation.members[0].preActivationStatus = "running";
        input.activationPlan = activation;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(false);
        input.members[0].proposalId = input.members[1].proposalId;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(false);
        input.materializationContractVersion = 99;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        auto input = BaseInput(false);
        input.selectedMemberCount = 3;
        (void)BuildRecommendationCampaignLaunchPlan({71, false}, input);
    });
    ExpectInvalid([] {
        (void)NormalizeRecommendationCampaignLaunchRequest({0, false});
    });

    assert(RecommendationCampaignLaunchPlanStateText(allNew.state) ==
           "ready_to_launch");
    assert(RecommendationCampaignLaunchExecutionDispositionText(
               allNew.members[0].executionDisposition) == "create");
    assert(RecommendationCampaignLaunchActivationDispositionText(
               allNew.members[0].activationDisposition) ==
           "create_after_execution");

    std::cout << "Experiment recommendation campaign launch tests passed\n";
    return 0;
}
