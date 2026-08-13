#include "../Sources/ExperimentRecommendationCampaignExecution.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignExecutionRequest Request(bool dryRun = false)
{
    return {17, dryRun};
}

RecommendationCampaignExecutionInput Input(int count = 2)
{
    RecommendationCampaignExecutionInput input;
    input.materializationId = 17;
    input.materializationContractVersion =
        kRecommendationCampaignMaterializationContractVersion;
    input.materializationIdentityCanonical = "materialization-v1";
    input.materializationIdentityHash = RecommendationCanonicalHash(
        input.materializationIdentityCanonical);
    input.selectedMemberCount = count;
    input.materializationComplete = true;
    input.workflowEvidenceConsistent = true;
    for (int ordinal = 1; ordinal <= count; ++ordinal)
    {
        RecommendationCampaignExecutionMemberInput member;
        member.memberOrdinal = ordinal;
        member.materializationMemberId = 100 + ordinal;
        member.proposalId = 200 + ordinal;
        member.authorizationReviewDecisionId = 300 + ordinal;
        member.authoritativeReviewApproved = true;
        input.members.push_back(member);
    }
    return input;
}

void Throws(const std::string& expected, const std::function<void()>& action)
{
    bool threw = false;
    try
    {
        action();
    }
    catch (const std::invalid_argument& error)
    {
        threw = true;
        assert(std::string{error.what()} == expected);
    }
    assert(threw);
}

} // namespace

int main()
{
    const auto ready = BuildRecommendationCampaignExecutionPlan(
        Request(), Input());
    assert(ready.state == RecommendationCampaignExecutionPlanState::ready);
    assert(ready.members.size() == 2);
    assert(ready.members[0].memberOrdinal == 1);
    assert(ready.members[1].memberOrdinal == 2);
    assert(ready.members[0].action ==
           RecommendationCampaignExecutionMemberAction::create);
    assert(ready.proposalsValidated == 2);
    assert(ready.alreadySatisfiedCount == 0);
    assert(ready.operationIdentityHash == RecommendationCanonicalHash(
        ready.operationIdentityCanonical));

    const auto deterministic = BuildRecommendationCampaignExecutionPlan(
        Request(true), Input());
    assert(deterministic.operationIdentityCanonical ==
           ready.operationIdentityCanonical);
    assert(deterministic.operationIdentityHash == ready.operationIdentityHash);
    assert(deterministic.request.dryRun);

    auto satisfiedInput = Input();
    for (std::size_t i = 0; i < satisfiedInput.members.size(); ++i)
    {
        satisfiedInput.members[i].executionId = 400 + i;
        satisfiedInput.members[i].experimentId = 500 + i;
        // A later review reversal cannot erase an immutable Phase 4C
        // execution, so exact execution retry remains satisfied.
        satisfiedInput.members[i].authoritativeReviewApproved = false;
    }
    const auto satisfied = BuildRecommendationCampaignExecutionPlan(
        Request(), satisfiedInput);
    assert(satisfied.state ==
           RecommendationCampaignExecutionPlanState::alreadySatisfied);
    assert(satisfied.alreadySatisfiedCount == 2);
    assert(satisfied.members[0].authorizationReviewDecisionId == 301);

    auto partial = satisfiedInput;
    partial.members[1].executionId.reset();
    partial.members[1].experimentId.reset();
    partial.members[1].authoritativeReviewApproved = true;
    Throws("campaign_execution_partial_operation_conflict", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), partial);
    });

    auto pending = Input();
    pending.members[0].authorizationReviewDecisionId.reset();
    pending.members[0].authoritativeReviewApproved = false;
    Throws("campaign_execution_proposal_not_approved", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), pending);
    });

    auto rejected = Input();
    rejected.members[1].authoritativeReviewApproved = false;
    Throws("campaign_execution_proposal_not_approved", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), rejected);
    });

    auto invalid = Request();
    invalid.materializationId = 0;
    Throws("campaign_execution_materialization_id_invalid", [&] {
        NormalizeRecommendationCampaignExecutionRequest(invalid);
    });

    auto zero = Input(0);
    Throws("campaign_execution_zero_members", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), zero);
    });

    auto countMismatch = Input();
    countMismatch.selectedMemberCount = 3;
    Throws("campaign_execution_member_count_mismatch", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), countMismatch);
    });

    auto badHash = Input();
    badHash.materializationIdentityHash = "bad";
    Throws("campaign_execution_materialization_identity_invalid", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), badHash);
    });

    auto badVersion = Input();
    badVersion.materializationContractVersion = 99;
    Throws("campaign_execution_materialization_identity_invalid", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), badVersion);
    });

    auto gap = Input();
    gap.members[1].memberOrdinal = 3;
    Throws("campaign_execution_member_identity_invalid", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), gap);
    });

    auto duplicateMember = Input();
    duplicateMember.members[1].materializationMemberId =
        duplicateMember.members[0].materializationMemberId;
    Throws("campaign_execution_member_identity_invalid", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), duplicateMember);
    });

    auto duplicateProposal = Input();
    duplicateProposal.members[1].proposalId =
        duplicateProposal.members[0].proposalId;
    Throws("campaign_execution_member_identity_invalid", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), duplicateProposal);
    });

    auto inconsistent = Input();
    inconsistent.workflowEvidenceConsistent = false;
    Throws("campaign_execution_workflow_evidence_invalid", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), inconsistent);
    });

    auto orphanExperiment = Input();
    orphanExperiment.members[0].experimentId = 999;
    Throws("campaign_execution_orphan_experiment_invalid", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), orphanExperiment);
    });

    auto oversized = Input(kMaximumRecommendationCampaignExecutionMembers + 1);
    Throws("campaign_execution_member_limit_exceeded", [&] {
        BuildRecommendationCampaignExecutionPlan(Request(), oversized);
    });

    assert(RecommendationCampaignExecutionPlanStateText(
               RecommendationCampaignExecutionPlanState::ready) == "ready");
    assert(RecommendationCampaignExecutionMemberActionText(
               RecommendationCampaignExecutionMemberAction::alreadySatisfied) ==
           "already_satisfied");

    std::cout << "Experiment recommendation campaign execution tests passed\n";
    return 0;
}
