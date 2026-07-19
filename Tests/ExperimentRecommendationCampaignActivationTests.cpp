#include "../Sources/ExperimentRecommendationCampaignActivation.hpp"

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

RecommendationCampaignActivationRequest Request(bool dryRun = false)
{
    return {19, dryRun};
}

RecommendationCampaignActivationInput Input(int count = 2)
{
    RecommendationCampaignActivationInput input;
    input.materializationId = 19;
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
        RecommendationCampaignActivationMemberInput member;
        member.memberOrdinal = ordinal;
        member.materializationMemberId = 100 + ordinal;
        member.recommendationId = 150 + ordinal;
        member.proposalId = 200 + ordinal;
        member.authorizationReviewDecisionId = 300 + ordinal;
        member.executionId = 400 + ordinal;
        member.experimentId = 500 + ordinal;
        member.evidenceState =
            RecommendationCampaignActivationEvidenceState::eligible;
        member.experimentStatus =
            kRecommendationConversionActivationPreviousStatus;
        member.experimentPhase =
            kRecommendationConversionActivationPreviousPhase;
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
    const auto ready = BuildRecommendationCampaignActivationPlan(
        Request(), Input());
    assert(ready.state == RecommendationCampaignActivationPlanState::ready);
    assert(ready.members.size() == 2);
    assert(ready.members[0].memberOrdinal == 1);
    assert(ready.members[1].memberOrdinal == 2);
    assert(ready.membersValidated == 2);
    assert(ready.alreadyActivatedCount == 0);
    assert(ready.operationIdentityHash == RecommendationCanonicalHash(
        ready.operationIdentityCanonical));
    assert(ready.members[0].postActivationStatus == "pending");
    assert(ready.members[0].postActivationPhase == "train");

    const auto dry = BuildRecommendationCampaignActivationPlan(
        Request(true), Input());
    assert(dry.request.dryRun);
    assert(dry.operationIdentityCanonical == ready.operationIdentityCanonical);
    assert(dry.operationIdentityHash == ready.operationIdentityHash);

    auto satisfiedInput = Input();
    for (std::size_t index = 0; index < satisfiedInput.members.size(); ++index)
    {
        auto& member = satisfiedInput.members[index];
        member.evidenceState =
            RecommendationCampaignActivationEvidenceState::existingIdentical;
        member.activationId = 600 + index;
        member.experimentStatus = "pending";
    }
    const auto satisfied = BuildRecommendationCampaignActivationPlan(
        Request(), satisfiedInput);
    assert(satisfied.state ==
           RecommendationCampaignActivationPlanState::alreadySatisfied);
    assert(satisfied.alreadyActivatedCount == 2);

    auto partial = satisfiedInput;
    partial.members[1].evidenceState =
        RecommendationCampaignActivationEvidenceState::eligible;
    partial.members[1].activationId.reset();
    partial.members[1].experimentStatus = "paused";
    Throws("campaign_activation_partial_operation_conflict", [&] {
        (void)BuildRecommendationCampaignActivationPlan(Request(), partial);
    });

    auto missingExecution = Input();
    missingExecution.members[0].executionId.reset();
    Throws("campaign_activation_member_identity_invalid", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), missingExecution);
    });

    auto invalidExperiment = Input();
    invalidExperiment.members[0].experimentId.reset();
    Throws("campaign_activation_member_identity_invalid", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), invalidExperiment);
    });

    auto invalidEvidence = Input();
    invalidEvidence.members[0].evidenceState =
        RecommendationCampaignActivationEvidenceState::invalid;
    invalidEvidence.members[0].evidenceDiagnostic =
        "campaign_activation_identity_conflict";
    Throws("campaign_activation_identity_conflict", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), invalidEvidence);
    });

    auto existingWrongState = satisfiedInput;
    existingWrongState.members[0].experimentStatus = "running";
    Throws("campaign_activation_existing_state_invalid", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), existingWrongState);
    });

    auto invalidRequest = Request();
    invalidRequest.materializationId = 0;
    Throws("campaign_activation_materialization_id_invalid", [&] {
        (void)NormalizeRecommendationCampaignActivationRequest(invalidRequest);
    });

    auto zero = Input(0);
    Throws("campaign_activation_zero_members", [&] {
        (void)BuildRecommendationCampaignActivationPlan(Request(), zero);
    });

    auto countMismatch = Input();
    countMismatch.selectedMemberCount = 3;
    Throws("campaign_activation_member_count_mismatch", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), countMismatch);
    });

    auto badVersion = Input();
    badVersion.materializationContractVersion = 99;
    Throws("campaign_activation_materialization_identity_invalid", [&] {
        (void)BuildRecommendationCampaignActivationPlan(Request(), badVersion);
    });

    auto duplicateMember = Input();
    duplicateMember.members[1].materializationMemberId =
        duplicateMember.members[0].materializationMemberId;
    Throws("campaign_activation_member_identity_invalid", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), duplicateMember);
    });

    auto duplicateExecution = Input();
    duplicateExecution.members[1].executionId =
        duplicateExecution.members[0].executionId;
    Throws("campaign_activation_member_identity_invalid", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), duplicateExecution);
    });

    auto gap = Input();
    gap.members[1].memberOrdinal = 3;
    Throws("campaign_activation_member_identity_invalid", [&] {
        (void)BuildRecommendationCampaignActivationPlan(Request(), gap);
    });

    auto oversized = Input(kMaximumRecommendationCampaignActivationMembers + 1);
    Throws("campaign_activation_member_limit_exceeded", [&] {
        (void)BuildRecommendationCampaignActivationPlan(
            Request(), oversized);
    });

    std::cout << "Experiment recommendation campaign activation tests passed\n";
    return 0;
}
