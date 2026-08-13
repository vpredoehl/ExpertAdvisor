#include "../Sources/ExperimentRecommendationCampaignHandoff.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignMaterialization.hpp"

#include <cassert>
#include <stdexcept>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignHandoffProposalFacts Proposal(
    long long id,
    const std::string& review = {},
    bool executed = false,
    bool activated = false)
{
    RecommendationCampaignHandoffProposalFacts value;
    value.proposalId = id;
    value.recommendationId = id;
    value.sourceExperimentId = 100 + id;
    value.identityCanonical = "proposal-" + std::to_string(id);
    value.identityHash = RecommendationCanonicalHash(value.identityCanonical);
    value.workflow.integrity =
        RecommendationConversionWorkflowIntegrity::consistent;
    value.workflow.state = RecommendationConversionWorkflowState::pendingReview;
    if (!review.empty())
    {
        value.latestReview = RecommendationConversionWorkflowReviewFact{
            1000 + id, id, review};
        value.workflow.state = review == "approve"
            ? RecommendationConversionWorkflowState::approvedNotExecuted
            : RecommendationConversionWorkflowState::rejected;
    }
    if (executed)
    {
        value.executionId = 2000 + id;
        value.workflow.state = RecommendationConversionWorkflowState::executedPaused;
    }
    if (activated)
    {
        value.activationId = 3000 + id;
        value.workflow.state = RecommendationConversionWorkflowState::activatedPending;
    }
    return value;
}

RecommendationCampaignHandoffMemberInput Member(
    int ordinal,
    const std::optional<RecommendationCampaignHandoffProposalFacts>& proposal)
{
    RecommendationCampaignHandoffMemberInput value;
    value.memberOrdinal = ordinal;
    value.rankingMemberId = 100 + ordinal;
    value.recommendationId = ordinal;
    value.sourceExperimentId = 100 + ordinal;
    value.rankingPosition = ordinal;
    value.selectedMemberIdentityCanonical =
        "member-" + std::to_string(ordinal);
    value.selectedMemberIdentityHash = RecommendationCanonicalHash(
        value.selectedMemberIdentityCanonical);
    value.conversionProposalId = ordinal;
    value.proposalIdentityCanonical = "proposal-" + std::to_string(ordinal);
    value.proposalIdentityHash = RecommendationCanonicalHash(
        value.proposalIdentityCanonical);
    value.proposal = proposal;
    return value;
}

RecommendationCampaignHandoffInput Input(
    std::optional<RecommendationCampaignHandoffProposalFacts> left,
    std::optional<RecommendationCampaignHandoffProposalFacts> right)
{
    RecommendationCampaignHandoffInput input;
    input.materializationId = 7;
    input.campaignApprovalId = 8;
    input.materializationContractVersion =
        kRecommendationCampaignMaterializationContractVersion;
    input.materializationIdentityCanonical = "materialization";
    input.materializationIdentityHash = RecommendationCanonicalHash(
        input.materializationIdentityCanonical);
    input.selectedMemberCount = 2;
    input.members = {Member(1, left), Member(2, right)};
    return input;
}

void ExpectInvalid(const RecommendationCampaignHandoffInput& input)
{
    try
    {
        (void)BuildRecommendationCampaignHandoff(input);
        assert(false);
    }
    catch (const std::invalid_argument&) {}
}

} // namespace

int main()
{
    auto result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1), Proposal(2)));
    assert(result.state ==
           RecommendationCampaignHandoffState::awaitingPhase4cReview);
    assert(result.members[0].memberOrdinal == 1);
    assert(result.members[1].memberOrdinal == 2);

    result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1, "approve"), Proposal(2)));
    assert(result.state == RecommendationCampaignHandoffState::partiallyReviewed);

    result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1, "approve"), Proposal(2, "reject")));
    assert(result.state == RecommendationCampaignHandoffState::reviewRejected);

    result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1, "approve"), Proposal(2, "approve")));
    assert(result.state ==
           RecommendationCampaignHandoffState::readyForPhase4cExecution);

    result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1, "approve", true), Proposal(2, "approve")));
    assert(result.state == RecommendationCampaignHandoffState::partiallyExecuted);

    result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1, "approve", true),
              Proposal(2, "approve", true)));
    assert(result.state ==
           RecommendationCampaignHandoffState::readyForPhase4cActivation);

    result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1, "approve", true, true),
              Proposal(2, "approve", true)));
    assert(result.state == RecommendationCampaignHandoffState::partiallyActivated);

    result = BuildRecommendationCampaignHandoff(
        Input(Proposal(1, "approve", true, true),
              Proposal(2, "approve", true, true)));
    assert(result.state == RecommendationCampaignHandoffState::fullyActivated);

    auto inconsistent = Proposal(1, "approve", true);
    inconsistent.workflow.integrity =
        RecommendationConversionWorkflowIntegrity::inconsistent;
    inconsistent.workflow.diagnosticCodes = {"execution_review_invalid"};
    result = BuildRecommendationCampaignHandoff(
        Input(inconsistent, Proposal(2, "approve")));
    assert(result.state == RecommendationCampaignHandoffState::inconsistent);
    assert(result.summary.integrityFailures == 1);

    result = BuildRecommendationCampaignHandoff(
        Input(std::nullopt, Proposal(2)));
    assert(result.state == RecommendationCampaignHandoffState::inconsistent);
    assert(result.members[0].diagnosticCodes.front() ==
           "linked_proposal_missing");
    assert(result.summary.proposalsPresent == 1);
    assert(result.summary.awaitingReview == 2);
    assert(result.summary.noAuthoritativeReview == 2);

    auto rejectedInconsistent = Proposal(1, "reject");
    rejectedInconsistent.workflow.integrity =
        RecommendationConversionWorkflowIntegrity::inconsistent;
    rejectedInconsistent.workflow.diagnosticCodes = {
        "execution_cardinality_invalid"};
    result = BuildRecommendationCampaignHandoff(
        Input(rejectedInconsistent, Proposal(2, "approve")));
    assert(result.state == RecommendationCampaignHandoffState::inconsistent);

    auto invalidReview = Proposal(1, "approve");
    invalidReview.latestReview->reviewDecisionId = 0;
    invalidReview.workflow.integrity =
        RecommendationConversionWorkflowIntegrity::inconsistent;
    invalidReview.workflow.diagnosticCodes = {"latest_review_invalid"};
    result = BuildRecommendationCampaignHandoff(
        Input(invalidReview, Proposal(2, "approve")));
    assert(result.state == RecommendationCampaignHandoffState::inconsistent);
    assert(result.members[0].reviewStatus ==
           RecommendationCampaignHandoffReviewStatus::awaitingReview);
    assert(result.members[0].reviewDecisionId == 0);
    assert(result.summary.approved == 1);
    assert(result.summary.awaitingReview == 1);
    assert(result.summary.noAuthoritativeReview == 1);
    assert((result.members[0].diagnosticCodes ==
            std::vector<std::string>{"latest_review_invalid"}));

    auto deterministic = Proposal(1);
    deterministic.identityCanonical = "wrong";
    deterministic.identityHash = RecommendationCanonicalHash("wrong");
    deterministic.workflow.integrity =
        RecommendationConversionWorkflowIntegrity::inconsistent;
    deterministic.workflow.diagnosticCodes = {
        "proposal_identity_invalid", "proposal_identity_invalid",
        "execution_cardinality_invalid"};
    result = BuildRecommendationCampaignHandoff(
        Input(deterministic, Proposal(2)));
    assert((result.members[0].diagnosticCodes == std::vector<std::string>{
        "proposal_identity_canonical_mismatch",
        "proposal_identity_hash_mismatch",
        "proposal_identity_invalid",
        "execution_cardinality_invalid"}));

    auto mismatch = Proposal(1);
    mismatch.identityCanonical = "wrong";
    mismatch.identityHash = RecommendationCanonicalHash("wrong");
    result = BuildRecommendationCampaignHandoff(
        Input(mismatch, Proposal(2)));
    assert(result.state == RecommendationCampaignHandoffState::inconsistent);
    assert(result.members[0].diagnosticCodes[0] ==
           "proposal_identity_canonical_mismatch");

    auto invalid = Input(Proposal(1), Proposal(2));
    invalid.selectedMemberCount = 0;
    ExpectInvalid(invalid);
    invalid = Input(Proposal(1), Proposal(2));
    invalid.materializationContractVersion = 999;
    ExpectInvalid(invalid);
    invalid = Input(Proposal(1), Proposal(2));
    invalid.materializationIdentityHash = "invalid";
    ExpectInvalid(invalid);
    invalid = Input(Proposal(1), Proposal(2));
    invalid.selectedMemberCount = 0;
    invalid.members.clear();
    ExpectInvalid(invalid);
    invalid = Input(Proposal(1), Proposal(2));
    std::swap(invalid.members[0], invalid.members[1]);
    ExpectInvalid(invalid);
    invalid = Input(Proposal(1), Proposal(2));
    invalid.members[1].memberOrdinal = 1;
    ExpectInvalid(invalid);
    invalid = Input(Proposal(1), Proposal(2));
    invalid.members[1].rankingMemberId = invalid.members[0].rankingMemberId;
    const auto paired = BuildRecommendationCampaignHandoff(invalid);
    assert(paired.summary.totalMembers == 2);
    invalid = Input(Proposal(1), Proposal(2));
    invalid.members[1].conversionProposalId =
        invalid.members[0].conversionProposalId;
    ExpectInvalid(invalid);

    assert(RecommendationCampaignHandoffStateText(
               RecommendationCampaignHandoffState::fullyActivated) ==
           "fully_activated");
    return 0;
}
