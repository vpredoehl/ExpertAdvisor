#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
#include <locale>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace EA::ExperimentRecommendation;

namespace
{

using Ratification = RecommendationCampaignFollowUpProposalRatification;
using RatificationDecision =
    RecommendationCampaignFollowUpProposalRatificationDecision;
using RatificationIdentity =
    RecommendationCampaignFollowUpProposalRatificationIdentity;
using Review = RecommendationCampaignFollowUpProposalReview;
using ReviewDecision =
    RecommendationCampaignFollowUpProposalReviewDecision;

static_assert(!std::is_default_constructible_v<Ratification>);
static_assert(std::is_copy_constructible_v<Ratification>);
static_assert(std::is_move_constructible_v<Ratification>);
static_assert(!std::is_copy_assignable_v<Ratification>);
static_assert(!std::is_move_assignable_v<Ratification>);
static_assert(!std::is_copy_assignable_v<RatificationIdentity>);
static_assert(!std::is_move_assignable_v<RatificationIdentity>);
static_assert(Ratification::readOnly);
static_assert(Ratification::databaseFree);
static_assert(!Ratification::persistent);
static_assert(Ratification::governanceRatification);
static_assert(
    Ratification::ratifiesAdvancementToNextSeparatelyControlledPhase);
static_assert(Ratification::eligibleReviewRequired);
static_assert(Ratification::separationOfDutiesRequired);
static_assert(!Ratification::phase6ECapabilityGranted);
static_assert(!Ratification::activated);
static_assert(!Ratification::executionAuthorized);
static_assert(!Ratification::followUpAuthorized);
static_assert(!Ratification::queued);
static_assert(!Ratification::scheduled);
static_assert(!Ratification::schedulerStarted);
static_assert(!Ratification::schedulerSignaled);
static_assert(!Ratification::workersStarted);
static_assert(!Ratification::experimentsCreated);
static_assert(!Ratification::experimentsModified);
static_assert(!Ratification::campaignSuccessDeclared);

class GroupedNumberPunctuation final : public std::numpunct<char>
{
protected:
    char do_thousands_sep() const override { return ','; }
    std::string do_grouping() const override { return "\1"; }
};

template <typename Function>
void AssertReason(Function&& function, const std::string& expected)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::exception& error)
    {
        threw = true;
        assert(error.what() == expected);
    }
    assert(threw);
}

Review BuildReview(
    ReviewDecision decision = ReviewDecision::approved,
    long long proposalId = 41,
    std::string reviewer = "reviewer@example.test",
    std::string basis = "Review evidence accepted.")
{
    const std::string proposalCanonical = "phase6a-proposal";
    return BuildRecommendationCampaignFollowUpProposalReview(
        kRecommendationCampaignFollowUpProposalReviewContractVersion,
        proposalId,
        kRecommendationCampaignFollowUpProposalContractVersion,
        proposalCanonical, RecommendationCanonicalHash(proposalCanonical),
        decision, std::move(reviewer), std::move(basis));
}

Ratification Build(
    long long reviewEventId = 73,
    std::string ratifier = "ratifier@example.test",
    std::string basis = "Ratified after independent review.")
{
    return BuildRecommendationCampaignFollowUpProposalRatification(reviewEventId,
        BuildReview(), std::move(ratifier), std::move(basis));
}

} // namespace

int main()
{
    const Review review = BuildReview();
    const Ratification ratification = Build();
    assert(ratification.reviewEventId == 73);
    assert(ratification.reviewContractVersion ==
        kRecommendationCampaignFollowUpProposalReviewContractVersion);
    assert(ratification.reviewCanonicalText == review.identity.canonicalText);
    assert(ratification.reviewIdentityHash == review.identity.hash);
    assert(ratification.followUpProposalId == review.followUpProposalId);
    assert(ratification.proposalCanonicalText == review.proposalCanonicalText);
    assert(ratification.proposalIdentityHash == review.proposalIdentityHash);
    assert(ratification.reviewDecision == ReviewDecision::approved);
    assert(ratification.reviewerIdentity == review.reviewerIdentity);
    assert(ratification.ratificationAuthorityRole ==
        kRecommendationCampaignFollowUpProposalRatificationAuthorityRole);
    assert(ratification.decision == RatificationDecision::ratified);
    assert(ratification.identity.contractVersion ==
        kRecommendationCampaignFollowUpProposalRatificationContractVersion);
    assert(ratification.identity.hash == "fnv1a64:caf47db05ce89408");
    assert(ratification.identity.hash ==
        RecommendationCanonicalHash(ratification.identity.canonicalText));
    ValidateRecommendationCampaignFollowUpProposalRatification(ratification);
    assert(Build() == ratification);

    const std::string expectedCanonical =
        "experiment_recommendation_campaign_follow_up_proposal_ratification_v1"
        ";ratification_contract_version=1"
        ";read_only=true"
        ";database_free=true"
        ";persistent=false"
        ";governance_ratification=true"
        ";ratifies_advancement_to_next_separately_controlled_phase=true"
        ";eligible_review_required=true"
        ";separation_of_duties_required=true"
        ";phase_6e_capability_granted=false"
        ";activated=false"
        ";execution_authorized=false"
        ";follow_up_authorized=false"
        ";queued=false"
        ";scheduled=false"
        ";scheduler_started=false"
        ";scheduler_signaled=false"
        ";workers_started=false"
        ";experiments_created=false"
        ";experiments_modified=false"
        ";campaign_success_declared=false"
        ";review_event_id=73"
        ";review_contract_version=1"
        ";review_canonical=" +
        std::to_string(review.identity.canonicalText.size()) + ":" +
        review.identity.canonicalText +
        ";review_identity_hash=24:" + review.identity.hash +
        ";review_decision=approved"
        ";reviewer=21:reviewer@example.test"
        ";follow_up_proposal_id=41"
        ";proposal_contract_version=1"
        ";proposal_canonical=16:phase6a-proposal"
        ";proposal_identity_hash=24:" + review.proposalIdentityHash +
        ";ratification_authority_role=29:follow_up_governance_ratifier"
        ";ratification_decision=ratified"
        ";ratifier=21:ratifier@example.test"
        ";basis=34:Ratified after independent review.";
    assert(ratification.identity.canonicalText == expectedCanonical);

    const std::locale originalLocale = std::locale();
    std::locale::global(
        std::locale(originalLocale, new GroupedNumberPunctuation));
    const Ratification localized = Build();
    std::locale::global(originalLocale);
    assert(localized == ratification);

    assert(Build(74).identity != ratification.identity);
    assert(Build(73, "other.ratifier").identity != ratification.identity);
    assert(Build(73, "ratifier@example.test", "Different basis.").identity !=
        ratification.identity);
    const std::string unicodeReason = "Ratified with UTF-8 \xe2\x9c\x93";
    assert(Build(73, "ratifier@example.test", unicodeReason).ratificationBasis ==
        unicodeReason);
    assert(Build(73,
               std::string(
                   kMaximumRecommendationCampaignFollowUpProposalRatifierIdentityBytes,
                   'a'),
               std::string(
                   kMaximumRecommendationCampaignFollowUpProposalRatificationBasisBytes,
                   'x'))
            .ratificationBasis.size() ==
        kMaximumRecommendationCampaignFollowUpProposalRatificationBasisBytes);
    const Review differentReview = BuildReview(
        ReviewDecision::approved, 42);
    assert(BuildRecommendationCampaignFollowUpProposalRatification(73,
               differentReview, "ratifier@example.test",
               "Ratified after independent review.")
            .identity != ratification.identity);

    assert(RecommendationCampaignFollowUpProposalRatificationDecisionText(
               RatificationDecision::ratified) == "ratified");
    assert(RecommendationCampaignFollowUpProposalRatificationDecisionFromText(
               "ratified") == RatificationDecision::ratified);

    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(73,
            BuildReview(ReviewDecision::approved, 41, "same.actor"),
            "same.actor", "Independent governance basis.");
    }, "ratification_separation_of_duties_violation");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, review.identity.hash, 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            "caller_injected_role", RatificationDecision::ratified,
            "ratifier", "basis");
    }, "ratification_authority_role_invalid");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, review.identity.hash, 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, "forged.reviewer",
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified, "ratifier", "basis");
    }, "review_identity_invalid");

    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(2, 73, 1,
            review.identity.canonicalText, review.identity.hash, 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "ratification_contract_unsupported");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(0, review,
            "ratifier", "basis");
    }, "persisted_review_event_id_invalid");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 2,
            review.identity.canonicalText, review.identity.hash, 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "review_contract_unsupported");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, "fnv1a64:0000000000000000", 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "review_identity_invalid");
    AssertReason([&]
    {
        const std::string badCanonical("review\0canonical", 16);
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            badCanonical, RecommendationCanonicalHash(badCanonical), 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "review_identity_invalid");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(73,
            BuildReview(ReviewDecision::rejected), "ratifier", "basis");
    }, "review_not_eligible_for_ratification");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, review.identity.hash, 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            static_cast<ReviewDecision>(99), review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "review_decision_invalid");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, review.identity.hash, 0, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "persisted_proposal_id_invalid");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, review.identity.hash, 41, 2,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "proposal_contract_unsupported");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, review.identity.hash, 41, 1,
            review.proposalCanonicalText, "fnv1a64:0000000000000000",
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            RatificationDecision::ratified,
            "ratifier", "basis");
    }, "proposal_identity_invalid");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalRatification(1, 73, 1,
            review.identity.canonicalText, review.identity.hash, 41, 1,
            review.proposalCanonicalText, review.proposalIdentityHash,
            ReviewDecision::approved, review.reviewerIdentity,
            kRecommendationCampaignFollowUpProposalRatificationAuthorityRole,
            static_cast<RatificationDecision>(99),
            "ratifier", "basis");
    }, "ratification_decision_invalid");
    AssertReason([&]
    {
        (void)Build(73, "", "basis");
    }, "ratifier_identity_invalid");
    AssertReason([&]
    {
        (void)Build(73, "bad identity", "basis");
    }, "ratifier_identity_invalid");
    AssertReason([&]
    {
        (void)Build(73,
            std::string(
                kMaximumRecommendationCampaignFollowUpProposalRatifierIdentityBytes +
                    1U,
                'a'),
            "basis");
    }, "ratifier_identity_invalid");
    AssertReason([&]
    {
        (void)Build(73, "ratifier", " \t\n\r");
    }, "ratification_basis_invalid");
    AssertReason([&]
    {
        (void)Build(73, "ratifier", std::string("bad\0basis", 10));
    }, "ratification_basis_invalid");
    AssertReason([&]
    {
        (void)Build(73, "ratifier", std::string("bad\x01" "basis", 9));
    }, "ratification_basis_invalid");
    AssertReason([&]
    {
        (void)Build(73, "ratifier", std::string("bad\x7f" "basis", 9));
    }, "ratification_basis_invalid");
    AssertReason([&]
    {
        (void)Build(73, "ratifier", std::string("\xc3\x28", 2));
    }, "ratification_basis_invalid");
    AssertReason([&]
    {
        (void)Build(73, "ratifier",
            std::string(
                kMaximumRecommendationCampaignFollowUpProposalRatificationBasisBytes +
                    1U,
                'x'));
    }, "ratification_basis_invalid");
    AssertReason([&]
    {
        (void)RecommendationCampaignFollowUpProposalRatificationDecisionFromText(
            "rejected");
    }, "ratification_decision_invalid");

    for (const std::string forbidden : {
             "activated=true", "execution_authorized=true",
             "follow_up_authorized=true", "queued=true", "scheduled=true",
             "scheduler_started=true", "scheduler_signaled=true",
             "workers_started=true", "experiments_created=true",
             "experiments_modified=true",
             "phase_6e_capability_granted=true",
             "campaign_success_declared=true"})
        assert(ratification.identity.canonicalText.find(forbidden) ==
            std::string::npos);

    return 0;
}
