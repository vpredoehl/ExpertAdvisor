#include "../Sources/ExperimentRecommendationCampaignFollowUpProposalReview.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
#include <locale>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace EA::ExperimentRecommendation;

namespace
{

using Decision = RecommendationCampaignFollowUpProposalReviewDecision;
using Review = RecommendationCampaignFollowUpProposalReview;
using ReviewIdentity = RecommendationCampaignFollowUpProposalReviewIdentity;

static_assert(!std::is_default_constructible_v<Review>);
static_assert(std::is_copy_constructible_v<Review>);
static_assert(std::is_move_constructible_v<Review>);
static_assert(!std::is_copy_assignable_v<Review>);
static_assert(!std::is_move_assignable_v<Review>);
static_assert(!std::is_copy_assignable_v<ReviewIdentity>);
static_assert(!std::is_move_assignable_v<ReviewIdentity>);
static_assert(Review::readOnly);
static_assert(Review::databaseFree);
static_assert(!Review::persistent);
static_assert(Review::administrativeReview);
static_assert(!Review::activated);
static_assert(!Review::executionAuthorized);
static_assert(!Review::followUpAuthorized);
static_assert(!Review::queued);
static_assert(!Review::scheduled);
static_assert(!Review::schedulerStarted);
static_assert(!Review::schedulerSignaled);
static_assert(!Review::workersStarted);
static_assert(!Review::experimentsCreated);
static_assert(!Review::experimentsModified);
static_assert(!Review::campaignSuccessDeclared);

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

Review Build(
    Decision decision = Decision::approved,
    long long proposalId = 41,
    std::string proposalCanonical = "phase6a-proposal",
    std::string reviewer = "operator@example.test",
    std::string reason = "Evidence reviewed manually.")
{
    const std::string proposalHash =
        RecommendationCanonicalHash(proposalCanonical);
    return BuildRecommendationCampaignFollowUpProposalReview(
        kRecommendationCampaignFollowUpProposalReviewContractVersion,
        proposalId,
        kRecommendationCampaignFollowUpProposalContractVersion,
        proposalCanonical, proposalHash, decision, std::move(reviewer),
        std::move(reason));
}

} // namespace

int main()
{
    const Review approved = Build();
    const Review rejected = Build(Decision::rejected);
    assert(approved.decision == Decision::approved);
    assert(rejected.decision == Decision::rejected);
    assert(approved.followUpProposalId == 41);
    assert(approved.proposalContractVersion ==
        kRecommendationCampaignFollowUpProposalContractVersion);
    assert(approved.identity.contractVersion ==
        kRecommendationCampaignFollowUpProposalReviewContractVersion);
    assert(approved.identity.hash ==
        RecommendationCanonicalHash(approved.identity.canonicalText));
    ValidateRecommendationCampaignFollowUpProposalReview(approved);
    ValidateRecommendationCampaignFollowUpProposalReview(rejected);

    const std::string expectedCanonical =
        "experiment_recommendation_campaign_follow_up_proposal_review_v1"
        ";review_contract_version=1"
        ";read_only=true"
        ";database_free=true"
        ";persistent=false"
        ";administrative_review=true"
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
        ";follow_up_proposal_id=41"
        ";proposal_contract_version=1"
        ";proposal_canonical=16:phase6a-proposal"
        ";proposal_identity_hash=24:" +
        approved.proposalIdentityHash +
        ";decision=approved"
        ";reviewer=21:operator@example.test"
        ";reason=27:Evidence reviewed manually.";
    assert(approved.identity.canonicalText == expectedCanonical);
    assert(Build() == approved);

    const std::locale originalLocale = std::locale();
    std::locale::global(
        std::locale(originalLocale, new GroupedNumberPunctuation));
    const Review localized = Build();
    std::locale::global(originalLocale);
    assert(localized == approved);

    assert(Build(Decision::approved, 42).identity != approved.identity);
    assert(Build(Decision::approved, 41, "phase6a-proposal-2").identity !=
        approved.identity);
    assert(Build(Decision::approved, 41, "phase6a-proposal",
               "second-reviewer")
            .identity != approved.identity);
    assert(Build(Decision::approved, 41, "phase6a-proposal",
               "operator@example.test", "A different reason.")
            .identity != approved.identity);
    assert(rejected.identity != approved.identity);

    assert(RecommendationCampaignFollowUpProposalReviewDecisionText(
               Decision::approved) == "approved");
    assert(RecommendationCampaignFollowUpProposalReviewDecisionText(
               Decision::rejected) == "rejected");
    assert(RecommendationCampaignFollowUpProposalReviewDecisionFromText(
               "approved") == Decision::approved);
    assert(RecommendationCampaignFollowUpProposalReviewDecisionFromText(
               "rejected") == Decision::rejected);

    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalReview(2, 41, 1,
            approved.proposalCanonicalText, approved.proposalIdentityHash,
            Decision::approved, "reviewer", "reason");
    }, "review_contract_unsupported");
    AssertReason([&]
    {
        (void)Build(Decision::approved, 0);
    }, "persisted_proposal_id_invalid");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalReview(1, 41, 2,
            approved.proposalCanonicalText, approved.proposalIdentityHash,
            Decision::approved, "reviewer", "reason");
    }, "proposal_contract_unsupported");
    AssertReason([&]
    {
        (void)BuildRecommendationCampaignFollowUpProposalReview(1, 41, 1,
            approved.proposalCanonicalText, "fnv1a64:0000000000000000",
            Decision::approved, "reviewer", "reason");
    }, "proposal_identity_invalid");
    AssertReason([&]
    {
        const std::string nulCanonical("proposal\0identity", 17);
        (void)BuildRecommendationCampaignFollowUpProposalReview(1, 41, 1,
            nulCanonical, RecommendationCanonicalHash(nulCanonical),
            Decision::approved, "reviewer", "reason");
    }, "proposal_identity_invalid");
    AssertReason([&]
    {
        (void)Build(static_cast<Decision>(99));
    }, "review_decision_invalid");
    AssertReason([&]
    {
        (void)Build(Decision::approved, 41, "phase6a-proposal", "", "reason");
    }, "reviewer_identity_invalid");
    AssertReason([&]
    {
        (void)Build(Decision::approved, 41, "phase6a-proposal", "bad identity",
            "reason");
    }, "reviewer_identity_invalid");
    AssertReason([&]
    {
        (void)Build(Decision::approved, 41, "phase6a-proposal", "reviewer",
            "   \t\n");
    }, "review_reason_invalid");
    AssertReason([&]
    {
        (void)Build(Decision::approved, 41, "phase6a-proposal", "reviewer",
            std::string("bad\0reason", 10));
    }, "review_reason_invalid");
    AssertReason([&]
    {
        (void)Build(Decision::approved, 41, "phase6a-proposal", "reviewer",
            std::string("\xc3\x28", 2));
    }, "review_reason_invalid");
    AssertReason([&]
    {
        (void)RecommendationCampaignFollowUpProposalReviewDecisionFromText(
            "pending");
    }, "review_decision_invalid");

    // No Phase 6C canonical can encode positive action authority.
    for (const std::string forbidden : {
             "activated=true", "execution_authorized=true",
             "follow_up_authorized=true", "queued=true", "scheduled=true",
             "scheduler_started=true", "scheduler_signaled=true",
             "workers_started=true", "experiments_created=true",
             "experiments_modified=true",
             "campaign_success_declared=true"})
        assert(approved.identity.canonicalText.find(forbidden) ==
            std::string::npos);

    return 0;
}
