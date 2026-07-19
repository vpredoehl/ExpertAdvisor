#include "../Sources/ExperimentRecommendationCampaignReview.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
#include <stdexcept>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignPlanningPolicy Policy()
{
    RecommendationCampaignPlanningPolicy policy;
    policy.enabled = true;
    policy.maximumSelectedRecommendations = 10;
    policy.maximumCandidatesConsidered = 100;
    policy.maximumPerSourceExperiment.reset();
    return policy;
}

RecommendationCampaignCandidateInput Candidate(
    long long id,
    int position,
    const std::string& symbol,
    int horizon,
    const std::string& family)
{
    RecommendationCampaignCandidateInput candidate;
    candidate.rankingSnapshotId = 7;
    candidate.rankingSnapshotIdentityCanonical = "review-snapshot";
    candidate.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        candidate.rankingSnapshotIdentityCanonical);
    candidate.rankingPolicyCanonical = "review-ranking-policy";
    candidate.rankingPolicyHash = RecommendationCanonicalHash(
        candidate.rankingPolicyCanonical);
    candidate.rankingVersion = 1;
    candidate.rankingMemberId = 100 + id;
    candidate.rankingPosition = position;
    candidate.rankingBucket = "advisory_ready";
    candidate.rankingScore = 0.8;
    candidate.recommendationId = id;
    candidate.sourceExperimentId = 500 + id;
    candidate.symbol = symbol;
    candidate.predictionHorizon = horizon;
    candidate.family = family;
    candidate.targetEpochs = 120;
    candidate.leaderScore = 0.75;
    candidate.inferenceAccuracy = 0.70;
    candidate.predictedNeutralProportion = 0.30;
    candidate.recommendationSemanticCanonical =
        "review-semantic-" + std::to_string(id);
    candidate.recommendationSemanticHash = RecommendationCanonicalHash(
        candidate.recommendationSemanticCanonical);
    candidate.recommendationInvocationCanonical =
        "review-invocation-" + std::to_string(id);
    candidate.recommendationInvocationHash = RecommendationCanonicalHash(
        candidate.recommendationInvocationCanonical);
    return candidate;
}

RecommendationCampaignPlan Plan(
    std::vector<RecommendationCampaignCandidateInput> candidates,
    const std::string& generatedAt = "display-time")
{
    RecommendationCampaignPlanInput input;
    input.policy = Policy();
    input.scope.rankingSnapshotId = 7;
    input.rankingSnapshotIdentityCanonical = "review-snapshot";
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = generatedAt;
    input.candidates = std::move(candidates);
    return PlanRecommendationCampaign(input);
}

} // namespace

int main()
{
    auto first = Candidate(1, 1, "eurusd", 12, "core_lr_mult");
    auto second = Candidate(2, 2, "gbpusd", 24, "batch_size");
    second.rankingBucket = "blocked";
    second.rankingScore.reset();
    auto third = Candidate(3, 3, "eurusd", 12, "core_lr_mult");
    third.recommendationInvocationCanonical =
        first.recommendationInvocationCanonical;
    third.recommendationInvocationHash = RecommendationCanonicalHash(
        third.recommendationInvocationCanonical);

    const auto plan = Plan({third, second, first});
    assert(RecommendationCampaignPlanOrderingIsDeterministic(plan));
    const auto review = ReviewRecommendationCampaignPlan(plan);
    assert(review.summary.candidateCount == 3);
    assert(review.summary.selectedCount == 1);
    assert(review.summary.excludedCount == 2);
    assert(review.summary.deterministicOrderingVerified);
    assert(review.selected.size() == 1);
    assert(review.selected[0].recommendationId == 1);
    assert(review.excluded.size() == 2);
    assert(review.excluded[0].recommendationId == 2);
    assert(review.excluded[0].reasons.size() == 1);
    assert(review.excluded[0].reasons[0] ==
           RecommendationCampaignReason::rankingNotAdvisoryReady);
    assert(review.excluded[1].recommendationId == 3);
    assert(review.excluded[1].reasons[0] ==
           RecommendationCampaignReason::duplicateConversionIdentity);

    assert(review.summary.duplicateGroupCount == 1);
    assert(review.summary.duplicateCandidateCount == 2);
    assert(review.duplicateGroups.size() == 1);
    assert(review.duplicateGroups[0].recommendationIds ==
           std::vector<long long>({1, 3}));
    assert(review.selected[0].duplicate);
    assert(review.excluded[1].duplicate);
    assert(review.selected[0].duplicateIdentityHash ==
           review.excluded[1].duplicateIdentityHash);

    assert(review.summary.consideredFamilyCount == 2);
    assert(review.summary.selectedFamilyCount == 1);
    assert(review.summary.consideredSymbolCount == 2);
    assert(review.summary.selectedSymbolCount == 1);
    assert(review.summary.consideredHorizonCount == 2);
    assert(review.summary.selectedHorizonCount == 1);
    assert(review.familyCoverage.size() == 2);
    assert(review.familyCoverage[0].value == "batch_size");
    assert(review.familyCoverage[1].value == "core_lr_mult");
    assert(review.familyCoverage[1].consideredCount == 2);
    assert(review.familyCoverage[1].selectedCount == 1);
    assert(review.symbolCoverage[0].value == "eurusd");
    assert(review.horizonCoverage[0].horizon == 12);

    assert(review.identityHash == RecommendationCanonicalHash(
        review.identityCanonical));
    const auto reordered = ReviewRecommendationCampaignPlan(
        Plan({first, third, second}, "different-display-time"));
    assert(review.identityCanonical == reordered.identityCanonical);
    assert(review.identityHash == reordered.identityHash);

    auto changed = first;
    changed.family = "batch_size";
    const auto changedReview = ReviewRecommendationCampaignPlan(Plan({changed}));
    assert(changedReview.identityHash !=
           ReviewRecommendationCampaignPlan(Plan({first})).identityHash);

    const auto empty = ReviewRecommendationCampaignPlan(Plan({}));
    assert(empty.summary.candidateCount == 0);
    assert(empty.selected.empty());
    assert(empty.excluded.empty());
    assert(empty.familyCoverage.empty());

    auto malformedOrder = plan;
    std::swap(malformedOrder.candidates[0], malformedOrder.candidates[1]);
    assert(!RecommendationCampaignPlanOrderingIsDeterministic(malformedOrder));
    try
    {
        (void)ReviewRecommendationCampaignPlan(malformedOrder);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_review_plan_order_invalid");
    }

    auto malformedIdentity = plan;
    malformedIdentity.identityHash = "not-the-plan-hash";
    try
    {
        (void)ReviewRecommendationCampaignPlan(malformedIdentity);
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()) ==
               "recommendation_campaign_review_plan_identity_invalid");
    }

    return 0;
}
