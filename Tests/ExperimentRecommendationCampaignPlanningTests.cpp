#include "../Sources/ExperimentRecommendationCampaignPlanning.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <locale>
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
    return policy;
}

RecommendationCampaignPlanningScope Scope()
{
    RecommendationCampaignPlanningScope scope;
    scope.rankingSnapshotId = 7;
    return scope;
}

RecommendationCampaignCandidateInput Candidate(
    long long recommendationId,
    int rank,
    double score = 0.8)
{
    RecommendationCampaignCandidateInput value;
    value.rankingSnapshotId = 7;
    value.rankingSnapshotIdentityCanonical = "snapshot-canonical";
    value.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        value.rankingSnapshotIdentityCanonical);
    value.rankingPolicyCanonical = "ranking-policy";
    value.rankingPolicyHash = RecommendationCanonicalHash(
        value.rankingPolicyCanonical);
    value.rankingVersion = 1;
    value.rankingMemberId = 1000 + recommendationId;
    value.rankingPosition = rank;
    value.rankingBucket = "advisory_ready";
    value.rankingScore = score;
    value.recommendationId = recommendationId;
    value.sourceExperimentId = 100 + recommendationId;
    value.symbol = recommendationId % 2 == 0 ? "eurusd" : "gbpusd";
    value.predictionHorizon = recommendationId % 2 == 0 ? 12 : 24;
    value.family = "core_lr_mult";
    value.targetEpochs = 120;
    value.leaderScore = 0.75;
    value.inferenceAccuracy = 0.70;
    value.predictedNeutralProportion = 0.30;
    value.recommendationSemanticCanonical =
        "semantic-" + std::to_string(recommendationId);
    value.recommendationSemanticHash = RecommendationCanonicalHash(
        value.recommendationSemanticCanonical);
    value.recommendationInvocationCanonical =
        "invocation-" + std::to_string(recommendationId);
    value.recommendationInvocationHash = RecommendationCanonicalHash(
        value.recommendationInvocationCanonical);
    return value;
}

RecommendationCampaignWorkflowEvidence Workflow(
    long long proposalId,
    RecommendationConversionWorkflowState state)
{
    RecommendationCampaignWorkflowEvidence value;
    value.proposalId = proposalId;
    value.proposalIdentityCanonical =
        "proposal-" + std::to_string(proposalId);
    value.proposalIdentityHash = RecommendationCanonicalHash(
        value.proposalIdentityCanonical);
    value.latestReviewDecisionId = proposalId + 10;
    value.latestReviewDisposition = state ==
            RecommendationConversionWorkflowState::rejected
        ? "rejected" : "approved";
    if (state == RecommendationConversionWorkflowState::executedPaused ||
        state == RecommendationConversionWorkflowState::activatedPending ||
        state == RecommendationConversionWorkflowState::schedulerClaimedOrRunning ||
        state == RecommendationConversionWorkflowState::completed ||
        state == RecommendationConversionWorkflowState::failed ||
        state == RecommendationConversionWorkflowState::cancelled)
    {
        value.executionId = proposalId + 20;
        value.executionIdentityCanonical =
            "execution-" + std::to_string(proposalId);
        value.executionIdentityHash = RecommendationCanonicalHash(
            *value.executionIdentityCanonical);
        value.convertedExperimentId = proposalId + 30;
    }
    if (state == RecommendationConversionWorkflowState::activatedPending ||
        state == RecommendationConversionWorkflowState::schedulerClaimedOrRunning ||
        state == RecommendationConversionWorkflowState::completed ||
        state == RecommendationConversionWorkflowState::failed ||
        state == RecommendationConversionWorkflowState::cancelled)
    {
        value.activationId = proposalId + 40;
        value.activationIdentityCanonical =
            "activation-" + std::to_string(proposalId);
        value.activationIdentityHash = RecommendationCanonicalHash(
            *value.activationIdentityCanonical);
    }
    value.state = state;
    value.integrity = RecommendationConversionWorkflowIntegrity::consistent;
    return value;
}

RecommendationCampaignPlan MakePlan(
    RecommendationCampaignPlanningPolicy policy,
    std::vector<RecommendationCampaignCandidateInput> candidates,
    std::string generatedAt = "display-only-time",
    RecommendationCampaignPlanningScope scope = Scope())
{
    RecommendationCampaignPlanInput input;
    input.policy = std::move(policy);
    input.scope = std::move(scope);
    input.rankingSnapshotIdentityCanonical = "snapshot-canonical";
    input.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        input.rankingSnapshotIdentityCanonical);
    input.generatedAt = std::move(generatedAt);
    input.candidates = std::move(candidates);
    return PlanRecommendationCampaign(input);
}

bool HasReason(
    const RecommendationCampaignPlanCandidate& candidate,
    RecommendationCampaignReason reason)
{
    return std::find(
        candidate.reasons.begin(), candidate.reasons.end(), reason) !=
        candidate.reasons.end();
}

void ExpectWorkflowExclusion(
    RecommendationConversionWorkflowState state,
    RecommendationCampaignReason reason)
{
    auto candidate = Candidate(1, 1);
    candidate.workflows.push_back(Workflow(20, state));
    const auto plan = MakePlan(Policy(), {candidate});
    assert(plan.summary.selectedCount == 0);
    assert(HasReason(plan.candidates[0], reason));
}

} // namespace

int main()
{
    auto disabled = Policy();
    disabled.enabled = false;
    auto plan = MakePlan(disabled, {Candidate(1, 1)});
    assert(HasReason(
        plan.candidates[0], RecommendationCampaignReason::planningDisabled));

    plan = MakePlan(Policy(), {});
    assert(plan.summary.candidateCount == 0);
    assert(plan.summary.selectedCount == 0);

    plan = MakePlan(
        Policy(), {Candidate(3, 3), Candidate(1, 1), Candidate(2, 2)});
    assert(plan.candidates[0].input.recommendationId == 1);
    assert(plan.candidates[1].input.recommendationId == 2);
    assert(plan.candidates[2].input.recommendationId == 3);

    auto tiedA = Candidate(2, 1);
    auto tiedB = Candidate(1, 1);
    plan = MakePlan(Policy(), {tiedA, tiedB});
    assert(plan.candidates[0].input.recommendationId == 1);

    auto duplicateKeyA = Candidate(1, 1);
    auto duplicateKeyB = Candidate(1, 1);
    duplicateKeyA.rankingMemberId = 1002;
    duplicateKeyB.rankingMemberId = 1001;
    plan = MakePlan(Policy(), {duplicateKeyA, duplicateKeyB});
    assert(plan.candidates[0].input.rankingMemberId == 1001);
    const auto reversedDuplicateKeys = MakePlan(
        Policy(), {duplicateKeyB, duplicateKeyA});
    assert(plan.identityHash == reversedDuplicateKeys.identityHash);

    auto policy = Policy();
    policy.maximumSelectedRecommendations = 1;
    plan = MakePlan(policy, {Candidate(1, 1), Candidate(2, 2)});
    assert(HasReason(
        plan.candidates[1], RecommendationCampaignReason::campaignLimitReached));

    policy = Policy();
    policy.maximumCandidatesConsidered = 1;
    plan = MakePlan(policy, {Candidate(1, 1), Candidate(2, 2)});
    assert(HasReason(
        plan.candidates[1], RecommendationCampaignReason::candidateLimitReached));

    policy = Policy();
    policy.maximumPerSymbol = 1;
    auto sameSymbol = Candidate(3, 2);
    sameSymbol.symbol = "gbpusd";
    plan = MakePlan(policy, {Candidate(1, 1), sameSymbol});
    assert(HasReason(
        plan.candidates[1], RecommendationCampaignReason::symbolLimitReached));

    policy = Policy();
    policy.maximumPerHorizon = 1;
    auto sameHorizon = Candidate(3, 2);
    sameHorizon.predictionHorizon = 24;
    plan = MakePlan(policy, {Candidate(1, 1), sameHorizon});
    assert(HasReason(
        plan.candidates[1], RecommendationCampaignReason::horizonLimitReached));

    policy = Policy();
    policy.maximumPerSourceExperiment = 1;
    auto sameSource = Candidate(2, 2);
    sameSource.sourceExperimentId = Candidate(1, 1).sourceExperimentId;
    plan = MakePlan(policy, {Candidate(1, 1), sameSource});
    assert(HasReason(
        plan.candidates[1],
        RecommendationCampaignReason::sourceExperimentConflict));

    policy.maximumPerSourceExperiment = 2;
    auto sameSource3 = Candidate(3, 3);
    sameSource3.sourceExperimentId = Candidate(1, 1).sourceExperimentId;
    plan = MakePlan(policy, {Candidate(1, 1), sameSource, sameSource3});
    assert(HasReason(
        plan.candidates[2],
        RecommendationCampaignReason::sourceExperimentLimitReached));

    auto duplicate = Candidate(2, 2);
    duplicate.sourceExperimentId = 999;
    duplicate.recommendationInvocationCanonical =
        Candidate(1, 1).recommendationInvocationCanonical;
    duplicate.recommendationInvocationHash = RecommendationCanonicalHash(
        duplicate.recommendationInvocationCanonical);
    policy = Policy();
    policy.maximumPerSourceExperiment.reset();
    plan = MakePlan(policy, {Candidate(1, 1), duplicate});
    assert(HasReason(
        plan.candidates[1],
        RecommendationCampaignReason::duplicateConversionIdentity));

    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::pendingReview,
        RecommendationCampaignReason::workflowPendingReview);
    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::approvedNotExecuted,
        RecommendationCampaignReason::workflowApprovedNotExecuted);
    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::executedPaused,
        RecommendationCampaignReason::workflowExecutedPaused);
    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::activatedPending,
        RecommendationCampaignReason::workflowActivatedOrSchedulerOwned);
    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::schedulerClaimedOrRunning,
        RecommendationCampaignReason::workflowActivatedOrSchedulerOwned);
    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::completed,
        RecommendationCampaignReason::workflowCompleted);
    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::failed,
        RecommendationCampaignReason::workflowFailedNotReconsiderable);
    ExpectWorkflowExclusion(
        RecommendationConversionWorkflowState::cancelled,
        RecommendationCampaignReason::workflowCancelledNotReconsiderable);

    auto rejected = Candidate(1, 1);
    rejected.workflows.push_back(Workflow(
        20, RecommendationConversionWorkflowState::rejected));
    plan = MakePlan(Policy(), {rejected});
    assert(HasReason(
        plan.candidates[0], RecommendationCampaignReason::
        workflowRejectedNotReconsiderable));
    policy = Policy();
    policy.reconsiderRejectedWorkflows = true;
    plan = MakePlan(policy, {rejected});
    assert(plan.summary.selectedCount == 1);

    auto failed = Candidate(1, 1);
    failed.workflows.push_back(Workflow(
        20, RecommendationConversionWorkflowState::failed));
    policy = Policy();
    policy.reconsiderFailedWorkflows = true;
    plan = MakePlan(policy, {failed});
    assert(plan.summary.selectedCount == 1);

    auto cancelled = Candidate(1, 1);
    cancelled.workflows.push_back(Workflow(
        20, RecommendationConversionWorkflowState::cancelled));
    policy = Policy();
    policy.reconsiderCancelledWorkflows = true;
    plan = MakePlan(policy, {cancelled});
    assert(plan.summary.selectedCount == 1);

    auto inconsistent = Candidate(1, 1);
    inconsistent.workflows.push_back(Workflow(
        20, RecommendationConversionWorkflowState::inconsistent));
    inconsistent.workflows[0].integrity =
        RecommendationConversionWorkflowIntegrity::inconsistent;
    plan = MakePlan(Policy(), {inconsistent});
    assert(HasReason(
        plan.candidates[0], RecommendationCampaignReason::workflowInconsistent));

    auto metric = Candidate(1, 1);
    policy = Policy();
    policy.minimumLeaderScore = 0.8;
    plan = MakePlan(policy, {metric});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::belowMinimumLeaderScore));
    policy = Policy();
    policy.minimumInferenceAccuracy = 0.8;
    plan = MakePlan(policy, {metric});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::belowMinimumInferenceAccuracy));
    policy = Policy();
    policy.maximumPredictedNeutralProportion = 0.2;
    plan = MakePlan(policy, {metric});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::predictedNeutralAboveMaximum));
    policy = Policy();
    metric.predictedNeutralProportion.reset();
    plan = MakePlan(policy, {metric});
    assert(HasReason(
        plan.candidates[0], RecommendationCampaignReason::
        predictedNeutralMetricUnavailable));
    metric.predictedNeutralProportion = 0.30;
    policy = Policy();
    policy.minimumProfitability = 0.1;
    plan = MakePlan(policy, {metric});
    assert(HasReason(
        plan.candidates[0], RecommendationCampaignReason::
        profitabilityMetricUnavailable));
    metric.profitabilityMetric = 0.05;
    metric.profitabilityMetricIdentity = "future-metric-v1";
    plan = MakePlan(policy, {metric});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::belowMinimumProfitability));

    const auto canonical = RecommendationCampaignPlanningPolicyCanonicalText(
        Policy());
    assert(canonical.find("experiment_recommendation_campaign_planning_policy_v2") == 0);
    assert(canonical.find(";donchian20_arms=preserve") != std::string::npos);
    assert(RecommendationCampaignPlanningPolicyHash(Policy()) ==
           RecommendationCanonicalHash(canonical));
    const std::string legacyCanonical =
        canonical.substr(0, canonical.find("_v2")) + "_v1" +
        canonical.substr(canonical.find(";contract_version="));
    const std::size_t legacyArmField = legacyCanonical.find(
        ";donchian20_arms=preserve");
    assert(legacyArmField != std::string::npos);
    const auto legacyPolicy =
        ParseRecommendationCampaignPlanningPolicyCanonicalText(
            legacyCanonical.substr(0, legacyArmField));
    assert(legacyPolicy.canonicalVersion == 1);
    assert(legacyPolicy.donchian20Arms.empty());
    assert(RecommendationCampaignPlanningPolicyCanonicalText(legacyPolicy) ==
           legacyCanonical.substr(0, legacyArmField));

    assert(ParseRecommendationCampaignDonchian20Arms("enabled") ==
           std::vector<Donchian20Mode>{Donchian20Mode::Enabled});
    const std::vector<Donchian20Mode> expectedPairedArms{
        Donchian20Mode::Enabled, Donchian20Mode::ZeroAblation};
    assert((ParseRecommendationCampaignDonchian20Arms(
                "zero_ablation:enabled") == expectedPairedArms));
    bool invalidArms = false;
    try { (void)ParseRecommendationCampaignDonchian20Arms("enabled:enabled"); }
    catch (const std::invalid_argument&) { invalidArms = true; }
    assert(invalidArms);
    invalidArms = false;
    try { (void)ParseRecommendationCampaignDonchian20Arms(""); }
    catch (const std::invalid_argument&) { invalidArms = true; }
    assert(invalidArms);
    policy = Policy();
    policy.donchian20Arms = {
        Donchian20Mode::Enabled, Donchian20Mode::ZeroAblation};
    const auto paired = MakePlan(policy, {Candidate(1, 1)});
    assert(paired.summary.candidateCount == 2);
    assert(paired.summary.selectedCount == 2);
    assert(paired.candidates[0].input.campaignDonchian20Mode ==
           Donchian20Mode::Enabled);
    assert(paired.candidates[1].input.campaignDonchian20Mode ==
           Donchian20Mode::ZeroAblation);
    assert(paired.candidates[0].input.recommendationInvocationCanonical ==
           paired.candidates[1].input.recommendationInvocationCanonical);
    assert(paired.identityHash != MakePlan(Policy(), {Candidate(1, 1)}).identityHash);

    const auto first = MakePlan(Policy(), {Candidate(2, 2), Candidate(1, 1)});
    const auto second = MakePlan(Policy(), {Candidate(1, 1), Candidate(2, 2)});
    assert(first.identityCanonical == second.identityCanonical);
    assert(first.identityHash == second.identityHash);
    assert(first.identityHash == MakePlan(
        Policy(), {Candidate(1, 1), Candidate(2, 2)},
        "different-display-time").identityHash);
    auto filteredScope = Scope();
    filteredScope.symbol = "eurusd";
    assert(first.identityHash != MakePlan(
        Policy(), {Candidate(1, 1), Candidate(2, 2)},
        "display-only-time", filteredScope).identityHash);
    auto unsafeScope = Scope();
    unsafeScope.symbol = " EURUSD ";
    assert(ValidateRecommendationCampaignPlanningScope(unsafeScope));
    policy = Policy();
    policy.maximumSelectedRecommendations = 1;
    assert(MakePlan(policy, {Candidate(1, 1), Candidate(2, 2)}).identityHash !=
           first.identityHash);

    auto changedWorkflow = Candidate(1, 1);
    changedWorkflow.workflows.push_back(Workflow(
        20, RecommendationConversionWorkflowState::pendingReview));
    assert(MakePlan(Policy(), {changedWorkflow}).identityHash !=
           MakePlan(Policy(), {Candidate(1, 1)}).identityHash);

    auto badIdentity = Candidate(1, 1);
    badIdentity.recommendationInvocationHash = "bad";
    plan = MakePlan(Policy(), {badIdentity});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::identityValidationFailed));

    auto wrongSnapshot = Candidate(1, 1);
    wrongSnapshot.rankingSnapshotIdentityCanonical = "different-snapshot";
    wrongSnapshot.rankingSnapshotIdentityHash = RecommendationCanonicalHash(
        wrongSnapshot.rankingSnapshotIdentityCanonical);
    plan = MakePlan(Policy(), {wrongSnapshot});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::identityValidationFailed));

    auto badProvenance = Candidate(1, 1);
    badProvenance.persistedProvenanceValid = false;
    plan = MakePlan(Policy(), {badProvenance});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::identityValidationFailed));

    auto inconsistentPolicy = Candidate(2, 2);
    inconsistentPolicy.rankingPolicyCanonical = "different-ranking-policy";
    inconsistentPolicy.rankingPolicyHash = RecommendationCanonicalHash(
        inconsistentPolicy.rankingPolicyCanonical);
    plan = MakePlan(Policy(), {Candidate(1, 1), inconsistentPolicy});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::identityValidationFailed));
    assert(HasReason(
        plan.candidates[1],
        RecommendationCampaignReason::identityValidationFailed));

    auto changedRankingPolicy = Candidate(1, 1);
    changedRankingPolicy.rankingPolicyCanonical = "changed-ranking-policy";
    changedRankingPolicy.rankingPolicyHash = RecommendationCanonicalHash(
        changedRankingPolicy.rankingPolicyCanonical);
    assert(MakePlan(Policy(), {changedRankingPolicy}).identityHash !=
           MakePlan(Policy(), {Candidate(1, 1)}).identityHash);

    auto unsupportedRanking = Candidate(1, 1);
    unsupportedRanking.rankingVersion = 2;
    plan = MakePlan(Policy(), {unsupportedRanking});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::unsupportedContractVersion));

    auto notReady = Candidate(1, 1);
    notReady.rankingBucket = "blocked";
    notReady.rankingScore.reset();
    plan = MakePlan(Policy(), {notReady});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::rankingNotAdvisoryReady));

    auto invalidNumeric = Candidate(1, 1);
    invalidNumeric.leaderScore = std::numeric_limits<double>::infinity();
    plan = MakePlan(Policy(), {invalidNumeric});
    assert(HasReason(
        plan.candidates[0],
        RecommendationCampaignReason::identityValidationFailed));

    std::vector<RecommendationCampaignCandidateInput> oversized;
    oversized.reserve(kMaximumRecommendationCampaignCandidates + 1);
    for (int i = 0; i <= kMaximumRecommendationCampaignCandidates; ++i)
        oversized.push_back(Candidate(i + 1, i + 1));
    bool rejectedOversized = false;
    try
    {
        (void)MakePlan(Policy(), std::move(oversized));
    }
    catch (const std::invalid_argument& error)
    {
        rejectedOversized = std::string(error.what()) ==
            "recommendation_campaign_input_limit_exceeded";
    }
    assert(rejectedOversized);

    for (const auto& candidate : first.candidates)
        assert(candidate.decision == RecommendationCampaignDecision::include ||
               candidate.decision == RecommendationCampaignDecision::exclude);

    const std::locale before = std::locale();
    std::locale::global(std::locale::classic());
    const std::string localeCanonical =
        RecommendationCampaignPlanningPolicyCanonicalText(Policy());
    std::locale::global(before);
    assert(localeCanonical == canonical);
}
