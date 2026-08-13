#include "../Sources/ExperimentRecommendationCampaignStatus.hpp"
#include "../Sources/ExperimentRecommendation.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationCampaignStatusMemberInput Member(int ordinal)
{
    RecommendationCampaignStatusMemberInput member;
    member.memberOrdinal = ordinal;
    member.materializationMemberId = 100 + ordinal;
    member.recommendationId = 200 + ordinal;
    member.sourceExperimentId = 300 + ordinal;
    member.rankingMemberId = 400 + ordinal;
    member.proposalId = 500 + ordinal;
    member.currentReviewDecisionId = 600 + ordinal;
    member.currentReviewDecision = "approve";
    return member;
}

RecommendationCampaignStatusExperimentEvidence Experiment(
    long long id,
    std::string status,
    std::string phase)
{
    RecommendationCampaignStatusExperimentEvidence experiment;
    experiment.experimentId = id;
    experiment.symbol = "eurusd";
    experiment.predictionHorizon = 12;
    experiment.status = std::move(status);
    experiment.phase = std::move(phase);
    experiment.targetEpochs = 120;
    experiment.invocationMode = "recommendation_conversion";
    experiment.invocationIdentityCanonical = "pure-status-invocation";
    experiment.invocationProvenanceValid = true;
    return experiment;
}

void Execute(RecommendationCampaignStatusMemberInput& member, bool activate)
{
    member.executionId = 700 + member.memberOrdinal;
    member.authorizationReviewDecisionId = 600 + member.memberOrdinal;
    member.experimentId = 800 + member.memberOrdinal;
    member.executionCount = 1;
    if (activate)
    {
        member.activationId = 900 + member.memberOrdinal;
        member.activationCount = 1;
    }
}

RecommendationCampaignStatusInput Input(
    std::vector<RecommendationCampaignStatusMemberInput> members)
{
    RecommendationCampaignStatusInput input;
    input.materializationId = 71;
    input.materializationContractVersion = 1;
    input.materializationIdentityCanonical = "materialization-status-test-v1";
    input.materializationIdentityHash = RecommendationCanonicalHash(
        input.materializationIdentityCanonical);
    input.selectedMemberCount = static_cast<int>(members.size());
    input.observedAt = "2026-07-19 12:00:00+00";
    input.members = std::move(members);
    return input;
}

RecommendationCampaignStatusSnapshot Snapshot(
    std::vector<RecommendationCampaignStatusMemberInput> members)
{
    return BuildRecommendationCampaignStatusSnapshot({71}, Input(std::move(members)));
}

void ExpectInvalid(
    RecommendationCampaignStatusInput input,
    const std::string& expected)
{
    bool threw = false;
    try { (void)BuildRecommendationCampaignStatusSnapshot({71}, std::move(input)); }
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
    assert(NormalizeRecommendationCampaignStatusRequest({71}).materializationId == 71);
    try { (void)NormalizeRecommendationCampaignStatusRequest({0}); assert(false); }
    catch (const std::invalid_argument&) {}

    const auto proposalOnly = Snapshot({Member(1)});
    assert(proposalOnly.aggregateStatus == RecommendationCampaignAggregateStatus::notExecuted);
    assert(proposalOnly.proposalOnlyCount == 1);
    assert(proposalOnly.members[0].operationalState ==
           RecommendationCampaignStatusOperationalState::proposalOnly);

    auto pausedMember = Member(1);
    Execute(pausedMember, false);
    pausedMember.experiment = Experiment(*pausedMember.experimentId, "paused", "train");
    const auto paused = Snapshot({pausedMember});
    assert(paused.aggregateStatus == RecommendationCampaignAggregateStatus::executedPaused);
    assert(paused.pausedCount == 1);

    auto pendingMember = pausedMember;
    pendingMember.activationId = 901;
    pendingMember.activationCount = 1;
    pendingMember.experiment->status = "pending";
    const auto pending = Snapshot({pendingMember});
    assert(pending.aggregateStatus == RecommendationCampaignAggregateStatus::queued);
    assert(pending.pendingCount == 1);
    assert(!pending.totalCurrentEpochs);
    assert(pending.totalTargetEpochs == std::optional<long long>{120});

    for (const auto& phase : {"train", "infer", "analyze"})
    {
        auto running = pendingMember;
        running.experiment->status = "running";
        running.experiment->phase = phase;
        running.experiment->workerStartedAt = "2026-07-19 12:01:00+00";
        running.experiment->workerPid = 1234;
        running.experiment->currentOperation = phase;
        running.experiment->currentEpoch = 20;
        const auto snapshot = Snapshot({running});
        assert(snapshot.runningCount == 1);
        assert(snapshot.activeWorkerCount == 1);
        assert(snapshot.aggregateStatus == RecommendationCampaignAggregateStatus::inProgress);
    }

    auto completed = pendingMember;
    completed.experiment->status = "completed";
    completed.experiment->phase = "done";
    completed.experiment->completedAt = "2026-07-19 12:02:00+00";
    completed.experiment->modelId = 1001;
    completed.experiment->modelLinkCount = 1;
    const auto trainOnlyCompleted = Snapshot({completed});
    assert(trainOnlyCompleted.inconsistentCount == 1);
    assert(trainOnlyCompleted.members[0].trainComplete);
    assert(!trainOnlyCompleted.members[0].inferConfigured);

    auto fullCompleted = completed;
    fullCompleted.experiment->inferStartPresent = true;
    fullCompleted.experiment->inferEndPresent = true;
    fullCompleted.experiment->inferConfigured = true;
    fullCompleted.experiment->inferenceResultCount = 1;
    fullCompleted.experiment->completedInferenceResultCount = 1;
    fullCompleted.experiment->analysisResultCount = 1;
    fullCompleted.experiment->completedAnalysisResultCount = 1;
    const auto pipelineCompleted = Snapshot({fullCompleted});
    assert(pipelineCompleted.members[0].inferComplete);
    assert(pipelineCompleted.members[0].analyzeComplete);

    auto invalidResultProvenance = fullCompleted;
    invalidResultProvenance.experiment->invocationProvenanceValid = false;
    assert(Snapshot({invalidResultProvenance}).inconsistentCount == 1);

    auto missingAnalysis = fullCompleted;
    missingAnalysis.experiment->completedAnalysisResultCount = 0;
    const auto invalidCompletion = Snapshot({missingAnalysis});
    assert(invalidCompletion.inconsistentCount == 1);
    assert(invalidCompletion.aggregateStatus == RecommendationCampaignAggregateStatus::inconsistent);

    auto failedAnalysis = fullCompleted;
    failedAnalysis.experiment->completedAnalysisResultCount = 0;
    failedAnalysis.experiment->failedAnalysisResultCount = 1;
    const auto failedAnalysisSnapshot = Snapshot({failedAnalysis});
    assert(failedAnalysisSnapshot.inconsistentCount == 1);
    assert(failedAnalysisSnapshot.snapshotIdentityHash !=
           pipelineCompleted.snapshotIdentityHash);

    auto unknownInference = fullCompleted;
    unknownInference.experiment->completedInferenceResultCount = 0;
    unknownInference.experiment->failedInferenceResultCount = 0;
    assert(Snapshot({unknownInference}).inconsistentCount == 1);

    auto failed = pendingMember;
    failed.experiment->status = "failed";
    failed.experiment->phase = "train";
    failed.experiment->completedAt = "2026-07-19 12:03:00+00";
    failed.experiment->errorMessage = "worker_failed";
    assert(Snapshot({failed}).failedCount == 1);

    auto cancelled = pendingMember;
    cancelled.experiment->status = "cancelled";
    cancelled.experiment->completedAt = "2026-07-19 12:04:00+00";
    assert(Snapshot({cancelled}).cancelledCount == 1);

    auto progressedPause = pendingMember;
    progressedPause.experiment->status = "paused";
    progressedPause.experiment->currentEpoch = 15;
    assert(Snapshot({progressedPause}).members[0].operationalState ==
           RecommendationCampaignStatusOperationalState::pausedAfterProgress);

    auto activationWithoutExecution = Member(1);
    activationWithoutExecution.activationId = 901;
    activationWithoutExecution.activationCount = 1;
    assert(Snapshot({activationWithoutExecution}).inconsistentCount == 1);

    auto executionWithoutExperiment = Member(1);
    Execute(executionWithoutExperiment, false);
    assert(Snapshot({executionWithoutExperiment}).inconsistentCount == 1);

    auto mismatchedExperiment = pendingMember;
    mismatchedExperiment.experiment->experimentId += 1;
    assert(Snapshot({mismatchedExperiment}).inconsistentCount == 1);

    auto invalidWorker = pendingMember;
    invalidWorker.experiment->status = "running";
    invalidWorker.experiment->phase = "train";
    const auto invalidWorkerSnapshot = Snapshot({invalidWorker});
    assert(invalidWorkerSnapshot.inconsistentCount == 1);
    assert(invalidWorkerSnapshot.activeWorkerCount == 0);

    auto duplicateExecution = pendingMember;
    duplicateExecution.executionCount = 2;
    const auto duplicateIdentity = Snapshot({duplicateExecution}).snapshotIdentityHash;
    duplicateExecution.executionCount = 3;
    assert(duplicateIdentity != Snapshot({duplicateExecution}).snapshotIdentityHash);

    auto first = pendingMember;
    auto second = failed;
    second.memberOrdinal = 2;
    second.materializationMemberId = 102;
    second.recommendationId = 202;
    second.sourceExperimentId = 302;
    second.rankingMemberId = 402;
    second.proposalId = 502;
    second.currentReviewDecisionId = 602;
    second.authorizationReviewDecisionId = 602;
    second.executionId = 702;
    second.activationId = 902;
    second.experimentId = 802;
    second.experiment->experimentId = 802;
    const auto mixed = Snapshot({first, second});
    assert(mixed.pendingCount == 1 && mixed.failedCount == 1);
    assert(mixed.aggregateStatus == RecommendationCampaignAggregateStatus::queued);
    const auto ordered = Snapshot({second, first});
    assert(ordered.members[0].memberOrdinal == 1);
    assert(ordered.members[1].memberOrdinal == 2);
    const auto reordered = Snapshot({first, second});
    assert(ordered.snapshotIdentityHash == reordered.snapshotIdentityHash);
    second.experiment->status = "cancelled";
    second.experiment->errorMessage.reset();
    assert(ordered.snapshotIdentityHash != Snapshot({first, second}).snapshotIdentityHash);
    auto changedFailureEvidence = failed;
    const auto failedIdentity = Snapshot({changedFailureEvidence}).snapshotIdentityHash;
    changedFailureEvidence.experiment->errorMessage = "different_failure";
    assert(failedIdentity != Snapshot({changedFailureEvidence}).snapshotIdentityHash);
    auto changedResultMultiplicity = fullCompleted;
    const auto resultIdentity = Snapshot({changedResultMultiplicity}).snapshotIdentityHash;
    changedResultMultiplicity.experiment->inferenceResultCount = 2;
    changedResultMultiplicity.experiment->failedInferenceResultCount = 1;
    assert(resultIdentity != Snapshot({changedResultMultiplicity}).snapshotIdentityHash);

    auto empty = Input({});
    empty.selectedMemberCount = 0;
    ExpectInvalid(std::move(empty), "campaign_status_zero_members");
    auto gap = Input({Member(2)});
    ExpectInvalid(std::move(gap), "campaign_status_materialization_members_invalid");
    auto mismatch = Input({Member(1)});
    mismatch.selectedMemberCount = 2;
    ExpectInvalid(std::move(mismatch), "campaign_status_member_count_mismatch");

    assert(!proposalOnly.minimumCurrentEpoch);
    assert(!proposalOnly.totalCurrentEpochs);
    assert(proposalOnly.completedPercent == 0.0);
    return 0;
}
