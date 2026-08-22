#include "../Sources/CheckpointPolicy.hpp"

#include <cassert>
#include <iostream>

using namespace EA::ExperimentScheduler;

namespace
{

CheckpointPolicyConfig BasePolicy()
{
    CheckpointPolicyConfig config;
    config.enabled = true;
    config.minLeaderScore = 0.7;
    config.minInferAccuracy = 0.6;
    config.topN = 2;
    config.scope = "symbol_horizon";
    config.stopMode = "next_checkpoint";
    config.graceEvals = 1;
    config.checkpointInterval = 20;
    config.targetEpochs = 100;
    config.currentEpoch = 20;
    return config;
}

CheckpointPolicyEvidenceIdentity BaseEvidence()
{
    CheckpointPolicyEvidenceIdentity evidence;
    evidence.checkpointEvalId = 11;
    evidence.parentExperimentId = 22;
    evidence.checkpointModelId = 33;
    evidence.checkpointEpoch = 20;
    evidence.observedCurrentEpoch = 20;
    evidence.analysisId = 44;
    evidence.inferenceEvalResultId = 55;
    evidence.symbol = "EURUSD";
    evidence.predictionHorizon = 12;
    evidence.inferenceFromDate = "2025-01-01";
    evidence.inferenceToDate = "2025-03-01";
    evidence.leaderScore = 0.65;
    evidence.inferenceAccuracy = 0.55;
    evidence.rankValue = 3;
    evidence.rankScope = "symbol_horizon";
    evidence.rankPopulationWatermark = "rank-population";
    evidence.completedEvalCount = 3;
    evidence.completedPopulationWatermark = "completed-population";
    return evidence;
}

void AssertDecision(
    const CheckpointPolicyDecision& decision,
    const std::string& expectedDecision,
    const std::string& expectedReason)
{
    assert(decision.decision == expectedDecision);
    assert(decision.reason == expectedReason);
}

}  // namespace

int main()
{
    const CheckpointPolicyConfig base = BasePolicy();
    const std::string canonical = CheckpointPolicyCanonicalText(base);
    assert(canonical ==
           "enabled=true|min_leader_score=0.69999999999999996|"
           "min_infer_accuracy=0.59999999999999998|top_n=2|"
           "rank_scope=symbol_horizon|stop_mode=next_checkpoint|"
           "grace_evals=1|checkpoint_interval=20|target_epochs=100");
    assert(CheckpointPolicySemanticHash(base) ==
           CheckpointPolicySemanticHash(base));

    CheckpointPolicyConfig changed = base;
    changed.minLeaderScore = 0.71;
    assert(CheckpointPolicySemanticHash(changed) !=
           CheckpointPolicySemanticHash(base));
    changed = base;
    changed.enabled = false;
    assert(CheckpointPolicySemanticHash(changed) !=
           CheckpointPolicySemanticHash(base));
    changed = base;
    changed.currentEpoch = 60;
    changed.status = "completed";
    changed.phase = "done";
    changed.policyRevision = 99;
    changed.persistedPolicyHash = "observational";
    changed.activeTrainingAttemptId = 991;
    assert(CheckpointPolicySemanticHash(changed) ==
           CheckpointPolicySemanticHash(base));

    AssertDecision(
        DecideCheckpointPolicyPure(
            base, {20, 0, 0.1, 0.1, 99}),
        "continue_grace",
        "completed_checkpoint_evals=0;grace_evals=1");

    // The three legacy continue rules remain OR-based.
    AssertDecision(
        DecideCheckpointPolicyPure(
            base, {20, 2, 0.8, 0.1, 99}),
        "continue",
        "passed_rules=leader_score");
    AssertDecision(
        DecideCheckpointPolicyPure(
            base, {20, 2, 0.1, 0.8, 99}),
        "continue",
        "passed_rules=infer_accuracy");
    AssertDecision(
        DecideCheckpointPolicyPure(
            base, {20, 2, 0.1, 0.1, 2}),
        "continue",
        "passed_rules=top_n");

    const CheckpointPolicyDecision stopped = DecideCheckpointPolicyPure(
        base, {20, 2, 0.1, 0.1, 3});
    AssertDecision(
        stopped,
        "stop_requested",
        "failed_rules=leader_score|infer_accuracy|top_n");
    assert(stopped.requestedStopEpoch == 40);

    changed = base;
    changed.stopMode = "current_checkpoint_if_possible";
    assert(DecideCheckpointPolicyPure(
               changed, {20, 2, 0.1, 0.1, 3})
               .requestedStopEpoch == 20);
    changed.stopMode = "mark_pruned_when_not_running";
    assert(DecideCheckpointPolicyPure(
               changed, {20, 2, 0.1, 0.1, 3})
               .requestedStopEpoch == 40);

    changed = base;
    changed.minLeaderScore.reset();
    changed.minInferAccuracy.reset();
    changed.topN.reset();
    AssertDecision(
        DecideCheckpointPolicyPure(changed, {20, 2, {}, {}, {}}),
        "skipped",
        "no_policy_rules_configured");

    CheckpointPolicyEvidenceIdentity evidence = BaseEvidence();
    const std::string watermark =
        CheckpointPolicyEvidenceWatermark(evidence);
    assert(watermark == CheckpointPolicyEvidenceWatermark(evidence));
    evidence.analysisId += 1;
    assert(watermark != CheckpointPolicyEvidenceWatermark(evidence));
    evidence = BaseEvidence();
    evidence.inferenceEvalResultId += 1;
    assert(watermark != CheckpointPolicyEvidenceWatermark(evidence));
    evidence = BaseEvidence();
    evidence.rankPopulationWatermark = "changed-population";
    assert(watermark != CheckpointPolicyEvidenceWatermark(evidence));
    evidence = BaseEvidence();
    evidence.observedCurrentEpoch = 21;
    assert(watermark != CheckpointPolicyEvidenceWatermark(evidence));

    std::cout << "CheckpointPolicyHardeningTests passed\n";
    return 0;
}
