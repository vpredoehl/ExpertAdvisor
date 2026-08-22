#include <cassert>
#include <climits>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "../Sources/ContinuationPolicyInheritance.hpp"

using namespace EA::ExperimentScheduler;

namespace
{

template <typename Function>
std::string ExceptionMessage(Function&& function)
{
    try
    {
        function();
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    assert(false && "expected an exception");
    return {};
}

ContinuationPolicyConfig BoundedPolicy()
{
    ContinuationPolicyConfig config;
    config.sourceExperimentId = 184;
    config.source.targetEpochs = 130;
    config.enabled = true;
    config.targetEpochs = 134;
    config.minEvals = 1;
    config.patience = 2;
    config.minInferAccuracy = 0.0;
    config.scope = "symbol_horizon";
    config.trendMode = "none";
    config.sourceMode = "final_model";
    config.includeExcluded = true;
    config.candidateExcluded = true;
    config.inheritToChild = true;
    config.targetIncrement = 4;
    config.maxTargetEpochs = 142;
    return config;
}

ContinuationPolicyConfig SequencePolicy(int currentTarget, int policyTarget)
{
    ContinuationPolicyConfig config = BoundedPolicy();
    config.sourceExperimentId = 300;
    config.source.targetEpochs = currentTarget;
    config.targetEpochs = policyTarget;
    config.progressionMode = "target_sequence";
    config.targetIncrement.reset();
    config.maxTargetEpochs.reset();
    config.targetSequence = std::vector<int>{140, 160, 180, 200, 220, 240};
    return config;
}

PersistedContinuationIdentity SatisfiedContinuation(
    const ContinuationPolicyConfig& config)
{
    PersistedContinuationIdentity persisted;
    persisted.decisionId = 12;
    persisted.sourceExperimentId = config.sourceExperimentId;
    persisted.sourceModelId = 663;
    persisted.sourceAnalysisId = 1001;
    persisted.sourceEpoch = config.source.targetEpochs;
    persisted.targetEpochs = *config.targetEpochs;
    persisted.decision = "continuation_queued";
    persisted.observedEvalCount = 2;
    persisted.patienceWindow = config.patience;
    persisted.inferAccuracy = 0.90;
    persisted.policyRevision = config.policyRevision;
    persisted.policyHash = ContinuationPolicySemanticHash(config);
    persisted.evidenceWatermark = "evidence-v1";
    persisted.queuedExperimentId = 185;
    persisted.queuedChildExists = true;
    persisted.queuedChildStatus = "completed";
    persisted.childParentExperimentId = config.sourceExperimentId;
    persisted.childSourceExperimentId = config.sourceExperimentId;
    persisted.childResumeModelId = persisted.sourceModelId;
    persisted.childSourceModelId = persisted.sourceModelId;
    persisted.childSourceEpoch = persisted.sourceEpoch;
    persisted.childTargetEpochs = persisted.targetEpochs;
    persisted.childGeneration =
        config.continuationSourceExperimentId.has_value()
            ? config.continuationGeneration + 1
            : 1;
    persisted.childPolicyInherited = true;
    persisted.childPolicySourceMode = config.sourceMode;
    persisted.sourceAnalysisScope = "final";
    persisted.sourceAnalysisValid = true;
    persisted.sourceModelOwnedBySource = true;
    return persisted;
}

} // namespace

int main()
{
    ContinuationPolicyConfig enablement = BoundedPolicy();
    for (const std::string& status : {"pending", "paused", "running", "completed"})
    {
        enablement.status = status;
        enablement.phase = status == "completed" ? "done" : "train";
        assert(!ContinuationPolicyEnablementError(enablement));
    }
    enablement.status = "running";
    for (const std::string& phase : {"train", "infer", "analyze"})
    {
        enablement.phase = phase;
        assert(!ContinuationPolicyEnablementError(enablement));
        assert(!ContinuationPolicySourceCompletionReady(enablement));
    }
    enablement.status = "cancelled";
    assert(ContinuationPolicyEnablementError(enablement) ==
           "policy_enablement_disallowed_for_cancelled_source");
    enablement.status = "failed";
    assert(ContinuationPolicyEnablementError(enablement) ==
           "policy_enablement_disallowed_for_failed_source");
    enablement.status = "unknown";
    assert(ContinuationPolicyEnablementError(enablement) ==
           "policy_enablement_disallowed_for_source_status");
    enablement = BoundedPolicy();
    enablement.status = "running";
    enablement.phase = "train";
    enablement.minInferAccuracy.reset();
    assert(ContinuationPolicyEnablementError(enablement) ==
           "at_least_one_threshold_ranking_or_trend_criterion_required");
    assert(!ContinuationPolicySourceCompletionReady(enablement));
    enablement.status = "completed";
    enablement.phase = "analyze";
    assert(!ContinuationPolicySourceCompletionReady(enablement));
    enablement.phase = "done";
    assert(ContinuationPolicySourceCompletionReady(enablement));

    const ContinuationPolicyUpdate update = ParseContinuationPolicyUpdate(
        "target_epochs=134,min_evals=1,patience=2,min_leader_score=0.25,"
        "min_infer_accuracy=0.5,min_improvement=0.01,max_degradation=0.02,"
        "top_n=3,scope=global,trend_mode=improving,source_mode=final_model,"
        "include_excluded=true,candidate_excluded=1,inherit_to_child=true,"
        "target_increment=4,max_target_epochs=142");
    ContinuationPolicyConfig parsed;
    ApplyContinuationPolicyUpdate(parsed, update);
    assert(parsed.targetEpochs == 134);
    assert(parsed.minEvals == 1 && parsed.patience == 2);
    assert(parsed.minLeaderScore == 0.25 && parsed.minInferAccuracy == 0.5);
    assert(parsed.minImprovement == 0.01 && parsed.maxDegradation == 0.02);
    assert(parsed.topN == 3 && parsed.scope == "global");
    assert(parsed.trendMode == "improving" && parsed.sourceMode == "final_model");
    assert(parsed.includeExcluded && parsed.candidateExcluded && parsed.inheritToChild);
    assert(parsed.targetIncrement == 4 && parsed.maxTargetEpochs == 142);

    ApplyContinuationPolicyUpdate(
        parsed,
        ParseContinuationPolicyUpdate(
            "target_epochs=null,min_leader_score=null,min_infer_accuracy=null,"
            "min_improvement=null,max_degradation=null,top_n=null,"
            "target_increment=null,max_target_epochs=null"));
    assert(!parsed.targetEpochs && !parsed.minLeaderScore && !parsed.minInferAccuracy);
    assert(!parsed.minImprovement && !parsed.maxDegradation && !parsed.topN);
    assert(!parsed.targetIncrement && !parsed.maxTargetEpochs);

    ContinuationPolicyUpdate sequenceUpdate = ParseContinuationPolicyUpdate(
        "progression_mode=target_sequence,target_sequence=140:160:180:200:220:240");
    assert(sequenceUpdate.progressionMode == "target_sequence");
    assert(sequenceUpdate.targetSequence ==
           std::optional<std::vector<int>>({140, 160, 180, 200, 220, 240}));
    ApplyContinuationPolicyUpdate(parsed, sequenceUpdate);
    assert(parsed.progressionMode == "target_sequence");
    assert(ContinuationTargetSequenceText(parsed.targetSequence) ==
           "140:160:180:200:220:240");
    ApplyContinuationPolicyUpdate(
        parsed,
        ParseContinuationPolicyUpdate(
            "progression_mode=null,target_sequence=null"));
    assert(!parsed.progressionMode && !parsed.targetSequence);

    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate(""); }) ==
           "--set-continuation-policy requires at least one key=value pair");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("min_evals=0"); }) ==
           "invalid min_evals value '0'");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("include_excluded=yes"); }) ==
           "invalid include_excluded value 'yes'; expected true or false");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("unknown=1"); }) ==
           "unsupported continuation policy key 'unknown'");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_epochs"); }) ==
           "--set-continuation-policy requires key=value pairs");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence="); }) ==
           "target_sequence_required");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence=140:x"); }) ==
           "invalid target_sequence value '140:x'");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence=140:0"); }) ==
           "target_sequence_contains_nonpositive_value");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence=140:-1"); }) ==
           "target_sequence_contains_nonpositive_value");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence=140:140"); }) ==
           "target_sequence_contains_duplicate");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence=160:140"); }) ==
           "target_sequence_must_be_strictly_increasing");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence=140:"); }) ==
           "invalid target_sequence value '140:'");
    assert(ExceptionMessage([] { ParseContinuationPolicyUpdate("target_sequence=140::160"); }) ==
           "invalid target_sequence value '140::160'");

    ContinuationPolicyConfig invalid = BoundedPolicy();
    invalid.minEvals = 0;
    invalid.patience = 0;
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "min_evals_must_be_positive");
    invalid = BoundedPolicy();
    invalid.scope = "invalid";
    assert(ContinuationPolicyConfigurationError(invalid, true) == "invalid_scope");
    invalid = BoundedPolicy();
    invalid.minInferAccuracy.reset();
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "at_least_one_threshold_ranking_or_trend_criterion_required");
    invalid = BoundedPolicy();
    invalid.includeExcluded = false;
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "excluded_candidate_requires_include_excluded");
    invalid = BoundedPolicy();
    invalid.maxTargetEpochs = 133;
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "policy_target_exceeds_max_target");

    ContinuationPolicyConfig sequence = SequencePolicy(120, 140);
    assert(!ContinuationPolicyConfigurationError(sequence, true));
    invalid = sequence;
    invalid.progressionMode = "adaptive";
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "invalid_progression_mode");
    invalid = sequence;
    invalid.targetSequence.reset();
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "target_sequence_required");
    invalid = sequence;
    invalid.targetIncrement = 20;
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "target_increment_disallowed_for_target_sequence");
    invalid = sequence;
    invalid.maxTargetEpochs = 220;
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "max_target_conflicts_with_sequence");
    invalid = sequence;
    invalid.targetEpochs = 150;
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "policy_target_not_in_sequence");
    invalid = sequence;
    invalid.targetEpochs = 160;
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "policy_target_is_not_next_sequence_target");
    invalid = SequencePolicy(240, 240);
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "sequence_has_no_target_after_current_epoch");
    invalid = sequence;
    invalid.targetSequence = std::vector<int>{140, 160, 160};
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "target_sequence_contains_duplicate");
    invalid = sequence;
    invalid.targetSequence = std::vector<int>{140, 180, 160};
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "target_sequence_must_be_strictly_increasing");
    invalid = BoundedPolicy();
    invalid.progressionMode = "fixed_increment";
    invalid.targetIncrement.reset();
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "fixed_increment_requires_target_increment");
    invalid = BoundedPolicy();
    invalid.targetSequence = std::vector<int>{134, 138, 142};
    assert(ContinuationPolicyConfigurationError(invalid, true) ==
           "fixed_increment_disallows_target_sequence");

    const ContinuationPolicyConfig bounded = BoundedPolicy();
    const std::string boundedCanonical =
        "target_epochs=134|min_evals=1|patience=2|min_leader_score=NULL|"
        "min_infer_accuracy=0|min_improvement=NULL|max_degradation=NULL|top_n=NULL|"
        "scope=symbol_horizon|trend_mode=none|source_mode=final_model|"
        "include_excluded=true|candidate_excluded=true|inherit_to_child=true|"
        "target_increment=4|max_target_epochs=142";
    assert(ContinuationPolicySemanticCanonicalText(bounded) == boundedCanonical);
    assert(ContinuationPolicySemanticHash(bounded) == "28815e783b815f0b");

    ContinuationPolicyConfig automatic = bounded;
    automatic.status = "completed";
    automatic.phase = "done";
    automatic.source.lastModelId = 663;
    PersistedContinuationIdentity satisfied = SatisfiedContinuation(automatic);
    auto satisfaction =
        CheckAutomaticContinuationSatisfaction(automatic, satisfied);
    assert(satisfaction.alreadySatisfied);
    assert(satisfaction.reason == "existing_continuation_matches");

    // A revision-only change and a progression-only semantic change do not
    // change the already-persisted child identity.
    automatic.policyRevision += 1;
    satisfaction = CheckAutomaticContinuationSatisfaction(automatic, satisfied);
    assert(satisfaction.alreadySatisfied);
    automatic.maxTargetEpochs = 146;
    satisfaction = CheckAutomaticContinuationSatisfaction(automatic, satisfied);
    assert(satisfaction.alreadySatisfied);
    assert(satisfaction.currentPolicyHash !=
           satisfaction.persistedDecisionPolicyHash);

    ContinuationPolicyConfig requiresEvaluation = automatic;
    requiresEvaluation.minInferAccuracy = 0.95;
    assert(!CheckAutomaticContinuationSatisfaction(
                requiresEvaluation, satisfied).alreadySatisfied);
    requiresEvaluation = automatic;
    requiresEvaluation.topN = 1;
    assert(CheckAutomaticContinuationSatisfaction(
               requiresEvaluation, satisfied).reason ==
           "ranking_requires_full_evaluation");
    requiresEvaluation = automatic;
    requiresEvaluation.minProfitabilityActionableCount = 10;
    assert(CheckAutomaticContinuationSatisfaction(
               requiresEvaluation, satisfied).reason ==
           "profitability_requires_full_evaluation");
    requiresEvaluation = automatic;
    requiresEvaluation.targetEpochs = 138;
    assert(CheckAutomaticContinuationSatisfaction(
               requiresEvaluation, satisfied).reason ==
           "derived_target_changed");

    PersistedContinuationIdentity changedEvidence = satisfied;
    changedEvidence.evidenceChangedAfterDecision = true;
    assert(CheckAutomaticContinuationSatisfaction(
               automatic, changedEvidence).reason ==
           "evidence_changed_after_decision");
    PersistedContinuationIdentity invalidOwnership = satisfied;
    invalidOwnership.sourceModelOwnedBySource = false;
    assert(CheckAutomaticContinuationSatisfaction(
               automatic, invalidOwnership).reason ==
           "persisted_source_identity_invalid");
    PersistedContinuationIdentity invalidAnalysis = satisfied;
    invalidAnalysis.sourceAnalysisValid = false;
    assert(CheckAutomaticContinuationSatisfaction(
               automatic, invalidAnalysis).reason ==
           "persisted_source_identity_invalid");
    PersistedContinuationIdentity missingChild = satisfied;
    missingChild.queuedChildExists = false;
    assert(CheckAutomaticContinuationSatisfaction(
               automatic, missingChild).reason == "queued_child_missing");
    PersistedContinuationIdentity cancelledChild = satisfied;
    cancelledChild.queuedChildStatus = "cancelled";
    assert(CheckAutomaticContinuationSatisfaction(
               automatic, cancelledChild).reason ==
           "queued_child_retryable_or_failed");
    PersistedContinuationIdentity wrongLineage = satisfied;
    wrongLineage.childSourceEpoch = satisfied.sourceEpoch - 1;
    assert(CheckAutomaticContinuationSatisfaction(
               automatic, wrongLineage).reason ==
           "queued_child_lineage_mismatch");
    requiresEvaluation = automatic;
    requiresEvaluation.sourceMode = "best_checkpoint";
    assert(CheckAutomaticContinuationSatisfaction(
               requiresEvaluation, satisfied).reason ==
           "source_selection_changed");
    PersistedContinuationIdentity checkpointSource = satisfied;
    checkpointSource.sourceCheckpointEvalId = 77;
    checkpointSource.sourceAnalysisScope = "checkpoint";
    checkpointSource.childPolicySourceMode = "best_checkpoint";
    checkpointSource.policyHash =
        ContinuationPolicySemanticHash(requiresEvaluation);
    assert(CheckAutomaticContinuationSatisfaction(
               requiresEvaluation, checkpointSource).alreadySatisfied);

    ContinuationPolicyConfig legacy = bounded;
    legacy.maxTargetEpochs.reset();
    assert(ContinuationPolicySemanticHash(legacy) == "e000ebb1a1203736");
    legacy.progressionMode = "fixed_increment";
    assert(ContinuationPolicySemanticHash(legacy) == "e000ebb1a1203736");

    ContinuationPolicyConfig differentProvenance = bounded;
    differentProvenance.policyInherited = true;
    differentProvenance.inheritedFromExperimentId = 999;
    differentProvenance.inheritedFromRevision = 77;
    differentProvenance.inheritedFromHash = "source-hash";
    differentProvenance.inheritanceStatus = "max_target_reached";
    differentProvenance.lastDecision = "queued";
    differentProvenance.lastReason = "runtime";
    differentProvenance.selectedModelId = 645;
    differentProvenance.queuedExperimentId = 185;
    assert(ContinuationPolicySemanticHash(differentProvenance) ==
           ContinuationPolicySemanticHash(bounded));

    ContinuationPolicyConfig behavioralChange = bounded;
    behavioralChange.targetEpochs = 138;
    assert(ContinuationPolicySemanticHash(behavioralChange) !=
           ContinuationPolicySemanticHash(bounded));
    behavioralChange = bounded;
    behavioralChange.targetIncrement = 8;
    assert(ContinuationPolicySemanticHash(behavioralChange) !=
           ContinuationPolicySemanticHash(bounded));
    behavioralChange = bounded;
    behavioralChange.maxTargetEpochs = 146;
    assert(ContinuationPolicySemanticHash(behavioralChange) !=
           ContinuationPolicySemanticHash(bounded));
    behavioralChange = bounded;
    behavioralChange.sourceMode = "best_checkpoint";
    assert(ContinuationPolicySemanticHash(behavioralChange) !=
           ContinuationPolicySemanticHash(bounded));

    const auto derived = DeriveContinuationChildPolicyTarget(130, 4);
    assert(derived.error.empty() && derived.targetEpochs == 134);
    assert(DeriveContinuationChildPolicyTarget(130, std::nullopt).error ==
           "inherit_to_child_requires_target_increment");
    assert(DeriveContinuationChildPolicyTarget(std::nullopt, 4).error ==
           "inherit_to_child_requires_target_epochs");
    assert(DeriveContinuationChildPolicyTarget(INT_MAX - 1, 4).error ==
           "inherited_target_epochs_overflow");

    const auto child134 = DeriveBoundedContinuationChildPolicy(134, 4, 142);
    assert(!child134.terminal && child134.inheritedPolicyTargetEpochs == 138);
    const auto child138 = DeriveBoundedContinuationChildPolicy(138, 4, 142);
    assert(!child138.terminal && child138.inheritedPolicyTargetEpochs == 142);
    const auto child142 = DeriveBoundedContinuationChildPolicy(142, 4, 142);
    assert(child142.terminal && !child142.inheritedPolicyTargetEpochs.has_value());
    assert(DeriveBoundedContinuationChildPolicy(146, 4, 142).error ==
           "policy_target_exceeds_max_target");
    assert(DeriveBoundedContinuationChildPolicy(140, 4, 142).error ==
           "target_increment_would_skip_past_max_target");

    const std::optional<std::vector<int>> campaign =
        std::vector<int>{140, 160, 180, 200, 220, 240};
    const auto sequence140 =
        DeriveSequenceContinuationChildPolicy(120, 140, campaign);
    assert(sequence140.childTrainingTargetEpochs == 140);
    assert(sequence140.inheritedPolicyTargetEpochs == 160);
    assert(!sequence140.terminal && sequence140.error.empty());
    const auto sequence160 =
        DeriveSequenceContinuationChildPolicy(140, 160, campaign);
    assert(sequence160.childTrainingTargetEpochs == 160);
    assert(sequence160.inheritedPolicyTargetEpochs == 180);
    const auto sequence240 =
        DeriveSequenceContinuationChildPolicy(220, 240, campaign);
    assert(sequence240.childTrainingTargetEpochs == 240);
    assert(sequence240.terminal && !sequence240.inheritedPolicyTargetEpochs);
    assert(DeriveSequenceContinuationChildPolicy(240, 240, campaign).error ==
           "sequence_has_no_target_after_current_epoch");

    const ContinuationChildPolicyPlan plan = PlanContinuationChildPolicy(bounded);
    assert(plan.inherit && !plan.terminal && plan.targetEpochs == 138);
    assert(plan.sourcePolicyHash == "28815e783b815f0b");
    assert(plan.policyHash == "d8ba2ada6edd5727");
    assert(plan.inheritedPolicy.source.targetEpochs == 134);
    assert(plan.inheritedPolicy.targetEpochs == 138);
    assert(plan.inheritedPolicy.policyRevision == 1);
    assert(plan.inheritedPolicy.enabled && plan.inheritedPolicy.inheritToChild);
    assert(!plan.inheritedPolicy.lastDecision && !plan.inheritedPolicy.lastReason);
    assert(!plan.inheritedPolicy.selectedModelId && !plan.inheritedPolicy.queuedExperimentId);

    ContinuationPolicyConfig terminalSource = bounded;
    terminalSource.source.targetEpochs = 138;
    terminalSource.targetEpochs = 142;
    terminalSource.lastDecision = "queued";
    terminalSource.queuedExperimentId = 999;
    const ContinuationChildPolicyPlan terminal =
        PlanContinuationChildPolicy(terminalSource);
    assert(terminal.inherit && terminal.terminal && terminal.targetEpochs == 0);
    assert(!terminal.inheritedPolicy.enabled && !terminal.inheritedPolicy.inheritToChild);
    assert(!terminal.inheritedPolicy.targetEpochs);
    assert(terminal.inheritedPolicy.inheritanceStatus == "max_target_reached");
    assert(!terminal.inheritedPolicy.lastDecision && !terminal.inheritedPolicy.queuedExperimentId);

    ContinuationPolicyConfig disabled = bounded;
    disabled.inheritToChild = false;
    const ContinuationChildPolicyPlan noInheritance =
        PlanContinuationChildPolicy(disabled);
    assert(!noInheritance.inherit && !noInheritance.terminal);

    const ContinuationChildPolicyPlan sequencePlan =
        PlanContinuationChildPolicy(sequence);
    assert(sequencePlan.inherit && !sequencePlan.terminal);
    assert(sequencePlan.progressionMode == "target_sequence");
    assert(sequencePlan.childTrainingTargetEpochs == 140);
    assert(sequencePlan.inheritedPolicyTargetEpochs == 160);
    assert(sequencePlan.targetEpochs == 160);
    assert(sequencePlan.progressionDiagnostic == "target_sequence_advanced");
    assert(sequencePlan.inheritedPolicy.progressionMode == "target_sequence");
    assert(sequencePlan.inheritedPolicy.targetSequence == sequence.targetSequence);
    assert(sequencePlan.inheritedPolicy.targetEpochs == 160);
    assert(!sequencePlan.inheritedPolicy.lastDecision &&
           !sequencePlan.inheritedPolicy.queuedExperimentId);

    ContinuationPolicyConfig sequenceTerminalSource = SequencePolicy(220, 240);
    const ContinuationChildPolicyPlan sequenceTerminal =
        PlanContinuationChildPolicy(sequenceTerminalSource);
    assert(sequenceTerminal.terminal);
    assert(sequenceTerminal.childTrainingTargetEpochs == 240);
    assert(!sequenceTerminal.inheritedPolicyTargetEpochs);
    assert(!sequenceTerminal.inheritedPolicy.enabled);
    assert(!sequenceTerminal.inheritedPolicy.inheritToChild);
    assert(!sequenceTerminal.inheritedPolicy.targetEpochs);
    assert(sequenceTerminal.inheritedPolicy.targetSequence ==
           sequenceTerminalSource.targetSequence);
    assert(sequenceTerminal.progressionDiagnostic == "target_sequence_terminal");
    ContinuationPolicyConfig invalidSequencePlan = sequence;
    invalidSequencePlan.targetEpochs = 160;
    assert(ExceptionMessage([&] { PlanContinuationChildPolicy(invalidSequencePlan); }) ==
           "policy_target_is_not_next_sequence_target");

    const std::string sequenceCanonical =
        ContinuationPolicySemanticCanonicalText(sequence);
    assert(sequenceCanonical.find(
               "|progression_mode=target_sequence|target_sequence=[140,160,180,200,220,240]") !=
           std::string::npos);
    assert(ContinuationPolicySemanticHash(sequence) !=
           ContinuationPolicySemanticHash(legacy));
    ContinuationPolicyConfig sequenceProvenance = sequence;
    sequenceProvenance.policyInherited = true;
    sequenceProvenance.inheritedFromExperimentId = 999;
    sequenceProvenance.inheritedFromRevision = 8;
    sequenceProvenance.inheritedFromHash = "unrelated";
    assert(ContinuationPolicySemanticHash(sequenceProvenance) ==
           ContinuationPolicySemanticHash(sequence));
    ContinuationPolicyConfig changedSequence = sequence;
    changedSequence.targetSequence =
        std::vector<int>{140, 160, 180, 200, 224, 240};
    assert(ContinuationPolicySemanticHash(changedSequence) !=
           ContinuationPolicySemanticHash(sequence));

    // Profitability Phase 2A is diagnostic-only. Source selection uses the
    // preexisting analysis metrics and epochs even when observations differ.
    ContinuationEvidence checkpointBest;
    checkpointBest.analysisId = 501;
    checkpointBest.checkpointEvalId = 601;
    checkpointBest.inferenceEvalResultId = 701;
    checkpointBest.modelId = 801;
    checkpointBest.completedEpoch = 80;
    checkpointBest.leaderScore = 0.9;
    checkpointBest.inferAccuracy = 0.7;
    checkpointBest.analysisScope = "checkpoint";
    checkpointBest.profitabilityUnavailableReason =
        "no_profitability_observation";

    ContinuationEvidence checkpointLatest = checkpointBest;
    checkpointLatest.analysisId = 502;
    checkpointLatest.checkpointEvalId = 602;
    checkpointLatest.inferenceEvalResultId = 702;
    checkpointLatest.modelId = 802;
    checkpointLatest.completedEpoch = 100;
    checkpointLatest.leaderScore = 0.8;
    ContinuationProfitabilityEvidence zeroActionable;
    zeroActionable.observationId = 901;
    zeroActionable.observationIdentityHash =
        "fnv1a64:1111111111111111";
    zeroActionable.inferenceEvalResultId = 702;
    zeroActionable.inferenceScope = "checkpoint";
    zeroActionable.checkpointEvalId = 602;
    zeroActionable.metricDefinitionHash =
        "fnv1a64:2222222222222222";
    zeroActionable.sourceContentHash =
        "fnv1a64:3333333333333333";
    zeroActionable.predictionCount = 10;
    checkpointLatest.profitability = zeroActionable;
    checkpointLatest.profitabilityUnavailableReason.clear();

    ContinuationEvidence finalEvidence = checkpointLatest;
    finalEvidence.analysisId = 503;
    finalEvidence.checkpointEvalId.reset();
    finalEvidence.inferenceEvalResultId = 703;
    finalEvidence.modelId = 803;
    finalEvidence.analysisScope = "final";
    finalEvidence.profitability.reset();
    finalEvidence.profitabilityUnavailableReason =
        "no_profitability_observation";

    const std::vector<ContinuationEvidence> sourceEvidence{
        checkpointLatest,
        finalEvidence,
        checkpointBest};
    ContinuationPolicyConfig sourceSelection;
    sourceSelection.source.lastModelId = 803;
    sourceSelection.sourceMode = "best_checkpoint";
    const auto selectedBest = SelectContinuationSourceEvidence(
        sourceSelection,
        sourceEvidence);
    assert(selectedBest && selectedBest->checkpointEvalId == 601);
    sourceSelection.sourceMode = "latest_checkpoint";
    const auto selectedLatest = SelectContinuationSourceEvidence(
        sourceSelection,
        sourceEvidence);
    assert(selectedLatest && selectedLatest->checkpointEvalId == 602);
    assert(selectedLatest->profitability.has_value());
    assert(selectedLatest->profitability->observationId == 901);
    sourceSelection.sourceMode = "final_model";
    const auto selectedFinal = SelectContinuationSourceEvidence(
        sourceSelection,
        sourceEvidence);
    assert(selectedFinal && selectedFinal->modelId == 803);
    assert(!selectedFinal->profitability.has_value());

    ContinuationEvidence profitableBest = checkpointBest;
    profitableBest.profitability = zeroActionable;
    profitableBest.profitability->observationId = 999;
    std::vector<ContinuationEvidence> profitabilityChanged{
        checkpointLatest,
        finalEvidence,
        profitableBest};
    sourceSelection.sourceMode = "best_checkpoint";
    assert(SelectContinuationSourceEvidence(
               sourceSelection,
               profitabilityChanged)->checkpointEvalId ==
           selectedBest->checkpointEvalId);

    // Correcting only the FINAL diagnostic provenance (including changing or
    // removing the previously selected inference row) cannot affect any
    // continuation source mode or evidence-derived policy input.
    ContinuationEvidence correctedFinal = finalEvidence;
    correctedFinal.inferenceEvalResultId = 704;
    correctedFinal.acceptModel = false;
    correctedFinal.profitability = zeroActionable;
    correctedFinal.profitability->observationId = 1000;
    correctedFinal.profitability->inferenceEvalResultId = 704;
    correctedFinal.profitability->inferenceScope = "final";
    correctedFinal.profitability->checkpointEvalId.reset();
    correctedFinal.profitabilityUnavailableReason.clear();
    const std::vector<ContinuationEvidence> provenanceCorrected{
        checkpointLatest,
        correctedFinal,
        checkpointBest};
    sourceSelection.sourceMode = "best_checkpoint";
    assert(SelectContinuationSourceEvidence(
               sourceSelection,
               provenanceCorrected)->analysisId == selectedBest->analysisId);
    sourceSelection.sourceMode = "latest_checkpoint";
    assert(SelectContinuationSourceEvidence(
               sourceSelection,
               provenanceCorrected)->analysisId == selectedLatest->analysisId);
    sourceSelection.sourceMode = "final_model";
    assert(SelectContinuationSourceEvidence(
               sourceSelection,
               provenanceCorrected)->analysisId == selectedFinal->analysisId);

    const std::vector<ContinuationEvidence> distinctBefore =
        DeduplicateContinuationEvidence(sourceEvidence);
    const std::vector<ContinuationEvidence> distinctAfter =
        DeduplicateContinuationEvidence(profitabilityChanged);
    assert(distinctBefore.size() == distinctAfter.size());
    assert(distinctBefore.size() == 2);
    for (size_t index = 0; index < distinctBefore.size(); ++index)
    {
        assert(distinctBefore[index].analysisId ==
               distinctAfter[index].analysisId);
        assert(distinctBefore[index].leaderScore ==
               distinctAfter[index].leaderScore);
        assert(distinctBefore[index].inferAccuracy ==
               distinctAfter[index].inferAccuracy);
    }
    assert(ContinuationEvidenceWatermark(distinctBefore) ==
           ContinuationEvidenceWatermark(distinctAfter));
    const std::vector<ContinuationEvidence> distinctProvenanceCorrected =
        DeduplicateContinuationEvidence(provenanceCorrected);
    assert(distinctBefore.size() == distinctProvenanceCorrected.size());
    assert(ContinuationEvidenceWatermark(distinctBefore) ==
           ContinuationEvidenceWatermark(distinctProvenanceCorrected));

    ContinuationPolicyConfig trendPolicy;
    trendPolicy.trendMode = "non_degrading";
    trendPolicy.patience = 2;
    trendPolicy.maxDegradation = 0.2;
    std::optional<std::string> trendMetricBefore;
    std::optional<double> trendValueBefore;
    std::string trendReasonBefore;
    const ContinuationTrendResult trendBefore = EvaluateContinuationTrend(
        trendPolicy,
        distinctBefore,
        trendMetricBefore,
        trendValueBefore,
        trendReasonBefore);
    std::optional<std::string> trendMetricAfter;
    std::optional<double> trendValueAfter;
    std::string trendReasonAfter;
    const ContinuationTrendResult trendAfter = EvaluateContinuationTrend(
        trendPolicy,
        distinctAfter,
        trendMetricAfter,
        trendValueAfter,
        trendReasonAfter);
    assert(trendBefore == trendAfter);
    assert(trendMetricBefore == trendMetricAfter);
    assert(trendValueBefore == trendValueAfter);
    assert(trendReasonBefore == trendReasonAfter);
    std::optional<std::string> trendMetricProvenanceCorrected;
    std::optional<double> trendValueProvenanceCorrected;
    std::string trendReasonProvenanceCorrected;
    const ContinuationTrendResult trendProvenanceCorrected =
        EvaluateContinuationTrend(
            trendPolicy,
            distinctProvenanceCorrected,
            trendMetricProvenanceCorrected,
            trendValueProvenanceCorrected,
            trendReasonProvenanceCorrected);
    assert(trendBefore == trendProvenanceCorrected);
    assert(trendMetricBefore == trendMetricProvenanceCorrected);
    assert(trendValueBefore == trendValueProvenanceCorrected);
    assert(trendReasonBefore == trendReasonProvenanceCorrected);

    const std::string unavailableDiagnostic =
        ContinuationProfitabilityEvidenceLogFields(checkpointBest);
    assert(unavailableDiagnostic.find(
               "profitability_evidence=unavailable") !=
           std::string::npos);
    assert(unavailableDiagnostic.find(
               "profitability_unavailable_reason=no_profitability_observation") !=
           std::string::npos);
    ContinuationEvidence unresolvedFinal = finalEvidence;
    unresolvedFinal.profitabilityUnavailableReason =
        "no_exact_final_inference_result";
    const std::string unresolvedFinalDiagnostic =
        ContinuationProfitabilityEvidenceLogFields(unresolvedFinal);
    assert(unresolvedFinalDiagnostic.find(
               "profitability_unavailable_reason=no_exact_final_inference_result") !=
           std::string::npos);
    const std::string zeroActionableDiagnostic =
        ContinuationProfitabilityEvidenceLogFields(checkpointLatest);
    assert(zeroActionableDiagnostic.find(
               "profitability_evidence=available") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_observation_id=901") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_inference_scope=checkpoint") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_inference_eval_result_id=702") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_checkpoint_eval_id=602") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_metric_definition_hash=fnv1a64:2222222222222222") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_actionable_count=0") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_aggregate_terminal_horizon_log_return_sum=0") !=
           std::string::npos);
    assert(zeroActionableDiagnostic.find(
               "profitability_average_terminal_horizon_log_return_per_actionable_prediction=NULL") !=
           std::string::npos);

    std::cout << "ContinuationPolicyInheritanceTests passed\n";
    return 0;
}
