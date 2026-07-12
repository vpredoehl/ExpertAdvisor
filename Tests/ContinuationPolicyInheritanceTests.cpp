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

    std::cout << "ContinuationPolicyInheritanceTests passed\n";
    return 0;
}
