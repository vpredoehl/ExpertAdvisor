#include "ContinuationPolicyInheritance.hpp"

#include <limits>
#include <stdexcept>

namespace EA::ExperimentScheduler
{

ContinuationTargetDerivation DeriveContinuationChildPolicyTarget(
    const std::optional<int>& childTargetEpochs,
    const std::optional<int>& targetIncrement)
{
    if (!childTargetEpochs.has_value())
        return {std::nullopt, "inherit_to_child_requires_target_epochs"};
    if (!targetIncrement.has_value())
        return {std::nullopt, "inherit_to_child_requires_target_increment"};
    if (*targetIncrement <= 0)
        return {std::nullopt, "target_increment_must_be_positive"};
    if (*childTargetEpochs > std::numeric_limits<int>::max() - *targetIncrement)
        return {std::nullopt, "inherited_target_epochs_overflow"};
    const int derived = *childTargetEpochs + *targetIncrement;
    if (derived <= *childTargetEpochs)
        return {std::nullopt, "inherited_target_must_exceed_child_target"};
    return {derived, {}};
}

BoundedContinuationTargetDerivation DeriveBoundedContinuationChildPolicy(
    int childTrainingTargetEpochs,
    const std::optional<int>& targetIncrement,
    const std::optional<int>& maxTargetEpochs)
{
    if (maxTargetEpochs.has_value())
    {
        if (childTrainingTargetEpochs > *maxTargetEpochs)
            return {false, std::nullopt, "policy_target_exceeds_max_target"};
        if (childTrainingTargetEpochs == *maxTargetEpochs)
            return {true, std::nullopt, {}};
    }
    const ContinuationTargetDerivation next =
        DeriveContinuationChildPolicyTarget(childTrainingTargetEpochs, targetIncrement);
    if (!next.error.empty())
        return {false, std::nullopt, next.error};
    if (maxTargetEpochs.has_value() && *next.targetEpochs > *maxTargetEpochs)
        return {false, std::nullopt, "target_increment_would_skip_past_max_target"};
    return {false, next.targetEpochs, {}};
}

SequenceContinuationTargetDerivation DeriveSequenceContinuationChildPolicy(
    int currentExperimentTargetEpochs,
    const std::optional<int>& policyTargetEpochs,
    const std::optional<std::vector<int>>& targetSequence)
{
    SequenceContinuationTargetDerivation result;
    if (!targetSequence.has_value() || targetSequence->empty())
    {
        result.error = "target_sequence_required";
        return result;
    }
    for (size_t index = 0; index < targetSequence->size(); ++index)
    {
        const int value = (*targetSequence)[index];
        if (value <= 0)
        {
            result.error = "target_sequence_contains_nonpositive_value";
            return result;
        }
        if (index != 0)
        {
            if (value == (*targetSequence)[index - 1])
            {
                result.error = "target_sequence_contains_duplicate";
                return result;
            }
            if (value < (*targetSequence)[index - 1])
            {
                result.error = "target_sequence_must_be_strictly_increasing";
                return result;
            }
        }
    }
    if (!policyTargetEpochs.has_value())
    {
        result.error = "policy_target_not_in_sequence";
        return result;
    }

    size_t policyIndex = targetSequence->size();
    size_t nextIndex = targetSequence->size();
    for (size_t index = 0; index < targetSequence->size(); ++index)
    {
        if ((*targetSequence)[index] == *policyTargetEpochs)
            policyIndex = index;
        if (nextIndex == targetSequence->size() &&
            (*targetSequence)[index] > currentExperimentTargetEpochs)
            nextIndex = index;
    }
    if (policyIndex == targetSequence->size())
    {
        result.error = "policy_target_not_in_sequence";
        return result;
    }
    if (nextIndex == targetSequence->size())
    {
        result.error = "sequence_has_no_target_after_current_epoch";
        return result;
    }
    if (policyIndex != nextIndex)
    {
        result.error = "policy_target_is_not_next_sequence_target";
        return result;
    }

    result.childTrainingTargetEpochs = *policyTargetEpochs;
    result.terminal = policyIndex + 1 == targetSequence->size();
    if (!result.terminal)
        result.inheritedPolicyTargetEpochs = (*targetSequence)[policyIndex + 1];
    return result;
}

ContinuationChildPolicyPlan PlanContinuationChildPolicy(
    const ContinuationPolicyConfig& sourceConfig)
{
    ContinuationChildPolicyPlan plan;
    if (!sourceConfig.inheritToChild)
        return plan;

    plan.inherit = true;
    if (!sourceConfig.targetEpochs.has_value())
        throw std::runtime_error("inherit_to_child_requires_target_epochs");
    plan.progressionMode =
        EffectiveContinuationProgressionMode(sourceConfig).value_or("fixed_increment");
    plan.childTrainingTargetEpochs = *sourceConfig.targetEpochs;
    if (plan.progressionMode == "target_sequence")
    {
        const SequenceContinuationTargetDerivation sequenceTarget =
            DeriveSequenceContinuationChildPolicy(
                sourceConfig.source.targetEpochs,
                sourceConfig.targetEpochs,
                sourceConfig.targetSequence);
        if (!sequenceTarget.error.empty())
            throw std::runtime_error(sequenceTarget.error);
        plan.terminal = sequenceTarget.terminal;
        plan.inheritedPolicyTargetEpochs =
            sequenceTarget.inheritedPolicyTargetEpochs;
        plan.progressionDiagnostic =
            plan.terminal ? "target_sequence_terminal" : "target_sequence_advanced";
    }
    else
    {
        const BoundedContinuationTargetDerivation boundedTarget =
            DeriveBoundedContinuationChildPolicy(
                *sourceConfig.targetEpochs,
                sourceConfig.targetIncrement,
                sourceConfig.maxTargetEpochs);
        if (!boundedTarget.error.empty())
            throw std::runtime_error(boundedTarget.error);
        plan.terminal = boundedTarget.terminal;
        plan.inheritedPolicyTargetEpochs =
            boundedTarget.inheritedPolicyTargetEpochs;
        plan.progressionDiagnostic =
            plan.terminal ? "fixed_increment_terminal" : "fixed_increment_advanced";
    }

    plan.sourcePolicyHash = ContinuationPolicySemanticHash(sourceConfig);
    plan.inheritedPolicy = sourceConfig;
    plan.inheritedPolicy.source.targetEpochs = *sourceConfig.targetEpochs;
    plan.inheritedPolicy.enabled = !plan.terminal;
    plan.inheritedPolicy.inheritToChild = !plan.terminal;
    plan.inheritedPolicy.inheritanceStatus =
        plan.terminal ? "max_target_reached" : "valid";
    if (plan.terminal)
    {
        plan.inheritedPolicy.targetEpochs.reset();
    }
    else
    {
        plan.targetEpochs = *plan.inheritedPolicyTargetEpochs;
        plan.inheritedPolicy.targetEpochs = plan.targetEpochs;
    }
    plan.inheritedPolicy.policyRevision = 1;
    plan.inheritedPolicy.lastDecision.reset();
    plan.inheritedPolicy.lastReason.reset();
    plan.inheritedPolicy.selectedModelId.reset();
    plan.inheritedPolicy.queuedExperimentId.reset();

    const std::optional<std::string> configError =
        ContinuationPolicyConfigurationError(
            plan.inheritedPolicy,
            !plan.terminal);
    if (configError.has_value())
        throw std::runtime_error("invalid_inherited_policy:" + *configError);

    plan.policyHash = ContinuationPolicySemanticHash(plan.inheritedPolicy);
    return plan;
}

ContinuationAutoSatisfactionResult CheckAutomaticContinuationSatisfaction(
    const ContinuationPolicyConfig& currentPolicy,
    const PersistedContinuationIdentity& persisted)
{
    ContinuationAutoSatisfactionResult result;
    result.currentPolicyHash = ContinuationPolicySemanticHash(currentPolicy);
    result.persistedDecisionPolicyHash = persisted.policyHash;
    const auto requireFullEvaluation = [&](const std::string& reason) {
        result.reason = reason;
        return result;
    };

    const std::optional<std::string> configurationError =
        ContinuationPolicyConfigurationError(currentPolicy, true);
    if (configurationError.has_value())
        return requireFullEvaluation("current_policy_invalid");
    if (!currentPolicy.enabled ||
        currentPolicy.status != "completed" || currentPolicy.phase != "done")
    {
        return requireFullEvaluation("source_not_automatic_evaluation_ready");
    }
    if (!currentPolicy.targetEpochs.has_value() ||
        persisted.targetEpochs != *currentPolicy.targetEpochs)
    {
        return requireFullEvaluation("derived_target_changed");
    }
    if (persisted.sourceExperimentId != currentPolicy.sourceExperimentId)
        return requireFullEvaluation("source_experiment_mismatch");
    if (persisted.decision != "continuation_queued" ||
        !persisted.queuedExperimentId.has_value())
    {
        return requireFullEvaluation("decision_not_continuation_queued");
    }
    if (!persisted.queuedChildExists)
        return requireFullEvaluation("queued_child_missing");
    if (persisted.queuedChildStatus == "cancelled" ||
        persisted.queuedChildStatus == "failed")
    {
        return requireFullEvaluation("queued_child_retryable_or_failed");
    }
    if (!persisted.sourceAnalysisValid || !persisted.sourceModelOwnedBySource)
        return requireFullEvaluation("persisted_source_identity_invalid");

    const int expectedGeneration =
        currentPolicy.continuationSourceExperimentId.has_value()
            ? currentPolicy.continuationGeneration + 1
            : 1;
    if (persisted.childParentExperimentId != currentPolicy.sourceExperimentId ||
        persisted.childSourceExperimentId != currentPolicy.sourceExperimentId ||
        persisted.childResumeModelId != persisted.sourceModelId ||
        persisted.childSourceModelId != persisted.sourceModelId ||
        persisted.childSourceEpoch != persisted.sourceEpoch ||
        persisted.childTargetEpochs != persisted.targetEpochs ||
        persisted.childGeneration != expectedGeneration)
    {
        return requireFullEvaluation("queued_child_lineage_mismatch");
    }
    if (persisted.evidenceChangedAfterDecision)
        return requireFullEvaluation("evidence_changed_after_decision");

    // Ranking depends on the live comparison population, not only this source's
    // evidence. Without loading that population, a queued top-N decision cannot
    // be proven current, so retain the full evaluator for that policy shape.
    if (currentPolicy.topN.has_value())
        return requireFullEvaluation("ranking_requires_full_evaluation");
    if (persisted.observedEvalCount < currentPolicy.minEvals)
        return requireFullEvaluation("minimum_evidence_changed");
    if (currentPolicy.minLeaderScore.has_value() &&
        (!persisted.leaderScore.has_value() ||
         *persisted.leaderScore < *currentPolicy.minLeaderScore))
    {
        return requireFullEvaluation("leader_threshold_changed");
    }
    if (currentPolicy.minInferAccuracy.has_value() &&
        (!persisted.inferAccuracy.has_value() ||
         *persisted.inferAccuracy < *currentPolicy.minInferAccuracy))
    {
        return requireFullEvaluation("inference_threshold_changed");
    }
    if (currentPolicy.trendMode != "none")
    {
        if (persisted.patienceWindow != currentPolicy.patience ||
            !persisted.trendMetric.has_value() ||
            !persisted.trendValue.has_value())
        {
            return requireFullEvaluation("trend_configuration_changed");
        }
        if (currentPolicy.trendMode == "non_degrading" &&
            *persisted.trendValue < -*currentPolicy.maxDegradation)
        {
            return requireFullEvaluation("trend_gate_changed");
        }
        if (currentPolicy.trendMode == "improving" &&
            *persisted.trendValue < *currentPolicy.minImprovement)
        {
            return requireFullEvaluation("trend_gate_changed");
        }
    }

    const bool persistedFinal =
        persisted.sourceAnalysisScope == "final" &&
        !persisted.sourceCheckpointEvalId.has_value();
    const bool persistedCheckpoint =
        persisted.sourceAnalysisScope == "checkpoint" &&
        persisted.sourceCheckpointEvalId.has_value();
    if (currentPolicy.sourceMode == "final_model")
    {
        if (!persistedFinal ||
            currentPolicy.source.lastModelId != persisted.sourceModelId)
        {
            return requireFullEvaluation("source_selection_changed");
        }
    }
    else
    {
        if (!persistedCheckpoint)
            return requireFullEvaluation("source_selection_changed");
        // Equal semantic hashes prove the checkpoint source mode is unchanged.
        // An inherited child also captures the source mode used when it was
        // created. Otherwise best_checkpoint/latest_checkpoint ambiguity is
        // deliberately resolved by the authoritative full evaluator.
        const bool checkpointModeProven =
            result.currentPolicyHash == persisted.policyHash ||
            (persisted.childPolicyInherited &&
             persisted.childPolicySourceMode == currentPolicy.sourceMode);
        if (!checkpointModeProven)
            return requireFullEvaluation("checkpoint_source_mode_requires_full_evaluation");
    }

    // A queued decision is immutable. Its semantic hash describes the policy
    // revision evaluated at queue time, while the current hash may differ after
    // progression-only or still-passing gate edits. Child identity is instead
    // authoritative: source/model/epoch, target, generation, and lineage must
    // all match. Fresh evidence or any gate/source ambiguity above falls back.
    result.alreadySatisfied = true;
    result.reason = "existing_continuation_matches";
    return result;
}

} // namespace EA::ExperimentScheduler
