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

ContinuationChildPolicyPlan PlanContinuationChildPolicy(
    const ContinuationPolicyConfig& sourceConfig)
{
    ContinuationChildPolicyPlan plan;
    if (!sourceConfig.inheritToChild)
        return plan;

    plan.inherit = true;
    if (!sourceConfig.targetEpochs.has_value())
        throw std::runtime_error("inherit_to_child_requires_target_epochs");
    const BoundedContinuationTargetDerivation boundedTarget =
        DeriveBoundedContinuationChildPolicy(
            *sourceConfig.targetEpochs,
            sourceConfig.targetIncrement,
            sourceConfig.maxTargetEpochs);
    if (!boundedTarget.error.empty())
        throw std::runtime_error(boundedTarget.error);

    plan.terminal = boundedTarget.terminal;
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
        plan.targetEpochs = *boundedTarget.inheritedPolicyTargetEpochs;
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

} // namespace EA::ExperimentScheduler
