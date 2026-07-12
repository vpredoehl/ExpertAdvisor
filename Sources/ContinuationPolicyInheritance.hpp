#pragma once

#include <optional>
#include <string>

#include "ContinuationPolicy.hpp"

namespace EA::ExperimentScheduler
{

struct ContinuationTargetDerivation
{
    std::optional<int> targetEpochs;
    std::string error;
};

struct BoundedContinuationTargetDerivation
{
    bool terminal = false;
    std::optional<int> inheritedPolicyTargetEpochs;
    std::string error;
};

struct SequenceContinuationTargetDerivation
{
    int childTrainingTargetEpochs = 0;
    std::optional<int> inheritedPolicyTargetEpochs;
    bool terminal = false;
    std::string error;
};

ContinuationTargetDerivation DeriveContinuationChildPolicyTarget(
    const std::optional<int>& childTargetEpochs,
    const std::optional<int>& targetIncrement);

BoundedContinuationTargetDerivation DeriveBoundedContinuationChildPolicy(
    int childTrainingTargetEpochs,
    const std::optional<int>& targetIncrement,
    const std::optional<int>& maxTargetEpochs);

SequenceContinuationTargetDerivation DeriveSequenceContinuationChildPolicy(
    int currentExperimentTargetEpochs,
    const std::optional<int>& policyTargetEpochs,
    const std::optional<std::vector<int>>& targetSequence);

struct ContinuationChildPolicyPlan
{
    bool inherit = false;
    bool terminal = false;
    std::string progressionMode;
    int childTrainingTargetEpochs = 0;
    std::optional<int> inheritedPolicyTargetEpochs;
    std::string progressionDiagnostic;
    int targetEpochs = 0;
    std::string policyHash;
    std::string sourcePolicyHash;
    ContinuationPolicyConfig inheritedPolicy;
};

ContinuationChildPolicyPlan PlanContinuationChildPolicy(
    const ContinuationPolicyConfig& sourceConfig);

} // namespace EA::ExperimentScheduler
