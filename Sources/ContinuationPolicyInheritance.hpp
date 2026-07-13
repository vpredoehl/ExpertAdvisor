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

struct PersistedContinuationIdentity
{
    long long decisionId = -1;
    long long sourceExperimentId = -1;
    long long sourceModelId = -1;
    long long sourceAnalysisId = -1;
    std::optional<long long> sourceCheckpointEvalId;
    int sourceEpoch = 0;
    int targetEpochs = 0;
    std::string decision;
    int observedEvalCount = 0;
    int patienceWindow = 0;
    std::optional<double> leaderScore;
    std::optional<double> inferAccuracy;
    std::optional<int> rankValue;
    std::optional<std::string> trendMetric;
    std::optional<double> trendValue;
    long long policyRevision = 0;
    std::string policyHash;
    std::string evidenceWatermark;
    std::optional<long long> queuedExperimentId;
    bool queuedChildExists = false;
    std::string queuedChildStatus;
    std::optional<long long> childParentExperimentId;
    std::optional<long long> childSourceExperimentId;
    std::optional<long long> childResumeModelId;
    std::optional<long long> childSourceModelId;
    std::optional<int> childSourceEpoch;
    int childTargetEpochs = 0;
    int childGeneration = 0;
    bool childPolicyInherited = false;
    std::string childPolicySourceMode;
    std::string sourceAnalysisScope;
    bool sourceAnalysisValid = false;
    bool sourceModelOwnedBySource = false;
    bool evidenceChangedAfterDecision = false;
};

struct ContinuationAutoSatisfactionResult
{
    bool alreadySatisfied = false;
    std::string reason;
    std::string currentPolicyHash;
    std::string persistedDecisionPolicyHash;
};

ContinuationAutoSatisfactionResult CheckAutomaticContinuationSatisfaction(
    const ContinuationPolicyConfig& currentPolicy,
    const PersistedContinuationIdentity& persisted);

} // namespace EA::ExperimentScheduler
