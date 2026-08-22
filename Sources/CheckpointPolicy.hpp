#pragma once

#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentScheduler
{

struct CheckpointPolicyConfig
{
    bool enabled = false;
    bool checkpointInferEnabled = false;
    std::optional<double> minLeaderScore;
    std::optional<double> minInferAccuracy;
    std::optional<int> topN;
    std::string scope = "symbol_horizon";
    std::string stopMode = "next_checkpoint";
    int graceEvals = 1;
    int checkpointInterval = 20;
    int targetEpochs = 0;
    std::optional<int> currentEpoch;
    std::optional<int> stopAfterCheckpointEpoch;
    std::string status;
    std::string phase;
    long long policyRevision = 1;
    std::optional<std::string> persistedPolicyHash;
    std::optional<long long> activeTrainingAttemptId;
};

struct CheckpointPolicyDecision
{
    std::string decision = "skipped";
    std::string reason;
    std::optional<int> rankValue;
    std::optional<int> requestedStopEpoch;
    std::vector<std::string> passedRules;
    std::vector<std::string> failedRules;
};

struct CheckpointPolicyDecisionInputs
{
    int checkpointEpoch = 0;
    int completedEvalCount = 0;
    std::optional<double> leaderScore;
    std::optional<double> inferenceAccuracy;
    std::optional<int> rankValue;
};

struct CheckpointPolicyEvidenceIdentity
{
    long long checkpointEvalId = -1;
    long long parentExperimentId = -1;
    long long checkpointModelId = -1;
    int checkpointEpoch = 0;
    std::optional<int> observedCurrentEpoch;
    long long analysisId = -1;
    long long inferenceEvalResultId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    std::string inferenceFromDate;
    std::string inferenceToDate;
    std::optional<double> leaderScore;
    std::optional<double> inferenceAccuracy;
    std::optional<int> rankValue;
    std::string rankScope;
    std::string rankPopulationWatermark;
    int completedEvalCount = 0;
    std::string completedPopulationWatermark;
    std::string checkpointStatus = "completed";
    std::string checkpointPhase = "done";
    std::string analysisStatus = "completed";
    std::string inferenceStatus = "completed";
    std::string inferenceScope = "checkpoint";
};

std::string CheckpointPolicyCanonicalText(const CheckpointPolicyConfig& config);
std::string StableCheckpointPolicyHash(const std::string& canonicalText);
std::string CheckpointPolicySemanticHash(const CheckpointPolicyConfig& config);
std::string CheckpointPolicyEvidenceCanonicalText(
    const CheckpointPolicyEvidenceIdentity& evidence);
std::string CheckpointPolicyEvidenceWatermark(
    const CheckpointPolicyEvidenceIdentity& evidence);

std::optional<int> RequestedCheckpointPolicyStopEpoch(
    int checkpointEpoch,
    const CheckpointPolicyConfig& config);
CheckpointPolicyDecision DecideCheckpointPolicyPure(
    const CheckpointPolicyConfig& config,
    const CheckpointPolicyDecisionInputs& inputs);

}  // namespace EA::ExperimentScheduler
