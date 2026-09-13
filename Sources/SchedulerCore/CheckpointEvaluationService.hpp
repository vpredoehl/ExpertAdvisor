#pragma once

#include "CheckpointPolicy.hpp"

#include <functional>
#include <iosfwd>
#include <optional>
#include <string>

namespace EA::SchedulerCore
{

struct CheckpointEvaluationRecord
{
    long long checkpointEvalId = -1;
    long long parentExperimentId = -1;
    long long checkpointModelId = -1;
    int checkpointEpoch = 0;
    std::string symbol;
    int predictionHorizon = 0;
};

struct ValidatedCheckpointPolicyEvidence
{
    std::optional<double> leaderScore;
    std::optional<double> inferenceAccuracy;
    long long analysisId = -1;
    long long inferenceEvalResultId = -1;
    std::string inferenceFromDate;
    std::string inferenceToDate;
};

struct CheckpointPolicyEvidenceLoadResult
{
    std::optional<ValidatedCheckpointPolicyEvidence> evidence;
    std::string rejectionReason;
};

struct CheckpointPolicyPopulation
{
    int count = 0;
    std::optional<int> rankValue;
    std::string watermark;
};

struct CheckpointPolicyDecisionContext
{
    EA::ExperimentScheduler::CheckpointPolicyDecision decision;
    CheckpointPolicyPopulation completedPopulation;
    CheckpointPolicyPopulation rankPopulation;
};

struct PersistedCheckpointPolicyDecision
{
    long long decisionId = -1;
    std::string identityStatus;
    bool reused = false;
};

struct CheckpointPolicyEvaluationResult
{
    bool evaluated = false;
    std::string decision = "skipped";
    std::string reason;
};

std::optional<std::string> CheckpointPolicyConfigurationError(
    const EA::ExperimentScheduler::CheckpointPolicyConfig& config,
    bool requireRules);

CheckpointPolicyDecisionContext PlanCheckpointPolicyDecision(
    const CheckpointEvaluationRecord& evaluation,
    const EA::ExperimentScheduler::CheckpointPolicyConfig& config,
    const ValidatedCheckpointPolicyEvidence& evidence,
    CheckpointPolicyPopulation completedPopulation,
    CheckpointPolicyPopulation rankPopulation);

EA::ExperimentScheduler::CheckpointPolicyEvidenceIdentity
MakeCheckpointPolicyEvidenceIdentity(
    const CheckpointEvaluationRecord& evaluation,
    const ValidatedCheckpointPolicyEvidence& evidence,
    const CheckpointPolicyDecisionContext& context,
    const EA::ExperimentScheduler::CheckpointPolicyConfig& config);

// Transaction ownership and persistence stay in the scheduler adapter. These
// operations expose only checkpoint-policy domain values, never database
// transaction types.
struct CheckpointEvaluationOperations
{
    std::function<bool()> schemaAvailable;
    std::function<std::optional<EA::ExperimentScheduler::CheckpointPolicyConfig>(
        long long parentExperimentId)> loadPolicy;
    std::function<EA::ExperimentScheduler::CheckpointPolicyConfig(
        const CheckpointEvaluationRecord&,
        EA::ExperimentScheduler::CheckpointPolicyConfig)>
        reconcilePolicyIdentity;
    std::function<CheckpointPolicyEvidenceLoadResult(
        const CheckpointEvaluationRecord&)> loadEvidence;
    std::function<CheckpointPolicyPopulation(long long parentExperimentId)>
        loadCompletedPopulation;
    std::function<CheckpointPolicyPopulation(
        const CheckpointEvaluationRecord&,
        const EA::ExperimentScheduler::CheckpointPolicyConfig&)>
        loadRankPopulation;
    std::function<PersistedCheckpointPolicyDecision(
        const CheckpointEvaluationRecord&,
        const EA::ExperimentScheduler::CheckpointPolicyConfig&,
        const EA::ExperimentScheduler::CheckpointPolicyDecision&,
        const ValidatedCheckpointPolicyEvidence&,
        const EA::ExperimentScheduler::CheckpointPolicyEvidenceIdentity&)>
        persistDecision;
    std::function<std::string(
        const CheckpointEvaluationRecord&,
        const EA::ExperimentScheduler::CheckpointPolicyConfig&,
        const EA::ExperimentScheduler::CheckpointPolicyDecision&,
        const PersistedCheckpointPolicyDecision&,
        const std::string& expectedEvidenceWatermark)>
        applyStopRequest;
};

class CheckpointEvaluationService final
{
public:
    CheckpointEvaluationService(
        CheckpointEvaluationOperations operations,
        std::ostream& output);

    CheckpointPolicyEvaluationResult evaluate(
        const CheckpointEvaluationRecord& evaluation);

private:
    CheckpointEvaluationOperations operations_;
    std::ostream& output_;
};

} // namespace EA::SchedulerCore
