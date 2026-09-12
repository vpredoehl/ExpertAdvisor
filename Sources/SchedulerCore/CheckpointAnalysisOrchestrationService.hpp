#pragma once

#include <functional>
#include <iosfwd>
#include <optional>
#include <string>

namespace EA::SchedulerCore
{

struct CheckpointAnalysisClaim
{
    long long checkpointEvalId = -1;
    long long experimentId = -1;
    long long checkpointModelId = -1;
    int checkpointEpoch = 0;
    long long workerAttemptId = -1;
};

struct CheckpointAnalysisWorkResult
{
    bool success = false;
    std::string error;
    long long inferenceResultId = -1;
};

struct CheckpointAnalysisFinalizationPlan
{
    bool complete = false;
    std::string attemptLifecycleState;
    std::string reconciliationResult;
    std::string diagnostic;
};

CheckpointAnalysisFinalizationPlan PlanCheckpointAnalysisFinalization(
    const CheckpointAnalysisWorkResult& work,
    const std::optional<long long>& currentInferenceResultId);

// Adapter boundary for transaction-scoped persistence, scheduler authority
// failpoints, analysis evidence calculation, and report generation.
struct CheckpointAnalysisOperations
{
    std::function<std::optional<CheckpointAnalysisClaim>()> claim;
    std::function<void(const CheckpointAnalysisClaim&)> afterClaim;
    std::function<CheckpointAnalysisWorkResult(
        const CheckpointAnalysisClaim&)> execute;
    std::function<void(
        const CheckpointAnalysisClaim&,
        const CheckpointAnalysisWorkResult&)> afterWork;
    std::function<bool(
        const CheckpointAnalysisClaim&,
        const CheckpointAnalysisWorkResult&)> finalize;
    std::function<void()> generateReports;
};

class CheckpointAnalysisOrchestrationService final
{
public:
    CheckpointAnalysisOrchestrationService(
        CheckpointAnalysisOperations operations,
        std::ostream& output,
        std::ostream& errors,
        long long processId);

    int runOne();

private:
    CheckpointAnalysisOperations operations_;
    std::ostream& output_;
    std::ostream& errors_;
    long long processId_ = -1;
};

} // namespace EA::SchedulerCore
