#pragma once

#include <optional>
#include <string>

namespace EA::ExperimentScheduler
{

bool IsExperimentSchedulerCommand(int argc, const char* argv[]);
int RunExperimentSchedulerCli(int argc, const char* argv[]);
bool RegisterSchedulerWorkerAttempt(
    long long workerAttemptId,
    const std::optional<long long>& experimentId,
    const std::optional<long long>& checkpointEvalId,
    const std::string& workerKind,
    const std::string& lifecyclePhase);

constexpr int AvailableInferProcessSlots(
    int maximum,
    int runningExperimentInference,
    int runningCheckpointInference) noexcept
{
    const int available =
        maximum -
        runningExperimentInference -
        runningCheckpointInference;
    return available > 0 ? available : 0;
}

} // namespace EA::ExperimentScheduler
