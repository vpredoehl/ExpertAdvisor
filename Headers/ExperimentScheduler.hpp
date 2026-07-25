#pragma once

namespace EA::ExperimentScheduler
{

bool IsExperimentSchedulerCommand(int argc, const char* argv[]);
int RunExperimentSchedulerCli(int argc, const char* argv[]);

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
