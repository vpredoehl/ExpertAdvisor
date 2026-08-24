#pragma once

#include <optional>
#include <string>

#include "SchedulerWorkerLimits.hpp"

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

} // namespace EA::ExperimentScheduler
