#pragma once

#include "SchedulerWorkerLimits.hpp"

namespace EA::SchedulerCore
{
struct SchedulerDaemonConfiguration;
}

namespace EA::ExperimentScheduler
{

bool IsExperimentSchedulerCommand(int argc, const char* argv[]);
int RunExperimentSchedulerCli(int argc, const char* argv[]);
int RunSchedulerDaemon(
    const EA::SchedulerCore::SchedulerDaemonConfiguration& configuration);

} // namespace EA::ExperimentScheduler
