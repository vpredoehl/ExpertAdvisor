#include "SchedulerEngine.hpp"

#include "ExperimentScheduler.hpp"

namespace EA::SchedulerCore
{

bool SchedulerEngine::recognizes(
    const CommandInvocation& invocation) const
{
    return EA::ExperimentScheduler::IsExperimentSchedulerCommand(
        invocation.argc, invocation.argv);
}

int SchedulerEngine::run(const CommandInvocation& invocation) const
{
    return EA::ExperimentScheduler::RunExperimentSchedulerCli(
        invocation.argc, invocation.argv);
}

bool SchedulerEngine::registerWorkerAttempt(
    const WorkerAttemptRegistration& registration) const
{
    return EA::ExperimentScheduler::RegisterSchedulerWorkerAttempt(
        registration.workerAttemptId,
        registration.experimentId,
        registration.checkpointEvalId,
        registration.workerKind,
        registration.lifecyclePhase);
}

} // namespace EA::SchedulerCore
