#include "WorkerControlService.hpp"

#include <unistd.h>

namespace EA::SchedulerCore
{

WorkerControlService::WorkerControlService(
    WorkerProcessController& processes)
    : processes_{processes}
{
}

WorkerTerminationResult WorkerControlService::terminateUncommittedWorker(
    const WorkerTerminationRequest& request) const
{
    WorkerTerminationResult result;
    if (!processes_.signal(
            request.processGroupId,
            WorkerProcessSignal::Terminate,
            true).success)
    {
        (void)processes_.signal(
            request.processGroupId,
            WorkerProcessSignal::Terminate);
    }
    for (int attempt = 0;
         attempt < request.gracefulPollAttempts;
         ++attempt)
    {
        const bool groupExists =
            processes_.isAlive(request.processGroupId, true).alive;
        const bool leaderExists =
            processes_.isAlive(request.processGroupId).alive;
        if (!groupExists && !leaderExists)
        {
            result.terminatedDuringGrace = true;
            return result;
        }
        if (request.pollIntervalMicroseconds != 0)
            ::usleep(request.pollIntervalMicroseconds);
    }

    result.killEscalated = true;
    (void)processes_.signal(
        request.processGroupId, WorkerProcessSignal::Kill, true);
    (void)processes_.signal(
        request.processGroupId, WorkerProcessSignal::Kill);
    return result;
}

} // namespace EA::SchedulerCore
