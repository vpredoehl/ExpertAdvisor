#pragma once

#include "WorkerProcessController.hpp"

#include <sys/types.h>

namespace EA::SchedulerCore
{

struct WorkerTerminationRequest
{
    pid_t processGroupId = -1;
    int gracefulPollAttempts = 30;
    unsigned int pollIntervalMicroseconds = 100000;
};

struct WorkerTerminationResult
{
    bool terminatedDuringGrace = false;
    bool killEscalated = false;
};

class WorkerControlService final
{
public:
    explicit WorkerControlService(WorkerProcessController& processes);

    WorkerTerminationResult terminateUncommittedWorker(
        const WorkerTerminationRequest& request) const;

private:
    WorkerProcessController& processes_;
};

} // namespace EA::SchedulerCore
