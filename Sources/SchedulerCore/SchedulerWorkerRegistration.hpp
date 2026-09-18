#pragma once

#include <optional>
#include <string>

namespace EA::SchedulerCore
{

struct SchedulerWorkerRegistration
{
    long long workerAttemptId = 0;
    std::optional<long long> experimentId;
    std::optional<long long> checkpointEvalId;
    std::string workerKind;
    std::string lifecyclePhase;
};

// Worker-process self-registration is intentionally separate from the daemon
// engine: worker identity is not scheduler authority identity.
bool RegisterSchedulerWorker(
    const SchedulerWorkerRegistration& registration);

} // namespace EA::SchedulerCore
