#pragma once

#include <optional>
#include <string>

namespace EA::SchedulerCore
{

struct CommandInvocation
{
    int argc = 0;
    const char** argv = nullptr;
};

struct WorkerAttemptRegistration
{
    long long workerAttemptId = 0;
    std::optional<long long> experimentId;
    std::optional<long long> checkpointEvalId;
    std::string workerKind;
    std::string lifecyclePhase;
};

// Typed process boundary for the existing production scheduler implementation.
// The compatibility entrypoints remain available in ExperimentScheduler.hpp;
// new executables can depend on this component without reading CLI globals.
class SchedulerEngine final
{
public:
    bool recognizes(const CommandInvocation& invocation) const;
    int run(const CommandInvocation& invocation) const;
    bool registerWorkerAttempt(
        const WorkerAttemptRegistration& registration) const;
};

} // namespace EA::SchedulerCore
