#pragma once

#include "SchedulerCore/SchedulerAuthorityService.hpp"

#include <string>

namespace EA::ExperimentScheduler
{

using EA::SchedulerCore::DecideSchedulerTakeover;
using EA::SchedulerCore::SchedulerOwnerProcessEvidence;
using EA::SchedulerCore::SchedulerTakeoverDecision;

inline bool WorkerAttemptConsumesCapacity(
    const std::string& lifecycleState) noexcept
{
    return lifecycleState == "reserved" ||
           lifecycleState == "spawned" ||
           lifecycleState == "running" ||
           lifecycleState == "observed" ||
           lifecycleState == "identity_ambiguous";
}

inline bool WorkerAttemptIsTerminal(
    const std::string& lifecycleState) noexcept
{
    return lifecycleState == "completed" ||
           lifecycleState == "failed" ||
           lifecycleState == "launch_failed" ||
           lifecycleState == "abandoned";
}

} // namespace EA::ExperimentScheduler
