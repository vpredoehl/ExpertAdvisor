#pragma once

#include <string>

namespace EA::ExperimentScheduler
{

enum class SchedulerOwnerProcessEvidence
{
    Valid,
    Missing,
    IdentityMismatch,
    Ambiguous
};

enum class SchedulerTakeoverDecision
{
    AcquireVacant,
    AcquireReleased,
    TakeOverExpiredDeadOwner,
    RejectValidOwner,
    RejectFreshLease,
    RejectAmbiguousOwner
};

inline SchedulerTakeoverDecision DecideSchedulerTakeover(
    bool hasOwner,
    bool explicitlyReleased,
    bool leaseExpired,
    SchedulerOwnerProcessEvidence processEvidence) noexcept
{
    if (!hasOwner)
        return SchedulerTakeoverDecision::AcquireVacant;
    if (explicitlyReleased)
        return SchedulerTakeoverDecision::AcquireReleased;
    if (!leaseExpired)
        return SchedulerTakeoverDecision::RejectFreshLease;
    if (processEvidence == SchedulerOwnerProcessEvidence::Missing ||
        processEvidence == SchedulerOwnerProcessEvidence::IdentityMismatch)
    {
        return SchedulerTakeoverDecision::TakeOverExpiredDeadOwner;
    }
    if (processEvidence == SchedulerOwnerProcessEvidence::Valid)
        return SchedulerTakeoverDecision::RejectValidOwner;
    return SchedulerTakeoverDecision::RejectAmbiguousOwner;
}

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
