#include "../Headers/SchedulerExecutablePath.hpp"
#include "../Headers/SchedulerOwnershipPolicy.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

using namespace EA::ExperimentScheduler;

int main()
{
    assert(
        DecideSchedulerTakeover(
            false,
            false,
            false,
            SchedulerOwnerProcessEvidence::Ambiguous) ==
        SchedulerTakeoverDecision::AcquireVacant);
    assert(
        DecideSchedulerTakeover(
            true,
            true,
            false,
            SchedulerOwnerProcessEvidence::Valid) ==
        SchedulerTakeoverDecision::AcquireReleased);
    assert(
        DecideSchedulerTakeover(
            true,
            false,
            true,
            SchedulerOwnerProcessEvidence::Missing) ==
        SchedulerTakeoverDecision::TakeOverExpiredDeadOwner);
    assert(
        DecideSchedulerTakeover(
            true,
            false,
            true,
            SchedulerOwnerProcessEvidence::IdentityMismatch) ==
        SchedulerTakeoverDecision::TakeOverExpiredDeadOwner);
    assert(
        DecideSchedulerTakeover(
            true,
            false,
            false,
            SchedulerOwnerProcessEvidence::Missing) ==
        SchedulerTakeoverDecision::RejectFreshLease);
    assert(
        DecideSchedulerTakeover(
            true,
            false,
            true,
            SchedulerOwnerProcessEvidence::Valid) ==
        SchedulerTakeoverDecision::RejectValidOwner);
    assert(
        DecideSchedulerTakeover(
            true,
            false,
            true,
            SchedulerOwnerProcessEvidence::Ambiguous) ==
        SchedulerTakeoverDecision::RejectAmbiguousOwner);

    for (const std::string state :
         {"reserved", "spawned", "running", "observed",
          "identity_ambiguous"})
    {
        assert(WorkerAttemptConsumesCapacity(state));
        assert(!WorkerAttemptIsTerminal(state));
    }
    for (const std::string state :
         {"completed", "failed", "launch_failed", "abandoned"})
    {
        assert(!WorkerAttemptConsumesCapacity(state));
        assert(WorkerAttemptIsTerminal(state));
    }
    assert(!WorkerAttemptConsumesCapacity("stopped"));
    assert(!WorkerAttemptIsTerminal("stopped"));
    assert(!WorkerAttemptConsumesCapacity("unsupported"));
    assert(!WorkerAttemptIsTerminal("unsupported"));

    const std::string executable = ResolveCanonicalExecutablePath();
    assert(!executable.empty());
    assert(executable.front() == '/');
    assert(std::filesystem::path(executable).is_absolute());
    assert(std::filesystem::exists(executable));
    assert(std::filesystem::canonical(executable) == executable);

    std::cout << "CANONICAL_EXECUTABLE_PATH=" << executable << "\n";
    std::cout << "SchedulerOwnershipPolicyTests passed\n";
    return 0;
}
