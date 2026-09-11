#include "../Sources/SchedulerCore/SchedulerAuthorityService.hpp"

#include <cassert>
#include <filesystem>
#include <optional>
#include <string>

namespace
{

class MemoryAuthorityRepository final
    : public EA::SchedulerCore::SchedulerAuthorityRepository
{
public:
    std::optional<EA::SchedulerCore::SchedulerProtocolState> protocol =
        EA::SchedulerCore::SchedulerProtocolState{
            EA::SchedulerCore::kSchedulerProtocolGeneration,
            "complete",
            std::nullopt};
    std::optional<EA::SchedulerCore::SchedulerLeaseState> lease =
        EA::SchedulerCore::SchedulerLeaseState{
            std::nullopt,
            4,
            "vacant",
            true,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt};
    bool acquireLeaseResult = true;
    bool renewResult = true;
    bool releaseResult = true;
    bool cutoverResult = true;
    bool displaceResult = true;
    int coordinationLocks = 0;
    int renewals = 0;
    int touches = 0;
    EA::SchedulerCore::SchedulerInvocationRecord registered;
    EA::SchedulerCore::SchedulerLeaseAcquisition acquired;
    std::string rejectedReason;
    std::string crashedInvocation;
    std::string releasedInvocation;
    EA::SchedulerCore::SchedulerProtocolCutoverUpdate cutover;
    std::string displacedForeignInvocation;
    std::string displacedReason;

    void acquireAuthorityCoordinationLock() override
    {
        ++coordinationLocks;
    }

    std::optional<EA::SchedulerCore::SchedulerProtocolState>
    loadSchedulerProtocolForUpdate() override
    {
        return protocol;
    }

    void registerSchedulerInvocation(
        const EA::SchedulerCore::SchedulerInvocationRecord& value) override
    {
        registered = value;
    }

    std::optional<EA::SchedulerCore::SchedulerLeaseState>
    loadSchedulerLeaseForUpdate() override
    {
        return lease;
    }

    void rejectSchedulerInvocation(
        std::string_view,
        std::string_view reason) override
    {
        rejectedReason = reason;
    }

    void markSchedulerInvocationCrashed(
        std::string_view invocation,
        std::string_view) override
    {
        crashedInvocation = invocation;
    }

    bool acquireSchedulerLease(
        const EA::SchedulerCore::SchedulerLeaseAcquisition& value) override
    {
        acquired = value;
        return acquireLeaseResult;
    }

    void markSchedulerInvocationOwner(std::string_view) override {}

    bool renewSchedulerLease(
        const EA::SchedulerCore::SchedulerAuthorityIdentity&,
        int) override
    {
        ++renewals;
        return renewResult;
    }

    void touchSchedulerInvocation(std::string_view) override
    {
        ++touches;
    }

    bool releaseSchedulerLease(
        const EA::SchedulerCore::SchedulerAuthorityIdentity&,
        std::string_view) override
    {
        return releaseResult;
    }

    void markSchedulerInvocationReleased(
        std::string_view invocation,
        std::string_view) override
    {
        releasedInvocation = invocation;
    }

    bool completeSchedulerProtocolCutover(
        const EA::SchedulerCore::SchedulerProtocolCutoverUpdate& value) override
    {
        cutover = value;
        return cutoverResult;
    }

    bool displaceSchedulerAuthorityForTest(
        const EA::SchedulerCore::SchedulerAuthorityIdentity&,
        std::string_view foreignInvocation,
        std::string_view reason,
        int) override
    {
        displacedForeignInvocation = foreignInvocation;
        displacedReason = reason;
        return displaceResult;
    }
};

class FixedProcessInspector final
    : public EA::SchedulerCore::SchedulerOwnerProcessInspector
{
public:
    EA::SchedulerCore::SchedulerOwnerProcessEvidence evidence =
        EA::SchedulerCore::SchedulerOwnerProcessEvidence::Ambiguous;
    int inspections = 0;

    EA::SchedulerCore::SchedulerOwnerProcessEvidence inspectOwner(
        int,
        int,
        const std::string&,
        const std::string&) override
    {
        ++inspections;
        return evidence;
    }
};

EA::SchedulerCore::SchedulerAuthorityAcquisitionRequest Request()
{
    return {123, 123, "start-identity", "/tmp/LSTM_Release",
            "/tmp/LSTM_Release --schedule-experiments"};
}

template <typename Function>
bool ThrowsAuthorityReason(Function&& function, const std::string& reason)
{
    try
    {
        function();
    }
    catch (const EA::SchedulerCore::SchedulerAuthorityLost& error)
    {
        return error.what() == reason;
    }
    return false;
}

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    MemoryAuthorityRepository repository;
    FixedProcessInspector inspector;
    SchedulerAuthorityService authority{repository, inspector};

    repository.protocol->requiredGeneration =
        kSchedulerProtocolGeneration - 1;
    const auto protocolRejected = authority.acquire(Request());
    assert(!protocolRejected.protocolAccepted);
    assert(!protocolRejected.acquired());
    assert(repository.registered.schedulerInvocationId.empty());
    assert(repository.coordinationLocks == 1);

    repository.protocol->requiredGeneration = kSchedulerProtocolGeneration;
    const auto acquired = authority.acquire(Request());
    assert(acquired.protocolAccepted && acquired.acquired());
    assert(acquired.authority.schedulerInvocationId.starts_with("scheduler:"));
    assert(acquired.authority.invocationNonce.size() == 48);
    assert(acquired.authority.fencingToken == 5);
    assert(repository.registered.schedulerInvocationId ==
           acquired.authority.schedulerInvocationId);
    assert(repository.acquired.fencingToken == 5);
    assert(repository.acquired.leaseSeconds == kSchedulerLeaseSeconds);
    assert(repository.acquired.transitionReason == "vacant");
    assert(inspector.inspections == 0);

    repository.lease = SchedulerLeaseState{
        "scheduler-current",
        10,
        "active",
        false,
        321,
        321,
        "current-start",
        "/tmp/LSTM_Release"};
    inspector.evidence = SchedulerOwnerProcessEvidence::Missing;
    const auto freshRejected = authority.acquire(Request());
    assert(!freshRejected.acquired());
    assert(freshRejected.decision ==
           SchedulerTakeoverDecision::RejectFreshLease);
    assert(repository.rejectedReason == "owner_lease_valid");
    assert(inspector.inspections == 1);

    repository.lease->expired = true;
    inspector.evidence = SchedulerOwnerProcessEvidence::Valid;
    const auto liveRejected = authority.acquire(Request());
    assert(!liveRejected.acquired());
    assert(liveRejected.decision ==
           SchedulerTakeoverDecision::RejectValidOwner);

    inspector.evidence = SchedulerOwnerProcessEvidence::IdentityMismatch;
    const auto takeover = authority.acquire(Request());
    assert(takeover.acquired());
    assert(takeover.authority.fencingToken == 11);
    assert(repository.crashedInvocation == "scheduler-current");
    assert(repository.acquired.transitionReason ==
           "expired_and_owner_identity_invalid");

    repository.lease = SchedulerLeaseState{
        takeover.authority.schedulerInvocationId,
        takeover.authority.fencingToken,
        "active",
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    authority.requireAndRenew(takeover.authority);
    assert(repository.renewals == 1 && repository.touches == 1);

    repository.lease->fencingToken += 1;
    assert(ThrowsAuthorityReason(
        [&] { authority.requireAndRenew(takeover.authority); },
        "scheduler_lease_owner_or_fence_mismatch"));
    repository.lease->fencingToken = takeover.authority.fencingToken;
    repository.renewResult = false;
    assert(ThrowsAuthorityReason(
        [&] { authority.requireAndRenew(takeover.authority); },
        "scheduler_lease_refresh_rejected"));

    repository.releaseResult = true;
    assert(authority.release(takeover.authority, "scheduler_exit"));
    assert(repository.releasedInvocation ==
           takeover.authority.schedulerInvocationId);
    repository.releaseResult = false;
    repository.releasedInvocation.clear();
    assert(!authority.release(takeover.authority, "stale_release"));
    assert(repository.releasedInvocation.empty());

    repository.protocol = SchedulerProtocolState{
        kSchedulerProtocolGeneration, "pending", std::nullopt};
    assert(authority.completeProtocolCutover(
               {"/tmp/LSTM_Release", "ps_complete"},
               [] { return "pid:1;start:x"; }) ==
           SchedulerProtocolCutoverResult::Completed);
    assert(repository.cutover.protocolGeneration ==
           kSchedulerProtocolGeneration);
    repository.protocol->cutoverState = "complete";
    bool resolvedUnusedActor = false;
    assert(authority.completeProtocolCutover(
               {}, [&] {
                   resolvedUnusedActor = true;
                   return "unused";
               }) ==
           SchedulerProtocolCutoverResult::AlreadyComplete);
    assert(!resolvedUnusedActor);

    authority.displaceForTest(
        takeover.authority, "scheduler-restarted", "after_reservation");
    assert(repository.displacedForeignInvocation == "scheduler-restarted");
    assert(repository.displacedReason == "test_failpoint:after_reservation");

    const std::string shell = std::filesystem::canonical("/bin/sh").string();
    const SchedulerOwnerProcessObservation valid{
        true, true, 22, 22, "start", shell,
        shell + " --schedule-experiments"};
    assert(EvaluateSchedulerOwnerProcess(22, 22, "start", shell, valid) ==
           SchedulerOwnerProcessEvidence::Valid);
    auto mismatch = valid;
    mismatch.commandLine = shell + " --train";
    assert(EvaluateSchedulerOwnerProcess(22, 22, "start", shell, mismatch) ==
           SchedulerOwnerProcessEvidence::IdentityMismatch);
    auto missing = valid;
    missing.exists = false;
    assert(EvaluateSchedulerOwnerProcess(22, 22, "start", shell, missing) ==
           SchedulerOwnerProcessEvidence::Missing);
    missing.inspectionSucceeded = false;
    assert(EvaluateSchedulerOwnerProcess(22, 22, "start", shell, missing) ==
           SchedulerOwnerProcessEvidence::Ambiguous);

    return 0;
}
