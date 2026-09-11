#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace EA::SchedulerCore
{

struct SchedulerProtocolState
{
    int requiredGeneration = 0;
    std::string cutoverState;
    std::optional<std::string> failureDiagnostic;
};

struct SchedulerInvocationRecord
{
    std::string schedulerInvocationId;
    int processPid = -1;
    int processGroupId = -1;
    std::string processStartIdentity;
    std::string canonicalExecutablePath;
    std::string commandLine;
    std::string invocationNonce;
    int protocolGeneration = 0;
};

struct SchedulerLeaseState
{
    std::optional<std::string> ownerSchedulerInvocationId;
    long long fencingToken = 0;
    std::string authorityState;
    bool expired = false;
    std::optional<int> ownerProcessPid;
    std::optional<int> ownerProcessGroupId;
    std::optional<std::string> ownerProcessStartIdentity;
    std::optional<std::string> ownerCanonicalExecutablePath;
};

struct SchedulerLeaseAcquisition
{
    std::string schedulerInvocationId;
    long long fencingToken = 0;
    int leaseSeconds = 0;
    std::string transitionReason;
};

struct SchedulerAuthorityIdentity
{
    std::string schedulerInvocationId;
    long long fencingToken = 0;
};

struct SchedulerProtocolCutoverUpdate
{
    int protocolGeneration = 0;
    std::string actor;
    std::string canonicalExecutablePath;
    std::string processEvidence;
};

// Transaction-scoped persistence needed by scheduler authority.  This
// interface intentionally contains domain values only; database-library types
// remain private to the production adapter.
class SchedulerAuthorityRepository
{
public:
    virtual ~SchedulerAuthorityRepository() = default;

    virtual void acquireAuthorityCoordinationLock() = 0;
    virtual std::optional<SchedulerProtocolState>
    loadSchedulerProtocolForUpdate() = 0;
    virtual void registerSchedulerInvocation(
        const SchedulerInvocationRecord& invocation) = 0;
    virtual std::optional<SchedulerLeaseState>
    loadSchedulerLeaseForUpdate() = 0;
    virtual void rejectSchedulerInvocation(
        std::string_view schedulerInvocationId,
        std::string_view terminalReason) = 0;
    virtual void markSchedulerInvocationCrashed(
        std::string_view schedulerInvocationId,
        std::string_view terminalReason) = 0;
    virtual bool acquireSchedulerLease(
        const SchedulerLeaseAcquisition& acquisition) = 0;
    virtual void markSchedulerInvocationOwner(
        std::string_view schedulerInvocationId) = 0;
    virtual bool renewSchedulerLease(
        const SchedulerAuthorityIdentity& authority,
        int leaseSeconds) = 0;
    virtual void touchSchedulerInvocation(
        std::string_view schedulerInvocationId) = 0;
    virtual bool releaseSchedulerLease(
        const SchedulerAuthorityIdentity& authority,
        std::string_view reason) = 0;
    virtual void markSchedulerInvocationReleased(
        std::string_view schedulerInvocationId,
        std::string_view reason) = 0;
    virtual bool completeSchedulerProtocolCutover(
        const SchedulerProtocolCutoverUpdate& update) = 0;
    virtual bool displaceSchedulerAuthorityForTest(
        const SchedulerAuthorityIdentity& authority,
        std::string_view foreignSchedulerInvocationId,
        std::string_view transitionReason,
        int leaseSeconds) = 0;
};

} // namespace EA::SchedulerCore
