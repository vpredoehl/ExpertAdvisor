#pragma once

#include "SchedulerAuthorityRepository.hpp"

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace EA::SchedulerCore
{

inline constexpr int kSchedulerProtocolGeneration = 52;
inline constexpr int kSchedulerLeaseSeconds = 90;

struct SchedulerAuthorityContext
{
    std::string schedulerInvocationId;
    std::string invocationNonce;
    long long fencingToken = 0;
    std::string canonicalExecutablePath;
    bool held = false;

    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] SchedulerAuthorityIdentity identity() const;
};

class SchedulerAuthorityLost final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

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

struct SchedulerOwnerProcessObservation
{
    bool exists = false;
    bool inspectionSucceeded = false;
    int processPid = -1;
    int processGroupId = -1;
    std::string processStartIdentity;
    std::string executablePath;
    std::string commandLine;
};

SchedulerTakeoverDecision DecideSchedulerTakeover(
    bool hasOwner,
    bool explicitlyReleased,
    bool leaseExpired,
    SchedulerOwnerProcessEvidence processEvidence) noexcept;
std::string GenerateSchedulerIdentityNonce();
const char* SchedulerOwnerProcessEvidenceText(
    SchedulerOwnerProcessEvidence evidence) noexcept;
const char* SchedulerTakeoverDecisionText(
    SchedulerTakeoverDecision decision) noexcept;
SchedulerOwnerProcessEvidence EvaluateSchedulerOwnerProcess(
    int expectedProcessPid,
    int expectedProcessGroupId,
    const std::string& expectedProcessStartIdentity,
    const std::string& expectedCanonicalExecutablePath,
    const SchedulerOwnerProcessObservation& observation);

class SchedulerOwnerProcessInspector
{
public:
    virtual ~SchedulerOwnerProcessInspector() = default;
    virtual SchedulerOwnerProcessEvidence inspectOwner(
        int processPid,
        int processGroupId,
        const std::string& processStartIdentity,
        const std::string& canonicalExecutablePath) = 0;
};

struct SchedulerAuthorityAcquisitionRequest
{
    int processPid = -1;
    int processGroupId = -1;
    std::string processStartIdentity;
    std::string canonicalExecutablePath;
    std::string commandLine;
};

struct SchedulerAuthorityAcquisitionResult
{
    SchedulerAuthorityContext authority;
    bool protocolAccepted = false;
    int databaseProtocolGeneration = 0;
    std::string databaseCutoverState = "missing";
    std::string protocolFailureReason =
        "explicit_safe_cutover_required";
    std::optional<std::string> previousLeaseOwner;
    long long previousFencingToken = 0;
    SchedulerTakeoverDecision decision =
        SchedulerTakeoverDecision::RejectAmbiguousOwner;

    [[nodiscard]] bool acquired() const noexcept;
};

enum class SchedulerProtocolCutoverResult
{
    Completed,
    AlreadyComplete,
    GenerationMismatch
};

struct SchedulerProtocolCutoverRequest
{
    std::string canonicalExecutablePath;
    std::string processEvidence;
};

class SchedulerAuthorityService final
{
public:
    SchedulerAuthorityService(
        SchedulerAuthorityRepository& repository,
        SchedulerOwnerProcessInspector& processInspector);

    SchedulerAuthorityAcquisitionResult acquire(
        const SchedulerAuthorityAcquisitionRequest& request);
    void requireAndRenew(const SchedulerAuthorityContext& authority);
    bool release(
        const SchedulerAuthorityContext& authority,
        std::string_view reason);
    SchedulerProtocolCutoverResult completeProtocolCutover(
        const SchedulerProtocolCutoverRequest& request,
        const std::function<std::string()>& resolveActor);
    void displaceForTest(
        const SchedulerAuthorityContext& authority,
        std::string_view foreignSchedulerInvocationId,
        std::string_view boundary);

private:
    SchedulerAuthorityRepository& repository_;
    SchedulerOwnerProcessInspector& processInspector_;
};

} // namespace EA::SchedulerCore
