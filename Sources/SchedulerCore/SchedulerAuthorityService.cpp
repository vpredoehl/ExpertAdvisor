#include "SchedulerAuthorityService.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>

namespace EA::SchedulerCore
{
namespace
{

bool IsAcquisitionDecision(SchedulerTakeoverDecision decision) noexcept
{
    return decision == SchedulerTakeoverDecision::AcquireVacant ||
           decision == SchedulerTakeoverDecision::AcquireReleased ||
           decision ==
               SchedulerTakeoverDecision::TakeOverExpiredDeadOwner;
}

std::string CanonicalizeObservedExecutable(const std::string& executable)
{
    if (executable.empty())
        return {};
    char* resolved = ::realpath(executable.c_str(), nullptr);
    if (resolved == nullptr)
        return {};
    std::string canonical{resolved};
    std::free(resolved);
    return canonical;
}

} // namespace

std::string GenerateSchedulerIdentityNonce()
{
    std::array<unsigned char, 24> bytes{};
    std::random_device random;
    for (unsigned char& value : bytes)
        value = static_cast<unsigned char>(random());
    const auto now = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
    for (size_t index = 0;
         index < sizeof(now) && index < bytes.size();
         ++index)
    {
        bytes[index] ^= static_cast<unsigned char>(
            (static_cast<unsigned long long>(now) >> (index * 8)) &
            0xffU);
    }

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char value : bytes)
        output << std::setw(2) << static_cast<unsigned int>(value);
    return output.str();
}

bool SchedulerAuthorityContext::complete() const noexcept
{
    return held && !schedulerInvocationId.empty() && fencingToken > 0 &&
           !canonicalExecutablePath.empty();
}

SchedulerAuthorityIdentity SchedulerAuthorityContext::identity() const
{
    return {schedulerInvocationId, fencingToken};
}

SchedulerTakeoverDecision DecideSchedulerTakeover(
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

const char* SchedulerOwnerProcessEvidenceText(
    SchedulerOwnerProcessEvidence evidence) noexcept
{
    switch (evidence)
    {
        case SchedulerOwnerProcessEvidence::Valid: return "validated";
        case SchedulerOwnerProcessEvidence::Missing: return "process_missing";
        case SchedulerOwnerProcessEvidence::IdentityMismatch:
            return "identity_mismatch";
        case SchedulerOwnerProcessEvidence::Ambiguous:
            return "inspection_ambiguous";
    }
    return "inspection_ambiguous";
}

const char* SchedulerTakeoverDecisionText(
    SchedulerTakeoverDecision decision) noexcept
{
    switch (decision)
    {
        case SchedulerTakeoverDecision::AcquireVacant:
            return "vacant";
        case SchedulerTakeoverDecision::AcquireReleased:
            return "explicitly_released";
        case SchedulerTakeoverDecision::TakeOverExpiredDeadOwner:
            return "expired_and_owner_identity_invalid";
        case SchedulerTakeoverDecision::RejectValidOwner:
            return "expired_but_owner_identity_valid";
        case SchedulerTakeoverDecision::RejectFreshLease:
            return "owner_lease_valid";
        case SchedulerTakeoverDecision::RejectAmbiguousOwner:
            return "owner_identity_ambiguous";
    }
    return "unknown";
}

SchedulerOwnerProcessEvidence EvaluateSchedulerOwnerProcess(
    int expectedProcessPid,
    int expectedProcessGroupId,
    const std::string& expectedProcessStartIdentity,
    const std::string& expectedCanonicalExecutablePath,
    const SchedulerOwnerProcessObservation& observation)
{
    if (observation.inspectionSucceeded && !observation.exists)
        return SchedulerOwnerProcessEvidence::Missing;
    if (!observation.inspectionSucceeded || !observation.exists)
        return SchedulerOwnerProcessEvidence::Ambiguous;

    const std::string observedExecutable =
        CanonicalizeObservedExecutable(observation.executablePath);
    if (observation.processPid != expectedProcessPid ||
        observation.processGroupId != expectedProcessGroupId ||
        observation.processStartIdentity != expectedProcessStartIdentity ||
        observedExecutable.empty() ||
        observedExecutable != expectedCanonicalExecutablePath ||
        observation.commandLine.find("--schedule-experiments") ==
            std::string::npos)
    {
        return SchedulerOwnerProcessEvidence::IdentityMismatch;
    }
    return SchedulerOwnerProcessEvidence::Valid;
}

bool SchedulerAuthorityAcquisitionResult::acquired() const noexcept
{
    return authority.complete();
}

SchedulerAuthorityService::SchedulerAuthorityService(
    SchedulerAuthorityRepository& repository,
    SchedulerOwnerProcessInspector& processInspector)
    : repository_{repository}, processInspector_{processInspector}
{
}

SchedulerAuthorityAcquisitionResult SchedulerAuthorityService::acquire(
    const SchedulerAuthorityAcquisitionRequest& request)
{
    if (request.processStartIdentity.empty())
    {
        throw std::runtime_error(
            "scheduler_process_start_identity_unavailable");
    }

    SchedulerAuthorityAcquisitionResult result;
    result.authority.invocationNonce = GenerateSchedulerIdentityNonce();
    result.authority.schedulerInvocationId =
        "scheduler:" + result.authority.invocationNonce;
    result.authority.canonicalExecutablePath =
        request.canonicalExecutablePath;

    repository_.acquireAuthorityCoordinationLock();
    const auto protocol = repository_.loadSchedulerProtocolForUpdate();
    if (protocol)
    {
        result.databaseProtocolGeneration = protocol->requiredGeneration;
        result.databaseCutoverState = protocol->cutoverState;
        if (protocol->failureDiagnostic)
            result.protocolFailureReason = *protocol->failureDiagnostic;
    }
    result.protocolAccepted =
        protocol &&
        protocol->requiredGeneration == kSchedulerProtocolGeneration &&
        protocol->cutoverState == "complete";
    if (!result.protocolAccepted)
        return result;

    repository_.registerSchedulerInvocation({
        result.authority.schedulerInvocationId,
        request.processPid,
        request.processGroupId,
        request.processStartIdentity,
        request.canonicalExecutablePath,
        request.commandLine,
        result.authority.invocationNonce,
        kSchedulerProtocolGeneration});

    const auto lease = repository_.loadSchedulerLeaseForUpdate();
    if (!lease)
        throw std::runtime_error("scheduler_lease_singleton_missing");
    result.previousLeaseOwner = lease->ownerSchedulerInvocationId;
    result.previousFencingToken = lease->fencingToken;
    const bool hasOwner = lease->ownerSchedulerInvocationId.has_value();
    const bool explicitlyReleased =
        lease->authorityState == "released" ||
        lease->authorityState == "vacant";

    SchedulerOwnerProcessEvidence ownerEvidence =
        SchedulerOwnerProcessEvidence::Ambiguous;
    if (!hasOwner)
        ownerEvidence = SchedulerOwnerProcessEvidence::Missing;
    else if (!explicitlyReleased && lease->ownerProcessPid &&
             lease->ownerProcessGroupId &&
             lease->ownerProcessStartIdentity &&
             lease->ownerCanonicalExecutablePath)
    {
        ownerEvidence = processInspector_.inspectOwner(
            *lease->ownerProcessPid,
            *lease->ownerProcessGroupId,
            *lease->ownerProcessStartIdentity,
            *lease->ownerCanonicalExecutablePath);
    }

    result.decision = DecideSchedulerTakeover(
        hasOwner,
        explicitlyReleased,
        lease->expired,
        ownerEvidence);
    if (!IsAcquisitionDecision(result.decision))
    {
        repository_.rejectSchedulerInvocation(
            result.authority.schedulerInvocationId,
            SchedulerTakeoverDecisionText(result.decision));
        return result;
    }

    if (lease->ownerSchedulerInvocationId &&
        *lease->ownerSchedulerInvocationId !=
            result.authority.schedulerInvocationId)
    {
        repository_.markSchedulerInvocationCrashed(
            *lease->ownerSchedulerInvocationId,
            SchedulerTakeoverDecisionText(result.decision));
    }

    result.authority.fencingToken = lease->fencingToken + 1;
    if (!repository_.acquireSchedulerLease({
            result.authority.schedulerInvocationId,
            result.authority.fencingToken,
            kSchedulerLeaseSeconds,
            SchedulerTakeoverDecisionText(result.decision)}))
    {
        throw std::runtime_error("scheduler_lease_acquire_failed");
    }
    repository_.markSchedulerInvocationOwner(
        result.authority.schedulerInvocationId);
    result.authority.held = true;
    return result;
}

void SchedulerAuthorityService::requireAndRenew(
    const SchedulerAuthorityContext& authority)
{
    if (!authority.complete())
        throw SchedulerAuthorityLost("scheduler_authority_not_held");

    repository_.acquireAuthorityCoordinationLock();
    const auto protocol = repository_.loadSchedulerProtocolForUpdate();
    if (!protocol ||
        protocol->requiredGeneration != kSchedulerProtocolGeneration ||
        protocol->cutoverState != "complete")
    {
        throw SchedulerAuthorityLost(
            "scheduler_protocol_cutover_not_complete");
    }

    const auto lease = repository_.loadSchedulerLeaseForUpdate();
    if (!lease || lease->authorityState != "active" ||
        lease->ownerSchedulerInvocationId !=
            std::optional<std::string>{authority.schedulerInvocationId} ||
        lease->fencingToken != authority.fencingToken)
    {
        throw SchedulerAuthorityLost(
            "scheduler_lease_owner_or_fence_mismatch");
    }

    if (!repository_.renewSchedulerLease(
            authority.identity(), kSchedulerLeaseSeconds))
    {
        throw SchedulerAuthorityLost("scheduler_lease_refresh_rejected");
    }
    repository_.touchSchedulerInvocation(authority.schedulerInvocationId);
}

bool SchedulerAuthorityService::release(
    const SchedulerAuthorityContext& authority,
    std::string_view reason)
{
    if (!authority.held)
        return false;
    const bool released = repository_.releaseSchedulerLease(
        authority.identity(), reason);
    if (released)
    {
        repository_.markSchedulerInvocationReleased(
            authority.schedulerInvocationId, reason);
    }
    return released;
}

SchedulerProtocolCutoverResult
SchedulerAuthorityService::completeProtocolCutover(
    const SchedulerProtocolCutoverRequest& request,
    const std::function<std::string()>& resolveActor)
{
    repository_.acquireAuthorityCoordinationLock();
    const auto protocol = repository_.loadSchedulerProtocolForUpdate();
    if (!protocol ||
        protocol->requiredGeneration != kSchedulerProtocolGeneration)
    {
        return SchedulerProtocolCutoverResult::GenerationMismatch;
    }
    if (protocol->cutoverState == "complete")
        return SchedulerProtocolCutoverResult::AlreadyComplete;
    if (!repository_.completeSchedulerProtocolCutover({
            kSchedulerProtocolGeneration,
            resolveActor(),
            request.canonicalExecutablePath,
            request.processEvidence}))
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "complete_scheduler_protocol_cutover:affected_rows=0");
    }
    return SchedulerProtocolCutoverResult::Completed;
}

void SchedulerAuthorityService::displaceForTest(
    const SchedulerAuthorityContext& authority,
    std::string_view foreignSchedulerInvocationId,
    std::string_view boundary)
{
    if (!repository_.displaceSchedulerAuthorityForTest(
            authority.identity(),
            foreignSchedulerInvocationId,
            "test_failpoint:" + std::string{boundary},
            kSchedulerLeaseSeconds))
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "inject_scheduler_authority_loss_" +
            std::string{boundary} + ":affected_rows=0");
    }
}

} // namespace EA::SchedulerCore
