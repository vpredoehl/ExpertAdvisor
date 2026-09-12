#include "WorkerAttemptLifecycleService.hpp"

#include "ExperimentCurrentOperation.hpp"
#include "SchedulerAuthorityService.hpp"

#include <stdexcept>

namespace EA::SchedulerCore
{
namespace
{

std::optional<ReservedWorkerAttempt> RequireReservation(
    WorkerAttemptReservationResult result,
    const char* insertFailure,
    const char* claimFailure)
{
    switch (result.status)
    {
        case WorkerAttemptReservationStatus::Reserved:
            if (!result.attempt)
                throw std::runtime_error(insertFailure);
            return std::move(result.attempt);
        case WorkerAttemptReservationStatus::LifecycleUnavailable:
            return std::nullopt;
        case WorkerAttemptReservationStatus::ReservationInsertFailed:
            throw std::runtime_error(insertFailure);
        case WorkerAttemptReservationStatus::LifecycleClaimFailed:
            throw std::runtime_error(claimFailure);
    }
    throw std::runtime_error(insertFailure);
}

} // namespace

WorkerAttemptLifecycleService::WorkerAttemptLifecycleService(
    SchedulerRepository& repository,
    WorkerAttemptLifecycleContext context)
    : repository_{repository}, context_{std::move(context)}
{
    if (context_.schedulerInvocationId.empty() ||
        context_.schedulerFencingToken <= 0 ||
        context_.canonicalExecutablePath.empty())
    {
        throw std::invalid_argument(
            "complete scheduler authority context required");
    }
}

std::string WorkerAttemptLifecycleService::launchIdentity(
    std::string_view commandIdentity,
    std::string_view nonce) const
{
    if (commandIdentity.empty() || nonce.empty())
        throw std::invalid_argument("worker launch identity components required");
    return context_.schedulerInvocationId + ":worker:" +
           std::string{nonce} + ":" + std::string{commandIdentity};
}

std::optional<ReservedWorkerAttempt>
WorkerAttemptLifecycleService::reserveExperiment(
    const ExperimentWorkerAttemptRequest& request)
{
    if (request.experimentId <= 0 || request.phase.empty() ||
        request.logPath.empty())
    {
        throw std::invalid_argument(
            "complete experiment worker reservation required");
    }
    const std::string commandIdentity =
        "experiment:" + std::to_string(request.experimentId) + ":" +
        request.phase;
    return RequireReservation(
        repository_.reserveExperimentWorkerAttempt({
            launchIdentity(commandIdentity, request.identityNonce),
            context_.schedulerInvocationId,
            context_.schedulerFencingToken,
            request.experimentId,
            request.phase,
            context_.canonicalExecutablePath,
            commandIdentity,
            EA::ExperimentLifecycle::
                RequireCanonicalCurrentOperationForPhase(request.phase),
            request.logPath,
            request.cancellationOnly}),
        "worker_attempt_reservation_insert_failed",
        "worker_attempt_lifecycle_claim_failed");
}

std::optional<ReservedWorkerAttempt>
WorkerAttemptLifecycleService::reserveCheckpoint(
    const CheckpointWorkerAttemptRequest& request)
{
    if (request.experimentId <= 0 || request.checkpointEvalId <= 0 ||
        request.logPath.empty())
    {
        throw std::invalid_argument(
            "complete checkpoint worker reservation required");
    }
    const std::string commandIdentity =
        "checkpoint_infer:" + std::to_string(request.checkpointEvalId);
    return RequireReservation(
        repository_.reserveCheckpointWorkerAttempt({
            launchIdentity(commandIdentity, request.identityNonce),
            context_.schedulerInvocationId,
            context_.schedulerFencingToken,
            request.experimentId,
            request.checkpointEvalId,
            context_.canonicalExecutablePath,
            commandIdentity,
            request.logPath}),
        "checkpoint_worker_attempt_reservation_insert_failed",
        "checkpoint_worker_attempt_lifecycle_claim_failed");
}

void WorkerAttemptLifecycleService::recordSpawned(
    const SpawnedWorkerAttemptEvidence& evidence)
{
    const auto result = repository_.persistSpawnedWorkerAttempt({
        evidence.attempt.workerAttemptId,
        context_.schedulerInvocationId,
        context_.schedulerFencingToken,
        evidence.attempt.experimentId,
        evidence.attempt.checkpointEvalId,
        evidence.attempt.phase,
        evidence.workerPid,
        evidence.processStartIdentity,
        context_.canonicalExecutablePath,
        evidence.commandLine});
    if (result == SpawnPersistenceResult::AttemptPreconditionRejected)
    {
        if (evidence.requireSchedulerAuthority)
            throw SchedulerAuthorityLost(
                "worker_attempt_spawn_persistence_fence_rejected");
        throw std::runtime_error(
            "child_spawn_evidence_persistence_fence_rejected");
    }
    if (result == SpawnPersistenceResult::LifecyclePreconditionRejected)
        throw std::runtime_error(
            "worker_attempt_spawn_lifecycle_predicate_rejected");
}

void WorkerAttemptLifecycleService::recordLaunchFailure(
    const WorkerAttemptLaunchFailure& failure)
{
    const auto result = repository_.persistWorkerAttemptLaunchFailure({
        failure.attempt.workerAttemptId,
        context_.schedulerInvocationId,
        context_.schedulerFencingToken,
        failure.attempt.experimentId,
        failure.attempt.checkpointEvalId,
        failure.attempt.phase,
        failure.exitCode,
        failure.diagnostic});
    if (result == LaunchFailurePersistenceResult::AttemptPreconditionRejected)
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "terminalize_exact_launch_failure_attempt:affected_rows=0");
    }
    if (result == LaunchFailurePersistenceResult::LifecyclePreconditionRejected)
    {
        throw std::runtime_error(
            "exact_attempt_predicate_rejected:"
            "clear_exact_launch_failure_binding:affected_rows=0");
    }
}

} // namespace EA::SchedulerCore
