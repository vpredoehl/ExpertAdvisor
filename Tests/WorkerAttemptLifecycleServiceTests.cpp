#include "SchedulerCore/WorkerAttemptLifecycleService.hpp"

#include "SchedulerCore/SchedulerAuthorityService.hpp"

#include <cassert>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

class MemoryRepository final : public EA::SchedulerCore::SchedulerRepository
{
public:
    EA::SchedulerCore::WorkerAttemptReservationResult reservationResult;
    EA::SchedulerCore::SpawnPersistenceResult spawnResult =
        EA::SchedulerCore::SpawnPersistenceResult::Updated;
    EA::SchedulerCore::LaunchFailurePersistenceResult failureResult =
        EA::SchedulerCore::LaunchFailurePersistenceResult::Updated;
    std::optional<EA::SchedulerCore::ExperimentWorkerAttemptReservation>
        experimentReservation;
    std::optional<EA::SchedulerCore::CheckpointWorkerAttemptReservation>
        checkpointReservation;
    std::optional<EA::SchedulerCore::SpawnedWorkerAttemptUpdate> spawned;
    std::optional<EA::SchedulerCore::WorkerAttemptLaunchFailureUpdate>
        launchFailure;

    std::vector<EA::SchedulerCore::PendingSchedulerExperimentRecord>
    loadPendingExperiments(std::string_view, bool) override
    {
        return {};
    }

    std::vector<EA::SchedulerCore::RunningSchedulerExperimentRecord>
    loadRunningExperiments() override
    {
        return {};
    }

    EA::SchedulerCore::SchedulerQueueSnapshot loadQueueSnapshot() override
    {
        return {};
    }

    int countWorkersConsumingCapacity(std::string_view) override
    {
        return 0;
    }

    std::optional<EA::SchedulerCore::PreemptionVictimRecord>
    loadPreemptionVictim(std::string_view, int) override
    {
        return std::nullopt;
    }

    EA::SchedulerCore::WorkerAttemptReservationResult
    reserveExperimentWorkerAttempt(
        const EA::SchedulerCore::ExperimentWorkerAttemptReservation& request)
        override
    {
        experimentReservation = request;
        return reservationResult;
    }

    EA::SchedulerCore::WorkerAttemptReservationResult
    reserveCheckpointWorkerAttempt(
        const EA::SchedulerCore::CheckpointWorkerAttemptReservation& request)
        override
    {
        checkpointReservation = request;
        return reservationResult;
    }

    EA::SchedulerCore::WorkerAttemptReservationResult
    reserveCheckpointAnalysisAttempt(
        const EA::SchedulerCore::CheckpointAnalysisAttemptReservation&) override
    {
        return {};
    }

    EA::SchedulerCore::SpawnPersistenceResult persistSpawnedWorkerAttempt(
        const EA::SchedulerCore::SpawnedWorkerAttemptUpdate& update) override
    {
        spawned = update;
        return spawnResult;
    }

    EA::SchedulerCore::LaunchFailurePersistenceResult
    persistWorkerAttemptLaunchFailure(
        const EA::SchedulerCore::WorkerAttemptLaunchFailureUpdate& update)
        override
    {
        launchFailure = update;
        return failureResult;
    }

    bool persistCheckpointAnalysisCompletion(
        const EA::SchedulerCore::CheckpointAnalysisCompletionUpdate&) override
    {
        return true;
    }

    EA::SchedulerCore::CheckpointAnalysisPersistenceResult
    persistCheckpointAnalysisTerminalState(
        const EA::SchedulerCore::CheckpointAnalysisTerminalUpdate&) override
    {
        return EA::SchedulerCore::CheckpointAnalysisPersistenceResult::Updated;
    }

    std::optional<EA::SchedulerCore::ExperimentTransitionRecord>
    loadExperimentTransition(long long, bool) override
    {
        return std::nullopt;
    }

    EA::SchedulerCore::ExperimentTransitionPersistenceResult
    applyExperimentTransition(
        const EA::SchedulerCore::ExperimentTransitionUpdate&) override
    {
        return EA::SchedulerCore::ExperimentTransitionPersistenceResult::Updated;
    }
};

EA::SchedulerCore::ReservedWorkerAttempt ExperimentAttempt()
{
    return {71,
            "scheduler-a:worker:nonce-a:experiment:41:train",
            41,
            std::nullopt,
            "experiment",
            "train",
            "train",
            "/tmp/train.log"};
}

template <typename Exception, typename Operation>
std::string Throws(Operation operation)
{
    try
    {
        operation();
    }
    catch (const Exception& error)
    {
        return error.what();
    }
    assert(false && "expected exception");
    return {};
}

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    MemoryRepository repository;
    repository.reservationResult = {
        WorkerAttemptReservationStatus::Reserved,
        ExperimentAttempt()};
    WorkerAttemptLifecycleService service{
        repository,
        {"scheduler-a", 17, "/tmp/LSTM_Release"}};

    const auto experiment = service.reserveExperiment(
        {41, "train", "/tmp/train.log", "nonce-a", true});
    assert(experiment && experiment->workerAttemptId == 71);
    assert(repository.experimentReservation);
    const auto& experimentRequest = *repository.experimentReservation;
    assert(experimentRequest.launchAttemptIdentity ==
           "scheduler-a:worker:nonce-a:experiment:41:train");
    assert(experimentRequest.schedulerInvocationId == "scheduler-a");
    assert(experimentRequest.schedulerFencingToken == 17);
    assert(experimentRequest.phase == "train");
    assert(experimentRequest.commandIdentity == "experiment:41:train");
    assert(experimentRequest.currentOperation == "train");
    assert(experimentRequest.cancellationOnly);

    ReservedWorkerAttempt checkpointAttempt{
        72,
        "scheduler-a:worker:nonce-b:checkpoint_infer:88",
        41,
        88,
        "checkpoint_infer",
        "infer",
        "infer",
        "/tmp/checkpoint.log"};
    repository.reservationResult = {
        WorkerAttemptReservationStatus::Reserved,
        checkpointAttempt};
    const auto checkpoint = service.reserveCheckpoint(
        {41, 88, "/tmp/checkpoint.log", "nonce-b"});
    assert(checkpoint && checkpoint->workerAttemptId == 72);
    assert(repository.checkpointReservation);
    const auto& checkpointRequest = *repository.checkpointReservation;
    assert(checkpointRequest.launchAttemptIdentity ==
           "scheduler-a:worker:nonce-b:checkpoint_infer:88");
    assert(checkpointRequest.commandIdentity == "checkpoint_infer:88");
    assert(checkpointRequest.checkpointEvalId == 88);

    repository.reservationResult = {
        WorkerAttemptReservationStatus::LifecycleUnavailable,
        std::nullopt};
    assert(!service.reserveExperiment(
        {41, "infer", "/tmp/infer.log", "nonce-c", false}));

    repository.reservationResult = {
        WorkerAttemptReservationStatus::ReservationInsertFailed,
        std::nullopt};
    assert(Throws<std::runtime_error>([&] {
               (void)service.reserveExperiment(
                   {41, "train", "/tmp/train.log", "nonce-d", false});
           }) == "worker_attempt_reservation_insert_failed");
    repository.reservationResult = {
        WorkerAttemptReservationStatus::LifecycleClaimFailed,
        std::nullopt};
    assert(Throws<std::runtime_error>([&] {
               (void)service.reserveCheckpoint(
                   {41, 88, "/tmp/checkpoint.log", "nonce-e"});
           }) == "checkpoint_worker_attempt_lifecycle_claim_failed");

    service.recordSpawned(
        {checkpointAttempt, 43210, "start-identity", "command identity", true});
    assert(repository.spawned);
    assert(repository.spawned->workerAttemptId == 72);
    assert(repository.spawned->checkpointEvalId == 88);
    assert(repository.spawned->workerPid == 43210);
    assert(repository.spawned->canonicalExecutablePath ==
           "/tmp/LSTM_Release");

    repository.spawnResult =
        SpawnPersistenceResult::AttemptPreconditionRejected;
    assert(Throws<SchedulerAuthorityLost>([&] {
               service.recordSpawned(
                   {checkpointAttempt, 43210, "start", "command", true});
           }) == "worker_attempt_spawn_persistence_fence_rejected");
    assert(Throws<std::runtime_error>([&] {
               service.recordSpawned(
                   {checkpointAttempt, 43210, "start", "command", false});
           }) == "child_spawn_evidence_persistence_fence_rejected");
    repository.spawnResult =
        SpawnPersistenceResult::LifecyclePreconditionRejected;
    assert(Throws<std::runtime_error>([&] {
               service.recordSpawned(
                   {checkpointAttempt, 43210, "start", "command", true});
           }) == "worker_attempt_spawn_lifecycle_predicate_rejected");

    repository.failureResult = LaunchFailurePersistenceResult::Updated;
    service.recordLaunchFailure(
        {ExperimentAttempt(), 127, "launch_failed_errno_2"});
    assert(repository.launchFailure);
    assert(repository.launchFailure->workerAttemptId == 71);
    assert(repository.launchFailure->diagnostic == "launch_failed_errno_2");

    repository.failureResult =
        LaunchFailurePersistenceResult::AttemptPreconditionRejected;
    assert(Throws<std::runtime_error>([&] {
               service.recordLaunchFailure(
                   {ExperimentAttempt(), 127, "failure"});
           }) ==
           "exact_attempt_predicate_rejected:"
           "terminalize_exact_launch_failure_attempt:affected_rows=0");
    repository.failureResult =
        LaunchFailurePersistenceResult::LifecyclePreconditionRejected;
    assert(Throws<std::runtime_error>([&] {
               service.recordLaunchFailure(
                   {ExperimentAttempt(), 127, "failure"});
           }) ==
           "exact_attempt_predicate_rejected:"
           "clear_exact_launch_failure_binding:affected_rows=0");

    assert(Throws<std::invalid_argument>([] {
               MemoryRepository invalidRepository;
               WorkerAttemptLifecycleService invalid{
                   invalidRepository, {"", 0, ""}};
           }) == "complete scheduler authority context required");
    return 0;
}
