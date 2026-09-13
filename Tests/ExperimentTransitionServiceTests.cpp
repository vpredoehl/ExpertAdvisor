#include "SchedulerCore/ExperimentTransitionService.hpp"

#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

using namespace EA::SchedulerCore;

class MemoryRepository final : public SchedulerRepository
{
public:
    std::optional<ExperimentTransitionRecord> transition;
    ExperimentTransitionPersistenceResult persistenceResult =
        ExperimentTransitionPersistenceResult::Updated;
    std::optional<ExperimentTransitionUpdate> applied;
    bool observedForUpdate = false;

    std::vector<PendingSchedulerExperimentRecord>
    loadPendingExperiments(std::string_view, bool) override { return {}; }
    std::vector<RunningSchedulerExperimentRecord>
    loadRunningExperiments() override { return {}; }
    SchedulerQueueSnapshot loadQueueSnapshot() override { return {}; }
    int countWorkersConsumingCapacity(std::string_view) override { return 0; }
    std::optional<PreemptionVictimRecord>
    loadPreemptionVictim(std::string_view, int) override
    {
        return std::nullopt;
    }
    WorkerAttemptReservationResult reserveExperimentWorkerAttempt(
        const ExperimentWorkerAttemptReservation&) override { return {}; }
    WorkerAttemptReservationResult reserveCheckpointWorkerAttempt(
        const CheckpointWorkerAttemptReservation&) override { return {}; }
    WorkerAttemptReservationResult reserveCheckpointAnalysisAttempt(
        const CheckpointAnalysisAttemptReservation&) override { return {}; }
    SpawnPersistenceResult persistSpawnedWorkerAttempt(
        const SpawnedWorkerAttemptUpdate&) override
    {
        return SpawnPersistenceResult::Updated;
    }
    LaunchFailurePersistenceResult persistWorkerAttemptLaunchFailure(
        const WorkerAttemptLaunchFailureUpdate&) override
    {
        return LaunchFailurePersistenceResult::Updated;
    }
    bool persistCheckpointAnalysisCompletion(
        const CheckpointAnalysisCompletionUpdate&) override { return true; }
    CheckpointAnalysisPersistenceResult persistCheckpointAnalysisTerminalState(
        const CheckpointAnalysisTerminalUpdate&) override
    {
        return CheckpointAnalysisPersistenceResult::Updated;
    }

    std::optional<ExperimentTransitionRecord> loadExperimentTransition(
        long long experimentId,
        bool forUpdate) override
    {
        observedForUpdate = forUpdate;
        if (transition &&
            transition->experiment.experimentId != experimentId)
        {
            return std::nullopt;
        }
        return transition;
    }

    ExperimentTransitionPersistenceResult applyExperimentTransition(
        const ExperimentTransitionUpdate& update) override
    {
        applied = update;
        return persistenceResult;
    }
};

class MemoryCheckpointSelector final
    : public ExperimentTransitionCheckpointSelector
{
public:
    RetryTrainingCheckpointSelection retry;
    RequeueTrainingCheckpointSelection requeue;
    int retryCalls = 0;
    int requeueCalls = 0;

    RetryTrainingCheckpointSelection selectRetryTrainingCheckpoint(
        const ExperimentTransitionRecord&) override
    {
        ++retryCalls;
        return retry;
    }

    RequeueTrainingCheckpointSelection selectRequeueTrainingCheckpoint(
        const ExperimentTransitionRecord&) override
    {
        ++requeueCalls;
        return requeue;
    }
};

ExperimentTransitionRecord Record(
    std::string status,
    std::string phase)
{
    ExperimentTransitionRecord record;
    record.experiment.experimentId = 41;
    record.experiment.targetEpochs = 80;
    record.status = std::move(status);
    record.phase = std::move(phase);
    record.currentEpoch = 20;
    record.schedulerPriority = "high";
    return record;
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

    {
        auto pending = Record("pending", "train");
        const auto cancel = PlanExperimentTransition(
            ExperimentTransitionAction::Cancel, pending);
        assert(cancel.accepted && cancel.newStatus == "cancelled");

        pending.status = "running";
        const auto rejectedCancel = PlanExperimentTransition(
            ExperimentTransitionAction::Cancel, pending);
        assert(!rejectedCancel.accepted);
        assert(rejectedCancel.rejectionReason ==
               "running_experiment_cannot_be_cancelled");

        auto completed = Record("completed", "done");
        const auto rejectedRequeue = PlanExperimentTransition(
            ExperimentTransitionAction::RequeueTraining, completed);
        assert(rejectedRequeue.rejectionReason ==
               "terminal_experiment_cannot_be_requeued");

        auto attached = Record("pending", "infer");
        attached.activeWorkerAttemptId = 90;
        const auto rejectedAttached = PlanExperimentTransition(
            ExperimentTransitionAction::RequeueTraining, attached);
        assert(rejectedAttached.rejectionReason ==
               "requeue_training_worker_attempt_attached");

        auto exhausted = Record("failed", "train");
        exhausted.currentEpoch = 80;
        const auto rejectedExhausted = PlanExperimentTransition(
            ExperimentTransitionAction::RequeueTraining, exhausted);
        assert(rejectedExhausted.rejectionReason ==
               "requeue_training_has_no_remaining_epochs");

        auto failedDone = Record("failed", "done");
        failedDone.experiment.lastModelId = 700;
        const auto retryDone = PlanExperimentTransition(
            ExperimentTransitionAction::RetryFailed, failedDone);
        assert(retryDone.accepted && retryDone.newPhase == "analyze");

        auto missingModel = Record("failed", "done");
        const auto rejectedAnalysis = PlanExperimentTransition(
            ExperimentTransitionAction::RequeueAnalysis, missingModel);
        assert(rejectedAnalysis.rejectionReason ==
               "requeue_analysis_requires_model_id");
    }

    {
        MemoryRepository repository;
        MemoryCheckpointSelector selector;
        std::ostringstream output;
        ExperimentTransitionService service{repository, selector, output};
        assert(service.run({ExperimentTransitionAction::Cancel, 41, false,
                            true}) == 1);
        assert(output.str() ==
               "SCHEDULER_CONTROL_REJECTED,action=cancel,experiment_id=41,"
               "reason=experiment_not_found\n");
    }

    {
        MemoryRepository repository;
        repository.transition = Record("paused", "train");
        MemoryCheckpointSelector selector;
        std::ostringstream output;
        ExperimentTransitionService service{repository, selector, output};
        assert(service.run({ExperimentTransitionAction::Cancel, 41, false,
                            true}) == 0);
        assert(repository.applied);
        assert(repository.applied->newStatus == "cancelled");
        assert(repository.applied->newPhase == "train");
        assert(output.str().find(
                   "SCHEDULER_CONTROL_APPLIED,action=cancel") !=
               std::string::npos);
    }

    {
        MemoryRepository repository;
        repository.transition = Record("failed", "train");
        repository.transition->experiment.resumeModelId = 600;
        MemoryCheckpointSelector selector;
        selector.retry = {600, 700, 60, true,
                          "newer_compatible_checkpoint"};
        std::ostringstream output;
        ExperimentTransitionService service{repository, selector, output};
        assert(service.run({ExperimentTransitionAction::RetryFailed, 41,
                            false, true}) == 0);
        assert(repository.observedForUpdate);
        assert(selector.retryCalls == 1);
        assert(repository.applied);
        assert(repository.applied->newStatus == "pending");
        assert(repository.applied->newPhase == "train");
        assert(repository.applied->selectedResumeModelId == 700);
        assert(output.str().find(
                   "SCHEDULER_RETRY_CHECKPOINT_SELECTION,experiment_id=41,") !=
               std::string::npos);
        assert(output.str().find("promotion=1") != std::string::npos);
        assert(output.str().find(
                   "SCHEDULER_CONTROL_APPLIED,action=retry_failed") !=
               std::string::npos);
    }

    {
        MemoryRepository repository;
        repository.transition = Record("pending", "infer");
        MemoryCheckpointSelector selector;
        selector.requeue = {std::nullopt, std::nullopt,
                            "no_valid_checkpoint"};
        std::ostringstream output;
        ExperimentTransitionService service{repository, selector, output};
        assert(service.run({ExperimentTransitionAction::RequeueTraining, 41,
                            false, true}) == 1);
        assert(!repository.applied);
        assert(output.str().find("reason=no_valid_checkpoint") !=
               std::string::npos);
    }

    {
        MemoryRepository repository;
        repository.transition = Record("failed", "infer");
        repository.transition->experiment.lastModelId = 700;
        MemoryCheckpointSelector selector;
        std::ostringstream output;
        ExperimentTransitionService service{repository, selector, output};
        assert(service.run({ExperimentTransitionAction::RequeueInference, 41,
                            true, true}) == 0);
        assert(!repository.observedForUpdate);
        assert(!repository.applied);
        assert(output.str().find(
                   "SCHEDULER_CONTROL_DRY_RUN,action=requeue_inference") !=
               std::string::npos);
    }

    {
        MemoryRepository repository;
        repository.transition = Record("failed", "infer");
        repository.transition->experiment.lastModelId = 700;
        MemoryCheckpointSelector selector;
        std::ostringstream output;
        ExperimentTransitionService service{repository, selector, output};
        assert(service.run({ExperimentTransitionAction::RequeueInference, 41,
                            false, true}) == 0);
        assert(repository.applied->action ==
               ExperimentTransitionAction::RequeueInference);
        assert(output.str().find(
                   "operator_forced_final_inference_rerun_requested=1") !=
               std::string::npos);
    }

    {
        MemoryRepository repository;
        repository.transition = Record("pending", "infer");
        repository.persistenceResult =
            ExperimentTransitionPersistenceResult::AtomicPreconditionRejected;
        MemoryCheckpointSelector selector;
        selector.requeue = {700, 40, "selected"};
        std::ostringstream output;
        ExperimentTransitionService service{repository, selector, output};
        assert(Throws<std::runtime_error>([&] {
                   (void)service.run({
                       ExperimentTransitionAction::RequeueTraining,
                       41,
                       false,
                       true});
               }) == "scheduler_control_atomic_predicate_changed");
    }

    return 0;
}
