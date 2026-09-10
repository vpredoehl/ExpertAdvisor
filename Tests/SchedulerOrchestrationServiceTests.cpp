#include "../Sources/SchedulerCore/ReconciliationService.hpp"
#include "../Sources/SchedulerCore/SchedulerAdmissionService.hpp"
#include "../Sources/SchedulerCore/WorkerControlService.hpp"

#include <cassert>
#include <cerrno>
#include <optional>
#include <string_view>
#include <vector>

namespace
{

class MemoryRepository final : public EA::SchedulerCore::SchedulerRepository
{
public:
    std::vector<EA::SchedulerCore::PendingSchedulerExperimentRecord> pending;
    EA::SchedulerCore::SchedulerQueueSnapshot queue;
    int used = 0;
    std::optional<EA::SchedulerCore::PreemptionVictimRecord> victim;
    int observedCandidateRank = -1;

    std::vector<EA::SchedulerCore::PendingSchedulerExperimentRecord>
    loadPendingExperiments(std::string_view, bool) override
    {
        return pending;
    }

    std::vector<EA::SchedulerCore::RunningSchedulerExperimentRecord>
    loadRunningExperiments() override
    {
        return {};
    }

    EA::SchedulerCore::SchedulerQueueSnapshot loadQueueSnapshot() override
    {
        return queue;
    }

    int countWorkersConsumingCapacity(std::string_view) override
    {
        return used;
    }

    std::optional<EA::SchedulerCore::PreemptionVictimRecord>
    loadPreemptionVictim(std::string_view, int candidateRank) override
    {
        observedCandidateRank = candidateRank;
        return victim;
    }

    EA::SchedulerCore::SpawnPersistenceResult persistSpawnedWorkerAttempt(
        const EA::SchedulerCore::SpawnedWorkerAttemptUpdate&) override
    {
        return EA::SchedulerCore::SpawnPersistenceResult::Updated;
    }

    EA::SchedulerCore::LaunchFailurePersistenceResult
    persistWorkerAttemptLaunchFailure(
        const EA::SchedulerCore::WorkerAttemptLaunchFailureUpdate&) override
    {
        return EA::SchedulerCore::
            LaunchFailurePersistenceResult::Updated;
    }
};

class RecordingProcessController final
    : public EA::SchedulerCore::WorkerProcessController
{
public:
    mutable std::vector<EA::SchedulerCore::WorkerProcessSignal> signals;
    bool groupTerminateSucceeds = true;
    bool alive = false;

    EA::SchedulerCore::ProcessLiveness isAlive(
        pid_t, bool = false) const override
    {
        return {alive, false, alive ? 0 : ESRCH};
    }

    EA::SchedulerCore::ProcessSignalResult signal(
        pid_t,
        EA::SchedulerCore::WorkerProcessSignal signal,
        bool processGroup = false) const override
    {
        signals.push_back(signal);
        const bool success =
            signal != EA::SchedulerCore::WorkerProcessSignal::Terminate ||
            !processGroup || groupTerminateSucceeds;
        return {
            success,
            EA::SchedulerCore::WorkerProcessSignalNumber(signal),
            success ? 0 : ESRCH};
    }

    EA::ExperimentScheduler::ObservedChildStatus observeChild(
        pid_t) const override
    {
        return {};
    }

    EA::SchedulerCore::SpawnResult spawn(
        const EA::SchedulerCore::WorkerLaunchRequest&) const override
    {
        return {};
    }

    std::string resolveExecutablePath() const override
    {
        return "/tmp/LSTM_Release";
    }
};

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    MemoryRepository repository;
    PendingSchedulerExperimentRecord high;
    high.experiment.experimentId = 1;
    high.schedulerPriority = "high";
    repository.pending.push_back(high);
    repository.used = 1;
    repository.victim = PreemptionVictimRecord{7, "low", 70};
    SchedulerAdmissionService admission{repository};
    const auto phase = admission.loadPhase("train", 2);
    assert(phase.candidates.size() == 1);
    assert(phase.hasCapacity());
    assert(phase.availableSlots() == 1);
    repository.used = 2;
    assert(!admission.hasCapacity("train", 2));
    const auto victim = admission.selectPreemptionVictim("train", "high");
    assert(victim && victim->experimentId == 7);
    assert(repository.observedCandidateRank == 0);
    assert(!admission.selectPreemptionVictim("analyze", "high"));

    const auto stopped = PlanAttemptObservation(
        {true, true, true, true, true});
    assert(stopped.action == AttemptObservationAction::RetainLive);
    assert(stopped.lifecycleState == "stopped");
    assert(!stopped.capacityConsumed);
    assert(!stopped.restoreRunningLifecycle);

    const auto resumed = PlanAttemptObservation(
        {true, true, true, true, false});
    assert(resumed.lifecycleState == "observed");
    assert(resumed.capacityConsumed);
    assert(resumed.restoreRunningLifecycle);

    const auto mismatch = PlanAttemptObservation(
        {true, true, false, false, false});
    assert(mismatch.lifecycleState == "identity_ambiguous");
    assert(mismatch.reconciliationResult == "identity_mismatch");
    assert(mismatch.capacityConsumed);

    const auto missing = PlanAttemptObservation(
        {true, false, false, false, false});
    assert(missing.action == AttemptObservationAction::ReconcileMissing);
    const auto inspectionFailure = PlanAttemptObservation(
        {false, false, false, false, false});
    assert(inspectionFailure.action == AttemptObservationAction::Defer);
    assert(inspectionFailure.diagnostic ==
           "process_identity_inspection_failed");

    assert(PlanMissingStoppedWorker(
               "train", "pending", true, "preemption", true)
               .disposition == MissingStoppedWorkerDisposition::
                                   RestartPreemptedTrainFromCheckpoint);
    assert(PlanMissingStoppedWorker(
               "train", "pending", true, "preemption", false)
               .disposition == MissingStoppedWorkerDisposition::
                                   FailPreemptedTrainWithoutCheckpoint);
    assert(PlanMissingStoppedWorker(
               "infer", "pending", true, "operator", false)
               .disposition ==
           MissingStoppedWorkerDisposition::RestartEligible);

    const auto recovered = PlanMissingProcessTerminalization(true);
    assert(recovered.attemptLifecycleState == "completed");
    assert(recovered.reconciliationResult ==
           "process_missing_result_recovered");
    const auto failed = PlanMissingProcessTerminalization(false);
    assert(failed.attemptLifecycleState == "failed");
    assert(failed.reconciliationResult == "process_missing_no_result");

    RecordingProcessController processes;
    WorkerControlService controls{processes};
    const auto terminated = controls.terminateUncommittedWorker(
        {123, 1, 0});
    assert(terminated.terminatedDuringGrace);
    assert(!terminated.killEscalated);
    assert(processes.signals.size() == 1);
    assert(processes.signals[0] == WorkerProcessSignal::Terminate);

    RecordingProcessController escalationProcesses;
    escalationProcesses.alive = true;
    escalationProcesses.groupTerminateSucceeds = false;
    WorkerControlService escalation{escalationProcesses};
    const auto escalated = escalation.terminateUncommittedWorker(
        {124, 1, 0});
    assert(!escalated.terminatedDuringGrace);
    assert(escalated.killEscalated);
    assert(escalationProcesses.signals.size() == 4);
    assert(escalationProcesses.signals[0] ==
           WorkerProcessSignal::Terminate);
    assert(escalationProcesses.signals[1] ==
           WorkerProcessSignal::Terminate);
    assert(escalationProcesses.signals[2] == WorkerProcessSignal::Kill);
    assert(escalationProcesses.signals[3] == WorkerProcessSignal::Kill);
    return 0;
}
