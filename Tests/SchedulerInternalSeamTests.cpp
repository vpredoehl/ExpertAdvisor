#include "../Sources/SchedulerCore/SchedulerPolicy.hpp"
#include "../Sources/SchedulerCore/SchedulerRepository.hpp"
#include "../Sources/SchedulerCore/WorkerProcessController.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

class MemorySchedulerRepository final
    : public EA::SchedulerCore::SchedulerRepository
{
public:
    std::vector<EA::SchedulerCore::PendingSchedulerExperimentRecord> pending;
    std::vector<EA::SchedulerCore::RunningSchedulerExperimentRecord> running;
    EA::SchedulerCore::SchedulerQueueSnapshot queue;
    int capacityUsed = 0;
    std::optional<EA::SchedulerCore::PreemptionVictimRecord> victim;
    EA::SchedulerCore::SpawnedWorkerAttemptUpdate observedSpawn;

    std::vector<EA::SchedulerCore::PendingSchedulerExperimentRecord>
    loadPendingExperiments(std::string_view, bool) override
    {
        return pending;
    }

    std::vector<EA::SchedulerCore::RunningSchedulerExperimentRecord>
    loadRunningExperiments() override
    {
        return running;
    }

    EA::SchedulerCore::SchedulerQueueSnapshot loadQueueSnapshot() override
    {
        return queue;
    }

    int countWorkersConsumingCapacity(std::string_view) override
    {
        return capacityUsed;
    }

    std::optional<EA::SchedulerCore::PreemptionVictimRecord>
    loadPreemptionVictim(std::string_view, int) override
    {
        return victim;
    }

    EA::SchedulerCore::SpawnPersistenceResult persistSpawnedWorkerAttempt(
        const EA::SchedulerCore::SpawnedWorkerAttemptUpdate& update) override
    {
        observedSpawn = update;
        return EA::SchedulerCore::HasCompleteSpawnedWorkerAttemptUpdate(update)
            ? EA::SchedulerCore::SpawnPersistenceResult::Updated
            : EA::SchedulerCore::SpawnPersistenceResult::
                  AttemptPreconditionRejected;
    }

    EA::SchedulerCore::LaunchFailurePersistenceResult
    persistWorkerAttemptLaunchFailure(
        const EA::SchedulerCore::WorkerAttemptLaunchFailureUpdate&) override
    {
        return EA::SchedulerCore::LaunchFailurePersistenceResult::Updated;
    }
};

EA::ExperimentScheduler::ObservedChildStatus WaitForChild(
    EA::SchedulerCore::WorkerProcessController& controller,
    pid_t pid)
{
    for (int attempt = 0; attempt < 2000; ++attempt)
    {
        const auto status = controller.observeChild(pid);
        if (status.kind != EA::ExperimentScheduler::ChildStatusKind::Running)
            return status;
        ::usleep(1000);
    }
    assert(false && "child did not terminate");
    return {};
}

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    std::vector<PendingSelectionMetadata> candidates{
        {"normal", "preemption", "2026-01-01 00:00:00", 4},
        {"normal", "operator", "2026-01-01 00:00:01", 3},
        {"high", "none", "2026-01-01 00:00:02", 2},
        {"normal", "operator", "2026-01-01 00:00:01", 1},
        {"low", "operator", "2025-01-01 00:00:00", 5}};
    std::sort(candidates.begin(), candidates.end(), PendingSelectionPrecedes);
    assert(candidates[0].experimentId == 2);
    assert(candidates[1].experimentId == 1);
    assert(candidates[2].experimentId == 3);
    assert(candidates[3].experimentId == 4);
    assert(candidates[4].experimentId == 5);

    MemorySchedulerRepository repository;
    PendingSchedulerExperimentRecord persisted;
    persisted.experiment.experimentId = 41;
    persisted.experiment.coreLrMult = std::nullopt;
    persisted.experiment.headLrMult = 2.5;
    persisted.experiment.inferStart = std::nullopt;
    persisted.experiment.inferEnd = "2026-01-01";
    persisted.experiment.lastModelId = std::nullopt;
    persisted.experiment.resumeModelId = 88;
    persisted.experiment.trainLogPath = std::nullopt;
    persisted.experiment.inferLogPath = "infer.log";
    persisted.experiment.analysisLogPath = std::nullopt;
    persisted.activeWorkerAttemptId = std::nullopt;
    repository.pending.push_back(persisted);
    const auto roundTrip = repository.loadPendingExperiments("train", false);
    assert(roundTrip.size() == 1);
    assert(!roundTrip[0].experiment.coreLrMult);
    assert(roundTrip[0].experiment.headLrMult == 2.5);
    assert(!roundTrip[0].experiment.inferStart);
    assert(roundTrip[0].experiment.inferEnd == "2026-01-01");
    assert(!roundTrip[0].experiment.lastModelId);
    assert(roundTrip[0].experiment.resumeModelId == 88);
    assert(!roundTrip[0].experiment.trainLogPath);
    assert(roundTrip[0].experiment.inferLogPath == "infer.log");
    assert(!roundTrip[0].experiment.analysisLogPath);
    assert(!roundTrip[0].activeWorkerAttemptId);

    SpawnedWorkerAttemptUpdate spawn{
        101,
        "scheduler-invocation",
        9,
        41,
        std::nullopt,
        "train",
        12345,
        "process-start",
        "/tmp/LSTM_Release",
        "/tmp/LSTM_Release --train"};
    assert(HasCompleteSpawnedWorkerAttemptUpdate(spawn));
    assert(repository.persistSpawnedWorkerAttempt(spawn) ==
           SpawnPersistenceResult::Updated);
    assert(repository.observedSpawn.checkpointEvalId == std::nullopt);
    spawn.processStartIdentity.clear();
    assert(!HasCompleteSpawnedWorkerAttemptUpdate(spawn));
    assert(repository.persistSpawnedWorkerAttempt(spawn) ==
           SpawnPersistenceResult::AttemptPreconditionRejected);

    assert(WorkerProcessSignalNumber(WorkerProcessSignal::Stop) == SIGSTOP);
    assert(WorkerProcessSignalNumber(WorkerProcessSignal::Resume) == SIGCONT);
    assert(WorkerProcessSignalNumber(WorkerProcessSignal::Terminate) == SIGTERM);
    assert(WorkerProcessSignalNumber(WorkerProcessSignal::Kill) == SIGKILL);
    assert(WorkerCommandLine({"/tmp/LSTM Release", "--train", "a=b"}) ==
           "/tmp/LSTM Release --train a=b");

    PosixWorkerProcessController processes;
    const auto missing = processes.isAlive(2147483647);
    assert(!missing.alive);
    assert(missing.errorNumber == ESRCH);
    assert(!processes.resolveExecutablePath().empty());

    char exitLog[] = "/tmp/ea_scheduler_process_exit.XXXXXX";
    const int exitFd = ::mkstemp(exitLog);
    assert(exitFd >= 0);
    assert(::close(exitFd) == 0);
    const auto child = processes.spawn({
        {"/bin/sh", "-c", "exit 23"}, exitLog, 700, "infer", true});
    assert(child.launched());
    const auto exited = WaitForChild(processes, child.pid);
    assert(exited.kind == EA::ExperimentScheduler::ChildStatusKind::Exited);
    assert(exited.exitCode == 23);
    assert(::unlink(exitLog) == 0);

    char signalLog[] = "/tmp/ea_scheduler_process_signal.XXXXXX";
    const int signalFd = ::mkstemp(signalLog);
    assert(signalFd >= 0);
    assert(::close(signalFd) == 0);
    const auto sleeper = processes.spawn({
        {"/bin/sleep", "30"},
        signalLog,
        701,
        "train",
        true});
    assert(sleeper.launched());
    assert(processes.isAlive(sleeper.pid).alive);
    const auto stopped = processes.signal(
        sleeper.pid, WorkerProcessSignal::Stop);
    assert(stopped.success && stopped.signalNumber == SIGSTOP);
    const auto resumed = processes.signal(
        sleeper.pid, WorkerProcessSignal::Resume);
    assert(resumed.success && resumed.signalNumber == SIGCONT);
    const auto terminated = processes.signal(
        sleeper.pid, WorkerProcessSignal::Terminate);
    assert(terminated.success && terminated.signalNumber == SIGTERM);
    const auto signaled = WaitForChild(processes, sleeper.pid);
    assert(signaled.kind == EA::ExperimentScheduler::ChildStatusKind::Signaled);
    assert(signaled.signalNumber == SIGTERM);
    assert(::unlink(signalLog) == 0);
    return 0;
}
