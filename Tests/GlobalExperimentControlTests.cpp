#include <algorithm>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "../Headers/ExperimentScheduler.hpp"
#include "../Sources/GlobalExperimentControl.hpp"

using namespace EA::GlobalExperimentControl;
using EA::ExperimentScheduler::AvailableInferProcessSlots;

class FakeProcesses final : public ProcessOperations
{
public:
    std::map<int, ProcessObservation> observations;
    std::vector<std::pair<int, int>> signals;
    std::vector<bool> waits;
    int callerPid = 900;
    int callerGroup = 900;

    ProcessObservation Observe(int pid) override
    {
        const auto found = observations.find(pid);
        if (found == observations.end())
        {
            ProcessObservation missing;
            missing.inspectionSucceeded = true;
            missing.pid = pid;
            return missing;
        }
        return found->second;
    }

    bool SignalProcessGroup(int group, int signal, int& error) override
    {
        signals.emplace_back(group, signal);
        error = 0;
        return true;
    }

    bool WaitForProcessGroupExit(
        int,
        std::chrono::milliseconds) override
    {
        if (waits.empty())
            return true;
        const bool value = waits.front();
        waits.erase(waits.begin());
        return value;
    }

    int CallerPid() const override { return callerPid; }
    int CallerProcessGroupId() const override { return callerGroup; }
};

ManagedWorker Worker(long long experimentId = 42,
                     const std::string& phase = "train")
{
    ManagedWorker worker;
    worker.experimentId = experimentId;
    worker.phase = phase;
    worker.lifecycleStatus = "running";
    worker.pid = 1200;
    worker.processGroupId = 1200;
    worker.executable = "/tmp/LSTM_Release";
    worker.commandLine =
        "/tmp/LSTM_Release --train --scheduler-experiment-id=42";
    worker.processStartIdentity = "1700000000:123456";
    return worker;
}

ProcessObservation Observation(
    const std::string& command =
        "/tmp/LSTM_Release --train --scheduler-experiment-id=42")
{
    ProcessObservation observation;
    observation.exists = true;
    observation.inspectionSucceeded = true;
    observation.pid = 1200;
    observation.processGroupId = 1200;
    observation.executable = "/tmp/LSTM_Release";
    observation.commandLine = command;
    observation.processStartIdentity = "1700000000:123456";
    return observation;
}

ManagedWorker CheckpointWorker(long long checkpointEvalId,
                               long long experimentId,
                               int pid,
                               const std::string& lifecycleStatus = "running")
{
    const std::string command =
        "/tmp/LSTM_Release --infer --model=1162 "
        "--scheduler-checkpoint-eval-id=" +
        std::to_string(checkpointEvalId);
    ManagedWorker worker;
    worker.experimentId = experimentId;
    worker.checkpointEvalId = checkpointEvalId;
    worker.phase = "checkpoint_infer";
    worker.lifecycleStatus = lifecycleStatus;
    worker.pid = pid;
    worker.processGroupId = pid;
    worker.executable = "/tmp/LSTM_Release";
    worker.commandLine = command;
    worker.processStartIdentity =
        "1700000000:" + std::to_string(pid);
    return worker;
}

ProcessObservation ObservationFor(const ManagedWorker& worker,
                                  const std::string& command)
{
    ProcessObservation observation;
    observation.exists = true;
    observation.inspectionSucceeded = true;
    observation.pid = worker.pid;
    observation.processGroupId = worker.processGroupId.value_or(-1);
    observation.executable = worker.executable.value_or("");
    observation.commandLine = command;
    observation.processStartIdentity =
        worker.processStartIdentity.value_or("");
    return observation;
}

SchedulerWorkerCandidate Candidate(
    int pid,
    const std::string& kind,
    const std::string& command,
    double cpuPercent = 0.0,
    double memPercent = 0.0,
    double rssMb = 0.0)
{
    return SchedulerWorkerCandidate{
        pid, kind, command, cpuPercent, memPercent, rssMb};
}

int main()
{
    Command pause;
    pause.action = Action::PauseAll;
    pause.confirmed = true;
    assert(!ValidateCommand(pause));

    Command missingConfirmation = pause;
    missingConfirmation.confirmed = false;
    assert(ValidateCommand(missingConfirmation));
    missingConfirmation.dryRun = true;
    assert(!ValidateCommand(missingConfirmation));

    Command cancel;
    cancel.action = Action::CancelAll;
    cancel.confirmed = true;
    assert(ValidateCommand(cancel));
    cancel.cancellationMode = CancellationMode::Immediate;
    assert(!ValidateCommand(cancel));

    Command invalidInfer = pause;
    invalidInfer.inferBeforeCancel = true;
    assert(ValidateCommand(invalidInfer));

    assert(NextCancellationCheckpoint(0, 20, 100, std::nullopt) == 20);
    assert(NextCancellationCheckpoint(21, 20, 100, 20) == 40);
    assert(NextCancellationCheckpoint(40, 20, 100, 40) == 40);
    assert(!NextCancellationCheckpoint(91, 20, 100, 80));
    assert(!NextCancellationCheckpoint(20, 0, 100, 20));
    assert(RequestExitCodeForPersistedStatus("completed") == 0);
    assert(RequestExitCodeForPersistedStatus("pending") == 0);
    assert(RequestExitCodeForPersistedStatus("partial") == 1);
    assert(RequestExitCodeForPersistedStatus("failed") == 1);
    assert(RequestExitCodeForPersistedStatus("applying") == 1);

    assert(AvailableInferProcessSlots(3, 1, 2) == 0);
    assert(AvailableInferProcessSlots(3, 1, 1) == 1);
    assert(AvailableInferProcessSlots(3, 4, 0) == 0);

    ControlSnapshot running;
    assert(NormalSchedulingAllowed(running));
    running.desiredState = "paused";
    assert(!NormalSchedulingAllowed(running));
    running.desiredState = "running";
    running.activeRequestId = 9;
    running.activeAction = "cancel_all";
    running.inferBeforeCancel = true;
    assert(!NormalSchedulingAllowed(running));
    assert(CancellationInferenceAllowed(running));

    FakeProcesses processes;
    processes.observations.emplace(1200, Observation());
    ManagedWorker worker = Worker();
    assert(ValidateManagedWorker(worker, processes).identity ==
           IdentityResult::Validated);

    SignalOutcome paused = PauseWorker(worker, processes);
    assert(paused.success);
    assert((processes.signals ==
            std::vector<std::pair<int, int>>{{1200, SIGSTOP}}));

    processes.signals.clear();
    processes.observations[1200].stopped = true;
    SignalOutcome resumed = ResumeWorker(worker, processes);
    assert(resumed.success);
    assert((processes.signals ==
            std::vector<std::pair<int, int>>{{1200, SIGCONT}}));

    processes.signals.clear();
    processes.observations[1200].stopped = true;
    processes.waits = {true};
    SignalOutcome cancelled =
        CancelWorker(worker, true, std::chrono::milliseconds(1), processes);
    assert(cancelled.success);
    assert((processes.signals ==
            std::vector<std::pair<int, int>>{
                {1200, SIGCONT}, {1200, SIGTERM}}));

    processes.signals.clear();
    processes.observations[1200].stopped = false;
    processes.waits = {false, true};
    cancelled =
        CancelWorker(worker, false, std::chrono::milliseconds(1), processes);
    assert(cancelled.success);
    assert(cancelled.result == "escalated");
    assert((processes.signals ==
            std::vector<std::pair<int, int>>{
                {1200, SIGTERM}, {1200, SIGKILL}}));

    FakeProcesses stale;
    stale.observations.emplace(
        1200,
        Observation("/usr/bin/python unrelated.py --scheduler-experiment-id=42"));
    assert(ValidateManagedWorker(worker, stale).identity ==
           IdentityResult::IdentityValidationFailed);
    assert(!PauseWorker(worker, stale).success);
    assert(stale.signals.empty());

    FakeProcesses prefixCollision;
    prefixCollision.observations.emplace(
        1200,
        Observation(
            "/tmp/LSTM_Release --train --scheduler-experiment-id=420"));
    assert(ValidateManagedWorker(worker, prefixCollision).identity ==
           IdentityResult::IdentityValidationFailed);
    assert(prefixCollision.signals.empty());

    FakeProcesses reusedPid;
    ProcessObservation reusedObservation = Observation();
    reusedObservation.processStartIdentity = "1700000001:654321";
    reusedPid.observations.emplace(1200, reusedObservation);
    assert(ValidateManagedWorker(worker, reusedPid).identity ==
           IdentityResult::IdentityValidationFailed);
    assert(!CancelWorker(
                worker, false, std::chrono::milliseconds(1), reusedPid)
                .success);
    assert(reusedPid.signals.empty());

    FakeProcesses phaseCollision;
    phaseCollision.observations.emplace(
        1200,
        Observation(
            "/tmp/LSTM_Release --train-start=2020-01-01 "
            "--scheduler-experiment-id=42"));
    assert(ValidateManagedWorker(worker, phaseCollision).identity ==
           IdentityResult::IdentityValidationFailed);
    assert(phaseCollision.signals.empty());

    FakeProcesses missing;
    assert(ValidateManagedWorker(worker, missing).identity ==
           IdentityResult::ProcessMissing);
    assert(!ResumeWorker(worker, missing).success);
    assert(missing.signals.empty());

    FakeProcesses scheduler;
    scheduler.observations.emplace(
        1200,
        Observation(
            "/tmp/LSTM_Release --schedule-experiments "
            "--scheduler-experiment-id=42 --train"));
    assert(ValidateManagedWorker(worker, scheduler).identity ==
           IdentityResult::IdentityValidationFailed);
    assert(scheduler.signals.empty());

    FakeProcesses unsafeGroup;
    ProcessObservation unsafe = Observation();
    unsafe.processGroupId = unsafeGroup.callerGroup;
    unsafeGroup.observations.emplace(1200, unsafe);
    worker.processGroupId = unsafeGroup.callerGroup;
    assert(ValidateManagedWorker(worker, unsafeGroup).identity ==
           IdentityResult::UnsafeProcessGroup);
    assert(!CancelWorker(
                worker, false, std::chrono::milliseconds(1), unsafeGroup)
                .success);
    assert(unsafeGroup.signals.empty());

    ManagedWorker inactive = Worker();
    inactive.lifecycleStatus = "cancelled";
    assert(ValidateManagedWorker(inactive, processes).identity ==
           IdentityResult::StalePid);

    {
        FakeProcesses classifierProcesses;
        ManagedWorker normal = Worker();
        classifierProcesses.observations.emplace(
            normal.pid, ObservationFor(normal, *normal.commandLine));
        const auto classified = ClassifySchedulerWorkers(
            {Candidate(normal.pid, "train", *normal.commandLine)},
            {normal},
            classifierProcesses);
        assert(classified.size() == 1);
        assert(classified[0].managed);
        assert(classified[0].experimentId == normal.experimentId);
        assert(!classified[0].checkpointEvalId);
    }

    {
        FakeProcesses classifierProcesses;
        ManagedWorker normalInfer = Worker(42, "infer");
        normalInfer.commandLine =
            "/tmp/LSTM_Release --infer --model=1163 "
            "--scheduler-experiment-id=42";
        ManagedWorker normalAnalyze = Worker(43, "analyze");
        normalAnalyze.pid = 1203;
        normalAnalyze.processGroupId = 1203;
        normalAnalyze.commandLine =
            "/tmp/LSTM_Release --analyze-experiment=43";
        normalAnalyze.processStartIdentity = "1700000000:1203";
        classifierProcesses.observations.emplace(
            normalInfer.pid,
            ObservationFor(normalInfer, *normalInfer.commandLine));
        classifierProcesses.observations.emplace(
            normalAnalyze.pid,
            ObservationFor(normalAnalyze, *normalAnalyze.commandLine));
        const auto classified = ClassifySchedulerWorkers(
            {
                Candidate(
                    normalInfer.pid, "infer", *normalInfer.commandLine),
                Candidate(
                    normalAnalyze.pid, "analyze",
                    *normalAnalyze.commandLine),
            },
            {normalInfer, normalAnalyze},
            classifierProcesses);
        assert(classified.size() == 2);
        assert(classified[0].managed);
        assert(classified[1].managed);
        const auto summary = SummarizeSchedulerWorkers(classified);
        assert(summary.managedInfer.workers == 1);
        assert(summary.managedAnalyze.workers == 1);
        assert(summary.unmanagedInfer.workers == 0);
        assert(summary.unmanagedAnalyze.workers == 0);
    }

    {
        // The parent experiment lifecycle is intentionally absent from the
        // worker authority. A running checkpoint row remains authoritative
        // whether its parent experiment is completed or still running.
        FakeProcesses classifierProcesses;
        ManagedWorker completedParentCheckpoint =
            CheckpointWorker(250613, 440, 1201);
        ManagedWorker runningParentCheckpoint =
            CheckpointWorker(250614, 441, 1202);
        classifierProcesses.observations.emplace(
            1201,
            ObservationFor(
                completedParentCheckpoint,
                *completedParentCheckpoint.commandLine));
        classifierProcesses.observations.emplace(
            1202,
            ObservationFor(
                runningParentCheckpoint,
                *runningParentCheckpoint.commandLine));
        const auto classified = ClassifySchedulerWorkers(
            {
                Candidate(
                    1201, "infer",
                    *completedParentCheckpoint.commandLine),
                Candidate(
                    1202, "infer",
                    *runningParentCheckpoint.commandLine),
            },
            {completedParentCheckpoint, runningParentCheckpoint},
            classifierProcesses);
        assert(classified.size() == 2);
        assert(classified[0].managed);
        assert(classified[0].checkpointEvalId == 250613);
        assert(classified[1].managed);
        assert(classified[1].checkpointEvalId == 250614);
    }

    {
        FakeProcesses classifierProcesses;
        ManagedWorker known = CheckpointWorker(250613, 440, 1201);
        const std::string unknownCommand =
            "/tmp/LSTM_Release --infer --model=1162 "
            "--scheduler-checkpoint-eval-id=999999";
        classifierProcesses.observations.emplace(
            1201, ObservationFor(known, unknownCommand));
        const auto unknownId = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", unknownCommand)},
            {known},
            classifierProcesses);
        assert(!unknownId[0].managed);
        assert(
            unknownId[0].reason ==
            "no_matching_active_checkpoint_evaluation");

        const auto missingRow = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", unknownCommand)},
            {},
            classifierProcesses);
        assert(!missingRow[0].managed);
        assert(
            missingRow[0].reason ==
            "no_matching_active_checkpoint_evaluation");

        ManagedWorker ordinaryInfer = Worker(440, "infer");
        ordinaryInfer.pid = 1201;
        ordinaryInfer.processGroupId = 1201;
        ordinaryInfer.commandLine =
            "/tmp/LSTM_Release --infer --model=1162 "
            "--scheduler-experiment-id=440";
        ordinaryInfer.processStartIdentity = "1700000000:1201";
        const auto noExperimentFallback = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", unknownCommand)},
            {ordinaryInfer},
            classifierProcesses);
        assert(!noExperimentFallback[0].managed);
        assert(
            noExperimentFallback[0].reason ==
            "no_matching_active_checkpoint_evaluation");
    }

    {
        FakeProcesses classifierProcesses;
        ManagedWorker checkpoint = CheckpointWorker(250613, 440, 1201);
        classifierProcesses.observations.emplace(
            1201, ObservationFor(checkpoint, *checkpoint.commandLine));

        ManagedWorker pidMismatch = checkpoint;
        pidMismatch.pid = 1301;
        pidMismatch.processGroupId = 1301;
        const auto wrongPid = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", *checkpoint.commandLine)},
            {pidMismatch},
            classifierProcesses);
        assert(!wrongPid[0].managed);

        const auto wrongKind = ClassifySchedulerWorkers(
            {Candidate(1201, "train", *checkpoint.commandLine)},
            {checkpoint},
            classifierProcesses);
        assert(!wrongKind[0].managed);

        ManagedWorker completed = checkpoint;
        completed.lifecycleStatus = "completed";
        const auto completedRow = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", *checkpoint.commandLine)},
            {completed},
            classifierProcesses);
        assert(!completedRow[0].managed);
        assert(completedRow[0].identity == IdentityResult::StalePid);

        ManagedWorker failed = checkpoint;
        failed.lifecycleStatus = "failed";
        const auto failedRow = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", *checkpoint.commandLine)},
            {failed},
            classifierProcesses);
        assert(!failedRow[0].managed);
        assert(failedRow[0].identity == IdentityResult::StalePid);

        classifierProcesses.observations[1201].processStartIdentity =
            "1700009999:1201";
        const auto reused = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", *checkpoint.commandLine)},
            {checkpoint},
            classifierProcesses);
        assert(!reused[0].managed);
        assert(
            reused[0].identity ==
            IdentityResult::IdentityValidationFailed);

        classifierProcesses.observations[1201] =
            ObservationFor(checkpoint, *checkpoint.commandLine);
        ManagedWorker corruptCommand = checkpoint;
        corruptCommand.commandLine =
            "/tmp/LSTM_Release --infer "
            "--scheduler-checkpoint-eval-id=250999";
        const auto persistedCommandMismatch = ClassifySchedulerWorkers(
            {Candidate(1201, "infer", *checkpoint.commandLine)},
            {corruptCommand},
            classifierProcesses);
        assert(!persistedCommandMismatch[0].managed);
        assert(
            persistedCommandMismatch[0].reason ==
            "persisted_worker_command_line_identity_mismatch");
    }

    {
        FakeProcesses classifierProcesses;
        ManagedWorker normal = Worker(42, "infer");
        normal.commandLine =
            "/tmp/LSTM_Release --infer --model=1163 "
            "--scheduler-experiment-id=42";
        ManagedWorker checkpointA =
            CheckpointWorker(250613, 440, 1201);
        ManagedWorker checkpointB =
            CheckpointWorker(250614, 441, 1202);
        classifierProcesses.observations.emplace(
            normal.pid, ObservationFor(normal, *normal.commandLine));
        classifierProcesses.observations.emplace(
            checkpointA.pid,
            ObservationFor(checkpointA, *checkpointA.commandLine));
        classifierProcesses.observations.emplace(
            checkpointB.pid,
            ObservationFor(checkpointB, *checkpointB.commandLine));
        const auto classified = ClassifySchedulerWorkers(
            {
                Candidate(
                    normal.pid, "infer", *normal.commandLine,
                    10.0, 1.0, 100.0),
                Candidate(
                    checkpointA.pid, "infer",
                    *checkpointA.commandLine,
                    20.0, 2.0, 200.0),
                Candidate(
                    checkpointB.pid, "infer",
                    *checkpointB.commandLine,
                    30.0, 3.0, 300.0),
            },
            {normal, checkpointA, checkpointB},
            classifierProcesses);
        assert(classified.size() == 3);
        assert(std::count_if(
                   classified.begin(), classified.end(),
                   [](const SchedulerWorkerClassification& worker) {
                       return worker.managed;
                   }) == 3);
        assert(std::count_if(
                   classified.begin(), classified.end(),
                   [](const SchedulerWorkerClassification& worker) {
                       return worker.managed && worker.kind == "infer";
                   }) == 3);
        const auto summary = SummarizeSchedulerWorkers(classified);
        assert(summary.managedInfer.workers == 3);
        assert(summary.managedInfer.cpuPercent == 60.0);
        assert(summary.managedInfer.memPercent == 6.0);
        assert(summary.managedInfer.rssMb == 600.0);
        assert(summary.unmanagedInfer.workers == 0);
    }

    std::cout << "GlobalExperimentControlTests passed\n";
    return 0;
}
