#include <cassert>
#include <cerrno>
#include <csignal>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "../Sources/GlobalExperimentControl.hpp"

using namespace EA::GlobalExperimentControl;

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

    std::cout << "GlobalExperimentControlTests passed\n";
    return 0;
}
