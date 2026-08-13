#include <cassert>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "../Sources/SchedulerChildStatus.hpp"
#include "../Sources/WorkerLifecycleDiagnostics.hpp"

using EA::ExperimentScheduler::ChildStatusKind;
using EA::ExperimentScheduler::ObserveChildStatusNonBlocking;
using EA::ExperimentScheduler::WriteSchedulerChildExecFailureDiagnostic;
using EA::ExperimentScheduler::ChildLifecycleDiagnosticState;
using EA::ExperimentScheduler::LogWorkerStarted;
using EA::ExperimentScheduler::IsCurrentWorkerAttemptEvidence;
using EA::ExperimentScheduler::MarkChildExitDiagnosticEmitted;
using EA::ExperimentScheduler::MarkChildTerminationObserved;
using EA::ExperimentScheduler::SchedulerChildLaunchFailureDiagnostic;
using EA::ExperimentScheduler::SchedulerLaunchFailureError;
using EA::ExperimentScheduler::ShouldReconcileSchedulerOrphan;

pid_t ForkSleepingChild()
{
    const pid_t pid = ::fork();
    assert(pid >= 0);
    if (pid == 0)
    {
        for (;;)
            ::pause();
    }
    return pid;
}

void WaitUntilObserved(pid_t pid,
                       ChildStatusKind expectedKind,
                       int expectedValue)
{
    for (int attempt = 0; attempt < 1000; ++attempt)
    {
        const auto observed = ObserveChildStatusNonBlocking(pid);
        if (observed.kind == ChildStatusKind::Running)
        {
            ::usleep(1000);
            continue;
        }
        assert(observed.kind == expectedKind);
        if (expectedKind == ChildStatusKind::Exited)
            assert(observed.exitCode == expectedValue);
        else
        {
            assert(observed.signalNumber == expectedValue);
            assert(observed.exitCode == -expectedValue);
        }
        return;
    }
    assert(false && "child status was not observed");
}

int main()
{
    assert(SchedulerLaunchFailureError(ENOENT) == "launch_failed_errno_2");
    const std::string launchFailure = SchedulerChildLaunchFailureDiagnostic(
        143,
        "analyze",
        ENOENT,
        "/missing/worker --analyze-experiment=143");
    assert(launchFailure.find(
               "SCHEDULER_CHILD_LAUNCH_FAILED,experiment_id=143,phase=analyze,operation=analyze,errno=2") == 0);
    assert(launchFailure.find("error=No such file or directory") != std::string::npos);
    assert(launchFailure.find("command_line=/missing/worker --analyze-experiment=143") !=
           std::string::npos);

    ChildLifecycleDiagnosticState diagnosticState;
    assert(MarkChildTerminationObserved(diagnosticState));
    assert(!MarkChildTerminationObserved(diagnosticState));
    assert(MarkChildExitDiagnosticEmitted(diagnosticState));
    assert(!MarkChildExitDiagnosticEmitted(diagnosticState));

    assert(ShouldReconcileSchedulerOrphan(false, false, false));
    assert(!ShouldReconcileSchedulerOrphan(true, false, false));
    assert(!ShouldReconcileSchedulerOrphan(false, true, false));
    assert(!ShouldReconcileSchedulerOrphan(false, false, true));
    assert(!IsCurrentWorkerAttemptEvidence(99.999, 100.0));
    assert(IsCurrentWorkerAttemptEvidence(100.0, 100.0));
    assert(IsCurrentWorkerAttemptEvidence(100.001, 100.0));

    char startupPath[] = "/tmp/analysis_worker_startup_XXXXXX";
    const int startupFd = ::mkstemp(startupPath);
    assert(startupFd >= 0);
    assert(::close(startupFd) == 0);
    {
        std::ofstream startupLog(startupPath, std::ios::trunc);
        assert(startupLog.is_open());
        std::streambuf* original = std::cout.rdbuf(startupLog.rdbuf());
        LogWorkerStarted(
            "ANALYSIS_WORKER_STARTED",
            "analyze",
            143,
            645);
        std::cout.rdbuf(original);
    }
    std::ifstream startupLog(startupPath);
    const std::string startupText{
        std::istreambuf_iterator<char>(startupLog),
        std::istreambuf_iterator<char>()};
    assert(!startupText.empty());
    assert(startupText.find(
               "ANALYSIS_WORKER_STARTED,experiment_id=143,pid=") == 0);
    assert(startupText.find(",operation=analyze,model_id=645") != std::string::npos);
    assert(::unlink(startupPath) == 0);

    int startedExitPipe[2];
    assert(::pipe(startedExitPipe) == 0);
    pid_t startedExitPid = ::fork();
    assert(startedExitPid >= 0);
    if (startedExitPid == 0)
    {
        ::close(startedExitPipe[0]);
        assert(::dup2(startedExitPipe[1], STDOUT_FILENO) >= 0);
        ::close(startedExitPipe[1]);
        LogWorkerStarted("TRAINING_WORKER_STARTED", "train", 200);
        _exit(0);
    }
    assert(::close(startedExitPipe[1]) == 0);
    WaitUntilObserved(startedExitPid, ChildStatusKind::Exited, 0);
    char startedExitText[256] = {};
    const ssize_t startedExitSize =
        ::read(startedExitPipe[0], startedExitText, sizeof(startedExitText));
    assert(startedExitSize > 0);
    assert(::close(startedExitPipe[0]) == 0);
    assert(std::string(startedExitText, static_cast<size_t>(startedExitSize)).find(
               "TRAINING_WORKER_STARTED,experiment_id=200") == 0);

    int startedSignalPipe[2];
    assert(::pipe(startedSignalPipe) == 0);
    pid_t startedSignalPid = ::fork();
    assert(startedSignalPid >= 0);
    if (startedSignalPid == 0)
    {
        ::close(startedSignalPipe[0]);
        assert(::dup2(startedSignalPipe[1], STDOUT_FILENO) >= 0);
        ::close(startedSignalPipe[1]);
        LogWorkerStarted("INFERENCE_WORKER_STARTED", "infer", 201, 700);
        for (;;)
            ::pause();
    }
    assert(::close(startedSignalPipe[1]) == 0);
    char startedSignalText[256] = {};
    const ssize_t startedSignalSize =
        ::read(startedSignalPipe[0], startedSignalText, sizeof(startedSignalText));
    assert(startedSignalSize > 0);
    assert(std::string(startedSignalText, static_cast<size_t>(startedSignalSize)).find(
               "INFERENCE_WORKER_STARTED,experiment_id=201") == 0);
    assert(::kill(startedSignalPid, SIGTERM) == 0);
    WaitUntilObserved(startedSignalPid, ChildStatusKind::Signaled, SIGTERM);
    assert(::close(startedSignalPipe[0]) == 0);

    int diagnosticPipe[2];
    assert(::pipe(diagnosticPipe) == 0);
    WriteSchedulerChildExecFailureDiagnostic(
        diagnosticPipe[1], 143, "analyze", 7, ENOENT);
    assert(::close(diagnosticPipe[1]) == 0);
    char diagnostic[256] = {};
    const ssize_t diagnosticSize = ::read(
        diagnosticPipe[0], diagnostic, sizeof(diagnostic));
    assert(diagnosticSize > 0);
    assert(::close(diagnosticPipe[0]) == 0);
    assert(std::string(diagnostic, static_cast<size_t>(diagnosticSize)) ==
           "SCHEDULER_CHILD_EXEC_FAILED,experiment_id=143,phase=analyze,errno=2,error=no_such_file_or_directory\n");

    int execPipe[2];
    assert(::pipe(execPipe) == 0);
    pid_t execPid = ::fork();
    assert(execPid >= 0);
    if (execPid == 0)
    {
        ::close(execPipe[0]);
        assert(::dup2(execPipe[1], STDERR_FILENO) >= 0);
        ::close(execPipe[1]);
        char executable[] = "/definitely/not/a/scheduler/worker";
        char* const argv[] = {executable, nullptr};
        ::execv(executable, argv);
        const int execError = errno;
        WriteSchedulerChildExecFailureDiagnostic(
            STDERR_FILENO, 143, "analyze", 7, execError);
        _exit(127);
    }
    assert(::close(execPipe[1]) == 0);
    WaitUntilObserved(execPid, ChildStatusKind::Exited, 127);
    char execDiagnostic[256] = {};
    const ssize_t execDiagnosticSize = ::read(
        execPipe[0], execDiagnostic, sizeof(execDiagnostic));
    assert(execDiagnosticSize > 0);
    assert(::close(execPipe[0]) == 0);
    assert(std::string(execDiagnostic, static_cast<size_t>(execDiagnosticSize)).find(
               "SCHEDULER_CHILD_EXEC_FAILED,experiment_id=143,phase=analyze,errno=2,error=no_such_file_or_directory") !=
           std::string::npos);

    pid_t pid = ::fork();
    assert(pid >= 0);
    if (pid == 0)
        _exit(0);
    WaitUntilObserved(pid, ChildStatusKind::Exited, 0);

    pid = ::fork();
    assert(pid >= 0);
    if (pid == 0)
        _exit(1);
    WaitUntilObserved(pid, ChildStatusKind::Exited, 1);

    pid = ForkSleepingChild();
    assert(ObserveChildStatusNonBlocking(pid).kind == ChildStatusKind::Running);
    assert(::kill(pid, SIGTERM) == 0);
    WaitUntilObserved(pid, ChildStatusKind::Signaled, SIGTERM);

    pid = ForkSleepingChild();
    assert(::kill(pid, SIGKILL) == 0);
    WaitUntilObserved(pid, ChildStatusKind::Signaled, SIGKILL);

    const auto noLongerOwned = ObserveChildStatusNonBlocking(pid);
    assert(noLongerOwned.kind == ChildStatusKind::WaitError);
    assert(noLongerOwned.errorNumber == ECHILD);

    std::cout << "SchedulerChildStatusTests passed\n";
    return 0;
}
