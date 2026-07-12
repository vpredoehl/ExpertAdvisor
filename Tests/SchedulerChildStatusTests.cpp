#include <cassert>
#include <csignal>
#include <iostream>
#include <string>
#include <unistd.h>

#include "../Sources/SchedulerChildStatus.hpp"

using EA::ExperimentScheduler::ChildStatusKind;
using EA::ExperimentScheduler::ObserveChildStatusNonBlocking;
using EA::ExperimentScheduler::WriteSchedulerChildExecFailureDiagnostic;

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
