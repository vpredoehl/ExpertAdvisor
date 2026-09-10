#include "WorkerProcessController.hpp"

#include "SchedulerExecutablePath.hpp"

#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sstream>
#include <unistd.h>

namespace EA::SchedulerCore
{

int WorkerProcessSignalNumber(WorkerProcessSignal signal) noexcept
{
    switch (signal)
    {
        case WorkerProcessSignal::Stop: return SIGSTOP;
        case WorkerProcessSignal::Resume: return SIGCONT;
        case WorkerProcessSignal::Terminate: return SIGTERM;
        case WorkerProcessSignal::Kill: return SIGKILL;
    }
    return 0;
}

std::string WorkerCommandLine(
    const std::vector<std::string>& arguments)
{
    std::ostringstream command;
    for (size_t index = 0; index < arguments.size(); ++index)
    {
        if (index != 0)
            command << ' ';
        command << arguments[index];
    }
    return command.str();
}

ProcessLiveness PosixWorkerProcessController::isAlive(
    pid_t pid,
    bool processGroup) const
{
    ProcessLiveness result;
    if (pid <= 0)
    {
        result.errorNumber = EINVAL;
        return result;
    }
    errno = 0;
    const pid_t target = processGroup ? -pid : pid;
    if (::kill(target, 0) == 0)
    {
        result.alive = true;
        return result;
    }
    result.errorNumber = errno;
    result.permissionDenied = errno == EPERM;
    result.alive = result.permissionDenied;
    return result;
}

ProcessSignalResult PosixWorkerProcessController::signal(
    pid_t pid,
    WorkerProcessSignal requestedSignal,
    bool processGroup) const
{
    ProcessSignalResult result;
    result.signalNumber = WorkerProcessSignalNumber(requestedSignal);
    if (pid <= 0 || result.signalNumber == 0)
    {
        result.errorNumber = EINVAL;
        return result;
    }
    errno = 0;
    result.success =
        ::kill(processGroup ? -pid : pid, result.signalNumber) == 0;
    if (!result.success)
        result.errorNumber = errno;
    return result;
}

EA::ExperimentScheduler::ObservedChildStatus
PosixWorkerProcessController::observeChild(pid_t pid) const
{
    return EA::ExperimentScheduler::ObserveChildStatusNonBlocking(pid);
}

SpawnResult PosixWorkerProcessController::spawn(
    const WorkerLaunchRequest& request) const
{
    SpawnResult result;
    if (request.arguments.empty())
    {
        result.failureStage = SpawnFailureStage::InvalidRequest;
        result.errorNumber = EINVAL;
        return result;
    }

    const int fd = ::open(
        request.logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
    {
        result.failureStage = SpawnFailureStage::OpenLog;
        result.errorNumber = errno;
        return result;
    }

    if (request.arguments.front().empty() ||
        request.arguments.front().front() != '/' ||
        ::access(request.arguments.front().c_str(), X_OK) != 0)
    {
        ::close(fd);
        result.failureStage = SpawnFailureStage::InvalidRequest;
        result.errorNumber = EINVAL;
        return result;
    }

    std::vector<char*> childArguments;
    childArguments.reserve(request.arguments.size() + 1);
    for (const std::string& argument : request.arguments)
        childArguments.push_back(const_cast<char*>(argument.c_str()));
    childArguments.push_back(nullptr);

    const char* const phaseData = request.phase.data();
    const size_t phaseLength = request.phase.size();
    const pid_t pid = ::fork();
    if (pid < 0)
    {
        result.failureStage = SpawnFailureStage::Fork;
        result.errorNumber = errno;
        ::close(fd);
        return result;
    }
    if (pid == 0)
    {
        if (request.createSession && ::setsid() < 0)
        {
            const int childError = errno;
            ::dup2(fd, STDOUT_FILENO);
            ::dup2(fd, STDERR_FILENO);
            ::close(fd);
            EA::ExperimentScheduler::WriteSchedulerChildExecFailureDiagnostic(
                STDERR_FILENO,
                request.experimentId,
                phaseData,
                phaseLength,
                childError);
            ::_exit(127);
        }
        ::signal(SIGHUP, SIG_IGN);
        ::dup2(fd, STDOUT_FILENO);
        ::dup2(fd, STDERR_FILENO);
        ::close(fd);
        ::execv(request.arguments.front().c_str(), childArguments.data());
        const int childError = errno;
        EA::ExperimentScheduler::WriteSchedulerChildExecFailureDiagnostic(
            STDERR_FILENO,
            request.experimentId,
            phaseData,
            phaseLength,
            childError);
        ::_exit(127);
    }

    ::close(fd);
    result.pid = pid;
    return result;
}

std::string PosixWorkerProcessController::resolveExecutablePath() const
{
    return EA::ExperimentScheduler::ResolveCanonicalExecutablePath();
}

WorkerProcessController& NativeWorkerProcessController()
{
    static PosixWorkerProcessController controller;
    return controller;
}

} // namespace EA::SchedulerCore
