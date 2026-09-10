#pragma once

#include "SchedulerChildStatus.hpp"

#include <string>
#include <vector>

#include <sys/types.h>

namespace EA::SchedulerCore
{

enum class WorkerProcessSignal
{
    Stop,
    Resume,
    Terminate,
    Kill
};

int WorkerProcessSignalNumber(WorkerProcessSignal signal) noexcept;

struct ProcessLiveness
{
    bool alive = false;
    bool permissionDenied = false;
    int errorNumber = 0;
};

struct ProcessSignalResult
{
    bool success = false;
    int signalNumber = 0;
    int errorNumber = 0;
};

struct WorkerLaunchRequest
{
    std::vector<std::string> arguments;
    std::string logPath;
    long long experimentId = -1;
    std::string phase;
    bool createSession = true;
};

enum class SpawnFailureStage
{
    None,
    InvalidRequest,
    OpenLog,
    Fork
};

struct SpawnResult
{
    pid_t pid = -1;
    SpawnFailureStage failureStage = SpawnFailureStage::None;
    int errorNumber = 0;

    [[nodiscard]] bool launched() const noexcept { return pid > 0; }
};

std::string WorkerCommandLine(const std::vector<std::string>& arguments);

class WorkerProcessController
{
public:
    virtual ~WorkerProcessController() = default;

    virtual ProcessLiveness isAlive(pid_t pid,
                                    bool processGroup = false) const = 0;
    virtual ProcessSignalResult signal(pid_t pid,
                                       WorkerProcessSignal signal,
                                       bool processGroup = false) const = 0;
    virtual EA::ExperimentScheduler::ObservedChildStatus observeChild(
        pid_t pid) const = 0;
    virtual SpawnResult spawn(const WorkerLaunchRequest& request) const = 0;
    virtual std::string resolveExecutablePath() const = 0;
};

class PosixWorkerProcessController final : public WorkerProcessController
{
public:
    ProcessLiveness isAlive(pid_t pid,
                            bool processGroup = false) const override;
    ProcessSignalResult signal(pid_t pid,
                               WorkerProcessSignal signal,
                               bool processGroup = false) const override;
    EA::ExperimentScheduler::ObservedChildStatus observeChild(
        pid_t pid) const override;
    SpawnResult spawn(const WorkerLaunchRequest& request) const override;
    std::string resolveExecutablePath() const override;
};

// Process-lifetime native controller used by the scheduler implementation.
// Tests can instantiate PosixWorkerProcessController directly or provide a
// WorkerProcessController implementation to extracted orchestration code.
WorkerProcessController& NativeWorkerProcessController();

} // namespace EA::SchedulerCore
