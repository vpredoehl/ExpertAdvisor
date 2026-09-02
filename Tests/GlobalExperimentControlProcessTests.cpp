#include "../Sources/GlobalExperimentControl.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <libproc.h>
#include <memory>
#include <mutex>
#include <optional>
#include <poll.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace EA::GlobalExperimentControl;

namespace
{

volatile sig_atomic_t gTerminateRequested = 0;

void HandleTerm(int)
{
    gTerminateRequested = 1;
}

bool HasArgument(int argc, char* argv[], const std::string& expected)
{
    for (int i = 1; i < argc; ++i)
    {
        if (argv[i] == expected)
            return true;
    }
    return false;
}

std::optional<int> IntegerOption(int argc,
                                 char* argv[],
                                 const std::string& prefix)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument{argv[i]};
        if (argument.rfind(prefix, 0) == 0)
            return std::stoi(argument.substr(prefix.size()));
    }
    return std::nullopt;
}

bool ProcessGroupExists(pid_t processGroupId)
{
    errno = 0;
    return ::kill(-processGroupId, 0) == 0 || errno == EPERM;
}

int ProcessGroupMemberCount(pid_t processGroupId)
{
    std::vector<pid_t> members(16);
    const int count = ::proc_listpgrppids(
        processGroupId,
        members.data(),
        static_cast<int>(members.size() * sizeof(pid_t)));
    return std::max(0, count);
}

template <typename Predicate>
bool WaitUntil(Predicate predicate,
               std::chrono::milliseconds timeout =
                   std::chrono::milliseconds(3000))
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do
    {
        if (predicate())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}

std::string CanonicalExecutableIdentity(const std::string& path)
{
    char* resolved = ::realpath(path.c_str(), nullptr);
    if (resolved == nullptr)
        return {};
    std::string result{resolved};
    std::free(resolved);
    return result;
}

class EmergencyProcessGroupRegistry
{
public:
    enum class Role
    {
        Launcher,
        WorkerPid,
        WorkerGroup
    };

    struct Entry
    {
        pid_t pid = -1;
        pid_t processGroupId = -1;
        std::string processStartIdentity;
        Role role = Role::Launcher;
        std::string expectedExecutable;
        std::string registrationCommandMarker;
        std::string managedCommandMarker;
    };

    EmergencyProcessGroupRegistry()
        : callerPid_(::getpid()),
          callerProcessGroupId_(::getpgrp()),
          processes_(CreateNativeProcessOperations())
    {
    }

    bool AddLauncher(pid_t pid, const std::string& expectedCommandMarker)
    {
        return AddPid(pid, Role::Launcher, expectedCommandMarker);
    }

    bool AddPendingWorker(pid_t pid,
                          const std::string& expectedCommandMarker)
    {
        return AddPid(pid, Role::WorkerPid, expectedCommandMarker);
    }

    bool PromoteWorkerGroup(pid_t pid,
                            pid_t processGroupId,
                            const std::string& expectedCommandMarker)
    {
        if (!SafePid(pid) || !SafeProcessGroup(processGroupId) ||
            pid != processGroupId)
            return false;
        const auto entry = Find(pid);
        if (entry == entries_.end())
            return false;
        const ProcessObservation observation = processes_->Observe(pid);
        if (!Matches(*entry, observation, false) ||
            observation.processGroupId != processGroupId ||
            observation.commandLine.find("--managed-test-worker") ==
                std::string::npos ||
            observation.commandLine.find(expectedCommandMarker) ==
                std::string::npos)
            return false;
        entry->processGroupId = processGroupId;
        entry->role = Role::WorkerGroup;
        entry->managedCommandMarker = expectedCommandMarker;
        return true;
    }

    void Remove(pid_t pid)
    {
        Reap(pid);
        entries_.erase(
            std::remove_if(
                entries_.begin(),
                entries_.end(),
                [pid](const Entry& entry) { return entry.pid == pid; }),
            entries_.end());
    }

    bool Cleanup()
    {
        if (cleaning_)
            return false;
        cleaning_ = true;
        bool confirmed = true;
        for (const Entry& entry : entries_)
        {
            // SIGCONT is necessary before SIGTERM because a paused disposable
            // group cannot run its termination handler.
            if (entry.role == Role::WorkerGroup &&
                !SignalDisposable(entry, SIGCONT))
                confirmed = false;
            if (!SignalDisposable(entry, SIGTERM))
                confirmed = false;
        }
        WaitForAll(std::chrono::milliseconds(500));
        for (const Entry& entry : entries_)
        {
            if (!OriginalDisposableExited(entry) &&
                !SignalDisposable(entry, SIGKILL))
                confirmed = false;
        }
        WaitForAll(std::chrono::milliseconds(3000));
        for (const Entry& entry : entries_)
        {
            if (!OriginalDisposableExited(entry))
            {
                confirmed = false;
                std::cerr
                    << "Emergency cleanup could not confirm original "
                    << (entry.role == Role::Launcher ? "launcher " : "worker ")
                    << "exit for pid " << entry.pid
                    << " and process group " << entry.processGroupId << "\n";
            }
            Reap(entry.pid);
        }
        entries_.clear();
        cleaning_ = false;
        return confirmed;
    }

    size_t Size() const { return entries_.size(); }

private:
    using Iterator = std::vector<Entry>::iterator;

    bool SafePid(pid_t pid) const
    {
        return pid > 1 && pid != callerPid_ &&
               pid != callerProcessGroupId_;
    }

    bool SafeProcessGroup(pid_t processGroupId) const
    {
        return processGroupId > 1 &&
               processGroupId != callerPid_ &&
               processGroupId != callerProcessGroupId_;
    }

    Iterator Find(pid_t pid)
    {
        return std::find_if(
            entries_.begin(),
            entries_.end(),
            [pid](const Entry& entry) { return entry.pid == pid; });
    }

    bool AddPid(pid_t pid,
                Role role,
                const std::string& expectedCommandMarker)
    {
        if (!SafePid(pid) || expectedCommandMarker.empty())
            return false;
        const std::optional<std::string> startIdentity =
            ReadProcessStartIdentity(pid);
        if (!startIdentity || startIdentity->empty())
            return false;
        const ProcessObservation observation = processes_->Observe(pid);
        Entry proposed{
            pid,
            -1,
            *startIdentity,
            role,
            CanonicalExecutableIdentity(observation.executable),
            expectedCommandMarker,
            {}};
        if (!Matches(proposed, observation, false))
            return false;
        const auto existing = Find(pid);
        if (existing != entries_.end())
        {
            return existing->processStartIdentity ==
                       proposed.processStartIdentity &&
                   existing->expectedExecutable ==
                       proposed.expectedExecutable &&
                   existing->registrationCommandMarker ==
                       proposed.registrationCommandMarker &&
                   existing->role == proposed.role;
        }
        entries_.push_back(std::move(proposed));
        return true;
    }

    bool Matches(const Entry& entry,
                 const ProcessObservation& observation,
                 bool requireGroup) const
    {
        if (!SafePid(entry.pid) ||
            entry.processStartIdentity.empty() ||
            entry.expectedExecutable.empty() ||
            entry.registrationCommandMarker.empty() ||
            !observation.exists || observation.permissionDenied ||
            !observation.inspectionSucceeded ||
            observation.pid != entry.pid ||
            observation.processStartIdentity !=
                entry.processStartIdentity ||
            CanonicalExecutableIdentity(observation.executable) !=
                entry.expectedExecutable ||
            observation.commandLine.find(entry.registrationCommandMarker) ==
                std::string::npos ||
            (!entry.managedCommandMarker.empty() &&
             observation.commandLine.find(entry.managedCommandMarker) ==
                 std::string::npos) ||
            observation.commandLine.find("--schedule-experiments") !=
                std::string::npos ||
            observation.commandLine.find("--scheduler-status") !=
                std::string::npos)
            return false;
        if (!requireGroup)
            return true;
        return entry.role == Role::WorkerGroup &&
               SafeProcessGroup(entry.processGroupId) &&
               entry.processGroupId == entry.pid &&
               observation.processGroupId == entry.processGroupId &&
               observation.commandLine.find("--managed-test-worker") !=
                   std::string::npos;
    }

    bool SignalDisposable(const Entry& entry, int signalNumber)
    {
        const bool groupSignal = entry.role == Role::WorkerGroup;
        const ProcessObservation observation = processes_->Observe(entry.pid);
        if (!Matches(entry, observation, groupSignal))
            return !observation.exists &&
                   (groupSignal
                        ? !ProcessGroupExists(entry.processGroupId)
                        : true);
        errno = 0;
        const int result = groupSignal
            ? ::kill(-entry.processGroupId, signalNumber)
            : ::kill(entry.pid, signalNumber);
        return result == 0 || errno == ESRCH;
    }

    bool OriginalDisposableExited(const Entry& entry)
    {
        const ProcessObservation observation = processes_->Observe(entry.pid);
        if (observation.exists)
        {
            if (observation.processStartIdentity !=
                entry.processStartIdentity)
                return false;
            return false;
        }
        if (entry.role == Role::WorkerGroup &&
            ProcessGroupExists(entry.processGroupId))
            return false;
        return observation.inspectionSucceeded;
    }

    static void Reap(pid_t leader)
    {
        int status = 0;
        while (::waitpid(leader, &status, WNOHANG) > 0)
        {
        }
    }

    void WaitForAll(std::chrono::milliseconds timeout)
    {
        (void)WaitUntil(
            [&] {
                bool allExited = true;
                for (const Entry& entry : entries_)
                {
                    Reap(entry.pid);
                    if (!OriginalDisposableExited(entry))
                        allExited = false;
                }
                return allExited;
            },
            timeout);
    }

    const pid_t callerPid_;
    const pid_t callerProcessGroupId_;
    std::unique_ptr<ProcessOperations> processes_;
    std::vector<Entry> entries_;
    bool cleaning_ = false;
};

EmergencyProcessGroupRegistry& EmergencyRegistry()
{
    static EmergencyProcessGroupRegistry registry;
    return registry;
}

void EmergencyCleanupAtExit()
{
    (void)EmergencyRegistry().Cleanup();
}

[[noreturn]] void FailCheck(const char* expression,
                            const char* file,
                            int line)
{
    std::cerr << file << ":" << line << ": check failed: "
              << expression << "\n";
    const bool cleanupConfirmed = EmergencyRegistry().Cleanup();
    std::_Exit(cleanupConfirmed ? 1 : 2);
}

#define CHECK(expression) \
    do \
    { \
        if (!(expression)) \
            FailCheck(#expression, __FILE__, __LINE__); \
    } while (false)

struct ReadyMessage
{
    unsigned int magic = 0;
    unsigned int stage = 0;
    pid_t pid = -1;
    pid_t processGroupId = -1;
    pid_t memberPid = -1;
};

constexpr unsigned int kReadyMagic = 0x45415259;
constexpr unsigned int kGroupEstablishedStage = 1;
constexpr unsigned int kWorkerReadyStage = 2;

bool WriteAll(int descriptor, const void* data, size_t size)
{
    const char* cursor = static_cast<const char*>(data);
    while (size > 0)
    {
        const ssize_t bytes = ::write(descriptor, cursor, size);
        if (bytes < 0 && errno == EINTR)
            continue;
        if (bytes <= 0)
            return false;
        cursor += bytes;
        size -= static_cast<size_t>(bytes);
    }
    return true;
}

bool ReadAllWithTimeout(int descriptor,
                        void* data,
                        size_t size,
                        std::chrono::milliseconds timeout)
{
    char* cursor = static_cast<char*>(data);
    size_t remaining = size;
    while (remaining > 0)
    {
        pollfd readyPoll{descriptor, POLLIN, 0};
        const int pollResult =
            ::poll(&readyPoll, 1, static_cast<int>(timeout.count()));
        if (pollResult != 1 ||
            (readyPoll.revents & (POLLIN | POLLHUP)) == 0)
            return false;
        const ssize_t bytes = ::read(descriptor, cursor, remaining);
        if (bytes < 0 && errno == EINTR)
            continue;
        if (bytes <= 0)
            return false;
        cursor += bytes;
        remaining -= static_cast<size_t>(bytes);
    }
    return true;
}

bool ReadReadyMessage(int descriptor,
                      ReadyMessage& message,
                      std::chrono::milliseconds timeout)
{
    return ReadAllWithTimeout(
        descriptor, &message, sizeof(message), timeout);
}

[[noreturn]] void RunManagedTestWorker(int argc, char* argv[])
{
    const std::optional<int> readyDescriptor =
        IntegerOption(argc, argv, "--ready-fd=");
    const std::optional<int> groupReleaseDescriptor =
        IntegerOption(argc, argv, "--group-release-fd=");
    if (!readyDescriptor)
        _exit(3);
    if (HasArgument(argc, argv, "--self-session") && ::setsid() < 0)
        _exit(7);

    struct sigaction action{};
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    const bool ignoreTerm = HasArgument(argc, argv, "--ignore-term");
    action.sa_handler = ignoreTerm ? SIG_IGN : HandleTerm;
    if (::sigaction(SIGTERM, &action, nullptr) != 0)
        _exit(4);

    ReadyMessage ready;
    ready.magic = kReadyMagic;
    ready.pid = ::getpid();
    ready.processGroupId = ::getpgrp();
    if (ready.processGroupId != ready.pid)
        _exit(6);

    if (groupReleaseDescriptor)
    {
        ready.stage = kGroupEstablishedStage;
        if (!WriteAll(*readyDescriptor, &ready, sizeof(ready)))
            _exit(6);
        char release = '\0';
        if (::read(*groupReleaseDescriptor, &release, 1) != 1 ||
            release != 'G')
            _exit(8);
        ::close(*groupReleaseDescriptor);
    }

    pid_t memberPid = -1;
    int leaderLifetimePipe[2] = {-1, -1};
    if (HasArgument(argc, argv, "--spawn-group-member"))
    {
        if (::pipe(leaderLifetimePipe) != 0)
            _exit(5);
        memberPid = ::fork();
        if (memberPid < 0)
            _exit(5);
        if (memberPid == 0)
        {
            ::close(*readyDescriptor);
            ::close(leaderLifetimePipe[1]);
            while (ignoreTerm || !gTerminateRequested)
            {
                pollfd lifetimePoll{
                    leaderLifetimePipe[0], POLLIN | POLLHUP, 0};
                const int pollResult = ::poll(&lifetimePoll, 1, -1);
                if (pollResult > 0)
                {
                    char unused = '\0';
                    if (::read(leaderLifetimePipe[0], &unused, 1) <= 0)
                        break;
                }
                else if (pollResult < 0 && errno != EINTR)
                {
                    break;
                }
            }
            _exit(0);
        }
        ::close(leaderLifetimePipe[0]);
    }

    // Group ownership has been acknowledged before a member can be created.
    // This final message proves the complete managed-worker fixture is ready.
    ready.stage = kWorkerReadyStage;
    ready.memberPid = memberPid;
    if (HasArgument(argc, argv, "--suppress-final-ready"))
    {
        while (ignoreTerm || !gTerminateRequested)
            ::pause();
    }
    if (!WriteAll(*readyDescriptor, &ready, sizeof(ready)))
    {
        if (leaderLifetimePipe[1] >= 0)
            ::close(leaderLifetimePipe[1]);
        if (memberPid > 1)
        {
            (void)::kill(memberPid, SIGKILL);
            while (::waitpid(memberPid, nullptr, 0) < 0 && errno == EINTR)
            {
            }
        }
        _exit(6);
    }
    ::close(*readyDescriptor);

    while (ignoreTerm || !gTerminateRequested)
        ::pause();
    if (leaderLifetimePipe[1] >= 0)
        ::close(leaderLifetimePipe[1]);
    if (memberPid > 1)
    {
        while (::waitpid(memberPid, nullptr, 0) < 0 && errno == EINTR)
        {
        }
    }
    _exit(0);
}

std::string CanonicalSelfPath(const char* argv0)
{
    const std::string resolved = CanonicalExecutableIdentity(argv0);
    return resolved.empty() ? argv0 : resolved;
}

std::string ExecutableMarker(const std::string& path)
{
    const size_t separator = path.find_last_of('/');
    return separator == std::string::npos
        ? path
        : path.substr(separator + 1);
}

int PrintManagedTestProcessIdentity(int pid)
{
    std::unique_ptr<ProcessOperations> processes =
        CreateNativeProcessOperations();
    const ProcessObservation observation = processes->Observe(pid);
    const std::string executable =
        CanonicalExecutableIdentity(observation.executable);
    if (!observation.exists || !observation.inspectionSucceeded ||
        observation.permissionDenied || observation.pid != pid ||
        observation.processGroupId <= 1 ||
        observation.processStartIdentity.empty() || executable.empty())
        return 1;
    std::cout << observation.pid << "|"
              << observation.processGroupId << "|"
              << observation.processStartIdentity << "|"
              << executable << "|"
              << observation.commandLine << "\n";
    return 0;
}

enum class SpawnFailureMode
{
    None,
    LauncherRegistration,
    WorkerRegistration,
    PidPublication,
    FinalReadinessTimeout
};

ManagedWorker SpawnWorker(const std::string& selfPath,
                          ProcessOperations& processes,
                          long long experimentId = 42,
                          bool ignoreTerm = false,
                          bool spawnGroupMember = false,
                          SpawnFailureMode failureMode =
                              SpawnFailureMode::None,
                          std::optional<long long> checkpointEvalId =
                              std::nullopt)
{
    const size_t registrySizeBefore = EmergencyRegistry().Size();
    int pidPipe[2] = {-1, -1};
    int readyPipe[2] = {-1, -1};
    int launcherReleasePipe[2] = {-1, -1};
    int workerReleasePipe[2] = {-1, -1};
    int groupReleasePipe[2] = {-1, -1};
    CHECK(::pipe(pidPipe) == 0);
    CHECK(::pipe(readyPipe) == 0);
    CHECK(::pipe(launcherReleasePipe) == 0);
    CHECK(::pipe(workerReleasePipe) == 0);
    CHECK(::pipe(groupReleasePipe) == 0);
    const pid_t launcherPid = ::fork();
    CHECK(launcherPid >= 0);
    if (launcherPid == 0)
    {
        ::close(pidPipe[0]);
        ::close(readyPipe[0]);
        ::close(launcherReleasePipe[1]);
        ::close(workerReleasePipe[1]);
        ::close(groupReleasePipe[1]);
        char release = '\0';
        if (::read(launcherReleasePipe[0], &release, 1) != 1 ||
            release != 'L')
            _exit(125);
        ::close(launcherReleasePipe[0]);
        const pid_t workerPid = ::fork();
        if (workerPid == 0)
        {
            ::close(pidPipe[1]);
            char workerRelease = '\0';
            if (::read(workerReleasePipe[0], &workerRelease, 1) != 1 ||
                workerRelease != 'W')
                _exit(123);
            ::close(workerReleasePipe[0]);
            if (::setsid() < 0)
                _exit(126);
            std::vector<std::string> arguments{
                selfPath,
                "--managed-test-worker",
                "--scheduler-experiment-id=" + std::to_string(experimentId),
                "--ready-fd=" + std::to_string(readyPipe[1]),
                "--group-release-fd=" +
                    std::to_string(groupReleasePipe[0])};
            if (checkpointEvalId)
            {
                arguments.emplace_back("--infer");
                arguments.emplace_back(
                    "--scheduler-checkpoint-eval-id=" +
                    std::to_string(*checkpointEvalId));
            }
            else
            {
                arguments.emplace_back("--train");
            }
            if (ignoreTerm)
                arguments.emplace_back("--ignore-term");
            if (spawnGroupMember)
                arguments.emplace_back("--spawn-group-member");
            if (failureMode == SpawnFailureMode::FinalReadinessTimeout)
                arguments.emplace_back("--suppress-final-ready");
            std::vector<char*> childArgv;
            childArgv.reserve(arguments.size() + 1);
            for (std::string& argument : arguments)
                childArgv.push_back(argument.data());
            childArgv.push_back(nullptr);
            ::execv(selfPath.c_str(), childArgv.data());
            _exit(127);
        }
        ::close(workerReleasePipe[0]);
        ::close(groupReleasePipe[0]);
        const bool pidPublished =
            workerPid > 0 &&
            failureMode != SpawnFailureMode::PidPublication &&
            WriteAll(pidPipe[1], &workerPid, sizeof(workerPid));
        ::close(pidPipe[1]);
        ::close(readyPipe[1]);
        if (!pidPublished && workerPid > 0)
        {
            (void)::kill(workerPid, SIGTERM);
            while (::waitpid(workerPid, nullptr, 0) < 0 && errno == EINTR)
            {
            }
        }
        _exit(pidPublished ? 0 : 124);
    }

    ::close(launcherReleasePipe[0]);
    ::close(pidPipe[1]);
    ::close(readyPipe[1]);
    ::close(workerReleasePipe[0]);
    ::close(groupReleasePipe[0]);
    const bool launcherRegistered = EmergencyRegistry().AddLauncher(
        launcherPid,
        failureMode == SpawnFailureMode::LauncherRegistration
            ? "--not-the-launcher-command"
            : ExecutableMarker(selfPath));
    const char release = 'L';
    const bool launcherReleased =
        launcherRegistered &&
        WriteAll(launcherReleasePipe[1], &release, 1);
    ::close(launcherReleasePipe[1]);
    if (!launcherRegistered || !launcherReleased)
    {
        ::close(pidPipe[0]);
        ::close(readyPipe[0]);
        ::close(workerReleasePipe[1]);
        ::close(groupReleasePipe[1]);
        int launcherStatus = 0;
        CHECK(::waitpid(launcherPid, &launcherStatus, 0) == launcherPid);
        EmergencyRegistry().Remove(launcherPid);
        CHECK(launcherRegistered);
        CHECK(launcherReleased);
    }
    CHECK(launcherRegistered);
    CHECK(launcherReleased);

    pid_t workerPid = -1;
    const bool pidPublished = ReadAllWithTimeout(
        pidPipe[0],
        &workerPid,
        sizeof(workerPid),
        std::chrono::milliseconds(3000));
    ::close(pidPipe[0]);
    int launcherStatus = 0;
    CHECK(::waitpid(launcherPid, &launcherStatus, 0) == launcherPid);
    EmergencyRegistry().Remove(launcherPid);
    if (!pidPublished)
    {
        ::close(readyPipe[0]);
        ::close(workerReleasePipe[1]);
        ::close(groupReleasePipe[1]);
        CHECK(pidPublished);
    }
    CHECK(workerPid > 1);
    CHECK(WIFEXITED(launcherStatus) && WEXITSTATUS(launcherStatus) == 0);

    const bool workerRegistered = EmergencyRegistry().AddPendingWorker(
        workerPid,
        failureMode == SpawnFailureMode::WorkerRegistration
            ? "--not-the-worker-command"
            : ExecutableMarker(selfPath));
    if (!workerRegistered)
    {
        ::close(workerReleasePipe[1]);
        ::close(groupReleasePipe[1]);
        ::close(readyPipe[0]);
        CHECK(WaitUntil([&] {
            const ProcessObservation observation =
                processes.Observe(workerPid);
            return !observation.exists &&
                   observation.inspectionSucceeded;
        }));
        CHECK(workerRegistered);
    }
    CHECK(workerRegistered);
    const char workerRelease = 'W';
    const bool workerReleased =
        WriteAll(workerReleasePipe[1], &workerRelease, 1);
    ::close(workerReleasePipe[1]);
    CHECK(workerReleased);

    ReadyMessage groupEstablished;
    CHECK(ReadReadyMessage(
        readyPipe[0],
        groupEstablished,
        std::chrono::milliseconds(3000)));
    CHECK(groupEstablished.magic == kReadyMagic);
    CHECK(groupEstablished.stage == kGroupEstablishedStage);
    CHECK(groupEstablished.pid == workerPid);
    CHECK(groupEstablished.processGroupId == workerPid);
    CHECK(groupEstablished.memberPid == -1);
    CHECK(EmergencyRegistry().PromoteWorkerGroup(
        workerPid,
        groupEstablished.processGroupId,
        "--managed-test-worker"));
    CHECK(EmergencyRegistry().PromoteWorkerGroup(
        workerPid,
        groupEstablished.processGroupId,
        "--managed-test-worker"));
    CHECK(EmergencyRegistry().Size() == registrySizeBefore + 1);

    const char groupRelease = 'G';
    const bool groupReleased =
        WriteAll(groupReleasePipe[1], &groupRelease, 1);
    ::close(groupReleasePipe[1]);
    CHECK(groupReleased);

    ReadyMessage ready;
    const bool receivedReady = ReadReadyMessage(
        readyPipe[0], ready, std::chrono::milliseconds(3000));
    ::close(readyPipe[0]);
    CHECK(receivedReady);
    CHECK(ready.magic == kReadyMagic);
    CHECK(ready.stage == kWorkerReadyStage);
    CHECK(ready.pid == workerPid);
    CHECK(ready.processGroupId == workerPid);
    CHECK(!spawnGroupMember || ready.memberPid > 1);

    ProcessObservation observation;
    CHECK(WaitUntil([&] {
        observation = processes.Observe(workerPid);
        return observation.exists && observation.inspectionSucceeded &&
               observation.processGroupId == workerPid &&
               observation.commandLine.find("--managed-test-worker") !=
                   std::string::npos &&
               observation.commandLine.find(
                   "--scheduler-experiment-id=" +
                   std::to_string(experimentId)) != std::string::npos;
    }));
    if (spawnGroupMember)
        CHECK(ProcessGroupMemberCount(workerPid) >= 2);

    ManagedWorker worker;
    worker.experimentId = experimentId;
    worker.checkpointEvalId = checkpointEvalId;
    worker.phase = checkpointEvalId ? "checkpoint_infer" : "train";
    worker.lifecycleStatus = "running";
    worker.pid = static_cast<int>(workerPid);
    worker.processGroupId = observation.processGroupId;
    worker.executable = observation.executable;
    worker.commandLine = observation.commandLine;
    worker.processStartIdentity = observation.processStartIdentity;
    CHECK(ValidateManagedWorker(worker, processes).identity ==
          IdentityResult::Validated);
    return worker;
}

void AssertGroupExited(const ManagedWorker& worker)
{
    CHECK(worker.processGroupId);
    CHECK(WaitUntil(
        [&] { return !ProcessGroupExists(*worker.processGroupId); },
        std::chrono::milliseconds(3000)));
    EmergencyRegistry().Remove(*worker.processGroupId);
}

class RecordingNativeProcesses final : public ProcessOperations
{
public:
    RecordingNativeProcesses()
        : native_(CreateNativeProcessOperations())
    {
    }

    ProcessObservation Observe(int pid) override
    {
        return native_->Observe(pid);
    }

    bool SignalProcessGroup(int group, int signal, int& error) override
    {
        signals.emplace_back(group, signal);
        return native_->SignalProcessGroup(group, signal, error);
    }

    bool WaitForProcessGroupExit(
        int group,
        std::chrono::milliseconds timeout) override
    {
        return native_->WaitForProcessGroupExit(group, timeout);
    }

    int CallerPid() const override { return native_->CallerPid(); }
    int CallerProcessGroupId() const override
    {
        return native_->CallerProcessGroupId();
    }

    std::vector<std::pair<int, int>> signals;

private:
    std::unique_ptr<ProcessOperations> native_;
};

class InterleavingProcesses final : public ProcessOperations
{
public:
    InterleavingProcesses(ProcessOperations& delegate,
                          std::function<void()> beforeFirstSignal)
        : delegate_(delegate),
          beforeFirstSignal_(std::move(beforeFirstSignal))
    {
    }

    ProcessObservation Observe(int pid) override
    {
        return delegate_.Observe(pid);
    }

    bool SignalProcessGroup(int group, int signal, int& error) override
    {
        if (!interleaved_)
        {
            interleaved_ = true;
            beforeFirstSignal_();
        }
        return delegate_.SignalProcessGroup(group, signal, error);
    }

    bool WaitForProcessGroupExit(
        int group,
        std::chrono::milliseconds timeout) override
    {
        return delegate_.WaitForProcessGroupExit(group, timeout);
    }

    int CallerPid() const override { return delegate_.CallerPid(); }
    int CallerProcessGroupId() const override
    {
        return delegate_.CallerProcessGroupId();
    }

private:
    ProcessOperations& delegate_;
    std::function<void()> beforeFirstSignal_;
    bool interleaved_ = false;
};

class AfterSignalProcesses final : public ProcessOperations
{
public:
    AfterSignalProcesses(ProcessOperations& delegate,
                         std::function<void()> afterFirstSignal)
        : delegate_(delegate),
          afterFirstSignal_(std::move(afterFirstSignal))
    {
    }

    ProcessObservation Observe(int pid) override
    {
        return delegate_.Observe(pid);
    }

    bool SignalProcessGroup(int group, int signal, int& error) override
    {
        const bool signaled =
            delegate_.SignalProcessGroup(group, signal, error);
        if (!interleaved_)
        {
            interleaved_ = true;
            afterFirstSignal_();
        }
        return signaled;
    }

    bool WaitForProcessGroupExit(
        int group,
        std::chrono::milliseconds timeout) override
    {
        return delegate_.WaitForProcessGroupExit(group, timeout);
    }

    int CallerPid() const override { return delegate_.CallerPid(); }
    int CallerProcessGroupId() const override
    {
        return delegate_.CallerProcessGroupId();
    }

private:
    ProcessOperations& delegate_;
    std::function<void()> afterFirstSignal_;
    bool interleaved_ = false;
};

enum class ObservationFailureMode
{
    PermissionDenied,
    InspectionFailed
};

class ObservationFailureProcesses final : public ProcessOperations
{
public:
    ObservationFailureProcesses(ProcessOperations& delegate,
                                int targetPid,
                                ObservationFailureMode mode)
        : delegate_(delegate), targetPid_(targetPid), mode_(mode)
    {
    }

    ProcessObservation Observe(int pid) override
    {
        ProcessObservation observation = delegate_.Observe(pid);
        if (pid == targetPid_ && observation.exists)
        {
            observation.inspectionSucceeded = false;
            observation.permissionDenied =
                mode_ == ObservationFailureMode::PermissionDenied;
        }
        return observation;
    }

    bool SignalProcessGroup(int group, int signal, int& error) override
    {
        return delegate_.SignalProcessGroup(group, signal, error);
    }

    bool WaitForProcessGroupExit(
        int group,
        std::chrono::milliseconds timeout) override
    {
        return delegate_.WaitForProcessGroupExit(group, timeout);
    }

    int CallerPid() const override { return delegate_.CallerPid(); }
    int CallerProcessGroupId() const override
    {
        return delegate_.CallerProcessGroupId();
    }

private:
    ProcessOperations& delegate_;
    int targetPid_;
    ObservationFailureMode mode_;
};

std::string Scalar(pqxx::connection& connection, const std::string& query)
{
    pqxx::work transaction{connection};
    const pqxx::row row = transaction.exec(query).one_row();
    const std::string value =
        row[0].is_null() ? std::string{} : row[0].as<std::string>();
    transaction.commit();
    return value;
}

bool ReconcileActiveCancellationForTest(pqxx::work& transaction,
                                        long long requestId)
{
    const pqxx::row request = transaction.exec_params(
        "SELECT COALESCE(application_owner,invocation_identity) "
        "FROM experiment_admin_request WHERE request_id=$1;",
        requestId).one_row();
    return ReconcileActiveCancellation(
        transaction, request[0].as<std::string>(), true);
}

void ResetCrashFixtures(pqxx::connection& connection)
{
    pqxx::work transaction{connection};
    transaction.exec(
        "SELECT set_config("
        "'expertadvisor.scheduler_protocol_generation','52',true);");
    transaction.exec(
        "UPDATE experiment_global_control SET desired_state='running',"
        "active_request_id=NULL,current_pause_request_id=NULL WHERE singleton;");
    transaction.exec(
        "DELETE FROM experiment_admin_worker_outcome "
        "WHERE experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "UPDATE experiment_checkpoint_eval "
        "SET status=CASE WHEN status='running' THEN 'failed' ELSE status END,"
        "active_scheduler_worker_attempt_id=NULL "
        "WHERE experiment_id BETWEEN 700000 AND 700099 "
        "OR parent_experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "UPDATE experiment "
        "SET status=CASE WHEN status='running' THEN 'failed' ELSE status END,"
        "active_scheduler_worker_attempt_id=NULL "
        "WHERE experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "DELETE FROM experiment_scheduler_worker_attempt "
        "WHERE experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "DELETE FROM experiment_checkpoint_eval "
        "WHERE experiment_id BETWEEN 700000 AND 700099 "
        "OR parent_experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "DELETE FROM experiment_admin_request "
        "WHERE action='resume_experiment' "
        "AND target_experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "DELETE FROM experiment WHERE experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "DELETE FROM matrix WHERE model_id IN ("
        "SELECT model_id FROM model "
        "WHERE experiment_id BETWEEN 700000 AND 700099);");
    transaction.exec(
        "DELETE FROM model WHERE experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "DELETE FROM experiment_admin_request "
        "WHERE invocation_identity LIKE 'crash-window-%' "
        "OR invocation_identity LIKE 'selective-resume-%' "
        "OR invocation_identity LIKE 'resume-all-source-pause-%';");
    transaction.commit();
}

void InsertRunningExperiment(pqxx::connection& connection,
                             const ManagedWorker& worker,
                             const std::string& controlState = "running")
{
    pqxx::work transaction{connection};
    transaction.exec(
        "SELECT set_config("
        "'expertadvisor.scheduler_protocol_generation','52',true);");
    transaction.exec_params(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,worker_pid,worker_process_group_id,"
        "worker_executable,worker_command_line,worker_process_start_identity,"
        "worker_control_state,current_operation,worker_started_at) "
        "VALUES ($1,'running',$2,$3,$4,$5,$6,$7,$8,$2,now());",
        worker.experimentId,
        worker.phase,
        worker.pid,
        *worker.processGroupId,
        *worker.executable,
        *worker.commandLine,
        *worker.processStartIdentity,
        controlState);
    const long long attemptId = transaction.exec_params(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,"
        "capacity_class,ownership_origin,lifecycle_state,worker_pid,"
        "worker_process_group_id,worker_process_start_identity,"
        "canonical_executable_path,command_line,command_identity,"
        "reserved_at,spawned_at,registered_at,last_observed_at) "
        "VALUES($1,$2,'experiment',$3,$3,'legacy_unverified','observed',"
        "$4,$5,$6,$7,$8,$9,now(),now(),now(),now()) "
        "RETURNING worker_attempt_id;",
        "test:exact-attempt:experiment:" +
            std::to_string(worker.experimentId),
        worker.experimentId,
        worker.phase,
        worker.pid,
        *worker.processGroupId,
        *worker.processStartIdentity,
        *worker.executable,
        *worker.commandLine,
        "experiment:" + std::to_string(worker.experimentId) +
            ":" + worker.phase)[0][0].as<long long>();
    transaction.exec_params(
        "UPDATE experiment SET active_scheduler_worker_attempt_id=$1 "
        "WHERE experiment_id=$2;",
        attemptId,
        worker.experimentId);
    transaction.commit();
}

void BindPersistedFixtureAttempt(
    pqxx::transaction_base& transaction,
    long long experimentId,
    const std::string& phase)
{
    transaction.exec(
        "SELECT set_config("
        "'expertadvisor.scheduler_protocol_generation','52',true);");
    const long long attemptId = transaction.exec_params(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,"
        "capacity_class,ownership_origin,lifecycle_state,worker_pid,"
        "worker_process_group_id,worker_process_start_identity,"
        "canonical_executable_path,command_line,command_identity,"
        "reserved_at,spawned_at,registered_at,last_observed_at) "
        "SELECT $1,e.experiment_id,'experiment',$2,$2,"
        "'legacy_unverified','observed',e.worker_pid,"
        "e.worker_process_group_id,e.worker_process_start_identity,"
        "e.worker_executable,e.worker_command_line,$3,"
        "now(),now(),now(),now() "
        "FROM experiment e WHERE e.experiment_id=$4 "
        "RETURNING worker_attempt_id;",
        "test:persisted-fixture-attempt:" +
            std::to_string(experimentId),
        phase,
        "experiment:" + std::to_string(experimentId) + ":" + phase,
        experimentId)[0][0].as<long long>();
    transaction.exec_params(
        "UPDATE experiment SET active_scheduler_worker_attempt_id=$1 "
        "WHERE experiment_id=$2;",
        attemptId,
        experimentId);
}

void InsertRunningCheckpointChild(pqxx::connection& connection,
                                  const ManagedWorker& worker)
{
    CHECK(worker.checkpointEvalId);
    pqxx::work transaction{connection};
    transaction.exec(
        "SELECT set_config("
        "'expertadvisor.scheduler_protocol_generation','52',true);");
    const long long checkpointModelId = transaction.exec_params(
        "INSERT INTO model(experiment_id,comment) "
        "VALUES ($1,'selective resume checkpoint child fixture') "
        "RETURNING model_id;",
        worker.experimentId)[0][0].as<long long>();
    transaction.exec_params(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "VALUES ($1,'train_config_meta',0,10,20);",
        checkpointModelId);
    transaction.exec_params(
        "INSERT INTO experiment_checkpoint_eval ("
        "checkpoint_eval_id,experiment_id,parent_experiment_id,"
        "checkpoint_epoch,checkpoint_model_id,status,phase,worker_pid,"
        "worker_process_group_id,worker_executable,worker_command_line,"
        "worker_process_start_identity,worker_control_state) "
        "VALUES ($1,$2,$2,20,$3,'running','infer',$4,$5,$6,$7,$8,'running');",
        *worker.checkpointEvalId,
        worker.experimentId,
        checkpointModelId,
        worker.pid,
        *worker.processGroupId,
        *worker.executable,
        *worker.commandLine,
        *worker.processStartIdentity);
    const long long attemptId = transaction.exec_params(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "launch_attempt_identity,experiment_id,checkpoint_eval_id,"
        "worker_kind,lifecycle_phase,capacity_class,ownership_origin,"
        "lifecycle_state,worker_pid,worker_process_group_id,"
        "worker_process_start_identity,canonical_executable_path,"
        "command_line,command_identity,reserved_at,spawned_at,"
        "registered_at,last_observed_at) "
        "VALUES($1,$2,$3,'checkpoint_infer','infer','infer',"
        "'legacy_unverified','observed',$4,$5,$6,$7,$8,$9,"
        "now(),now(),now(),now()) RETURNING worker_attempt_id;",
        "test:exact-attempt:checkpoint:" +
            std::to_string(*worker.checkpointEvalId),
        worker.experimentId,
        *worker.checkpointEvalId,
        worker.pid,
        *worker.processGroupId,
        *worker.processStartIdentity,
        *worker.executable,
        *worker.commandLine,
        "checkpoint_infer:" +
            std::to_string(*worker.checkpointEvalId))[0][0]
            .as<long long>();
    transaction.exec_params(
        "UPDATE experiment_checkpoint_eval "
        "SET active_scheduler_worker_attempt_id=$1 "
        "WHERE checkpoint_eval_id=$2;",
        attemptId,
        *worker.checkpointEvalId);
    transaction.commit();
}

long long SeedRequest(pqxx::connection& connection,
                      Action action,
                      const ManagedWorker& worker,
                      const std::string& outcomeStatus,
                      const std::string& workerControlState = "running")
{
    InsertRunningExperiment(connection, worker, workerControlState);
    pqxx::work transaction{connection};
    const std::string actionName = ToString(action);
    const std::string resultingState =
        action == Action::PauseAll ? "paused" : "running";
    const std::optional<std::string> cancellationMode =
        action == Action::CancelAll
            ? std::optional<std::string>{"immediate"}
            : std::nullopt;
    const std::string invocationIdentity =
        "crash-window-" + actionName + "-" +
        std::to_string(worker.experimentId);
    const long long requestId = transaction.exec_params(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,"
        "invocation_identity,requester_identity,application_owner,"
        "application_lease_until,status,previous_global_state,"
        "resulting_global_state,scheduler_running_observed,target_count) "
        "VALUES ($1,$2,false,$3,'crash-fixture-requester',$3,"
        "now()-interval '1 second','applying','running',$4,false,1) "
        "RETURNING request_id;",
        actionName,
        cancellationMode,
        invocationIdentity,
        resultingState)[0][0].as<long long>();
    std::optional<long long> sourcePauseRequestId;
    if (action == Action::ResumeAll)
    {
        sourcePauseRequestId = transaction.exec_params(
            "INSERT INTO experiment_admin_request ("
            "action,invocation_identity,status,completed_at,"
            "previous_global_state,resulting_global_state,target_count,"
            "successful_count) "
            "VALUES ('pause_all',$1,'completed',now(),'running','paused',1,1) "
            "RETURNING request_id;",
            "resume-all-source-pause-" +
                std::to_string(worker.experimentId))[0][0].as<long long>();
        transaction.exec_params(
            "UPDATE experiment SET worker_global_pause_request_id=$1 "
            "WHERE experiment_id=$2;",
            *sourcePauseRequestId,
            worker.experimentId);
    }
    transaction.exec_params(
        "UPDATE experiment_global_control SET desired_state=$1,"
        "active_request_id=$2,current_pause_request_id=$3,"
        "revision=revision+1,updated_at=now() "
        "WHERE singleton;",
        resultingState,
        requestId,
        action == Action::PauseAll
            ? std::optional<long long>{requestId}
            : sourcePauseRequestId);
    if (action == Action::CancelAll)
        transaction.exec_params(
            "UPDATE experiment SET cancellation_request_id=$1,"
            "cancel_infer_before=false,updated_at=now() "
            "WHERE experiment_id=$2;",
            requestId,
            worker.experimentId);
    const std::string detail =
        action == Action::CancelAll
            ? "immediate_cancellation"
            : "worker_validation_and_signal_planned";
    transaction.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,checkpoint_eval_id,"
        "worker_kind,phase,"
        "lifecycle_status,worker_pid,worker_process_group_id,"
        "worker_process_start_identity,worker_executable,worker_command_line,"
        "worker_attempt_id,cancellation_checkpoint_epoch,"
        "source_pause_request_id,inference_action,outcome_status,detail) "
        "VALUES ("
        "$1,$2,$3,NULL,'experiment',$4,'running',$5,$6,$7,$8,$9,"
        "(SELECT active_scheduler_worker_attempt_id FROM experiment "
        " WHERE experiment_id=$3),NULL,$10,"
        "'none',$11,$12);",
        requestId,
        "experiment:" + std::to_string(worker.experimentId),
        worker.experimentId,
        worker.phase,
        worker.pid,
        *worker.processGroupId,
        *worker.processStartIdentity,
        *worker.executable,
        *worker.commandLine,
        sourcePauseRequestId,
        outcomeStatus,
        detail);
    transaction.commit();

    CHECK(Scalar(
              connection,
              "SELECT action||':'||COALESCE(cancellation_mode,'NULL')||':'||"
              "infer_before_cancel::text||':'||invocation_identity||':'||"
              "requester_identity||':'||application_owner||':'||status||':'||"
              "previous_global_state||':'||resulting_global_state||':'||"
              "scheduler_running_observed::text||':'||target_count::text||':'||"
              "successful_count::text||':'||already_satisfied_count::text||':'||"
              "missing_count::text||':'||rejected_count::text||':'||"
              "failed_count::text||':'||result_summary::text||':'||"
              "(requested_at IS NOT NULL)::text||':'||"
              "(completed_at IS NULL)::text||':'||"
              "(application_lease_until<now())::text "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          actionName + ":" +
              (cancellationMode ? *cancellationMode : "NULL") +
              ":false:" + invocationIdentity +
              ":crash-fixture-requester:" + invocationIdentity +
              ":applying:running:" + resultingState +
              ":false:1:0:0:0:0:0:{}:true:true:true");
    CHECK(Scalar(
              connection,
              "SELECT desired_state||':'||active_request_id::text "
              "FROM experiment_global_control WHERE singleton") ==
          resultingState + ":" + std::to_string(requestId));
    if (action == Action::CancelAll)
        CHECK(Scalar(
                  connection,
                  "SELECT cancellation_request_id::text||':'||"
                  "cancel_infer_before::text FROM experiment "
                  "WHERE experiment_id=" +
                      std::to_string(worker.experimentId)) ==
              std::to_string(requestId) + ":false");
    return requestId;
}

Command ReplayCommand(
    Action action,
    const std::string& identity,
    CancellationMode cancellationMode = CancellationMode::Immediate)
{
    Command command;
    command.action = action;
    if (action == Action::CancelAll)
        command.cancellationMode = cancellationMode;
    command.confirmed = true;
    command.invocationIdentity = identity;
    command.terminationGrace = std::chrono::milliseconds(300);
    return command;
}

int Replay(const std::string& connectionString,
           const Command& command,
           RecordingNativeProcesses& processes,
           std::string& outputText)
{
    std::ostringstream output;
    std::ostringstream error;
    const int result = RunCommandWithProcessOperationsForTesting(
        connectionString, command, output, error, processes);
    outputText = output.str();
    if (!error.str().empty())
        std::cerr << error.str();
    return result;
}

void CheckStableIdentities(pqxx::connection& connection,
                           long long requestId,
                           long long experimentId)
{
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity LIKE 'crash-window-%'") == "1");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE request_id=" + std::to_string(requestId)) == "1");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_worker_outcome "
              "WHERE request_id=" + std::to_string(requestId)) == "1");
    CHECK(Scalar(
              connection,
              "SELECT worker_identity FROM experiment_admin_worker_outcome "
              "WHERE request_id=" + std::to_string(requestId)) ==
          "experiment:" + std::to_string(experimentId));
}

void CleanupWorker(const ManagedWorker& worker, ProcessOperations& processes)
{
    if (processes.Observe(worker.pid).exists)
    {
        const SignalOutcome cleanup = CancelWorker(
            worker, true, std::chrono::milliseconds(300), processes);
        CHECK(cleanup.success);
    }
    AssertGroupExited(worker);
}

void TestCancellationAfterSignalBeforeAccounting(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    RecordingNativeProcesses processes;
    const ManagedWorker worker =
        SpawnWorker(selfPath, processes, 700004);
    const long long requestId =
        SeedRequest(connection, Action::CancelAll, worker, "planned");

    const SignalOutcome terminated = CancelWorker(
        worker, false, std::chrono::milliseconds(300), processes);
    CHECK(terminated.success);
    CHECK(terminated.signals == std::vector<int>{SIGTERM});
    CHECK((processes.signals ==
           std::vector<std::pair<int, int>>{
               {*worker.processGroupId, SIGTERM}}));
    AssertGroupExited(worker);
    processes.signals.clear();

    std::string replayOutput;
    CHECK(Replay(
              connectionString,
              ReplayCommand(
                  Action::CancelAll, "crash-window-replay-terminated"),
              processes,
              replayOutput) == 0);
    CHECK(processes.signals.empty());
    CheckStableIdentities(connection, requestId, worker.experimentId);
    CHECK(Scalar(
              connection,
              "SELECT identity_result||':'||signal_result||':'||"
              "outcome_status FROM experiment_admin_worker_outcome "
              "WHERE request_id=" + std::to_string(requestId)) ==
          "process_missing:process_missing:completed");
    CHECK(Scalar(
              connection,
              "SELECT status FROM experiment WHERE "
              "experiment_id=700004") == "cancelled");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL FROM "
              "experiment_global_control WHERE singleton") == "t");

    ResetCrashFixtures(connection);
}

void TestAccountedBeforeReconciliation(const std::string& connectionString,
                                       pqxx::connection& connection)
{
    pqxx::work planTransaction{connection};
    const long long requestId = planTransaction.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,application_owner,application_lease_until,status,"
        "requested_at,"
        "previous_global_state,resulting_global_state,"
        "scheduler_running_observed,target_count) VALUES ("
        "'cancel_all','after_next_checkpoint',false,"
        "'crash-window-accounted','crash-fixture-requester',"
        "'crash-window-accounted',now()-interval '3 seconds','applying',"
        "now()-interval '33 seconds',"
        "'running','running',false,2) RETURNING request_id;")[0][0].as<long long>();
    planTransaction.exec_params(
        "UPDATE experiment_global_control SET desired_state='running',"
        "active_request_id=$1,revision=revision+1,"
        "updated_at=now()-interval '33 seconds' "
        "WHERE singleton;",
        requestId);
    planTransaction.exec_params(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,current_epoch,checkpoint_interval,"
        "target_epochs,worker_pid,worker_process_group_id,worker_executable,"
        "worker_command_line,worker_process_start_identity,"
        "worker_control_state,worker_started_at,cancellation_request_id,"
        "cancel_after_checkpoint_epoch,stop_after_checkpoint_epoch,"
        "current_operation,error_message,updated_at) VALUES ("
        "700010,'running','infer',10,20,100,910010,910010,'LSTM_Release',"
        "'LSTM_Release --infer --scheduler-experiment-id=700010',"
        "'1700000010:10','running',now()-interval '34 seconds',$1,"
        "NULL,NULL,'infer',NULL,now()-interval '33 seconds'),"
        "(700011,'running','train',10,20,100,910011,910011,'LSTM_Release',"
        "'LSTM_Release --train --scheduler-experiment-id=700011',"
        "'1700000011:11','running',now()-interval '34 seconds',$1,"
        "20,20,'train',NULL,now()-interval '33 seconds');",
        requestId);
    const long long restartModelId = planTransaction.exec(
        "INSERT INTO model(experiment_id,comment) VALUES "
        "(700011,'periodic training checkpoint') RETURNING model_id;")
                                         [0][0]
                                             .as<long long>();
    planTransaction.exec_params(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "VALUES ($1,'train_config_meta',0,10,10);",
        restartModelId);
    planTransaction.exec_params(
        "UPDATE experiment SET last_model_id=$1 "
        "WHERE experiment_id=700011;",
        restartModelId);
    planTransaction.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,worker_kind,phase,"
        "lifecycle_status,worker_pid,worker_process_group_id,"
        "worker_process_start_identity,requested_signal,"
        "signal_result,cancellation_checkpoint_epoch,inference_action,"
        "outcome_status,detail,created_at,updated_at) VALUES ("
        "$1,'experiment:700010',700010,'experiment','infer','running',"
        "910010,910010,'1700000010:10',NULL,'not_attempted',NULL,"
        "'none','planned','immediate_cancellation',"
        "now()-interval '33 seconds',now()-interval '33 seconds'),"
        "($1,'experiment:700011',700011,'experiment','train','running',"
        "910011,910011,'1700000011:11',NULL,'not_attempted',20,'none',"
        "'pending_checkpoint','continue_to_next_durable_checkpoint',"
        "now()-interval '33 seconds',now()-interval '33 seconds');",
        requestId);
    planTransaction.commit();

    // Mirror the immediate-cancellation accounting transaction after one
    // second. The request lease, experiment materialization, and outcome update
    // retain the exact transaction timestamp relationships.
    pqxx::work immediateAccountingTransaction{connection};
    const std::string accountedAt = immediateAccountingTransaction.exec(
        "SELECT (now()-interval '32 seconds')::text;")[0][0].as<std::string>();
    immediateAccountingTransaction.exec_params(
        "UPDATE experiment_admin_request SET "
        "application_lease_until=$1::timestamptz+interval '30 seconds' "
        "WHERE request_id=$2 AND application_owner='crash-window-accounted';",
        accountedAt,
        requestId);
    immediateAccountingTransaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "identity_result='validated',signal_result='signaled',"
        "requested_signal=$1,outcome_status='completed',detail='validated',"
        "updated_at=$2 WHERE request_id=$3 "
        "AND worker_identity='experiment:700010';",
        std::to_string(SIGTERM),
        accountedAt,
        requestId);
    immediateAccountingTransaction.exec_params(
        "UPDATE experiment SET status='cancelled',"
        "completed_at=COALESCE(completed_at,$1),"
        "cancellation_completed_at=$1,worker_pid=NULL,"
        "worker_process_group_id=NULL,"
        "current_operation='infer',"
        "error_message='cancelled_by_global_request',updated_at=$1 "
        "WHERE experiment_id=700010 AND status='running';",
        accountedAt);
    immediateAccountingTransaction.commit();

    // Mirror the subsequent missing-worker checkpoint-requeue transaction.
    // Its later lease refresh is the active request boundary observed on replay.
    pqxx::work missingAccountingTransaction{connection};
    const std::string missingAccountedAt = missingAccountingTransaction.exec(
        "SELECT (now()-interval '31 seconds')::text;")[0][0].as<std::string>();
    missingAccountingTransaction.exec_params(
        "UPDATE experiment_admin_request SET "
        "application_lease_until=$1::timestamptz+interval '30 seconds' "
        "WHERE request_id=$2 AND application_owner='crash-window-accounted';",
        missingAccountedAt,
        requestId);
    missingAccountingTransaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "identity_result='process_missing',signal_result='process_missing',"
        "detail='missing_worker_requeued_from_durable_checkpoint',"
        "outcome_status='pending_checkpoint',updated_at=$1 "
        "WHERE request_id=$2 AND worker_identity='experiment:700011';",
        missingAccountedAt,
        requestId);
    missingAccountingTransaction.exec_params(
        "UPDATE experiment SET status='pending',phase='train',worker_pid=NULL,"
        "worker_process_group_id=NULL,worker_control_state='running',"
        "last_model_id=$1,"
        "current_operation='train',"
        "error_message='cancellation_worker_restart_required',updated_at=$2 "
        "WHERE experiment_id=700011 AND status='running';",
        restartModelId,
        missingAccountedAt);
    missingAccountingTransaction.commit();

    CHECK(Scalar(
              connection,
              "SELECT action||':'||cancellation_mode||':'||"
              "infer_before_cancel::text||':'||requester_identity||':'||"
              "application_owner||':'||status||':'||previous_global_state||':'||"
              "resulting_global_state||':'||"
              "scheduler_running_observed::text||':'||target_count::text||':'||"
              "successful_count::text||':'||already_satisfied_count::text||':'||"
              "missing_count::text||':'||rejected_count::text||':'||"
              "failed_count::text||':'||result_summary::text||':'||"
              "(requested_at IS NOT NULL)::text||':'||"
              "(completed_at IS NULL)::text||':'||"
              "(application_lease_until<now())::text "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "cancel_all:after_next_checkpoint:false:crash-fixture-requester:"
          "crash-window-accounted:applying:running:running:"
          "false:2:0:0:0:0:0:{}:true:true:true");
    CHECK(Scalar(
              connection,
              "SELECT desired_state||':'||active_request_id::text "
              "FROM experiment_global_control WHERE singleton") ==
          "running:" + std::to_string(requestId));
    CHECK(Scalar(
              connection,
              "SELECT bool_and(cancellation_request_id=" +
                  std::to_string(requestId) +
                  ") FROM experiment WHERE experiment_id IN (700010,700011)") ==
          "t");
    CHECK(Scalar(
              connection,
              "SELECT status||':'||phase||':'||"
              "(worker_pid IS NULL)::text||':'||"
              "(worker_process_group_id IS NULL)::text||':'||"
              "worker_executable||':'||worker_command_line||':'||"
              "worker_process_start_identity||':'||worker_control_state||':'||"
              "(worker_started_at IS NOT NULL)::text||':'||"
              "last_model_id::text||':'||"
              "(resume_model_id IS NULL)::text||':'||"
              "current_operation||':'||error_message||':'||"
              "cancel_after_checkpoint_epoch::text||':'||"
              "stop_after_checkpoint_epoch::text "
              "FROM experiment WHERE experiment_id=700011") ==
          "pending:train:true:true:LSTM_Release:"
          "LSTM_Release --train --scheduler-experiment-id=700011:"
          "1700000011:11:running:true:" + std::to_string(restartModelId) +
              ":true:train:"
              "cancellation_worker_restart_required:20:20");
    CHECK(Scalar(
              connection,
              "WITH cfg AS (SELECT model_id,max(value) FILTER "
              "(WHERE col_idx=10) AS completed_epochs FROM matrix "
              "WHERE param_name='train_config_meta' AND row_idx=0 "
              "GROUP BY model_id) SELECT m.model_id::text FROM model m "
              "LEFT JOIN cfg ON cfg.model_id=m.model_id "
              "WHERE m.experiment_id=700011 ORDER BY "
              "cfg.completed_epochs DESC NULLS LAST,m.model_id DESC LIMIT 1") ==
          std::to_string(restartModelId));
    CHECK(Scalar(
              connection,
              "SELECT worker_pid::text||':'||worker_process_group_id::text||':'||"
              "worker_process_start_identity||':'||phase||':'||"
              "lifecycle_status||':'||COALESCE(requested_signal,'NULL')||':'||"
              "identity_result||':'||signal_result||':'||outcome_status "
              "FROM experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId) +
                  " AND worker_identity='experiment:700010'") ==
          "910010:910010:1700000010:10:infer:running:" +
              std::to_string(SIGTERM) + ":validated:signaled:completed");
    CHECK(Scalar(
              connection,
              "SELECT e.status||':'||e.phase||':'||"
              "(e.worker_pid IS NULL)::text||':'||"
              "(e.worker_process_group_id IS NULL)::text||':'||"
              "e.worker_control_state||':'||"
              "(e.worker_started_at IS NOT NULL)::text||':'||"
              "e.current_operation||':'||e.error_message||':'||"
              "(e.completed_at=e.cancellation_completed_at)::text||':'||"
              "(e.completed_at=e.updated_at)::text||':'||"
              "(e.completed_at=o.updated_at)::text||':'||"
              "(o.created_at<o.updated_at)::text||':'||"
              "(r.requested_at<o.updated_at)::text||':'||"
              "(o.updated_at<pending.updated_at)::text||':'||"
              "(r.application_lease_until="
              "pending.updated_at+interval '30 seconds')::text "
              "FROM experiment e "
              "JOIN experiment_admin_worker_outcome o "
              "ON o.experiment_id=e.experiment_id "
              "JOIN experiment_admin_worker_outcome pending "
              "ON pending.request_id=o.request_id "
              "AND pending.worker_identity='experiment:700011' "
              "JOIN experiment_admin_request r "
              "ON r.request_id=o.request_id "
              "WHERE e.experiment_id=700010 AND r.request_id=" +
                  std::to_string(requestId)) ==
          "cancelled:infer:true:true:running:true:"
          "infer:cancelled_by_global_request:"
          "true:true:true:true:true:true:true");
    CHECK(Scalar(
              connection,
              "SELECT cancellation_checkpoint_epoch::text||':'||"
              "outcome_status||':'||detail FROM "
              "experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId) +
                  " AND worker_identity='experiment:700011'") ==
          "20:pending_checkpoint:"
          "missing_worker_requeued_from_durable_checkpoint");

    RecordingNativeProcesses processes;
    std::string replayOutput;
    CHECK(Replay(
              connectionString,
              ReplayCommand(
                  Action::CancelAll,
                  "crash-window-replay-accounted-pending",
                  CancellationMode::AfterNextCheckpoint),
              processes,
              replayOutput) == 0);
    CHECK(processes.signals.empty());
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity LIKE 'crash-window-%'") == "1");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id::text FROM "
              "experiment_global_control WHERE singleton") ==
          std::to_string(requestId));
    CHECK(Scalar(
              connection,
              "SELECT status FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) == "pending");
    CHECK(Scalar(
              connection,
              "SELECT requested_signal||':'||signal_result||':'||"
              "outcome_status FROM experiment_admin_worker_outcome "
              "WHERE request_id=" + std::to_string(requestId) +
                  " AND worker_identity='experiment:700010'") ==
          std::to_string(SIGTERM) + ":signaled:completed");
    CHECK(Scalar(
              connection,
              "SELECT (e.completed_at=e.cancellation_completed_at)::text||':'||"
              "(e.completed_at=e.updated_at)::text||':'||"
              "(e.completed_at=o.updated_at)::text FROM experiment e "
              "JOIN experiment_admin_worker_outcome o "
              "ON o.experiment_id=e.experiment_id "
              "WHERE e.experiment_id=700010 AND o.request_id=" +
                  std::to_string(requestId)) == "true:true:true");
    CHECK(Scalar(
              connection,
              "SELECT detail FROM experiment_admin_worker_outcome "
              "WHERE request_id=" + std::to_string(requestId) +
                  " AND worker_identity='experiment:700010'") ==
          "validated");

    {
        pqxx::work completeTransaction{connection};
        const long long terminalModelId = completeTransaction.exec(
            "INSERT INTO model(experiment_id,comment) VALUES "
            "(700011,'periodic training checkpoint') RETURNING model_id;")
                                              [0][0]
                                                  .as<long long>();
        completeTransaction.exec_params(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
            "VALUES ($1,'train_config_meta',0,10,20);",
            terminalModelId);
        completeTransaction.exec_params(
            "UPDATE experiment SET current_epoch=20,worker_pid=NULL,"
            "worker_process_group_id=NULL,"
            "current_operation='train',"
            "stopped_at_checkpoint_epoch=20,"
            "stopped_at_checkpoint_model_id=$1,last_model_id=$1,"
            "status='cancelled',phase='train',exit_code=0,"
            "completed_at=now(),cancellation_completed_at=now(),"
            "error_message='cancelled_at_requested_checkpoint',"
            "updated_at=now() WHERE experiment_id=700011;",
            terminalModelId);
        completeTransaction.exec_params(
            "UPDATE experiment_admin_worker_outcome SET "
            "cancellation_checkpoint_model_id=$1,inference_action='none',"
            "outcome_status='completed',"
            "detail='cancellation_checkpoint_reached',updated_at=now() "
            "WHERE request_id=$2 AND experiment_id=700011;",
            terminalModelId,
            requestId);
        completeTransaction.commit();
    }
    processes.signals.clear();
    CHECK(Replay(
              connectionString,
              ReplayCommand(
                  Action::CancelAll,
                  "crash-window-replay-accounted-complete",
                  CancellationMode::AfterNextCheckpoint),
              processes,
              replayOutput) == 0);
    CHECK(processes.signals.empty());
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity LIKE 'crash-window-%'") == "1");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL FROM "
              "experiment_global_control WHERE singleton") == "t");
    CHECK(Scalar(
              connection,
              "SELECT status FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) == "completed");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_worker_outcome "
              "WHERE request_id=" + std::to_string(requestId)) == "2");
    CHECK(Scalar(
              connection,
              "SELECT bool_and(status='cancelled') FROM experiment "
              "WHERE experiment_id IN (700010,700011)") == "t");
    CHECK(Scalar(
              connection,
              "SELECT requested_signal||':'||signal_result||':'||"
              "outcome_status||':'||detail FROM "
              "experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId) +
                  " AND worker_identity='experiment:700010'") ==
          std::to_string(SIGTERM) +
              ":signaled:completed:validated");
    CHECK(Scalar(
              connection,
              "SELECT current_epoch::text||':'||"
              "stopped_at_checkpoint_epoch::text||':'||"
              "(stopped_at_checkpoint_model_id=last_model_id)::text||':'||"
              "current_operation||':'||error_message FROM experiment "
              "WHERE experiment_id=700011") ==
          "20:20:true:train:"
          "cancelled_at_requested_checkpoint");
    ResetCrashFixtures(connection);
}

int RunSelectiveResume(const std::string& connectionString,
                       long long experimentId,
                       const std::string& invocationIdentity,
                       ProcessOperations& processes,
                       std::string& outputText,
                       bool dryRun = false,
                       bool confirmed = true)
{
    ExperimentResumeCommand command;
    command.experimentId = experimentId;
    command.dryRun = dryRun;
    command.confirmed = confirmed;
    command.invocationIdentity = invocationIdentity;
    std::ostringstream output;
    std::ostringstream error;
    const int result =
        RunExperimentResumeCommandWithProcessOperationsForTesting(
            connectionString, command, output, error, processes);
    outputText = output.str() + error.str();
    return result;
}

void TestSelectiveResumeFromGlobalPause(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    RecordingNativeProcesses processes;
    const ManagedWorker first = SpawnWorker(selfPath, processes, 700020);
    const ManagedWorker second = SpawnWorker(selfPath, processes, 700021);
    const ManagedWorker third = SpawnWorker(selfPath, processes, 700022);
    InsertRunningExperiment(connection, first);
    InsertRunningExperiment(connection, second);
    InsertRunningExperiment(connection, third);

    Command pause;
    pause.action = Action::PauseAll;
    pause.confirmed = true;
    pause.invocationIdentity = "selective-resume-pause-1";
    std::ostringstream pauseOutput;
    std::ostringstream pauseError;
    const int pauseResult = RunCommandWithProcessOperationsForTesting(
        connectionString, pause, pauseOutput, pauseError, processes);
    CHECK(pauseResult == 0);
    CHECK(pauseError.str().empty());
    CHECK(WaitUntil([&] {
        return processes.Observe(first.pid).stopped &&
               processes.Observe(second.pid).stopped &&
               processes.Observe(third.pid).stopped;
    }));
    const long long firstPauseRequest = std::stoll(Scalar(
        connection,
        "SELECT current_pause_request_id::text "
        "FROM experiment_global_control WHERE singleton"));
    CHECK(Scalar(
              connection,
              "SELECT string_agg(experiment_id::text||':'||"
              "worker_control_state||':'||worker_global_pause_request_id::text,"
              "',' ORDER BY experiment_id) FROM experiment "
              "WHERE experiment_id BETWEEN 700020 AND 700022") ==
          "700020:paused:" + std::to_string(firstPauseRequest) +
              ",700021:paused:" + std::to_string(firstPauseRequest) +
              ",700022:paused:" + std::to_string(firstPauseRequest));

    processes.signals.clear();
    std::string output;
    CHECK(RunSelectiveResume(
              connectionString, 700020, "selective-resume-dry-run",
              processes, output, true) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_DRY_RUN") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(first.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text||':'||"
              "worker_control_state||':'||"
              "worker_global_pause_request_id::text "
              "FROM experiment WHERE experiment_id=700020") ==
          "paused:false:paused:" + std::to_string(firstPauseRequest));
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE action='resume_experiment'") == "0");

    CHECK(RunSelectiveResume(
              connectionString, 700020, "selective-resume-confirmation",
              processes, output, false, false) == 0);
    CHECK(output.find("Use --yes to apply.") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(first.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text "
              "FROM experiment WHERE experiment_id=700020") ==
          "paused:false");

    CHECK(RunSelectiveResume(
              connectionString, 700020, "selective-resume-first",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("worker_state=stopped") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=queued_for_admission") != std::string::npos);
    CHECK(processes.signals.empty());

    // Selective resume releases the paused experiment back to scheduler
    // admission.  It deliberately does not SIGCONT the exact stopped worker.
    CHECK(processes.Observe(first.pid).stopped);
    CHECK(processes.Observe(second.pid).stopped);
    CHECK(processes.Observe(third.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT desired_state||':'||"
              "(active_request_id IS NULL)::text||':'||"
              "current_pause_request_id::text "
              "FROM experiment_global_control WHERE singleton") ==
          "paused:true:" + std::to_string(firstPauseRequest));
    {
        pqxx::work transaction{connection};
        CHECK(!NormalSchedulingAllowed(LoadControlSnapshot(transaction)));
        transaction.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text||':'||"
              "worker_control_state||':'||"
              "worker_global_pause_request_id::text||':'||"
              "(active_scheduler_worker_attempt_id IS NOT NULL)::text "
              "FROM experiment WHERE experiment_id=700020") ==
          "pending:true:paused:" + std::to_string(firstPauseRequest) +
              ":true");

    processes.signals.clear();
    CHECK(RunSelectiveResume(
              connectionString, 700020, "selective-resume-replay",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("worker_state=stopped") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=already_satisfied") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(first.pid).stopped);

    CHECK(RunSelectiveResume(
              connectionString, 700021, "selective-resume-second",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("worker_state=stopped") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=queued_for_admission") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(second.pid).stopped);
    CHECK(processes.Observe(third.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text "
              "FROM experiment WHERE experiment_id=700021") ==
          "pending:true");

    processes.signals.clear();
    Command resumeAll;
    resumeAll.action = Action::ResumeAll;
    resumeAll.confirmed = true;
    resumeAll.invocationIdentity = "selective-resume-all";
    std::ostringstream resumeAllOutput;
    std::ostringstream resumeAllError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString, resumeAll, resumeAllOutput, resumeAllError,
              processes) == 0);
    CHECK(resumeAllError.str().empty());
    CHECK(processes.signals.empty());
    CHECK(resumeAllOutput.str().find(
              "GLOBAL_EXPERIMENT_CONTROL_SUMMARY") != std::string::npos);
    CHECK(resumeAllOutput.str().find(
              "action=resume_all,status=completed") != std::string::npos);
    CHECK(resumeAllOutput.str().find(
              "signal_attempted=0") != std::string::npos);
    CHECK(processes.Observe(first.pid).stopped);
    CHECK(processes.Observe(second.pid).stopped);
    CHECK(processes.Observe(third.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT string_agg("
              "experiment_id::text||':'||status||':'||"
              "resume_requested::text,',' ORDER BY experiment_id) "
              "FROM experiment "
              "WHERE experiment_id BETWEEN 700020 AND 700022") ==
          "700020:pending:true,"
          "700021:pending:true,"
          "700022:pending:true");
    CHECK(Scalar(
              connection,
              "SELECT desired_state||':'||"
              "(current_pause_request_id IS NULL)::text "
              "FROM experiment_global_control WHERE singleton") ==
          "running:true");
    CHECK(Scalar(
              connection,
              "SELECT bool_and(worker_global_pause_request_id IS NULL) "
              "FROM experiment WHERE experiment_id BETWEEN 700020 AND 700022") ==
          "t");

    processes.signals.clear();
    pause.invocationIdentity = "selective-resume-pause-2";
    std::ostringstream secondPauseOutput;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString, pause, secondPauseOutput, pauseError,
              processes) == 0);
    CHECK(WaitUntil([&] {
        return processes.Observe(first.pid).stopped &&
               processes.Observe(second.pid).stopped &&
               processes.Observe(third.pid).stopped;
    }));
    const long long secondPauseRequest = std::stoll(Scalar(
        connection,
        "SELECT current_pause_request_id::text "
        "FROM experiment_global_control WHERE singleton"));
    CHECK(secondPauseRequest != firstPauseRequest);

    processes.signals.clear();
    CHECK(RunSelectiveResume(
              connectionString, 700020, "selective-resume-before-repause",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("worker_state=stopped") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=queued_for_admission") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(first.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text "
              "FROM experiment WHERE experiment_id=700020") ==
          "pending:true");
    pause.invocationIdentity = "selective-resume-pause-3";
    std::ostringstream thirdPauseOutput;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString, pause, thirdPauseOutput, pauseError,
              processes) == 0);
    CHECK(WaitUntil([&] { return processes.Observe(first.pid).stopped; }));
    const long long thirdPauseRequest = std::stoll(Scalar(
        connection,
        "SELECT current_pause_request_id::text "
        "FROM experiment_global_control WHERE singleton"));
    CHECK(thirdPauseRequest != secondPauseRequest);
    CHECK(Scalar(
              connection,
              "SELECT string_agg("
              "experiment_id::text||':'||worker_global_pause_request_id::text,"
              "',' ORDER BY experiment_id) "
              "FROM experiment "
              "WHERE experiment_id BETWEEN 700020 AND 700022") ==
          "700020:" + std::to_string(thirdPauseRequest) +
              ",700021:" + std::to_string(secondPauseRequest) +
              ",700022:" + std::to_string(secondPauseRequest));

    processes.signals.clear();
    CHECK(RunSelectiveResume(
              connectionString, 700020,
              "selective-resume-before-cancel-all",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("worker_state=stopped") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=queued_for_admission") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(first.pid).stopped);
    Command cancelAll;
    cancelAll.action = Action::CancelAll;
    cancelAll.cancellationMode = CancellationMode::Immediate;
    cancelAll.confirmed = true;
    cancelAll.invocationIdentity = "selective-resume-cancel-all";
    cancelAll.terminationGrace = std::chrono::milliseconds(300);
    std::ostringstream cancelOutput;
    std::ostringstream cancelError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString, cancelAll, cancelOutput, cancelError,
              processes) == 0);
    CHECK(cancelError.str().empty());

    AssertGroupExited(first);
    AssertGroupExited(second);
    AssertGroupExited(third);
    CHECK(Scalar(
              connection,
              "SELECT bool_and(status='cancelled') FROM experiment "
              "WHERE experiment_id BETWEEN 700020 AND 700022") == "t");
    CHECK(Scalar(
              connection,
              "SELECT bool_and(worker_global_pause_request_id IS NULL) "
              "FROM experiment "
              "WHERE experiment_id BETWEEN 700020 AND 700022") == "t");

    {
        pqxx::work transaction{connection};
        transaction.exec(
            "INSERT INTO experiment (experiment_id,status,phase) "
            "VALUES (700023,'paused','train'),(700024,'running','train');");
        transaction.commit();
    }
    processes.signals.clear();
    CHECK(RunSelectiveResume(
              connectionString, 700023, "selective-resume-lifecycle",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("worker_state=none") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=queued_for_admission") != std::string::npos);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text "
              "FROM experiment WHERE experiment_id=700023") ==
          "pending:true");
    CHECK(processes.signals.empty());

    CHECK(RunSelectiveResume(
              connectionString, 700024, "selective-resume-ordinary-running",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=running") != std::string::npos);
    CHECK(output.find("resume_requested=false") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=already_satisfied") != std::string::npos);
    CHECK(processes.signals.empty());

    ResetCrashFixtures(connection);

    const ManagedWorker missing =
        SpawnWorker(selfPath, processes, 700025);
    InsertRunningExperiment(connection, missing);
    pause.invocationIdentity = "selective-resume-missing-pause";
    std::ostringstream missingPauseOutput;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString, pause, missingPauseOutput, pauseError,
              processes) == 0);
    CHECK(WaitUntil([&] { return processes.Observe(missing.pid).stopped; }));
    CHECK(CancelWorker(
              missing, true, std::chrono::milliseconds(300), processes).success);
    AssertGroupExited(missing);
    processes.signals.clear();
    CHECK(RunSelectiveResume(
              connectionString, 700025, "selective-resume-missing",
              processes, output) == 0);
    CHECK(output.find("SCHEDULER_CONTROL_APPLIED") != std::string::npos);
    CHECK(output.find("new_status=pending") != std::string::npos);
    CHECK(output.find("resume_requested=true") != std::string::npos);
    CHECK(output.find("worker_state=stopped") != std::string::npos);
    CHECK(output.find("signal=none") != std::string::npos);
    CHECK(output.find("result=queued_for_admission") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(!processes.Observe(missing.pid).exists);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text||':'||"
              "(worker_pid IS NOT NULL)::text||':'||"
              "(active_scheduler_worker_attempt_id IS NOT NULL)::text "
              "FROM experiment WHERE experiment_id=700025") ==
          "pending:true:true:true");
    ResetCrashFixtures(connection);

    const ManagedWorker genericPredicateRace =
        SpawnWorker(selfPath, processes, 700029);
    InsertRunningExperiment(connection, genericPredicateRace);
    Command racedPause;
    racedPause.action = Action::PauseAll;
    racedPause.confirmed = true;
    racedPause.invocationIdentity = "generic-pause-state-race";
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment SET worker_command_line=$1 "
            "WHERE experiment_id=$2;",
            "generic-replacement-command",
            genericPredicateRace.experimentId);
        transaction.commit();
    }
    std::ostringstream genericRaceOutput;
    std::ostringstream genericRaceError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              racedPause,
              genericRaceOutput,
              genericRaceError,
              processes) == 1);
    CHECK(genericRaceOutput.str().find(
              "action=pause_all,status=partial") !=
          std::string::npos);
    CHECK(genericRaceOutput.str().find(
              "failed_count=1") !=
          std::string::npos);
    CHECK(!processes.Observe(genericPredicateRace.pid).stopped);
    CHECK(processes.signals.empty());
    CHECK(Scalar(
              connection,
              "SELECT (c.active_request_id IS NULL "
              "AND c.current_pause_request_id=r.request_id "
              "AND c.desired_state='paused' "
              "AND e.worker_control_state='running' "
              "AND e.worker_global_pause_request_id IS NULL "
              "AND r.application_lease_until IS NULL "
              "AND o.outcome_status='failed' "
              "AND o.signal_result='identity_validation_failed' "
              "AND o.detail LIKE '%exact%attempt%')::text "
              "FROM experiment_admin_request r "
              "JOIN experiment_global_control c ON c.singleton "
              "JOIN experiment_admin_worker_outcome o "
              "ON o.request_id=r.request_id "
              "JOIN experiment e ON e.experiment_id=o.experiment_id "
              "WHERE r.invocation_identity='generic-pause-state-race'") ==
          "true");
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment SET worker_command_line=$1 "
            "WHERE experiment_id=$2;",
            genericPredicateRace.commandLine,
            genericPredicateRace.experimentId);
        transaction.exec_params(
            "UPDATE experiment_admin_worker_outcome "
            "SET worker_command_line=$1 "
            "WHERE request_id=("
            " SELECT request_id FROM experiment_admin_request "
            " WHERE invocation_identity="
            "'generic-pause-state-race') "
            "AND experiment_id=$2;",
            genericPredicateRace.commandLine,
            genericPredicateRace.experimentId);
        transaction.exec(
            "UPDATE experiment_admin_request "
            "SET application_lease_until=now()-interval '1 second' "
            "WHERE invocation_identity='generic-pause-state-race';");
        transaction.commit();
    }
    processes.signals.clear();
    racedPause.invocationIdentity = "generic-pause-state-race-recovery";
    std::ostringstream genericRecoveryOutput;
    std::ostringstream genericRecoveryError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              racedPause,
              genericRecoveryOutput,
              genericRecoveryError,
              processes) == 0);
    CHECK((processes.signals ==
           std::vector<std::pair<int, int>>{
               {*genericPredicateRace.processGroupId, SIGSTOP}}));
    CHECK(genericRecoveryOutput.str().find(
              "action=pause_all,status=completed") !=
          std::string::npos);
    CHECK(genericRecoveryOutput.str().find(
              "stopped_workers=1") !=
          std::string::npos);
    CHECK(genericRecoveryOutput.str().find(
              "failed_count=0") !=
          std::string::npos);
    CHECK(Scalar(
              connection,
              "SELECT e.worker_control_state||':'||"
              "(e.worker_global_pause_request_id="
              "c.current_pause_request_id)::text||':'||"
              "(c.active_request_id IS NULL)::text "
              "FROM experiment e CROSS JOIN experiment_global_control c "
              "WHERE e.experiment_id=700029 AND c.singleton") ==
          "paused:true:true");
    Command genericCleanup;
    genericCleanup.action = Action::ResumeAll;
    genericCleanup.confirmed = true;
    genericCleanup.invocationIdentity = "generic-pause-state-race-cleanup";
    std::ostringstream genericCleanupOutput;
    std::ostringstream genericCleanupError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              genericCleanup,
              genericCleanupOutput,
              genericCleanupError,
              processes) == 0);
    CleanupWorker(genericPredicateRace, processes);
    ResetCrashFixtures(connection);

    const ManagedWorker changedDuringSignal =
        SpawnWorker(selfPath, processes, 700027);
    InsertRunningExperiment(connection, changedDuringSignal);
    pause.invocationIdentity = "selective-resume-state-race-pause";
    std::ostringstream stateRacePauseOutput;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString, pause, stateRacePauseOutput, pauseError,
              processes) == 0);
    CHECK(WaitUntil(
        [&] { return processes.Observe(changedDuringSignal.pid).stopped; }));
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment SET worker_command_line=$1 "
            "WHERE experiment_id=$2;",
            "replacement-command",
            changedDuringSignal.experimentId);
        transaction.commit();
    }
    processes.signals.clear();
    CHECK(RunSelectiveResume(
              connectionString, changedDuringSignal.experimentId,
              "selective-resume-state-race",
              processes, output) == 1);
    CHECK(output.find("reason=stale_control_evidence") !=
          std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(changedDuringSignal.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT worker_control_state FROM experiment "
              "WHERE experiment_id=700027") == "paused");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity="
              "'selective-resume-state-race'") == "0");
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment SET worker_command_line=$1 "
            "WHERE experiment_id=$2;",
            changedDuringSignal.commandLine,
            changedDuringSignal.experimentId);
        transaction.commit();
    }
    processes.signals.clear();
    std::string stateRaceReplayOutput;
    CHECK(RunSelectiveResume(
              connectionString,
              changedDuringSignal.experimentId,
              "selective-resume-state-race-recovery",
              processes,
              stateRaceReplayOutput) == 0);
    CHECK(stateRaceReplayOutput.find(
              "new_status=pending") != std::string::npos);
    CHECK(stateRaceReplayOutput.find(
              "resume_requested=true") != std::string::npos);
    CHECK(stateRaceReplayOutput.find(
              "worker_state=stopped") != std::string::npos);
    CHECK(stateRaceReplayOutput.find(
              "signal=none") != std::string::npos);
    CHECK(stateRaceReplayOutput.find(
              "result=queued_for_admission") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(changedDuringSignal.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text||':'||"
              "worker_control_state||':'||"
              "(active_scheduler_worker_attempt_id IS NOT NULL)::text "
              "FROM experiment WHERE experiment_id=700027") ==
          "pending:true:paused:true");
    CleanupWorker(changedDuringSignal, processes);
    ResetCrashFixtures(connection);

    const ManagedWorker identityMismatch =
        SpawnWorker(selfPath, processes, 700026);
    InsertRunningExperiment(connection, identityMismatch);
    pause.invocationIdentity = "selective-resume-identity-pause";
    std::ostringstream identityPauseOutput;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString, pause, identityPauseOutput, pauseError,
              processes) == 0);
    CHECK(WaitUntil(
        [&] { return processes.Observe(identityMismatch.pid).stopped; }));
    // Corruption-defense only: runtime privileges normally preserve the exact
    // experiment/attempt identity agreement established by pause.
    // Production-reachable replacement-process identity failures are covered
    // separately by the native process fixtures in RunRealProcessTests.
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment SET worker_executable='wrong-executable' "
            "WHERE experiment_id=$1;",
            identityMismatch.experimentId);
        transaction.commit();
    }
    processes.signals.clear();
    CHECK(RunSelectiveResume(
              connectionString, 700026, "selective-resume-identity-failure",
              processes, output) == 1);
    CHECK(output.find("reason=stale_control_evidence") !=
          std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(identityMismatch.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity="
              "'selective-resume-identity-failure'") == "0");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL "
              "FROM experiment_global_control WHERE singleton") == "t");
    resumeAll.invocationIdentity =
        "selective-resume-identity-resume-all";
    std::ostringstream unresolvedResumeOutput;
    std::ostringstream unresolvedResumeError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              resumeAll,
              unresolvedResumeOutput,
              unresolvedResumeError,
              processes) == 0);
    CHECK(unresolvedResumeError.str().empty());
    CHECK(unresolvedResumeOutput.str().find(
              "action=resume_all,status=completed") !=
          std::string::npos);
    CHECK(unresolvedResumeOutput.str().find(
              "signal_attempted=0") != std::string::npos);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text||':'||"
              "worker_control_state||':'||"
              "(worker_global_pause_request_id IS NULL)::text||':'||"
              "(active_scheduler_worker_attempt_id IS NOT NULL)::text "
              "FROM experiment WHERE experiment_id=700026") ==
          "pending:true:paused:true:true");
    CHECK(processes.Observe(identityMismatch.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT desired_state||':'||"
              "(active_request_id IS NULL)::text||':'||"
              "(current_pause_request_id IS NULL)::text "
              "FROM experiment_global_control WHERE singleton") ==
          "running:true:true");
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment SET worker_executable=$1 "
            "WHERE experiment_id=$2;",
            identityMismatch.executable,
            identityMismatch.experimentId);
        transaction.commit();
    }
    CleanupWorker(identityMismatch, processes);
    ResetCrashFixtures(connection);
}

void TestConcurrentSelectiveResumeIsIdempotent(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    RecordingNativeProcesses processes;
    const ManagedWorker worker =
        SpawnWorker(selfPath, processes, 700030);
    InsertRunningExperiment(connection, worker);

    Command pause;
    pause.action = Action::PauseAll;
    pause.confirmed = true;
    pause.invocationIdentity = "selective-resume-owner-fence-pause";
    std::ostringstream pauseOutput;
    std::ostringstream pauseError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              pause,
              pauseOutput,
              pauseError,
              processes) == 0);
    CHECK(WaitUntil([&] { return processes.Observe(worker.pid).stopped; }));
    const long long oldPauseRequest = std::stoll(Scalar(
        connection,
        "SELECT current_pause_request_id::text "
        "FROM experiment_global_control WHERE singleton"));
    processes.signals.clear();
    int firstResult = -1;
    int secondResult = -1;
    std::string firstOutput;
    std::string secondOutput;
    std::thread firstResume([&] {
        firstResult = RunSelectiveResume(
            connectionString,
            worker.experimentId,
            "concurrent-selective-resume-first",
            processes,
            firstOutput);
    });
    std::thread secondResume([&] {
        secondResult = RunSelectiveResume(
            connectionString,
            worker.experimentId,
            "concurrent-selective-resume-second",
            processes,
            secondOutput);
    });
    firstResume.join();
    secondResume.join();
    CHECK(firstResult == 0);
    CHECK(secondResult == 0);
    CHECK((firstOutput.find("result=queued_for_admission") !=
               std::string::npos) !=
          (secondOutput.find("result=queued_for_admission") !=
               std::string::npos));
    CHECK((firstOutput.find("result=already_satisfied") !=
               std::string::npos) !=
          (secondOutput.find("result=already_satisfied") !=
               std::string::npos));
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(worker.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text||':'||"
              "worker_control_state||':'||"
              "(active_scheduler_worker_attempt_id IS NOT NULL)::text "
              "FROM experiment WHERE experiment_id=700030") ==
          "pending:true:paused:true");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity IN "
              "('concurrent-selective-resume-first',"
              "'concurrent-selective-resume-second')") == "0");

    Command newerPause;
    newerPause.action = Action::PauseAll;
    newerPause.confirmed = true;
    newerPause.invocationIdentity =
        "selective-resume-owner-fence-new-pause";
    std::ostringstream output;
    std::ostringstream error;
    const int newerPauseResult = RunCommandWithProcessOperationsForTesting(
        connectionString,
        newerPause,
        output,
        error,
        processes);
    CHECK(newerPauseResult == 0);
    CHECK(error.str().empty());
    CHECK(output.str().find("status=completed") != std::string::npos);
    const long long newerPauseRequest = std::stoll(Scalar(
        connection,
        "SELECT current_pause_request_id::text "
        "FROM experiment_global_control WHERE singleton"));
    CHECK(newerPauseRequest != oldPauseRequest);
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text||':'||"
              "worker_control_state||':'||"
              "worker_global_pause_request_id::text "
              "FROM experiment WHERE experiment_id=700030") ==
          "paused:false:paused:" + std::to_string(newerPauseRequest));
    CHECK(processes.Observe(worker.pid).stopped);

    Command resumeAll;
    resumeAll.action = Action::ResumeAll;
    resumeAll.confirmed = true;
    resumeAll.invocationIdentity =
        "selective-resume-owner-fence-cleanup";
    std::ostringstream cleanupOutput;
    std::ostringstream cleanupError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              resumeAll,
              cleanupOutput,
              cleanupError,
              processes) == 0);
    CleanupWorker(worker, processes);
    ResetCrashFixtures(connection);
}

void TestSchedulerCancellationOwnerClaims(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    RecordingNativeProcesses processes;
    const ManagedWorker worker =
        SpawnWorker(selfPath, processes, 700036);
    const long long requestId =
        SeedRequest(connection, Action::CancelAll, worker, "planned");
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment_admin_request SET "
            "application_owner='foreign-live-owner',"
            "application_lease_until=now()+interval '5 minutes' "
            "WHERE request_id=$1;",
            requestId);
        transaction.commit();
    }
    {
        pqxx::work transaction{connection};
        AcquireCoordinationLock(transaction);
        CHECK(!ReconcileActiveCancellation(
            transaction, "scheduler-owner", true));
        transaction.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT application_owner||':'||status "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "foreign-live-owner:applying");
    {
        pqxx::work transaction{connection};
        transaction.exec_params(
            "UPDATE experiment_admin_request SET "
            "application_lease_until=now()-interval '1 second' "
            "WHERE request_id=$1;",
            requestId);
        transaction.commit();
    }
    {
        pqxx::work transaction{connection};
        AcquireCoordinationLock(transaction);
        CHECK(ReconcileActiveCancellation(
            transaction, "scheduler-owner", true));
        transaction.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT application_owner||':'||status "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "scheduler-owner:pending");
    {
        pqxx::work transaction{connection};
        AcquireCoordinationLock(transaction);
        CHECK(!ReconcileActiveCancellation(
            transaction, "foreign-live-owner", false));
        transaction.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT application_owner FROM experiment_admin_request "
              "WHERE request_id=" + std::to_string(requestId)) ==
          "scheduler-owner");
    CleanupWorker(worker, processes);
    ResetCrashFixtures(connection);
    (void)connectionString;
}

void TestSelectiveReplayAfterLifecycleTransition(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    (void)selfPath;
    RecordingNativeProcesses processes;
    const std::vector<std::string> terminalStates{
        "completed", "failed", "cancelled"};
    long long experimentId = 700031;
    for (const std::string& terminalState : terminalStates)
    {
        {
            pqxx::work transaction{connection};
            transaction.exec_params(
                "INSERT INTO experiment(experiment_id,status,phase) "
                "VALUES($1,$2,'train');",
                experimentId,
                terminalState);
            transaction.commit();
        }

        std::string replayOutput;
        CHECK(RunSelectiveResume(
                  connectionString,
                  experimentId,
                  "selective-resume-terminal-replay-" + terminalState,
                  processes,
                  replayOutput) == 1);
        CHECK(replayOutput.find(
                  "reason=resume_requires_paused_status") !=
              std::string::npos);
        CHECK(processes.signals.empty());
        CHECK(Scalar(
                  connection,
                  "SELECT status||':'||resume_requested::text "
                  "FROM experiment WHERE experiment_id=" +
                      std::to_string(experimentId)) ==
              terminalState + ":false");
        CHECK(Scalar(
                  connection,
                  "SELECT count(*)::text FROM experiment_admin_request "
                  "WHERE invocation_identity='selective-resume-terminal-replay-" +
                      terminalState + "'") == "0");
        ++experimentId;
    }
    ResetCrashFixtures(connection);
}

void TestPrimarySelectiveResumeAndCheckpointChildDeparture(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    RecordingNativeProcesses processes;
    const ManagedWorker primary =
        SpawnWorker(selfPath, processes, 700040);
    const ManagedWorker child =
        SpawnWorker(
            selfPath,
            processes,
            700040,
            false,
            false,
            SpawnFailureMode::None,
            8700040);
    InsertRunningExperiment(connection, primary);
    InsertRunningCheckpointChild(connection, child);

    Command pause;
    pause.action = Action::PauseAll;
    pause.confirmed = true;
    pause.invocationIdentity = "selective-resume-primary-child-pause";
    std::ostringstream pauseOutput;
    std::ostringstream pauseError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              pause,
              pauseOutput,
              pauseError,
              processes) == 0);
    CHECK(WaitUntil([&] {
        return processes.Observe(primary.pid).stopped;
    }));
    CHECK(!processes.Observe(child.pid).stopped);
    const long long pauseRequest = std::stoll(Scalar(
        connection,
        "SELECT current_pause_request_id::text "
        "FROM experiment_global_control WHERE singleton"));
    CHECK(Scalar(
              connection,
              "SELECT r.target_count::text||':'||"
              "count(o.*)::text FROM experiment_admin_request r "
              "JOIN experiment_admin_worker_outcome o USING(request_id) "
              "WHERE r.request_id=" + std::to_string(pauseRequest) +
                  " GROUP BY r.target_count") == "1:1");

    processes.signals.clear();
    std::string selectiveOutput;
    CHECK(RunSelectiveResume(
              connectionString,
              primary.experimentId,
              "selective-resume-primary-only",
              processes,
              selectiveOutput) == 0);
    CHECK(selectiveOutput.find("result=queued_for_admission") !=
          std::string::npos);
    CHECK(selectiveOutput.find("signal=none") != std::string::npos);
    CHECK(processes.signals.empty());
    CHECK(processes.Observe(primary.pid).stopped);
    CHECK(!processes.Observe(child.pid).stopped);
    CHECK(Scalar(
              connection,
              "SELECT e.status||':'||e.resume_requested::text||':'||"
              "e.worker_control_state||':'||"
              "e.worker_global_pause_request_id::text||':'||"
              "ce.worker_control_state||':'||"
              "(ce.worker_global_pause_request_id IS NULL)::text "
              "FROM experiment e "
              "JOIN experiment_checkpoint_eval ce "
              "ON ce.parent_experiment_id=e.experiment_id "
              "WHERE e.experiment_id=700040") ==
          "pending:true:paused:" + std::to_string(pauseRequest) +
              ":running:true");

    CHECK(CancelWorker(
              child,
              true,
              std::chrono::milliseconds(300),
              processes).success);
    AssertGroupExited(child);
    {
        pqxx::work transaction{connection};
        const long long attemptId = transaction.exec(
            "SELECT active_scheduler_worker_attempt_id "
            "FROM experiment_checkpoint_eval "
            "WHERE checkpoint_eval_id=8700040;")[0][0].as<long long>();
        CHECK(transaction.exec_params(
            "UPDATE experiment_scheduler_worker_attempt "
            "SET lifecycle_state='failed',completed_at=now(),"
            "last_observed_at=now() "
            "WHERE worker_attempt_id=$1 "
            "AND lifecycle_state='observed';",
            attemptId).affected_rows() == 1);
        CHECK(transaction.exec_params(
            "UPDATE experiment_checkpoint_eval SET status='failed',"
            "phase='done',worker_pid=NULL,worker_process_group_id=NULL,"
            "active_scheduler_worker_attempt_id=NULL,"
            "completed_at=now(),updated_at=now() "
            "WHERE checkpoint_eval_id=8700040 "
            "AND active_scheduler_worker_attempt_id=$1;",
            attemptId).affected_rows() == 1);
        transaction.commit();
    }
    processes.signals.clear();
    Command resumeAll;
    resumeAll.action = Action::ResumeAll;
    resumeAll.confirmed = true;
    resumeAll.invocationIdentity =
        "selective-resume-primary-child-resume-all";
    std::ostringstream resumeOutput;
    std::ostringstream resumeError;
    CHECK(RunCommandWithProcessOperationsForTesting(
              connectionString,
              resumeAll,
              resumeOutput,
              resumeError,
              processes) == 0);
    CHECK(processes.signals.empty());
    CHECK(resumeOutput.str().find(
              "target_count=0,resume_requested_count=0,"
              "signal_attempted=0") !=
          std::string::npos);
    CHECK(Scalar(
              connection,
              "SELECT e.worker_global_pause_request_id IS NULL AND "
              "ce.worker_global_pause_request_id IS NULL "
              "FROM experiment e "
              "JOIN experiment_checkpoint_eval ce "
              "ON ce.parent_experiment_id=e.experiment_id "
              "WHERE e.experiment_id=700040") == "t");
    CHECK(Scalar(
              connection,
              "SELECT status||':'||resume_requested::text "
              "FROM experiment WHERE experiment_id=700040") ==
          "pending:true");
    CHECK(Scalar(
              connection,
              "SELECT desired_state||':'||"
              "(current_pause_request_id IS NULL)::text||':'||"
              "(active_request_id IS NULL)::text "
              "FROM experiment_global_control WHERE singleton") ==
          "running:true:true");

    CleanupWorker(primary, processes);
    ResetCrashFixtures(connection);
}

void TestCancellationWithSelectivelyReleasedPrimaryAndStoppedChild(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    for (const CancellationMode mode :
         {CancellationMode::Immediate,
          CancellationMode::AfterNextCheckpoint})
    {
        const bool requestedAfterCheckpoint =
            mode == CancellationMode::AfterNextCheckpoint;
        const long long experimentId =
            requestedAfterCheckpoint ? 700051 : 700050;
        const long long checkpointEvalId =
            requestedAfterCheckpoint ? 8700051 : 8700050;
        RecordingNativeProcesses processes;
        const ManagedWorker primary =
            SpawnWorker(selfPath, processes, experimentId);
        const ManagedWorker child =
            SpawnWorker(
                selfPath,
                processes,
                experimentId,
                false,
                false,
                SpawnFailureMode::None,
                checkpointEvalId);
        InsertRunningExperiment(connection, primary);
        InsertRunningCheckpointChild(connection, child);

        Command pause;
        pause.action = Action::PauseAll;
        pause.confirmed = true;
        pause.invocationIdentity =
            "selective-resume-cancel-mix-pause-" +
            std::string{ToString(mode)};
        std::ostringstream pauseOutput;
        std::ostringstream pauseError;
        CHECK(RunCommandWithProcessOperationsForTesting(
                  connectionString,
                  pause,
                  pauseOutput,
                  pauseError,
                  processes) == 0);
        CHECK(WaitUntil([&] {
            return processes.Observe(primary.pid).stopped;
        }));
        CHECK(!processes.Observe(child.pid).stopped);

        std::string selectiveOutput;
        CHECK(RunSelectiveResume(
                  connectionString,
                  experimentId,
                  "selective-resume-cancel-mix-release-" +
                      std::string{ToString(mode)},
                  processes,
                  selectiveOutput) == 0);
        CHECK(selectiveOutput.find("result=queued_for_admission") !=
              std::string::npos);
        CHECK(processes.Observe(primary.pid).stopped);
        CHECK(!processes.Observe(child.pid).stopped);

        Command cancel;
        cancel.action = Action::CancelAll;
        cancel.cancellationMode = mode;
        cancel.confirmed = true;
        cancel.invocationIdentity =
            "selective-resume-cancel-mix-" +
            std::string{ToString(mode)};
        cancel.terminationGrace = std::chrono::milliseconds(300);
        std::ostringstream cancelOutput;
        std::ostringstream cancelError;
        CHECK(RunCommandWithProcessOperationsForTesting(
                  connectionString,
                  cancel,
                  cancelOutput,
                  cancelError,
                  processes) == 0);
        CHECK(cancelError.str().empty());
        AssertGroupExited(primary);
        AssertGroupExited(child);
        CHECK(cancelOutput.str().find("target_count=2") !=
              std::string::npos);
        CHECK(Scalar(
                  connection,
                  "SELECT e.status||':'||"
                  "(e.worker_global_pause_request_id IS NULL)::text||':'||"
                  "ce.status||':'||"
                  "(ce.worker_global_pause_request_id IS NULL)::text "
                  "FROM experiment e "
                  "JOIN experiment_checkpoint_eval ce "
                  "ON ce.checkpoint_eval_id=" +
                      std::to_string(checkpointEvalId) +
                      " WHERE e.experiment_id=" +
                      std::to_string(experimentId)) ==
              "cancelled:true:failed:true");
        CHECK(Scalar(
                  connection,
                  "SELECT bool_and(outcome_status='completed') "
                  "FROM experiment_admin_worker_outcome o "
                  "JOIN experiment_admin_request r USING(request_id) "
                  "WHERE r.invocation_identity='selective-resume-cancel-mix-" +
                      std::string{ToString(mode)} + "'") == "t");
        CHECK(Scalar(
                  connection,
                  "SELECT active_request_id IS NULL AND "
                  "current_pause_request_id IS NULL "
                  "FROM experiment_global_control WHERE singleton") == "t");
        ResetCrashFixtures(connection);
    }
}
void TestCancellationInspectionFailuresRemainRecoverable(
    const std::string& selfPath,
    const std::string& connectionString,
    pqxx::connection& connection)
{
    struct FailureCase
    {
        long long experimentId;
        ObservationFailureMode mode;
        const char* identityResult;
        const char* signalResult;
        const char* suffix;
    };
    for (const FailureCase& failure :
         {FailureCase{
              700060,
              ObservationFailureMode::PermissionDenied,
              "permission_denied",
              "permission_failure",
              "permission"},
          FailureCase{
              700061,
              ObservationFailureMode::InspectionFailed,
              "inspection_failed",
              "identity_validation_failed",
              "inspection"}})
    {
        RecordingNativeProcesses processes;
        const ManagedWorker worker =
            SpawnWorker(selfPath, processes, failure.experimentId);
        InsertRunningExperiment(connection, worker);

        Command pause;
        pause.action = Action::PauseAll;
        pause.confirmed = true;
        pause.invocationIdentity =
            "cancellation-observation-pause-" +
            std::string{failure.suffix};
        std::ostringstream pauseOutput;
        std::ostringstream pauseError;
        CHECK(RunCommandWithProcessOperationsForTesting(
                  connectionString,
                  pause,
                  pauseOutput,
                  pauseError,
                  processes) == 0);
        CHECK(WaitUntil([&] { return processes.Observe(worker.pid).stopped; }));
        const long long pauseRequestId = std::stoll(Scalar(
            connection,
            "SELECT current_pause_request_id::text "
            "FROM experiment_global_control WHERE singleton"));

        processes.signals.clear();
        ObservationFailureProcesses unavailable(
            processes, worker.pid, failure.mode);
        Command cancel;
        cancel.action = Action::CancelAll;
        cancel.cancellationMode = CancellationMode::Immediate;
        cancel.confirmed = true;
        cancel.invocationIdentity =
            "cancellation-observation-" + std::string{failure.suffix};
        cancel.terminationGrace = std::chrono::milliseconds(300);
        std::ostringstream cancelOutput;
        std::ostringstream cancelError;
        CHECK(RunCommandWithProcessOperationsForTesting(
                  connectionString,
                  cancel,
                  cancelOutput,
                  cancelError,
                  unavailable) == 0);
        CHECK(processes.signals.empty());
        CHECK(processes.Observe(worker.pid).stopped);
        CHECK(cancelOutput.str().find(
                  "action=cancel_all,status=pending,result=pending,"
                  "replay=0,signal_attempted=0") != std::string::npos);
        CHECK(cancelOutput.str().find(
                  "worker_identity=experiment:" +
                  std::to_string(failure.experimentId) +
                  ",identity_result=" + failure.identityResult +
                  ",outcome_status=planned,signal_result=" +
                  failure.signalResult) != std::string::npos);
        const long long cancelRequestId = std::stoll(Scalar(
            connection,
            "SELECT active_request_id::text "
            "FROM experiment_global_control WHERE singleton"));
        CHECK(Scalar(
                  connection,
                  "SELECT e.status||':'||e.worker_control_state||':'||"
                  "e.worker_global_pause_request_id::text||':'||"
                  "e.cancellation_request_id::text||':'||"
                  "c.current_pause_request_id::text||':'||"
                  "c.active_request_id::text "
                  "FROM experiment e CROSS JOIN experiment_global_control c "
                  "WHERE e.experiment_id=" +
                      std::to_string(failure.experimentId) +
                      " AND c.singleton") ==
              "paused:paused:" + std::to_string(pauseRequestId) + ":" +
                  std::to_string(cancelRequestId) + ":" +
                  std::to_string(pauseRequestId) + ":" +
                  std::to_string(cancelRequestId));

        processes.signals.clear();
        Command replay = cancel;
        replay.invocationIdentity =
            "cancellation-observation-recovery-" +
            std::string{failure.suffix};
        std::ostringstream replayOutput;
        std::ostringstream replayError;
        CHECK(RunCommandWithProcessOperationsForTesting(
                  connectionString,
                  replay,
                  replayOutput,
                  replayError,
                  processes) == 0);
        AssertGroupExited(worker);
        CHECK(replayOutput.str().find(
                  "action=cancel_all,status=completed,result=completed,"
                  "replay=1,signal_attempted=1") != std::string::npos);
        CHECK(Scalar(
                  connection,
                  "SELECT active_request_id IS NULL AND "
                  "current_pause_request_id IS NULL "
                  "FROM experiment_global_control WHERE singleton") == "t");
        ResetCrashFixtures(connection);
    }
}

void TestTerminalCompletedCheckpointReconciliation(
    const std::string& connectionString,
    pqxx::connection& connection)
{
    pqxx::work fixture{connection};
    const long long requestId = fixture.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,application_owner,application_lease_until,status,"
        "previous_global_state,resulting_global_state,"
        "scheduler_running_observed,target_count,successful_count,"
        "already_satisfied_count,missing_count,rejected_count,failed_count,"
        "result_summary) VALUES ("
        "'cancel_all','after_next_checkpoint',false,"
        "'crash-window-terminal-completed','crash-fixture-requester',"
        "'crash-window-terminal-completed',NULL,'pending',"
        "'running','running',false,6,0,0,0,6,0,"
        "'{\"target_count\":6,\"successful_count\":0,"
        "\"already_satisfied_count\":0,\"missing_count\":0,"
        "\"rejected_count\":6,\"failed_count\":0,"
        "\"pending_count\":6}'::jsonb) RETURNING request_id;")
                                    [0][0]
                                        .as<long long>();
    fixture.exec_params(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,current_epoch,checkpoint_interval,"
        "target_epochs,infer_start,infer_end,completed_at,updated_at,"
        "current_operation,cancellation_request_id,"
        "cancel_after_checkpoint_epoch,stop_after_checkpoint_epoch,"
        "stopped_at_checkpoint_epoch) "
        "SELECT experiment_id,'completed','done',80,20,100,"
        "'2020-02-02'::date,'2020-03-01'::date,"
        "now()-interval '1 hour',now()-interval '1 hour',"
        "'analyze', $1,80,80,80 "
        "FROM generate_series(700020,700025) AS experiment_id;",
        requestId);
    fixture.exec(
        "INSERT INTO experiment (experiment_id,status,phase,current_epoch,"
        "current_operation) VALUES "
        "(700026,'pending','train',0,'train');");
    fixture.exec(
        "INSERT INTO model(experiment_id,comment) "
        "SELECT experiment_id,'periodic training checkpoint' "
        "FROM generate_series(700020,700025) AS experiment_id;");
    fixture.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT model_id,'train_config_meta',0,10,80 "
        "FROM model WHERE experiment_id BETWEEN 700020 AND 700025;");
    fixture.exec(
        "UPDATE experiment e SET stopped_at_checkpoint_model_id=m.model_id,"
        "last_model_id=m.model_id "
        "FROM model m WHERE m.experiment_id=e.experiment_id "
        "AND e.experiment_id BETWEEN 700020 AND 700025;");
    fixture.exec_params(
        "INSERT INTO experiment_checkpoint_eval ("
        "experiment_id,parent_experiment_id,checkpoint_epoch,"
        "checkpoint_model_id,status,phase,completed_at,"
        "cancellation_request_id) "
        "SELECT e.experiment_id,e.experiment_id,80,"
        "e.stopped_at_checkpoint_model_id,'completed','done',"
        "now()-interval '50 minutes',$1 "
        "FROM experiment e "
        "WHERE e.experiment_id BETWEEN 700020 AND 700025;",
        requestId);
    fixture.exec(
        "INSERT INTO experiment_analysis_result(experiment_id) "
        "SELECT experiment_id FROM generate_series(700020,700025) "
        "AS experiment_id;");
    fixture.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,worker_kind,phase,"
        "lifecycle_status,worker_pid,worker_process_group_id,"
        "worker_process_start_identity,identity_result,signal_result,"
        "cancellation_checkpoint_epoch,inference_action,outcome_status,"
        "detail) "
        "SELECT $1,'experiment:'||experiment_id::text,experiment_id,"
        "'experiment','train','running',910000+experiment_id::int,"
        "910000+experiment_id::int,'1700000000:'||experiment_id::text,"
        "'identity_validation_failed','identity_validation_failed',"
        "80,'none','pending_checkpoint',"
        "'missing_worker_requeued_from_durable_checkpoint' "
        "FROM generate_series(700020,700025) AS experiment_id;",
        requestId);
    fixture.exec_params(
        "UPDATE experiment_global_control SET desired_state='running',"
        "active_request_id=$1,revision=revision+1,updated_at=now() "
        "WHERE singleton;",
        requestId);
    fixture.commit();

    const std::string completedExperimentEvidence = Scalar(
        connection,
        "SELECT string_agg("
        "e.experiment_id::text||':'||e.status||':'||e.phase||':'||"
        "e.current_epoch::text||':'||e.stop_after_checkpoint_epoch::text||':'||"
        "e.stopped_at_checkpoint_epoch::text||':'||"
        "e.stopped_at_checkpoint_model_id::text||':'||e.last_model_id::text||':'||"
        "e.current_operation||':'||(e.completed_at IS NOT NULL)::text||':'||"
        "(e.cancellation_completed_at IS NULL)::text,',' "
        "ORDER BY e.experiment_id) "
        "FROM experiment e WHERE e.experiment_id BETWEEN 700020 AND 700025;");
    const std::string artifactEvidence = Scalar(
        connection,
        "SELECT "
        "(SELECT count(*) FROM model "
        " WHERE experiment_id BETWEEN 700020 AND 700025)::text||':'||"
        "(SELECT count(*) FROM matrix tm JOIN model m USING(model_id) "
        " WHERE m.experiment_id BETWEEN 700020 AND 700025)::text||':'||"
        "(SELECT count(*) FROM experiment_checkpoint_eval "
        " WHERE experiment_id BETWEEN 700020 AND 700025)::text||':'||"
        "(SELECT count(*) FROM experiment_analysis_result "
        " WHERE experiment_id BETWEEN 700020 AND 700025)::text;");

    // Reconcile the durable cancellation directly. PauseAll/ResumeAll now
    // use the scheduler priority-queue control path and are no longer
    // administrative-request reconciliation entry points.
    {
        pqxx::work reconciliation{connection};
        AcquireCoordinationLock(reconciliation);
        CHECK(ReconcileActiveCancellationForTest(
            reconciliation, requestId));
        reconciliation.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||target_count::text||':'||"
              "successful_count::text||':'||already_satisfied_count::text||':'||"
              "missing_count::text||':'||rejected_count::text||':'||"
              "failed_count::text||':'||"
              "(result_summary->>'pending_count')||':'||"
              "(result_summary->>'target_count')||':'||"
              "(result_summary->>'rejected_count')||':'||"
              "(completed_at IS NOT NULL)::text "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "partial:6:0:0:0:6:0:0:6:6:true");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text||':'||"
              "count(*) FILTER (WHERE outcome_status='completed')::text||':'||"
              "count(*) FILTER (WHERE "
              "cancellation_checkpoint_model_id IS NOT NULL)::text||':'||"
              "bool_and(cancellation_checkpoint_model_id="
              "(SELECT stopped_at_checkpoint_model_id FROM experiment e "
              " WHERE e.experiment_id=o.experiment_id))::text||':'||"
              "bool_and(detail="
              "'cancellation_checkpoint_reconciled_from_terminal_experiment')"
              "::text FROM experiment_admin_worker_outcome o "
              "WHERE request_id=" + std::to_string(requestId)) ==
          "6:6:6:true:true");
    CHECK(Scalar(
              connection,
              "SELECT desired_state||':'||(active_request_id IS NULL)::text "
              "FROM experiment_global_control WHERE singleton") ==
          "running:true");
    CHECK(Scalar(
              connection,
              "SELECT status||':'||phase||':'||"
              "(cancellation_request_id IS NULL)::text "
              "FROM experiment WHERE experiment_id=700026") ==
          "pending:train:true");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment "
              "WHERE experiment_id BETWEEN 700020 AND 700026") == "7");
    CHECK(Scalar(
              connection,
              "SELECT string_agg("
              "e.experiment_id::text||':'||e.status||':'||e.phase||':'||"
              "e.current_epoch::text||':'||"
              "e.stop_after_checkpoint_epoch::text||':'||"
              "e.stopped_at_checkpoint_epoch::text||':'||"
              "e.stopped_at_checkpoint_model_id::text||':'||"
              "e.last_model_id::text||':'||e.current_operation||':'||"
              "(e.completed_at IS NOT NULL)::text||':'||"
              "(e.cancellation_completed_at IS NULL)::text,',' "
              "ORDER BY e.experiment_id) FROM experiment e "
              "WHERE e.experiment_id BETWEEN 700020 AND 700025") ==
          completedExperimentEvidence);
    CHECK(Scalar(
              connection,
              "SELECT "
              "(SELECT count(*) FROM model "
              " WHERE experiment_id BETWEEN 700020 AND 700025)::text||':'||"
              "(SELECT count(*) FROM matrix tm JOIN model m USING(model_id) "
              " WHERE m.experiment_id BETWEEN 700020 AND 700025)::text||':'||"
              "(SELECT count(*) FROM experiment_checkpoint_eval "
              " WHERE experiment_id BETWEEN 700020 AND 700025)::text||':'||"
              "(SELECT count(*) FROM experiment_analysis_result "
              " WHERE experiment_id BETWEEN 700020 AND 700025)::text") ==
          artifactEvidence);

    const std::string terminalTimestamps = Scalar(
        connection,
        "SELECT r.completed_at::text||':'||"
        "string_agg(o.updated_at::text,',' ORDER BY o.experiment_id) "
        "FROM experiment_admin_request r "
        "JOIN experiment_admin_worker_outcome o USING(request_id) "
        "WHERE r.request_id=" + std::to_string(requestId) +
            " GROUP BY r.completed_at;");
    {
        pqxx::connection restarted{connectionString};
        pqxx::work replay{restarted};
        AcquireCoordinationLock(replay);
        ReconcileActiveCancellationForTest(replay, requestId);
        replay.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT r.completed_at::text||':'||"
              "string_agg(o.updated_at::text,',' ORDER BY o.experiment_id) "
              "FROM experiment_admin_request r "
              "JOIN experiment_admin_worker_outcome o USING(request_id) "
              "WHERE r.request_id=" + std::to_string(requestId) +
                  " GROUP BY r.completed_at;") ==
          terminalTimestamps);

    ResetCrashFixtures(connection);
}

void TestUnresolvedCheckpointRemainsActive(
    const std::string& connectionString,
    pqxx::connection& connection)
{
    pqxx::work fixture{connection};
    const long long requestId = fixture.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,status,previous_global_state,"
        "resulting_global_state,target_count,rejected_count,result_summary) "
        "VALUES ('cancel_all','after_next_checkpoint',false,"
        "'crash-window-genuinely-unresolved','crash-fixture-requester',"
        "'pending','running','running',1,1,"
        "'{\"target_count\":1,\"rejected_count\":1,"
        "\"pending_count\":1}'::jsonb) RETURNING request_id;")
                                    [0][0]
                                        .as<long long>();
    fixture.exec_params(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,current_epoch,checkpoint_interval,"
        "target_epochs,cancellation_request_id,cancel_after_checkpoint_epoch,"
        "stop_after_checkpoint_epoch,last_checkpoint_stop_decision_epoch,"
        "current_operation) VALUES "
        "(700030,'pending','train',60,20,100,$1,80,80,60,"
        "'train');",
        requestId);
    const long long modelId = fixture.exec(
        "INSERT INTO model(experiment_id,comment) VALUES "
        "(700030,'periodic training checkpoint') RETURNING model_id;")
                                  [0][0]
                                      .as<long long>();
    fixture.exec_params(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "VALUES ($1,'train_config_meta',0,10,60);",
        modelId);
    fixture.exec_params(
        "UPDATE experiment SET last_model_id=$1 "
        "WHERE experiment_id=700030;",
        modelId);
    fixture.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,worker_kind,phase,"
        "lifecycle_status,identity_result,signal_result,"
        "cancellation_checkpoint_epoch,inference_action,outcome_status,detail) "
        "VALUES ($1,'experiment:700030',700030,'experiment','train','running',"
        "'identity_validation_failed','identity_validation_failed',80,'none',"
        "'pending_checkpoint',"
        "'missing_worker_requeued_from_durable_checkpoint');",
        requestId);
    fixture.exec_params(
        "UPDATE experiment_global_control SET desired_state='running',"
        "active_request_id=$1,revision=revision+1,updated_at=now() "
        "WHERE singleton;",
        requestId);
    fixture.commit();

    for (int attempt = 0; attempt < 2; ++attempt)
    {
        pqxx::connection restarted{connectionString};
        pqxx::work replay{restarted};
        AcquireCoordinationLock(replay);
        ReconcileActiveCancellationForTest(replay, requestId);
        replay.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||(completed_at IS NULL)::text||':'||"
              "target_count::text||':'||rejected_count::text||':'||"
              "(result_summary->>'pending_count') "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "pending:true:1:1:1");
    CHECK(Scalar(
              connection,
              "SELECT outcome_status||':'||"
              "(cancellation_checkpoint_model_id IS NULL)::text||':'||detail "
              "FROM experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId)) ==
          "pending_checkpoint:true:"
          "missing_worker_requeued_from_durable_checkpoint");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id::text FROM experiment_global_control "
              "WHERE singleton") == std::to_string(requestId));
    CHECK(Scalar(
              connection,
              "SELECT status||':'||phase||':'||current_epoch::text||':'||"
              "last_model_id::text FROM experiment "
              "WHERE experiment_id=700030") ==
          "pending:train:60:" + std::to_string(modelId));

    RecordingNativeProcesses processes;
    std::string resumeOutput;
    CHECK(Replay(
              connectionString,
              ReplayCommand(
                  Action::ResumeAll,
                  "crash-window-resume-while-unresolved"),
              processes,
              resumeOutput) == 1);
    CHECK(processes.signals.empty());
    CHECK(Scalar(
              connection,
              "SELECT active_request_id::text FROM experiment_global_control "
              "WHERE singleton") == std::to_string(requestId));
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity="
              "'crash-window-resume-while-unresolved'") == "0");

    ResetCrashFixtures(connection);
}

void TestImmediateAndCurrentBoundaryInferenceIdentity(
    const std::string& connectionString,
    pqxx::connection& connection)
{
    pqxx::work fixture{connection};
    fixture.exec(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,current_epoch,checkpoint_interval,"
        "target_epochs,infer_start,infer_end,worker_pid,"
        "worker_process_group_id,worker_executable,worker_command_line,"
        "worker_process_start_identity,current_operation) VALUES "
        "(700040,'running','train',45,20,100,'2020-02-02','2020-03-01',"
        "970040,970040,'/tmp/LSTM_Release',"
        "'/tmp/LSTM_Release --train --scheduler-experiment-id=700040',"
        "'1700000040:40','train');");
    fixture.exec(
        "INSERT INTO model(experiment_id,comment) VALUES "
        "(700040,'periodic training checkpoint');");
    fixture.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT model_id,'train_config_meta',0,10,40 "
        "FROM model WHERE experiment_id=700040;");
    fixture.exec(
        "UPDATE experiment e SET last_model_id=m.model_id "
        "FROM model m WHERE m.experiment_id=e.experiment_id "
        "AND e.experiment_id=700040;");
    BindPersistedFixtureAttempt(fixture, 700040, "train");
    fixture.commit();

    RecordingNativeProcesses processes;
    Command immediate = ReplayCommand(
        Action::CancelAll,
        "crash-window-exact-inference-immediate",
        CancellationMode::Immediate);
    immediate.inferBeforeCancel = true;
    std::string output;
    CHECK(Replay(
              connectionString, immediate, processes, output) == 0);
    CHECK(processes.signals.empty());
    const long long immediateRequestId = std::stoll(Scalar(
        connection,
        "SELECT request_id::text FROM experiment_admin_request "
        "WHERE invocation_identity="
        "'crash-window-exact-inference-immediate'"));
    CHECK(Scalar(
              connection,
              "SELECT o.cancellation_checkpoint_epoch::text||':'||"
              "o.cancellation_checkpoint_model_id::text||':'||"
              "ce.checkpoint_epoch::text||':'||ce.checkpoint_model_id::text||':'||"
              "o.inference_action||':'||o.outcome_status "
              "FROM experiment_admin_worker_outcome o "
              "JOIN experiment_checkpoint_eval ce "
              "ON ce.cancellation_request_id=o.request_id "
              "AND COALESCE(ce.parent_experiment_id,ce.experiment_id)="
              "o.experiment_id "
              "AND ce.checkpoint_epoch=o.cancellation_checkpoint_epoch "
              "AND ce.checkpoint_model_id=o.cancellation_checkpoint_model_id "
              "WHERE o.request_id=" + std::to_string(immediateRequestId) +
                  " AND o.experiment_id=700040") ==
          "40:" +
              Scalar(connection,
                     "SELECT last_model_id::text FROM experiment "
                     "WHERE experiment_id=700040") +
              ":40:" +
              Scalar(connection,
                     "SELECT last_model_id::text FROM experiment "
                     "WHERE experiment_id=700040") +
              ":queued:awaiting_inference");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id=700040") == "1");

    // Replaying the durable request before inference finishes must neither
    // select a new identity nor create a duplicate evaluation.
    CHECK(Replay(
              connectionString, immediate, processes, output) == 0);
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id=700040") == "1");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_admin_request "
              "WHERE invocation_identity="
              "'crash-window-exact-inference-immediate'") == "1");
    {
        pqxx::work update{connection};
        update.exec_params(
            "UPDATE experiment_checkpoint_eval SET status='completed',"
            "phase='done',completed_at=now(),updated_at=now() "
            "WHERE cancellation_request_id=$1;",
            immediateRequestId);
        update.commit();
    }
    {
        pqxx::connection restarted{connectionString};
        pqxx::work reconcile{restarted};
        AcquireCoordinationLock(reconcile);
        ReconcileActiveCancellationForTest(
            reconcile, immediateRequestId);
        reconcile.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||target_count::text||':'||"
              "missing_count::text||':'||failed_count::text||':'||"
              "(result_summary->>'pending_count') "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(immediateRequestId)) ==
          "completed:1:1:0:0");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL "
              "FROM experiment_global_control WHERE singleton") == "t");

    // Seed a separate running fixture so a new request can exercise the
    // after-next-checkpoint current-boundary selection.
    {
        pqxx::work boundaryFixture{connection};
        boundaryFixture.exec(
            "INSERT INTO experiment ("
            "experiment_id,status,phase,current_epoch,checkpoint_interval,"
            "target_epochs,infer_start,infer_end,worker_pid,"
            "worker_process_group_id,worker_executable,worker_command_line,"
            "worker_process_start_identity,current_operation) VALUES "
            "(700041,'running','train',40,20,100,"
            "'2020-02-02','2020-03-01',970041,970041,"
            "'/tmp/LSTM_Release',"
            "'/tmp/LSTM_Release --train --scheduler-experiment-id=700041',"
            "'1700000041:41','train');");
        BindPersistedFixtureAttempt(
            boundaryFixture, 700041, "train");
        boundaryFixture.exec(
            "INSERT INTO model(experiment_id,comment) VALUES "
            "(700041,'periodic training checkpoint');");
        boundaryFixture.exec(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
            "SELECT model_id,'train_config_meta',0,10,40 "
            "FROM model WHERE experiment_id=700041;");
        boundaryFixture.exec(
            "UPDATE experiment e SET last_model_id=m.model_id "
            "FROM model m WHERE m.experiment_id=e.experiment_id "
            "AND e.experiment_id=700041;");
        boundaryFixture.commit();
    }
    processes.signals.clear();
    Command boundary = ReplayCommand(
        Action::CancelAll,
        "crash-window-exact-inference-current-boundary",
        CancellationMode::AfterNextCheckpoint);
    boundary.inferBeforeCancel = true;
    CHECK(Replay(
              connectionString, boundary, processes, output) == 0);
    CHECK(processes.signals.empty());
    const long long boundaryRequestId = std::stoll(Scalar(
        connection,
        "SELECT request_id::text FROM experiment_admin_request "
        "WHERE invocation_identity="
        "'crash-window-exact-inference-current-boundary'"));
    CHECK(Scalar(
              connection,
              "SELECT o.cancellation_checkpoint_epoch::text||':'||"
              "o.cancellation_checkpoint_model_id::text||':'||"
              "ce.checkpoint_epoch::text||':'||ce.checkpoint_model_id::text||':'||"
              "o.inference_action||':'||o.outcome_status "
              "FROM experiment_admin_worker_outcome o "
              "JOIN experiment_checkpoint_eval ce "
              "ON ce.cancellation_request_id=o.request_id "
              "AND COALESCE(ce.parent_experiment_id,ce.experiment_id)="
              "o.experiment_id "
              "AND ce.checkpoint_epoch=o.cancellation_checkpoint_epoch "
              "AND ce.checkpoint_model_id=o.cancellation_checkpoint_model_id "
              "WHERE o.request_id=" + std::to_string(boundaryRequestId)) ==
          "40:" +
              Scalar(connection,
                     "SELECT last_model_id::text FROM experiment "
                     "WHERE experiment_id=700041") +
              ":40:" +
              Scalar(connection,
                     "SELECT last_model_id::text FROM experiment "
                     "WHERE experiment_id=700041") +
              ":queued:awaiting_inference");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id=700041") == "1");
    {
        pqxx::work update{connection};
        update.exec_params(
            "UPDATE experiment_checkpoint_eval SET status='failed',"
            "phase='done',completed_at=now(),updated_at=now(),"
            "error_message='current_boundary_inference_failed' "
            "WHERE cancellation_request_id=$1;",
            boundaryRequestId);
        update.commit();
    }
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        pqxx::connection restarted{connectionString};
        pqxx::work reconcile{restarted};
        AcquireCoordinationLock(reconcile);
        ReconcileActiveCancellationForTest(
            reconcile, boundaryRequestId);
        reconcile.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT inference_action||':'||outcome_status||':'||detail "
              "FROM experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(boundaryRequestId)) ==
          "failed:partial:current_boundary_inference_failed");
    CHECK(Scalar(
              connection,
              "SELECT status||':'||target_count::text||':'||"
              "missing_count::text||':'||failed_count::text||':'||"
              "(result_summary->>'pending_count') "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(boundaryRequestId)) ==
          "partial:1:1:1:0");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL "
              "FROM experiment_global_control WHERE singleton") == "t");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id=700041") == "1");

    ResetCrashFixtures(connection);
}

void TestTerminalInferenceReconciliation(
    const std::string& connectionString,
    pqxx::connection& connection)
{
    pqxx::work fixture{connection};
    const long long requestId = fixture.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,status,previous_global_state,"
        "resulting_global_state,target_count,rejected_count,result_summary) "
        "VALUES ('cancel_all','after_next_checkpoint',true,"
        "'crash-window-terminal-inference-matrix','crash-fixture-requester',"
        "'pending','running','running',6,6,"
        "'{\"target_count\":6,\"rejected_count\":6,"
        "\"pending_count\":6}'::jsonb) RETURNING request_id;")
                                    [0][0]
                                        .as<long long>();
    fixture.exec_params(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,current_epoch,checkpoint_interval,"
        "target_epochs,infer_start,infer_end,completed_at,updated_at,"
        "current_operation,cancellation_request_id,"
        "cancel_after_checkpoint_epoch,stop_after_checkpoint_epoch,"
        "stopped_at_checkpoint_epoch) "
        "SELECT experiment_id,'completed','done',80,20,100,"
        "CASE WHEN experiment_id=700055 THEN NULL ELSE '2020-02-02'::date END,"
        "CASE WHEN experiment_id=700055 THEN NULL ELSE '2020-03-01'::date END,"
        "now()-interval '1 hour',now()-interval '1 hour','analyze',"
        "$1,80,80,80 FROM generate_series(700050,700055) AS experiment_id;",
        requestId);
    fixture.exec(
        "INSERT INTO model(experiment_id,comment) "
        "SELECT experiment_id,'periodic training checkpoint' "
        "FROM generate_series(700050,700055) AS experiment_id;");
    fixture.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT model_id,'train_config_meta',0,10,80 FROM model "
        "WHERE experiment_id BETWEEN 700050 AND 700055;");
    fixture.exec(
        "UPDATE experiment e SET stopped_at_checkpoint_model_id=m.model_id,"
        "last_model_id=m.model_id FROM model m "
        "WHERE m.experiment_id=e.experiment_id "
        "AND e.experiment_id BETWEEN 700050 AND 700055;");
    fixture.exec(
        "INSERT INTO experiment_checkpoint_eval ("
        "experiment_id,parent_experiment_id,checkpoint_epoch,"
        "checkpoint_model_id,status,phase,worker_pid,started_at,"
        "completed_at,error_message) "
        "SELECT e.experiment_id,e.experiment_id,80,"
        "e.stopped_at_checkpoint_model_id,"
        "CASE e.experiment_id WHEN 700050 THEN 'completed' "
        "WHEN 700051 THEN 'failed' WHEN 700052 THEN 'pending' "
        "ELSE 'running' END,"
        "CASE WHEN e.experiment_id IN (700050,700051) THEN 'done' "
        "ELSE 'infer' END,"
        "CASE WHEN e.experiment_id=700053 THEN 970053 ELSE NULL END,"
        "CASE WHEN e.experiment_id=700053 "
        "THEN now()-interval '40 minutes' ELSE NULL END,"
        "CASE WHEN e.experiment_id IN (700050,700051) "
        "THEN now()-interval '30 minutes' ELSE NULL END,"
        "CASE WHEN e.experiment_id=700051 "
        "THEN 'existing_exact_inference_failed' ELSE NULL END "
        "FROM experiment e WHERE e.experiment_id BETWEEN 700050 AND 700053;");
    fixture.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,worker_kind,phase,"
        "lifecycle_status,identity_result,signal_result,"
        "cancellation_checkpoint_epoch,inference_action,outcome_status,detail) "
        "SELECT $1,'experiment:'||experiment_id::text,experiment_id,"
        "'experiment','train','running','identity_validation_failed',"
        "'identity_validation_failed',80,'none','pending_checkpoint',"
        "'missing_worker_requeued_from_durable_checkpoint' "
        "FROM generate_series(700050,700055) AS experiment_id;",
        requestId);
    fixture.exec_params(
        "UPDATE experiment_global_control SET desired_state='running',"
        "active_request_id=$1,revision=revision+1,updated_at=now() "
        "WHERE singleton;",
        requestId);
    fixture.commit();

    for (int attempt = 0; attempt < 2; ++attempt)
    {
        pqxx::connection restarted{connectionString};
        pqxx::work reconcile{restarted};
        AcquireCoordinationLock(reconcile);
        ReconcileActiveCancellationForTest(reconcile, requestId);
        reconcile.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT string_agg(experiment_id::text||':'||inference_action||"
              "':'||outcome_status||':'||"
              "(cancellation_checkpoint_epoch=80)::text||':'||"
              "(cancellation_checkpoint_model_id IS NOT NULL)::text,',' "
              "ORDER BY experiment_id) "
              "FROM experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId)) ==
          "700050:completed:completed:true:true,"
          "700051:failed:partial:true:true,"
          "700052:queued:awaiting_inference:true:true,"
          "700053:running:awaiting_inference:true:true,"
          "700054:queued:awaiting_inference:true:true,"
          "700055:no_checkpoint:partial:true:true");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text||':'||"
              "count(*) FILTER (WHERE cancellation_request_id=" +
                  std::to_string(requestId) +
                  ")::text FROM experiment_checkpoint_eval "
                  "WHERE parent_experiment_id BETWEEN 700050 AND 700055") ==
          "5:5");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM "
              "experiment_admin_worker_outcome o "
              "LEFT JOIN experiment_checkpoint_eval ce "
              "ON ce.cancellation_request_id=o.request_id "
              "AND COALESCE(ce.parent_experiment_id,ce.experiment_id)="
              "o.experiment_id "
              "AND ce.checkpoint_epoch=o.cancellation_checkpoint_epoch "
              "AND ce.checkpoint_model_id=o.cancellation_checkpoint_model_id "
              "WHERE o.request_id=" + std::to_string(requestId) +
                  " AND o.inference_action='queued' "
                  "AND ce.checkpoint_eval_id IS NULL") == "0");
    CHECK(Scalar(
              connection,
              "SELECT status||':'||failed_count::text||':'||"
              "(result_summary->>'pending_count') "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "pending:2:3");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id::text FROM experiment_global_control "
              "WHERE singleton") == std::to_string(requestId));

    {
        pqxx::work update{connection};
        update.exec_params(
            "UPDATE experiment_checkpoint_eval SET "
            "status=CASE WHEN parent_experiment_id=700053 "
            "THEN 'failed' ELSE 'completed' END,phase='done',"
            "worker_pid=NULL,completed_at=now(),updated_at=now(),"
            "error_message=CASE WHEN parent_experiment_id=700053 "
            "THEN 'running_exact_inference_failed' ELSE NULL END "
            "WHERE cancellation_request_id=$1 "
            "AND parent_experiment_id IN (700052,700053,700054);",
            requestId);
        update.commit();
    }
    {
        pqxx::connection restarted{connectionString};
        pqxx::work reconcile{restarted};
        AcquireCoordinationLock(reconcile);
        ReconcileActiveCancellationForTest(reconcile, requestId);
        reconcile.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||target_count::text||':'||"
              "successful_count::text||':'||rejected_count::text||':'||"
              "failed_count::text||':'||(result_summary->>'pending_count')||':'||"
              "(completed_at IS NOT NULL)::text "
              "FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "partial:6:0:6:3:0:true");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL FROM "
              "experiment_global_control WHERE singleton") == "t");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id BETWEEN 700050 AND 700055") == "5");
    const std::string terminalState = Scalar(
        connection,
        "SELECT r.completed_at::text||':'||"
        "string_agg(o.updated_at::text,',' ORDER BY o.experiment_id) "
        "FROM experiment_admin_request r "
        "JOIN experiment_admin_worker_outcome o USING(request_id) "
        "WHERE r.request_id=" + std::to_string(requestId) +
            " GROUP BY r.completed_at;");
    {
        pqxx::connection restarted{connectionString};
        pqxx::work replay{restarted};
        AcquireCoordinationLock(replay);
        ReconcileActiveCancellationForTest(replay, requestId);
        replay.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT r.completed_at::text||':'||"
              "string_agg(o.updated_at::text,',' ORDER BY o.experiment_id) "
              "FROM experiment_admin_request r "
              "JOIN experiment_admin_worker_outcome o USING(request_id) "
              "WHERE r.request_id=" + std::to_string(requestId) +
                  " GROUP BY r.completed_at;") ==
          terminalState);

    ResetCrashFixtures(connection);
}

void TestLegacyInferenceUpgradeMatrix(
    const std::string& connectionString,
    pqxx::connection& connection)
{
    pqxx::work fixture{connection};
    const long long requestId = fixture.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,status,previous_global_state,"
        "resulting_global_state,target_count,result_summary) VALUES ("
        "'cancel_all','after_next_checkpoint',true,"
        "'crash-window-legacy-upgrade-matrix','crash-fixture-requester',"
        "'pending','running','running',16,"
        "'{\"target_count\":16,\"pending_count\":15,\"failed_count\":1}'::jsonb) "
        "RETURNING request_id;")[0][0].as<long long>();
    const long long foreignRequestId = fixture.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,status,previous_global_state,"
        "resulting_global_state) VALUES ("
        "'cancel_all','immediate',true,"
        "'crash-window-legacy-foreign-owner','crash-fixture-requester',"
        "'completed','running','running') RETURNING request_id;")
                                           [0][0]
                                               .as<long long>();
    fixture.exec_params(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,current_epoch,checkpoint_interval,"
        "target_epochs,infer_start,infer_end,completed_at,"
        "cancellation_completed_at,current_operation,cancellation_request_id,"
        "cancel_after_checkpoint_epoch,stop_after_checkpoint_epoch,"
        "stopped_at_checkpoint_epoch) "
        "SELECT experiment_id,'cancelled','train',80,20,100,"
        "'2020-02-02'::date,'2020-03-01'::date,now(),now(),"
        "'train',$1,80,80,80 "
        "FROM generate_series(700060,700074) AS experiment_id;",
        requestId);
    fixture.exec(
        "INSERT INTO experiment (experiment_id,status,phase,current_epoch,"
        "current_operation) VALUES "
        "(700075,'completed','done',80,'analyze'),"
        "(700076,'completed','done',80,'analyze');");
    fixture.exec(
        "INSERT INTO model(experiment_id,comment) "
        "SELECT experiment_id,'periodic training checkpoint' "
        "FROM generate_series(700060,700076) AS experiment_id;");
    fixture.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT model_id,'train_config_meta',0,10,80 FROM model "
        "WHERE experiment_id BETWEEN 700060 AND 700076;");
    fixture.exec(
        "INSERT INTO model(experiment_id,comment) VALUES "
        "(700065,'periodic training checkpoint epoch 60'),"
        "(700067,'conflicting periodic training checkpoint');");
    fixture.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT model_id,'train_config_meta',0,10,"
        "CASE WHEN experiment_id=700065 THEN 60 ELSE 80 END "
        "FROM model WHERE experiment_id IN (700065,700067) "
        "AND comment NOT LIKE 'periodic training checkpoint';");
    fixture.exec(
        "UPDATE experiment e SET stopped_at_checkpoint_model_id=m.model_id,"
        "last_model_id=m.model_id FROM model m "
        "WHERE m.experiment_id=e.experiment_id "
        "AND m.comment='periodic training checkpoint' "
        "AND e.experiment_id BETWEEN 700060 AND 700074;");
    fixture.exec(
        "UPDATE experiment SET stopped_at_checkpoint_model_id=NULL,"
        "last_model_id=NULL WHERE experiment_id=700068;");
    fixture.exec(
        "UPDATE experiment SET stopped_at_checkpoint_model_id=("
        "SELECT model_id FROM model WHERE experiment_id=700076),"
        "last_model_id=(SELECT model_id FROM model "
        "WHERE experiment_id=700076) WHERE experiment_id=700069;");

    fixture.exec_params(
        "INSERT INTO experiment_checkpoint_eval ("
        "experiment_id,parent_experiment_id,checkpoint_epoch,"
        "checkpoint_model_id,status,phase,started_at,completed_at,worker_pid,"
        "error_message,cancellation_request_id) "
        "SELECT e.experiment_id,e.experiment_id,80,"
        "e.stopped_at_checkpoint_model_id,"
        "CASE e.experiment_id "
        "WHEN 700060 THEN 'completed' WHEN 700061 THEN 'failed' "
        "WHEN 700062 THEN 'pending' WHEN 700063 THEN 'running' "
        "WHEN 700066 THEN 'pending' WHEN 700070 THEN 'completed' "
        "WHEN 700071 THEN 'completed' WHEN 700072 THEN 'failed' "
        "WHEN 700073 THEN 'completed' ELSE 'failed' END,"
        "CASE e.experiment_id "
        "WHEN 700061 THEN 'infer' WHEN 700062 THEN 'infer' "
        "WHEN 700063 THEN 'infer' WHEN 700066 THEN 'infer' "
        "WHEN 700070 THEN 'infer' WHEN 700072 THEN 'analyze' "
        "WHEN 700074 THEN 'infer' ELSE 'done' END,"
        "CASE WHEN e.experiment_id=700063 THEN now()-interval '5 minutes' "
        "ELSE NULL END,"
        "CASE WHEN e.experiment_id IN "
        "(700060,700061,700070,700072,700073,700074) "
        "THEN now()-interval '1 minute' ELSE NULL END,"
        "CASE WHEN e.experiment_id=700063 THEN 970063 ELSE NULL END,"
        "CASE WHEN e.experiment_id IN (700061,700074) "
        "THEN 'legacy_exact_failure' ELSE NULL END,"
        "CASE WHEN e.experiment_id=700066 THEN $2::bigint "
        "ELSE $1::bigint END "
        "FROM experiment e WHERE e.experiment_id IN "
        "(700060,700061,700062,700063,700066,700070,700071,"
        "700072,700073,700074);",
        requestId,
        foreignRequestId);
    fixture.exec_params(
        "INSERT INTO experiment_checkpoint_eval ("
        "experiment_id,parent_experiment_id,checkpoint_epoch,"
        "checkpoint_model_id,status,phase,cancellation_request_id) "
        "SELECT 700065,700065,round(tm.value)::int,m.model_id,"
        "'pending','infer',$1 FROM model m JOIN matrix tm USING(model_id) "
        "WHERE m.experiment_id=700065;",
        requestId);

    fixture.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,worker_kind,phase,"
        "lifecycle_status,identity_result,signal_result,"
        "cancellation_checkpoint_epoch,cancellation_checkpoint_model_id,"
        "inference_action,outcome_status,detail) "
        "SELECT $1,'experiment:'||experiment_id::text,experiment_id,"
        "'experiment','train','running','process_missing','process_missing',"
        "CASE WHEN experiment_id IN (700067,700068,700069) THEN 80 "
        "WHEN experiment_id=700073 THEN 80 ELSE NULL END,"
        "CASE WHEN experiment_id=700067 THEN ("
        " SELECT model_id FROM model WHERE experiment_id=700067 "
        " AND comment='conflicting periodic training checkpoint') "
        "WHEN experiment_id=700074 THEN ("
        " SELECT model_id FROM model WHERE experiment_id=700074) "
        "ELSE NULL END,"
        "CASE WHEN experiment_id IN (700067,700068,700069) "
        "THEN 'none' ELSE 'queued' END,"
        "CASE WHEN experiment_id IN (700067,700068,700069) "
        "THEN 'pending_checkpoint' ELSE 'awaiting_inference' END,"
        "'pre_correction_legacy_row_shape' "
        "FROM generate_series(700060,700074) AS experiment_id;",
        requestId);
    fixture.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,worker_kind,phase,"
        "lifecycle_status,identity_result,signal_result,inference_action,"
        "outcome_status,detail) VALUES ("
        "$1,'experiment:700075',700075,'experiment','train','running',"
        "'inspection_failed','signaling_failure','failed','failed',"
        "'pre_correction_terminal_failure');",
        requestId);
    fixture.exec_params(
        "UPDATE experiment_global_control SET desired_state='running',"
        "active_request_id=$1,revision=revision+1,updated_at=now() "
        "WHERE singleton;",
        requestId);
    fixture.commit();

    for (int attempt = 0; attempt < 2; ++attempt)
    {
        pqxx::connection restarted{connectionString};
        pqxx::work reconcile{restarted};
        AcquireCoordinationLock(reconcile);
        ReconcileActiveCancellationForTest(reconcile, requestId);
        reconcile.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT string_agg(experiment_id::text||':'||"
              "outcome_status||':'||inference_action,',' "
              "ORDER BY experiment_id) "
              "FROM experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId)) ==
          "700060:completed:completed,"
          "700061:partial:failed,"
          "700062:awaiting_inference:queued,"
          "700063:awaiting_inference:running,"
          "700064:awaiting_inference:queued,"
          "700065:partial:failed,"
          "700066:partial:failed,"
          "700067:partial:failed,"
          "700068:partial:failed,"
          "700069:partial:failed,"
          "700070:partial:failed,"
          "700071:partial:failed,"
          "700072:partial:failed,"
          "700073:completed:completed,"
          "700074:partial:failed,"
          "700075:failed:failed");
    CHECK(Scalar(
              connection,
              "SELECT status||':'||failed_count::text||':'||"
              "(result_summary->>'pending_count')||':'||"
              "(completed_at IS NULL)::text FROM experiment_admin_request "
              "WHERE request_id=" + std::to_string(requestId)) ==
          "pending:11:3:true");
    CHECK(Scalar(
              connection,
              "SELECT bool_and(cancellation_checkpoint_epoch=80 "
              "AND cancellation_checkpoint_model_id IS NOT NULL)::text "
              "FROM experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId) +
                  " AND experiment_id IN "
                  "(700060,700061,700062,700063,700064,700073,700074)") ==
          "true");
    CHECK(Scalar(
              connection,
              "SELECT cancellation_request_id::text FROM "
              "experiment_checkpoint_eval WHERE parent_experiment_id=700066") ==
          std::to_string(foreignRequestId));
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id=700064 "
              "AND cancellation_request_id=" + std::to_string(requestId)) ==
          "1");

    {
        pqxx::work finish{connection};
        finish.exec_params(
            "UPDATE experiment_checkpoint_eval SET "
            "status=CASE WHEN parent_experiment_id=700063 "
            "THEN 'failed' ELSE 'completed' END,"
            "phase=CASE WHEN parent_experiment_id=700063 "
            "THEN 'infer' ELSE 'done' END,worker_pid=NULL,"
            "completed_at=now(),error_message=CASE "
            "WHEN parent_experiment_id=700063 "
            "THEN 'legacy_running_failure' ELSE NULL END,updated_at=now() "
            "WHERE cancellation_request_id=$1 "
            "AND parent_experiment_id IN (700062,700063,700064);",
            requestId);
        finish.commit();
    }
    {
        pqxx::connection restarted{connectionString};
        pqxx::work reconcile{restarted};
        AcquireCoordinationLock(reconcile);
        ReconcileActiveCancellationForTest(reconcile, requestId);
        reconcile.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||failed_count::text||':'||"
              "(result_summary->>'pending_count')||':'||"
              "(completed_at IS NOT NULL)::text FROM experiment_admin_request "
              "WHERE request_id=" + std::to_string(requestId)) ==
          "partial:12:0:true");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL FROM "
              "experiment_global_control WHERE singleton") == "t");

    const std::string terminalAudit = Scalar(
        connection,
        "SELECT r.completed_at::text||':'||"
        "string_agg(o.updated_at::text,',' ORDER BY o.experiment_id)||':'||"
        "(SELECT count(*)::text FROM experiment_checkpoint_eval ce "
        "WHERE ce.parent_experiment_id BETWEEN 700060 AND 700074)||':'||"
        "(SELECT revision::text FROM experiment_global_control "
        "WHERE singleton) "
        "FROM experiment_admin_request r "
        "JOIN experiment_admin_worker_outcome o USING(request_id) "
        "WHERE r.request_id=" + std::to_string(requestId) +
            " GROUP BY r.completed_at;");
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        pqxx::connection restarted{connectionString};
        pqxx::work replay{restarted};
        AcquireCoordinationLock(replay);
        ReconcileActiveCancellationForTest(replay, requestId);
        replay.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT r.completed_at::text||':'||"
              "string_agg(o.updated_at::text,',' ORDER BY o.experiment_id)||':'||"
              "(SELECT count(*)::text FROM experiment_checkpoint_eval ce "
              "WHERE ce.parent_experiment_id BETWEEN 700060 AND 700074)||':'||"
              "(SELECT revision::text FROM experiment_global_control "
              "WHERE singleton) "
              "FROM experiment_admin_request r "
              "JOIN experiment_admin_worker_outcome o USING(request_id) "
              "WHERE r.request_id=" + std::to_string(requestId) +
                  " GROUP BY r.completed_at;") ==
          terminalAudit);

    // Terminal cancellation reconciliation is already proven idempotent
    // above. PauseAll/ResumeAll now use the scheduler priority-queue path
    // and therefore are not generic post-reconciliation administrative
    // commands for this legacy cancellation fixture.
    ResetCrashFixtures(connection);
}

void TestProductionCheckpointStopOwnership(pqxx::connection& connection)
{
    std::unique_ptr<ProcessOperations> nativeProcesses =
        CreateNativeProcessOperations();
    const ProcessObservation self =
        nativeProcesses->Observe(static_cast<int>(::getpid()));
    CHECK(self.exists);
    CHECK(self.inspectionSucceeded);
    CHECK(self.pid == static_cast<int>(::getpid()));
    CHECK(self.processGroupId == static_cast<int>(::getpgrp()));
    CHECK(!self.processStartIdentity.empty());
    CHECK(!self.executable.empty());
    CHECK(!self.commandLine.empty());

    pqxx::work fixture{connection};
    fixture.exec(
        "SELECT set_config("
        "'expertadvisor.scheduler_protocol_generation','52',true);");
    fixture.exec_params(
        "INSERT INTO experiment_scheduler_invocation("
        "scheduler_invocation_id,process_pid,process_group_id,"
        "process_start_identity,canonical_executable_path,command_line,"
        "invocation_nonce,status,protocol_generation) VALUES("
        "'checkpoint-stop-process-test-owner',$1,$2,$3,$4,$5,"
        "'checkpoint-stop-process-test-nonce','owner',52) "
        "ON CONFLICT (scheduler_invocation_id) DO NOTHING;",
        self.pid,
        self.processGroupId,
        self.processStartIdentity,
        self.executable,
        self.commandLine);
    const long long requestId = fixture.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,status,previous_global_state,"
        "resulting_global_state,target_count,result_summary) VALUES ("
        "'cancel_all','after_next_checkpoint',true,"
        "'crash-window-production-checkpoint-stop',"
        "'crash-fixture-requester','pending','running','running',4,"
        "'{\"target_count\":4,\"pending_count\":4}'::jsonb) "
        "RETURNING request_id;")[0][0].as<long long>();
    const long long foreignRequestId = fixture.exec(
        "INSERT INTO experiment_admin_request ("
        "action,cancellation_mode,infer_before_cancel,invocation_identity,"
        "requester_identity,status,previous_global_state,"
        "resulting_global_state) VALUES ("
        "'cancel_all','immediate',true,"
        "'crash-window-production-foreign-owner',"
        "'crash-fixture-requester','completed','running','running') "
        "RETURNING request_id;")[0][0].as<long long>();
    fixture.exec_params(
        "INSERT INTO experiment ("
        "experiment_id,status,phase,current_epoch,checkpoint_interval,"
        "target_epochs,infer_start,infer_end,worker_pid,"
        "worker_process_group_id,worker_process_start_identity,"
        "worker_executable,worker_command_line,"
        "cancellation_request_id,cancel_infer_before,"
        "cancel_after_checkpoint_epoch,"
        "stop_after_checkpoint_epoch,last_checkpoint_stop_decision_epoch,"
        "current_operation) "
        "SELECT experiment_id,'running','train',79,20,100,"
        "'2020-02-02'::date,'2020-03-01'::date,"
        "$2,$3,$4,$5,$6,$1,true,80,80,80,'train' "
        "FROM generate_series(700080,700083) AS experiment_id;",
        requestId,
        self.pid,
        self.processGroupId,
        self.processStartIdentity,
        self.executable,
        self.commandLine);
    fixture.exec(
        "INSERT INTO model(experiment_id,comment) "
        "SELECT experiment_id,'periodic training checkpoint' "
        "FROM generate_series(700080,700083) AS experiment_id;");
    fixture.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT model_id,'train_config_meta',0,10,80 FROM model "
        "WHERE experiment_id BETWEEN 700080 AND 700083;");
    fixture.exec(
        "INSERT INTO model(experiment_id,comment) VALUES "
        "(700083,'persisted conflicting checkpoint');");
    fixture.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT model_id,'train_config_meta',0,10,80 FROM model "
        "WHERE experiment_id=700083 "
        "AND comment='persisted conflicting checkpoint';");
    fixture.exec_params(
        "WITH attempts AS ("
        " INSERT INTO experiment_scheduler_worker_attempt("
        " launch_attempt_identity,scheduler_invocation_id,"
        " scheduler_fencing_token,experiment_id,worker_kind,"
        " lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,"
        " worker_pid,worker_process_group_id,worker_process_start_identity,"
        " canonical_executable_path,command_line,command_identity,"
        " reserved_at,spawned_at,registered_at,last_observed_at"
        " ) SELECT "
        " 'checkpoint-stop-process-test:'||experiment_id::text,"
        " 'checkpoint-stop-process-test-owner',1,experiment_id,"
        " 'experiment','train','train','scheduler_launch','running',"
        " $1,$2,$3,$4,$5,'experiment:'||experiment_id::text||':train',"
        " clock_timestamp(),clock_timestamp(),clock_timestamp(),"
        " clock_timestamp() "
        " FROM generate_series(700080,700083) AS experiment_id "
        " RETURNING worker_attempt_id,experiment_id"
        ") UPDATE experiment e "
        "SET active_scheduler_worker_attempt_id=a.worker_attempt_id "
        "FROM attempts a WHERE e.experiment_id=a.experiment_id;",
        self.pid,
        self.processGroupId,
        self.processStartIdentity,
        self.executable,
        self.commandLine);
    fixture.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,worker_kind,phase,"
        "lifecycle_status,cancellation_checkpoint_epoch,"
        "cancellation_checkpoint_model_id,inference_action,"
        "outcome_status,detail,worker_attempt_id) "
        "SELECT $1,'experiment:'||experiment_id::text,experiment_id,"
        "'experiment','train','running',80,"
        "CASE WHEN experiment_id=700083 THEN ("
        " SELECT model_id FROM model WHERE experiment_id=700083 "
        " AND comment='persisted conflicting checkpoint') ELSE NULL END,"
        "'none','pending_checkpoint','production_checkpoint_pending',"
        "a.worker_attempt_id "
        "FROM generate_series(700080,700083) AS experiment_id "
        "JOIN experiment_scheduler_worker_attempt a USING(experiment_id) "
        "WHERE a.lifecycle_phase='train' "
        "AND a.launch_attempt_identity LIKE "
        "'checkpoint-stop-process-test:%';",
        requestId);
    fixture.exec_params(
        "INSERT INTO experiment_checkpoint_eval ("
        "experiment_id,parent_experiment_id,checkpoint_epoch,"
        "checkpoint_model_id,status,phase,cancellation_request_id) "
        "SELECT e.experiment_id,e.experiment_id,80,m.model_id,"
        "'pending','infer',CASE WHEN e.experiment_id=700080 "
        "THEN NULL::bigint WHEN e.experiment_id=700081 THEN $1::bigint "
        "ELSE $2::bigint END "
        "FROM experiment e JOIN model m "
        "ON m.experiment_id=e.experiment_id "
        "AND m.comment='periodic training checkpoint' "
        "WHERE e.experiment_id IN (700080,700081,700082);",
        requestId,
        foreignRequestId);
    fixture.exec_params(
        "UPDATE experiment_global_control SET active_request_id=$1,"
        "revision=revision+1,updated_at=now() WHERE singleton;",
        requestId);
    fixture.commit();

    for (long long experimentId = 700080;
         experimentId <= 700083;
         ++experimentId)
    {
        pqxx::work record{connection};
        AcquireCoordinationLock(record);
        const long long modelId = record.exec_params(
            "SELECT model_id FROM model WHERE experiment_id=$1 "
            "AND comment='periodic training checkpoint';",
            experimentId)[0][0].as<long long>();
        const long long attemptId = record.exec_params(
            "SELECT active_scheduler_worker_attempt_id "
            "FROM experiment WHERE experiment_id=$1;",
            experimentId)[0][0].as<long long>();
        const CheckpointStopRecordResult result =
            RecordCheckpointStopReached(
                record, experimentId, attemptId, 80, modelId);
        CHECK(result.recorded);
        record.commit();
    }
    const std::string productionOutcomes = Scalar(
        connection,
        "SELECT string_agg(experiment_id::text||':'||outcome_status||"
        "':'||inference_action,',' ORDER BY experiment_id) "
        "FROM experiment_admin_worker_outcome WHERE request_id=" +
            std::to_string(requestId));
    if (productionOutcomes !=
          "700080:awaiting_inference:queued,"
          "700081:awaiting_inference:queued,"
          "700082:partial:failed,"
          "700083:partial:failed")
        std::cerr << "production_checkpoint_outcomes="
                  << productionOutcomes << '\n';
    CHECK(productionOutcomes ==
          "700080:awaiting_inference:queued,"
          "700081:awaiting_inference:queued,"
          "700082:partial:failed,"
          "700083:partial:failed");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM "
              "experiment_scheduler_worker_attempt "
              "WHERE experiment_id BETWEEN 700080 AND 700083 "
              "AND capacity_class='train' "
              "AND lifecycle_state IN "
              "('reserved','spawned','running','observed',"
              "'identity_ambiguous')") == "0");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment "
              "WHERE experiment_id BETWEEN 700080 AND 700083 "
              "AND active_scheduler_worker_attempt_id IS NOT NULL") == "0");
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM "
              "experiment_scheduler_worker_attempt "
              "WHERE experiment_id BETWEEN 700080 AND 700083 "
              "AND lifecycle_state='completed' "
              "AND reconciliation_result='checkpoint_stop_completed' "
              "AND diagnostic LIKE "
              "'checkpoint_stop;checkpoint_epoch=80;"
              "checkpoint_model_id=%;next_phase=cancelled;"
              "worker_attempt_id=%'") == "4");
    CHECK(Scalar(
              connection,
              "SELECT string_agg(parent_experiment_id::text||':'||"
              "cancellation_request_id::text,',' "
              "ORDER BY parent_experiment_id) "
              "FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id BETWEEN 700080 AND 700082") ==
          "700080:" + std::to_string(requestId) + ",700081:" +
              std::to_string(requestId) + ",700082:" +
              std::to_string(foreignRequestId));
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM experiment_checkpoint_eval "
              "WHERE parent_experiment_id=700083") == "0");

    const std::string replayAudit = Scalar(
        connection,
        "SELECT o.updated_at::text||':'||ce.updated_at::text||':'||"
        "ce.cancellation_request_id::text FROM "
        "experiment_admin_worker_outcome o "
        "JOIN experiment_checkpoint_eval ce "
        "ON ce.parent_experiment_id=o.experiment_id "
        "WHERE o.request_id=" + std::to_string(requestId) +
            " AND o.experiment_id=700081;");
    {
        pqxx::work replay{connection};
        AcquireCoordinationLock(replay);
        const long long modelId = replay.exec(
            "SELECT model_id FROM model WHERE experiment_id=700081 "
            "AND comment='periodic training checkpoint';")[0][0].as<long long>();
        const long long attemptId = replay.exec(
            "SELECT worker_attempt_id FROM "
            "experiment_scheduler_worker_attempt "
            "WHERE experiment_id=700081 AND lifecycle_phase='train';")
                                        [0][0].as<long long>();
        const CheckpointStopRecordResult result =
            RecordCheckpointStopReached(
                replay, 700081, attemptId, 80, modelId);
        CHECK(result.recorded);
        replay.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT o.updated_at::text||':'||ce.updated_at::text||':'||"
              "ce.cancellation_request_id::text FROM "
              "experiment_admin_worker_outcome o "
              "JOIN experiment_checkpoint_eval ce "
              "ON ce.parent_experiment_id=o.experiment_id "
              "WHERE o.request_id=" + std::to_string(requestId) +
                  " AND o.experiment_id=700081;") ==
          replayAudit);

    {
        pqxx::work finish{connection};
        finish.exec_params(
            "UPDATE experiment_checkpoint_eval SET status='completed',"
            "phase='done',completed_at=now(),updated_at=now() "
            "WHERE cancellation_request_id=$1 "
            "AND parent_experiment_id IN (700080,700081);",
            requestId);
        AcquireCoordinationLock(finish);
        ReconcileActiveCancellationForTest(finish, requestId);
        finish.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||failed_count::text||':'||"
              "(result_summary->>'pending_count') FROM "
              "experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) ==
          "partial:2:0");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL FROM "
              "experiment_global_control WHERE singleton") == "t");

    {
        pqxx::work staleFixture{connection};
        const long long staleRequestId = staleFixture.exec(
            "INSERT INTO experiment_admin_request ("
            "action,cancellation_mode,infer_before_cancel,invocation_identity,"
            "requester_identity,application_owner,status,"
            "previous_global_state,resulting_global_state,target_count,"
            "result_summary) VALUES ("
            "'cancel_all','after_next_checkpoint',true,"
            "'crash-window-stale-checkpoint-stop',"
            "'crash-fixture-requester','crash-window-stale-checkpoint-stop',"
            "'pending','running','running',1,"
            "'{\"target_count\":1,\"pending_count\":1}'::jsonb) "
            "RETURNING request_id;")[0][0].as<long long>();
        staleFixture.exec_params(
            "INSERT INTO experiment ("
            "experiment_id,status,phase,current_epoch,checkpoint_interval,"
            "target_epochs,infer_start,infer_end,cancellation_request_id,"
            "cancel_infer_before,cancel_after_checkpoint_epoch,"
            "stop_after_checkpoint_epoch,current_operation) VALUES ("
            "700084,'pending','train',79,20,100,'2020-02-02','2020-03-01',"
            "$1,true,80,80,'train');",
            staleRequestId);
        const long long staleModelId = staleFixture.exec(
            "INSERT INTO model(experiment_id,comment) VALUES "
            "(700084,'periodic training checkpoint') RETURNING model_id;")
                                               [0][0]
                                                   .as<long long>();
        staleFixture.exec_params(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
            "VALUES ($1,'train_config_meta',0,10,80);",
            staleModelId);
        staleFixture.exec_params(
            "INSERT INTO experiment_admin_worker_outcome ("
            "request_id,worker_identity,experiment_id,worker_kind,phase,"
            "lifecycle_status,cancellation_checkpoint_epoch,inference_action,"
            "outcome_status,detail) VALUES ("
            "$1,'experiment:700084',700084,'experiment','train','running',"
            "80,'none','pending_checkpoint','stale_worker_fixture');",
            staleRequestId);
        staleFixture.exec_params(
            "UPDATE experiment_global_control SET active_request_id=$1,"
            "revision=revision+1,updated_at=now() WHERE singleton;",
            staleRequestId);
        staleFixture.commit();

        pqxx::work staleRecord{connection};
        AcquireCoordinationLock(staleRecord);
        const CheckpointStopRecordResult staleResult =
            RecordCheckpointStopReached(
                staleRecord,
                700084,
                staleRecord.exec(
                    "SELECT worker_attempt_id FROM "
                    "experiment_scheduler_worker_attempt "
                    "WHERE experiment_id=700081;")[0][0].as<long long>(),
                80,
                staleModelId);
        CHECK(!staleResult.recorded);
        CHECK(staleResult.detail == "exact_active_train_attempt_mismatch");
        staleRecord.commit();

        CHECK(Scalar(
                  connection,
                  "SELECT status||':'||current_epoch::text||':'||"
                  "(stopped_at_checkpoint_epoch IS NULL)::text||':'||"
                  "(stopped_at_checkpoint_model_id IS NULL)::text "
                  "FROM experiment WHERE experiment_id=700084") ==
              "pending:79:true:true");
        CHECK(Scalar(
                  connection,
                  "SELECT outcome_status||':'||inference_action||':'||detail "
                  "FROM experiment_admin_worker_outcome "
                  "WHERE request_id=" + std::to_string(staleRequestId)) ==
              "pending_checkpoint:none:stale_worker_fixture");
        CHECK(Scalar(
                  connection,
                  "SELECT count(*)::text FROM experiment_checkpoint_eval "
                  "WHERE parent_experiment_id=700084") == "0");
    }

    // Ordinary scenario-37 transition with a production-faithful durable
    // scheduler attempt. Claim a replacement, then prove old-attempt replay,
    // wrong-attempt rejection, and a new connection (restart boundary) are
    // all nonmutating.
    long long ordinaryAttemptId = -1;
    long long ordinaryModelId = -1;
    {
        pqxx::work ordinaryFixture{connection};
        ordinaryFixture.exec(
            "SELECT set_config("
            "'expertadvisor.scheduler_protocol_generation','52',true);");
        ordinaryFixture.exec_params(
            "INSERT INTO experiment("
            "experiment_id,status,phase,current_epoch,checkpoint_interval,"
            "target_epochs,infer_start,infer_end,worker_pid,"
            "worker_process_group_id,worker_process_start_identity,"
            "worker_executable,worker_command_line,worker_control_state,"
            "stop_after_checkpoint_epoch,last_checkpoint_stop_decision_epoch,"
            "current_operation) VALUES("
            "700085,'running','train',19,20,80,"
            "'2020-02-02','2020-03-01',$1,$2,$3,$4,$5,'running',"
            "20,20,'train');",
            self.pid,
            self.processGroupId,
            self.processStartIdentity,
            self.executable,
            self.commandLine);
        ordinaryModelId = ordinaryFixture.exec(
            "INSERT INTO model(experiment_id,comment) VALUES("
            "700085,'periodic training checkpoint') RETURNING model_id;")
                                      [0][0].as<long long>();
        ordinaryFixture.exec_params(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
            "VALUES($1,'train_config_meta',0,10,20);",
            ordinaryModelId);
        ordinaryAttemptId = ordinaryFixture.exec_params(
            "INSERT INTO experiment_scheduler_worker_attempt("
            "launch_attempt_identity,scheduler_invocation_id,"
            "scheduler_fencing_token,experiment_id,worker_kind,"
            "lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,"
            "worker_pid,worker_process_group_id,"
            "worker_process_start_identity,canonical_executable_path,"
            "command_line,command_identity,reserved_at,spawned_at,"
            "registered_at,last_observed_at) VALUES("
            "'checkpoint-stop-process-test:700085',"
            "'checkpoint-stop-process-test-owner',1,700085,'experiment',"
            "'train','train','scheduler_launch','running',$1,$2,$3,$4,$5,"
            "'experiment:700085:train',clock_timestamp(),clock_timestamp(),"
            "clock_timestamp(),clock_timestamp()) "
            "RETURNING worker_attempt_id;",
            self.pid,
            self.processGroupId,
            self.processStartIdentity,
            self.executable,
            self.commandLine)[0][0].as<long long>();
        ordinaryFixture.exec_params(
            "UPDATE experiment SET active_scheduler_worker_attempt_id=$1 "
            "WHERE experiment_id=700085;",
            ordinaryAttemptId);
        ordinaryFixture.commit();
    }
    {
        pqxx::work transition{connection};
        const CheckpointStopRecordResult result =
            RecordCheckpointStopReached(
                transition,
                700085,
                ordinaryAttemptId,
                20,
                ordinaryModelId);
        CHECK(result.recorded);
        CHECK(!result.cancellationRequested);
        CHECK(result.workerAttemptId ==
              std::optional<long long>{ordinaryAttemptId});
        transition.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT status||':'||phase||':'||current_operation||':'||"
              "(active_scheduler_worker_attempt_id IS NULL)::text||':'||"
              "(worker_pid IS NULL)::text||':'||"
              "(worker_process_group_id IS NULL)::text||':'||"
              "(worker_process_start_identity IS NULL)::text||':'||"
              "(worker_executable IS NULL)::text||':'||"
              "(worker_command_line IS NULL)::text||':'||"
              "stopped_at_checkpoint_epoch::text||':'||"
              "stopped_at_checkpoint_model_id::text "
              "FROM experiment WHERE experiment_id=700085") ==
          "pending:infer:infer:true:true:true:true:true:true:20:" +
              std::to_string(ordinaryModelId));
    CHECK(Scalar(
              connection,
              "SELECT lifecycle_state||':'||capacity_class||':'||"
              "reconciliation_result FROM "
              "experiment_scheduler_worker_attempt "
              "WHERE worker_attempt_id=" +
                  std::to_string(ordinaryAttemptId)) ==
          "completed:train:checkpoint_stop_completed");

    long long replacementAttemptId = -1;
    {
        pqxx::work claim{connection};
        claim.exec(
            "SELECT set_config("
            "'expertadvisor.scheduler_protocol_generation','52',true);");
        replacementAttemptId = claim.exec(
            "INSERT INTO experiment_scheduler_worker_attempt("
            "launch_attempt_identity,scheduler_invocation_id,"
            "scheduler_fencing_token,experiment_id,worker_kind,"
            "lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,"
            "canonical_executable_path,command_identity) VALUES("
            "'checkpoint-stop-process-test:700085:infer',"
            "'checkpoint-stop-process-test-owner',1,700085,'experiment',"
            "'infer','infer','scheduler_launch','reserved','/tmp/test',"
            "'experiment:700085:infer') RETURNING worker_attempt_id;")
                                   [0][0].as<long long>();
        const pqxx::result claimed = claim.exec_params(
            "UPDATE experiment SET status='running',phase='infer',"
            "current_operation='infer',"
            "active_scheduler_worker_attempt_id=$1 "
            "WHERE experiment_id=700085 AND status='pending' "
            "AND phase='infer' "
            "AND active_scheduler_worker_attempt_id IS NULL "
            "RETURNING experiment_id;",
            replacementAttemptId);
        CHECK(claimed.size() == 1);
        claim.commit();
    }
    const std::string claimedSnapshot = Scalar(
        connection,
        "SELECT e.status||':'||e.phase||':'||"
        "e.active_scheduler_worker_attempt_id::text||':'||"
        "replacement.lifecycle_state||':'||old.lifecycle_state||':'||"
        "(SELECT count(*)::text FROM "
        "experiment_scheduler_worker_attempt capacity "
        "WHERE capacity.experiment_id=e.experiment_id "
        "AND capacity.capacity_class='train' "
        "AND capacity.lifecycle_state IN "
        "('reserved','spawned','running','observed',"
        "'identity_ambiguous')) "
        "FROM experiment e "
        "JOIN experiment_scheduler_worker_attempt replacement "
        "ON replacement.worker_attempt_id="
        "e.active_scheduler_worker_attempt_id "
        "JOIN experiment_scheduler_worker_attempt old "
        "ON old.worker_attempt_id=" +
            std::to_string(ordinaryAttemptId) +
            " WHERE e.experiment_id=700085");
    CHECK(claimedSnapshot ==
          "running:infer:" + std::to_string(replacementAttemptId) +
              ":reserved:completed:0");

    {
        pqxx::work replay{connection};
        const CheckpointStopRecordResult result =
            RecordCheckpointStopReached(
                replay,
                700085,
                ordinaryAttemptId,
                20,
                ordinaryModelId);
        CHECK(result.recorded);
        CHECK(result.detail == "checkpoint_stop_replay");
        replay.commit();
    }
    {
        pqxx::work wrong{connection};
        const CheckpointStopRecordResult result =
            RecordCheckpointStopReached(
                wrong,
                700085,
                replacementAttemptId,
                20,
                ordinaryModelId);
        CHECK(!result.recorded);
        CHECK(result.detail == "exact_active_train_attempt_mismatch");
        wrong.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT e.status||':'||e.phase||':'||"
              "e.active_scheduler_worker_attempt_id::text||':'||"
              "a.lifecycle_state FROM experiment e "
              "JOIN experiment_scheduler_worker_attempt a "
              "ON a.worker_attempt_id="
              "e.active_scheduler_worker_attempt_id "
              "WHERE e.experiment_id=700085") ==
          "running:infer:" + std::to_string(replacementAttemptId) +
              ":reserved");
    {
        pqxx::connection restarted{connection.connection_string()};
        pqxx::work replayAfterRestart{restarted};
        const CheckpointStopRecordResult result =
            RecordCheckpointStopReached(
                replayAfterRestart,
                700085,
                ordinaryAttemptId,
                20,
                ordinaryModelId);
        CHECK(result.recorded);
        CHECK(result.detail == "checkpoint_stop_replay");
        replayAfterRestart.commit();
    }
    CHECK(Scalar(
              connection,
              "SELECT count(*)::text FROM "
              "experiment_scheduler_worker_attempt "
              "WHERE experiment_id=700085") == "2");
    ResetCrashFixtures(connection);
}

int RunDatabaseLegacyControlTests(const std::string& selfPath,
                                  const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    ResetCrashFixtures(connection);
    std::cout << "RUN TestCancellationAfterSignalBeforeAccounting"
              << std::endl;
    TestCancellationAfterSignalBeforeAccounting(
        selfPath, connectionString, connection);
    std::cout << "RUN TestAccountedBeforeReconciliation" << std::endl;
    TestAccountedBeforeReconciliation(connectionString, connection);
    std::cout << "RUN TestTerminalCompletedCheckpointReconciliation" << std::endl;
    TestTerminalCompletedCheckpointReconciliation(
        connectionString, connection);
    std::cout << "RUN TestUnresolvedCheckpointRemainsActive" << std::endl;
    TestUnresolvedCheckpointRemainsActive(connectionString, connection);
    std::cout << "RUN TestImmediateAndCurrentBoundaryInferenceIdentity"
              << std::endl;
    TestImmediateAndCurrentBoundaryInferenceIdentity(
        connectionString, connection);
    std::cout << "RUN TestTerminalInferenceReconciliation" << std::endl;
    TestTerminalInferenceReconciliation(connectionString, connection);
    std::cout << "RUN TestLegacyInferenceUpgradeMatrix" << std::endl;
    TestLegacyInferenceUpgradeMatrix(connectionString, connection);
    std::cout << "RUN TestProductionCheckpointStopOwnership" << std::endl;
    TestProductionCheckpointStopOwnership(connection);
    std::cout << "GlobalExperimentControlLegacyControlTests passed\n";
    return 0;
}

int RunDatabasePriorityControlTests(const std::string& selfPath,
                                    const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    ResetCrashFixtures(connection);
    std::cout << "RUN TestSelectiveResumeFromGlobalPause" << std::endl;
    TestSelectiveResumeFromGlobalPause(
        selfPath, connectionString, connection);
    std::cout << "RUN TestConcurrentSelectiveResumeIsIdempotent" << std::endl;
    TestConcurrentSelectiveResumeIsIdempotent(
        selfPath, connectionString, connection);
    std::cout << "RUN TestSchedulerCancellationOwnerClaims" << std::endl;
    TestSchedulerCancellationOwnerClaims(
        selfPath, connectionString, connection);
    std::cout << "RUN TestSelectiveReplayAfterLifecycleTransition" << std::endl;
    TestSelectiveReplayAfterLifecycleTransition(
        selfPath, connectionString, connection);
    std::cout << "RUN TestPrimarySelectiveResumeAndCheckpointChildDeparture"
              << std::endl;
    TestPrimarySelectiveResumeAndCheckpointChildDeparture(
        selfPath, connectionString, connection);
    std::cout
        << "RUN TestCancellationWithSelectivelyReleasedPrimaryAndStoppedChild"
        << std::endl;
    TestCancellationWithSelectivelyReleasedPrimaryAndStoppedChild(
        selfPath, connectionString, connection);
    std::cout
        << "RUN TestCancellationInspectionFailuresRemainRecoverable"
        << std::endl;
    TestCancellationInspectionFailuresRemainRecoverable(
        selfPath, connectionString, connection);
    std::cout << "GlobalExperimentControlPriorityControlTests passed\n";
    return 0;
}

int RunRealProcessTests(const std::string& selfPath)
{
    const std::string executableMarker = ExecutableMarker(selfPath);
    CHECK(!EmergencyRegistry().AddLauncher(0, executableMarker));
    CHECK(!EmergencyRegistry().AddLauncher(1, executableMarker));
    CHECK(!EmergencyRegistry().AddLauncher(::getpid(), executableMarker));
    CHECK(!EmergencyRegistry().AddPendingWorker(
        ::getpgrp(), executableMarker));
    CHECK(!EmergencyRegistry().PromoteWorkerGroup(
        ::getpid(), ::getpgrp(), "--managed-test-worker"));
    CHECK(EmergencyRegistry().Size() == 0);
    CHECK(EmergencyRegistry().Cleanup());

    std::unique_ptr<ProcessOperations> processes =
        CreateNativeProcessOperations();

    ManagedWorker pauseResume =
        SpawnWorker(selfPath, *processes);
    SignalOutcome pause = PauseWorker(pauseResume, *processes);
    CHECK(pause.success &&
          pause.signals == std::vector<int>{SIGSTOP});
    CHECK(WaitUntil(
        [&] { return processes->Observe(pauseResume.pid).stopped; }));
    SignalOutcome resume = ResumeWorker(pauseResume, *processes);
    CHECK(resume.success &&
          resume.signals == std::vector<int>{SIGCONT});
    CHECK(WaitUntil([&] {
        const ProcessObservation observation =
            processes->Observe(pauseResume.pid);
        return observation.exists && observation.inspectionSucceeded &&
               !observation.stopped;
    }));
    SignalOutcome pauseResumeCleanup = CancelWorker(
        pauseResume, false, std::chrono::milliseconds(1000), *processes);
    CHECK(pauseResumeCleanup.success);
    AssertGroupExited(pauseResume);

    ManagedWorker graceful =
        SpawnWorker(selfPath, *processes);
    SignalOutcome gracefulTerm = CancelWorker(
        graceful, false, std::chrono::milliseconds(1000), *processes);
    CHECK(gracefulTerm.success);
    CHECK(gracefulTerm.result == "signaled");
    CHECK(gracefulTerm.signals == std::vector<int>{SIGTERM});
    AssertGroupExited(graceful);

    ManagedWorker killEscalation =
        SpawnWorker(selfPath, *processes, 42, true);
    SignalOutcome killed = CancelWorker(
        killEscalation, false, std::chrono::milliseconds(100), *processes);
    CHECK(killed.success);
    CHECK(killed.result == "escalated");
    CHECK((killed.signals == std::vector<int>{SIGTERM, SIGKILL}));
    AssertGroupExited(killEscalation);

    ManagedWorker grouped =
        SpawnWorker(selfPath, *processes, 42, true, true);
    CHECK(ProcessGroupMemberCount(*grouped.processGroupId) >= 2);
    SignalOutcome groupedKill = CancelWorker(
        grouped, false, std::chrono::milliseconds(100), *processes);
    CHECK(groupedKill.success);
    CHECK(groupedKill.result == "escalated");
    AssertGroupExited(grouped);

    ManagedWorker reusedPid =
        SpawnWorker(selfPath, *processes);
    const std::string actualStartIdentity =
        *reusedPid.processStartIdentity;
    reusedPid.processStartIdentity = actualStartIdentity + "-different";
    SignalOutcome rejected = PauseWorker(reusedPid, *processes);
    CHECK(!rejected.success);
    CHECK(rejected.identity == IdentityResult::IdentityValidationFailed);
    CHECK(rejected.signals.empty());
    CHECK(processes->Observe(reusedPid.pid).exists);
    reusedPid.processStartIdentity = actualStartIdentity;
    SignalOutcome rejectedCleanup = CancelWorker(
        reusedPid, false, std::chrono::milliseconds(1000), *processes);
    CHECK(rejectedCleanup.success);
    AssertGroupExited(reusedPid);

    // Production-reachable stale worker-row fixtures. Each mismatch models a
    // replacement process occupying a recorded PID while one independently
    // persisted identity component still describes the departed worker.
    ManagedWorker executableReplacement =
        SpawnWorker(selfPath, *processes, 43);
    ManagedWorker staleExecutable = executableReplacement;
    staleExecutable.executable = "/tmp/replacement-LSTM_Release";
    SignalOutcome executableRejected =
        PauseWorker(staleExecutable, *processes);
    CHECK(!executableRejected.success);
    CHECK(executableRejected.identity ==
          IdentityResult::IdentityValidationFailed);
    CHECK(executableRejected.signals.empty());
    CleanupWorker(executableReplacement, *processes);

    ManagedWorker commandReplacement =
        SpawnWorker(selfPath, *processes, 44);
    ManagedWorker staleCommand = commandReplacement;
    staleCommand.commandLine =
        *commandReplacement.commandLine + " --replacement-process";
    SignalOutcome commandRejected = PauseWorker(staleCommand, *processes);
    CHECK(!commandRejected.success);
    CHECK(commandRejected.identity ==
          IdentityResult::IdentityValidationFailed);
    CHECK(commandRejected.signals.empty());
    CleanupWorker(commandReplacement, *processes);

    ManagedWorker groupReplacement =
        SpawnWorker(selfPath, *processes, 45);
    ManagedWorker staleGroup = groupReplacement;
    staleGroup.processGroupId = *groupReplacement.processGroupId + 100000;
    SignalOutcome groupRejected = PauseWorker(staleGroup, *processes);
    CHECK(!groupRejected.success);
    CHECK(groupRejected.identity ==
          IdentityResult::IdentityValidationFailed);
    CHECK(groupRejected.signals.empty());
    CleanupWorker(groupReplacement, *processes);

    ManagedWorker exitedWorker =
        SpawnWorker(selfPath, *processes, 46);
    CHECK(CancelWorker(
              exitedWorker,
              false,
              std::chrono::milliseconds(1000),
              *processes).success);
    AssertGroupExited(exitedWorker);
    SignalOutcome departedRejected = PauseWorker(exitedWorker, *processes);
    CHECK(!departedRejected.success);
    CHECK(departedRejected.identity == IdentityResult::ProcessMissing);
    CHECK(departedRejected.signals.empty());

    return 0;
}

[[noreturn]] void RunEmergencyCleanupSelfTest(const std::string& selfPath)
{
    const pid_t partialLauncher = ::fork();
    CHECK(partialLauncher >= 0);
    if (partialLauncher == 0)
    {
        while (true)
            ::pause();
    }
    CHECK(EmergencyRegistry().AddLauncher(
        partialLauncher, ExecutableMarker(selfPath)));

    std::unique_ptr<ProcessOperations> processes =
        CreateNativeProcessOperations();
    (void)SpawnWorker(selfPath, *processes, 700099, true, true);
    (void)std::freopen("/dev/null", "w", stderr);
    FailCheck("intentional_emergency_cleanup_self_test", __FILE__, __LINE__);
}

[[noreturn]] void RunSpawnFailureSelfTest(
    const std::string& selfPath,
    SpawnFailureMode failureMode)
{
    std::unique_ptr<ProcessOperations> processes =
        CreateNativeProcessOperations();
    (void)std::freopen("/dev/null", "w", stderr);
    (void)SpawnWorker(
        selfPath,
        *processes,
        700098,
        true,
        failureMode == SpawnFailureMode::FinalReadinessTimeout,
        failureMode);
    FailCheck("spawn_failure_injection_did_not_fail", __FILE__, __LINE__);
}

void TestSpawnFailurePaths(const std::string& selfPath)
{
    const std::vector<std::string> failureOptions{
        "--launcher-registration-failure-self-test",
        "--worker-registration-failure-self-test",
        "--pid-publication-failure-self-test",
        "--readiness-timeout-self-test"};
    for (const std::string& failureOption : failureOptions)
    {
        const pid_t testPid = ::fork();
        CHECK(testPid >= 0);
        if (testPid == 0)
        {
            ::execl(
                selfPath.c_str(),
                selfPath.c_str(),
                failureOption.c_str(),
                static_cast<char*>(nullptr));
            _exit(127);
        }
        int status = 0;
        CHECK(::waitpid(testPid, &status, 0) == testPid);
        CHECK(WIFEXITED(status));
        CHECK(WEXITSTATUS(status) == 1);
    }
}

void TestEmergencyCleanupPath(const std::string& selfPath)
{
    const pid_t testPid = ::fork();
    CHECK(testPid >= 0);
    if (testPid == 0)
    {
        ::execl(
            selfPath.c_str(),
            selfPath.c_str(),
            "--emergency-cleanup-self-test",
            static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    CHECK(::waitpid(testPid, &status, 0) == testPid);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 1);
}

} // namespace

int main(int argc, char* argv[])
{
    if (::signal(SIGPIPE, SIG_IGN) == SIG_ERR)
        return 3;
    if (HasArgument(argc, argv, "--managed-test-worker"))
        RunManagedTestWorker(argc, argv);
    if (const std::optional<int> inspectedPid =
            IntegerOption(argc, argv, "--inspect-managed-test-process="))
        return PrintManagedTestProcessIdentity(*inspectedPid);

    // Construct the registry before registering the callback. C++ registers
    // the function-local static destructor at construction time, so reverse
    // atexit order runs cleanup while the registry is still alive.
    (void)EmergencyRegistry();
    CHECK(std::atexit(EmergencyCleanupAtExit) == 0);
    const std::string selfPath = CanonicalSelfPath(argv[0]);
    if (HasArgument(argc, argv, "--emergency-cleanup-self-test"))
        RunEmergencyCleanupSelfTest(selfPath);
    if (HasArgument(
            argc, argv, "--launcher-registration-failure-self-test"))
        RunSpawnFailureSelfTest(
            selfPath, SpawnFailureMode::LauncherRegistration);
    if (HasArgument(
            argc, argv, "--worker-registration-failure-self-test"))
        RunSpawnFailureSelfTest(
            selfPath, SpawnFailureMode::WorkerRegistration);
    if (HasArgument(argc, argv, "--pid-publication-failure-self-test"))
        RunSpawnFailureSelfTest(
            selfPath, SpawnFailureMode::PidPublication);
    if (HasArgument(argc, argv, "--readiness-timeout-self-test"))
        RunSpawnFailureSelfTest(
            selfPath, SpawnFailureMode::FinalReadinessTimeout);
    if (argc == 3 &&
        std::string{argv[1]} == "--database-legacy-control-tests")
        return RunDatabaseLegacyControlTests(selfPath, argv[2]);
    if (argc == 3 &&
        std::string{argv[1]} == "--database-priority-control-tests")
        return RunDatabasePriorityControlTests(selfPath, argv[2]);
    const int result = RunRealProcessTests(selfPath);
    TestSpawnFailurePaths(selfPath);
    TestEmergencyCleanupPath(selfPath);
    std::cout << "GlobalExperimentControlProcessTests passed\n";
    return result;
}
