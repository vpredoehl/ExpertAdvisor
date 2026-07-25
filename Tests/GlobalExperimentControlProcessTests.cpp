#include "../Sources/GlobalExperimentControl.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <libproc.h>
#include <memory>
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
                              SpawnFailureMode::None)
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
                "--train",
                "--scheduler-experiment-id=" + std::to_string(experimentId),
                "--ready-fd=" + std::to_string(readyPipe[1]),
                "--group-release-fd=" +
                    std::to_string(groupReleasePipe[0])};
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
    worker.phase = "train";
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

std::string Scalar(pqxx::connection& connection, const std::string& query)
{
    pqxx::work transaction{connection};
    const pqxx::row row = transaction.exec(query).one_row();
    const std::string value =
        row[0].is_null() ? std::string{} : row[0].as<std::string>();
    transaction.commit();
    return value;
}

void ResetCrashFixtures(pqxx::connection& connection)
{
    pqxx::work transaction{connection};
    transaction.exec(
        "UPDATE experiment_global_control SET desired_state='running',"
        "active_request_id=NULL WHERE singleton;");
    transaction.exec(
        "DELETE FROM experiment_checkpoint_eval "
        "WHERE experiment_id BETWEEN 700000 AND 700099 "
        "OR parent_experiment_id BETWEEN 700000 AND 700099;");
    transaction.exec(
        "DELETE FROM experiment_admin_worker_outcome "
        "WHERE experiment_id BETWEEN 700000 AND 700099;");
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
        "WHERE invocation_identity LIKE 'crash-window-%';");
    transaction.commit();
}

void InsertRunningExperiment(pqxx::connection& connection,
                             const ManagedWorker& worker,
                             const std::string& controlState = "running")
{
    pqxx::work transaction{connection};
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
    transaction.exec_params(
        "UPDATE experiment_global_control SET desired_state=$1,"
        "active_request_id=$2,revision=revision+1,updated_at=now() "
        "WHERE singleton;",
        resultingState,
        requestId);
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
        "worker_process_start_identity,cancellation_checkpoint_epoch,"
        "inference_action,outcome_status,detail) VALUES ("
        "$1,$2,$3,NULL,'experiment',$4,'running',$5,$6,$7,NULL,"
        "'none',$8,$9);",
        requestId,
        "experiment:" + std::to_string(worker.experimentId),
        worker.experimentId,
        worker.phase,
        worker.pid,
        *worker.processGroupId,
        *worker.processStartIdentity,
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

void TestPlanCommittedBeforeSignal(const std::string& selfPath,
                                   const std::string& connectionString,
                                   pqxx::connection& connection)
{
    RecordingNativeProcesses processes;
    const ManagedWorker worker =
        SpawnWorker(selfPath, processes, 700001);
    const long long requestId =
        SeedRequest(connection, Action::PauseAll, worker, "planned");

    std::string replayOutput;
    CHECK(Replay(
              connectionString,
              ReplayCommand(Action::PauseAll, "crash-window-replay-plan"),
              processes,
              replayOutput) == 0);
    CHECK(replayOutput.find(
              "GLOBAL_EXPERIMENT_CONTROL_RETRY,request_id=" +
              std::to_string(requestId)) != std::string::npos);
    CHECK((processes.signals ==
           std::vector<std::pair<int, int>>{
               {*worker.processGroupId, SIGSTOP}}));
    CHECK(WaitUntil(
        [&] { return processes.Observe(worker.pid).stopped; }));
    CheckStableIdentities(connection, requestId, worker.experimentId);
    CHECK(Scalar(
              connection,
              "SELECT requested_signal FROM "
              "experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId)) == std::to_string(SIGSTOP));
    CHECK(Scalar(
              connection,
              "SELECT signal_result||':'||outcome_status FROM "
              "experiment_admin_worker_outcome WHERE request_id=" +
                  std::to_string(requestId)) == "signaled:completed");
    CHECK(Scalar(
              connection,
              "SELECT status FROM experiment_admin_request WHERE request_id=" +
                  std::to_string(requestId)) == "completed");
    CHECK(Scalar(
              connection,
              "SELECT active_request_id IS NULL FROM "
              "experiment_global_control WHERE singleton") == "t");

    CleanupWorker(worker, processes);
    ResetCrashFixtures(connection);
}

void TestAfterSignalBeforeAccounting(const std::string& selfPath,
                                     const std::string& connectionString,
                                     pqxx::connection& connection)
{
    {
        RecordingNativeProcesses processes;
        const ManagedWorker worker =
            SpawnWorker(selfPath, processes, 700002);
        const long long requestId =
            SeedRequest(connection, Action::PauseAll, worker, "planned");
        CHECK(PauseWorker(worker, processes).success);
        CHECK((processes.signals ==
               std::vector<std::pair<int, int>>{
                   {*worker.processGroupId, SIGSTOP}}));
        CHECK(WaitUntil(
            [&] { return processes.Observe(worker.pid).stopped; }));
        processes.signals.clear();

        std::string replayOutput;
        CHECK(Replay(
                  connectionString,
                  ReplayCommand(
                      Action::PauseAll, "crash-window-replay-paused"),
                  processes,
                  replayOutput) == 0);
        CHECK(processes.signals.empty());
        CheckStableIdentities(connection, requestId, worker.experimentId);
        CHECK(Scalar(
                  connection,
                  "SELECT signal_result||':'||outcome_status FROM "
                  "experiment_admin_worker_outcome WHERE request_id=" +
                      std::to_string(requestId)) ==
              "already_requested_state:completed");
        CHECK(Scalar(
                  connection,
                  "SELECT requested_signal IS NULL FROM "
                  "experiment_admin_worker_outcome WHERE request_id=" +
                      std::to_string(requestId)) == "t");
        CHECK(Scalar(
                  connection,
                  "SELECT worker_control_state FROM experiment WHERE "
                  "experiment_id=700002") == "paused");
        CleanupWorker(worker, processes);
        ResetCrashFixtures(connection);
    }

    {
        RecordingNativeProcesses processes;
        const ManagedWorker worker =
            SpawnWorker(selfPath, processes, 700003);
        CHECK(PauseWorker(worker, processes).success);
        CHECK(WaitUntil(
            [&] { return processes.Observe(worker.pid).stopped; }));
        const long long requestId = SeedRequest(
            connection, Action::ResumeAll, worker, "planned", "paused");
        CHECK(ResumeWorker(worker, processes).success);
        CHECK((processes.signals ==
               std::vector<std::pair<int, int>>{
                   {*worker.processGroupId, SIGSTOP},
                   {*worker.processGroupId, SIGCONT}}));
        CHECK(WaitUntil([&] {
            const ProcessObservation observation =
                processes.Observe(worker.pid);
            return observation.exists && observation.inspectionSucceeded &&
                   !observation.stopped;
        }));
        processes.signals.clear();

        std::string replayOutput;
        CHECK(Replay(
                  connectionString,
                  ReplayCommand(
                      Action::ResumeAll, "crash-window-replay-running"),
                  processes,
                  replayOutput) == 0);
        CHECK(processes.signals.empty());
        CheckStableIdentities(connection, requestId, worker.experimentId);
        CHECK(Scalar(
                  connection,
                  "SELECT signal_result||':'||outcome_status FROM "
                  "experiment_admin_worker_outcome WHERE request_id=" +
                      std::to_string(requestId)) ==
              "already_requested_state:completed");
        CHECK(Scalar(
                  connection,
                  "SELECT worker_control_state FROM experiment WHERE "
                  "experiment_id=700003") == "running");
        CleanupWorker(worker, processes);
        ResetCrashFixtures(connection);
    }

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
        "current_operation='cancelled_by_global_request',"
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
        "current_operation='cancel_checkpoint_restart_pending',"
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
              ":true:cancel_checkpoint_restart_pending:"
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
          "cancelled_by_global_request:cancelled_by_global_request:"
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
            "current_operation='cancel_checkpoint_reached',"
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
          "20:20:true:cancel_checkpoint_reached:"
          "cancelled_at_requested_checkpoint");
    ResetCrashFixtures(connection);
}

int RunDatabaseCrashWindowTests(const std::string& selfPath,
                                const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    ResetCrashFixtures(connection);
    TestPlanCommittedBeforeSignal(selfPath, connectionString, connection);
    TestAfterSignalBeforeAccounting(selfPath, connectionString, connection);
    TestAccountedBeforeReconciliation(connectionString, connection);
    std::cout << "GlobalExperimentControlCrashWindowTests passed\n";
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
        std::string{argv[1]} == "--database-crash-window-tests")
        return RunDatabaseCrashWindowTests(selfPath, argv[2]);
    const int result = RunRealProcessTests(selfPath);
    TestSpawnFailurePaths(selfPath);
    TestEmergencyCleanupPath(selfPath);
    std::cout << "GlobalExperimentControlProcessTests passed\n";
    return result;
}
