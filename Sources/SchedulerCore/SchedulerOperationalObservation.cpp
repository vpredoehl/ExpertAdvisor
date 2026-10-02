#include "SchedulerOperationalObservation.hpp"
#include "SchedulerOperationalObservationInternal.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libproc.h>
#include <sstream>
#include <stdexcept>
#include <sys/sysctl.h>
#include <unistd.h>

namespace EA::GlobalExperimentControl
{
namespace
{

std::string QuoteShellPid(int pid)
{
    if (pid <= 0)
        throw std::invalid_argument("process id must be positive");
    return std::to_string(pid);
}

bool ContainsExactToken(const std::string& command,
                        const std::string& token)
{
    size_t position = 0;
    while ((position = command.find(token, position)) != std::string::npos)
    {
        const bool start =
            position == 0 ||
            std::isspace(static_cast<unsigned char>(command[position - 1]));
        const size_t end = position + token.size();
        if (start &&
            (end == command.size() ||
             std::isspace(static_cast<unsigned char>(command[end]))))
            return true;
        position += token.size();
    }
    return false;
}

bool ContainsWorkerIdentity(const std::string& command,
                            long long experimentId,
                            const std::string& phase)
{
    using SchedulerObservationDetail::ContainsExactOptionValue;
    if (command.find("--schedule-experiments") != std::string::npos ||
        command.find("--scheduler-status") != std::string::npos)
        return false;
    if (phase == "train")
        return ContainsExactToken(command, "--train") &&
               ContainsExactOptionValue(
                   command, "--scheduler-experiment-id", experimentId);
    if (phase == "infer")
        return ContainsExactToken(command, "--infer") &&
               ContainsExactOptionValue(
                   command, "--scheduler-experiment-id", experimentId);
    if (phase == "analyze")
        return ContainsExactOptionValue(
            command, "--analyze-experiment", experimentId);
    if (phase == "checkpoint_infer")
        return ContainsExactToken(command, "--infer") &&
               command.find("--scheduler-checkpoint-eval-id=") !=
                   std::string::npos;
    return false;
}

std::optional<std::string> CanonicalizeObservedExecutablePath(
    const std::string& executablePath)
{
    if (executablePath.empty() || executablePath.front() != '/')
        return std::nullopt;

    errno = 0;
    char* canonicalExecutable = ::realpath(executablePath.c_str(), nullptr);
    if (canonicalExecutable != nullptr)
    {
        std::string result{canonicalExecutable};
        std::free(canonicalExecutable);
        return result;
    }

    // A live process may outlive its executable's directory entry. Preserve
    // the kernel-recorded absolute path only for that observed-live case.
    if (errno == ENOENT)
        return executablePath;

    return std::nullopt;
}

class PosixNativeProcessObservationBackend final
    : public NativeProcessObservationBackend
{
public:
    bool ProcessExists(int pid, int& errorNumber) override
    {
        errno = 0;
        if (::kill(static_cast<pid_t>(pid), 0) == 0)
        {
            errorNumber = 0;
            return true;
        }
        errorNumber = errno;
        return false;
    }

    std::optional<std::string> ReadStartIdentity(
        int pid,
        int& errorNumber) override
    {
        errno = 0;
        const std::optional<std::string> identity =
            ReadProcessStartIdentity(pid);
        errorNumber = errno;
        return identity;
    }

    std::optional<NativeProcessStatus> ReadStatus(
        int pid,
        int& errorNumber) override
    {
        const std::string command =
            "ps -p " + QuoteShellPid(pid) +
            " -o pid= -o pgid= -o state= -o command=";
        FILE* pipe = ::popen(command.c_str(), "r");
        if (pipe == nullptr)
        {
            errorNumber = errno;
            return std::nullopt;
        }
        char buffer[16384] = {};
        std::string line;
        while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
            line += buffer;
        const int status = ::pclose(pipe);
        if (status != 0 || line.empty())
        {
            errorNumber = EIO;
            return std::nullopt;
        }

        std::istringstream input(line);
        NativeProcessStatus observed;
        if (!(input >> observed.pid >> observed.processGroupId >> observed.state) ||
            observed.pid != pid)
        {
            errorNumber = EIO;
            return std::nullopt;
        }
        std::getline(input, observed.commandLine);
        const size_t begin = observed.commandLine.find_first_not_of(" \t");
        if (begin != std::string::npos)
            observed.commandLine.erase(0, begin);
        errorNumber = 0;
        return observed;
    }

    std::optional<std::string> ReadProcPidPath(
        int pid,
        int& errorNumber) override
    {
        char executablePath[PROC_PIDPATHINFO_MAXSIZE] = {};
        errno = 0;
        const int executableLength = ::proc_pidpath(
            pid, executablePath, sizeof(executablePath));
        if (executableLength <= 0)
        {
            errorNumber = errno;
            return std::nullopt;
        }
        errorNumber = 0;
        return std::string{executablePath};
    }

    std::optional<std::string> ReadKernelExecutablePath(
        int pid,
        int& errorNumber) override
    {
        int mib[] = {CTL_KERN, KERN_PROCARGS2, pid};
        size_t size = 0;
        errno = 0;
        if (::sysctl(mib, 3, nullptr, &size, nullptr, 0) != 0 ||
            size <= sizeof(int))
        {
            errorNumber = errno != 0 ? errno : EIO;
            return std::nullopt;
        }
        std::vector<char> buffer(size, '\0');
        errno = 0;
        if (::sysctl(mib, 3, buffer.data(), &size, nullptr, 0) != 0 ||
            size <= sizeof(int))
        {
            errorNumber = errno != 0 ? errno : EIO;
            return std::nullopt;
        }
        const char* const executable = buffer.data() + sizeof(int);
        const size_t available = size - sizeof(int);
        const void* const terminator =
            std::memchr(executable, '\0', available);
        if (terminator == nullptr || executable == terminator)
        {
            errorNumber = EIO;
            return std::nullopt;
        }
        errorNumber = 0;
        return std::string{
            executable,
            static_cast<size_t>(
                static_cast<const char*>(terminator) - executable)};
    }
};

class NativeProcessObserver final : public ProcessObserver
{
public:
    explicit NativeProcessObserver(
        std::unique_ptr<NativeProcessObservationBackend> backend)
        : backend_(std::move(backend))
    {
        if (!backend_)
            throw std::invalid_argument(
                "native_process_observation_backend_required");
    }

    ProcessObservation Observe(int pid) override
    {
        ProcessObservation observation;
        observation.pid = pid;
        if (pid <= 0)
            return observation;

        int errorNumber = 0;
        if (!backend_->ProcessExists(pid, errorNumber))
        {
            if (errorNumber == ESRCH)
            {
                observation.inspectionSucceeded = true;
                return observation;
            }
            if (errorNumber == EPERM)
            {
                observation.exists = true;
                observation.permissionDenied = true;
                return observation;
            }
            return observation;
        }
        observation.exists = true;
        const std::optional<std::string> startIdentityBefore =
            backend_->ReadStartIdentity(pid, errorNumber);
        if (!startIdentityBefore)
        {
            if (errorNumber == EPERM)
                observation.permissionDenied = true;
            return observation;
        }

        const std::optional<NativeProcessStatus> status =
            backend_->ReadStatus(pid, errorNumber);
        if (!status || status->pid != pid)
            return observation;
        if (!status->state.empty() && status->state[0] == 'Z')
        {
            observation.exists = false;
            observation.inspectionSucceeded = true;
            observation.processGroupId = status->processGroupId;
            return observation;
        }
        observation.processGroupId = status->processGroupId;
        observation.stopped =
            !status->state.empty() &&
            (status->state[0] == 'T' || status->state[0] == 't');
        observation.commandLine = status->commandLine;

        std::optional<std::string> executablePath =
            backend_->ReadProcPidPath(pid, errorNumber);
        if (!executablePath)
        {
            if (errorNumber != ENOENT)
                return observation;
            executablePath =
                backend_->ReadKernelExecutablePath(pid, errorNumber);
            if (!executablePath)
                return observation;
        }
        const std::optional<std::string> canonicalExecutable =
            CanonicalizeObservedExecutablePath(*executablePath);
        if (!canonicalExecutable)
            return observation;
        observation.executable = *canonicalExecutable;
        const std::optional<std::string> startIdentityAfter =
            backend_->ReadStartIdentity(pid, errorNumber);
        if (!startIdentityAfter ||
            *startIdentityAfter != *startIdentityBefore)
            return observation;
        observation.processStartIdentity = *startIdentityAfter;
        observation.inspectionSucceeded = true;
        return observation;
    }

    int CallerPid() const override { return static_cast<int>(::getpid()); }
    int CallerProcessGroupId() const override
    {
        return static_cast<int>(::getpgrp());
    }

private:
    std::unique_ptr<NativeProcessObservationBackend> backend_;
};

enum class ManagedWorkerLifecyclePrecondition
{
    Active,
    SchedulerStoppedAdmission,
    PausedStopReconciliation,
    ResumedStoppedCancellation
};

ValidatedWorker ValidateManagedWorkerWithLifecyclePrecondition(
    const ManagedWorker& worker,
    ProcessObserver& processes,
    ManagedWorkerLifecyclePrecondition precondition)
{
    using SchedulerObservationDetail::ContainsExactOptionValue;
    ValidatedWorker result;
    result.worker = worker;
    const bool activeLifecycle = worker.lifecycleStatus == "running";
    const bool exactStoppedAdmissionLifecycle =
        worker.lifecycleStatus == "pending" &&
        worker.attemptLifecycleState == "stopped" &&
        worker.workerAttemptId.has_value() &&
        worker.workerKind == "experiment" &&
        worker.capacityClass == worker.phase &&
        !worker.launchAttemptIdentity.empty();
    const bool exactPausedStopLifecycle =
        worker.lifecycleStatus == "paused" &&
        worker.attemptLifecycleState == "stopped" &&
        worker.workerAttemptId.has_value() &&
        worker.workerKind == "experiment" &&
        worker.capacityClass == worker.phase &&
        !worker.launchAttemptIdentity.empty();
    bool lifecycleAllowed = activeLifecycle;
    if (precondition ==
        ManagedWorkerLifecyclePrecondition::SchedulerStoppedAdmission)
        lifecycleAllowed = exactStoppedAdmissionLifecycle;
    else if (precondition ==
             ManagedWorkerLifecyclePrecondition::PausedStopReconciliation)
        lifecycleAllowed = exactPausedStopLifecycle;
    else if (precondition ==
             ManagedWorkerLifecyclePrecondition::ResumedStoppedCancellation)
        lifecycleAllowed =
            exactStoppedAdmissionLifecycle || exactPausedStopLifecycle;
    if (!lifecycleAllowed || worker.pid <= 0)
    {
        result.identity = IdentityResult::StalePid;
        result.detail =
            precondition == ManagedWorkerLifecyclePrecondition::Active
                ? "database_row_is_not_an_active_managed_worker"
                : (precondition ==
                           ManagedWorkerLifecyclePrecondition::
                               ResumedStoppedCancellation
                       ? "database_row_is_not_a_resumed_authoritative_stopped_worker"
                       : "database_row_is_not_an_authoritative_stopped_worker");
        return result;
    }
    if (!worker.authoritativeBindingMatches)
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "lifecycle_and_worker_attempt_identity_mismatch";
        return result;
    }
    result.observation = processes.Observe(worker.pid);
    if (!result.observation.exists)
    {
        result.identity = result.observation.inspectionSucceeded
            ? IdentityResult::ProcessMissing
            : IdentityResult::InspectionFailed;
        result.detail = result.observation.inspectionSucceeded
            ? "pid_does_not_exist"
            : "process_inspection_failed";
        return result;
    }
    if (result.observation.permissionDenied)
    {
        result.identity = IdentityResult::PermissionDenied;
        result.detail = "process_inspection_permission_denied";
        return result;
    }
    if (!result.observation.inspectionSucceeded)
    {
        result.identity = IdentityResult::InspectionFailed;
        result.detail = "process_inspection_failed";
        return result;
    }
    if (!worker.processGroupId || !worker.executable ||
        worker.executable->empty() || !worker.commandLine ||
        worker.commandLine->empty() || !worker.processStartIdentity ||
        worker.processStartIdentity->empty())
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "persisted_worker_identity_incomplete";
        return result;
    }
    if (result.observation.pid != worker.pid ||
        !ContainsWorkerIdentity(result.observation.commandLine,
                                worker.experimentId,
                                worker.phase))
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "pid_command_does_not_match_experiment_and_phase";
        return result;
    }
    if (worker.phase == "checkpoint_infer" &&
        (!worker.checkpointEvalId ||
         !ContainsExactOptionValue(
             result.observation.commandLine,
             "--scheduler-checkpoint-eval-id",
             *worker.checkpointEvalId)))
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "checkpoint_eval_identity_mismatch";
        return result;
    }
    if (worker.workerAttemptId &&
        worker.ownershipOrigin != "legacy_unverified" &&
        !ContainsExactOptionValue(
            result.observation.commandLine,
            "--scheduler-worker-attempt-id",
            *worker.workerAttemptId))
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "worker_attempt_identity_mismatch";
        return result;
    }
    if (*worker.executable != result.observation.executable)
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "executable_identity_mismatch";
        return result;
    }
    if (*worker.commandLine != result.observation.commandLine)
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "command_line_identity_mismatch";
        return result;
    }
    if (*worker.processGroupId != result.observation.processGroupId)
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "process_group_identity_mismatch";
        return result;
    }
    if (*worker.processStartIdentity !=
        result.observation.processStartIdentity)
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "process_start_identity_mismatch";
        return result;
    }
    if (result.observation.processGroupId <= 1 ||
        result.observation.processGroupId == processes.CallerProcessGroupId() ||
        result.observation.processGroupId == processes.CallerPid() ||
        result.observation.commandLine.find("--schedule-experiments") !=
            std::string::npos)
    {
        result.identity = IdentityResult::UnsafeProcessGroup;
        result.detail = "unsafe_or_scheduler_process_group";
        return result;
    }
    const bool requiresStoppedObservation =
        precondition ==
            ManagedWorkerLifecyclePrecondition::SchedulerStoppedAdmission ||
        precondition ==
            ManagedWorkerLifecyclePrecondition::PausedStopReconciliation;
    if (requiresStoppedObservation && !result.observation.stopped)
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "authoritative_stopped_worker_is_not_stopped";
        return result;
    }
    result.identity = IdentityResult::Validated;
    result.detail = "validated";
    return result;
}

ProcessExecutionState ExecutionStateFor(
    const ProcessObservation& observation,
    bool candidateStopped = false)
{
    if (observation.inspectionSucceeded)
    {
        if (!observation.exists)
            return ProcessExecutionState::Missing;
        return observation.stopped
            ? ProcessExecutionState::Stopped
            : ProcessExecutionState::Running;
    }
    if (observation.exists)
        return candidateStopped
            ? ProcessExecutionState::Stopped
            : ProcessExecutionState::Unknown;
    return ProcessExecutionState::Unknown;
}

ValidatedWorker ValidateWorkerForStatusClassification(
    const ManagedWorker& worker,
    ProcessObserver& processes)
{
    if (worker.lifecycleStatus == "paused")
        return ValidatePausedManagedWorker(worker, processes);
    return ValidateManagedWorker(worker, processes);
}

void PopulateAuthoritativeClassification(
    SchedulerWorkerClassification& classification,
    const ManagedWorker& worker,
    ProcessObserver& processes,
    bool candidateStopped)
{
    classification.authoritative = true;
    classification.experimentId = worker.experimentId;
    classification.checkpointEvalId = worker.checkpointEvalId;
    classification.lifecycleStatus = worker.lifecycleStatus;
    classification.attemptLifecycleState = worker.attemptLifecycleState;
    classification.expectedExecutable = worker.executable;
    const ValidatedWorker validated =
        ValidateWorkerForStatusClassification(worker, processes);
    classification.identity = validated.identity;
    classification.executionState =
        ExecutionStateFor(validated.observation, candidateStopped);
    if (!validated.observation.executable.empty())
        classification.observedExecutable = validated.observation.executable;
    if (classification.expectedExecutable &&
        classification.observedExecutable)
    {
        classification.executableIdentityMatch =
            *classification.expectedExecutable ==
            *classification.observedExecutable;
    }
    classification.managed =
        validated.identity == IdentityResult::Validated;
    classification.reason = classification.managed
        ? "validated"
        : validated.detail;
}

} // namespace

namespace SchedulerObservationDetail
{

bool ContainsExactOptionValue(const std::string& command,
                              const std::string& option,
                              long long value)
{
    const std::string expected = std::to_string(value);
    size_t position = 0;
    while ((position = command.find(option, position)) != std::string::npos)
    {
        const bool tokenStart =
            position == 0 ||
            std::isspace(static_cast<unsigned char>(command[position - 1]));
        size_t valueStart = position + option.size();
        if (tokenStart && valueStart < command.size() &&
            command[valueStart] == '=')
            ++valueStart;
        else if (tokenStart && valueStart < command.size() &&
                 std::isspace(
                     static_cast<unsigned char>(command[valueStart])))
        {
            while (valueStart < command.size() &&
                   std::isspace(
                       static_cast<unsigned char>(command[valueStart])))
                ++valueStart;
        }
        else
        {
            position += option.size();
            continue;
        }
        const size_t valueEnd = valueStart + expected.size();
        if (command.compare(valueStart, expected.size(), expected) == 0 &&
            (valueEnd == command.size() ||
             std::isspace(static_cast<unsigned char>(command[valueEnd]))))
            return true;
        position += option.size();
    }
    return false;
}

ValidatedWorker ValidateResumedStoppedWorkerForCancellation(
    const ManagedWorker& worker,
    ProcessObserver& processes)
{
    return ValidateManagedWorkerWithLifecyclePrecondition(
        worker,
        processes,
        ManagedWorkerLifecyclePrecondition::ResumedStoppedCancellation);
}

} // namespace SchedulerObservationDetail

const char* ToString(IdentityResult result)
{
    switch (result)
    {
        case IdentityResult::Validated: return "validated";
        case IdentityResult::ProcessMissing: return "process_missing";
        case IdentityResult::StalePid: return "stale_pid";
        case IdentityResult::IdentityValidationFailed:
            return "identity_validation_failed";
        case IdentityResult::UnsafeProcessGroup: return "unsafe_process_group";
        case IdentityResult::PermissionDenied: return "permission_denied";
        case IdentityResult::InspectionFailed: return "inspection_failed";
    }
    return "inspection_failed";
}

const char* ToString(ProcessExecutionState state)
{
    switch (state)
    {
        case ProcessExecutionState::Unknown: return "unknown";
        case ProcessExecutionState::Running: return "running";
        case ProcessExecutionState::Stopped: return "stopped";
        case ProcessExecutionState::Missing: return "missing";
    }
    return "unknown";
}

std::optional<std::string> ReadProcessStartIdentity(int pid)
{
    if (pid <= 0)
    {
        errno = EINVAL;
        return std::nullopt;
    }
    proc_bsdinfo info{};
    errno = 0;
    const int bytes = ::proc_pidinfo(
        pid, PROC_PIDTBSDINFO, 0, &info, sizeof(info));
    if (bytes != static_cast<int>(sizeof(info)) ||
        info.pbi_pid != static_cast<uint32_t>(pid) ||
        (info.pbi_start_tvsec == 0 && info.pbi_start_tvusec == 0))
        return std::nullopt;
    return std::to_string(info.pbi_start_tvsec) + ":" +
           std::to_string(info.pbi_start_tvusec);
}

std::unique_ptr<ProcessObserver> CreateNativeProcessObserver()
{
    return std::make_unique<NativeProcessObserver>(
        std::make_unique<PosixNativeProcessObservationBackend>());
}

std::unique_ptr<ProcessObserver> CreateNativeProcessObserverForTesting(
    std::unique_ptr<NativeProcessObservationBackend> backend)
{
    return std::make_unique<NativeProcessObserver>(std::move(backend));
}

ValidatedWorker ValidateManagedWorker(const ManagedWorker& worker,
                                      ProcessObserver& processes)
{
    return ValidateManagedWorkerWithLifecyclePrecondition(
        worker,
        processes,
        ManagedWorkerLifecyclePrecondition::Active);
}

ValidatedWorker ValidateStoppedWorkerForSchedulerAdmission(
    const ManagedWorker& worker,
    ProcessObserver& processes)
{
    return ValidateManagedWorkerWithLifecyclePrecondition(
        worker,
        processes,
        ManagedWorkerLifecyclePrecondition::SchedulerStoppedAdmission);
}

ValidatedWorker ValidatePausedManagedWorker(
    const ManagedWorker& worker,
    ProcessObserver& processes)
{
    return ValidateManagedWorkerWithLifecyclePrecondition(
        worker,
        processes,
        ManagedWorkerLifecyclePrecondition::PausedStopReconciliation);
}

std::vector<SchedulerWorkerClassification> ClassifySchedulerWorkers(
    const std::vector<SchedulerWorkerCandidate>& candidates,
    const std::vector<ManagedWorker>& authoritativeWorkers,
    ProcessObserver& processes)
{
    using SchedulerObservationDetail::ContainsExactOptionValue;
    std::vector<SchedulerWorkerClassification> classifications;
    classifications.reserve(candidates.size() + authoritativeWorkers.size());
    std::vector<bool> representedAuthorities(
        authoritativeWorkers.size(), false);

    for (const SchedulerWorkerCandidate& candidate : candidates)
    {
        SchedulerWorkerClassification classification;
        classification.pid = candidate.pid;
        classification.kind = candidate.kind;
        classification.cpuPercent = candidate.cpuPercent;
        classification.memPercent = candidate.memPercent;
        classification.rssMb = candidate.rssMb;
        classification.executionState = candidate.stopped
            ? ProcessExecutionState::Stopped
            : ProcessExecutionState::Running;

        const bool checkpointTagged =
            candidate.commandLine.find("--scheduler-checkpoint-eval-id") !=
            std::string::npos;
        std::vector<size_t> matches;
        for (size_t workerIndex = 0;
             workerIndex < authoritativeWorkers.size();
             ++workerIndex)
        {
            const ManagedWorker& worker =
                authoritativeWorkers[workerIndex];
            const std::string expectedKind =
                worker.phase == "checkpoint_infer" ? "infer" : worker.phase;
            if (worker.pid != candidate.pid ||
                expectedKind != candidate.kind ||
                worker.checkpointEvalId.has_value() != checkpointTagged)
            {
                continue;
            }
            if (checkpointTagged &&
                (!worker.checkpointEvalId ||
                 !ContainsExactOptionValue(
                     candidate.commandLine,
                     "--scheduler-checkpoint-eval-id",
                     *worker.checkpointEvalId)))
            {
                continue;
            }
            matches.push_back(workerIndex);
        }

        if (matches.empty())
        {
            classification.reason = checkpointTagged
                ? "no_matching_active_checkpoint_evaluation"
                : "no_matching_running_experiment";
            classifications.push_back(std::move(classification));
            continue;
        }
        if (matches.size() != 1)
        {
            classification.reason = "ambiguous_authoritative_worker";
            classifications.push_back(std::move(classification));
            continue;
        }

        const size_t workerIndex = matches.front();
        representedAuthorities[workerIndex] = true;
        const ManagedWorker& worker = authoritativeWorkers[workerIndex];
        classification.authoritative = true;
        classification.experimentId = worker.experimentId;
        classification.checkpointEvalId = worker.checkpointEvalId;
        classification.lifecycleStatus = worker.lifecycleStatus;
        classification.attemptLifecycleState =
            worker.attemptLifecycleState;
        classification.expectedExecutable = worker.executable;
        if (!worker.commandLine || worker.commandLine->empty() ||
            !ContainsWorkerIdentity(
                *worker.commandLine, worker.experimentId, worker.phase) ||
            (worker.phase == "checkpoint_infer" &&
             (!worker.checkpointEvalId ||
              !ContainsExactOptionValue(
                  *worker.commandLine,
                  "--scheduler-checkpoint-eval-id",
                  *worker.checkpointEvalId))))
        {
            classification.reason =
                "persisted_worker_command_line_identity_mismatch";
            classifications.push_back(std::move(classification));
            continue;
        }

        PopulateAuthoritativeClassification(
            classification, worker, processes, candidate.stopped);
        classifications.push_back(std::move(classification));
    }

    for (size_t workerIndex = 0;
         workerIndex < authoritativeWorkers.size();
         ++workerIndex)
    {
        if (representedAuthorities[workerIndex])
            continue;
        const ManagedWorker& worker = authoritativeWorkers[workerIndex];
        const bool pidRepresented = std::any_of(
            candidates.begin(), candidates.end(),
            [&](const SchedulerWorkerCandidate& candidate) {
                return candidate.pid == worker.pid;
            });
        if (pidRepresented)
            continue;

        SchedulerWorkerClassification classification;
        classification.pid = worker.pid;
        classification.kind = worker.phase == "checkpoint_infer"
            ? "infer" : worker.phase;
        classification.detected = false;
        PopulateAuthoritativeClassification(
            classification, worker, processes, false);
        classifications.push_back(std::move(classification));
    }
    return classifications;
}

SchedulerWorkerClassificationSummary SummarizeSchedulerWorkers(
    const std::vector<SchedulerWorkerClassification>& classifications)
{
    SchedulerWorkerClassificationSummary summary;
    for (const auto& classification : classifications)
    {
        SchedulerWorkerAggregate* aggregate = nullptr;
        SchedulerWorkerAggregate* stateAggregate = nullptr;
        if (classification.kind == "train")
        {
            if (classification.managed)
            {
                aggregate = &summary.managedTrain;
                if (classification.executionState ==
                    ProcessExecutionState::Running)
                    stateAggregate = &summary.managedRunningTrain;
                else if (classification.executionState ==
                             ProcessExecutionState::Stopped &&
                         classification.lifecycleStatus == "paused")
                    stateAggregate = &summary.managedPausedTrain;
            }
            else if (classification.authoritative)
            {
                aggregate = classification.executionState ==
                                ProcessExecutionState::Missing
                    ? &summary.expectedMissingTrain
                    : &summary.identityMismatchTrain;
            }
            else
                aggregate = &summary.unmanagedTrain;
        }
        else if (classification.kind == "infer")
        {
            if (classification.managed)
            {
                aggregate = &summary.managedInfer;
                if (classification.executionState ==
                    ProcessExecutionState::Running)
                    stateAggregate = &summary.managedRunningInfer;
                else if (classification.executionState ==
                             ProcessExecutionState::Stopped &&
                         classification.lifecycleStatus == "paused")
                    stateAggregate = &summary.managedPausedInfer;
            }
            else if (classification.authoritative)
            {
                aggregate = classification.executionState ==
                                ProcessExecutionState::Missing
                    ? &summary.expectedMissingInfer
                    : &summary.identityMismatchInfer;
            }
            else
                aggregate = &summary.unmanagedInfer;
        }
        else if (classification.kind == "analyze")
        {
            if (classification.managed)
            {
                aggregate = &summary.managedAnalyze;
                if (classification.executionState ==
                    ProcessExecutionState::Running)
                    stateAggregate = &summary.managedRunningAnalyze;
                else if (classification.executionState ==
                             ProcessExecutionState::Stopped &&
                         classification.lifecycleStatus == "paused")
                    stateAggregate = &summary.managedPausedAnalyze;
            }
            else if (classification.authoritative)
            {
                aggregate = classification.executionState ==
                                ProcessExecutionState::Missing
                    ? &summary.expectedMissingAnalyze
                    : &summary.identityMismatchAnalyze;
            }
            else
                aggregate = &summary.unmanagedAnalyze;
        }
        if (aggregate == nullptr)
            continue;
        const auto add = [&](SchedulerWorkerAggregate& target) {
            ++target.workers;
            target.cpuPercent += classification.cpuPercent;
            target.memPercent += classification.memPercent;
            target.rssMb += classification.rssMb;
        };
        add(*aggregate);
        if (stateAggregate != nullptr)
            add(*stateAggregate);
    }
    return summary;
}

ControlSnapshot LoadControlSnapshot(pqxx::transaction_base& transaction)
{
    pqxx::result rows = transaction.exec(
        "SELECT c.desired_state,c.active_request_id,c.current_pause_request_id,"
        "r.action,"
        "r.cancellation_mode,COALESCE(r.infer_before_cancel,false) "
        "FROM experiment_global_control c "
        "LEFT JOIN experiment_admin_request r ON r.request_id=c.active_request_id "
        "WHERE c.singleton=true;");
    if (rows.empty())
        throw std::runtime_error(
            "global experiment control migration 046 is required");
    ControlSnapshot snapshot;
    snapshot.desiredState = rows[0][0].as<std::string>();
    if (!rows[0][1].is_null())
        snapshot.activeRequestId = rows[0][1].as<long long>();
    if (!rows[0][2].is_null())
        snapshot.currentPauseRequestId = rows[0][2].as<long long>();
    if (!rows[0][3].is_null())
        snapshot.activeAction = rows[0][3].as<std::string>();
    if (!rows[0][4].is_null())
        snapshot.cancellationMode = rows[0][4].as<std::string>();
    snapshot.inferBeforeCancel = rows[0][5].as<bool>();
    return snapshot;
}

} // namespace EA::GlobalExperimentControl
