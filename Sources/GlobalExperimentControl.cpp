#include "GlobalExperimentControl.hpp"
#include "ExperimentCurrentOperation.hpp"
#include "SchedulerOwnershipRepository.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <functional>
#include <libproc.h>
#include <sstream>
#include <stdexcept>
#include <sys/sysctl.h>
#include <thread>
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
             std::isspace(
                 static_cast<unsigned char>(command[valueEnd]))))
            return true;
        position += option.size();
    }
    return false;
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
               command.find("--scheduler-checkpoint-eval-id=") != std::string::npos;
    return false;
}

std::optional<std::string> CanonicalizeExecutablePath(
    const std::string& executablePath)
{
    if (executablePath.empty() || executablePath.front() != '/')
        return std::nullopt;
    char* canonicalExecutable = ::realpath(executablePath.c_str(), nullptr);
    if (canonicalExecutable == nullptr)
        return std::nullopt;
    std::string result{canonicalExecutable};
    std::free(canonicalExecutable);
    return result;
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

class PosixProcessOperations final : public ProcessOperations
{
public:
    PosixProcessOperations()
        : PosixProcessOperations(
              std::make_unique<PosixNativeProcessObservationBackend>())
    {
    }

    explicit PosixProcessOperations(
        std::unique_ptr<NativeProcessObservationBackend> backend)
        : backend_(std::move(backend))
    {
        if (!backend_)
            throw std::invalid_argument("native_process_observation_backend_required");
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
            // macOS can report ENOENT from proc_pidpath for a live stopped
            // process. KERN_PROCARGS2 supplies the kernel-recorded exec path;
            // it is deliberately not reconstructed from ps/argv text.
            if (errorNumber != ENOENT)
                return observation;
            executablePath =
                backend_->ReadKernelExecutablePath(pid, errorNumber);
            if (!executablePath)
                return observation;
        }
        const std::optional<std::string> canonicalExecutable =
            CanonicalizeExecutablePath(*executablePath);
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

    bool SignalProcessGroup(int processGroupId,
                            int signalNumber,
                            int& errorNumber) override
    {
        errorNumber = 0;
        if (processGroupId <= 1)
        {
            errorNumber = EINVAL;
            return false;
        }
        if (::kill(-static_cast<pid_t>(processGroupId), signalNumber) == 0)
            return true;
        errorNumber = errno;
        return false;
    }

    bool WaitForProcessGroupExit(
        int processGroupId,
        std::chrono::milliseconds timeout) override
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        do
        {
            errno = 0;
            if (::kill(-static_cast<pid_t>(processGroupId), 0) != 0 &&
                errno == ESRCH)
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while (std::chrono::steady_clock::now() < deadline);
        errno = 0;
        return ::kill(-static_cast<pid_t>(processGroupId), 0) != 0 &&
               errno == ESRCH;
    }

    int CallerPid() const override { return static_cast<int>(::getpid()); }
    int CallerProcessGroupId() const override
    {
        return static_cast<int>(::getpgrp());
    }

private:
    std::unique_ptr<NativeProcessObservationBackend> backend_;
};

std::string ActionState(Action action)
{
    return action == Action::PauseAll ? "paused" : "running";
}

std::string CurrentRequester()
{
    if (const char* user = std::getenv("USER"); user != nullptr && *user != '\0')
        return user;
    return "unknown";
}

bool SchedulerObservedRunning()
{
    FILE* pipe = ::popen("ps -axo command", "r");
    if (pipe == nullptr)
        return false;
    char buffer[8192];
    bool found = false;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        const std::string command{buffer};
        if (command.find("LSTM_Release") != std::string::npos &&
            command.find("--schedule-experiments") != std::string::npos)
        {
            found = true;
            break;
        }
    }
    ::pclose(pipe);
    return found;
}

struct DbTarget
{
    ManagedWorker worker;
    std::string workerIdentity;
    std::optional<long long> workerGlobalPauseRequestId;
    std::optional<long long> sourcePauseRequestId;
    std::optional<long long> checkpointEvalId;
    bool checkpointWorker = false;
    std::optional<int> currentEpoch;
    int checkpointInterval = 0;
    int targetEpochs = 0;
    std::optional<int> lastCheckpointStopDecisionEpoch;
    std::optional<int> latestCheckpointEpoch;
    std::optional<long long> latestCheckpointModelId;
    bool hasInferenceRange = false;
    std::optional<int> cancellationCheckpoint;
    std::optional<long long> cancellationCheckpointModelId;
    bool signalImmediately = false;
    bool resumeBeforeAction = false;
    bool inferenceQueued = false;
    bool inferenceAlreadyCompleted = false;
    bool inferenceAlreadyRunning = false;
    bool inferenceFailed = false;
    bool inferenceRequested = false;
    bool latestCheckpointAmbiguous = false;
    std::string inferenceFailureDetail;
    bool frozenReplayTarget = false;
    bool authoritativeRowExists = false;
    bool authoritativeActive = false;
    bool authoritativeExactMatch = false;
    bool authoritativeExactTerminalDeparture = false;
    bool authoritativePendingCancellation = false;
    std::string authoritativeMismatchDetail;
    std::string plan;
};

using SignalAuthorization = std::function<bool(
    const ManagedWorker&,
    const ProcessObservation&,
    int,
    int&,
    std::string&)>;

SignalOutcome PauseWorkerAuthorized(
    const ManagedWorker& worker,
    ProcessOperations& processes,
    const SignalAuthorization& authorize);
SignalOutcome ResumeWorkerAuthorized(
    const ManagedWorker& worker,
    ProcessOperations& processes,
    const SignalAuthorization& authorize);
SignalOutcome CancelWorkerAuthorized(
    const ManagedWorker& worker,
    bool resumeFirst,
    std::chrono::milliseconds grace,
    ProcessOperations& processes,
    const SignalAuthorization& authorize);

void RequireAffectedRows(const pqxx::result& result,
                         pqxx::result::size_type expected,
                         const std::string& mutation)
{
    if (result.affected_rows() != expected)
    {
        throw std::runtime_error(
            "guarded_mutation_predicate_mismatch:" + mutation +
            ":expected=" + std::to_string(expected) +
            ":actual=" + std::to_string(result.affected_rows()));
    }
}

bool OptionalTextEqual(const std::optional<std::string>& expected,
                       const pqxx::field& actual)
{
    return expected ? (!actual.is_null() &&
                       actual.as<std::string>() == *expected)
                    : actual.is_null();
}

bool OptionalIntegerEqual(const std::optional<int>& expected,
                          const pqxx::field& actual)
{
    return expected ? (!actual.is_null() && actual.as<int>() == *expected)
                    : actual.is_null();
}

void LoadAuthoritativeTargetState(pqxx::transaction_base& transaction,
                                  long long requestId,
                                  Action action,
                                  DbTarget& target)
{
    pqxx::result rows;
    if (target.checkpointWorker)
    {
        rows = transaction.exec_params(
            "SELECT status,phase,worker_pid,worker_process_group_id,"
            "worker_process_start_identity,worker_executable,"
            "worker_command_line,worker_control_state,"
            "worker_global_pause_request_id,cancellation_request_id,"
            "active_scheduler_worker_attempt_id "
            "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            *target.checkpointEvalId);
    }
    else
    {
        rows = transaction.exec_params(
            "SELECT status,phase,worker_pid,worker_process_group_id,"
            "worker_process_start_identity,worker_executable,"
            "worker_command_line,worker_control_state,"
            "worker_global_pause_request_id,cancellation_request_id,"
            "active_scheduler_worker_attempt_id "
            "FROM experiment WHERE experiment_id=$1;",
            target.worker.experimentId);
    }
    target.authoritativeRowExists = !rows.empty();
    target.authoritativeActive =
        !rows.empty() && rows[0][0].as<std::string>() == "running" &&
        (!target.checkpointWorker ||
         rows[0][1].as<std::string>() == "infer");
    target.authoritativePendingCancellation =
        action == Action::CancelAll && !target.checkpointWorker &&
        !rows.empty() && rows[0][0].as<std::string>() == "pending" &&
        !rows[0][9].is_null() &&
        rows[0][9].as<long long>() == requestId;
    target.authoritativeExactMatch =
        target.authoritativeActive &&
        rows[0][1].as<std::string>() ==
            (target.checkpointWorker ? "infer" : target.worker.phase) &&
        (!rows[0][2].is_null() &&
         rows[0][2].as<int>() == target.worker.pid) &&
        OptionalIntegerEqual(target.worker.processGroupId, rows[0][3]) &&
        OptionalTextEqual(target.worker.processStartIdentity, rows[0][4]) &&
        OptionalTextEqual(target.worker.executable, rows[0][5]) &&
        OptionalTextEqual(target.worker.commandLine, rows[0][6]) &&
        target.worker.workerAttemptId.has_value() &&
        !rows[0][10].is_null() &&
        rows[0][10].as<long long>() ==
            *target.worker.workerAttemptId;

    if (!target.authoritativeActive &&
        !rows.empty() &&
        rows[0][10].is_null() &&
        target.worker.workerAttemptId)
    {
        target.authoritativeExactTerminalDeparture =
            transaction.exec_params(
                "SELECT EXISTS ("
                "SELECT 1 FROM experiment_scheduler_worker_attempt a "
                "WHERE a.worker_attempt_id=$1 "
                "AND a.worker_kind=$2 "
                "AND a.experiment_id=$3 "
                "AND a.checkpoint_eval_id IS NOT DISTINCT FROM $4 "
                "AND a.lifecycle_phase=$5 "
                "AND a.capacity_class=$5 "
                "AND a.lifecycle_state IN "
                "('completed','failed','launch_failed','abandoned') "
                "AND a.worker_pid IS NOT DISTINCT FROM $6 "
                "AND a.worker_process_group_id IS NOT DISTINCT FROM $7 "
                "AND a.worker_process_start_identity IS NOT DISTINCT FROM $8 "
                "AND a.canonical_executable_path IS NOT DISTINCT FROM $9 "
                "AND a.command_line IS NOT DISTINCT FROM $10);",
                *target.worker.workerAttemptId,
                target.checkpointWorker ? "checkpoint_infer" : "experiment",
                target.worker.experimentId,
                target.checkpointEvalId,
                target.checkpointWorker ? "infer" : target.worker.phase,
                target.worker.pid > 0
                    ? std::optional<int>{target.worker.pid}
                    : std::nullopt,
                target.worker.processGroupId,
                target.worker.processStartIdentity,
                target.worker.executable,
                target.worker.commandLine)[0][0].as<bool>();
    }

    if (target.authoritativeExactMatch && action == Action::CancelAll)
    {
        target.authoritativeExactMatch =
            !rows[0][9].is_null() &&
            rows[0][9].as<long long>() == requestId;
    }
    if (target.authoritativeExactMatch && action == Action::ResumeAll)
    {
        target.authoritativeExactMatch =
            target.sourcePauseRequestId &&
            rows[0][7].as<std::string>() == "paused" &&
            !rows[0][8].is_null() &&
            rows[0][8].as<long long>() ==
                *target.sourcePauseRequestId;
    }

    if (!target.authoritativeRowExists)
        target.authoritativeMismatchDetail =
            "authoritative_lifecycle_row_missing";
    else if (!target.authoritativeActive)
        target.authoritativeMismatchDetail =
            "authoritative_lifecycle_not_active";
    else if (rows[0][2].is_null())
        target.authoritativeMismatchDetail =
            "authoritative_active_worker_pid_missing";
    else if (rows[0][2].as<int>() != target.worker.pid)
        target.authoritativeMismatchDetail =
            "authoritative_worker_pid_replaced_since_plan_frozen";
    else if (!OptionalIntegerEqual(target.worker.processGroupId, rows[0][3]))
        target.authoritativeMismatchDetail =
            "authoritative_worker_process_group_replaced_since_plan_frozen";
    else if (!OptionalTextEqual(
                 target.worker.processStartIdentity, rows[0][4]))
        target.authoritativeMismatchDetail =
            "authoritative_worker_start_identity_replaced_since_plan_frozen";
    else if (!OptionalTextEqual(target.worker.executable, rows[0][5]))
        target.authoritativeMismatchDetail =
            "authoritative_worker_executable_replaced_since_plan_frozen";
    else if (!OptionalTextEqual(target.worker.commandLine, rows[0][6]))
        target.authoritativeMismatchDetail =
            "authoritative_worker_command_replaced_since_plan_frozen";
    else if (!target.worker.workerAttemptId ||
             rows[0][10].is_null() ||
             rows[0][10].as<long long>() !=
                 *target.worker.workerAttemptId)
        target.authoritativeMismatchDetail =
            "authoritative_worker_attempt_replaced_since_plan_frozen";
    else if (action == Action::CancelAll &&
             (rows[0][9].is_null() ||
              rows[0][9].as<long long>() != requestId))
        target.authoritativeMismatchDetail =
            "authoritative_foreign_cancellation_request";
    else if (action == Action::ResumeAll)
        target.authoritativeMismatchDetail =
            "authoritative_pause_generation_or_control_state_changed";
    else
        target.authoritativeMismatchDetail =
            "authoritative_worker_state_changed_since_plan_frozen";

    if (target.authoritativeExactMatch)
    {
        target.resumeBeforeAction =
            rows[0][7].as<std::string>() == "paused";
        if (!rows[0][8].is_null())
            target.workerGlobalPauseRequestId =
                rows[0][8].as<long long>();
    }
}

std::vector<DbTarget> LoadTargets(pqxx::transaction_base& transaction)
{
    pqxx::result rows = transaction.exec(
        "SELECT e.experiment_id, e.status, e.phase, e.worker_pid, "
        "e.worker_process_group_id, e.worker_executable, e.worker_command_line, "
        "e.worker_process_start_identity, "
        "e.worker_control_state, e.worker_global_pause_request_id, "
        "e.current_epoch, e.checkpoint_interval, "
        "e.target_epochs,e.last_checkpoint_stop_decision_epoch,"
        "e.infer_start IS NOT NULL AND e.infer_end IS NOT NULL, "
        "cp.completed_epoch, cp.model_id,cp.same_epoch_count,"
        "e.active_scheduler_worker_attempt_id,"
        "a.worker_kind,a.capacity_class,a.lifecycle_state,"
        "a.launch_attempt_identity "
        "FROM experiment e "
        "LEFT JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
        "LEFT JOIN LATERAL ("
        "  SELECT round(tm.value)::int AS completed_epoch, m.model_id,"
        "   count(*) OVER (PARTITION BY round(tm.value)::int) AS same_epoch_count "
        "  FROM model m "
        "  JOIN matrix tm ON tm.model_id=m.model_id "
        "   AND tm.param_name='train_config_meta' "
        "   AND tm.row_idx=0 AND tm.col_idx=10 "
        "  WHERE m.experiment_id=e.experiment_id "
        "  AND (COALESCE(m.comment,'') ILIKE '%periodic training checkpoint%' "
        "       OR m.model_id=e.stopped_at_checkpoint_model_id "
        "       OR m.model_id=e.last_model_id) "
        "  ORDER BY round(tm.value)::int DESC, m.model_id DESC LIMIT 1"
        ") cp ON true "
        "WHERE e.status IN ('pending','paused','running') "
        "ORDER BY e.experiment_id;");

    std::vector<DbTarget> targets;
    targets.reserve(rows.size());
    for (const auto& row : rows)
    {
        DbTarget target;
        target.worker.experimentId = row[0].as<long long>();
        target.workerIdentity =
            "experiment:" + std::to_string(target.worker.experimentId);
        target.worker.lifecycleStatus = row[1].as<std::string>();
        target.worker.phase = row[2].as<std::string>();
        if (!row[3].is_null())
            target.worker.pid = row[3].as<int>();
        if (!row[4].is_null())
            target.worker.processGroupId = row[4].as<int>();
        if (!row[5].is_null())
            target.worker.executable = row[5].as<std::string>();
        if (!row[6].is_null())
            target.worker.commandLine = row[6].as<std::string>();
        if (!row[7].is_null())
            target.worker.processStartIdentity = row[7].as<std::string>();
        target.resumeBeforeAction =
            !row[8].is_null() && row[8].as<std::string>() == "paused";
        if (!row[9].is_null())
            target.workerGlobalPauseRequestId = row[9].as<long long>();
        if (!row[10].is_null())
            target.currentEpoch = row[10].as<int>();
        target.checkpointInterval = row[11].as<int>();
        target.targetEpochs = row[12].as<int>();
        if (!row[13].is_null())
            target.lastCheckpointStopDecisionEpoch = row[13].as<int>();
        target.hasInferenceRange = row[14].as<bool>();
        if (!row[15].is_null())
            target.latestCheckpointEpoch = row[15].as<int>();
        if (!row[16].is_null())
            target.latestCheckpointModelId = row[16].as<long long>();
        target.latestCheckpointAmbiguous =
            !row[17].is_null() && row[17].as<int>() > 1;
        if (!row[18].is_null())
            target.worker.workerAttemptId = row[18].as<long long>();
        if (!row[19].is_null())
            target.worker.workerKind = row[19].as<std::string>();
        if (!row[20].is_null())
            target.worker.capacityClass = row[20].as<std::string>();
        if (!row[21].is_null())
            target.worker.attemptLifecycleState =
                row[21].as<std::string>();
        if (!row[22].is_null())
            target.worker.launchAttemptIdentity =
                row[22].as<std::string>();
        targets.push_back(std::move(target));
    }

    pqxx::result checkpointWorkers = transaction.exec(
        "SELECT COALESCE(ce.parent_experiment_id,ce.experiment_id),"
        "ce.worker_pid,ce.worker_process_group_id,ce.worker_executable,"
        "ce.worker_command_line,ce.worker_process_start_identity,"
        "ce.worker_control_state,ce.worker_global_pause_request_id,"
        "ce.checkpoint_eval_id,ce.checkpoint_epoch,ce.checkpoint_model_id,"
        "ce.active_scheduler_worker_attempt_id,"
        "a.worker_kind,a.capacity_class,a.lifecycle_state,"
        "a.launch_attempt_identity "
        "FROM experiment_checkpoint_eval ce "
        "LEFT JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=ce.active_scheduler_worker_attempt_id "
        "WHERE ce.status='running' AND ce.phase='infer' "
        "ORDER BY ce.checkpoint_eval_id;");
    for (const auto& row : checkpointWorkers)
    {
        DbTarget target;
        target.worker.experimentId = row[0].as<long long>();
        target.worker.lifecycleStatus = "running";
        target.worker.phase = "checkpoint_infer";
        if (!row[1].is_null())
            target.worker.pid = row[1].as<int>();
        if (!row[2].is_null())
            target.worker.processGroupId = row[2].as<int>();
        if (!row[3].is_null())
            target.worker.executable = row[3].as<std::string>();
        if (!row[4].is_null())
            target.worker.commandLine = row[4].as<std::string>();
        if (!row[5].is_null())
            target.worker.processStartIdentity = row[5].as<std::string>();
        target.resumeBeforeAction =
            !row[6].is_null() && row[6].as<std::string>() == "paused";
        if (!row[7].is_null())
            target.workerGlobalPauseRequestId = row[7].as<long long>();
        target.checkpointEvalId = row[8].as<long long>();
        target.worker.checkpointEvalId = target.checkpointEvalId;
        if (!row[9].is_null())
            target.cancellationCheckpoint = row[9].as<int>();
        if (!row[10].is_null())
            target.cancellationCheckpointModelId = row[10].as<long long>();
        if (!row[11].is_null())
            target.worker.workerAttemptId = row[11].as<long long>();
        if (!row[12].is_null())
            target.worker.workerKind = row[12].as<std::string>();
        if (!row[13].is_null())
            target.worker.capacityClass = row[13].as<std::string>();
        if (!row[14].is_null())
            target.worker.attemptLifecycleState =
                row[14].as<std::string>();
        if (!row[15].is_null())
            target.worker.launchAttemptIdentity =
                row[15].as<std::string>();
        target.checkpointWorker = true;
        target.workerIdentity =
            "checkpoint_eval:" + std::to_string(*target.checkpointEvalId);
        targets.push_back(std::move(target));
    }
    return targets;
}

std::vector<DbTarget> LoadRetryTargets(pqxx::transaction_base& transaction,
                                       long long requestId,
                                       Action action)
{
    pqxx::result outcomes = transaction.exec_params(
        "SELECT worker_identity,experiment_id,checkpoint_eval_id,worker_kind,"
        "phase,lifecycle_status,worker_pid,worker_process_group_id,"
        "worker_process_start_identity,worker_executable,worker_command_line,"
        "source_pause_request_id,cancellation_checkpoint_epoch,"
        "cancellation_checkpoint_model_id,inference_action,outcome_status,"
        "identity_result,signal_result,worker_attempt_id "
        "FROM experiment_admin_worker_outcome WHERE request_id=$1 "
        "ORDER BY worker_identity;",
        requestId);

    std::vector<DbTarget> retryTargets;
    retryTargets.reserve(outcomes.size());
    for (const pqxx::row& outcome : outcomes)
    {
        DbTarget target;
        target.frozenReplayTarget = true;
        target.workerIdentity = outcome[0].as<std::string>();
        target.worker.experimentId = outcome[1].as<long long>();
        if (!outcome[2].is_null())
        {
            target.checkpointEvalId = outcome[2].as<long long>();
            target.worker.checkpointEvalId = target.checkpointEvalId;
        }
        target.checkpointWorker =
            outcome[3].as<std::string>() == "checkpoint_infer";
        target.worker.phase = outcome[4].as<std::string>();
        target.worker.lifecycleStatus = outcome[5].as<std::string>();
        if (!outcome[6].is_null())
            target.worker.pid = outcome[6].as<int>();
        if (!outcome[7].is_null())
            target.worker.processGroupId = outcome[7].as<int>();
        if (!outcome[8].is_null())
            target.worker.processStartIdentity = outcome[8].as<std::string>();
        if (!outcome[9].is_null())
            target.worker.executable = outcome[9].as<std::string>();
        if (!outcome[10].is_null())
            target.worker.commandLine = outcome[10].as<std::string>();
        if (!outcome[11].is_null())
            target.sourcePauseRequestId = outcome[11].as<long long>();
        if (!outcome[12].is_null())
            target.cancellationCheckpoint = outcome[12].as<int>();
        if (!outcome[13].is_null())
        {
            target.cancellationCheckpointModelId =
                outcome[13].as<long long>();
            target.latestCheckpointModelId = outcome[13].as<long long>();
        }
        const std::string outcomeStatus = outcome[15].as<std::string>();
        const std::string identityResult = outcome[16].as<std::string>();
        const std::string signalResult = outcome[17].as<std::string>();
        if (!outcome[18].is_null())
            target.worker.workerAttemptId =
                outcome[18].as<long long>();
        target.worker.workerKind =
            target.checkpointWorker ? "checkpoint_infer"
                                    : "experiment";
        target.worker.capacityClass =
            target.checkpointWorker ? "infer"
                                    : target.worker.phase;
        const bool retryableUnresolved =
            identityResult == "stale_pid" ||
            identityResult == "identity_validation_failed" ||
            identityResult == "unsafe_process_group" ||
            identityResult == "permission_denied" ||
            identityResult == "inspection_failed" ||
            signalResult == "stale_pid" ||
            signalResult == "identity_validation_failed" ||
            signalResult == "permission_failure" ||
            signalResult == "signaling_failure";
        if (outcomeStatus == "planned")
        {
            target.signalImmediately = action == Action::CancelAll;
        }
        else if (outcomeStatus == "pending_checkpoint")
        {
            target.plan = "pending_checkpoint";
        }
        else if ((outcomeStatus == "failed" ||
                  outcomeStatus == "partial") &&
                 retryableUnresolved)
        {
            target.signalImmediately = action == Action::CancelAll;
        }
        else
        {
            target.plan = "already_accounted";
        }
        const std::string inferenceAction = outcome[14].as<std::string>();
        target.inferenceRequested = inferenceAction != "none";
        target.inferenceQueued =
            inferenceAction == "queued" || inferenceAction == "running";
        target.inferenceAlreadyCompleted =
            inferenceAction == "already_completed" ||
            inferenceAction == "completed";
        target.inferenceFailed =
            inferenceAction == "failed" ||
            inferenceAction == "no_checkpoint" ||
            inferenceAction == "identity_ambiguous";
        LoadAuthoritativeTargetState(
            transaction, requestId, action, target);
        if (outcomeStatus == "pending_checkpoint" &&
            target.authoritativePendingCancellation)
            target.plan = "already_accounted";
        retryTargets.push_back(std::move(target));
    }
    return retryTargets;
}

std::vector<DbTarget> LoadPauseGenerationTargets(
    pqxx::transaction_base& transaction,
    long long pauseRequestId)
{
    std::vector<DbTarget> currentTargets = LoadTargets(transaction);
    pqxx::result members = transaction.exec_params(
        "SELECT worker_identity,experiment_id,checkpoint_eval_id,worker_kind,"
        "phase,lifecycle_status,worker_pid,worker_process_group_id,"
        "worker_process_start_identity,worker_executable,"
        "worker_command_line,worker_attempt_id "
        "FROM experiment_admin_worker_outcome "
        "WHERE request_id=$1 AND outcome_status='completed' "
        "AND identity_result='validated' "
        "AND signal_result IN ('signaled','already_requested_state') "
        "ORDER BY worker_identity;",
        pauseRequestId);

    std::vector<DbTarget> targets;
    targets.reserve(members.size());
    for (const pqxx::row& member : members)
    {
        const std::string workerIdentity = member[0].as<std::string>();
        const auto current = std::find_if(
            currentTargets.begin(),
            currentTargets.end(),
            [&](const DbTarget& candidate) {
                return candidate.workerIdentity == workerIdentity;
            });

        DbTarget target;
        target.workerIdentity = workerIdentity;
        target.worker.experimentId = member[1].as<long long>();
        if (!member[2].is_null())
        {
            target.checkpointEvalId = member[2].as<long long>();
            target.worker.checkpointEvalId = target.checkpointEvalId;
        }
        target.checkpointWorker =
            member[3].as<std::string>() == "checkpoint_infer";
        target.worker.phase = member[4].as<std::string>();
        target.worker.lifecycleStatus = member[5].as<std::string>();
        if (!member[6].is_null())
            target.worker.pid = member[6].as<int>();
        if (!member[7].is_null())
            target.worker.processGroupId = member[7].as<int>();
        if (!member[8].is_null())
            target.worker.processStartIdentity = member[8].as<std::string>();
        if (!member[9].is_null())
            target.worker.executable = member[9].as<std::string>();
        if (!member[10].is_null())
            target.worker.commandLine = member[10].as<std::string>();
        if (!member[11].is_null())
            target.worker.workerAttemptId =
                member[11].as<long long>();
        target.worker.workerKind =
            target.checkpointWorker ? "checkpoint_infer"
                                    : "experiment";
        target.worker.capacityClass =
            target.checkpointWorker ? "infer"
                                    : target.worker.phase;
        target.sourcePauseRequestId = pauseRequestId;
        target.frozenReplayTarget = true;

        const bool exactCurrentWorker =
            current != currentTargets.end() &&
            current->worker.lifecycleStatus == "running" &&
            current->worker.workerAttemptId ==
                target.worker.workerAttemptId &&
            current->worker.pid == target.worker.pid &&
            current->worker.processGroupId == target.worker.processGroupId &&
            current->worker.processStartIdentity ==
                target.worker.processStartIdentity &&
            current->worker.executable == target.worker.executable &&
            current->worker.commandLine == target.worker.commandLine;
        if (exactCurrentWorker)
        {
            target.resumeBeforeAction = current->resumeBeforeAction;
            target.workerGlobalPauseRequestId =
                current->workerGlobalPauseRequestId;
            if (!current->resumeBeforeAction ||
                current->workerGlobalPauseRequestId !=
                    std::optional<long long>{pauseRequestId})
                target.plan = "already_satisfied";
        }
        else
        {
            target.plan = "generation_member_requires_reconciliation";
        }
        LoadAuthoritativeTargetState(
            transaction, pauseRequestId, Action::ResumeAll, target);
        targets.push_back(std::move(target));
    }
    return targets;
}

void InsertOutcome(pqxx::transaction_base& transaction,
                   long long requestId,
                   const DbTarget& target,
                   const std::string& outcomeStatus,
                   const std::string& inferenceAction,
                   const std::string& detail)
{
    const pqxx::result inserted = transaction.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,checkpoint_eval_id,"
        "worker_kind,phase,lifecycle_status,worker_pid,"
        "worker_process_group_id,worker_process_start_identity,"
        "worker_executable,worker_command_line,source_pause_request_id,"
        "cancellation_checkpoint_epoch,cancellation_checkpoint_model_id,"
        "inference_action,outcome_status,detail,worker_attempt_id) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19) "
        "ON CONFLICT (request_id,worker_identity) DO NOTHING;",
        requestId,
        target.workerIdentity,
        target.worker.experimentId,
        target.checkpointEvalId,
        target.checkpointWorker ? "checkpoint_infer" : "experiment",
        target.worker.phase,
        target.worker.lifecycleStatus,
        target.worker.pid > 0 ? std::optional<int>{target.worker.pid}
                              : std::optional<int>{},
        target.worker.processGroupId,
        target.worker.processStartIdentity,
        target.worker.executable,
        target.worker.commandLine,
        target.sourcePauseRequestId,
        target.cancellationCheckpoint,
        target.cancellationCheckpointModelId
            ? target.cancellationCheckpointModelId
            : target.latestCheckpointModelId,
        inferenceAction,
        outcomeStatus,
        detail,
        target.worker.workerAttemptId);
    RequireAffectedRows(inserted, 1, "insert_frozen_worker_outcome");
}

void MarkResumeAllAlreadySatisfied(pqxx::transaction_base& transaction,
                                   long long requestId,
                                   const DbTarget& target,
                                   const std::string& detail)
{
    InsertOutcome(
        transaction, requestId, target, "completed", "none", detail);
    const pqxx::result updated = transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "identity_result='validated',"
        "signal_result='already_requested_state',"
        "outcome_status='completed',detail=$1,updated_at=now() "
        "WHERE request_id=$2 AND worker_identity=$3;",
        detail,
        requestId,
        target.workerIdentity);
    RequireAffectedRows(
        updated, 1, "mark_resume_all_already_satisfied");
}

std::optional<long long> QueueCancellationInference(
    pqxx::transaction_base& transaction,
    long long requestId,
    DbTarget& target)
{
    if (!target.latestCheckpointEpoch || !target.latestCheckpointModelId ||
        target.latestCheckpointAmbiguous)
    {
        target.inferenceFailureDetail = target.latestCheckpointAmbiguous
            ? "durable_checkpoint_identity_ambiguous"
            : "no_valid_durable_checkpoint_for_inference";
        return std::nullopt;
    }

    target.cancellationCheckpoint = target.latestCheckpointEpoch;
    target.cancellationCheckpointModelId = target.latestCheckpointModelId;

    pqxx::result modelEvidence = transaction.exec_params(
        "SELECT count(*) FILTER (WHERE m.experiment_id=$1)::int,"
        "count(tm.*) FILTER (WHERE tm.param_name='train_config_meta' "
        "AND tm.row_idx=0 AND tm.col_idx=10 "
        "AND round(tm.value)::int=$2)::int "
        "FROM model m LEFT JOIN matrix tm ON tm.model_id=m.model_id "
        "WHERE m.model_id=$3 GROUP BY m.model_id;",
        target.worker.experimentId,
        *target.latestCheckpointEpoch,
        *target.latestCheckpointModelId);
    if (modelEvidence.empty() || modelEvidence[0][0].as<int>() != 1)
    {
        target.inferenceFailureDetail =
            "cancellation_checkpoint_model_foreign_or_missing";
        return std::nullopt;
    }
    if (modelEvidence[0][1].as<int>() != 1)
    {
        target.inferenceFailureDetail =
            "cancellation_checkpoint_epoch_metadata_invalid";
        return std::nullopt;
    }

    pqxx::result existing = transaction.exec_params(
        "SELECT checkpoint_eval_id,experiment_id,status,phase,"
        "cancellation_request_id,started_at,completed_at,worker_pid,"
        "error_message "
        "FROM experiment_checkpoint_eval "
        "WHERE parent_experiment_id=$1 "
        "AND checkpoint_epoch=$2 AND checkpoint_model_id=$3 "
        "ORDER BY checkpoint_eval_id;",
        target.worker.experimentId,
        *target.latestCheckpointEpoch,
        *target.latestCheckpointModelId);
    if (existing.size() > 1)
    {
        target.inferenceFailed = true;
        target.inferenceFailureDetail =
            "exact_checkpoint_evaluation_ambiguous";
        return std::nullopt;
    }
    if (existing.size() == 1)
    {
        const std::string status = existing[0][2].as<std::string>();
        const std::string phase = existing[0][3].as<std::string>();
        const bool belongsToOtherCancellation =
            !existing[0][4].is_null() &&
            existing[0][4].as<long long>() != requestId;
        const bool crossExperiment =
            existing[0][1].as<long long>() != target.worker.experimentId;
        const bool workerPresent = !existing[0][7].is_null();
        const bool started = !existing[0][5].is_null();
        const bool completed = !existing[0][6].is_null();
        std::string lifecycleFailure;
        if (status == "completed" &&
            (phase != "done" || !completed || workerPresent))
            lifecycleFailure =
                "exact_checkpoint_evaluation_completed_lifecycle_invalid";
        else if (status == "failed" &&
                 ((phase != "infer" && phase != "done") ||
                  !completed || workerPresent))
            lifecycleFailure =
                "exact_checkpoint_evaluation_failed_lifecycle_invalid";
        else if (status == "pending" &&
                 (phase != "infer" || started || completed || workerPresent))
            lifecycleFailure =
                "exact_checkpoint_evaluation_pending_lifecycle_invalid";
        else if (status == "running" &&
                 (phase != "infer" || !started || completed ||
                  !workerPresent))
            lifecycleFailure =
                "exact_checkpoint_evaluation_running_lifecycle_invalid";
        else if (status != "completed" && status != "failed" &&
                 status != "pending" && status != "running")
            lifecycleFailure =
                "exact_checkpoint_evaluation_status_invalid";
        else if ((status == "pending" || status == "running") &&
                 !target.hasInferenceRange)
            lifecycleFailure = "cancellation_inference_range_missing";

        if (crossExperiment)
            target.inferenceFailureDetail =
                "exact_checkpoint_evaluation_cross_experiment";
        else if (belongsToOtherCancellation)
            target.inferenceFailureDetail =
                "exact_checkpoint_evaluation_owned_by_other_cancellation";
        else if (!lifecycleFailure.empty())
            target.inferenceFailureDetail = lifecycleFailure;
        else if (status == "failed")
            target.inferenceFailureDetail =
                existing[0][8].is_null()
                    ? "cancellation_checkpoint_inference_failed"
                    : existing[0][8].as<std::string>();
        target.inferenceFailed =
            crossExperiment || belongsToOtherCancellation ||
            !lifecycleFailure.empty() || status == "failed";
        target.inferenceAlreadyCompleted =
            !target.inferenceFailed && status == "completed";
        target.inferenceAlreadyRunning =
            !target.inferenceFailed && status == "running";
        if (crossExperiment || belongsToOtherCancellation ||
            !lifecycleFailure.empty())
            return existing[0][0].as<long long>();
        if (existing[0][4].is_null())
        {
            const pqxx::result assigned = transaction.exec_params(
                "UPDATE experiment_checkpoint_eval "
                "SET cancellation_request_id=$1,updated_at=now() "
                "WHERE checkpoint_eval_id=$2 "
                "AND cancellation_request_id IS NULL;",
                requestId,
                existing[0][0].as<long long>());
            RequireAffectedRows(
                assigned, 1, "assign_cancellation_inference_request");
        }
        if (target.inferenceFailed)
            return existing[0][0].as<long long>();
        target.inferenceQueued =
            !target.inferenceAlreadyCompleted && !target.inferenceFailed;
        return existing[0][0].as<long long>();
    }

    if (!target.hasInferenceRange)
    {
        target.inferenceFailureDetail =
            "cancellation_inference_range_missing";
        return std::nullopt;
    }

    pqxx::result inserted = transaction.exec_params(
        "INSERT INTO experiment_checkpoint_eval ("
        "experiment_id,parent_experiment_id,checkpoint_epoch,"
        "checkpoint_model_id,symbol,prediction_horizon,cancellation_request_id) "
        "SELECT experiment_id,experiment_id,$1,$2,symbol,prediction_horizon,$3 "
        "FROM experiment WHERE experiment_id=$4 "
        "RETURNING checkpoint_eval_id;",
        *target.latestCheckpointEpoch,
        *target.latestCheckpointModelId,
        requestId,
        target.worker.experimentId);
    if (inserted.empty())
    {
        target.inferenceFailureDetail =
            "exact_checkpoint_evaluation_materialization_failed";
        return std::nullopt;
    }
    RequireAffectedRows(
        inserted, 1, "insert_cancellation_inference_request");
    target.inferenceQueued = true;
    return inserted[0][0].as<long long>();
}

void UpdateSignalOutcome(pqxx::transaction_base& transaction,
                         long long requestId,
                         const DbTarget& target,
                         const SignalOutcome& outcome,
                         bool cancelled,
                         const std::string& applicationOwner,
                         const std::string& action)
{
    std::ostringstream signalList;
    for (size_t i = 0; i < outcome.signals.size(); ++i)
    {
        if (i)
            signalList << "|";
        signalList << outcome.signals[i];
    }
    std::string persistedSignalResult = outcome.result;
    if (persistedSignalResult == "unsafe_process_group" ||
        persistedSignalResult == "inspection_failed")
        persistedSignalResult = "identity_validation_failed";
    if (persistedSignalResult == "permission_denied")
        persistedSignalResult = "permission_failure";
    const bool safelyDetached =
        outcome.identity == IdentityResult::ProcessMissing;
    const bool reconciled = outcome.success || safelyDetached;
    const std::string reconciledStatus =
        cancelled && target.inferenceQueued &&
                !target.inferenceAlreadyCompleted
            ? "awaiting_inference"
            : (cancelled && target.inferenceRequested &&
                       target.inferenceFailed
                   ? "partial"
                   : (cancelled && safelyDetached &&
                              outcome.identity !=
                                  IdentityResult::ProcessMissing
                          ? "partial"
                          : "completed"));
    const pqxx::result updated = transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "identity_result=$1,signal_result=$2,requested_signal=$3,"
        "outcome_status=$4,detail=$5,updated_at=now() "
        "WHERE request_id=$6 AND worker_identity=$7 "
        "AND outcome_status IN "
        "('planned','pending_checkpoint','failed','partial') "
        "AND EXISTS ("
        " SELECT 1 FROM experiment_admin_request r "
        " JOIN experiment_global_control c "
        " ON c.active_request_id=r.request_id "
        " WHERE c.singleton AND r.request_id=$6 "
        " AND r.application_owner=$8 AND r.action=$9 "
        " AND r.application_lease_until>now());",
        ToString(outcome.identity),
        persistedSignalResult,
        signalList.str().empty() ? std::optional<std::string>{}
                                 : std::optional<std::string>{signalList.str()},
        reconciled ? reconciledStatus : (cancelled ? "planned" : "failed"),
        outcome.detail,
        requestId,
        target.workerIdentity,
        applicationOwner,
        action);
    RequireAffectedRows(updated, 1, "persist_worker_signal_outcome");
}

void UpdateRequestAccounting(pqxx::transaction_base& transaction,
                             long long requestId,
                             bool keepPending,
                             const std::string& applicationOwner,
                             const std::string& action)
{
    const pqxx::result updated = transaction.exec_params(
        "WITH totals AS ("
        " SELECT count(*)::int AS target_count,"
        " count(*) FILTER (WHERE outcome_status='completed' "
        " AND identity_result NOT IN ('process_missing','stale_pid',"
        " 'identity_validation_failed','unsafe_process_group') "
        " AND signal_result<>'already_requested_state')::int AS successful,"
        " count(*) FILTER (WHERE signal_result='already_requested_state')::int AS already,"
        " count(*) FILTER (WHERE identity_result='process_missing')::int AS missing,"
        " count(*) FILTER (WHERE identity_result IN "
        " ('stale_pid','identity_validation_failed','unsafe_process_group'))::int AS rejected,"
        " count(*) FILTER (WHERE outcome_status IN ('failed','partial'))::int AS failed,"
        " count(*) FILTER (WHERE outcome_status IN "
        " ('planned','pending_checkpoint','awaiting_inference'))::int AS pending"
        " FROM experiment_admin_worker_outcome WHERE request_id=$1"
        ") UPDATE experiment_admin_request r SET "
        "target_count=totals.target_count,successful_count=totals.successful,"
        "already_satisfied_count=totals.already,missing_count=totals.missing,"
        "rejected_count=totals.rejected,failed_count=totals.failed,"
        "status=CASE WHEN $2 OR totals.pending>0 THEN 'pending' "
        " WHEN totals.failed>0 OR totals.rejected>0 THEN 'partial' "
        " ELSE 'completed' END,"
        "completed_at=CASE WHEN $2 OR totals.pending>0 THEN NULL "
        "ELSE COALESCE(r.completed_at,now()) END,"
        "result_summary=jsonb_build_object("
        "'target_count',totals.target_count,'successful_count',totals.successful,"
        "'already_satisfied_count',totals.already,'missing_count',totals.missing,"
        "'rejected_count',totals.rejected,'failed_count',totals.failed,"
        "'pending_count',totals.pending) "
        "FROM totals,experiment_global_control c "
        "WHERE r.request_id=$1 AND r.application_owner=$3 "
        "AND r.action=$4 AND r.application_lease_until>now() "
        "AND c.singleton AND c.active_request_id=r.request_id;",
        requestId,
        keepPending,
        applicationOwner,
        action);
    RequireAffectedRows(updated, 1, "update_request_accounting");
}

std::optional<DbTarget> LoadExperimentResumeTarget(
    pqxx::transaction_base& transaction,
    long long experimentId,
    bool forUpdate)
{
    std::string sql =
        "SELECT e.experiment_id,e.status,e.phase,e.worker_pid,"
        "e.worker_process_group_id,e.worker_executable,"
        "e.worker_command_line,e.worker_process_start_identity,"
        "e.worker_control_state,e.worker_global_pause_request_id,"
        "e.active_scheduler_worker_attempt_id,a.worker_kind,"
        "a.capacity_class,a.lifecycle_state,a.launch_attempt_identity "
        "FROM experiment e "
        "LEFT JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
        "WHERE e.experiment_id=$1";
    if (forUpdate)
        sql += " FOR UPDATE OF e";
    pqxx::result rows = transaction.exec_params(sql, experimentId);
    if (rows.empty())
        return std::nullopt;

    DbTarget target;
    target.worker.experimentId = rows[0][0].as<long long>();
    target.workerIdentity =
        "experiment:" + std::to_string(target.worker.experimentId);
    target.worker.lifecycleStatus = rows[0][1].as<std::string>();
    target.worker.phase = rows[0][2].as<std::string>();
    if (!rows[0][3].is_null())
        target.worker.pid = rows[0][3].as<int>();
    if (!rows[0][4].is_null())
        target.worker.processGroupId = rows[0][4].as<int>();
    if (!rows[0][5].is_null())
        target.worker.executable = rows[0][5].as<std::string>();
    if (!rows[0][6].is_null())
        target.worker.commandLine = rows[0][6].as<std::string>();
    if (!rows[0][7].is_null())
        target.worker.processStartIdentity = rows[0][7].as<std::string>();
    target.resumeBeforeAction =
        rows[0][8].as<std::string>() == "paused";
    if (!rows[0][9].is_null())
        target.workerGlobalPauseRequestId = rows[0][9].as<long long>();
    if (!rows[0][10].is_null())
        target.worker.workerAttemptId = rows[0][10].as<long long>();
    if (!rows[0][11].is_null())
        target.worker.workerKind = rows[0][11].as<std::string>();
    if (!rows[0][12].is_null())
        target.worker.capacityClass = rows[0][12].as<std::string>();
    if (!rows[0][13].is_null())
        target.worker.attemptLifecycleState =
            rows[0][13].as<std::string>();
    if (!rows[0][14].is_null())
        target.worker.launchAttemptIdentity =
            rows[0][14].as<std::string>();
    return target;
}

bool ActiveManagedPhase(const std::string& phase)
{
    return phase == "train" || phase == "infer" || phase == "analyze";
}

bool HasMatchingPauseEvidence(pqxx::transaction_base& transaction,
                              const DbTarget& target,
                              long long pauseRequestId)
{
    pqxx::result rows = transaction.exec_params(
        "SELECT 1 "
        "FROM experiment_admin_worker_outcome o "
        "JOIN experiment_admin_request r ON r.request_id=o.request_id "
        "WHERE o.request_id=$1 AND o.worker_identity=$2 "
        "AND o.worker_kind='experiment' AND o.experiment_id=$3 "
        "AND o.phase=$4 AND o.lifecycle_status='running' "
        "AND o.worker_pid=$5 AND o.worker_process_group_id=$6 "
        "AND o.worker_process_start_identity=$7 "
        "AND o.worker_executable=$8 AND o.worker_command_line=$9 "
        "AND o.identity_result='validated' "
        "AND o.signal_result IN ('signaled','already_requested_state') "
        "AND o.outcome_status='completed' "
        "AND r.action='pause_all' "
        "AND r.status IN ('completed','partial') LIMIT 1;",
        pauseRequestId,
        target.workerIdentity,
        target.worker.experimentId,
        target.worker.phase,
        target.worker.pid > 0 ? std::optional<int>{target.worker.pid}
                              : std::optional<int>{},
        target.worker.processGroupId,
        target.worker.processStartIdentity,
        target.worker.executable,
        target.worker.commandLine);
    return !rows.empty();
}

std::optional<long long> CompletedSelectiveReleaseRequestId(
    pqxx::transaction_base& transaction,
    long long experimentId,
    long long pauseRequestId)
{
    pqxx::result rows = transaction.exec_params(
        "SELECT r.request_id FROM experiment_admin_request r "
        "JOIN experiment_admin_worker_outcome o "
        "ON o.request_id=r.request_id "
        "WHERE r.action='resume_experiment' "
        "AND r.target_experiment_id=$1 "
        "AND r.status='completed' "
        "AND o.source_pause_request_id=$2 "
        "AND o.outcome_status='completed' "
        "AND o.signal_result IN ('signaled','already_requested_state') "
        "LIMIT 1;",
        experimentId,
        pauseRequestId);
    if (rows.empty())
        return std::nullopt;
    return rows[0][0].as<long long>();
}

std::optional<DbTarget> LoadSelectiveResumePlan(
    pqxx::transaction_base& transaction,
    long long requestId)
{
    pqxx::result rows = transaction.exec_params(
        "SELECT worker_identity,experiment_id,phase,lifecycle_status,"
        "worker_pid,worker_process_group_id,worker_process_start_identity,"
        "worker_executable,worker_command_line,source_pause_request_id,"
        "worker_attempt_id,outcome_status "
        "FROM experiment_admin_worker_outcome "
        "WHERE request_id=$1 AND worker_kind='experiment';",
        requestId);
    if (rows.empty())
        return std::nullopt;

    DbTarget target;
    target.workerIdentity = rows[0][0].as<std::string>();
    target.worker.experimentId = rows[0][1].as<long long>();
    target.worker.phase = rows[0][2].as<std::string>();
    target.worker.lifecycleStatus = rows[0][3].as<std::string>();
    if (!rows[0][4].is_null())
        target.worker.pid = rows[0][4].as<int>();
    if (!rows[0][5].is_null())
        target.worker.processGroupId = rows[0][5].as<int>();
    if (!rows[0][6].is_null())
        target.worker.processStartIdentity = rows[0][6].as<std::string>();
    if (!rows[0][7].is_null())
        target.worker.executable = rows[0][7].as<std::string>();
    if (!rows[0][8].is_null())
        target.worker.commandLine = rows[0][8].as<std::string>();
    if (!rows[0][9].is_null())
        target.sourcePauseRequestId = rows[0][9].as<long long>();
    if (!rows[0][10].is_null())
        target.worker.workerAttemptId = rows[0][10].as<long long>();
    target.resumeBeforeAction = true;
    target.frozenReplayTarget = true;
    target.plan = rows[0][11].as<std::string>();
    return target;
}

bool OwnsActiveRequest(pqxx::transaction_base& transaction,
                       long long requestId,
                       const std::string& invocationIdentity,
                       const std::string& action,
                       std::optional<long long> pauseRequestId = std::nullopt)
{
    pqxx::result owned = transaction.exec_params(
        "SELECT 1 FROM experiment_admin_request r "
        "JOIN experiment_global_control c ON c.active_request_id=r.request_id "
        "WHERE c.singleton AND r.request_id=$1 "
        "AND r.application_owner=$2 AND r.action=$3 "
        "AND r.application_lease_until>now() "
        "AND ($4::bigint IS NULL OR c.current_pause_request_id=$4) "
        "FOR UPDATE OF r,c;",
        requestId,
        invocationIdentity,
        action,
        pauseRequestId);
    return !owned.empty();
}

bool HasUnresolvedWorkerOutcome(pqxx::transaction_base& transaction,
                                long long requestId,
                                bool failedIsUnresolved = true)
{
    return transaction.exec_params(
        "SELECT EXISTS ("
        " SELECT 1 FROM experiment_admin_worker_outcome "
        " WHERE request_id=$1 AND ("
        " outcome_status IN ('planned','pending_checkpoint',"
        "'awaiting_inference') "
        " OR ($2 AND outcome_status='failed')));",
        requestId,
        failedIsUnresolved)[0][0].as<bool>();
}

void ClearPauseGenerationAfterResolvedRequest(
    pqxx::transaction_base& transaction,
    long long requestId,
    long long pauseRequestId,
    const std::string& applicationOwner,
    const std::string& action)
{
    (void)transaction.exec_params(
        "UPDATE experiment e SET worker_global_pause_request_id=NULL,"
        "updated_at=now() "
        "FROM experiment_admin_request r,experiment_global_control c "
        "WHERE e.worker_global_pause_request_id=$1 "
        "AND r.request_id=$2 AND r.application_owner=$3 "
        "AND r.action=$4 AND r.application_lease_until>now() "
        "AND c.singleton AND c.active_request_id=r.request_id;",
        pauseRequestId,
        requestId,
        applicationOwner,
        action);
    (void)transaction.exec_params(
        "UPDATE experiment_checkpoint_eval ce "
        "SET worker_global_pause_request_id=NULL,updated_at=now() "
        "FROM experiment_admin_request r,experiment_global_control c "
        "WHERE ce.worker_global_pause_request_id=$1 "
        "AND r.request_id=$2 AND r.application_owner=$3 "
        "AND r.action=$4 AND r.application_lease_until>now() "
        "AND c.singleton AND c.active_request_id=r.request_id;",
        pauseRequestId,
        requestId,
        applicationOwner,
        action);
    const pqxx::row remaining = transaction.exec_params(
        "SELECT "
        "(SELECT count(*) FROM experiment "
        " WHERE worker_global_pause_request_id=$1)+"
        "(SELECT count(*) FROM experiment_checkpoint_eval "
        " WHERE worker_global_pause_request_id=$1);",
        pauseRequestId).one_row();
    if (remaining[0].as<long long>() != 0)
        throw std::runtime_error(
            "guarded_mutation_predicate_mismatch:"
            "pause_generation_member_cleanup");
    const pqxx::result generationCleared = transaction.exec_params(
        "UPDATE experiment_global_control c "
        "SET current_pause_request_id=NULL,updated_at=now() "
        "FROM experiment_admin_request r "
        "WHERE c.singleton AND c.current_pause_request_id=$1 "
        "AND c.active_request_id=$2 AND r.request_id=c.active_request_id "
        "AND r.application_owner=$3 AND r.action=$4 "
        "AND r.application_lease_until>now();",
        pauseRequestId,
        requestId,
        applicationOwner,
        action);
    RequireAffectedRows(
        generationCleared, 1, "clear_resolved_pause_generation");
}

SignalOutcome FrozenTargetAuthorizationFailure(
    const DbTarget& target,
    ProcessOperations& processes)
{
    SignalOutcome outcome;
    if (target.authoritativeActive)
    {
        outcome.identity = IdentityResult::IdentityValidationFailed;
        outcome.result = "identity_validation_failed";
        outcome.detail = target.authoritativeMismatchDetail;
        return outcome;
    }

    const ValidatedWorker frozen =
        ValidateManagedWorker(target.worker, processes);
    outcome.identity = frozen.identity;
    outcome.detail = frozen.detail;
    if (frozen.identity == IdentityResult::ProcessMissing)
    {
        outcome.result = "process_missing";
        if (target.authoritativeExactTerminalDeparture)
        {
            outcome.success = true;
            outcome.detail =
                "exact_terminal_attempt_process_departed";
        }
        return outcome;
    }
    if (frozen.identity == IdentityResult::PermissionDenied)
    {
        outcome.result = "permission_failure";
        return outcome;
    }
    if (frozen.identity == IdentityResult::InspectionFailed)
    {
        outcome.result = "inspection_failed";
        return outcome;
    }
    if (frozen.identity == IdentityResult::Validated)
    {
        outcome.identity = IdentityResult::IdentityValidationFailed;
        outcome.result = "identity_validation_failed";
        outcome.detail =
            target.authoritativeMismatchDetail +
            "_while_frozen_process_remains_live";
        return outcome;
    }
    outcome.result = ToString(frozen.identity);
    return outcome;
}

SignalAuthorization ExactAttemptSignalAuthorization(
    const std::string& connectionString,
    long long requestId,
    const std::string& applicationOwner,
    const std::string& action,
    ProcessOperations& processes)
{
    return [
        connectionString,
        requestId,
        applicationOwner,
        action,
        &processes](
            const ManagedWorker& worker,
            const ProcessObservation& observation,
            int signalNumber,
            int& errorNumber,
            std::string& rejection) {
        if (!worker.workerAttemptId)
        {
            rejection = "active_worker_attempt_identity_missing";
            errorNumber = EINVAL;
            return false;
        }
        try
        {
            pqxx::connection connection{connectionString};
            pqxx::work transaction{connection};
            transaction.exec("SET TRANSACTION READ WRITE;");
            AcquireCoordinationLock(transaction);
            if (!OwnsActiveRequest(
                    transaction,
                    requestId,
                    applicationOwner,
                    action))
            {
                rejection =
                    "administrative_request_ownership_lost_before_signal";
                errorNumber = EPERM;
                transaction.commit();
                return false;
            }

            EA::SchedulerOwnership::ExactAttemptExpectation expected;
            expected.workerAttemptId = *worker.workerAttemptId;
            expected.experimentId = worker.experimentId;
            expected.checkpointEvalId = worker.checkpointEvalId;
            expected.workerKind = worker.checkpointEvalId
                ? "checkpoint_infer"
                : "experiment";
            expected.lifecyclePhase = worker.checkpointEvalId
                ? "infer"
                : worker.phase;
            expected.capacityClass = worker.checkpointEvalId
                ? "infer"
                : worker.phase;
            expected.requireSignalable = true;
            expected.requireCompleteProcessIdentity = true;
            const auto exact =
                EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
                    transaction, expected, true);
            const std::string expectedCommandIdentity =
                worker.checkpointEvalId
                    ? "checkpoint_infer:" +
                          std::to_string(*worker.checkpointEvalId)
                    : "experiment:" +
                          std::to_string(worker.experimentId) +
                          ":" + worker.phase;
            if (!exact ||
                exact->commandIdentity != expectedCommandIdentity ||
                exact->workerPid !=
                    std::optional<int>{observation.pid} ||
                exact->processGroupId !=
                    std::optional<int>{
                        observation.processGroupId} ||
                exact->processStartIdentity !=
                    std::optional<std::string>{
                        observation.processStartIdentity} ||
                exact->canonicalExecutablePath !=
                    std::optional<std::string>{
                        observation.executable} ||
                exact->commandLine !=
                    std::optional<std::string>{
                        observation.commandLine} ||
                (exact->ownershipOrigin != "legacy_unverified" &&
                 !ContainsExactOptionValue(
                     observation.commandLine,
                     "--scheduler-worker-attempt-id",
                     exact->workerAttemptId)))
            {
                rejection =
                    "exact_active_worker_attempt_verification_failed";
                errorNumber = EINVAL;
                transaction.commit();
                return false;
            }

            const bool signaled = processes.SignalProcessGroup(
                observation.processGroupId,
                signalNumber,
                errorNumber);
            if (!signaled)
                rejection = std::strerror(errorNumber);
            transaction.commit();
            return signaled;
        }
        catch (const std::exception& error)
        {
            rejection =
                std::string{
                    "exact_attempt_signal_verification_error:"} +
                error.what();
            errorNumber = EIO;
            return false;
        }
    };
}

std::optional<EA::SchedulerOwnership::ExactAttemptSnapshot>
LockExactTargetForMutation(
    pqxx::transaction_base& transaction,
    const DbTarget& target,
    bool requireCompleteProcessIdentity)
{
    if (!target.worker.workerAttemptId)
        return std::nullopt;
    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId =
        *target.worker.workerAttemptId;
    expected.experimentId = target.worker.experimentId;
    expected.checkpointEvalId = target.worker.checkpointEvalId;
    expected.workerKind = target.checkpointWorker
        ? "checkpoint_infer"
        : "experiment";
    expected.lifecyclePhase = target.checkpointWorker
        ? "infer"
        : target.worker.phase;
    expected.capacityClass = target.checkpointWorker
        ? "infer"
        : target.worker.phase;
    expected.requireCompleteProcessIdentity =
        requireCompleteProcessIdentity;
    return EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
        transaction, expected, true);
}

SignalOutcome ApplyTargetSignal(const DbTarget& target,
                                Action action,
                                std::chrono::milliseconds terminationGrace,
                                ProcessOperations& processes,
                                const std::string& connectionString,
                                long long requestId,
                                const std::string& applicationOwner)
{
    if (target.frozenReplayTarget &&
        !target.authoritativeExactMatch)
        return FrozenTargetAuthorizationFailure(target, processes);
    const SignalAuthorization authorize =
        ExactAttemptSignalAuthorization(
            connectionString,
            requestId,
            applicationOwner,
            ToString(action),
            processes);
    if (action == Action::PauseAll)
        return PauseWorkerAuthorized(
            target.worker, processes, authorize);
    if (action == Action::ResumeAll)
        return ResumeWorkerAuthorized(
            target.worker, processes, authorize);
    return CancelWorkerAuthorized(
        target.worker,
        target.resumeBeforeAction,
        terminationGrace,
        processes,
        authorize);
}

struct SelectiveAccounting
{
    std::string action;
    std::string status;
    int targetCount = 0;
    int successfulCount = 0;
    int alreadySatisfiedCount = 0;
    int missingCount = 0;
    int rejectedCount = 0;
    int failedCount = 0;
    std::string identityResult = "not_checked";
    std::string signalResult = "not_attempted";
    std::string outcomeStatus = "planned";
    std::string workerIdentity;
    std::string requestedSignal;
    std::string detail;
};

SelectiveAccounting LoadSelectiveAccounting(
    pqxx::transaction_base& transaction,
    long long requestId)
{
    const pqxx::row row = transaction.exec_params(
        "SELECT r.action,r.status,r.target_count,r.successful_count,"
        "r.already_satisfied_count,r.missing_count,r.rejected_count,"
        "r.failed_count,o.identity_result,o.signal_result,o.outcome_status,"
        "o.worker_identity,COALESCE(o.requested_signal,''),"
        "COALESCE(o.detail,'') "
        "FROM experiment_admin_request r "
        "JOIN experiment_admin_worker_outcome o ON o.request_id=r.request_id "
        "WHERE r.request_id=$1 AND o.worker_kind='experiment';",
        requestId).one_row();
    SelectiveAccounting accounting;
    accounting.action = row[0].as<std::string>();
    accounting.status = row[1].as<std::string>();
    accounting.targetCount = row[2].as<int>();
    accounting.successfulCount = row[3].as<int>();
    accounting.alreadySatisfiedCount = row[4].as<int>();
    accounting.missingCount = row[5].as<int>();
    accounting.rejectedCount = row[6].as<int>();
    accounting.failedCount = row[7].as<int>();
    accounting.identityResult = row[8].as<std::string>();
    accounting.signalResult = row[9].as<std::string>();
    accounting.outcomeStatus = row[10].as<std::string>();
    accounting.workerIdentity = row[11].as<std::string>();
    accounting.requestedSignal = row[12].as<std::string>();
    accounting.detail = row[13].as<std::string>();
    return accounting;
}

std::string SelectiveResultFromAccounting(
    const SelectiveAccounting& accounting,
    bool replay)
{
    if (accounting.status == "pending" ||
        accounting.status == "applying")
        return "pending";
    if (accounting.outcomeStatus == "failed")
    {
        if (accounting.signalResult == "signaled")
            return "stale_control_evidence";
        if (accounting.rejectedCount > 0)
            return "identity_validation_failed";
        return "signal_failed";
    }
    if (accounting.signalResult == "signaled")
        return replay ? "already_resumed"
                      : "globally_suspended_worker_resumed";
    if (accounting.signalResult == "already_requested_state")
        return "already_resumed";
    if (accounting.signalResult == "process_missing")
        return "worker_departed";
    return "signal_failed";
}

void PrintSelectiveResult(std::ostream& output,
                          long long experimentId,
                          long long requestId,
                          long long pauseRequestId,
                          const SelectiveAccounting& accounting,
                          bool replay,
                          bool signalAttempted)
{
    output << "SCHEDULER_CONTROL_RESULT,request_id=" << requestId
           << ",action=" << accounting.action
           << ",status=" << accounting.status
           << ",result="
           << SelectiveResultFromAccounting(accounting, replay)
           << ",replay=" << (replay ? 1 : 0)
           << ",signal_attempted=" << (signalAttempted ? 1 : 0)
           << ",target_count=" << accounting.targetCount
           << ",successful_count=" << accounting.successfulCount
           << ",already_satisfied_count="
           << accounting.alreadySatisfiedCount
           << ",missing_count=" << accounting.missingCount
           << ",rejected_count=" << accounting.rejectedCount
           << ",failed_count=" << accounting.failedCount
           << ",experiment_id="
           << experimentId
           << ",source_pause_request_id=" << pauseRequestId
           << ",global_state=paused"
           << "\n"
           << "SCHEDULER_CONTROL_OUTCOME,request_id=" << requestId
           << ",worker_identity=" << accounting.workerIdentity
           << ",identity_result=" << accounting.identityResult
           << ",outcome_status=" << accounting.outcomeStatus
           << ",signal_result=" << accounting.signalResult
           << ",requested_signal=" << accounting.requestedSignal
           << ",detail=" << accounting.detail << "\n";
}

std::string PersistedSignalResult(const SignalOutcome& outcome)
{
    if (outcome.result == "unsafe_process_group" ||
        outcome.result == "inspection_failed" ||
        outcome.result == "permission_denied")
    {
        if (outcome.result == "permission_denied")
            return "permission_failure";
        return "identity_validation_failed";
    }
    return outcome.result;
}

std::string PersistedRequestResult(const std::string& status)
{
    if (status == "completed")
        return "completed";
    if (status == "pending" || status == "applying")
        return "pending";
    if (status == "partial")
        return "partial";
    return "failed";
}

void TerminalizeCancellationOutcome(
    pqxx::transaction_base& transaction,
    long long requestId,
    const std::string& workerIdentity,
    const std::string& detail)
{
    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "outcome_status='partial',"
        "inference_action=CASE "
        "WHEN $1='cancellation_inference_range_missing' "
        "THEN 'no_checkpoint' "
        "WHEN outcome_status='awaiting_inference' "
        "OR inference_action IN ('queued','running','already_completed') "
        "OR EXISTS (SELECT 1 FROM experiment_admin_request r "
        "WHERE r.request_id=$2 AND r.infer_before_cancel) "
        "THEN 'failed' ELSE inference_action END,"
        "detail=$1,updated_at=now() "
        "WHERE request_id=$2 AND worker_identity=$3 "
        "AND outcome_status IN "
        "('planned','pending_checkpoint','awaiting_inference');",
        detail,
        requestId,
        workerIdentity);
}

std::string CheckpointModelEvidenceFailure(
    pqxx::transaction_base& transaction,
    long long experimentId,
    int checkpointEpoch,
    long long checkpointModelId)
{
    pqxx::result evidence = transaction.exec_params(
        "SELECT m.experiment_id,"
        "count(tm.*) FILTER (WHERE tm.param_name='train_config_meta' "
        "AND tm.row_idx=0 AND tm.col_idx=10 "
        "AND round(tm.value)::int=$1)::int "
        "FROM model m LEFT JOIN matrix tm ON tm.model_id=m.model_id "
        "WHERE m.model_id=$2 GROUP BY m.experiment_id;",
        checkpointEpoch,
        checkpointModelId);
    if (evidence.empty())
        return "cancellation_checkpoint_model_missing";
    if (evidence[0][0].as<long long>() != experimentId)
        return "cancellation_checkpoint_model_cross_experiment";
    if (evidence[0][1].as<int>() != 1)
        return "cancellation_checkpoint_epoch_metadata_invalid";
    return {};
}

struct CancellationExperimentEvidence
{
    std::string status;
    std::string phase;
    std::optional<int> currentEpoch;
    bool completed = false;
    bool cancellationCompleted = false;
    std::optional<long long> cancellationRequestId;
    std::optional<int> cancelAfterEpoch;
    std::optional<int> stopAfterEpoch;
    std::optional<int> stoppedEpoch;
    std::optional<long long> stoppedModelId;
    bool hasInferenceRange = false;
};

std::optional<CancellationExperimentEvidence>
LoadCancellationExperimentEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    pqxx::result rows = transaction.exec_params(
        "SELECT status,phase,current_epoch,completed_at,"
        "cancellation_completed_at,cancellation_request_id,"
        "cancel_after_checkpoint_epoch,stop_after_checkpoint_epoch,"
        "stopped_at_checkpoint_epoch,stopped_at_checkpoint_model_id,"
        "infer_start IS NOT NULL AND infer_end IS NOT NULL "
        "FROM experiment WHERE experiment_id=$1;",
        experimentId);
    if (rows.empty())
        return std::nullopt;
    CancellationExperimentEvidence evidence;
    evidence.status = rows[0][0].as<std::string>();
    evidence.phase = rows[0][1].as<std::string>();
    if (!rows[0][2].is_null())
        evidence.currentEpoch = rows[0][2].as<int>();
    evidence.completed = !rows[0][3].is_null();
    evidence.cancellationCompleted = !rows[0][4].is_null();
    if (!rows[0][5].is_null())
        evidence.cancellationRequestId = rows[0][5].as<long long>();
    if (!rows[0][6].is_null())
        evidence.cancelAfterEpoch = rows[0][6].as<int>();
    if (!rows[0][7].is_null())
        evidence.stopAfterEpoch = rows[0][7].as<int>();
    if (!rows[0][8].is_null())
        evidence.stoppedEpoch = rows[0][8].as<int>();
    if (!rows[0][9].is_null())
        evidence.stoppedModelId = rows[0][9].as<long long>();
    evidence.hasInferenceRange = rows[0][10].as<bool>();
    return evidence;
}

bool IsTerminalExperiment(const CancellationExperimentEvidence& evidence)
{
    return evidence.status == "completed" ||
           evidence.status == "cancelled" ||
           evidence.status == "failed";
}

std::string TerminalExperimentLifecycleFailure(
    const CancellationExperimentEvidence& evidence)
{
    if (evidence.status == "completed")
        return evidence.phase == "done" && evidence.completed
            ? std::string{}
            : "terminal_experiment_completed_lifecycle_invalid";
    if (evidence.status == "cancelled")
        return evidence.phase == "train" && evidence.completed &&
                       evidence.cancellationCompleted
            ? std::string{}
            : "terminal_experiment_cancelled_lifecycle_invalid";
    if (evidence.status == "failed")
        return evidence.phase == "done" && evidence.completed
            ? "terminal_experiment_failed_before_cancellation_reconciliation"
            : "terminal_experiment_failed_lifecycle_invalid";
    return {};
}

std::optional<std::pair<int, long long>>
RecoverIdentityFromOwnedEvaluation(
    pqxx::transaction_base& transaction,
    long long requestId,
    long long experimentId,
    const std::optional<int>& persistedEpoch,
    const std::optional<long long>& persistedModelId,
    std::string& failure)
{
    pqxx::result candidates = transaction.exec_params(
        "SELECT ce.checkpoint_epoch,ce.checkpoint_model_id "
        "FROM experiment_checkpoint_eval ce "
        "JOIN model m ON m.model_id=ce.checkpoint_model_id "
        "AND m.experiment_id=$1 "
        "WHERE ce.cancellation_request_id=$2 "
        "AND ce.parent_experiment_id=$1 AND ce.experiment_id=$1 "
        "AND ($3::integer IS NULL OR ce.checkpoint_epoch=$3) "
        "AND ($4::bigint IS NULL OR ce.checkpoint_model_id=$4) "
        "AND (SELECT count(*) FROM matrix tm "
        " WHERE tm.model_id=ce.checkpoint_model_id "
        " AND tm.param_name='train_config_meta' "
        " AND tm.row_idx=0 AND tm.col_idx=10 "
        " AND round(tm.value)::int=ce.checkpoint_epoch)=1 "
        "ORDER BY ce.checkpoint_eval_id;",
        experimentId,
        requestId,
        persistedEpoch,
        persistedModelId);
    if (candidates.size() == 1)
        return std::pair<int, long long>{
            candidates[0][0].as<int>(),
            candidates[0][1].as<long long>()};
    if (candidates.size() > 1)
    {
        failure = "legacy_cancellation_inference_identity_ambiguous";
        return std::nullopt;
    }

    pqxx::result related = transaction.exec_params(
        "SELECT "
        "count(*) FILTER (WHERE cancellation_request_id IS NOT NULL "
        "AND cancellation_request_id<>$2)::int AS foreign_owned,"
        "count(*) FILTER (WHERE cancellation_request_id IS NULL)::int "
        "AS unowned,"
        "count(*) FILTER (WHERE cancellation_request_id=$2 "
        "AND experiment_id<>$1)::int AS cross_experiment,"
        "count(*) FILTER (WHERE cancellation_request_id=$2)::int AS owned "
        "FROM experiment_checkpoint_eval "
        "WHERE parent_experiment_id=$1 "
        "AND ($3::integer IS NULL OR checkpoint_epoch=$3) "
        "AND ($4::bigint IS NULL OR checkpoint_model_id=$4);",
        experimentId,
        requestId,
        persistedEpoch,
        persistedModelId);
    if (related[0][0].as<int>() > 0)
        failure =
            "legacy_cancellation_inference_evaluation_foreign_owned";
    else if (related[0][2].as<int>() > 0)
        failure =
            "legacy_cancellation_inference_evaluation_cross_experiment";
    else if (related[0][3].as<int>() > 0)
        failure =
            "legacy_cancellation_inference_evaluation_malformed";
    else if (related[0][1].as<int>() > 0)
        failure =
            "legacy_cancellation_inference_unowned_evaluation_not_proof";
    else
        failure = "legacy_cancellation_inference_identity_unprovable";
    return std::nullopt;
}

std::optional<std::pair<int, long long>> RecoverDurableStoppedIdentity(
    pqxx::transaction_base& transaction,
    long long requestId,
    long long experimentId,
    const CancellationExperimentEvidence& evidence,
    const std::optional<int>& persistedEpoch,
    const std::optional<long long>& persistedModelId,
    std::string& failure)
{
    if (!evidence.cancellationRequestId ||
        *evidence.cancellationRequestId != requestId)
    {
        failure = "terminal_experiment_cancellation_request_mismatch";
        return std::nullopt;
    }
    if (!evidence.cancelAfterEpoch || !evidence.stopAfterEpoch ||
        !evidence.stoppedEpoch || !evidence.currentEpoch)
    {
        failure = "terminal_checkpoint_epoch_evidence_missing";
        return std::nullopt;
    }
    const int epoch = *evidence.stoppedEpoch;
    if (*evidence.cancelAfterEpoch != epoch ||
        *evidence.stopAfterEpoch != epoch ||
        *evidence.currentEpoch != epoch)
    {
        failure = "terminal_checkpoint_epoch_evidence_conflicting";
        return std::nullopt;
    }
    if (persistedEpoch && *persistedEpoch != epoch)
    {
        failure = "cancellation_checkpoint_epoch_conflict";
        return std::nullopt;
    }
    if (!evidence.stoppedModelId)
    {
        failure = "terminal_checkpoint_stopped_model_missing";
        return std::nullopt;
    }
    if (persistedModelId &&
        *persistedModelId != *evidence.stoppedModelId)
    {
        failure = "cancellation_checkpoint_model_conflict";
        return std::nullopt;
    }
    failure = CheckpointModelEvidenceFailure(
        transaction,
        experimentId,
        epoch,
        *evidence.stoppedModelId);
    if (!failure.empty())
        return std::nullopt;
    return std::pair<int, long long>{epoch, *evidence.stoppedModelId};
}

void ReconcileCancellationInferenceOutcome(
    pqxx::transaction_base& transaction,
    long long requestId,
    const std::string& workerIdentity,
    long long experimentId,
    const CancellationExperimentEvidence& experiment,
    std::optional<int> persistedEpoch,
    std::optional<long long> persistedModelId)
{
    std::string failure;
    std::optional<std::pair<int, long long>> identity;
    if (persistedEpoch && persistedModelId)
    {
        failure = CheckpointModelEvidenceFailure(
            transaction,
            experimentId,
            *persistedEpoch,
            *persistedModelId);
        if (failure.empty())
            identity = std::pair<int, long long>{
                *persistedEpoch, *persistedModelId};
    }
    else
    {
        identity = RecoverIdentityFromOwnedEvaluation(
            transaction,
            requestId,
            experimentId,
            persistedEpoch,
            persistedModelId,
            failure);
        if (!identity &&
            failure !=
                "legacy_cancellation_inference_identity_ambiguous")
        {
            std::string durableFailure;
            const auto durable = RecoverDurableStoppedIdentity(
                transaction,
                requestId,
                experimentId,
                experiment,
                persistedEpoch,
                persistedModelId,
                durableFailure);
            if (durable)
            {
                identity = durable;
                failure.clear();
            }
            else if (failure ==
                     "legacy_cancellation_inference_identity_unprovable")
                failure = durableFailure;
        }
    }
    if (!identity)
    {
        TerminalizeCancellationOutcome(
            transaction, requestId, workerIdentity, failure);
        return;
    }

    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "cancellation_checkpoint_epoch=COALESCE("
        "cancellation_checkpoint_epoch,$1),"
        "cancellation_checkpoint_model_id=COALESCE("
        "cancellation_checkpoint_model_id,$2),updated_at=now() "
        "WHERE request_id=$3 AND worker_identity=$4 "
        "AND (cancellation_checkpoint_epoch IS NULL "
        "OR cancellation_checkpoint_model_id IS NULL) "
        "AND (cancellation_checkpoint_epoch IS NULL "
        "OR cancellation_checkpoint_epoch=$1) "
        "AND (cancellation_checkpoint_model_id IS NULL "
        "OR cancellation_checkpoint_model_id=$2);",
        identity->first,
        identity->second,
        requestId,
        workerIdentity);

    DbTarget target;
    target.worker.experimentId = experimentId;
    target.latestCheckpointEpoch = identity->first;
    target.latestCheckpointModelId = identity->second;
    target.hasInferenceRange = experiment.hasInferenceRange;
    const std::optional<long long> evaluationId =
        QueueCancellationInference(transaction, requestId, target);
    if (!evaluationId || target.inferenceFailed)
    {
        TerminalizeCancellationOutcome(
            transaction,
            requestId,
            workerIdentity,
            target.inferenceFailureDetail.empty()
                ? "cancellation_checkpoint_evaluation_materialization_invalid"
                : target.inferenceFailureDetail);
        return;
    }

    const std::string inferenceAction =
        target.inferenceAlreadyCompleted
            ? "completed"
            : (target.inferenceAlreadyRunning ? "running" : "queued");
    const std::string outcomeStatus =
        target.inferenceAlreadyCompleted ? "completed" : "awaiting_inference";
    const std::string detail =
        target.inferenceAlreadyCompleted
            ? "cancellation_checkpoint_inference_completed"
            : "cancellation_checkpoint_reached";
    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "inference_action=$1,outcome_status=$2,detail=$3,updated_at=now() "
        "WHERE request_id=$4 AND worker_identity=$5 "
        "AND outcome_status IN "
        "('planned','pending_checkpoint','awaiting_inference') "
        "AND (inference_action IS DISTINCT FROM $1 "
        "OR outcome_status IS DISTINCT FROM $2 "
        "OR detail IS DISTINCT FROM $3);",
        inferenceAction,
        outcomeStatus,
        detail,
        requestId,
        workerIdentity);
}

} // namespace

int RequestExitCodeForPersistedStatus(const std::string& status)
{
    return status == "completed" || status == "pending" ? 0 : 1;
}

const char* ToString(Action action)
{
    switch (action)
    {
        case Action::PauseAll: return "pause_all";
        case Action::ResumeAll: return "resume_all";
        case Action::CancelAll: return "cancel_all";
    }
    return "unknown";
}

const char* ToString(CancellationMode mode)
{
    return mode == CancellationMode::Immediate
        ? "immediate"
        : "after_next_checkpoint";
}

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

std::optional<std::string> ValidateCommand(const Command& command)
{
    if (command.action == Action::CancelAll)
    {
        if (!command.cancellationMode)
            return "cancellation requires exactly one of --immediate or "
                   "--after-next-checkpoint";
    }
    else
    {
        if (command.cancellationMode)
            return "cancellation modes require --cancel-all-experiments";
        if (command.inferBeforeCancel)
            return "--infer-before-cancel requires --cancel-all-experiments";
    }
    if (!command.dryRun && !command.confirmed)
        return "administrative write requires --yes (or use --dry-run)";
    if (command.terminationGrace.count() <= 0)
        return "termination grace must be positive";
    return std::nullopt;
}

std::optional<int> NextCancellationCheckpoint(
    int currentEpoch,
    int checkpointInterval,
    int targetEpochs,
    std::optional<int> latestDurableCheckpointEpoch)
{
    if (currentEpoch < 0 || checkpointInterval <= 0 || targetEpochs <= 0)
        return std::nullopt;
    if (latestDurableCheckpointEpoch &&
        *latestDurableCheckpointEpoch == currentEpoch &&
        currentEpoch > 0 &&
        currentEpoch % checkpointInterval == 0 &&
        currentEpoch < targetEpochs)
        return currentEpoch;
    const int next =
        ((currentEpoch / checkpointInterval) + 1) * checkpointInterval;
    return next < targetEpochs ? std::optional<int>{next} : std::nullopt;
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

std::unique_ptr<ProcessOperations> CreateNativeProcessOperations()
{
    return std::make_unique<PosixProcessOperations>();
}

std::unique_ptr<ProcessOperations> CreateNativeProcessOperationsForTesting(
    std::unique_ptr<NativeProcessObservationBackend> backend)
{
    return std::make_unique<PosixProcessOperations>(std::move(backend));
}

ValidatedWorker ValidateManagedWorker(const ManagedWorker& worker,
                                      ProcessOperations& processes)
{
    ValidatedWorker result;
    result.worker = worker;
    if (worker.lifecycleStatus != "running" || worker.pid <= 0)
    {
        result.identity = IdentityResult::StalePid;
        result.detail = "database_row_is_not_an_active_managed_worker";
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
    result.identity = IdentityResult::Validated;
    result.detail = "validated";
    return result;
}

std::vector<SchedulerWorkerClassification> ClassifySchedulerWorkers(
    const std::vector<SchedulerWorkerCandidate>& candidates,
    const std::vector<ManagedWorker>& authoritativeWorkers,
    ProcessOperations& processes)
{
    std::vector<SchedulerWorkerClassification> classifications;
    classifications.reserve(candidates.size());

    for (const SchedulerWorkerCandidate& candidate : candidates)
    {
        SchedulerWorkerClassification classification;
        classification.pid = candidate.pid;
        classification.kind = candidate.kind;
        classification.cpuPercent = candidate.cpuPercent;
        classification.memPercent = candidate.memPercent;
        classification.rssMb = candidate.rssMb;

        const bool checkpointTagged =
            candidate.commandLine.find("--scheduler-checkpoint-eval-id") !=
            std::string::npos;
        std::vector<const ManagedWorker*> matches;
        for (const ManagedWorker& worker : authoritativeWorkers)
        {
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
            matches.push_back(&worker);
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

        const ManagedWorker& worker = *matches.front();
        classification.experimentId = worker.experimentId;
        classification.checkpointEvalId = worker.checkpointEvalId;
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

        const ValidatedWorker validated =
            ValidateManagedWorker(worker, processes);
        classification.identity = validated.identity;
        classification.managed =
            validated.identity == IdentityResult::Validated;
        classification.reason = classification.managed
            ? "validated"
            : validated.detail;
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
        if (classification.kind == "train")
        {
            aggregate = classification.managed
                ? &summary.managedTrain
                : &summary.unmanagedTrain;
        }
        else if (classification.kind == "infer")
        {
            aggregate = classification.managed
                ? &summary.managedInfer
                : &summary.unmanagedInfer;
        }
        else if (classification.kind == "analyze")
        {
            aggregate = classification.managed
                ? &summary.managedAnalyze
                : &summary.unmanagedAnalyze;
        }
        if (aggregate == nullptr)
            continue;
        ++aggregate->workers;
        aggregate->cpuPercent += classification.cpuPercent;
        aggregate->memPercent += classification.memPercent;
        aggregate->rssMb += classification.rssMb;
    }
    return summary;
}

namespace
{

bool SendAuthorizedSignal(
    const ManagedWorker& worker,
    const ProcessObservation& observation,
    int signalNumber,
    ProcessOperations& processes,
    const SignalAuthorization& authorize,
    int& errorNumber,
    std::string& rejection)
{
    if (authorize)
    {
        return authorize(
            worker,
            observation,
            signalNumber,
            errorNumber,
            rejection);
    }
    return processes.SignalProcessGroup(
        observation.processGroupId,
        signalNumber,
        errorNumber);
}

SignalOutcome PauseWorkerAuthorized(
    const ManagedWorker& worker,
    ProcessOperations& processes,
    const SignalAuthorization& authorize)
{
    SignalOutcome outcome;
    const ValidatedWorker validated = ValidateManagedWorker(worker, processes);
    outcome.identity = validated.identity;
    outcome.detail = validated.detail;
    if (validated.identity != IdentityResult::Validated)
    {
        outcome.result = validated.identity == IdentityResult::ProcessMissing
            ? "process_missing"
            : (validated.identity == IdentityResult::PermissionDenied
                   ? "permission_failure"
                   : ToString(validated.identity));
        return outcome;
    }
    if (validated.observation.stopped)
    {
        outcome.result = "already_requested_state";
        outcome.success = true;
        return outcome;
    }
    int errorNumber = 0;
    std::string rejection;
    outcome.signals.push_back(SIGSTOP);
    if (!SendAuthorizedSignal(
            worker,
            validated.observation,
            SIGSTOP,
            processes,
            authorize,
            errorNumber,
            rejection))
    {
        if (errorNumber == ESRCH)
        {
            outcome.identity = IdentityResult::ProcessMissing;
            outcome.result = "process_missing";
            outcome.detail = "process_group_disappeared_before_sigstop";
            return outcome;
        }
        outcome.result = errorNumber == EPERM
            ? "permission_failure"
            : (errorNumber == EINVAL
                   ? "identity_validation_failed"
                   : "signaling_failure");
        outcome.detail = rejection.empty()
            ? std::strerror(errorNumber)
            : rejection;
        return outcome;
    }
    outcome.result = "signaled";
    outcome.success = true;
    return outcome;
}

SignalOutcome ResumeWorkerAuthorized(
    const ManagedWorker& worker,
    ProcessOperations& processes,
    const SignalAuthorization& authorize)
{
    SignalOutcome outcome;
    const ValidatedWorker validated = ValidateManagedWorker(worker, processes);
    outcome.identity = validated.identity;
    outcome.detail = validated.detail;
    if (validated.identity != IdentityResult::Validated)
    {
        outcome.result = validated.identity == IdentityResult::ProcessMissing
            ? "process_missing"
            : (validated.identity == IdentityResult::PermissionDenied
                   ? "permission_failure"
                   : ToString(validated.identity));
        return outcome;
    }
    if (!validated.observation.stopped)
    {
        outcome.result = "already_requested_state";
        outcome.success = true;
        return outcome;
    }
    int errorNumber = 0;
    std::string rejection;
    outcome.signals.push_back(SIGCONT);
    if (!SendAuthorizedSignal(
            worker,
            validated.observation,
            SIGCONT,
            processes,
            authorize,
            errorNumber,
            rejection))
    {
        if (errorNumber == ESRCH)
        {
            outcome.identity = IdentityResult::ProcessMissing;
            outcome.result = "process_missing";
            outcome.detail = "process_group_disappeared_before_sigcont";
            return outcome;
        }
        outcome.result = errorNumber == EPERM
            ? "permission_failure"
            : (errorNumber == EINVAL
                   ? "identity_validation_failed"
                   : "signaling_failure");
        outcome.detail = rejection.empty()
            ? std::strerror(errorNumber)
            : rejection;
        return outcome;
    }
    outcome.result = "signaled";
    outcome.success = true;
    return outcome;
}

SignalOutcome CancelWorkerAuthorized(
    const ManagedWorker& worker,
    bool resumeFirst,
    std::chrono::milliseconds grace,
    ProcessOperations& processes,
    const SignalAuthorization& authorize)
{
    SignalOutcome outcome;
    ValidatedWorker validated = ValidateManagedWorker(worker, processes);
    outcome.identity = validated.identity;
    outcome.detail = validated.detail;
    if (validated.identity != IdentityResult::Validated)
    {
        outcome.result = validated.identity == IdentityResult::ProcessMissing
            ? "process_missing"
            : (validated.identity == IdentityResult::PermissionDenied
                   ? "permission_failure"
                   : ToString(validated.identity));
        return outcome;
    }
    int errorNumber = 0;
    std::string rejection;
    if (resumeFirst)
    {
        outcome.signals.push_back(SIGCONT);
        if (!SendAuthorizedSignal(
                worker,
                validated.observation,
                SIGCONT,
                processes,
                authorize,
                errorNumber,
                rejection))
        {
            if (errorNumber == ESRCH)
            {
                outcome.identity = IdentityResult::ProcessMissing;
                outcome.result = "process_missing";
                outcome.detail = "process_group_disappeared_before_sigcont";
                return outcome;
            }
            outcome.result = errorNumber == EPERM
                ? "permission_failure"
                : (errorNumber == EINVAL
                       ? "identity_validation_failed"
                       : "signaling_failure");
            outcome.detail = rejection.empty()
                ? std::strerror(errorNumber)
                : rejection;
            return outcome;
        }
    }
    validated = ValidateManagedWorker(worker, processes);
    if (validated.identity != IdentityResult::Validated)
    {
        outcome.identity = validated.identity;
        outcome.result = validated.identity == IdentityResult::ProcessMissing
            ? "process_missing"
            : (validated.identity == IdentityResult::PermissionDenied
                   ? "permission_failure"
                   : ToString(validated.identity));
        outcome.detail = validated.detail;
        return outcome;
    }
    outcome.signals.push_back(SIGTERM);
    rejection.clear();
    if (!SendAuthorizedSignal(
            worker,
            validated.observation,
            SIGTERM,
            processes,
            authorize,
            errorNumber,
            rejection))
    {
        if (errorNumber == ESRCH)
        {
            outcome.identity = IdentityResult::ProcessMissing;
            outcome.result = "process_missing";
            outcome.detail = "process_group_disappeared_before_sigterm";
            return outcome;
        }
        outcome.result = errorNumber == EPERM
            ? "permission_failure"
            : (errorNumber == EINVAL
                   ? "identity_validation_failed"
                   : "signaling_failure");
        outcome.detail = rejection.empty()
            ? std::strerror(errorNumber)
            : rejection;
        return outcome;
    }
    if (processes.WaitForProcessGroupExit(
            validated.observation.processGroupId, grace))
    {
        outcome.result = "signaled";
        outcome.success = true;
        return outcome;
    }
    // Validate every persisted identity component again immediately before
    // destructive escalation. A missing leader makes the group unsafe to
    // signal because its identity can no longer be proven.
    const ValidatedWorker beforeKill =
        ValidateManagedWorker(worker, processes);
    if (beforeKill.identity != IdentityResult::Validated)
    {
        outcome.identity = beforeKill.identity;
        outcome.result = beforeKill.identity == IdentityResult::ProcessMissing
            ? "process_missing"
            : (beforeKill.identity == IdentityResult::PermissionDenied
                   ? "permission_failure"
                   : ToString(beforeKill.identity));
        outcome.detail = beforeKill.detail;
        return outcome;
    }
    outcome.signals.push_back(SIGKILL);
    rejection.clear();
    if (!SendAuthorizedSignal(
            worker,
            beforeKill.observation,
            SIGKILL,
            processes,
            authorize,
            errorNumber,
            rejection))
    {
        outcome.result = errorNumber == EPERM
            ? "permission_failure"
            : (errorNumber == EINVAL
                   ? "identity_validation_failed"
                   : "signaling_failure");
        outcome.detail = rejection.empty()
            ? std::strerror(errorNumber)
            : rejection;
        return outcome;
    }
    outcome.result = "escalated";
    outcome.success = processes.WaitForProcessGroupExit(
        beforeKill.observation.processGroupId, grace);
    if (!outcome.success)
        outcome.detail = "process_still_exists_after_sigkill";
    return outcome;
}

} // namespace

SignalOutcome PauseWorker(const ManagedWorker& worker,
                          ProcessOperations& processes)
{
    return PauseWorkerAuthorized(
        worker, processes, SignalAuthorization{});
}

SignalOutcome ResumeWorker(const ManagedWorker& worker,
                           ProcessOperations& processes)
{
    return ResumeWorkerAuthorized(
        worker, processes, SignalAuthorization{});
}

SignalOutcome CancelWorker(const ManagedWorker& worker,
                           bool resumeFirst,
                           std::chrono::milliseconds grace,
                           ProcessOperations& processes)
{
    return CancelWorkerAuthorized(
        worker,
        resumeFirst,
        grace,
        processes,
        SignalAuthorization{});
}

namespace
{

bool IsReconcilableExperimentWorker(const std::string& workerKind,
                                   const std::string& phase,
                                   const std::string& capacityClass)
{
    return workerKind == "experiment" &&
           (phase == "train" || phase == "infer" || phase == "analyze") &&
           capacityClass == phase;
}

void PrintWorkerAttemptReconciliationResult(
    std::ostream& output,
    const char* outcome,
    const EA::SchedulerOwnership::ExactAttemptSnapshot* exact,
    const ValidatedWorker* validated,
    const std::string& proposedState,
    const std::string& reason)
{
    const bool eligible = std::string{outcome} == "eligible" ||
                          std::string{outcome} == "applied";
    output << "WORKER_ATTEMPT_RECONCILIATION"
           << ",outcome=" << outcome
           << ",worker_attempt_id="
           << (exact ? std::to_string(exact->workerAttemptId) : "unknown")
           << ",experiment_id="
           << (exact ? std::to_string(exact->experimentId) : "unknown")
           << ",worker_kind=" << (exact ? exact->workerKind : "unknown")
           << ",phase=" << (exact ? exact->lifecyclePhase : "unknown")
           << ",current_lifecycle_state="
           << (exact ? exact->lifecycleState : "unknown")
           << ",proposed_lifecycle_state=" << proposedState
           << ",durable_pid=" << (exact && exact->workerPid
               ? std::to_string(*exact->workerPid) : "NULL")
           << ",durable_pgid=" << (exact && exact->processGroupId
               ? std::to_string(*exact->processGroupId) : "NULL")
           << ",process_presence=" << (validated
               ? (validated->identity == IdentityResult::ProcessMissing
                      ? "absent"
                      : (validated->observation.exists ? "live" : "unknown"))
               : "unknown")
           << ",live_pid=" << (validated
               ? (validated->observation.exists
                      ? std::to_string(validated->observation.pid)
                      : "NULL")
               : "NULL")
           << ",live_pgid=" << (validated
               ? (validated->observation.exists
                      ? std::to_string(validated->observation.processGroupId)
                      : "NULL")
               : "NULL")
           << ",process_start_identity_result="
           << (validated && validated->identity == IdentityResult::ProcessMissing
               ? "not_required_absent"
               : (validated && validated->identity == IdentityResult::Validated
                      ? "matched" : "not_matched"))
           << ",executable_identity_result="
           << (validated && validated->identity == IdentityResult::ProcessMissing
               ? "not_required_absent"
               : (validated && validated->identity == IdentityResult::Validated
                      ? "matched" : "not_matched"))
           << ",command_identity_result="
           << (validated && validated->identity == IdentityResult::ProcessMissing
               ? "not_required_absent"
               : (validated && validated->identity == IdentityResult::Validated
                      ? "matched" : "not_matched"))
           << ",lifecycle_binding_result="
           << (exact ? "matched" : "not_matched")
           << ",eligible=" << (eligible ? "true" : "false")
           << ",reason=" << reason << std::endl;
}

int RunWorkerAttemptReconciliationCommandImpl(
    const std::string& connectionString,
    const WorkerAttemptReconciliationCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes)
{
    if (command.workerAttemptId <= 0)
    {
        error << "WORKER_ATTEMPT_RECONCILIATION_REJECTED"
              << ",reason=worker_attempt_id_required" << std::endl;
        return 2;
    }
    if (!command.dryRun && !command.confirmed)
    {
        error << "WORKER_ATTEMPT_RECONCILIATION_REJECTED"
              << ",worker_attempt_id=" << command.workerAttemptId
              << ",reason=administrative_write_requires_yes" << std::endl;
        return 2;
    }

    try
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        EA::SchedulerOwnership::SetCorrectedSchedulerProtocolSession(transaction);

        // Coordinate with scheduler transactions without claiming or
        // refreshing scheduler ownership. Historical attempt fence identity
        // remains an exact predicate; this command cannot dispatch work.
        AcquireCoordinationLock(transaction);
        const pqxx::result candidate = transaction.exec_params(
            "SELECT worker_attempt_id,experiment_id,checkpoint_eval_id,"
            "worker_kind,lifecycle_phase,capacity_class,"
            "scheduler_invocation_id,scheduler_fencing_token "
            "FROM experiment_scheduler_worker_attempt "
            "WHERE worker_attempt_id=$1 FOR UPDATE;",
            command.workerAttemptId);
        if (candidate.size() != 1)
        {
            PrintWorkerAttemptReconciliationResult(
                output, "rejected", nullptr, nullptr, "observed",
                "worker_attempt_not_found");
            transaction.commit();
            return 1;
        }

        EA::SchedulerOwnership::ExactAttemptExpectation expected;
        expected.workerAttemptId = candidate[0][0].as<long long>();
        expected.experimentId = candidate[0][1].as<long long>();
        if (!candidate[0][2].is_null())
            expected.checkpointEvalId = candidate[0][2].as<long long>();
        expected.workerKind = candidate[0][3].as<std::string>();
        expected.lifecyclePhase = candidate[0][4].as<std::string>();
        expected.capacityClass = candidate[0][5].as<std::string>();
        if (!candidate[0][6].is_null())
            expected.schedulerInvocationId = candidate[0][6].as<std::string>();
        if (!candidate[0][7].is_null())
            expected.schedulerFencingToken = candidate[0][7].as<long long>();
        expected.requiredLifecycleState = "identity_ambiguous";
        expected.requireCompleteProcessIdentity = true;

        const auto exact = EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
            transaction, expected, true);
        if (!exact || exact->checkpointEvalId ||
            !IsReconcilableExperimentWorker(
                expected.workerKind, expected.lifecyclePhase,
                expected.capacityClass) ||
            exact->commandIdentity !=
                "experiment:" + std::to_string(expected.experimentId) +
                    ":" + expected.lifecyclePhase)
        {
            PrintWorkerAttemptReconciliationResult(
                output, "rejected", exact ? &*exact : nullptr, nullptr,
                "observed", "exact_ambiguous_attempt_verification_failed");
            transaction.commit();
            return 1;
        }

        ManagedWorker worker;
        worker.workerAttemptId = exact->workerAttemptId;
        worker.workerKind = exact->workerKind;
        worker.capacityClass = exact->capacityClass;
        worker.attemptLifecycleState = exact->lifecycleState;
        worker.launchAttemptIdentity = exact->launchAttemptIdentity;
        worker.experimentId = exact->experimentId;
        worker.phase = exact->lifecyclePhase;
        worker.lifecycleStatus = exact->lifecycleStatus;
        worker.pid = *exact->workerPid;
        worker.processGroupId = exact->processGroupId;
        worker.processStartIdentity = exact->processStartIdentity;
        worker.executable = exact->canonicalExecutablePath;
        worker.commandLine = exact->commandLine;

        // Observe immediately before the locked guarded transition. This
        // command never calls any ProcessOperations signalling method.
        const ValidatedWorker validated = ValidateManagedWorker(worker, processes);
        if (validated.identity == IdentityResult::ProcessMissing &&
            validated.observation.inspectionSucceeded &&
            !validated.observation.permissionDenied)
        {
            // ProcessMissing is admitted only after Observe positively proved
            // absence (inspectionSucceeded=true). All other incomplete or
            // uncertain observations remain on the fail-closed live path
            // below. The existing scheduler orphan contract defines the
            // no-result terminal outcome for this exact case.
            constexpr const char* kAbsentReason =
                "exact_process_absent_no_result";
            PrintWorkerAttemptReconciliationResult(
                output, command.dryRun ? "eligible" : "applying", &*exact,
                &validated, "failed", kAbsentReason);
            if (command.dryRun)
            {
                transaction.commit();
                return 0;
            }

            const pqxx::result terminalized = transaction.exec_params(
                "UPDATE experiment_scheduler_worker_attempt a SET "
                "lifecycle_state='failed',"
                "completed_at=clock_timestamp(),"
                "last_observed_at=clock_timestamp(),"
                "exit_code=-1,"
                "reconciliation_result='process_missing_no_result',"
                "diagnostic='exact_process_identity_absent_no_result' "
                "WHERE a.worker_attempt_id=$1 "
                "AND a.experiment_id=$2 AND a.checkpoint_eval_id IS NULL "
                "AND a.worker_kind=$3 AND a.lifecycle_phase=$4 "
                "AND a.capacity_class=$5 "
                "AND a.lifecycle_state='identity_ambiguous' "
                "AND a.scheduler_invocation_id IS NOT DISTINCT FROM $6 "
                "AND a.scheduler_fencing_token IS NOT DISTINCT FROM $7 "
                "AND a.worker_pid=$8 AND a.worker_process_group_id=$9 "
                "AND a.worker_process_start_identity=$10 "
                "AND a.canonical_executable_path=$11 "
                "AND a.command_line=$12 AND a.command_identity=$13 "
                "AND EXISTS ("
                " SELECT 1 FROM experiment e "
                " WHERE e.experiment_id=$2 "
                " AND e.active_scheduler_worker_attempt_id=$1 "
                " AND e.status='running' AND e.phase=$4 "
                " AND e.worker_pid=$8 "
                " AND e.worker_process_group_id=$9 "
                " AND e.worker_process_start_identity=$10 "
                " AND e.worker_executable=$11 "
                " AND e.worker_command_line=$12) "
                "RETURNING a.worker_attempt_id;",
                exact->workerAttemptId, exact->experimentId,
                exact->workerKind, exact->lifecyclePhase,
                exact->capacityClass, exact->schedulerInvocationId,
                exact->schedulerFencingToken, *exact->workerPid,
                *exact->processGroupId, *exact->processStartIdentity,
                *exact->canonicalExecutablePath, *exact->commandLine,
                exact->commandIdentity);
            EA::SchedulerOwnership::RequireAffectedExactlyOne(
                terminalized,
                "terminalize_exact_absent_worker_attempt");

            const pqxx::result failedLifecycle = transaction.exec_params(
                "UPDATE experiment SET status='failed',"
                "worker_pid=NULL,worker_process_group_id=NULL,"
                "completed_at=clock_timestamp(),exit_code=-1,"
                "error_message='worker_process_missing_no_result',"
                "active_scheduler_worker_attempt_id=NULL,"
                "updated_at=clock_timestamp() "
                "WHERE experiment_id=$1 "
                "AND active_scheduler_worker_attempt_id=$2 "
                "AND status='running' AND phase=$3 "
                "AND worker_pid=$4 "
                "AND worker_process_group_id=$5 "
                "AND worker_process_start_identity=$6 "
                "AND worker_executable=$7 "
                "AND worker_command_line=$8 "
                "RETURNING experiment_id;",
                exact->experimentId, exact->workerAttemptId,
                exact->lifecyclePhase, *exact->workerPid,
                *exact->processGroupId, *exact->processStartIdentity,
                *exact->canonicalExecutablePath, *exact->commandLine);
            EA::SchedulerOwnership::RequireAffectedExactlyOne(
                failedLifecycle,
                "clear_exact_absent_worker_lifecycle_binding");
            transaction.commit();
            PrintWorkerAttemptReconciliationResult(
                output, "applied", &*exact, &validated, "failed",
                kAbsentReason);
            return 0;
        }
        if (validated.identity != IdentityResult::Validated)
        {
            PrintWorkerAttemptReconciliationResult(
                output, "rejected", &*exact, &validated, "observed",
                "live_process_identity_verification_failed:" + validated.detail);
            transaction.commit();
            return 1;
        }

        PrintWorkerAttemptReconciliationResult(
            output, command.dryRun ? "eligible" : "applying", &*exact,
            &validated, "observed", "exact_identity_verified");
        if (command.dryRun)
        {
            transaction.commit();
            return 0;
        }

        const pqxx::result transitioned = transaction.exec_params(
            "UPDATE experiment_scheduler_worker_attempt a SET "
            "lifecycle_state='observed',"
            "reconciliation_result='valid_process_observed',"
            "diagnostic='exact_attempt_reconciled_by_administrator',"
            "last_observed_at=clock_timestamp() "
            "FROM experiment e "
            "WHERE a.worker_attempt_id=$1 AND a.lifecycle_state='identity_ambiguous' "
            "AND a.experiment_id=$2 AND a.checkpoint_eval_id IS NULL "
            "AND a.worker_kind=$3 AND a.lifecycle_phase=$4 AND a.capacity_class=$5 "
            "AND a.scheduler_invocation_id IS NOT DISTINCT FROM $6 "
            "AND a.scheduler_fencing_token IS NOT DISTINCT FROM $7 "
            "AND a.worker_pid=$8 AND a.worker_process_group_id=$9 "
            "AND a.worker_process_start_identity=$10 "
            "AND a.canonical_executable_path=$11 AND a.command_line=$12 "
            "AND a.command_identity=$13 "
            "AND e.experiment_id=a.experiment_id "
            "AND e.active_scheduler_worker_attempt_id=a.worker_attempt_id "
            "AND e.status='running' AND e.phase=$4 "
            "AND e.worker_pid=a.worker_pid "
            "AND e.worker_process_group_id=a.worker_process_group_id "
            "AND e.worker_process_start_identity=a.worker_process_start_identity "
            "AND e.worker_executable=a.canonical_executable_path "
            "AND e.worker_command_line=a.command_line "
            "RETURNING a.worker_attempt_id;",
            exact->workerAttemptId, exact->experimentId, exact->workerKind,
            exact->lifecyclePhase, exact->capacityClass,
            exact->schedulerInvocationId, exact->schedulerFencingToken,
            *exact->workerPid, *exact->processGroupId,
            *exact->processStartIdentity, *exact->canonicalExecutablePath,
            *exact->commandLine, exact->commandIdentity);
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            transitioned, "reconcile_exact_identity_ambiguous_attempt");
        // The applied outcome is a post-commit observation.  If commit fails,
        // control transfers to the error path and no applied result is
        // emitted; the transaction destructor then aborts the mutation.
        transaction.commit();
        PrintWorkerAttemptReconciliationResult(
            output, "applied", &*exact, &validated, "observed",
            "exact_identity_verified");
        return 0;
    }
    catch (const std::exception& exception)
    {
        error << "WORKER_ATTEMPT_RECONCILIATION_ERROR"
              << ",worker_attempt_id=" << command.workerAttemptId
              << ",error=" << exception.what() << std::endl;
        return 2;
    }
}

} // namespace

int RunWorkerAttemptReconciliationCommand(
    const std::string& connectionString,
    const WorkerAttemptReconciliationCommand& command,
    std::ostream& output,
    std::ostream& error)
{
    std::unique_ptr<ProcessOperations> processes = CreateNativeProcessOperations();
    return RunWorkerAttemptReconciliationCommandImpl(
        connectionString, command, output, error, *processes);
}

int RunWorkerAttemptReconciliationCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const WorkerAttemptReconciliationCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes)
{
    return RunWorkerAttemptReconciliationCommandImpl(
        connectionString, command, output, error, processes);
}

void AcquireCoordinationLock(pqxx::transaction_base& transaction)
{
    transaction.exec_params(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,0));",
        kCoordinationLockName);
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

bool NormalSchedulingAllowed(const ControlSnapshot& snapshot)
{
    return snapshot.desiredState == "running" &&
           !snapshot.activeRequestId.has_value();
}

bool CancellationInferenceAllowed(const ControlSnapshot& snapshot)
{
    return snapshot.activeRequestId.has_value() &&
           snapshot.activeAction == "cancel_all" &&
           snapshot.inferBeforeCancel;
}

bool CancellationCheckpointTrainAllowed(const ControlSnapshot& snapshot)
{
    return snapshot.activeRequestId.has_value() &&
           snapshot.activeAction == "cancel_all" &&
           snapshot.cancellationMode == "after_next_checkpoint";
}

CheckpointStopRecordResult RecordCheckpointStopReached(
    pqxx::work& transaction,
    const std::optional<long long>& experimentId,
    long long workerAttemptId,
    int epoch,
    long long modelId)
{
    CheckpointStopRecordResult result;
    if (!experimentId)
    {
        result.detail = "experiment_id_missing";
        return result;
    }
    if (workerAttemptId <= 0)
    {
        result.detail = "worker_attempt_id_missing";
        return result;
    }

    AcquireCoordinationLock(transaction);

    EA::SchedulerOwnership::ExactAttemptExpectation expected;
    expected.workerAttemptId = workerAttemptId;
    expected.experimentId = *experimentId;
    expected.workerKind = "experiment";
    expected.lifecyclePhase = "train";
    expected.capacityClass = "train";
    expected.requireCompleteProcessIdentity = true;

    // A replay deliberately does not require the lifecycle to remain bound to
    // the old attempt: the atomic transition cleared that binding and a later
    // phase may already own a replacement attempt.
    if (const auto terminal =
            EA::SchedulerOwnership::LockAndVerifyExactTerminalAttempt(
                transaction,
                expected,
                "completed",
                "checkpoint_stop_completed",
                true))
    {
        const pqxx::result replay = transaction.exec_params(
            "SELECT cancellation_request_id,cancel_infer_before,status,phase,"
            "stopped_at_checkpoint_epoch,stopped_at_checkpoint_model_id "
            "FROM experiment WHERE experiment_id=$1 FOR UPDATE;",
            *experimentId);
        const std::string epochEvidence =
            "checkpoint_epoch=" + std::to_string(epoch);
        const std::string modelEvidence =
            "checkpoint_model_id=" + std::to_string(modelId);
        const std::string attemptEvidence =
            "worker_attempt_id=" + std::to_string(workerAttemptId);
        if (replay.size() != 1 ||
            replay[0][4].is_null() ||
            replay[0][4].as<int>() != epoch ||
            replay[0][5].is_null() ||
            replay[0][5].as<long long>() != modelId ||
            terminal->diagnostic.find(epochEvidence) == std::string::npos ||
            terminal->diagnostic.find(modelEvidence) == std::string::npos ||
            terminal->diagnostic.find(attemptEvidence) == std::string::npos)
        {
            result.detail = "checkpoint_stop_terminal_replay_mismatch";
            return result;
        }
        result.recorded = true;
        result.workerAttemptId = workerAttemptId;
        result.cancellationRequested = !replay[0][0].is_null();
        result.inferenceRequested =
            result.cancellationRequested && replay[0][1].as<bool>();
        if (result.cancellationRequested)
            result.cancellationRequestId =
                replay[0][0].as<long long>();
        result.detail = result.cancellationRequested
            ? "cancellation_checkpoint_replay"
            : "checkpoint_stop_replay";
        return result;
    }

    const auto exact =
        EA::SchedulerOwnership::LockAndVerifyExactActiveAttempt(
            transaction, expected, true);
    const std::optional<std::string> currentStartIdentity =
        ReadProcessStartIdentity(static_cast<int>(::getpid()));
    if (!exact ||
        exact->workerPid !=
            std::optional<int>{static_cast<int>(::getpid())} ||
        exact->processGroupId !=
            std::optional<int>{static_cast<int>(::getpgrp())} ||
        exact->processStartIdentity != currentStartIdentity ||
        exact->commandIdentity !=
            "experiment:" + std::to_string(*experimentId) + ":train")
    {
        result.detail = "exact_active_train_attempt_mismatch";
        return result;
    }

    pqxx::result experiments = transaction.exec_params(
        "SELECT cancellation_request_id,cancel_infer_before,"
        "infer_start IS NOT NULL AND infer_end IS NOT NULL,status,phase,"
        "stopped_at_checkpoint_epoch,stopped_at_checkpoint_model_id,"
        "EXISTS (SELECT 1 FROM experiment_admin_request r "
        "JOIN experiment_global_control c "
        "ON c.active_request_id=r.request_id "
        "WHERE c.singleton AND r.request_id=experiment.cancellation_request_id "
        "AND r.action='cancel_all'),"
        "active_scheduler_worker_attempt_id,"
        "last_checkpoint_stop_decision_epoch,stop_after_checkpoint_epoch "
        "FROM experiment WHERE experiment_id=$1 FOR UPDATE;",
        *experimentId);
    if (experiments.empty())
    {
        result.detail = "experiment_missing";
        return result;
    }

    if (experiments[0][3].as<std::string>() != "running" ||
        experiments[0][4].as<std::string>() != "train" ||
        experiments[0][8].is_null() ||
        experiments[0][8].as<long long>() != workerAttemptId ||
        experiments[0][9].is_null() ||
        experiments[0][9].as<int>() != epoch ||
        experiments[0][10].is_null() ||
        experiments[0][10].as<int>() > epoch)
    {
        result.detail = "checkpoint_stop_decision_or_lifecycle_mismatch";
        return result;
    }

    const std::string modelFailure = CheckpointModelEvidenceFailure(
        transaction, *experimentId, epoch, modelId);
    if (!modelFailure.empty())
    {
        result.detail = modelFailure;
        return result;
    }

    const auto terminalizeExactAttempt =
        [&](const std::string& nextPhase) {
        const std::string diagnostic =
            "checkpoint_stop;checkpoint_epoch=" +
            std::to_string(epoch) +
            ";checkpoint_model_id=" + std::to_string(modelId) +
            ";next_phase=" + nextPhase +
            ";worker_attempt_id=" + std::to_string(workerAttemptId);
        const pqxx::result terminalized = transaction.exec_params(
            "UPDATE experiment_scheduler_worker_attempt a SET "
            "lifecycle_state='completed',"
            "completed_at=clock_timestamp(),"
            "last_observed_at=clock_timestamp(),"
            "reconciliation_result='checkpoint_stop_completed',"
            "diagnostic=$1 "
            "WHERE a.worker_attempt_id=$2 "
            "AND a.experiment_id=$3 "
            "AND a.checkpoint_eval_id IS NULL "
            "AND a.worker_kind='experiment' "
            "AND a.lifecycle_phase='train' "
            "AND a.capacity_class='train' "
            "AND a.scheduler_invocation_id IS NOT DISTINCT FROM $4 "
            "AND a.scheduler_fencing_token IS NOT DISTINCT FROM $5 "
            "AND a.worker_pid=$6 "
            "AND a.worker_process_group_id=$7 "
            "AND a.worker_process_start_identity=$8 "
            "AND a.canonical_executable_path=$9 "
            "AND a.command_line=$10 "
            "AND a.command_identity=$11 "
            "AND a.lifecycle_state IN ('spawned','running','observed') "
            "AND EXISTS ("
            " SELECT 1 FROM experiment e "
            " WHERE e.experiment_id=$3 "
            " AND e.status='running' AND e.phase='train' "
            " AND e.active_scheduler_worker_attempt_id="
            "a.worker_attempt_id "
            " AND e.last_checkpoint_stop_decision_epoch=$12 "
            " AND e.stop_after_checkpoint_epoch IS NOT NULL "
            " AND e.stop_after_checkpoint_epoch<=$12"
            ") "
            "AND EXISTS ("
            " SELECT 1 FROM model m JOIN matrix tm "
            " ON tm.model_id=m.model_id "
            " WHERE m.model_id=$13 AND m.experiment_id=$3 "
            " AND tm.param_name='train_config_meta' "
            " AND tm.row_idx=0 AND tm.col_idx=10 "
            " AND round(tm.value)::int=$12"
            ") "
            "RETURNING a.worker_attempt_id;",
            diagnostic,
            workerAttemptId,
            *experimentId,
            exact->schedulerInvocationId,
            exact->schedulerFencingToken,
            *exact->workerPid,
            *exact->processGroupId,
            *exact->processStartIdentity,
            *exact->canonicalExecutablePath,
            *exact->commandLine,
            exact->commandIdentity,
            epoch,
            modelId);
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            terminalized,
            "terminalize_exact_checkpoint_stop_train_attempt");
        result.workerAttemptId = workerAttemptId;
    };

    result.cancellationRequested = !experiments[0][0].is_null();
    if (!result.cancellationRequested)
    {
        const std::string nextPhase =
            experiments[0][2].as<bool>() ? "infer" : "analyze";
        terminalizeExactAttempt(nextPhase);
        pqxx::result updated = transaction.exec_params(
            "UPDATE experiment SET current_epoch=$1,worker_pid=NULL,"
            "worker_process_group_id=NULL,"
            "worker_process_start_identity=NULL,worker_executable=NULL,"
            "worker_command_line=NULL,worker_control_state='running',"
            "worker_global_pause_request_id=NULL,"
            "active_scheduler_worker_attempt_id=NULL,"
            "current_operation=$4,"
            "stopped_at_checkpoint_epoch=$1,"
            "stopped_at_checkpoint_model_id=$2,last_model_id=$2,"
            "status='pending',phase=$4,exit_code=0,error_message=NULL,"
            "updated_at=now() WHERE experiment_id=$3 "
            "AND status='running' AND phase='train' "
            "AND active_scheduler_worker_attempt_id=$5 "
            "AND last_checkpoint_stop_decision_epoch=$1 "
            "AND stop_after_checkpoint_epoch IS NOT NULL "
            "AND stop_after_checkpoint_epoch<=$1 "
            "RETURNING experiment_id;",
            epoch,
            modelId,
            *experimentId,
            nextPhase,
            workerAttemptId);
        EA::SchedulerOwnership::RequireAffectedExactlyOne(
            updated,
            "advance_exact_checkpoint_stop_experiment");
        result.recorded = true;
        result.detail = "checkpoint_stopped";
        return result;
    }

    const long long requestId = experiments[0][0].as<long long>();
    const bool activeCancellation = experiments[0][7].as<bool>();
    if (!activeCancellation)
    {
        result.detail = "cancellation_request_not_active";
        return result;
    }
    const auto reconcileUnderPersistedOwner = [&] {
        const pqxx::result owner = transaction.exec_params(
            "SELECT application_owner FROM experiment_admin_request "
            "WHERE request_id=$1 AND action='cancel_all';",
            requestId);
        if (!owner.empty() && !owner[0][0].is_null())
            ReconcileActiveCancellation(
                transaction, owner[0][0].as<std::string>(), false);
    };
    result.cancellationRequestId = requestId;
    const bool inferBeforeCancel = experiments[0][1].as<bool>();
    result.inferenceRequested = inferBeforeCancel;
    const bool hasInferenceRange = experiments[0][2].as<bool>();
    pqxx::result outcomes = transaction.exec_params(
        "SELECT worker_identity,cancellation_checkpoint_epoch,"
        "cancellation_checkpoint_model_id,outcome_status,worker_attempt_id "
        "FROM experiment_admin_worker_outcome "
        "WHERE request_id=$1 AND experiment_id=$2 "
        "AND worker_kind='experiment' ORDER BY worker_identity FOR UPDATE;",
        requestId,
        *experimentId);

    std::string failure = CheckpointModelEvidenceFailure(
        transaction, *experimentId, epoch, modelId);
    std::optional<int> persistedEpoch;
    std::optional<long long> persistedModelId;
    std::string workerIdentity;
    if (outcomes.size() != 1)
        failure = outcomes.empty()
            ? "cancellation_worker_outcome_missing"
            : "cancellation_worker_outcome_ambiguous";
    else
    {
        workerIdentity = outcomes[0][0].as<std::string>();
        if (!outcomes[0][1].is_null())
            persistedEpoch = outcomes[0][1].as<int>();
        if (!outcomes[0][2].is_null())
            persistedModelId = outcomes[0][2].as<long long>();
        if (persistedEpoch && *persistedEpoch != epoch)
            failure = "cancellation_checkpoint_epoch_conflict";
        else if (persistedModelId && *persistedModelId != modelId)
            failure = "cancellation_checkpoint_model_conflict";
        else if (outcomes[0][4].is_null() ||
                 outcomes[0][4].as<long long>() != workerAttemptId)
            failure = "cancellation_worker_attempt_identity_mismatch";
    }

    DbTarget target;
    target.worker.experimentId = *experimentId;
    target.latestCheckpointEpoch = epoch;
    target.latestCheckpointModelId = modelId;
    target.hasInferenceRange = hasInferenceRange;
    std::optional<long long> evaluationId;
    if (failure.empty() && inferBeforeCancel && hasInferenceRange)
    {
        evaluationId =
            QueueCancellationInference(transaction, requestId, target);
        if (!evaluationId || target.inferenceFailed)
            failure = target.inferenceFailureDetail.empty()
                ? "cancellation_checkpoint_evaluation_materialization_invalid"
                : target.inferenceFailureDetail;
    }
    else if (failure.empty() && inferBeforeCancel)
        failure = "cancellation_inference_range_missing";

    if (!inferBeforeCancel)
    {
        transaction.exec_params(
            "UPDATE experiment_checkpoint_eval SET status='failed',"
            "phase='done',completed_at=COALESCE(completed_at,now()),"
            "updated_at=now(),"
            "error_message='suppressed_by_global_cancellation' "
            "WHERE parent_experiment_id=$1 AND experiment_id=$1 "
            "AND checkpoint_epoch=$2 AND checkpoint_model_id=$3 "
            "AND status='pending' "
            "AND (cancellation_request_id IS NULL "
            "OR cancellation_request_id=$4);",
            *experimentId,
            epoch,
            modelId,
            requestId);
    }

    terminalizeExactAttempt("cancelled");
    const pqxx::result experimentUpdated = transaction.exec_params(
        "UPDATE experiment SET current_epoch=$1,worker_pid=NULL,"
        "worker_process_group_id=NULL,"
        "worker_process_start_identity=NULL,worker_executable=NULL,"
        "worker_command_line=NULL,worker_control_state='running',"
        "worker_global_pause_request_id=NULL,"
        "active_scheduler_worker_attempt_id=NULL,"
        "current_operation='train',"
        "stopped_at_checkpoint_epoch=$1,"
        "stopped_at_checkpoint_model_id=$2,last_model_id=$2,"
        "status='cancelled',phase='train',exit_code=0,"
        "completed_at=COALESCE(completed_at,now()),"
        "cancellation_completed_at=COALESCE(cancellation_completed_at,now()),"
        "error_message='cancelled_at_requested_checkpoint',updated_at=now() "
        "WHERE experiment_id=$3 "
        "AND cancellation_request_id=$4 "
        "AND status='running' AND phase='train' "
        "AND active_scheduler_worker_attempt_id=$5 "
        "AND last_checkpoint_stop_decision_epoch=$1 "
        "AND stop_after_checkpoint_epoch IS NOT NULL "
        "AND stop_after_checkpoint_epoch<=$1 "
        "RETURNING experiment_id;",
        epoch,
        modelId,
        *experimentId,
        requestId,
        workerAttemptId);
    RequireAffectedRows(
        experimentUpdated, 1, "record_cancellation_checkpoint_stop");
    result.recorded = true;

    if (outcomes.size() != 1)
    {
        result.detail = failure;
        return result;
    }
    if (!failure.empty())
    {
        TerminalizeCancellationOutcome(
            transaction, requestId, workerIdentity, failure);
        reconcileUnderPersistedOwner();
        result.detail = failure;
        return result;
    }

    const std::string inferenceAction =
        !inferBeforeCancel
            ? "none"
            : (target.inferenceAlreadyCompleted
                   ? "completed"
                   : (target.inferenceAlreadyRunning ? "running" : "queued"));
    const std::string outcomeStatus =
        !inferBeforeCancel || target.inferenceAlreadyCompleted
            ? "completed"
            : "awaiting_inference";
    const std::string detail =
        target.inferenceAlreadyCompleted
            ? "cancellation_checkpoint_inference_completed"
            : "cancellation_checkpoint_reached";
    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "cancellation_checkpoint_epoch=COALESCE("
        "cancellation_checkpoint_epoch,$1),"
        "cancellation_checkpoint_model_id=COALESCE("
        "cancellation_checkpoint_model_id,$2),"
        "inference_action=$3,outcome_status=$4,detail=$5,updated_at=now() "
        "WHERE request_id=$6 AND worker_identity=$7 "
        "AND outcome_status IN "
        "('planned','pending_checkpoint','awaiting_inference') "
        "AND (cancellation_checkpoint_epoch IS NULL "
        "OR cancellation_checkpoint_epoch=$1) "
        "AND (cancellation_checkpoint_model_id IS NULL "
        "OR cancellation_checkpoint_model_id=$2) "
        "AND (cancellation_checkpoint_epoch IS NULL "
        "OR cancellation_checkpoint_model_id IS NULL "
        "OR inference_action IS DISTINCT FROM $3 "
        "OR outcome_status IS DISTINCT FROM $4 "
        "OR detail IS DISTINCT FROM $5);",
        epoch,
        modelId,
        inferenceAction,
        outcomeStatus,
        detail,
        requestId,
        workerIdentity);
    reconcileUnderPersistedOwner();
    result.detail = detail;
    return result;
}

bool ReconcileActiveCancellation(pqxx::work& transaction,
                                 const std::string& applicationOwner,
                                 bool claimExpiredLease)
{
    if (applicationOwner.empty())
        throw std::invalid_argument(
            "cancellation reconciliation requires an application owner");

    const pqxx::result active = transaction.exec(
        "SELECT r.request_id,r.application_owner,"
        "COALESCE(r.application_lease_until>now(),false) AS live_lease,"
        "c.current_pause_request_id "
        "FROM experiment_global_control c "
        "JOIN experiment_admin_request r "
        "ON r.request_id=c.active_request_id "
        "WHERE c.singleton AND r.action='cancel_all' "
        "FOR UPDATE OF c,r;");
    if (active.empty())
        return false;

    const long long requestId = active[0][0].as<long long>();
    const std::optional<long long> pauseRequestId =
        active[0][3].is_null()
            ? std::optional<long long>{}
            : std::optional<long long>{
                  active[0][3].as<long long>()};
    const bool liveLease = active[0][2].as<bool>();
    const bool sameOwner =
        !active[0][1].is_null() &&
        active[0][1].as<std::string>() == applicationOwner;
    if (liveLease && !sameOwner)
        return false;
    if (!sameOwner && !claimExpiredLease)
        return false;

    const pqxx::result claimed = transaction.exec_params(
        "UPDATE experiment_admin_request r SET application_owner=$2,"
        "application_lease_until=now()+interval '30 seconds' "
        "FROM experiment_global_control c "
        "WHERE r.request_id=$1 AND r.action='cancel_all' "
        "AND c.singleton AND c.active_request_id=r.request_id "
        "AND (r.application_owner=$2 "
        "OR r.application_lease_until IS NULL "
        "OR r.application_lease_until<=now());",
        requestId,
        applicationOwner);
    RequireAffectedRows(
        claimed, 1, "claim_cancellation_reconciliation_owner");

    const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
    if (!snapshot.activeRequestId ||
        *snapshot.activeRequestId != requestId ||
        snapshot.activeAction != "cancel_all")
        throw std::runtime_error(
            "claimed_cancellation_request_is_no_longer_active");

    pqxx::result outcomes = transaction.exec_params(
        "SELECT worker_identity,experiment_id,outcome_status,"
        "inference_action,cancellation_checkpoint_epoch,"
        "cancellation_checkpoint_model_id "
        "FROM experiment_admin_worker_outcome "
        "WHERE request_id=$1 AND worker_kind='experiment' "
        "AND outcome_status IN "
        "('planned','pending_checkpoint','awaiting_inference') "
        "ORDER BY experiment_id,worker_identity FOR UPDATE;",
        requestId);
    for (const auto& row : outcomes)
    {
        const std::string workerIdentity = row[0].as<std::string>();
        const long long experimentId = row[1].as<long long>();
        const std::string outcomeStatus = row[2].as<std::string>();
        const std::string inferenceAction = row[3].as<std::string>();
        std::optional<int> checkpointEpoch;
        std::optional<long long> checkpointModelId;
        if (!row[4].is_null())
            checkpointEpoch = row[4].as<int>();
        if (!row[5].is_null())
            checkpointModelId = row[5].as<long long>();

        const auto experiment =
            LoadCancellationExperimentEvidence(transaction, experimentId);
        if (!experiment)
        {
            TerminalizeCancellationOutcome(
                transaction,
                requestId,
                workerIdentity,
                "cancellation_experiment_evidence_missing");
            continue;
        }
        const bool terminal = IsTerminalExperiment(*experiment);
        if (terminal)
        {
            const std::string lifecycleFailure =
                TerminalExperimentLifecycleFailure(*experiment);
            if (!lifecycleFailure.empty())
            {
                TerminalizeCancellationOutcome(
                    transaction,
                    requestId,
                    workerIdentity,
                    lifecycleFailure);
                continue;
            }
            if (!experiment->cancellationRequestId ||
                *experiment->cancellationRequestId != requestId)
            {
                TerminalizeCancellationOutcome(
                    transaction,
                    requestId,
                    workerIdentity,
                    "terminal_experiment_cancellation_request_mismatch");
                continue;
            }
        }

        if (outcomeStatus == "pending_checkpoint")
        {
            if (!terminal)
                continue;
            std::string failure;
            const auto identity = RecoverDurableStoppedIdentity(
                transaction,
                requestId,
                experimentId,
                *experiment,
                checkpointEpoch,
                checkpointModelId,
                failure);
            if (!identity)
            {
                TerminalizeCancellationOutcome(
                    transaction,
                    requestId,
                    workerIdentity,
                    failure);
                continue;
            }
            checkpointEpoch = identity->first;
            checkpointModelId = identity->second;
            transaction.exec_params(
                "UPDATE experiment_admin_worker_outcome SET "
                "cancellation_checkpoint_epoch=COALESCE("
                "cancellation_checkpoint_epoch,$1),"
                "cancellation_checkpoint_model_id=COALESCE("
                "cancellation_checkpoint_model_id,$2),updated_at=now() "
                "WHERE request_id=$3 AND worker_identity=$4 "
                "AND (cancellation_checkpoint_epoch IS NULL "
                "OR cancellation_checkpoint_model_id IS NULL);",
                *checkpointEpoch,
                *checkpointModelId,
                requestId,
                workerIdentity);
            if (!snapshot.inferBeforeCancel)
            {
                transaction.exec_params(
                    "UPDATE experiment_admin_worker_outcome SET "
                    "outcome_status='completed',"
                    "detail=CASE WHEN $1='completed' "
                    "THEN 'cancellation_checkpoint_reconciled_from_terminal_experiment' "
                    "ELSE 'cancellation_checkpoint_reached' END,"
                    "updated_at=now() "
                    "WHERE request_id=$2 AND worker_identity=$3 "
                    "AND outcome_status='pending_checkpoint';",
                    experiment->status,
                    requestId,
                    workerIdentity);
                continue;
            }
            ReconcileCancellationInferenceOutcome(
                transaction,
                requestId,
                workerIdentity,
                experimentId,
                *experiment,
                checkpointEpoch,
                checkpointModelId);
            continue;
        }

        if (outcomeStatus == "awaiting_inference" ||
            (outcomeStatus == "planned" &&
             snapshot.inferBeforeCancel &&
             inferenceAction != "none"))
        {
            ReconcileCancellationInferenceOutcome(
                transaction,
                requestId,
                workerIdentity,
                experimentId,
                *experiment,
                checkpointEpoch,
                checkpointModelId);
            continue;
        }

        if (outcomeStatus == "planned" && terminal)
        {
            transaction.exec_params(
                "UPDATE experiment_admin_worker_outcome SET "
                "outcome_status='completed',"
                "detail='terminal_cancellation_reconciled_after_restart',"
                "updated_at=now() "
                "WHERE request_id=$1 AND worker_identity=$2 "
                "AND outcome_status='planned';",
                requestId,
                workerIdentity);
        }
    }

    UpdateRequestAccounting(
        transaction,
        requestId,
        false,
        applicationOwner,
        "cancel_all");
    const pqxx::row request = transaction.exec_params(
        "SELECT status FROM experiment_admin_request "
        "WHERE request_id=$1 AND action='cancel_all' "
        "AND application_owner=$2 "
        "AND application_lease_until>now() FOR UPDATE;",
        requestId,
        applicationOwner).one_row();
    const bool unresolved =
        HasUnresolvedWorkerOutcome(transaction, requestId, false);
    if (request[0].as<std::string>() != "pending" && !unresolved)
    {
        if (pauseRequestId)
            ClearPauseGenerationAfterResolvedRequest(
                transaction,
                requestId,
                *pauseRequestId,
                applicationOwner,
                "cancel_all");
        const pqxx::result cleared = transaction.exec_params(
            "UPDATE experiment_global_control c "
            "SET active_request_id=NULL,revision=revision+1,updated_at=now() "
            "FROM experiment_admin_request r "
            "WHERE c.singleton AND c.active_request_id=$1 "
            "AND r.request_id=c.active_request_id "
            "AND r.action='cancel_all' AND r.application_owner=$2 "
            "AND r.application_lease_until>now();",
            requestId,
            applicationOwner);
        RequireAffectedRows(
            cleared, 1, "clear_reconciled_cancellation_gate");
    }
    return true;
}

int RunCommand(const std::string& connectionString,
               const Command& command,
               std::ostream& output,
               std::ostream& error)
{
    PosixProcessOperations processes;
    return RunCommandWithProcessOperationsForTesting(
        connectionString, command, output, error, processes);
}

int RunCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const Command& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes)
{
    if (const auto validation = ValidateCommand(command))
    {
        error << "GLOBAL_EXPERIMENT_CONTROL_REJECTED,reason="
              << *validation << "\n";
        return 1;
    }

    std::vector<DbTarget> targets;
    std::string previousState;
    std::optional<long long> applicablePauseRequestId;
    std::string committedMachineOutput;
    std::string finalPersistedStatus = "failed";
    if (command.dryRun)
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        AcquireCoordinationLock(transaction);
        const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
        previousState = snapshot.desiredState;
        applicablePauseRequestId = snapshot.currentPauseRequestId;
        targets = LoadTargets(transaction);
        const size_t dryRunTargetCount =
            command.action == Action::ResumeAll
                ? static_cast<size_t>(std::count_if(
                      targets.begin(), targets.end(),
                      [&](const DbTarget& target) {
                          return applicablePauseRequestId &&
                                 target.worker.lifecycleStatus == "running" &&
                                 target.resumeBeforeAction &&
                                 target.workerGlobalPauseRequestId ==
                                     applicablePauseRequestId;
                      }))
                : targets.size();
        output << "GLOBAL_EXPERIMENT_CONTROL_DRY_RUN,action="
               << ToString(command.action)
               << ",previous_state=" << previousState
               << ",resulting_state=" << ActionState(command.action)
               << ",targets=" << dryRunTargetCount << "\n";
        for (DbTarget& target : targets)
        {
            if (command.action == Action::ResumeAll &&
                (!applicablePauseRequestId ||
                 target.workerGlobalPauseRequestId !=
                     applicablePauseRequestId ||
                 !target.resumeBeforeAction))
                continue;
            if (command.action == Action::CancelAll &&
                command.cancellationMode ==
                    CancellationMode::AfterNextCheckpoint &&
                target.worker.lifecycleStatus == "running" &&
                target.worker.phase == "train")
            {
                target.cancellationCheckpoint = NextCancellationCheckpoint(
                    target.currentEpoch.value_or(0),
                    target.checkpointInterval,
                    target.targetEpochs,
                    target.lastCheckpointStopDecisionEpoch ==
                            target.currentEpoch
                        ? std::nullopt
                        : target.latestCheckpointEpoch);
            }
            ValidatedWorker validation;
            if (target.worker.lifecycleStatus == "running" &&
                target.worker.pid > 0)
                validation = ValidateManagedWorker(target.worker, processes);
            std::string intendedSignal = "none";
            std::string intendedPlan = "observe_only";
            if (target.worker.lifecycleStatus != "running" &&
                command.action == Action::CancelAll)
                intendedPlan = "cancel_without_launch";
            else if (target.worker.lifecycleStatus == "running")
            {
                if (command.action == Action::PauseAll)
                {
                    intendedSignal =
                        validation.identity == IdentityResult::Validated &&
                                !validation.observation.stopped
                            ? "SIGSTOP"
                            : "none";
                    intendedPlan = "validate_then_pause";
                }
                else if (command.action == Action::ResumeAll)
                {
                    intendedSignal =
                        validation.identity == IdentityResult::Validated &&
                                validation.observation.stopped
                            ? "SIGCONT"
                            : "none";
                    intendedPlan = "validate_then_resume";
                }
                else if (target.cancellationCheckpoint &&
                         target.worker.phase == "train" &&
                         target.latestCheckpointEpoch !=
                             target.cancellationCheckpoint)
                {
                    intendedSignal =
                        target.resumeBeforeAction ? "SIGCONT" : "none";
                    intendedPlan = "continue_to_checkpoint";
                }
                else
                {
                    intendedSignal = target.resumeBeforeAction
                        ? "SIGCONT|SIGTERM|SIGKILL_if_needed"
                        : "SIGTERM|SIGKILL_if_needed";
                    intendedPlan = "cancel_immediately";
                }
            }
            output << "GLOBAL_EXPERIMENT_CONTROL_TARGET,experiment_id="
                   << target.worker.experimentId
                   << ",status=" << target.worker.lifecycleStatus
                   << ",phase=" << target.worker.phase
                   << ",pid="
                   << (target.worker.pid > 0
                           ? std::to_string(target.worker.pid)
                           : "NULL")
                   << ",pgid="
                   << (target.worker.processGroupId
                           ? std::to_string(*target.worker.processGroupId)
                           : "NULL")
                   << ",identity="
                   << (target.worker.lifecycleStatus == "running" &&
                               target.worker.pid > 0
                           ? ToString(validation.identity)
                           : "not_checked")
                   << ",checkpoint_target="
                   << (target.cancellationCheckpoint
                           ? std::to_string(*target.cancellationCheckpoint)
                           : "NULL")
                   << ",plan=" << intendedPlan
                   << ",intended_signal=" << intendedSignal
                   << ",inference="
                   << (command.inferBeforeCancel
                           ? (command.cancellationMode ==
                                          CancellationMode::
                                              AfterNextCheckpoint &&
                                      target.cancellationCheckpoint &&
                                      target.latestCheckpointEpoch !=
                                          target.cancellationCheckpoint
                                  ? "exact_future_cancellation_checkpoint"
                                  : (target.latestCheckpointAmbiguous
                                  ? "identity_ambiguous"
                                  : (target.latestCheckpointModelId
                                         ? "latest_durable_checkpoint"
                                         : "no_checkpoint")))
                           : "none")
                   << "\n";
        }
        transaction.commit();
        return 0;
    }

    long long requestId = -1;
    bool retryingPersistedRequest = false;
    const bool schedulerRunningObserved = SchedulerObservedRunning();
    const std::string invocationIdentity =
        command.invocationIdentity.empty()
            ? std::string{"pid:"} + std::to_string(::getpid())
            : command.invocationIdentity;
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        AcquireCoordinationLock(transaction);
        ControlSnapshot snapshot = LoadControlSnapshot(transaction);
        const auto matchesCommandShape =
            [&command](const ControlSnapshot& candidate) {
                return candidate.activeAction &&
                       *candidate.activeAction == ToString(command.action) &&
                       candidate.inferBeforeCancel ==
                           command.inferBeforeCancel &&
                       ((!command.cancellationMode &&
                         !candidate.cancellationMode) ||
                        (command.cancellationMode &&
                         candidate.cancellationMode &&
                         *candidate.cancellationMode ==
                             ToString(*command.cancellationMode)));
            };
        if (snapshot.activeRequestId && !matchesCommandShape(snapshot))
        {
            ReconcileActiveCancellation(
                transaction, invocationIdentity, true);
            snapshot = LoadControlSnapshot(transaction);
        }
        applicablePauseRequestId = snapshot.currentPauseRequestId;
        if (snapshot.activeRequestId)
        {
            const bool sameRequestShape = matchesCommandShape(snapshot);
            if (!sameRequestShape)
            {
                error << "GLOBAL_EXPERIMENT_CONTROL_REJECTED,reason="
                         "conflicting_administrative_request_active,"
                         "request_id="
                      << *snapshot.activeRequestId << "\n";
                transaction.commit();
                return 1;
            }
            pqxx::result claim = transaction.exec_params(
                "SELECT application_owner,"
                "application_lease_until IS NOT NULL "
                "AND application_lease_until>now() "
                "FROM experiment_admin_request WHERE request_id=$1 FOR UPDATE;",
                *snapshot.activeRequestId);
            if (!claim.empty() && claim[0][1].as<bool>() &&
                !claim[0][0].is_null() &&
                claim[0][0].as<std::string>() != invocationIdentity)
            {
                error << "GLOBAL_EXPERIMENT_CONTROL_REJECTED,reason="
                         "administrative_request_application_in_progress,"
                         "request_id="
                      << *snapshot.activeRequestId << "\n";
                transaction.commit();
                return 1;
            }
            requestId = *snapshot.activeRequestId;
            previousState = snapshot.desiredState;
            retryingPersistedRequest = true;
            const pqxx::result claimed = transaction.exec_params(
                "UPDATE experiment_admin_request r "
                "SET application_owner=$1,"
                "application_lease_until=now()+interval '30 seconds' "
                "FROM experiment_global_control c "
                "WHERE r.request_id=$2 AND r.action=$3 "
                "AND c.singleton AND c.active_request_id=r.request_id "
                "AND (r.application_owner=$1 "
                "OR r.application_lease_until IS NULL "
                "OR r.application_lease_until<=now());",
                invocationIdentity,
                requestId,
                ToString(command.action));
            RequireAffectedRows(
                claimed, 1, "claim_existing_administrative_request");
            if (command.action == Action::PauseAll)
            {
                applicablePauseRequestId = requestId;
                const pqxx::result restored = transaction.exec_params(
                    "UPDATE experiment_global_control "
                    "SET current_pause_request_id=$1 "
                    "FROM experiment_admin_request r "
                    "WHERE singleton AND active_request_id=$1 "
                    "AND r.request_id=active_request_id "
                    "AND r.application_owner=$2 "
                    "AND r.application_lease_until>now();",
                    requestId,
                    invocationIdentity);
                RequireAffectedRows(
                    restored, 1, "restore_owned_pause_generation");
            }
            targets =
                LoadRetryTargets(transaction, requestId, command.action);
            transaction.commit();
        }
        else
        {
            previousState = snapshot.desiredState;
            if (command.action == Action::ResumeAll)
            {
                targets = applicablePauseRequestId
                    ? LoadPauseGenerationTargets(
                          transaction, *applicablePauseRequestId)
                    : std::vector<DbTarget>{};
            }
            else
            {
                targets = LoadTargets(transaction);
            }

            pqxx::result inserted = transaction.exec_params(
                "INSERT INTO experiment_admin_request ("
                "action,cancellation_mode,infer_before_cancel,"
                "invocation_identity,requester_identity,application_owner,"
                "application_lease_until,previous_global_state,"
                "resulting_global_state,scheduler_running_observed,target_count) "
                "VALUES ($1,$2,$3,$4,$5,$4,now()+interval '30 seconds',"
                "$6,$7,$8,$9) RETURNING request_id;",
                ToString(command.action),
                command.cancellationMode
                    ? std::optional<std::string>{
                          ToString(*command.cancellationMode)}
                    : std::optional<std::string>{},
                command.inferBeforeCancel,
                invocationIdentity,
                command.requesterIdentity.value_or(CurrentRequester()),
                previousState,
                ActionState(command.action),
                schedulerRunningObserved,
                static_cast<int>(targets.size()));
            requestId = inserted[0][0].as<long long>();

            const pqxx::result gated = transaction.exec_params(
                "UPDATE experiment_global_control SET desired_state=$1,"
                "active_request_id=$2,"
                "current_pause_request_id=CASE WHEN $3 THEN $2 "
                "ELSE current_pause_request_id END,"
                "revision=revision+1,updated_at=now() "
                "WHERE singleton=true;",
                ActionState(command.action),
                requestId,
                command.action == Action::PauseAll);
            RequireAffectedRows(
                gated, 1, "activate_new_administrative_request");
            if (command.action == Action::PauseAll)
                applicablePauseRequestId = requestId;

            for (DbTarget& target : targets)
            {
            if (command.action == Action::PauseAll ||
                command.action == Action::ResumeAll)
            {
                if (command.action == Action::PauseAll &&
                    target.worker.lifecycleStatus != "running")
                    continue;
                if (command.action == Action::ResumeAll)
                {
                    if (target.plan == "already_satisfied")
                    {
                        MarkResumeAllAlreadySatisfied(
                            transaction,
                            requestId,
                            target,
                            "pause_generation_member_already_released");
                        continue;
                    }
                    if (target.authoritativeExactMatch &&
                        (!applicablePauseRequestId ||
                         target.workerGlobalPauseRequestId !=
                             applicablePauseRequestId ||
                         !target.resumeBeforeAction))
                    {
                        MarkResumeAllAlreadySatisfied(
                            transaction,
                            requestId,
                            target,
                            "pause_generation_member_already_reconciled");
                        target.plan = "already_satisfied";
                        continue;
                    }
                }
                if (command.action == Action::ResumeAll)
                    target.sourcePauseRequestId =
                        applicablePauseRequestId;
                InsertOutcome(transaction, requestId, target, "planned", "none",
                              "worker_validation_and_signal_planned");
                continue;
            }

            if (target.checkpointWorker)
            {
                if (command.inferBeforeCancel)
                {
                    pqxx::result authorized = transaction.exec_params(
                        "SELECT 1 FROM experiment_checkpoint_eval "
                        "WHERE checkpoint_eval_id=$1 "
                        "AND cancellation_request_id=$2;",
                        *target.checkpointEvalId,
                        requestId);
                    if (!authorized.empty())
                    {
                        target.plan = "already_accounted";
                        continue;
                    }
                }
                target.signalImmediately = true;
                const pqxx::result assigned = transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval "
                    "SET cancellation_request_id=$1,updated_at=now() "
                    "WHERE checkpoint_eval_id=$2 "
                    "AND cancellation_request_id IS NULL;",
                    requestId,
                    *target.checkpointEvalId);
                RequireAffectedRows(
                    assigned, 1, "assign_checkpoint_cancellation_request");
                InsertOutcome(
                    transaction,
                    requestId,
                    target,
                    "planned",
                    "running",
                    "active_checkpoint_inference_cancellation");
                continue;
            }

            const pqxx::result assigned = transaction.exec_params(
                "UPDATE experiment SET cancellation_request_id=$1,"
                "cancel_infer_before=$2,updated_at=now() "
                "WHERE experiment_id=$3 "
                "AND cancellation_request_id IS NULL;",
                requestId,
                command.inferBeforeCancel,
                target.worker.experimentId);
            RequireAffectedRows(
                assigned, 1, "assign_experiment_cancellation_request");
            if (target.worker.lifecycleStatus != "running")
            {
                const pqxx::result cancelledQueued = transaction.exec_params(
                    "UPDATE experiment SET status='cancelled',"
                    "completed_at=COALESCE(completed_at,now()),"
                    "cancellation_completed_at=now(),worker_pid=NULL,"
                    "worker_process_group_id=NULL,updated_at=now() "
                    "WHERE experiment_id=$1 "
                    "AND status IN ('pending','paused') "
                    "AND cancellation_request_id=$2;",
                    target.worker.experimentId,
                    requestId);
                RequireAffectedRows(
                    cancelledQueued, 1, "cancel_nonrunning_experiment");
                InsertOutcome(transaction, requestId, target, "completed",
                              "none", "queued_or_paused_experiment_cancelled");
                continue;
            }

            const bool afterCheckpoint =
                command.cancellationMode ==
                    CancellationMode::AfterNextCheckpoint &&
                target.worker.phase == "train";
            if (afterCheckpoint)
            {
                target.cancellationCheckpoint = NextCancellationCheckpoint(
                    target.currentEpoch.value_or(0),
                    target.checkpointInterval,
                    target.targetEpochs,
                    target.lastCheckpointStopDecisionEpoch ==
                            target.currentEpoch
                        ? std::nullopt
                        : target.latestCheckpointEpoch);
            }
            const bool useCurrentBoundary =
                afterCheckpoint && target.cancellationCheckpoint &&
                target.latestCheckpointEpoch &&
                *target.cancellationCheckpoint ==
                    *target.latestCheckpointEpoch;
            if (afterCheckpoint && target.cancellationCheckpoint &&
                !useCurrentBoundary)
            {
                const pqxx::result checkpointPlanned =
                    transaction.exec_params(
                    "UPDATE experiment SET stop_after_checkpoint_epoch=$1,"
                    "cancel_after_checkpoint_epoch=$1,updated_at=now() "
                    "WHERE experiment_id=$2 "
                    "AND status='running' AND cancellation_request_id=$3;",
                    *target.cancellationCheckpoint,
                    target.worker.experimentId,
                    requestId);
                RequireAffectedRows(
                    checkpointPlanned,
                    1,
                    "persist_cancellation_checkpoint_plan");
                InsertOutcome(transaction, requestId, target,
                              "pending_checkpoint", "none",
                              "continue_to_next_durable_checkpoint");
                target.plan = "pending_checkpoint";
                continue;
            }

            target.signalImmediately = true;
            if (command.inferBeforeCancel)
            {
                target.inferenceRequested = true;
                if (!QueueCancellationInference(
                        transaction, requestId, target))
                {
                    target.inferenceFailed = true;
                    InsertOutcome(transaction, requestId, target, "planned",
                                  target.latestCheckpointAmbiguous
                                      ? "identity_ambiguous"
                                      : "no_checkpoint",
                                  target.latestCheckpointAmbiguous
                                      ? "durable_checkpoint_identity_ambiguous"
                                      : "no_valid_durable_checkpoint_for_inference");
                }
                else
                {
                    InsertOutcome(transaction, requestId, target, "planned",
                                  target.inferenceAlreadyCompleted
                                      ? "already_completed"
                                      : (target.inferenceFailed
                                             ? "failed"
                                             : (target.inferenceAlreadyRunning
                                             ? "running"
                                             : "queued")),
                                  useCurrentBoundary
                                      ? "current_durable_checkpoint_selected"
                                      : "latest_durable_checkpoint_selected");
                }
            }
            else
            {
                InsertOutcome(transaction, requestId, target, "planned",
                              "none",
                              afterCheckpoint
                                  ? "no_future_checkpoint_cancel_immediately"
                                  : "immediate_cancellation");
            }
            }
            if (command.action == Action::CancelAll)
            {
                transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval SET status='failed',"
                    "phase='done',completed_at=now(),updated_at=now(),"
                    "error_message='cancelled_by_global_request' "
                    "WHERE status='pending' "
                    "AND cancellation_request_id IS DISTINCT FROM $1;",
                    requestId);
            }
            transaction.commit();
        }
    }

    bool anySignalAttempted = false;
    bool postSignalPredicateMismatch = false;
    for (DbTarget& target : targets)
    {
        if (command.action == Action::CancelAll &&
            !target.signalImmediately &&
            target.plan != "pending_checkpoint")
            continue;
        if (command.action == Action::CancelAll &&
            target.plan == "pending_checkpoint")
        {
            if (target.worker.lifecycleStatus != "running")
                continue;
            SignalOutcome resume;
            if (target.frozenReplayTarget &&
                !target.authoritativeExactMatch)
                resume =
                    FrozenTargetAuthorizationFailure(target, processes);
            else if (target.resumeBeforeAction)
                resume = ResumeWorkerAuthorized(
                    target.worker,
                    processes,
                    ExactAttemptSignalAuthorization(
                        connectionString,
                        requestId,
                        invocationIdentity,
                        "cancel_all",
                        processes));
            else
            {
                const ValidatedWorker validation =
                    ValidateManagedWorker(target.worker, processes);
                resume.identity = validation.identity;
                resume.detail = validation.detail;
                resume.success =
                    validation.identity == IdentityResult::Validated;
                resume.result = resume.success
                    ? "already_requested_state"
                    : ToString(validation.identity);
            }
            std::string persistedResumeResult = resume.result;
            if (persistedResumeResult == "unsafe_process_group" ||
                persistedResumeResult == "inspection_failed")
                persistedResumeResult = "identity_validation_failed";
            if (persistedResumeResult == "permission_denied")
                persistedResumeResult = "permission_failure";
            pqxx::connection connection{connectionString};
            pqxx::work transaction{connection};
            transaction.exec("SET TRANSACTION READ WRITE;");
            AcquireCoordinationLock(transaction);
            if (!OwnsActiveRequest(
                    transaction,
                    requestId,
                    invocationIdentity,
                    ToString(command.action)))
            {
                error << "GLOBAL_EXPERIMENT_CONTROL_REJECTED,reason="
                         "administrative_request_ownership_lost,request_id="
                      << requestId << "\n";
                transaction.commit();
                return 1;
            }
            const pqxx::result leaseRefreshed = transaction.exec_params(
                "UPDATE experiment_admin_request "
                "SET application_lease_until=now()+interval '30 seconds' "
                "WHERE request_id=$1 AND application_owner=$2 "
                "AND application_lease_until>now();",
                requestId,
                invocationIdentity);
            RequireAffectedRows(
                leaseRefreshed, 1, "refresh_checkpoint_cancellation_lease");
            const auto exactCheckpointAttempt =
                LockExactTargetForMutation(
                    transaction, target, true);
            if (!exactCheckpointAttempt)
            {
                postSignalPredicateMismatch = true;
                resume.success = false;
                resume.identity =
                    IdentityResult::IdentityValidationFailed;
                resume.result = "identity_validation_failed";
                resume.detail =
                    "exact_active_attempt_changed_during_checkpoint_cancel";
            }
            if (resume.success && exactCheckpointAttempt)
            {
                const pqxx::result resumed = transaction.exec_params(
                    "UPDATE experiment SET worker_control_state='running',"
                    "updated_at=now() WHERE experiment_id=$1 "
                    "AND status='running' AND phase=$3 AND worker_pid=$2 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $4 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $5 "
                    "AND worker_executable IS NOT DISTINCT FROM $6 "
                    "AND worker_command_line IS NOT DISTINCT FROM $7 "
                    "AND cancellation_request_id=$8 "
                    "AND active_scheduler_worker_attempt_id=$9;",
                    target.worker.experimentId,
                    target.worker.pid,
                    target.worker.phase,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    requestId,
                    target.worker.workerAttemptId);
                if (resumed.affected_rows() != 1)
                {
                    postSignalPredicateMismatch = true;
                    resume.success = false;
                    resume.identity =
                        IdentityResult::IdentityValidationFailed;
                    resume.detail =
                        "worker_state_changed_after_checkpoint_resume_signal";
                }
            }
            if (!resume.success &&
                resume.identity == IdentityResult::ProcessMissing &&
                exactCheckpointAttempt)
            {
                const pqxx::result terminal =
                    transaction.exec_params(
                        "UPDATE experiment_scheduler_worker_attempt a SET "
                        "lifecycle_state='failed',"
                        "completed_at=clock_timestamp(),"
                        "last_observed_at=clock_timestamp(),"
                        "reconciliation_result="
                        "'cancellation_checkpoint_process_missing',"
                        "diagnostic='exact_process_absence_observed' "
                        "WHERE a.worker_attempt_id=$1 "
                        "AND a.lifecycle_state IN "
                        "('spawned','running','observed') "
                        "AND EXISTS (SELECT 1 FROM experiment e "
                        " WHERE e.experiment_id=$2 "
                        " AND e.active_scheduler_worker_attempt_id="
                        "a.worker_attempt_id) "
                        "RETURNING a.worker_attempt_id;",
                        *target.worker.workerAttemptId,
                        target.worker.experimentId);
                EA::SchedulerOwnership::RequireAffectedExactlyOne(
                    terminal,
                    "checkpoint_cancel_terminalize_missing_attempt");
                if (target.latestCheckpointModelId)
                {
                    const pqxx::result requeued = transaction.exec_params(
                        "UPDATE experiment SET status='pending',phase='train',"
                        "worker_pid=NULL,worker_process_group_id=NULL,"
                        "active_scheduler_worker_attempt_id=NULL,"
                        "worker_control_state='running',last_model_id=$1,"
                        "current_operation='train',"
                        "error_message='cancellation_worker_restart_required',"
                        "updated_at=now() "
                        "WHERE experiment_id=$2 AND status='running' "
                        "AND phase=$3 AND worker_pid=$4 "
                        "AND worker_process_group_id IS NOT DISTINCT FROM $5 "
                        "AND worker_process_start_identity IS NOT DISTINCT FROM $6 "
                        "AND worker_executable IS NOT DISTINCT FROM $7 "
                        "AND worker_command_line IS NOT DISTINCT FROM $8 "
                        "AND cancellation_request_id=$9 "
                        "AND active_scheduler_worker_attempt_id=$10;",
                        *target.latestCheckpointModelId,
                        target.worker.experimentId,
                        target.worker.phase,
                        target.worker.pid,
                        target.worker.processGroupId,
                        target.worker.processStartIdentity,
                        target.worker.executable,
                        target.worker.commandLine,
                        requestId,
                        target.worker.workerAttemptId);
                    if (target.authoritativeExactMatch)
                        RequireAffectedRows(
                            requeued,
                            1,
                            "requeue_missing_cancellation_worker");
                }
                else
                {
                    const pqxx::result cancelledMissing =
                        transaction.exec_params(
                        "UPDATE experiment SET status='cancelled',"
                        "worker_pid=NULL,worker_process_group_id=NULL,"
                        "active_scheduler_worker_attempt_id=NULL,"
                        "completed_at=now(),cancellation_completed_at=now(),"
                        "error_message='cancelled_missing_worker_no_checkpoint',"
                        "updated_at=now() WHERE experiment_id=$1 "
                        "AND status='running' AND phase=$2 AND worker_pid=$3 "
                        "AND worker_process_group_id IS NOT DISTINCT FROM $4 "
                        "AND worker_process_start_identity IS NOT DISTINCT FROM $5 "
                        "AND worker_executable IS NOT DISTINCT FROM $6 "
                        "AND worker_command_line IS NOT DISTINCT FROM $7 "
                        "AND cancellation_request_id=$8 "
                        "AND active_scheduler_worker_attempt_id=$9;",
                        target.worker.experimentId,
                        target.worker.phase,
                        target.worker.pid,
                        target.worker.processGroupId,
                        target.worker.processStartIdentity,
                        target.worker.executable,
                        target.worker.commandLine,
                        requestId,
                        target.worker.workerAttemptId);
                    if (target.authoritativeExactMatch)
                        RequireAffectedRows(
                            cancelledMissing,
                            1,
                            "cancel_missing_checkpoint_worker_without_checkpoint");
                }
            }
            const std::string checkpointOutcome =
                resume.success || (resume.identity ==
                                       IdentityResult::ProcessMissing &&
                                   target.latestCheckpointModelId)
                    ? "pending_checkpoint"
                    : (resume.identity == IdentityResult::ProcessMissing
                           ? "partial"
                           : "partial");
            const std::string checkpointDetail =
                resume.success
                    ? (target.resumeBeforeAction
                           ? "resumed_to_reach_cancellation_checkpoint"
                           : "running_to_cancellation_checkpoint_validated")
                    : (resume.identity == IdentityResult::ProcessMissing
                           ? (target.latestCheckpointModelId
                                  ? "missing_worker_requeued_from_durable_checkpoint"
                                  : "missing_worker_no_restart_checkpoint")
                           : resume.detail);
            const pqxx::result outcomeUpdated = transaction.exec_params(
                "UPDATE experiment_admin_worker_outcome o SET "
                "identity_result=$1,signal_result=$2,"
                "outcome_status=$3,detail=$4,updated_at=now() "
                "WHERE request_id=$5 AND worker_identity=$6 "
                "AND outcome_status='pending_checkpoint' "
                "AND EXISTS (SELECT 1 FROM experiment_admin_request r "
                "JOIN experiment_global_control c "
                "ON c.active_request_id=r.request_id "
                "WHERE c.singleton AND r.request_id=o.request_id "
                "AND r.application_owner=$7 AND r.action='cancel_all' "
                "AND r.application_lease_until>now());",
                ToString(resume.identity),
                persistedResumeResult,
                checkpointOutcome,
                checkpointDetail,
                requestId,
                target.workerIdentity,
                invocationIdentity);
            RequireAffectedRows(
                outcomeUpdated,
                1,
                "persist_checkpoint_cancellation_resume_outcome");
            transaction.commit();
            continue;
        }
        if (target.worker.lifecycleStatus != "running")
            continue;
        if (target.plan == "already_satisfied" ||
            target.plan == "already_accounted")
            continue;

        SignalOutcome signal = ApplyTargetSignal(
            target,
            command.action,
            command.terminationGrace,
            processes,
            connectionString,
            requestId,
            invocationIdentity);
        anySignalAttempted =
            anySignalAttempted || !signal.signals.empty();

        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        AcquireCoordinationLock(transaction);
        if (!OwnsActiveRequest(
                transaction,
                requestId,
                invocationIdentity,
                ToString(command.action)))
        {
            error << "GLOBAL_EXPERIMENT_CONTROL_REJECTED,reason="
                     "administrative_request_ownership_lost,request_id="
                  << requestId << "\n";
            transaction.commit();
            return 1;
        }
        const pqxx::result leaseRefreshed = transaction.exec_params(
            "UPDATE experiment_admin_request "
            "SET application_lease_until=now()+interval '30 seconds' "
            "WHERE request_id=$1 AND application_owner=$2 "
            "AND application_lease_until>now();",
            requestId,
            invocationIdentity);
        RequireAffectedRows(
            leaseRefreshed, 1, "refresh_administrative_request_lease");
        const auto exactMutationAttempt =
            target.authoritativeExactTerminalDeparture
                ? std::optional<
                      EA::SchedulerOwnership::ExactAttemptSnapshot>{}
                : LockExactTargetForMutation(
                      transaction,
                      target,
                      true);
        if (!exactMutationAttempt &&
            !target.authoritativeExactTerminalDeparture)
        {
            postSignalPredicateMismatch = true;
            signal.success = false;
            signal.identity =
                IdentityResult::IdentityValidationFailed;
            signal.result = "identity_validation_failed";
            signal.detail =
                "exact_active_worker_attempt_changed_after_signal";
        }
        if (command.action == Action::PauseAll &&
            signal.success && exactMutationAttempt)
        {
            pqxx::result paused;
            if (target.checkpointWorker)
                paused = transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval "
                    "SET worker_control_state='paused',"
                    "worker_global_pause_request_id=$3,updated_at=now() "
                    "WHERE checkpoint_eval_id=$1 AND status='running' "
                    "AND phase='infer' AND worker_pid=$2 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $4 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $5 "
                    "AND worker_executable IS NOT DISTINCT FROM $6 "
                    "AND worker_command_line IS NOT DISTINCT FROM $7 "
                    "AND active_scheduler_worker_attempt_id=$8;",
                    *target.checkpointEvalId,
                    target.worker.pid,
                    requestId,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    target.worker.workerAttemptId);
            else
                paused = transaction.exec_params(
                    "UPDATE experiment SET worker_control_state='paused',"
                    "worker_global_pause_request_id=$3,updated_at=now() "
                    "WHERE experiment_id=$1 "
                    "AND status='running' AND phase=$4 AND worker_pid=$2 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $5 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $6 "
                    "AND worker_executable IS NOT DISTINCT FROM $7 "
                    "AND worker_command_line IS NOT DISTINCT FROM $8 "
                    "AND active_scheduler_worker_attempt_id=$9;",
                    target.worker.experimentId,
                    target.worker.pid,
                    requestId,
                    target.worker.phase,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    target.worker.workerAttemptId);
            if (paused.affected_rows() != 1)
            {
                postSignalPredicateMismatch = true;
                signal.success = false;
                signal.identity =
                    IdentityResult::IdentityValidationFailed;
                signal.detail =
                    "worker_state_changed_after_validated_pause_signal";
            }
        }
        else if (command.action == Action::ResumeAll &&
                 signal.success && exactMutationAttempt)
        {
            pqxx::result resumed;
            if (target.checkpointWorker)
                resumed = transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval "
                    "SET worker_control_state='running',"
                    "worker_global_pause_request_id=NULL,updated_at=now() "
                    "WHERE checkpoint_eval_id=$1 AND status='running' "
                    "AND phase='infer' AND worker_pid=$2 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $4 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $5 "
                    "AND worker_executable IS NOT DISTINCT FROM $6 "
                    "AND worker_command_line IS NOT DISTINCT FROM $7 "
                    "AND worker_control_state='paused' "
                    "AND worker_global_pause_request_id=$3 "
                    "AND active_scheduler_worker_attempt_id=$8;",
                    *target.checkpointEvalId,
                    target.worker.pid,
                    target.sourcePauseRequestId,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    target.worker.workerAttemptId);
            else
                resumed = transaction.exec_params(
                    "UPDATE experiment SET worker_control_state='running',"
                    "worker_global_pause_request_id=NULL,updated_at=now() "
                    "WHERE experiment_id=$1 "
                    "AND status='running' AND phase=$4 AND worker_pid=$2 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $5 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $6 "
                    "AND worker_executable IS NOT DISTINCT FROM $7 "
                    "AND worker_command_line IS NOT DISTINCT FROM $8 "
                    "AND worker_control_state='paused' "
                    "AND worker_global_pause_request_id=$3 "
                    "AND active_scheduler_worker_attempt_id=$9;",
                    target.worker.experimentId,
                    target.worker.pid,
                    target.sourcePauseRequestId,
                    target.worker.phase,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    target.worker.workerAttemptId);
            if (resumed.affected_rows() != 1)
            {
                postSignalPredicateMismatch = true;
                signal.success = false;
                signal.identity =
                    IdentityResult::IdentityValidationFailed;
                signal.detail =
                    "worker_state_changed_after_validated_resume_signal";
            }
        }
        else if (command.action == Action::CancelAll &&
                 exactMutationAttempt &&
                 (signal.success ||
                  signal.identity == IdentityResult::ProcessMissing))
        {
            const pqxx::result attemptTerminal =
                transaction.exec_params(
                    "UPDATE experiment_scheduler_worker_attempt a SET "
                    "lifecycle_state='failed',"
                    "completed_at=clock_timestamp(),"
                    "last_observed_at=clock_timestamp(),"
                    "signal_number=$1,"
                    "reconciliation_result='global_control_cancel',"
                    "diagnostic=$2 "
                    "WHERE a.worker_attempt_id=$3 "
                    "AND a.lifecycle_state IN "
                    "('spawned','running','observed') "
                    "AND EXISTS ("
                    " SELECT 1 FROM experiment e "
                    " WHERE $4::bigint IS NULL "
                    " AND e.experiment_id=$5 "
                    " AND e.active_scheduler_worker_attempt_id="
                    "a.worker_attempt_id "
                    " UNION ALL "
                    " SELECT 1 FROM experiment_checkpoint_eval ce "
                    " WHERE $4::bigint IS NOT NULL "
                    " AND ce.checkpoint_eval_id=$4 "
                    " AND ce.active_scheduler_worker_attempt_id="
                    "a.worker_attempt_id"
                    ") RETURNING a.worker_attempt_id;",
                    signal.signals.empty()
                        ? std::optional<int>{}
                        : std::optional<int>{
                              signal.signals.back()},
                    signal.identity ==
                            IdentityResult::ProcessMissing
                        ? "exact_process_absence_observed"
                        : "exact_attempt_signaled_by_global_control",
                    *target.worker.workerAttemptId,
                    target.checkpointEvalId,
                    target.worker.experimentId);
            EA::SchedulerOwnership::RequireAffectedExactlyOne(
                attemptTerminal,
                "global_control_terminalize_exact_attempt");
            pqxx::result cancelled;
            if (target.checkpointWorker)
            {
                cancelled = transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval SET status='failed',"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "active_scheduler_worker_attempt_id=NULL,"
                    "completed_at=now(),updated_at=now(),"
                    "error_message='cancelled_by_global_request' "
                    "WHERE checkpoint_eval_id=$1 AND status='running' "
                    "AND phase='infer' AND worker_pid=$2 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $3 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $4 "
                    "AND worker_executable IS NOT DISTINCT FROM $5 "
                    "AND worker_command_line IS NOT DISTINCT FROM $6 "
                    "AND cancellation_request_id=$7 "
                    "AND active_scheduler_worker_attempt_id=$8;",
                    *target.checkpointEvalId,
                    target.worker.pid,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    requestId,
                    target.worker.workerAttemptId);
            }
            else
            {
                const std::string currentOperation =
                    EA::ExperimentLifecycle::
                        RequireCanonicalCurrentOperationForPhase(
                            target.worker.phase);
                cancelled = transaction.exec_params(
                    "UPDATE experiment SET status='cancelled',"
                    "completed_at=COALESCE(completed_at,now()),"
                    "cancellation_completed_at=now(),"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "active_scheduler_worker_attempt_id=NULL,"
                    "current_operation=$10,"
                    "error_message=CASE WHEN $1 THEN "
                    "'cancelled_after_sigkill' ELSE 'cancelled_by_global_request' END,"
                    "updated_at=now() WHERE experiment_id=$2 "
                    "AND status='running' AND phase=$3 AND worker_pid=$4 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $5 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $6 "
                    "AND worker_executable IS NOT DISTINCT FROM $7 "
                    "AND worker_command_line IS NOT DISTINCT FROM $8 "
                    "AND cancellation_request_id=$9 "
                    "AND active_scheduler_worker_attempt_id=$11;",
                    signal.result == "escalated",
                    target.worker.experimentId,
                    target.worker.phase,
                    target.worker.pid,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    requestId,
                    currentOperation,
                    target.worker.workerAttemptId);
            }
            const bool exactLifecycleMutationExpected =
                !target.frozenReplayTarget ||
                target.authoritativeExactMatch;
            if (exactLifecycleMutationExpected &&
                cancelled.affected_rows() != 1)
            {
                postSignalPredicateMismatch = true;
                signal.success = false;
                signal.identity =
                    IdentityResult::IdentityValidationFailed;
                signal.detail =
                    "worker_state_changed_after_validated_cancellation_signal";
            }
        }
        UpdateSignalOutcome(
            transaction,
            requestId,
            target,
            signal,
            command.action == Action::CancelAll,
            invocationIdentity,
            ToString(command.action));
        transaction.commit();
    }

    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        AcquireCoordinationLock(transaction);
        if (!OwnsActiveRequest(
                transaction,
                requestId,
                invocationIdentity,
                ToString(command.action)))
        {
            error << "GLOBAL_EXPERIMENT_CONTROL_REJECTED,reason="
                     "administrative_request_ownership_lost,request_id="
                  << requestId << "\n";
            transaction.commit();
            return 1;
        }
        if (command.action == Action::CancelAll)
        {
            if (!ReconcileActiveCancellation(
                    transaction, invocationIdentity, false))
                throw std::runtime_error(
                    "owned_cancellation_reconciliation_was_not_authorized");
        }
        else
        {
            UpdateRequestAccounting(
                transaction,
                requestId,
                false,
                invocationIdentity,
                ToString(command.action));
            const pqxx::row state = transaction.exec_params(
                "SELECT status FROM experiment_admin_request "
                "WHERE request_id=$1 AND application_owner=$2 "
                "AND application_lease_until>now() FOR UPDATE;",
                requestId,
                invocationIdentity).one_row();
            const bool unresolved =
                HasUnresolvedWorkerOutcome(transaction, requestId);
            const bool terminal =
                state[0].as<std::string>() != "pending" &&
                state[0].as<std::string>() != "applying";
            if (terminal && !unresolved)
            {
                if (command.action == Action::ResumeAll &&
                    applicablePauseRequestId)
                    ClearPauseGenerationAfterResolvedRequest(
                        transaction,
                        requestId,
                        *applicablePauseRequestId,
                        invocationIdentity,
                        ToString(command.action));
                const pqxx::result gateCleared =
                    transaction.exec_params(
                        "UPDATE experiment_global_control c "
                        "SET active_request_id=NULL,"
                        "revision=revision+1,updated_at=now() "
                        "FROM experiment_admin_request r "
                        "WHERE c.singleton AND c.active_request_id=$1 "
                        "AND r.request_id=c.active_request_id "
                        "AND r.application_owner=$2 AND r.action=$3 "
                        "AND r.application_lease_until>now();",
                        requestId,
                        invocationIdentity,
                        ToString(command.action));
                RequireAffectedRows(
                    gateCleared,
                    1,
                    "clear_resolved_administrative_request_gate");
            }
        }
        const pqxx::row summary = transaction.exec_params(
            "SELECT action,status,target_count,successful_count,"
            "already_satisfied_count,missing_count,rejected_count,failed_count "
            "FROM experiment_admin_request WHERE request_id=$1;",
            requestId).one_row();
        if (!postSignalPredicateMismatch)
        {
            const pqxx::result leaseCleared = transaction.exec_params(
                "UPDATE experiment_admin_request "
                "SET application_lease_until=NULL "
                "WHERE request_id=$1 AND application_owner=$2;",
                requestId,
                invocationIdentity);
            RequireAffectedRows(
                leaseCleared, 1, "release_administrative_request_lease");
        }
        finalPersistedStatus = summary[1].as<std::string>();
        std::ostringstream committed;
        if (retryingPersistedRequest)
            committed << "GLOBAL_EXPERIMENT_CONTROL_RETRY,request_id="
                      << requestId << ",persisted_plan=1\n";
        committed
            << "GLOBAL_EXPERIMENT_CONTROL_SUMMARY,request_id="
            << requestId
            << ",action=" << summary[0].as<std::string>()
            << ",status=" << finalPersistedStatus
            << ",result="
            << PersistedRequestResult(finalPersistedStatus)
            << ",replay=" << (retryingPersistedRequest ? 1 : 0)
            << ",signal_attempted=" << (anySignalAttempted ? 1 : 0)
            << ",target_count=" << summary[2].as<int>()
            << ",successful_count=" << summary[3].as<int>()
            << ",already_satisfied_count=" << summary[4].as<int>()
            << ",missing_count=" << summary[5].as<int>()
            << ",rejected_count=" << summary[6].as<int>()
            << ",failed_count=" << summary[7].as<int>()
            << ",global_state=" << ActionState(command.action)
            << "\n";
        pqxx::result outcomes = transaction.exec_params(
            "SELECT worker_identity,experiment_id,phase,"
            "COALESCE(worker_pid::text,'NULL'),"
            "identity_result,outcome_status,signal_result,"
            "COALESCE(requested_signal,''),"
            "COALESCE(cancellation_checkpoint_epoch::text,'NULL'),"
            "inference_action,COALESCE(detail,'') "
            "FROM experiment_admin_worker_outcome WHERE request_id=$1 "
            "ORDER BY worker_identity;",
            requestId);
        for (const auto& row : outcomes)
        {
            committed
                << "GLOBAL_EXPERIMENT_CONTROL_OUTCOME,request_id="
                << requestId
                << ",worker_identity=" << row[0].as<std::string>()
                << ",identity_result=" << row[4].as<std::string>()
                << ",outcome_status=" << row[5].as<std::string>()
                << ",signal_result=" << row[6].as<std::string>()
                << ",requested_signal=" << row[7].as<std::string>()
                << ",detail=" << row[10].as<std::string>()
                << ",experiment_id=" << row[1].as<long long>()
                << ",phase=" << row[2].as<std::string>()
                << ",pid=" << row[3].as<std::string>()
                << ",checkpoint_target=" << row[8].as<std::string>()
                << ",inference_action=" << row[9].as<std::string>()
                << "\n";
        }
        transaction.commit();
        committedMachineOutput = committed.str();
    }
    output << committedMachineOutput;
    return RequestExitCodeForPersistedStatus(finalPersistedStatus);
}

int RunExperimentResumeCommand(const std::string& connectionString,
                               const ExperimentResumeCommand& command,
                               std::ostream& output,
                               std::ostream& error)
{
    PosixProcessOperations processes;
    return RunExperimentResumeCommandWithProcessOperationsForTesting(
        connectionString, command, output, error, processes);
}

int RunExperimentResumeCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const ExperimentResumeCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes)
{
    if (command.experimentId <= 0)
    {
        error << "SCHEDULER_CONTROL_REJECTED,action=resume,experiment_id="
              << command.experimentId
              << ",reason=invalid_experiment_id,result=invalid_lifecycle\n";
        return 1;
    }

    const bool willApply = command.confirmed && !command.dryRun;
    const std::string invocationIdentity =
        command.invocationIdentity.empty()
            ? std::string{"pid:"} + std::to_string(::getpid())
            : command.invocationIdentity;
    const bool schedulerRunningObserved =
        willApply ? SchedulerObservedRunning() : false;
    long long requestId = -1;
    long long pauseRequestId = -1;
    DbTarget target;
    bool retrying = false;
    bool alreadyAccounted = false;
    std::ostringstream deferredMachineOutput;

    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec(willApply
            ? "SET TRANSACTION READ WRITE;"
            : "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        AcquireCoordinationLock(transaction);
        const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
        const auto loaded = LoadExperimentResumeTarget(
            transaction, command.experimentId, willApply);
        auto reject = [&](const std::string& reason) {
            output << "SCHEDULER_CONTROL_REJECTED,action=resume,experiment_id="
                   << command.experimentId
                   << ",reason=" << reason << ",result=" << reason << "\n";
        };

        if (snapshot.activeRequestId)
        {
            std::string activeSql =
                "SELECT action,target_experiment_id,application_owner,"
                "application_lease_until IS NOT NULL "
                "AND application_lease_until>now() "
                "FROM experiment_admin_request "
                "WHERE request_id=$1";
            if (willApply)
                activeSql += " FOR UPDATE";
            pqxx::result active = transaction.exec_params(
                activeSql,
                *snapshot.activeRequestId);
            const bool sameRequest =
                !active.empty() &&
                active[0][0].as<std::string>() == "resume_experiment" &&
                !active[0][1].is_null() &&
                active[0][1].as<long long>() == command.experimentId;
            const bool ownedByOther =
                sameRequest && active[0][3].as<bool>() &&
                !active[0][2].is_null() &&
                active[0][2].as<std::string>() != invocationIdentity;
            if (!sameRequest || ownedByOther || !willApply)
            {
                reject(ownedByOther
                    ? "administrative_request_application_in_progress"
                    : "conflicting_administrative_request_active");
                transaction.commit();
                return 1;
            }
            requestId = *snapshot.activeRequestId;
            retrying = true;
            const std::optional<DbTarget> persistedPlan =
                LoadSelectiveResumePlan(transaction, requestId);
            if (!persistedPlan || !persistedPlan->sourcePauseRequestId)
            {
                reject("stale_control_evidence");
                transaction.commit();
                return 1;
            }
            target = *persistedPlan;
            pauseRequestId = *target.sourcePauseRequestId;
            if (snapshot.desiredState != "paused" ||
                snapshot.currentPauseRequestId !=
                    std::optional<long long>{pauseRequestId})
            {
                reject("stale_control_evidence");
                transaction.commit();
                return 1;
            }
            const pqxx::result claimed = transaction.exec_params(
                "UPDATE experiment_admin_request r "
                "SET application_owner=$1,"
                "application_lease_until=now()+interval '30 seconds' "
                "FROM experiment_global_control c "
                "WHERE r.request_id=$2 AND r.action='resume_experiment' "
                "AND c.singleton AND c.active_request_id=r.request_id "
                "AND c.current_pause_request_id=$3 "
                "AND (r.application_owner=$1 "
                "OR r.application_lease_until IS NULL "
                "OR r.application_lease_until<=now());",
                invocationIdentity,
                requestId,
                pauseRequestId);
            RequireAffectedRows(
                claimed, 1, "claim_selective_resume_request");
            LoadAuthoritativeTargetState(
                transaction, requestId, Action::ResumeAll, target);
            alreadyAccounted =
                target.plan == "completed";
            deferredMachineOutput
                << "SCHEDULER_CONTROL_ATTEMPT,action=resume,experiment_id="
                << command.experimentId
                << ",current_status="
                << (loaded ? loaded->worker.lifecycleStatus : "missing")
                << ",current_phase="
                << (loaded ? loaded->worker.phase : target.worker.phase)
                << "\n";
            transaction.commit();
        }
        else
        {
            if (!loaded)
            {
                output << "SCHEDULER_CONTROL_REJECTED,action=resume,"
                       << "experiment_id=" << command.experimentId
                       << ",reason=experiment_not_found,"
                       << "result=invalid_lifecycle\n";
                transaction.commit();
                return 1;
            }
            target = *loaded;
            deferredMachineOutput
                << "SCHEDULER_CONTROL_ATTEMPT,action=resume,experiment_id="
                << target.worker.experimentId
                << ",current_status=" << target.worker.lifecycleStatus
                << ",current_phase=" << target.worker.phase << "\n";

            if (target.worker.lifecycleStatus == "paused")
            {
                deferredMachineOutput
                    << "Experiment " << target.worker.experimentId << "\n"
                    << "Current: status=paused phase="
                    << target.worker.phase
                    << "\nRequested: status=pending phase="
                    << target.worker.phase << "\n";
                if (command.dryRun)
                {
                    deferredMachineOutput
                        << "SCHEDULER_CONTROL_DRY_RUN,action=resume,"
                        << "experiment_id=" << target.worker.experimentId
                        << ",new_status=pending,new_phase="
                        << target.worker.phase
                        << ",result=lifecycle_resumed\n";
                    transaction.commit();
                    output << deferredMachineOutput.str();
                    return 0;
                }
                if (!command.confirmed)
                {
                    deferredMachineOutput << "Use --yes to apply.\n";
                    transaction.commit();
                    output << deferredMachineOutput.str();
                    return 0;
                }
                const pqxx::result lifecycleResumed =
                    transaction.exec_params(
                    "UPDATE experiment SET status='pending',updated_at=now() "
                    "WHERE experiment_id=$1 AND status='paused';",
                    target.worker.experimentId);
                RequireAffectedRows(
                    lifecycleResumed, 1, "resume_paused_experiment_lifecycle");
                deferredMachineOutput
                    << "SCHEDULER_CONTROL_APPLIED,action=resume,"
                    << "experiment_id=" << target.worker.experimentId
                    << ",new_status=pending,new_phase="
                    << target.worker.phase
                    << ",result=lifecycle_resumed\n";
                transaction.commit();
                output << deferredMachineOutput.str();
                return 0;
            }
            if (target.worker.lifecycleStatus != "running" ||
                !ActiveManagedPhase(target.worker.phase))
            {
                reject("invalid_lifecycle");
                transaction.commit();
                return 1;
            }
            if (!snapshot.currentPauseRequestId ||
                !target.workerGlobalPauseRequestId)
            {
                reject("not_globally_suspended");
                transaction.commit();
                return 1;
            }
            pauseRequestId = *snapshot.currentPauseRequestId;
            if (*target.workerGlobalPauseRequestId != pauseRequestId ||
                snapshot.desiredState != "paused")
            {
                reject("stale_control_evidence");
                transaction.commit();
                return 1;
            }
            if (!target.resumeBeforeAction)
            {
                if (const auto completedRequestId =
                        CompletedSelectiveReleaseRequestId(
                            transaction,
                            command.experimentId,
                            pauseRequestId))
                {
                    const SelectiveAccounting accounting =
                        LoadSelectiveAccounting(
                            transaction, *completedRequestId);
                    transaction.commit();
                    output << deferredMachineOutput.str();
                    PrintSelectiveResult(
                        output,
                        command.experimentId,
                        *completedRequestId,
                        pauseRequestId,
                        accounting,
                        true,
                        false);
                    return RequestExitCodeForPersistedStatus(
                        accounting.status);
                }
                reject("not_globally_suspended");
                transaction.commit();
                return 1;
            }
            if (!HasMatchingPauseEvidence(
                    transaction, target, pauseRequestId))
            {
                reject("stale_control_evidence");
                transaction.commit();
                return 1;
            }

            deferredMachineOutput
                << "SCHEDULER_CONTROL_TRANSITION,action=resume,"
                << "experiment_id=" << target.worker.experimentId
                << ",control_state=paused->running,"
                << "lifecycle_status=running"
                << ",phase=" << target.worker.phase
                << ",source_pause_request_id=" << pauseRequestId << "\n";
            if (command.dryRun)
            {
                const ValidatedWorker dryValidation =
                    ValidateManagedWorker(target.worker, processes);
                const std::string intendedSignal =
                    dryValidation.identity == IdentityResult::Validated &&
                            dryValidation.observation.stopped
                        ? "SIGCONT"
                        : "none";
                deferredMachineOutput
                    << "SCHEDULER_CONTROL_DRY_RUN,action=resume,"
                    << "experiment_id=" << target.worker.experimentId
                    << ",result=globally_suspended_worker_resumed"
                    << ",identity=" << ToString(dryValidation.identity)
                    << ",intended_signal=" << intendedSignal
                    << ",global_state=paused\n";
                transaction.commit();
                output << deferredMachineOutput.str();
                return 0;
            }
            if (!command.confirmed)
            {
                deferredMachineOutput << "Use --yes to apply.\n";
                transaction.commit();
                output << deferredMachineOutput.str();
                return 0;
            }

            target.sourcePauseRequestId = pauseRequestId;
            requestId = transaction.exec_params(
                "INSERT INTO experiment_admin_request ("
                "action,target_experiment_id,invocation_identity,"
                "requester_identity,application_owner,application_lease_until,"
                "previous_global_state,resulting_global_state,"
                "scheduler_running_observed,target_count) "
                "VALUES ('resume_experiment',$1,$2,$3,$2,"
                "now()+interval '30 seconds','paused','paused',$4,1) "
                "RETURNING request_id;",
                command.experimentId,
                invocationIdentity,
                command.requesterIdentity.value_or(CurrentRequester()),
                schedulerRunningObserved)[0][0].as<long long>();
            InsertOutcome(
                transaction, requestId, target, "planned", "none",
                "selective_global_pause_release_planned");
            const pqxx::result gated = transaction.exec_params(
                "UPDATE experiment_global_control SET active_request_id=$1,"
                "revision=revision+1,updated_at=now() "
                "WHERE singleton AND desired_state='paused' "
                "AND current_pause_request_id=$2 "
                "AND active_request_id IS NULL;",
                requestId,
                pauseRequestId);
            RequireAffectedRows(
                gated, 1, "activate_selective_resume_request");
            transaction.commit();
        }
    }

    if (retrying)
        deferredMachineOutput
            << "SCHEDULER_CONTROL_RETRY,action=resume,experiment_id="
            << command.experimentId << ",request_id=" << requestId
            << ",persisted_plan=1\n";

    SignalOutcome signal;
    bool signalAttempted = false;
    bool postSignalPredicateMismatch = false;
    if (!alreadyAccounted && target.worker.pid <= 0)
    {
        signal.identity = IdentityResult::ProcessMissing;
        signal.result = "process_missing";
        signal.detail = "globally_suspended_worker_pid_missing";
    }
    else if (!alreadyAccounted)
    {
        signal = target.frozenReplayTarget &&
                         !target.authoritativeExactMatch
            ? FrozenTargetAuthorizationFailure(target, processes)
            : ResumeWorkerAuthorized(
                  target.worker,
                  processes,
                  ExactAttemptSignalAuthorization(
                      connectionString,
                      requestId,
                      invocationIdentity,
                      "resume_experiment",
                      processes));
        signalAttempted = !signal.signals.empty();
    }

    SelectiveAccounting accounting;
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        AcquireCoordinationLock(transaction);
        if (!OwnsActiveRequest(
                transaction,
                requestId,
                invocationIdentity,
                "resume_experiment",
                pauseRequestId))
        {
            error << "SCHEDULER_CONTROL_REJECTED,action=resume,experiment_id="
                  << command.experimentId
                  << ",request_id=" << requestId
                  << ",reason=administrative_request_ownership_lost"
                  << ",result=administrative_request_ownership_lost"
                  << ",replay=" << (retrying ? 1 : 0)
                  << ",signal_attempted=" << (signalAttempted ? 1 : 0)
                  << "\n";
            transaction.commit();
            return 1;
        }

        if (!alreadyAccounted)
        {
            const auto exactSelectiveAttempt =
                target.authoritativeExactTerminalDeparture
                    ? std::optional<
                          EA::SchedulerOwnership::ExactAttemptSnapshot>{}
                    : LockExactTargetForMutation(
                          transaction, target, true);
            if (!exactSelectiveAttempt &&
                !target.authoritativeExactTerminalDeparture)
            {
                postSignalPredicateMismatch = true;
                signal.success = false;
                signal.identity =
                    IdentityResult::IdentityValidationFailed;
                signal.result = "identity_validation_failed";
                signal.detail =
                    "exact_active_attempt_changed_after_selective_resume";
            }
            std::ostringstream signalList;
            for (size_t i = 0; i < signal.signals.size(); ++i)
            {
                if (i)
                    signalList << "|";
                signalList << signal.signals[i];
            }
            if (signal.success && exactSelectiveAttempt)
            {
                const pqxx::result released = transaction.exec_params(
                    "UPDATE experiment SET worker_control_state='running',"
                    "updated_at=now() WHERE experiment_id=$1 "
                    "AND status='running' AND phase=$3 AND worker_pid=$4 "
                    "AND worker_process_group_id IS NOT DISTINCT FROM $5 "
                    "AND worker_process_start_identity IS NOT DISTINCT FROM $6 "
                    "AND worker_executable IS NOT DISTINCT FROM $7 "
                    "AND worker_command_line IS NOT DISTINCT FROM $8 "
                    "AND worker_control_state='paused' "
                    "AND worker_global_pause_request_id=$2 "
                    "AND active_scheduler_worker_attempt_id=$9;",
                    target.worker.experimentId,
                    pauseRequestId,
                    target.worker.phase,
                    target.worker.pid,
                    target.worker.processGroupId,
                    target.worker.processStartIdentity,
                    target.worker.executable,
                    target.worker.commandLine,
                    target.worker.workerAttemptId);
                if (released.affected_rows() != 1)
                {
                    postSignalPredicateMismatch = true;
                    signal.success = false;
                    signal.identity =
                        IdentityResult::IdentityValidationFailed;
                    signal.detail =
                        "worker_state_changed_after_validated_resume_signal";
                }
            }
            const bool safelyReconciled =
                signal.success;
            const pqxx::result outcomeUpdated = transaction.exec_params(
                "UPDATE experiment_admin_worker_outcome SET "
                "identity_result=$1,signal_result=$2,requested_signal=$3,"
                "outcome_status=$4,detail=$5,updated_at=now() "
                "WHERE request_id=$6 AND worker_identity=$7 "
                "AND outcome_status IN ('planned','failed','partial') "
                "AND EXISTS (SELECT 1 FROM experiment_admin_request r "
                "JOIN experiment_global_control c "
                "ON c.active_request_id=r.request_id "
                "WHERE c.singleton AND r.request_id=$6 "
                "AND r.application_owner=$8 "
                "AND r.action='resume_experiment' "
                "AND r.application_lease_until>now() "
                "AND c.current_pause_request_id=$9);",
                ToString(signal.identity),
                PersistedSignalResult(signal),
                signalList.str().empty()
                    ? std::optional<std::string>{}
                    : std::optional<std::string>{signalList.str()},
                safelyReconciled ? "completed" : "failed",
                signal.detail,
                requestId,
                target.workerIdentity,
                invocationIdentity,
                pauseRequestId);
            RequireAffectedRows(
                outcomeUpdated,
                1,
                "persist_selective_resume_outcome");
        }
        UpdateRequestAccounting(
            transaction,
            requestId,
            false,
            invocationIdentity,
            "resume_experiment");
        accounting = LoadSelectiveAccounting(transaction, requestId);
        const bool unresolved =
            HasUnresolvedWorkerOutcome(transaction, requestId);
        if (accounting.status == "completed" && !unresolved)
        {
            const pqxx::result gateCleared = transaction.exec_params(
                "UPDATE experiment_global_control c "
                "SET active_request_id=NULL,revision=revision+1,"
                "updated_at=now() "
                "FROM experiment_admin_request r "
                "WHERE c.singleton AND c.active_request_id=$1 "
                "AND c.current_pause_request_id=$3 "
                "AND r.request_id=c.active_request_id "
                "AND r.application_owner=$2 "
                "AND r.action='resume_experiment' "
                "AND r.application_lease_until>now();",
                requestId,
                invocationIdentity,
                pauseRequestId);
            RequireAffectedRows(
                gateCleared, 1, "clear_selective_resume_gate");
        }
        if (!postSignalPredicateMismatch)
        {
            const pqxx::result leaseCleared = transaction.exec_params(
                "UPDATE experiment_admin_request "
                "SET application_lease_until=NULL "
                "WHERE request_id=$1 AND application_owner=$2;",
                requestId,
                invocationIdentity);
            RequireAffectedRows(
                leaseCleared, 1, "release_selective_resume_lease");
        }
        transaction.commit();
    }

    output << deferredMachineOutput.str();
    PrintSelectiveResult(
        output,
        command.experimentId,
        requestId,
        pauseRequestId,
        accounting,
        retrying,
        signalAttempted);
    return RequestExitCodeForPersistedStatus(accounting.status);
}

} // namespace EA::GlobalExperimentControl
