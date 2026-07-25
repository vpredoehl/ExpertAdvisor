#include "GlobalExperimentControl.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <libproc.h>
#include <sstream>
#include <stdexcept>
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

std::string FirstCommandToken(const std::string& command)
{
    const size_t begin = command.find_first_not_of(" \t");
    if (begin == std::string::npos)
        return {};
    const size_t end = command.find_first_of(" \t", begin);
    return command.substr(begin, end == std::string::npos
                                     ? std::string::npos
                                     : end - begin);
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

class PosixProcessOperations final : public ProcessOperations
{
public:
    ProcessObservation Observe(int pid) override
    {
        ProcessObservation observation;
        observation.pid = pid;
        if (pid <= 0)
            return observation;

        errno = 0;
        if (::kill(static_cast<pid_t>(pid), 0) != 0)
        {
            if (errno == ESRCH)
            {
                observation.inspectionSucceeded = true;
                return observation;
            }
            if (errno == EPERM)
            {
                observation.exists = true;
                observation.permissionDenied = true;
                return observation;
            }
            return observation;
        }
        observation.exists = true;
        const std::optional<std::string> startIdentityBefore =
            ReadProcessStartIdentity(pid);
        if (!startIdentityBefore)
        {
            if (errno == EPERM)
                observation.permissionDenied = true;
            return observation;
        }

        const std::string command =
            "ps -p " + QuoteShellPid(pid) +
            " -o pid= -o pgid= -o state= -o command=";
        FILE* pipe = ::popen(command.c_str(), "r");
        if (pipe == nullptr)
            return observation;
        char buffer[16384] = {};
        std::string line;
        while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
            line += buffer;
        const int status = ::pclose(pipe);
        if (status != 0 || line.empty())
            return observation;

        std::istringstream input(line);
        int observedPid = -1;
        int processGroupId = -1;
        std::string processState;
        if (!(input >> observedPid >> processGroupId >> processState) ||
            observedPid != pid)
            return observation;
        if (!processState.empty() && processState[0] == 'Z')
        {
            observation.exists = false;
            observation.inspectionSucceeded = true;
            observation.processGroupId = processGroupId;
            return observation;
        }
        std::string observedCommand;
        std::getline(input, observedCommand);
        const size_t begin = observedCommand.find_first_not_of(" \t");
        if (begin != std::string::npos)
            observedCommand.erase(0, begin);
        observation.processGroupId = processGroupId;
        observation.stopped =
            !processState.empty() &&
            (processState[0] == 'T' || processState[0] == 't');
        observation.commandLine = observedCommand;
        observation.executable = FirstCommandToken(observedCommand);
        const std::optional<std::string> startIdentityAfter =
            ReadProcessStartIdentity(pid);
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
    bool signalImmediately = false;
    bool resumeBeforeAction = false;
    bool inferenceQueued = false;
    bool inferenceAlreadyCompleted = false;
    bool inferenceAlreadyRunning = false;
    bool inferenceFailed = false;
    bool inferenceRequested = false;
    bool latestCheckpointAmbiguous = false;
    std::string plan;
};

std::vector<DbTarget> LoadTargets(pqxx::transaction_base& transaction)
{
    pqxx::result rows = transaction.exec(
        "SELECT e.experiment_id, e.status, e.phase, e.worker_pid, "
        "e.worker_process_group_id, e.worker_executable, e.worker_command_line, "
        "e.worker_process_start_identity, "
        "e.worker_control_state, e.current_epoch, e.checkpoint_interval, "
        "e.target_epochs,e.last_checkpoint_stop_decision_epoch,"
        "e.infer_start IS NOT NULL AND e.infer_end IS NOT NULL, "
        "cp.completed_epoch, cp.model_id,cp.same_epoch_count "
        "FROM experiment e "
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
            target.currentEpoch = row[9].as<int>();
        target.checkpointInterval = row[10].as<int>();
        target.targetEpochs = row[11].as<int>();
        if (!row[12].is_null())
            target.lastCheckpointStopDecisionEpoch = row[12].as<int>();
        target.hasInferenceRange = row[13].as<bool>();
        if (!row[14].is_null())
            target.latestCheckpointEpoch = row[14].as<int>();
        if (!row[15].is_null())
            target.latestCheckpointModelId = row[15].as<long long>();
        target.latestCheckpointAmbiguous =
            !row[16].is_null() && row[16].as<int>() > 1;
        targets.push_back(std::move(target));
    }

    pqxx::result checkpointWorkers = transaction.exec(
        "SELECT COALESCE(ce.parent_experiment_id,ce.experiment_id),"
        "ce.worker_pid,ce.worker_process_group_id,ce.worker_executable,"
        "ce.worker_command_line,ce.worker_process_start_identity,"
        "ce.worker_control_state,ce.checkpoint_eval_id "
        "FROM experiment_checkpoint_eval ce "
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
        target.checkpointEvalId = row[7].as<long long>();
        target.worker.checkpointEvalId = target.checkpointEvalId;
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
    std::vector<DbTarget> targets = LoadTargets(transaction);
    pqxx::result outcomes = transaction.exec_params(
        "SELECT worker_identity,outcome_status,inference_action,"
        "cancellation_checkpoint_epoch "
        "FROM experiment_admin_worker_outcome WHERE request_id=$1;",
        requestId);

    std::vector<DbTarget> retryTargets;
    retryTargets.reserve(outcomes.size());
    for (DbTarget& target : targets)
    {
        const auto outcome = std::find_if(
            outcomes.begin(),
            outcomes.end(),
            [&target](const pqxx::row& row) {
                return row[0].as<std::string>() == target.workerIdentity;
            });
        if (outcome == outcomes.end())
            continue;

        const std::string outcomeStatus = (*outcome)[1].as<std::string>();
        if (!(*outcome)[3].is_null())
            target.cancellationCheckpoint = (*outcome)[3].as<int>();
        if (outcomeStatus == "planned")
        {
            target.signalImmediately = action == Action::CancelAll;
        }
        else if (outcomeStatus == "pending_checkpoint")
        {
            target.plan = "pending_checkpoint";
        }
        else
        {
            target.plan = "already_accounted";
        }
        const std::string inferenceAction = (*outcome)[2].as<std::string>();
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
        retryTargets.push_back(std::move(target));
    }
    return retryTargets;
}

void InsertOutcome(pqxx::transaction_base& transaction,
                   long long requestId,
                   const DbTarget& target,
                   const std::string& outcomeStatus,
                   const std::string& inferenceAction,
                   const std::string& detail)
{
    transaction.exec_params(
        "INSERT INTO experiment_admin_worker_outcome ("
        "request_id,worker_identity,experiment_id,checkpoint_eval_id,"
        "worker_kind,phase,lifecycle_status,worker_pid,"
        "worker_process_group_id,worker_process_start_identity,"
        "cancellation_checkpoint_epoch,"
        "inference_action,outcome_status,detail) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14) "
        "ON CONFLICT (request_id,worker_identity) DO UPDATE SET "
        "worker_pid=EXCLUDED.worker_pid,"
        "worker_process_group_id=EXCLUDED.worker_process_group_id,"
        "worker_process_start_identity=EXCLUDED.worker_process_start_identity,"
        "cancellation_checkpoint_epoch=EXCLUDED.cancellation_checkpoint_epoch,"
        "inference_action=EXCLUDED.inference_action,"
        "outcome_status=EXCLUDED.outcome_status,"
        "detail=EXCLUDED.detail,updated_at=now();",
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
        target.cancellationCheckpoint,
        inferenceAction,
        outcomeStatus,
        detail);
}

std::optional<long long> QueueCancellationInference(
    pqxx::transaction_base& transaction,
    long long requestId,
    DbTarget& target)
{
    if (!target.latestCheckpointEpoch || !target.latestCheckpointModelId ||
        !target.hasInferenceRange || target.latestCheckpointAmbiguous)
        return std::nullopt;

    pqxx::result existing = transaction.exec_params(
        "SELECT checkpoint_eval_id,status,phase,cancellation_request_id "
        "FROM experiment_checkpoint_eval "
        "WHERE COALESCE(parent_experiment_id,experiment_id)=$1 "
        "AND checkpoint_epoch=$2 AND checkpoint_model_id=$3 "
        "ORDER BY checkpoint_eval_id LIMIT 1;",
        target.worker.experimentId,
        *target.latestCheckpointEpoch,
        *target.latestCheckpointModelId);
    if (!existing.empty())
    {
        const std::string status = existing[0][1].as<std::string>();
        const std::string phase = existing[0][2].as<std::string>();
        const bool belongsToOtherCancellation =
            !existing[0][3].is_null() &&
            existing[0][3].as<long long>() != requestId;
        target.inferenceFailed =
            status == "failed" || status == "skipped" ||
            belongsToOtherCancellation ||
            (status == "running" && phase != "infer");
        target.inferenceAlreadyCompleted =
            !target.inferenceFailed &&
            (status == "completed" || phase == "analyze" ||
             phase == "done");
        target.inferenceAlreadyRunning =
            status == "running" && phase == "infer";
        if (belongsToOtherCancellation)
            return existing[0][0].as<long long>();
        transaction.exec_params(
            "UPDATE experiment_checkpoint_eval "
            "SET cancellation_request_id=$1,"
            "status=CASE WHEN $3 THEN 'completed' ELSE status END,"
            "phase=CASE WHEN $3 THEN 'done' ELSE phase END,"
            "completed_at=CASE WHEN $3 THEN COALESCE(completed_at,now()) "
            "ELSE completed_at END,"
            "updated_at=now() WHERE checkpoint_eval_id=$2;",
            requestId,
            existing[0][0].as<long long>(),
            target.inferenceAlreadyCompleted);
        target.inferenceQueued =
            !target.inferenceAlreadyCompleted && !target.inferenceFailed;
        return existing[0][0].as<long long>();
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
        return std::nullopt;
    target.inferenceQueued = true;
    return inserted[0][0].as<long long>();
}

void UpdateSignalOutcome(pqxx::transaction_base& transaction,
                         long long requestId,
                         const DbTarget& target,
                         const SignalOutcome& outcome,
                         bool cancelled)
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
        outcome.identity == IdentityResult::ProcessMissing ||
        outcome.identity == IdentityResult::StalePid ||
        outcome.identity == IdentityResult::IdentityValidationFailed;
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
    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome SET "
        "identity_result=$1,signal_result=$2,requested_signal=$3,"
        "outcome_status=$4,detail=$5,updated_at=now() "
        "WHERE request_id=$6 AND worker_identity=$7;",
        ToString(outcome.identity),
        persistedSignalResult,
        signalList.str().empty() ? std::optional<std::string>{}
                                 : std::optional<std::string>{signalList.str()},
        reconciled ? reconciledStatus : (cancelled ? "planned" : "failed"),
        outcome.detail,
        requestId,
        target.workerIdentity);
}

void UpdateRequestAccounting(pqxx::transaction_base& transaction,
                             long long requestId,
                             bool keepPending)
{
    transaction.exec_params(
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
        "completed_at=CASE WHEN $2 OR totals.pending>0 THEN NULL ELSE now() END,"
        "result_summary=jsonb_build_object("
        "'target_count',totals.target_count,'successful_count',totals.successful,"
        "'already_satisfied_count',totals.already,'missing_count',totals.missing,"
        "'rejected_count',totals.rejected,'failed_count',totals.failed,"
        "'pending_count',totals.pending) "
        "FROM totals WHERE r.request_id=$1;",
        requestId,
        keepPending);
}

} // namespace

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
        worker.executable->empty() || !worker.processStartIdentity ||
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
    if (FirstCommandToken(*worker.executable) != result.observation.executable &&
        *worker.executable != result.observation.executable)
    {
        result.identity = IdentityResult::IdentityValidationFailed;
        result.detail = "executable_identity_mismatch";
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

SignalOutcome PauseWorker(const ManagedWorker& worker,
                          ProcessOperations& processes)
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
    outcome.signals.push_back(SIGSTOP);
    if (!processes.SignalProcessGroup(
            validated.observation.processGroupId, SIGSTOP, errorNumber))
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
            : "signaling_failure";
        outcome.detail = std::strerror(errorNumber);
        return outcome;
    }
    outcome.result = "signaled";
    outcome.success = true;
    return outcome;
}

SignalOutcome ResumeWorker(const ManagedWorker& worker,
                           ProcessOperations& processes)
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
    outcome.signals.push_back(SIGCONT);
    if (!processes.SignalProcessGroup(
            validated.observation.processGroupId, SIGCONT, errorNumber))
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
            : "signaling_failure";
        outcome.detail = std::strerror(errorNumber);
        return outcome;
    }
    outcome.result = "signaled";
    outcome.success = true;
    return outcome;
}

SignalOutcome CancelWorker(const ManagedWorker& worker,
                           bool resumeFirst,
                           std::chrono::milliseconds grace,
                           ProcessOperations& processes)
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
    if (resumeFirst)
    {
        outcome.signals.push_back(SIGCONT);
        if (!processes.SignalProcessGroup(
                validated.observation.processGroupId, SIGCONT, errorNumber))
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
                : "signaling_failure";
            outcome.detail = std::strerror(errorNumber);
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
    if (!processes.SignalProcessGroup(
            validated.observation.processGroupId, SIGTERM, errorNumber))
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
            : "signaling_failure";
        outcome.detail = std::strerror(errorNumber);
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
    if (!processes.SignalProcessGroup(
            beforeKill.observation.processGroupId, SIGKILL, errorNumber))
    {
        outcome.result = errorNumber == EPERM
            ? "permission_failure"
            : "signaling_failure";
        outcome.detail = std::strerror(errorNumber);
        return outcome;
    }
    outcome.result = "escalated";
    outcome.success = processes.WaitForProcessGroupExit(
        beforeKill.observation.processGroupId, grace);
    if (!outcome.success)
        outcome.detail = "process_still_exists_after_sigkill";
    return outcome;
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
        "SELECT c.desired_state,c.active_request_id,r.action,"
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
        snapshot.activeAction = rows[0][2].as<std::string>();
    if (!rows[0][3].is_null())
        snapshot.cancellationMode = rows[0][3].as<std::string>();
    snapshot.inferBeforeCancel = rows[0][4].as<bool>();
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

void ReconcileActiveCancellation(pqxx::work& transaction)
{
    const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
    if (!snapshot.activeRequestId || snapshot.activeAction != "cancel_all")
        return;
    const long long requestId = *snapshot.activeRequestId;

    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome o SET "
        "outcome_status='completed',inference_action='completed',"
        "detail='cancellation_checkpoint_inference_completed',updated_at=now() "
        "FROM experiment_checkpoint_eval ce "
        "WHERE o.request_id=$1 AND o.outcome_status='awaiting_inference' "
        "AND o.worker_kind='experiment' "
        "AND ce.cancellation_request_id=o.request_id "
        "AND COALESCE(ce.parent_experiment_id,ce.experiment_id)=o.experiment_id "
        "AND ce.status='completed';",
        requestId);
    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome o SET "
        "outcome_status='partial',inference_action='failed',"
        "detail=COALESCE(ce.error_message,'cancellation_checkpoint_inference_failed'),"
        "updated_at=now() "
        "FROM experiment_checkpoint_eval ce "
        "WHERE o.request_id=$1 AND o.outcome_status='awaiting_inference' "
        "AND o.worker_kind='experiment' "
        "AND ce.cancellation_request_id=o.request_id "
        "AND COALESCE(ce.parent_experiment_id,ce.experiment_id)=o.experiment_id "
        "AND ce.status='failed';",
        requestId);
    transaction.exec_params(
        "UPDATE experiment_admin_worker_outcome o SET "
        "outcome_status=CASE WHEN e.cancel_infer_before "
        " THEN 'awaiting_inference' ELSE 'completed' END,"
        "inference_action=CASE WHEN e.cancel_infer_before "
        " THEN 'queued' ELSE inference_action END,"
        "detail='cancellation_checkpoint_reached',updated_at=now() "
        "FROM experiment e WHERE o.request_id=$1 "
        "AND o.worker_kind='experiment' "
        "AND o.experiment_id=e.experiment_id "
        "AND o.outcome_status='pending_checkpoint' "
        "AND e.status='cancelled';",
        requestId);

    UpdateRequestAccounting(transaction, requestId, false);
    pqxx::result request = transaction.exec_params(
        "SELECT status FROM experiment_admin_request WHERE request_id=$1;",
        requestId);
    if (!request.empty() && request[0][0].as<std::string>() != "pending")
    {
        transaction.exec_params(
            "UPDATE experiment_global_control SET active_request_id=NULL,"
            "revision=revision+1,updated_at=now() "
            "WHERE singleton=true AND active_request_id=$1;",
            requestId);
    }
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
    if (command.dryRun)
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        AcquireCoordinationLock(transaction);
        const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
        previousState = snapshot.desiredState;
        targets = LoadTargets(transaction);
        output << "GLOBAL_EXPERIMENT_CONTROL_DRY_RUN,action="
               << ToString(command.action)
               << ",previous_state=" << previousState
               << ",resulting_state=" << ActionState(command.action)
               << ",targets=" << targets.size() << "\n";
        for (DbTarget& target : targets)
        {
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
        const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
        if (snapshot.activeRequestId)
        {
            const bool sameRequestShape =
                snapshot.activeAction &&
                *snapshot.activeAction == ToString(command.action) &&
                snapshot.inferBeforeCancel == command.inferBeforeCancel &&
                ((!command.cancellationMode &&
                  !snapshot.cancellationMode) ||
                 (command.cancellationMode &&
                  snapshot.cancellationMode &&
                  *snapshot.cancellationMode ==
                      ToString(*command.cancellationMode)));
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
            transaction.exec_params(
                "UPDATE experiment_admin_request SET application_owner=$1,"
                "application_lease_until=now()+interval '30 seconds' "
                "WHERE request_id=$2;",
                invocationIdentity,
                requestId);
            transaction.exec_params(
                "UPDATE experiment_admin_worker_outcome o SET "
                "identity_result='process_missing',"
                "signal_result='process_missing',"
                "outcome_status=CASE WHEN $2='cancel_all' "
                "AND o.inference_action IN ('queued','running') "
                "THEN 'awaiting_inference' ELSE 'completed' END,"
                "detail='worker_became_inactive_before_request_retry',"
                "updated_at=now() WHERE o.request_id=$1 "
                "AND o.outcome_status='planned' AND ("
                " (o.worker_kind='experiment' AND NOT EXISTS ("
                "   SELECT 1 FROM experiment e "
                "   WHERE e.experiment_id=o.experiment_id "
                "   AND e.status='running')) "
                " OR (o.worker_kind='checkpoint_infer' AND NOT EXISTS ("
                "   SELECT 1 FROM experiment_checkpoint_eval ce "
                "   WHERE ce.checkpoint_eval_id=o.checkpoint_eval_id "
                "   AND ce.status='running' AND ce.phase='infer')));",
                requestId,
                ToString(command.action));
            targets =
                LoadRetryTargets(transaction, requestId, command.action);
            transaction.commit();
        }
        else
        {
            previousState = snapshot.desiredState;
            targets = LoadTargets(transaction);

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

            transaction.exec_params(
                "UPDATE experiment_global_control SET desired_state=$1,"
                "active_request_id=$2,revision=revision+1,updated_at=now() "
                "WHERE singleton=true;",
                ActionState(command.action),
                requestId);

            for (DbTarget& target : targets)
            {
            if (command.action == Action::PauseAll ||
                command.action == Action::ResumeAll)
            {
                if (target.worker.lifecycleStatus != "running")
                    continue;
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
                transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval "
                    "SET cancellation_request_id=$1,updated_at=now() "
                    "WHERE checkpoint_eval_id=$2;",
                    requestId,
                    *target.checkpointEvalId);
                InsertOutcome(
                    transaction,
                    requestId,
                    target,
                    "planned",
                    "running",
                    "active_checkpoint_inference_cancellation");
                continue;
            }

            transaction.exec_params(
                "UPDATE experiment SET cancellation_request_id=$1,"
                "cancel_infer_before=$2,updated_at=now() "
                "WHERE experiment_id=$3;",
                requestId,
                command.inferBeforeCancel,
                target.worker.experimentId);
            if (target.worker.lifecycleStatus != "running")
            {
                transaction.exec_params(
                    "UPDATE experiment SET status='cancelled',"
                    "completed_at=COALESCE(completed_at,now()),"
                    "cancellation_completed_at=now(),worker_pid=NULL,"
                    "worker_process_group_id=NULL,updated_at=now() "
                    "WHERE experiment_id=$1 "
                    "AND status IN ('pending','paused');",
                    target.worker.experimentId);
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
                transaction.exec_params(
                    "UPDATE experiment SET stop_after_checkpoint_epoch=$1,"
                    "cancel_after_checkpoint_epoch=$1,updated_at=now() "
                    "WHERE experiment_id=$2;",
                    *target.cancellationCheckpoint,
                    target.worker.experimentId);
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

    if (retryingPersistedRequest)
        output << "GLOBAL_EXPERIMENT_CONTROL_RETRY,request_id="
               << requestId << ",persisted_plan=1\n";

    int failed = 0;
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
            if (target.resumeBeforeAction)
                resume = ResumeWorker(target.worker, processes);
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
            if (!resume.success &&
                resume.identity != IdentityResult::ProcessMissing)
                ++failed;
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
            transaction.exec_params(
                "UPDATE experiment_admin_request "
                "SET application_lease_until=now()+interval '30 seconds' "
                "WHERE request_id=$1 AND application_owner=$2;",
                requestId,
                invocationIdentity);
            transaction.exec_params(
                "UPDATE experiment_admin_worker_outcome SET "
                "identity_result=$1,signal_result=$2,"
                "detail=$3,updated_at=now() "
                "WHERE request_id=$4 AND worker_identity=$5;",
                ToString(resume.identity),
                persistedResumeResult,
                resume.success
                    ? (target.resumeBeforeAction
                           ? "resumed_to_reach_cancellation_checkpoint"
                           : "running_to_cancellation_checkpoint_validated")
                    : resume.detail,
                requestId,
                target.workerIdentity);
            if (resume.success)
            {
                transaction.exec_params(
                    "UPDATE experiment SET worker_control_state='running',"
                    "updated_at=now() WHERE experiment_id=$1 "
                    "AND status='running' AND worker_pid=$2;",
                    target.worker.experimentId,
                    target.worker.pid);
            }
            else if (resume.identity == IdentityResult::ProcessMissing ||
                     resume.identity == IdentityResult::StalePid ||
                     resume.identity ==
                         IdentityResult::IdentityValidationFailed)
            {
                if (target.latestCheckpointModelId)
                {
                    transaction.exec_params(
                        "UPDATE experiment SET status='pending',phase='train',"
                        "worker_pid=NULL,worker_process_group_id=NULL,"
                        "worker_control_state='running',last_model_id=$1,"
                        "current_operation='cancel_checkpoint_restart_pending',"
                        "error_message='cancellation_worker_restart_required',"
                        "updated_at=now() "
                        "WHERE experiment_id=$2 AND status='running';",
                        *target.latestCheckpointModelId,
                        target.worker.experimentId);
                    transaction.exec_params(
                        "UPDATE experiment_admin_worker_outcome SET "
                        "outcome_status='pending_checkpoint',"
                        "detail='missing_worker_requeued_from_durable_checkpoint',"
                        "updated_at=now() WHERE request_id=$1 "
                        "AND worker_identity=$2;",
                        requestId,
                        target.workerIdentity);
                }
                else
                {
                    transaction.exec_params(
                        "UPDATE experiment SET status='cancelled',"
                        "worker_pid=NULL,worker_process_group_id=NULL,"
                        "completed_at=now(),cancellation_completed_at=now(),"
                        "error_message='cancelled_missing_worker_no_checkpoint',"
                        "updated_at=now() WHERE experiment_id=$1 "
                        "AND status='running';",
                        target.worker.experimentId);
                    transaction.exec_params(
                        "UPDATE experiment_admin_worker_outcome SET "
                        "outcome_status='partial',"
                        "detail='missing_worker_no_restart_checkpoint',"
                        "updated_at=now() WHERE request_id=$1 "
                        "AND worker_identity=$2;",
                        requestId,
                        target.workerIdentity);
                }
            }
            transaction.commit();
            continue;
        }
        if (target.worker.lifecycleStatus != "running")
            continue;
        if (target.plan == "already_satisfied" ||
            target.plan == "already_accounted")
            continue;

        SignalOutcome signal;
        if (command.action == Action::PauseAll)
            signal = PauseWorker(target.worker, processes);
        else if (command.action == Action::ResumeAll)
            signal = ResumeWorker(target.worker, processes);
        else
            signal = CancelWorker(target.worker,
                                  target.resumeBeforeAction,
                                  command.terminationGrace,
                                  processes);
        if (!signal.success &&
            signal.identity != IdentityResult::ProcessMissing)
            ++failed;

        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        AcquireCoordinationLock(transaction);
        transaction.exec_params(
            "UPDATE experiment_admin_request "
            "SET application_lease_until=now()+interval '30 seconds' "
            "WHERE request_id=$1 AND application_owner=$2;",
            requestId,
            invocationIdentity);
        UpdateSignalOutcome(transaction, requestId, target, signal,
                            command.action == Action::CancelAll);
        if (command.action != Action::CancelAll &&
            signal.identity == IdentityResult::ProcessMissing)
        {
            if (target.checkpointWorker)
                transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval "
                    "SET worker_pid=NULL,worker_process_group_id=NULL,"
                    "worker_control_state='running',updated_at=now() "
                    "WHERE checkpoint_eval_id=$1 AND status='running' "
                    "AND worker_pid=$2;",
                    *target.checkpointEvalId,
                    target.worker.pid);
            else
                transaction.exec_params(
                    "UPDATE experiment SET worker_pid=NULL,"
                    "worker_process_group_id=NULL,"
                    "worker_control_state='running',updated_at=now() "
                    "WHERE experiment_id=$1 AND status='running' "
                    "AND worker_pid=$2;",
                    target.worker.experimentId,
                    target.worker.pid);
        }
        if (command.action == Action::PauseAll && signal.success)
        {
            if (target.checkpointWorker)
                transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval "
                    "SET worker_control_state='paused',updated_at=now() "
                    "WHERE checkpoint_eval_id=$1 AND status='running' "
                    "AND worker_pid=$2;",
                    *target.checkpointEvalId,
                    target.worker.pid);
            else
                transaction.exec_params(
                    "UPDATE experiment SET worker_control_state='paused',"
                    "updated_at=now() WHERE experiment_id=$1 "
                    "AND status='running' AND worker_pid=$2;",
                    target.worker.experimentId,
                    target.worker.pid);
        }
        else if (command.action == Action::ResumeAll && signal.success)
        {
            if (target.checkpointWorker)
                transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval "
                    "SET worker_control_state='running',updated_at=now() "
                    "WHERE checkpoint_eval_id=$1 AND status='running' "
                    "AND worker_pid=$2;",
                    *target.checkpointEvalId,
                    target.worker.pid);
            else
                transaction.exec_params(
                    "UPDATE experiment SET worker_control_state='running',"
                    "updated_at=now() WHERE experiment_id=$1 "
                    "AND status='running' AND worker_pid=$2;",
                    target.worker.experimentId,
                    target.worker.pid);
        }
        else if (command.action == Action::CancelAll &&
                 (signal.success ||
                  signal.identity == IdentityResult::ProcessMissing ||
                  signal.identity == IdentityResult::StalePid ||
                  signal.identity ==
                      IdentityResult::IdentityValidationFailed))
        {
            if (target.checkpointWorker)
            {
                transaction.exec_params(
                    "UPDATE experiment_checkpoint_eval SET status='failed',"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "completed_at=now(),updated_at=now(),"
                    "error_message='cancelled_by_global_request' "
                    "WHERE checkpoint_eval_id=$1 AND status='running';",
                    *target.checkpointEvalId);
            }
            else
            {
                transaction.exec_params(
                    "UPDATE experiment SET status='cancelled',"
                    "completed_at=COALESCE(completed_at,now()),"
                    "cancellation_completed_at=now(),"
                    "worker_pid=NULL,worker_process_group_id=NULL,"
                    "current_operation='cancelled_by_global_request',"
                    "error_message=CASE WHEN $1 THEN "
                    "'cancelled_after_sigkill' ELSE 'cancelled_by_global_request' END,"
                    "updated_at=now() WHERE experiment_id=$2 "
                    "AND status='running';",
                    signal.result == "escalated",
                    target.worker.experimentId);
            }
        }
        transaction.commit();
    }

    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION READ WRITE;");
        AcquireCoordinationLock(transaction);
        ReconcileActiveCancellation(transaction);
        const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
        if (command.action != Action::CancelAll)
        {
            UpdateRequestAccounting(transaction, requestId, false);
            transaction.exec_params(
                "UPDATE experiment_global_control SET active_request_id=NULL,"
                "revision=revision+1,updated_at=now() "
                "WHERE singleton=true AND active_request_id=$1;",
                requestId);
        }
        else if (snapshot.activeRequestId &&
                 *snapshot.activeRequestId == requestId)
        {
            UpdateRequestAccounting(transaction, requestId, false);
            pqxx::result state = transaction.exec_params(
                "SELECT status FROM experiment_admin_request "
                "WHERE request_id=$1;",
                requestId);
            if (!state.empty() &&
                state[0][0].as<std::string>() != "pending")
            {
                transaction.exec_params(
                    "UPDATE experiment_global_control "
                    "SET active_request_id=NULL,revision=revision+1,"
                    "updated_at=now() WHERE singleton=true "
                    "AND active_request_id=$1;",
                    requestId);
            }
        }
        pqxx::result summary = transaction.exec_params(
            "SELECT status,target_count,successful_count,"
            "already_satisfied_count,missing_count,rejected_count,failed_count "
            "FROM experiment_admin_request WHERE request_id=$1;",
            requestId);
        transaction.exec_params(
            "UPDATE experiment_admin_request SET application_lease_until=NULL "
            "WHERE request_id=$1 AND application_owner=$2;",
            requestId,
            invocationIdentity);
        if (!summary.empty())
        {
            output << "GLOBAL_EXPERIMENT_CONTROL_SUMMARY,request_id="
                   << requestId
                   << ",action=" << ToString(command.action)
                   << ",status=" << summary[0][0].as<std::string>()
                   << ",target_count=" << summary[0][1].as<int>()
                   << ",successful_count=" << summary[0][2].as<int>()
                   << ",already_satisfied_count=" << summary[0][3].as<int>()
                   << ",missing_count=" << summary[0][4].as<int>()
                   << ",rejected_count=" << summary[0][5].as<int>()
                   << ",failed_count=" << summary[0][6].as<int>()
                   << ",global_state=" << ActionState(command.action)
                   << "\n";
        }
        pqxx::result outcomes = transaction.exec_params(
            "SELECT experiment_id,phase,COALESCE(worker_pid::text,'NULL'),"
            "identity_result,signal_result,"
            "COALESCE(cancellation_checkpoint_epoch::text,'NULL'),"
            "inference_action,outcome_status,COALESCE(detail,'') "
            "FROM experiment_admin_worker_outcome WHERE request_id=$1 "
            "ORDER BY experiment_id;",
            requestId);
        for (const auto& row : outcomes)
        {
            output << "GLOBAL_EXPERIMENT_CONTROL_OUTCOME,request_id="
                   << requestId
                   << ",experiment_id=" << row[0].as<long long>()
                   << ",phase=" << row[1].as<std::string>()
                   << ",pid=" << row[2].as<std::string>()
                   << ",identity=" << row[3].as<std::string>()
                   << ",signal_result=" << row[4].as<std::string>()
                   << ",checkpoint_target=" << row[5].as<std::string>()
                   << ",inference_action=" << row[6].as<std::string>()
                   << ",outcome_status=" << row[7].as<std::string>()
                   << ",detail=" << row[8].as<std::string>() << "\n";
        }
        transaction.commit();
    }
    return failed == 0 ? 0 : 1;
}

} // namespace EA::GlobalExperimentControl
