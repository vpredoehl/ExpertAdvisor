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
        "ce.worker_control_state,ce.checkpoint_eval_id,"
        "ce.checkpoint_epoch,ce.checkpoint_model_id "
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
        if (!row[8].is_null())
            target.cancellationCheckpoint = row[8].as<int>();
        if (!row[9].is_null())
            target.cancellationCheckpointModelId = row[9].as<long long>();
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
        "cancellation_checkpoint_epoch,cancellation_checkpoint_model_id "
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
        if (!(*outcome)[4].is_null())
            target.cancellationCheckpointModelId =
                (*outcome)[4].as<long long>();
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
        "cancellation_checkpoint_epoch,cancellation_checkpoint_model_id,"
        "inference_action,outcome_status,detail) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15) "
        "ON CONFLICT (request_id,worker_identity) DO UPDATE SET "
        "worker_pid=EXCLUDED.worker_pid,"
        "worker_process_group_id=EXCLUDED.worker_process_group_id,"
        "worker_process_start_identity=EXCLUDED.worker_process_start_identity,"
        "cancellation_checkpoint_epoch=COALESCE("
        "experiment_admin_worker_outcome.cancellation_checkpoint_epoch,"
        "EXCLUDED.cancellation_checkpoint_epoch),"
        "cancellation_checkpoint_model_id=COALESCE("
        "experiment_admin_worker_outcome.cancellation_checkpoint_model_id,"
        "EXCLUDED.cancellation_checkpoint_model_id),"
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
        target.cancellationCheckpointModelId,
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
        transaction.exec_params(
            "UPDATE experiment_checkpoint_eval SET cancellation_request_id=$1,"
            "updated_at=now() WHERE checkpoint_eval_id=$2 "
            "AND cancellation_request_id IS NULL;",
            requestId,
            existing[0][0].as<long long>());
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
        "completed_at=CASE WHEN $2 OR totals.pending>0 THEN NULL "
        "ELSE COALESCE(r.completed_at,now()) END,"
        "result_summary=jsonb_build_object("
        "'target_count',totals.target_count,'successful_count',totals.successful,"
        "'already_satisfied_count',totals.already,'missing_count',totals.missing,"
        "'rejected_count',totals.rejected,'failed_count',totals.failed,"
        "'pending_count',totals.pending) "
        "FROM totals WHERE r.request_id=$1;",
        requestId,
        keepPending);
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

CheckpointStopRecordResult RecordCheckpointStopReached(
    pqxx::work& transaction,
    const std::optional<long long>& experimentId,
    int epoch,
    long long modelId)
{
    CheckpointStopRecordResult result;
    if (!experimentId)
    {
        result.detail = "experiment_id_missing";
        return result;
    }

    pqxx::result experiments = transaction.exec_params(
        "SELECT cancellation_request_id,cancel_infer_before,"
        "infer_start IS NOT NULL AND infer_end IS NOT NULL "
        "FROM experiment WHERE experiment_id=$1 FOR UPDATE;",
        *experimentId);
    if (experiments.empty())
    {
        result.detail = "experiment_missing";
        return result;
    }

    result.cancellationRequested = !experiments[0][0].is_null();
    if (!result.cancellationRequested)
    {
        pqxx::result updated = transaction.exec_params(
            "UPDATE experiment SET current_epoch=$1,worker_pid=NULL,"
            "worker_process_group_id=NULL,"
            "current_operation='checkpoint_stopped',"
            "stopped_at_checkpoint_epoch=$1,"
            "stopped_at_checkpoint_model_id=$2,last_model_id=$2,"
            "status='pending',"
            "phase=CASE WHEN infer_start IS NOT NULL AND infer_end IS NOT NULL "
            "THEN 'infer' ELSE 'analyze' END,exit_code=0,error_message=NULL,"
            "updated_at=now() WHERE experiment_id=$3 "
            "AND status='running' AND phase='train';",
            epoch,
            modelId,
            *experimentId);
        result.recorded = updated.affected_rows() == 1;
        result.detail = result.recorded
            ? "checkpoint_stopped"
            : "experiment_not_running_train";
        return result;
    }

    const long long requestId = experiments[0][0].as<long long>();
    result.cancellationRequestId = requestId;
    const bool inferBeforeCancel = experiments[0][1].as<bool>();
    result.inferenceRequested = inferBeforeCancel;
    const bool hasInferenceRange = experiments[0][2].as<bool>();
    pqxx::result outcomes = transaction.exec_params(
        "SELECT worker_identity,cancellation_checkpoint_epoch,"
        "cancellation_checkpoint_model_id,outcome_status "
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

    transaction.exec_params(
        "UPDATE experiment SET current_epoch=$1,worker_pid=NULL,"
        "worker_process_group_id=NULL,"
        "current_operation='cancel_checkpoint_reached',"
        "stopped_at_checkpoint_epoch=$1,"
        "stopped_at_checkpoint_model_id=$2,last_model_id=$2,"
        "status='cancelled',phase='train',exit_code=0,"
        "completed_at=COALESCE(completed_at,now()),"
        "cancellation_completed_at=COALESCE(cancellation_completed_at,now()),"
        "error_message='cancelled_at_requested_checkpoint',updated_at=now() "
        "WHERE experiment_id=$3 "
        "AND ((status='running' AND phase='train') "
        "OR (status='cancelled' AND phase='train' "
        "AND stopped_at_checkpoint_epoch=$1 "
        "AND stopped_at_checkpoint_model_id=$2));",
        epoch,
        modelId,
        *experimentId);
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
        ReconcileActiveCancellation(transaction);
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
    ReconcileActiveCancellation(transaction);
    result.detail = detail;
    return result;
}

void ReconcileActiveCancellation(pqxx::work& transaction)
{
    const ControlSnapshot snapshot = LoadControlSnapshot(transaction);
    if (!snapshot.activeRequestId || snapshot.activeAction != "cancel_all")
        return;
    const long long requestId = *snapshot.activeRequestId;

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
            ReconcileActiveCancellation(transaction);
            snapshot = LoadControlSnapshot(transaction);
        }
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
