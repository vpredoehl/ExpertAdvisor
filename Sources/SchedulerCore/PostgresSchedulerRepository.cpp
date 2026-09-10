#include "PostgresSchedulerRepository.hpp"

#include <stdexcept>

namespace EA::SchedulerCore
{
namespace
{

template <typename T>
std::optional<T> OptionalCell(const pqxx::row& row, int index)
{
    if (row[index].is_null())
        return std::nullopt;
    return row[index].as<T>();
}

SchedulerExperimentRecord MapExperiment(const pqxx::row& row)
{
    SchedulerExperimentRecord record;
    record.experimentId = row[0].as<long long>();
    record.symbol = row[1].as<std::string>();
    record.predictionHorizon = row[2].as<int>();
    record.cNextThreshold = row[3].as<double>();
    record.coreLrMult = OptionalCell<double>(row, 4);
    record.headLrMult = OptionalCell<double>(row, 5);
    record.targetEpochs = row[6].as<int>();
    record.checkpointInterval = row[7].as<int>();
    record.trainStart = row[8].as<std::string>();
    record.trainEnd = row[9].as<std::string>();
    record.inferStart = OptionalCell<std::string>(row, 10);
    record.inferEnd = OptionalCell<std::string>(row, 11);
    record.lastModelId = OptionalCell<long long>(row, 12);
    record.resumeModelId = OptionalCell<long long>(row, 13);
    record.trainLogPath = OptionalCell<std::string>(row, 14);
    record.inferLogPath = OptionalCell<std::string>(row, 15);
    record.analysisLogPath = OptionalCell<std::string>(row, 16);
    record.donchian20Mode = row[17].as<std::string>();
    record.featureWarmupScope = row[18].as<std::string>();
    record.donchianLookback = row[19].as<std::string>();
    record.featureAblationMask = row[20].as<std::string>();
    record.resumeExpandInputWidth = row[21].as<bool>();
    record.trainingObjectiveCanonical = row[22].as<std::string>();
    record.trainingObjectiveHash = row[23].as<std::string>();
    return record;
}

constexpr const char* kExperimentProjection =
    "experiment_id, symbol, prediction_horizon, c_next_threshold, "
    "core_lr_mult, head_lr_mult, target_epochs, checkpoint_interval, "
    "train_start::text, train_end::text, infer_start::text, infer_end::text, "
    "last_model_id, resume_model_id, train_log_path, infer_log_path, "
    "analysis_log_path, donchian20_mode, feature_warmup_scope, "
    "donchian_lookback, feature_ablation_mask, resume_expand_input_width, "
    "training_objective_canonical, training_objective_hash";

} // namespace

PostgresSchedulerRepository::PostgresSchedulerRepository(
    pqxx::transaction_base& transaction)
    : transaction_{transaction}
{
}

std::vector<PendingSchedulerExperimentRecord>
PostgresSchedulerRepository::loadPendingExperiments(
    std::string_view phase,
    bool cancellationOnly)
{
    std::string sql =
        "SELECT " + std::string{kExperimentProjection} + ","
        "scheduler_priority,resume_requested,scheduler_resume_origin,"
        "active_scheduler_worker_attempt_id "
        "FROM experiment "
        "WHERE status = 'pending' AND phase = $1 ";
    if (cancellationOnly)
    {
        sql +=
            "AND cancellation_request_id=("
            " SELECT active_request_id FROM experiment_global_control "
            " WHERE singleton=true) "
            "AND cancel_after_checkpoint_epoch IS NOT NULL ";
    }
    sql +=
        "ORDER BY CASE scheduler_priority WHEN 'high' THEN 0 "
        "WHEN 'normal' THEN 1 ELSE 2 END ASC,"
        "CASE scheduler_resume_origin WHEN 'operator' THEN 0 "
        "WHEN 'preemption' THEN 1 ELSE 2 END ASC,"
        "updated_at ASC,experiment_id ASC;";

    const pqxx::result rows = transaction_.exec(
        sql, pqxx::params{phase});
    std::vector<PendingSchedulerExperimentRecord> records;
    records.reserve(rows.size());
    for (const auto& row : rows)
    {
        PendingSchedulerExperimentRecord record;
        record.experiment = MapExperiment(row);
        record.schedulerPriority = row[24].as<std::string>();
        record.resumeRequested = row[25].as<bool>();
        record.schedulerResumeOrigin = row[26].as<std::string>();
        record.activeWorkerAttemptId = OptionalCell<long long>(row, 27);
        records.push_back(std::move(record));
    }
    return records;
}

std::vector<RunningSchedulerExperimentRecord>
PostgresSchedulerRepository::loadRunningExperiments()
{
    const pqxx::result rows = transaction_.exec(
        "SELECT " + std::string{kExperimentProjection} + ", phase, worker_pid, "
        "extract(epoch from COALESCE(worker_started_at, updated_at))::double precision "
        "FROM experiment WHERE status = 'running' "
        "ORDER BY updated_at ASC, experiment_id ASC;");

    std::vector<RunningSchedulerExperimentRecord> records;
    records.reserve(rows.size());
    for (const auto& row : rows)
    {
        records.push_back(RunningSchedulerExperimentRecord{
            MapExperiment(row),
            row[24].as<std::string>(),
            OptionalCell<int>(row, 25),
            row[26].as<double>()});
    }
    return records;
}

SchedulerQueueSnapshot PostgresSchedulerRepository::loadQueueSnapshot()
{
    SchedulerQueueSnapshot snapshot;
    const pqxx::result rows = transaction_.exec(
        "SELECT phase, status, count(*) "
        "FROM experiment "
        "WHERE status IN ('pending', 'running') "
        "AND phase IN ('train', 'infer', 'analyze') "
        "GROUP BY phase, status;");
    for (const auto& row : rows)
    {
        const std::string phase = row[0].as<std::string>();
        const std::string status = row[1].as<std::string>();
        const int count = row[2].as<int>();
        if (phase == "train" && status == "pending")
            snapshot.pendingTrain = count;
        else if (phase == "infer" && status == "pending")
            snapshot.pendingInfer = count;
        else if (phase == "analyze" && status == "pending")
            snapshot.pendingAnalyze = count;
        else if (phase == "train" && status == "running")
            snapshot.runningTrain = count;
        else if (phase == "infer" && status == "running")
            snapshot.runningInfer = count;
        else if (phase == "analyze" && status == "running")
            snapshot.runningAnalyze = count;
    }
    return snapshot;
}

SpawnPersistenceResult
PostgresSchedulerRepository::persistSpawnedWorkerAttempt(
    const SpawnedWorkerAttemptUpdate& update)
{
    if (!HasCompleteSpawnedWorkerAttemptUpdate(update))
        return SpawnPersistenceResult::AttemptPreconditionRejected;

    const pqxx::result spawned = transaction_.exec(
        "UPDATE experiment_scheduler_worker_attempt SET "
        "lifecycle_state='spawned',worker_pid=$1,"
        "worker_process_group_id=$1,worker_process_start_identity=$2,"
        "canonical_executable_path=$3,command_line=$4,"
        "spawned_at=COALESCE(spawned_at,clock_timestamp()),"
        "last_observed_at=clock_timestamp() "
        "WHERE worker_attempt_id=$5 AND scheduler_invocation_id=$6 "
        "AND scheduler_fencing_token=$7 "
        "AND (lifecycle_state='reserved' OR ("
        " lifecycle_state='spawned' AND worker_pid=$1 "
        " AND worker_process_group_id=$1 "
        " AND worker_process_start_identity=$2 "
        " AND canonical_executable_path=$3 "
        " AND command_line=$4)) RETURNING worker_attempt_id;",
        pqxx::params{
            update.workerPid,
            update.processStartIdentity,
            update.canonicalExecutablePath,
            update.commandLine,
            update.workerAttemptId,
            update.schedulerInvocationId,
            update.schedulerFencingToken});
    if (spawned.size() != 1)
        return SpawnPersistenceResult::AttemptPreconditionRejected;

    pqxx::result lifecycle;
    if (update.checkpointEvalId)
    {
        lifecycle = transaction_.exec(
            "UPDATE experiment_checkpoint_eval SET "
            "worker_pid=$1,worker_process_group_id=$1,"
            "worker_process_start_identity=$2,worker_executable=$3,"
            "worker_command_line=$4,updated_at=clock_timestamp() "
            "WHERE checkpoint_eval_id=$5 AND status='running' "
            "AND phase='infer' AND active_scheduler_worker_attempt_id=$6 "
            "RETURNING checkpoint_eval_id;",
            pqxx::params{
                update.workerPid,
                update.processStartIdentity,
                update.canonicalExecutablePath,
                update.commandLine,
                *update.checkpointEvalId,
                update.workerAttemptId});
    }
    else
    {
        lifecycle = transaction_.exec(
            "UPDATE experiment SET worker_pid=$1,worker_process_group_id=$1,"
            "worker_process_start_identity=$2,worker_executable=$3,"
            "worker_command_line=$4,updated_at=clock_timestamp() "
            "WHERE experiment_id=$5 AND status='running' AND phase=$6 "
            "AND active_scheduler_worker_attempt_id=$7 RETURNING experiment_id;",
            pqxx::params{
                update.workerPid,
                update.processStartIdentity,
                update.canonicalExecutablePath,
                update.commandLine,
                update.experimentId,
                update.phase,
                update.workerAttemptId});
    }
    if (lifecycle.size() != 1)
        return SpawnPersistenceResult::LifecyclePreconditionRejected;
    return SpawnPersistenceResult::Updated;
}

LaunchFailurePersistenceResult
PostgresSchedulerRepository::persistWorkerAttemptLaunchFailure(
    const WorkerAttemptLaunchFailureUpdate& update)
{
    const pqxx::result terminal = transaction_.exec(
        "UPDATE experiment_scheduler_worker_attempt a SET "
        "lifecycle_state='launch_failed',completed_at=clock_timestamp(),"
        "exit_code=$1,reconciliation_result='launch_failed',diagnostic=$2 "
        "WHERE a.worker_attempt_id=$3 AND a.scheduler_invocation_id=$4 "
        "AND a.scheduler_fencing_token=$5 "
        "AND a.lifecycle_state IN ('reserved','spawned') AND EXISTS ("
        " SELECT 1 FROM experiment e WHERE $6::bigint IS NULL "
        " AND e.experiment_id=$7 AND e.status='running' AND e.phase=$8 "
        " AND e.active_scheduler_worker_attempt_id=a.worker_attempt_id "
        " UNION ALL SELECT 1 FROM experiment_checkpoint_eval ce "
        " WHERE $6::bigint IS NOT NULL AND ce.checkpoint_eval_id=$6 "
        " AND ce.status='running' AND ce.phase='infer' "
        " AND ce.active_scheduler_worker_attempt_id=a.worker_attempt_id"
        ") RETURNING a.worker_attempt_id;",
        pqxx::params{
            update.exitCode,
            update.diagnostic,
            update.workerAttemptId,
            update.schedulerInvocationId,
            update.schedulerFencingToken,
            update.checkpointEvalId,
            update.experimentId,
            update.phase});
    if (terminal.size() != 1)
        return LaunchFailurePersistenceResult::AttemptPreconditionRejected;

    pqxx::result lifecycle;
    if (update.checkpointEvalId)
    {
        lifecycle = transaction_.exec(
            "UPDATE experiment_checkpoint_eval SET status='failed',"
            "worker_pid=NULL,worker_process_group_id=NULL,"
            "completed_at=clock_timestamp(),error_message=$1,"
            "updated_at=clock_timestamp(),active_scheduler_worker_attempt_id=NULL "
            "WHERE checkpoint_eval_id=$2 AND active_scheduler_worker_attempt_id=$3 "
            "AND status='running' AND phase='infer' RETURNING checkpoint_eval_id;",
            pqxx::params{
                update.diagnostic,
                *update.checkpointEvalId,
                update.workerAttemptId});
    }
    else
    {
        lifecycle = transaction_.exec(
            "UPDATE experiment SET status='failed',worker_pid=NULL,"
            "worker_process_group_id=NULL,completed_at=clock_timestamp(),"
            "exit_code=$1,error_message=$2,updated_at=clock_timestamp(),"
            "active_scheduler_worker_attempt_id=NULL "
            "WHERE experiment_id=$3 AND active_scheduler_worker_attempt_id=$4 "
            "AND status='running' AND phase=$5 RETURNING experiment_id;",
            pqxx::params{
                update.exitCode,
                update.diagnostic,
                update.experimentId,
                update.workerAttemptId,
                update.phase});
    }
    if (lifecycle.size() != 1)
        return LaunchFailurePersistenceResult::LifecyclePreconditionRejected;
    return LaunchFailurePersistenceResult::Updated;
}

} // namespace EA::SchedulerCore
