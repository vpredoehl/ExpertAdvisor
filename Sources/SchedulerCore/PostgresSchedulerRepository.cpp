#include "PostgresSchedulerRepository.hpp"

#include "GlobalExperimentControl.hpp"

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

int PostgresSchedulerRepository::countWorkersConsumingCapacity(
    std::string_view capacityClass)
{
    return transaction_.exec(
        "SELECT count(*) FROM experiment_scheduler_worker_attempt "
        "WHERE capacity_class=$1 AND lifecycle_state IN "
        "('reserved','spawned','running','observed',"
        "'identity_ambiguous');",
        pqxx::params{capacityClass}).one_row()[0].as<int>();
}

std::optional<PreemptionVictimRecord>
PostgresSchedulerRepository::loadPreemptionVictim(
    std::string_view phase,
    int candidatePriorityRank)
{
    const pqxx::result rows = transaction_.exec(
        "SELECT e.experiment_id,e.scheduler_priority,"
        "e.active_scheduler_worker_attempt_id "
        "FROM experiment e "
        "JOIN experiment_scheduler_worker_attempt a "
        "ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
        "WHERE e.status='running' AND e.phase=$1 "
        "AND a.worker_kind='experiment' "
        "AND a.lifecycle_phase=$1 AND a.capacity_class=$1 "
        "AND a.lifecycle_state IN ('spawned','running','observed') "
        "AND e.cancellation_request_id IS NULL "
        "AND e.cancel_after_checkpoint_epoch IS NULL "
        "AND e.stop_after_checkpoint_epoch IS NULL "
        "AND e.worker_global_pause_request_id IS NULL "
        "AND e.worker_control_state='running' "
        "AND CASE e.scheduler_priority WHEN 'high' THEN 0 "
        "WHEN 'normal' THEN 1 ELSE 2 END > $2 "
        "ORDER BY CASE e.scheduler_priority WHEN 'low' THEN 0 "
        "WHEN 'normal' THEN 1 ELSE 2 END,"
        "e.worker_started_at DESC NULLS LAST,e.experiment_id DESC LIMIT 1;",
        pqxx::params{phase, candidatePriorityRank});
    if (rows.empty())
        return std::nullopt;
    return PreemptionVictimRecord{
        rows[0][0].as<long long>(),
        rows[0][1].as<std::string>(),
        rows[0][2].as<long long>()};
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

void PostgresSchedulerRepository::acquireAuthorityCoordinationLock()
{
    transaction_.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,0));",
        pqxx::params{
            EA::GlobalExperimentControl::kCoordinationLockName});
}

std::optional<SchedulerProtocolState>
PostgresSchedulerRepository::loadSchedulerProtocolForUpdate()
{
    const pqxx::result rows = transaction_.exec(
        "SELECT required_generation,cutover_state,failure_diagnostic "
        "FROM experiment_scheduler_protocol "
        "WHERE singleton=true FOR UPDATE;");
    if (rows.size() != 1)
        return std::nullopt;
    return SchedulerProtocolState{
        rows[0][0].as<int>(),
        rows[0][1].as<std::string>(),
        OptionalCell<std::string>(rows[0], 2)};
}

void PostgresSchedulerRepository::registerSchedulerInvocation(
    const SchedulerInvocationRecord& invocation)
{
    transaction_.exec(
        "INSERT INTO experiment_scheduler_invocation("
        "scheduler_invocation_id,process_pid,process_group_id,"
        "process_start_identity,canonical_executable_path,command_line,"
        "invocation_nonce,status,protocol_generation) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,'starting',$8);",
        pqxx::params{
            invocation.schedulerInvocationId,
            invocation.processPid,
            invocation.processGroupId,
            invocation.processStartIdentity,
            invocation.canonicalExecutablePath,
            invocation.commandLine,
            invocation.invocationNonce,
            invocation.protocolGeneration});
}

std::optional<SchedulerLeaseState>
PostgresSchedulerRepository::loadSchedulerLeaseForUpdate()
{
    const pqxx::result rows = transaction_.exec(
        "SELECT l.owner_scheduler_invocation_id,l.fencing_token,"
        "l.authority_state,(l.expires_at <= clock_timestamp()) AS expired,"
        "i.process_pid,i.process_group_id,i.process_start_identity,"
        "i.canonical_executable_path "
        "FROM experiment_scheduler_lease l "
        "LEFT JOIN experiment_scheduler_invocation i "
        "ON i.scheduler_invocation_id=l.owner_scheduler_invocation_id "
        "WHERE l.singleton=true FOR UPDATE OF l;");
    if (rows.size() != 1)
        return std::nullopt;
    return SchedulerLeaseState{
        OptionalCell<std::string>(rows[0], 0),
        rows[0][1].as<long long>(),
        rows[0][2].as<std::string>(),
        rows[0][3].is_null() || rows[0][3].as<bool>(),
        OptionalCell<int>(rows[0], 4),
        OptionalCell<int>(rows[0], 5),
        OptionalCell<std::string>(rows[0], 6),
        OptionalCell<std::string>(rows[0], 7)};
}

void PostgresSchedulerRepository::rejectSchedulerInvocation(
    std::string_view schedulerInvocationId,
    std::string_view terminalReason)
{
    transaction_.exec(
        "UPDATE experiment_scheduler_invocation SET "
        "status='rejected',ended_at=clock_timestamp(),terminal_reason=$1 "
        "WHERE scheduler_invocation_id=$2 AND status='starting';",
        pqxx::params{terminalReason, schedulerInvocationId});
}

void PostgresSchedulerRepository::markSchedulerInvocationCrashed(
    std::string_view schedulerInvocationId,
    std::string_view terminalReason)
{
    transaction_.exec(
        "UPDATE experiment_scheduler_invocation SET "
        "status=CASE WHEN status='released' THEN status ELSE 'crashed' END,"
        "ended_at=COALESCE(ended_at,clock_timestamp()),"
        "terminal_reason=COALESCE(terminal_reason,$1) "
        "WHERE scheduler_invocation_id=$2;",
        pqxx::params{terminalReason, schedulerInvocationId});
}

bool PostgresSchedulerRepository::acquireSchedulerLease(
    const SchedulerLeaseAcquisition& acquisition)
{
    const pqxx::result updated = transaction_.exec(
        "UPDATE experiment_scheduler_lease SET "
        "owner_scheduler_invocation_id=$1,fencing_token=$2,"
        "authority_state='active',acquired_at=clock_timestamp(),"
        "heartbeat_at=clock_timestamp(),"
        "expires_at=clock_timestamp()+make_interval(secs=>$3),"
        "released_at=NULL,transition_reason=$4 "
        "WHERE singleton=true RETURNING fencing_token;",
        pqxx::params{
            acquisition.schedulerInvocationId,
            acquisition.fencingToken,
            acquisition.leaseSeconds,
            acquisition.transitionReason});
    return updated.size() == 1;
}

void PostgresSchedulerRepository::markSchedulerInvocationOwner(
    std::string_view schedulerInvocationId)
{
    transaction_.exec(
        "UPDATE experiment_scheduler_invocation SET "
        "status='owner',ownership_acquired_at=clock_timestamp(),"
        "last_heartbeat_at=clock_timestamp() "
        "WHERE scheduler_invocation_id=$1 AND status='starting';",
        pqxx::params{schedulerInvocationId});
}

bool PostgresSchedulerRepository::renewSchedulerLease(
    const SchedulerAuthorityIdentity& authority,
    int leaseSeconds)
{
    const pqxx::result refreshed = transaction_.exec(
        "UPDATE experiment_scheduler_lease SET "
        "heartbeat_at=clock_timestamp(),"
        "expires_at=clock_timestamp()+make_interval(secs=>$1) "
        "WHERE singleton=true AND authority_state='active' "
        "AND owner_scheduler_invocation_id=$2 AND fencing_token=$3 "
        "RETURNING fencing_token;",
        pqxx::params{
            leaseSeconds,
            authority.schedulerInvocationId,
            authority.fencingToken});
    return refreshed.size() == 1;
}

void PostgresSchedulerRepository::touchSchedulerInvocation(
    std::string_view schedulerInvocationId)
{
    transaction_.exec(
        "UPDATE experiment_scheduler_invocation SET "
        "last_heartbeat_at=clock_timestamp() "
        "WHERE scheduler_invocation_id=$1 AND status='owner';",
        pqxx::params{schedulerInvocationId});
}

bool PostgresSchedulerRepository::releaseSchedulerLease(
    const SchedulerAuthorityIdentity& authority,
    std::string_view reason)
{
    const pqxx::result released = transaction_.exec(
        "UPDATE experiment_scheduler_lease SET "
        "authority_state='released',heartbeat_at=clock_timestamp(),"
        "expires_at=clock_timestamp(),released_at=clock_timestamp(),"
        "transition_reason=$1 "
        "WHERE singleton=true AND authority_state='active' "
        "AND owner_scheduler_invocation_id=$2 AND fencing_token=$3 "
        "RETURNING fencing_token;",
        pqxx::params{
            reason,
            authority.schedulerInvocationId,
            authority.fencingToken});
    return released.size() == 1;
}

void PostgresSchedulerRepository::markSchedulerInvocationReleased(
    std::string_view schedulerInvocationId,
    std::string_view reason)
{
    transaction_.exec(
        "UPDATE experiment_scheduler_invocation SET "
        "status='released',ownership_released_at=clock_timestamp(),"
        "ended_at=clock_timestamp(),terminal_reason=$1 "
        "WHERE scheduler_invocation_id=$2 AND status='owner';",
        pqxx::params{reason, schedulerInvocationId});
}

bool PostgresSchedulerRepository::completeSchedulerProtocolCutover(
    const SchedulerProtocolCutoverUpdate& update)
{
    const pqxx::result completed = transaction_.exec(
        "UPDATE experiment_scheduler_protocol SET "
        "cutover_state='complete',cutover_completed_at=clock_timestamp(),"
        "cutover_completed_by=$1,cutover_executable_path=$2,"
        "cutover_process_evidence=$3,failure_diagnostic=NULL,"
        "updated_at=clock_timestamp() "
        "WHERE singleton=true AND required_generation=$4 "
        "AND cutover_state IN ('pending','failed') "
        "RETURNING required_generation;",
        pqxx::params{
            update.actor,
            update.canonicalExecutablePath,
            update.processEvidence,
            update.protocolGeneration});
    return completed.size() == 1;
}

bool PostgresSchedulerRepository::displaceSchedulerAuthorityForTest(
    const SchedulerAuthorityIdentity& authority,
    std::string_view foreignSchedulerInvocationId,
    std::string_view transitionReason,
    int leaseSeconds)
{
    const pqxx::result displaced = transaction_.exec(
        "UPDATE experiment_scheduler_lease SET "
        "owner_scheduler_invocation_id=$1,fencing_token=fencing_token+1,"
        "authority_state='active',heartbeat_at=clock_timestamp(),"
        "expires_at=clock_timestamp()+make_interval(secs=>$2),"
        "transition_reason=$3 "
        "WHERE singleton=true AND owner_scheduler_invocation_id=$4 "
        "AND fencing_token=$5 RETURNING fencing_token;",
        pqxx::params{
            foreignSchedulerInvocationId,
            leaseSeconds,
            transitionReason,
            authority.schedulerInvocationId,
            authority.fencingToken});
    return displaced.size() == 1;
}

} // namespace EA::SchedulerCore
