#include "PostgresSchedulerRepository.hpp"

#include "GlobalExperimentControl.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

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

bool TableExists(
    pqxx::transaction_base& transaction,
    std::string_view tableName)
{
    return !transaction.exec(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=$1 LIMIT 1;",
        pqxx::params{tableName}).empty();
}

bool ColumnExists(
    pqxx::transaction_base& transaction,
    std::string_view tableName,
    std::string_view columnName)
{
    return !transaction.exec(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=$1 "
        "AND column_name=$2 LIMIT 1;",
        pqxx::params{tableName, columnName}).empty();
}

std::string CheckpointPolicyPopulationWatermark(const pqxx::result& rows)
{
    std::ostringstream canonical;
    canonical << "population_size=" << rows.size();
    for (const auto& row : rows)
    {
        canonical << '|';
        for (pqxx::row::size_type column = 0; column < row.size(); ++column)
        {
            if (column != 0)
                canonical << ',';
            if (row[column].is_null())
            {
                canonical << "NULL";
                continue;
            }
            const std::string value = row[column].as<std::string>();
            canonical << value.size() << ':' << value;
        }
    }
    return EA::ExperimentScheduler::StableCheckpointPolicyHash(
        canonical.str());
}

} // namespace

PostgresSchedulerRepository::PostgresSchedulerRepository(
    pqxx::transaction_base& transaction)
    : transaction_{transaction}
{
}

std::optional<AuthoritativeFinalInferenceResult>
PostgresSchedulerRepository::findAuthoritativeFinalInferenceResultForWorkerAttempt(
    long long experimentId,
    long long workerAttemptId)
{
    // This is the durable completion contract shared by scheduler-owned orphan
    // recovery and exact-attempt reconciliation. A profitability observation
    // or log marker is deliberately insufficient.
    const pqxx::row relation = transaction_.exec(
        "SELECT to_regclass('inference_eval_result');").one_row();
    if (relation[0].is_null())
        return std::nullopt;

    const pqxx::result rows = transaction_.exec(
        "SELECT r.id,r.model_id,"
        "e.operator_forced_final_inference_rerun_requested "
        "FROM experiment e "
        "JOIN experiment_scheduler_worker_attempt a "
        " ON a.worker_attempt_id=$2 AND a.experiment_id=e.experiment_id "
        " AND a.checkpoint_eval_id IS NULL "
        " AND a.worker_kind='experiment' "
        " AND a.lifecycle_phase='infer' AND a.capacity_class='infer' "
        "JOIN model m ON m.model_id=e.last_model_id "
        " AND m.experiment_id=e.experiment_id "
        "JOIN inference_eval_result r ON r.model_id=e.last_model_id "
        " AND r.symbol=e.symbol "
        " AND r.prediction_horizon=e.prediction_horizon "
        " AND abs(r.threshold_logret-e.c_next_threshold)<=1e-7 "
        " AND r.from_date=e.infer_start::date::text "
        " AND r.to_date=e.infer_end::date::text "
        " AND r.status='completed' AND r.inference_scope='final' "
        " AND r.checkpoint_eval_id IS NULL "
        " AND r.completed_at>=a.reserved_at "
        "WHERE e.experiment_id=$1 AND e.phase='infer' "
        "AND e.active_scheduler_worker_attempt_id=a.worker_attempt_id "
        "AND (e.status='running' OR ("
        " e.status='pending' AND e.resume_requested "
        " AND e.scheduler_resume_origin='preemption' "
        " AND e.worker_control_state='paused' "
        " AND a.lifecycle_state='stopped')) "
        "ORDER BY r.completed_at DESC,r.id DESC LIMIT 1;",
        pqxx::params{experimentId, workerAttemptId});
    if (rows.empty())
        return std::nullopt;
    return AuthoritativeFinalInferenceResult{
        rows[0][0].as<long long>(),
        rows[0][1].as<long long>(),
        rows[0][2].as<bool>()};
}

HistoricalFailedInferenceRecoveryDiscovery
PostgresSchedulerRepository::findHistoricalFailedInferenceRecoveryEvidence(
    long long experimentId)
{
    const pqxx::result experiment = transaction_.exec(
        "SELECT 1 FROM experiment e "
        "WHERE e.experiment_id=$1 AND e.status='failed' "
        "AND e.phase='infer' AND e.exit_code=0 "
        "AND e.error_message='child_exit_code_0;phase=infer;exit_code=0' "
        "AND e.active_scheduler_worker_attempt_id IS NULL "
        "AND e.last_model_id IS NOT NULL;",
        pqxx::params{experimentId});
    if (experiment.empty())
    {
        return {
            HistoricalFailedInferenceRecoveryEvidenceStatus::
                ExperimentNotEligible,
            std::nullopt};
    }

    // Do not add LIMIT here. More than one attempt/result pair is ambiguous
    // evidence and must be rejected rather than resolved by ordering.
    const pqxx::result rows = transaction_.exec(
        "SELECT e.experiment_id,a.worker_attempt_id,m.model_id,r.id "
        "FROM experiment e "
        "JOIN model m ON m.model_id=e.last_model_id "
        " AND m.experiment_id=e.experiment_id "
        "JOIN experiment_scheduler_worker_attempt a "
        " ON a.experiment_id=e.experiment_id "
        " AND a.checkpoint_eval_id IS NULL "
        " AND a.worker_kind='experiment' "
        " AND a.lifecycle_phase='infer' "
        " AND a.capacity_class='infer' "
        " AND a.lifecycle_state='failed' "
        " AND a.exit_code=0 "
        " AND a.completed_at IS NOT NULL "
        " AND a.reconciliation_result='parent_observed_exit' "
        "JOIN inference_eval_result r ON r.model_id=m.model_id "
        " AND r.symbol=e.symbol "
        " AND r.prediction_horizon=e.prediction_horizon "
        " AND abs(r.threshold_logret-e.c_next_threshold)<=1e-7 "
        " AND r.from_date=e.infer_start::date::text "
        " AND r.to_date=e.infer_end::date::text "
        " AND r.status='completed' "
        " AND r.inference_scope='final' "
        " AND r.checkpoint_eval_id IS NULL "
        " AND r.completed_at>=a.reserved_at "
        "WHERE e.experiment_id=$1 AND e.status='failed' "
        "AND e.phase='infer' AND e.exit_code=0 "
        "AND e.error_message='child_exit_code_0;phase=infer;exit_code=0' "
        "AND e.active_scheduler_worker_attempt_id IS NULL "
        "AND e.last_model_id IS NOT NULL "
        "ORDER BY a.worker_attempt_id,r.id "
        "FOR UPDATE OF e,a,m,r;",
        pqxx::params{experimentId});
    if (rows.empty())
    {
        return {
            HistoricalFailedInferenceRecoveryEvidenceStatus::
                NoQualifyingEvidence,
            std::nullopt};
    }
    if (rows.size() != 1)
    {
        return {
            HistoricalFailedInferenceRecoveryEvidenceStatus::
                AmbiguousEvidence,
            std::nullopt};
    }

    return {
        HistoricalFailedInferenceRecoveryEvidenceStatus::Eligible,
        HistoricalFailedInferenceRecoveryEvidence{
            rows[0][0].as<long long>(),
            rows[0][1].as<long long>(),
            rows[0][2].as<long long>(),
            rows[0][3].as<long long>()}};
}

HistoricalFailedInferenceRecoveryPersistenceResult
PostgresSchedulerRepository::applyHistoricalFailedInferenceRecovery(
    const HistoricalFailedInferenceRecoveryEvidence& evidence)
{
    // Re-prove the unique relationship in the UPDATE snapshot. This rejects
    // state changes after discovery, including a newly ambiguous attempt or
    // result, before either durable row can be committed.
    const pqxx::result attempt = transaction_.exec(
        "WITH candidates AS ("
        " SELECT e.experiment_id,a.worker_attempt_id,m.model_id,r.id "
        " FROM experiment e "
        " JOIN model m ON m.model_id=e.last_model_id "
        "  AND m.experiment_id=e.experiment_id "
        " JOIN experiment_scheduler_worker_attempt a "
        "  ON a.experiment_id=e.experiment_id "
        "  AND a.checkpoint_eval_id IS NULL "
        "  AND a.worker_kind='experiment' "
        "  AND a.lifecycle_phase='infer' "
        "  AND a.capacity_class='infer' "
        "  AND a.lifecycle_state='failed' AND a.exit_code=0 "
        "  AND a.completed_at IS NOT NULL "
        "  AND a.reconciliation_result='parent_observed_exit' "
        " JOIN inference_eval_result r ON r.model_id=m.model_id "
        "  AND r.symbol=e.symbol "
        "  AND r.prediction_horizon=e.prediction_horizon "
        "  AND abs(r.threshold_logret-e.c_next_threshold)<=1e-7 "
        "  AND r.from_date=e.infer_start::date::text "
        "  AND r.to_date=e.infer_end::date::text "
        "  AND r.status='completed' AND r.inference_scope='final' "
        "  AND r.checkpoint_eval_id IS NULL "
        "  AND r.completed_at>=a.reserved_at "
        " WHERE e.experiment_id=$1 AND e.status='failed' "
        " AND e.phase='infer' AND e.exit_code=0 "
        " AND e.error_message='child_exit_code_0;phase=infer;exit_code=0' "
        " AND e.active_scheduler_worker_attempt_id IS NULL "
        " AND e.last_model_id IS NOT NULL"
        "), unique_candidate AS ("
        " SELECT min(experiment_id) AS experiment_id,"
        " min(worker_attempt_id) AS worker_attempt_id,"
        " min(model_id) AS model_id,min(id) AS inference_result_id "
        " FROM candidates HAVING count(*)=1"
        ") UPDATE experiment_scheduler_worker_attempt a SET "
        "lifecycle_state='completed',"
        "reconciliation_result="
        "'historical_completed_inference_result_recovered',"
        "diagnostic="
        "'historical_failed_final_inference_recovered_from_durable_result',"
        "reconciled_at=clock_timestamp() "
        "FROM unique_candidate c "
        "WHERE a.worker_attempt_id=$2 "
        "AND c.experiment_id=$1 AND c.worker_attempt_id=$2 "
        "AND c.model_id=$3 AND c.inference_result_id=$4 "
        "AND a.lifecycle_state='failed' AND a.exit_code=0 "
        "AND a.completed_at IS NOT NULL "
        "AND a.reconciliation_result='parent_observed_exit' "
        "RETURNING a.worker_attempt_id;",
        pqxx::params{
            evidence.experimentId,
            evidence.workerAttemptId,
            evidence.modelId,
            evidence.inferenceResultId});
    if (attempt.size() != 1)
    {
        return HistoricalFailedInferenceRecoveryPersistenceResult::
            AtomicPreconditionRejected;
    }

    const pqxx::result lifecycle = transaction_.exec(
        "UPDATE experiment e SET "
        "status='pending',phase='analyze',"
        "worker_pid=NULL,worker_process_group_id=NULL,"
        "worker_process_start_identity=NULL,worker_executable=NULL,"
        "worker_command_line=NULL,worker_started_at=NULL,"
        "worker_control_state='paused',worker_global_pause_request_id=NULL,"
        "active_scheduler_worker_attempt_id=NULL,current_operation=NULL,"
        "completed_at=NULL,exit_code=0,error_message=NULL,"
        "operator_forced_final_inference_rerun_requested=false,"
        "resume_requested=false,scheduler_resume_origin='none',"
        "updated_at=clock_timestamp() "
        "WHERE e.experiment_id=$1 AND e.status='failed' "
        "AND e.phase='infer' AND e.exit_code=0 "
        "AND e.error_message='child_exit_code_0;phase=infer;exit_code=0' "
        "AND e.active_scheduler_worker_attempt_id IS NULL "
        "AND e.last_model_id=$3 "
        "AND EXISTS ("
        " SELECT 1 FROM model m "
        " JOIN experiment_scheduler_worker_attempt a "
        "  ON a.worker_attempt_id=$2 "
        "  AND a.experiment_id=e.experiment_id "
        "  AND a.checkpoint_eval_id IS NULL "
        "  AND a.worker_kind='experiment' "
        "  AND a.lifecycle_phase='infer' "
        "  AND a.capacity_class='infer' "
        "  AND a.lifecycle_state='completed' AND a.exit_code=0 "
        "  AND a.completed_at IS NOT NULL "
        "  AND a.reconciliation_result="
        "'historical_completed_inference_result_recovered' "
        " JOIN inference_eval_result r ON r.id=$4 "
        "  AND r.model_id=m.model_id AND r.symbol=e.symbol "
        "  AND r.prediction_horizon=e.prediction_horizon "
        "  AND abs(r.threshold_logret-e.c_next_threshold)<=1e-7 "
        "  AND r.from_date=e.infer_start::date::text "
        "  AND r.to_date=e.infer_end::date::text "
        "  AND r.status='completed' AND r.inference_scope='final' "
        "  AND r.checkpoint_eval_id IS NULL "
        "  AND r.completed_at>=a.reserved_at "
        " WHERE m.model_id=$3 AND m.experiment_id=e.experiment_id"
        ") AND NOT EXISTS ("
        " SELECT 1 FROM experiment_scheduler_worker_attempt a2 "
        " JOIN inference_eval_result r2 ON r2.model_id=e.last_model_id "
        "  AND r2.symbol=e.symbol "
        "  AND r2.prediction_horizon=e.prediction_horizon "
        "  AND abs(r2.threshold_logret-e.c_next_threshold)<=1e-7 "
        "  AND r2.from_date=e.infer_start::date::text "
        "  AND r2.to_date=e.infer_end::date::text "
        "  AND r2.status='completed' AND r2.inference_scope='final' "
        "  AND r2.checkpoint_eval_id IS NULL "
        "  AND r2.completed_at>=a2.reserved_at "
        " WHERE a2.experiment_id=e.experiment_id "
        " AND a2.checkpoint_eval_id IS NULL "
        " AND a2.worker_kind='experiment' "
        " AND a2.lifecycle_phase='infer' AND a2.capacity_class='infer' "
        " AND a2.lifecycle_state='failed' AND a2.exit_code=0 "
        " AND a2.completed_at IS NOT NULL "
        " AND a2.reconciliation_result='parent_observed_exit'"
        ") AND NOT EXISTS ("
        " SELECT 1 FROM inference_eval_result r3 "
        " JOIN experiment_scheduler_worker_attempt a3 "
        "  ON a3.worker_attempt_id=$2 "
        " WHERE r3.id<>$4 AND r3.model_id=e.last_model_id "
        " AND r3.symbol=e.symbol "
        " AND r3.prediction_horizon=e.prediction_horizon "
        " AND abs(r3.threshold_logret-e.c_next_threshold)<=1e-7 "
        " AND r3.from_date=e.infer_start::date::text "
        " AND r3.to_date=e.infer_end::date::text "
        " AND r3.status='completed' AND r3.inference_scope='final' "
        " AND r3.checkpoint_eval_id IS NULL "
        " AND r3.completed_at>=a3.reserved_at"
        ") RETURNING e.experiment_id;",
        pqxx::params{
            evidence.experimentId,
            evidence.workerAttemptId,
            evidence.modelId,
            evidence.inferenceResultId});
    if (lifecycle.size() != 1)
    {
        return HistoricalFailedInferenceRecoveryPersistenceResult::
            AtomicPreconditionRejected;
    }
    return HistoricalFailedInferenceRecoveryPersistenceResult::Updated;
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

WorkerAttemptReservationResult
PostgresSchedulerRepository::reserveExperimentWorkerAttempt(
    const ExperimentWorkerAttemptReservation& reservation)
{
    const pqxx::result lifecycle = transaction_.exec(
        "SELECT status,phase,active_scheduler_worker_attempt_id,"
        "cancellation_request_id,cancel_after_checkpoint_epoch "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{reservation.experimentId});
    if (lifecycle.size() != 1 ||
        lifecycle[0][0].as<std::string>() != "pending" ||
        lifecycle[0][1].as<std::string>() != reservation.phase ||
        !lifecycle[0][2].is_null() ||
        (reservation.cancellationOnly &&
         (lifecycle[0][3].is_null() || lifecycle[0][4].is_null())))
    {
        return {WorkerAttemptReservationStatus::LifecycleUnavailable,
                std::nullopt};
    }

    const pqxx::result inserted = transaction_.exec(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "launch_attempt_identity,scheduler_invocation_id,"
        "scheduler_fencing_token,experiment_id,worker_kind,"
        "lifecycle_phase,capacity_class,ownership_origin,"
        "lifecycle_state,canonical_executable_path,"
        "semantic_layout_version,model_input_width,semantic_worker_role,"
        "source_commit,executable_sha256,runtime_identity,canonical_manifest_path,"
        "command_identity,log_path) "
        "VALUES($1,$2,$3,$4,'experiment',$5,$5,"
        "'scheduler_launch','reserved',$6,"
        "CASE WHEN $5='analyze' THEN NULL ELSE $7::integer END,"
        "CASE WHEN $5='analyze' THEN NULL ELSE $8::bigint END,"
        "CASE WHEN $5='analyze' THEN NULL ELSE $9 END,"
        "CASE WHEN $5='analyze' THEN NULL ELSE $10 END,"
        "CASE WHEN $5='analyze' THEN NULL ELSE $11 END,"
        "CASE WHEN $5='analyze' THEN NULL ELSE $12 END,"
        "CASE WHEN $5='analyze' THEN NULL ELSE $13 END,$14,$15) "
        "RETURNING worker_attempt_id;",
        pqxx::params{
            reservation.launchAttemptIdentity,
            reservation.schedulerInvocationId,
            reservation.schedulerFencingToken,
            reservation.experimentId,
            reservation.phase,
            reservation.canonicalExecutablePath,
            reservation.semanticLayoutVersion,
            reservation.modelInputWidth,
            reservation.semanticWorkerRole,
            reservation.sourceCommit,
            reservation.executableSha256,
            reservation.runtimeIdentity,
            reservation.canonicalManifestPath,
            reservation.commandIdentity,
            reservation.logPath});
    if (inserted.size() != 1)
    {
        return {WorkerAttemptReservationStatus::ReservationInsertFailed,
                std::nullopt};
    }

    ReservedWorkerAttempt attempt;
    attempt.workerAttemptId = inserted[0][0].as<long long>();
    attempt.launchAttemptIdentity = reservation.launchAttemptIdentity;
    attempt.experimentId = reservation.experimentId;
    attempt.workerKind = "experiment";
    attempt.phase = reservation.phase;
    attempt.capacityClass = reservation.phase;
    attempt.logPath = reservation.logPath;
    attempt.canonicalExecutablePath =
        reservation.canonicalExecutablePath;

    const std::string logColumn =
        reservation.phase == "train"
            ? "train_log_path"
            : (reservation.phase == "infer"
                   ? "infer_log_path"
                   : "analysis_log_path");
    const std::string claimSql =
        std::string{
            "UPDATE experiment SET status='running',resume_requested=false,"
            "scheduler_resume_origin='none',"
            "started_at=COALESCE(started_at,clock_timestamp()),"
            "worker_started_at=clock_timestamp(),worker_pid=NULL,"
            "worker_process_group_id=NULL,"
            "worker_process_start_identity=NULL,worker_executable=$1,"
            "current_operation=$2,worker_control_state='running',"
            "active_scheduler_worker_attempt_id=$3,"} +
        logColumn + "=$4,updated_at=clock_timestamp() "
        "WHERE experiment_id=$5 AND status='pending' AND phase=$6 "
        "AND active_scheduler_worker_attempt_id IS NULL "
        "RETURNING experiment_id;";
    const pqxx::result claimed = transaction_.exec(
        claimSql,
        pqxx::params{
            reservation.canonicalExecutablePath,
            reservation.currentOperation,
            attempt.workerAttemptId,
            reservation.logPath,
            reservation.experimentId,
            reservation.phase});
    if (claimed.size() != 1)
    {
        return {WorkerAttemptReservationStatus::LifecycleClaimFailed,
                std::nullopt};
    }
    return {WorkerAttemptReservationStatus::Reserved, std::move(attempt)};
}

WorkerAttemptReservationResult
PostgresSchedulerRepository::reserveCheckpointWorkerAttempt(
    const CheckpointWorkerAttemptReservation& reservation)
{
    const pqxx::result lifecycle = transaction_.exec(
        "SELECT status,phase,active_scheduler_worker_attempt_id "
        "FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
        pqxx::params{reservation.checkpointEvalId});
    if (lifecycle.size() != 1 ||
        lifecycle[0][0].as<std::string>() != "pending" ||
        lifecycle[0][1].as<std::string>() != "infer" ||
        !lifecycle[0][2].is_null())
    {
        return {WorkerAttemptReservationStatus::LifecycleUnavailable,
                std::nullopt};
    }

    const pqxx::result inserted = transaction_.exec(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "launch_attempt_identity,scheduler_invocation_id,"
        "scheduler_fencing_token,experiment_id,checkpoint_eval_id,"
        "worker_kind,lifecycle_phase,capacity_class,ownership_origin,"
        "lifecycle_state,canonical_executable_path,semantic_layout_version,model_input_width,semantic_worker_role,source_commit,executable_sha256,runtime_identity,canonical_manifest_path,command_identity,log_path) "
        "VALUES($1,$2,$3,$4,$5,'checkpoint_infer','infer','infer',"
        "'scheduler_launch','reserved',$6,$7,$8,$9,$10,$11,$12,$13,$14,$15) "
        "RETURNING worker_attempt_id;",
        pqxx::params{
            reservation.launchAttemptIdentity,
            reservation.schedulerInvocationId,
            reservation.schedulerFencingToken,
            reservation.experimentId,
            reservation.checkpointEvalId,
            reservation.canonicalExecutablePath,
            reservation.semanticLayoutVersion,
            reservation.modelInputWidth,
            reservation.semanticWorkerRole,
            reservation.sourceCommit,
            reservation.executableSha256,
            reservation.runtimeIdentity,
            reservation.canonicalManifestPath,
            reservation.commandIdentity,
            reservation.logPath});
    if (inserted.size() != 1)
    {
        return {WorkerAttemptReservationStatus::ReservationInsertFailed,
                std::nullopt};
    }

    ReservedWorkerAttempt attempt;
    attempt.workerAttemptId = inserted[0][0].as<long long>();
    attempt.launchAttemptIdentity = reservation.launchAttemptIdentity;
    attempt.experimentId = reservation.experimentId;
    attempt.checkpointEvalId = reservation.checkpointEvalId;
    attempt.workerKind = "checkpoint_infer";
    attempt.phase = "infer";
    attempt.capacityClass = "infer";
    attempt.logPath = reservation.logPath;
    attempt.canonicalExecutablePath =
        reservation.canonicalExecutablePath;

    const pqxx::result claimed = transaction_.exec(
        "UPDATE experiment_checkpoint_eval SET status='running',phase='infer',"
        "worker_pid=NULL,worker_process_group_id=NULL,"
        "worker_process_start_identity=NULL,worker_executable=$1,"
        "worker_control_state='running',infer_log_path=$2,"
        "active_scheduler_worker_attempt_id=$3,"
        "started_at=COALESCE(started_at,clock_timestamp()),"
        "infer_started_at=COALESCE(infer_started_at,clock_timestamp()),"
        "updated_at=clock_timestamp(),error_message=NULL "
        "WHERE checkpoint_eval_id=$4 AND status='pending' AND phase='infer' "
        "AND active_scheduler_worker_attempt_id IS NULL "
        "RETURNING checkpoint_eval_id;",
        pqxx::params{
            reservation.canonicalExecutablePath,
            reservation.logPath,
            attempt.workerAttemptId,
            reservation.checkpointEvalId});
    if (claimed.size() != 1)
    {
        return {WorkerAttemptReservationStatus::LifecycleClaimFailed,
                std::nullopt};
    }
    return {WorkerAttemptReservationStatus::Reserved, std::move(attempt)};
}

WorkerAttemptReservationResult
PostgresSchedulerRepository::reserveCheckpointAnalysisAttempt(
    const CheckpointAnalysisAttemptReservation& reservation)
{
    const pqxx::result inserted = transaction_.exec(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "launch_attempt_identity,scheduler_invocation_id,"
        "scheduler_fencing_token,experiment_id,checkpoint_eval_id,"
        "worker_kind,lifecycle_phase,capacity_class,ownership_origin,"
        "lifecycle_state,canonical_executable_path,command_line,"
        "command_identity) VALUES($1,$2,$3,$4,$5,'checkpoint_analyze',"
        "'analyze','analyze','scheduler_in_process','running',$6,$7,$8) "
        "RETURNING worker_attempt_id;",
        pqxx::params{
            reservation.launchAttemptIdentity,
            reservation.schedulerInvocationId,
            reservation.schedulerFencingToken,
            reservation.experimentId,
            reservation.checkpointEvalId,
            reservation.canonicalExecutablePath,
            reservation.commandLine,
            reservation.commandIdentity});
    if (inserted.size() != 1)
    {
        return {WorkerAttemptReservationStatus::ReservationInsertFailed,
                std::nullopt};
    }

    ReservedWorkerAttempt attempt;
    attempt.workerAttemptId = inserted[0][0].as<long long>();
    attempt.launchAttemptIdentity = reservation.launchAttemptIdentity;
    attempt.experimentId = reservation.experimentId;
    attempt.checkpointEvalId = reservation.checkpointEvalId;
    attempt.workerKind = "checkpoint_analyze";
    attempt.phase = "analyze";
    attempt.capacityClass = "analyze";
    attempt.canonicalExecutablePath =
        reservation.canonicalExecutablePath;

    std::string sql =
        "UPDATE experiment_checkpoint_eval SET status='running',"
        "phase='analyze',active_scheduler_worker_attempt_id=$1,"
        "worker_pid=NULL,worker_process_group_id=NULL,"
        "worker_process_start_identity=NULL,worker_executable=$2,"
        "worker_command_line=$3,updated_at=clock_timestamp(),"
        "error_message=NULL";
    if (reservation.recordAnalyzeStartedAt)
    {
        sql += ",analyze_started_at="
               "COALESCE(analyze_started_at,clock_timestamp())";
    }
    sql +=
        " WHERE checkpoint_eval_id=$4 AND status='pending' "
        "AND phase='analyze' AND active_scheduler_worker_attempt_id IS NULL "
        "RETURNING checkpoint_eval_id;";
    const pqxx::result claimed = transaction_.exec(
        sql,
        pqxx::params{
            attempt.workerAttemptId,
            reservation.canonicalExecutablePath,
            reservation.commandLine,
            reservation.checkpointEvalId});
    if (claimed.size() != 1)
    {
        return {WorkerAttemptReservationStatus::LifecycleClaimFailed,
                std::nullopt};
    }
    return {WorkerAttemptReservationStatus::Reserved, std::move(attempt)};
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
        "AND canonical_executable_path=$3 "
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

bool PostgresSchedulerRepository::persistCheckpointAnalysisCompletion(
    const CheckpointAnalysisCompletionUpdate& update)
{
    std::string sql =
        "UPDATE experiment_checkpoint_eval SET status='completed',"
        "phase='done',completed_at=clock_timestamp(),"
        "updated_at=clock_timestamp(),error_message=NULL";
    if (update.recordAnalyzeCompletedAt)
        sql += ",analyze_completed_at=clock_timestamp()";
    sql +=
        ",analysis_id=$1 WHERE checkpoint_eval_id=$2 "
        "AND status='running' AND phase='analyze' "
        "AND active_scheduler_worker_attempt_id=$3 "
        "RETURNING checkpoint_eval_id;";
    const pqxx::result completed = transaction_.exec(
        sql,
        pqxx::params{
            update.analysisId,
            update.checkpointEvalId,
            update.workerAttemptId});
    return completed.size() == 1;
}

CheckpointAnalysisPersistenceResult
PostgresSchedulerRepository::persistCheckpointAnalysisTerminalState(
    const CheckpointAnalysisTerminalUpdate& update)
{
    const pqxx::result terminal = transaction_.exec(
        "UPDATE experiment_scheduler_worker_attempt a SET "
        "lifecycle_state=$1,completed_at=clock_timestamp(),"
        "last_observed_at=clock_timestamp(),reconciliation_result=$2,"
        "diagnostic=$3 WHERE a.worker_attempt_id=$4 "
        "AND a.scheduler_invocation_id=$5 AND a.scheduler_fencing_token=$6 "
        "AND a.worker_kind='checkpoint_analyze' "
        "AND a.lifecycle_state='running' AND EXISTS (SELECT 1 "
        "FROM experiment_checkpoint_eval ce WHERE ce.checkpoint_eval_id=$7 "
        "AND ce.active_scheduler_worker_attempt_id=a.worker_attempt_id) "
        "RETURNING a.worker_attempt_id;",
        pqxx::params{
            update.attemptLifecycleState,
            update.reconciliationResult,
            update.diagnostic,
            update.workerAttemptId,
            update.schedulerInvocationId,
            update.schedulerFencingToken,
            update.checkpointEvalId});
    if (terminal.size() != 1)
    {
        return CheckpointAnalysisPersistenceResult::
            AttemptPreconditionRejected;
    }

    pqxx::result lifecycle;
    if (update.complete)
    {
        lifecycle = transaction_.exec(
            "UPDATE experiment_checkpoint_eval SET "
            "active_scheduler_worker_attempt_id=NULL "
            "WHERE checkpoint_eval_id=$1 "
            "AND active_scheduler_worker_attempt_id=$2 "
            "AND status='completed' AND phase='done' "
            "RETURNING checkpoint_eval_id;",
            pqxx::params{update.checkpointEvalId, update.workerAttemptId});
    }
    else
    {
        lifecycle = transaction_.exec(
            "UPDATE experiment_checkpoint_eval SET status='failed',"
            "completed_at=clock_timestamp(),error_message=$1,"
            "active_scheduler_worker_attempt_id=NULL,"
            "updated_at=clock_timestamp() WHERE checkpoint_eval_id=$2 "
            "AND active_scheduler_worker_attempt_id=$3 "
            "AND status='running' AND phase='analyze' "
            "RETURNING checkpoint_eval_id;",
            pqxx::params{
                update.diagnostic,
                update.checkpointEvalId,
                update.workerAttemptId});
    }
    if (lifecycle.size() != 1)
    {
        return CheckpointAnalysisPersistenceResult::
            LifecyclePreconditionRejected;
    }
    return CheckpointAnalysisPersistenceResult::Updated;
}

std::optional<ExperimentTransitionRecord>
PostgresSchedulerRepository::loadExperimentTransition(
    long long experimentId,
    bool forUpdate)
{
    std::string sql =
        "SELECT " + std::string{kExperimentProjection} + ","
        "status,phase,current_epoch,scheduler_priority,"
        "scheduler_resume_origin,active_scheduler_worker_attempt_id,"
        "worker_pid,worker_process_group_id,worker_process_start_identity,"
        "worker_executable,worker_command_line,"
        "EXISTS (SELECT 1 FROM experiment_scheduler_worker_attempt a "
        "WHERE a.experiment_id=experiment.experiment_id "
        "AND a.worker_kind='experiment' "
        "AND a.lifecycle_state IN ('reserved','spawned','running','observed',"
        "'stopped','identity_ambiguous')) "
        "FROM experiment WHERE experiment_id=$1";
    if (forUpdate)
        sql += " FOR UPDATE";
    sql += ";";

    const pqxx::result rows = transaction_.exec(
        sql, pqxx::params{experimentId});
    if (rows.empty())
        return std::nullopt;

    ExperimentTransitionRecord record;
    record.experiment = MapExperiment(rows[0]);
    record.status = rows[0][24].as<std::string>();
    record.phase = rows[0][25].as<std::string>();
    record.currentEpoch = OptionalCell<int>(rows[0], 26);
    record.schedulerPriority = rows[0][27].as<std::string>();
    record.schedulerResumeOrigin = rows[0][28].as<std::string>();
    record.activeWorkerAttemptId = OptionalCell<long long>(rows[0], 29);
    record.workerPid = OptionalCell<int>(rows[0], 30);
    record.workerProcessGroupId = OptionalCell<long long>(rows[0], 31);
    record.workerProcessStartIdentity =
        OptionalCell<std::string>(rows[0], 32);
    record.workerExecutable = OptionalCell<std::string>(rows[0], 33);
    record.workerCommandLine = OptionalCell<std::string>(rows[0], 34);
    record.hasAttachedWorkerAttempt = rows[0][35].as<bool>();
    return record;
}

ExperimentTransitionPersistenceResult
PostgresSchedulerRepository::applyExperimentTransition(
    const ExperimentTransitionUpdate& update)
{
    std::string sql =
        "UPDATE experiment SET status=$1,phase=$2,updated_at=now()";

    if (update.action == ExperimentTransitionAction::Cancel)
    {
        sql +=
            ",completed_at=now(),resume_requested=false,"
            "scheduler_resume_origin='none'";
    }
    else
    {
        sql +=
            ",started_at=NULL,worker_started_at=NULL,completed_at=NULL,"
            "exit_code=NULL,error_message=NULL,worker_pid=NULL,"
            "worker_process_group_id=NULL,worker_process_start_identity=NULL,"
            "worker_executable=NULL,worker_command_line=NULL,"
            "worker_control_state='running',worker_global_pause_request_id=NULL,"
            "active_scheduler_worker_attempt_id=NULL";
        if (update.action != ExperimentTransitionAction::RequeueTraining)
            sql += ",current_operation=NULL";
        if (update.action == ExperimentTransitionAction::RequeueAnalysis ||
            update.action == ExperimentTransitionAction::RequeueInference)
        {
            sql +=
                ",resume_requested=true,"
                "scheduler_resume_origin='operator'";
        }
        else
        {
            sql +=
                ",resume_requested=false,"
                "scheduler_resume_origin='none'";
        }
    }

    if (update.action == ExperimentTransitionAction::RequeueTraining)
    {
        if (!update.selectedResumeModelId)
        {
            throw std::invalid_argument(
                "requeue_training_requires_selected_checkpoint");
        }
        sql += ",current_operation='train',last_model_id=$3";
    }
    if (update.action == ExperimentTransitionAction::RequeueInference)
    {
        sql +=
            ",operator_forced_final_inference_rerun_requested=true";
    }
    if (update.selectedResumeModelId)
        sql += ",resume_model_id=$3";

    const bool selectedResumeModel = update.selectedResumeModelId.has_value();
    const int experimentParameter = selectedResumeModel ? 4 : 3;
    sql += " WHERE experiment_id=$" +
           std::to_string(experimentParameter);
    if (update.action == ExperimentTransitionAction::RequeueTraining)
    {
        sql +=
            " AND status=$5 AND phase=$6 AND scheduler_priority=$7 "
            "AND scheduler_resume_origin=$8 "
            "AND active_scheduler_worker_attempt_id IS NULL "
            "AND worker_pid IS NULL "
            "AND worker_process_group_id IS NULL "
            "AND worker_process_start_identity IS NULL "
            "AND worker_executable IS NULL "
            "AND worker_command_line IS NULL "
            "AND COALESCE(current_epoch,-1)<target_epochs "
            "AND NOT EXISTS (SELECT 1 "
            "FROM experiment_scheduler_worker_attempt a "
            "WHERE a.experiment_id=experiment.experiment_id "
            "AND a.worker_kind='experiment' "
            "AND a.lifecycle_state IN ('reserved','spawned','running',"
            "'observed','stopped','identity_ambiguous'))";
    }
    sql += " RETURNING experiment_id;";

    pqxx::result updated;
    if (update.action == ExperimentTransitionAction::RequeueTraining)
    {
        updated = transaction_.exec(
            sql,
            pqxx::params{
                update.newStatus,
                update.newPhase,
                update.selectedResumeModelId,
                update.experimentId,
                update.previousStatus,
                update.previousPhase,
                update.previousSchedulerPriority,
                update.previousSchedulerResumeOrigin});
    }
    else if (selectedResumeModel)
    {
        updated = transaction_.exec(
            sql,
            pqxx::params{
                update.newStatus,
                update.newPhase,
                update.selectedResumeModelId,
                update.experimentId});
    }
    else
    {
        updated = transaction_.exec(
            sql,
            pqxx::params{
                update.newStatus,
                update.newPhase,
                update.experimentId});
    }
    return updated.size() == 1
        ? ExperimentTransitionPersistenceResult::Updated
        : ExperimentTransitionPersistenceResult::AtomicPreconditionRejected;
}

bool PostgresSchedulerRepository::checkpointPolicySchemaAvailable()
{
    return ColumnExists(
               transaction_, "experiment", "checkpoint_policy_enabled") &&
           ColumnExists(
               transaction_, "experiment", "checkpoint_policy_revision") &&
           ColumnExists(
               transaction_, "experiment", "checkpoint_policy_hash") &&
           ColumnExists(
               transaction_, "experiment",
               "checkpoint_policy_last_decision_id") &&
           ColumnExists(
               transaction_, "experiment",
               "checkpoint_policy_stop_decision_id") &&
           ColumnExists(
               transaction_, "experiment_analysis_result",
               "analysis_scope") &&
           ColumnExists(
               transaction_, "experiment_analysis_result",
               "checkpoint_eval_id") &&
           ColumnExists(
               transaction_, "experiment_analysis_result",
               "checkpoint_epoch") &&
           ColumnExists(
               transaction_, "experiment_analysis_result",
               "parent_experiment_id") &&
           ColumnExists(
               transaction_, "experiment_checkpoint_eval", "analysis_id") &&
           ColumnExists(
               transaction_, "experiment_checkpoint_decision",
               "evidence_watermark") &&
           TableExists(transaction_, "experiment_checkpoint_decision");
}

std::optional<EA::ExperimentScheduler::CheckpointPolicyConfig>
PostgresSchedulerRepository::loadCheckpointPolicyForEvaluation(
    long long parentExperimentId)
{
    const pqxx::result rows = transaction_.exec(
        "SELECT checkpoint_policy_enabled, "
        "checkpoint_policy_min_leader_score, "
        "checkpoint_policy_min_infer_accuracy, checkpoint_policy_top_n, "
        "checkpoint_policy_scope, checkpoint_policy_stop_mode, "
        "checkpoint_policy_grace_evals, checkpoint_interval, target_epochs, "
        "current_epoch, stop_after_checkpoint_epoch, status, phase, "
        "(checkpoint_infer_enabled OR opportunistic_checkpoint_infer), "
        "checkpoint_policy_revision, checkpoint_policy_hash, "
        "active_scheduler_worker_attempt_id FROM experiment "
        "WHERE experiment_id=$1 FOR UPDATE;",
        pqxx::params{parentExperimentId});
    if (rows.empty())
        return std::nullopt;

    EA::ExperimentScheduler::CheckpointPolicyConfig config;
    config.enabled = rows[0][0].as<bool>();
    config.minLeaderScore = OptionalCell<double>(rows[0], 1);
    config.minInferAccuracy = OptionalCell<double>(rows[0], 2);
    config.topN = OptionalCell<int>(rows[0], 3);
    config.scope = rows[0][4].is_null()
        ? "symbol_horizon"
        : rows[0][4].as<std::string>();
    config.stopMode = rows[0][5].is_null()
        ? "next_checkpoint"
        : rows[0][5].as<std::string>();
    config.graceEvals = rows[0][6].is_null()
        ? 1
        : rows[0][6].as<int>();
    config.checkpointInterval = rows[0][7].as<int>();
    config.targetEpochs = rows[0][8].as<int>();
    config.currentEpoch = OptionalCell<int>(rows[0], 9);
    config.stopAfterCheckpointEpoch = OptionalCell<int>(rows[0], 10);
    config.status = rows[0][11].as<std::string>();
    config.phase = rows[0][12].as<std::string>();
    config.checkpointInferEnabled = rows[0][13].as<bool>();
    config.policyRevision = rows[0][14].as<long long>();
    config.persistedPolicyHash = OptionalCell<std::string>(rows[0], 15);
    config.activeTrainingAttemptId = OptionalCell<long long>(rows[0], 16);
    return config;
}

EA::ExperimentScheduler::CheckpointPolicyConfig
PostgresSchedulerRepository::reconcileCheckpointPolicyIdentity(
    const CheckpointEvaluationRecord& evaluation,
    EA::ExperimentScheduler::CheckpointPolicyConfig config)
{
    const std::string derivedPolicyHash =
        EA::ExperimentScheduler::CheckpointPolicySemanticHash(config);
    if (!config.persistedPolicyHash)
    {
        transaction_.exec(
            "UPDATE experiment SET checkpoint_policy_hash=$1, "
            "updated_at=now() WHERE experiment_id=$2 "
            "AND checkpoint_policy_hash IS NULL;",
            pqxx::params{derivedPolicyHash, evaluation.parentExperimentId});
        config.persistedPolicyHash = derivedPolicyHash;
    }
    else if (*config.persistedPolicyHash != derivedPolicyHash)
    {
        const pqxx::result revised = transaction_.exec(
            "UPDATE experiment SET checkpoint_policy_revision="
            "checkpoint_policy_revision+1, checkpoint_policy_hash=$1, "
            "updated_at=now() WHERE experiment_id=$2 "
            "RETURNING checkpoint_policy_revision;",
            pqxx::params{derivedPolicyHash, evaluation.parentExperimentId});
        config.policyRevision = revised.one_row()[0].as<long long>();
        config.persistedPolicyHash = derivedPolicyHash;
    }
    return config;
}

CheckpointPolicyEvidenceLoadResult
PostgresSchedulerRepository::loadCheckpointPolicyEvidence(
    const CheckpointEvaluationRecord& evaluation)
{
    const pqxx::result rows = transaction_.exec(
        "SELECT a.analysis_id, ir.id, a.infer_accuracy, a.leader_score, "
        "ir.from_date, ir.to_date FROM experiment_checkpoint_eval ce "
        "JOIN experiment e ON e.experiment_id="
        "COALESCE(ce.parent_experiment_id,ce.experiment_id) "
        "JOIN experiment_analysis_result a ON a.analysis_id=ce.analysis_id "
        "JOIN inference_eval_result ir "
        "ON ir.checkpoint_eval_id=ce.checkpoint_eval_id "
        "WHERE ce.checkpoint_eval_id=$1 AND ce.status='completed' "
        "AND ce.phase='done' "
        "AND COALESCE(ce.parent_experiment_id,ce.experiment_id)=$2 "
        "AND ce.checkpoint_model_id=$3 AND ce.checkpoint_epoch=$4 "
        "AND a.analysis_scope='checkpoint' "
        "AND a.analysis_status='completed' "
        "AND a.checkpoint_eval_id=ce.checkpoint_eval_id "
        "AND a.parent_experiment_id=e.experiment_id "
        "AND a.experiment_id=e.experiment_id "
        "AND a.model_id=ce.checkpoint_model_id "
        "AND a.checkpoint_epoch=ce.checkpoint_epoch "
        "AND a.symbol=e.symbol "
        "AND a.prediction_horizon=e.prediction_horizon "
        "AND ir.inference_scope='checkpoint' AND ir.status='completed' "
        "AND ir.parent_experiment_id=e.experiment_id "
        "AND ir.model_id=ce.checkpoint_model_id "
        "AND ir.checkpoint_epoch=ce.checkpoint_epoch "
        "AND ir.symbol=e.symbol "
        "AND ir.prediction_horizon=e.prediction_horizon "
        "AND ir.threshold_logret=e.c_next_threshold "
        "AND ir.from_date::date=e.infer_start::date "
        "AND ir.to_date::date=e.infer_end::date "
        "AND a.infer_accuracy IS NOT DISTINCT FROM ir.accuracy "
        "AND e.symbol=$5 AND e.prediction_horizon=$6;",
        pqxx::params{
            evaluation.checkpointEvalId,
            evaluation.parentExperimentId,
            evaluation.checkpointModelId,
            evaluation.checkpointEpoch,
            evaluation.symbol,
            evaluation.predictionHorizon});
    if (rows.size() != 1)
    {
        const pqxx::result state = transaction_.exec(
            "SELECT status, phase, analysis_id, "
            "(SELECT count(*) FROM inference_eval_result ir "
            "WHERE ir.checkpoint_eval_id="
            "experiment_checkpoint_eval.checkpoint_eval_id "
            "AND ir.inference_scope='checkpoint' "
            "AND ir.status='completed') FROM experiment_checkpoint_eval "
            "WHERE checkpoint_eval_id=$1;",
            pqxx::params{evaluation.checkpointEvalId});
        std::string reason;
        if (state.empty())
            reason = "checkpoint_eval_not_found";
        else if (state[0][0].as<std::string>() != "completed")
            reason = "checkpoint_eval_not_completed";
        else if (state[0][1].as<std::string>() != "done")
            reason = "checkpoint_eval_not_done";
        else if (state[0][2].is_null())
            reason = "checkpoint_analysis_not_linked";
        else if (state[0][3].as<int>() != 1)
            reason = "checkpoint_inference_result_not_exactly_one";
        else
            reason = "checkpoint_analysis_inference_identity_mismatch";
        return {std::nullopt, std::move(reason)};
    }

    ValidatedCheckpointPolicyEvidence evidence;
    evidence.analysisId = rows[0][0].as<long long>();
    evidence.inferenceEvalResultId = rows[0][1].as<long long>();
    evidence.inferenceAccuracy = OptionalCell<double>(rows[0], 2);
    evidence.leaderScore = OptionalCell<double>(rows[0], 3);
    evidence.inferenceFromDate = rows[0][4].as<std::string>();
    evidence.inferenceToDate = rows[0][5].as<std::string>();
    return {std::move(evidence), {}};
}

CheckpointPolicyPopulation
PostgresSchedulerRepository::loadCompletedCheckpointPolicyPopulation(
    long long parentExperimentId)
{
    const pqxx::result rows = transaction_.exec(
        "SELECT ce.checkpoint_eval_id::text, a.analysis_id::text, "
        "ir.id::text, ce.checkpoint_epoch::text, a.leader_score::text, "
        "a.infer_accuracy::text, ce.status, ce.phase, a.analysis_status "
        "FROM experiment_checkpoint_eval ce JOIN experiment e "
        "ON e.experiment_id="
        "COALESCE(ce.parent_experiment_id,ce.experiment_id) "
        "JOIN experiment_analysis_result a ON a.analysis_id=ce.analysis_id "
        "LEFT JOIN inference_eval_result ir "
        "ON ir.checkpoint_eval_id=ce.checkpoint_eval_id "
        "AND ir.inference_scope='checkpoint' AND ir.status='completed' "
        "WHERE COALESCE(ce.parent_experiment_id,ce.experiment_id)=$1 "
        "AND ce.status='completed' AND ce.phase='done' "
        "AND a.analysis_scope='checkpoint' "
        "AND a.analysis_status='completed' "
        "AND a.checkpoint_eval_id=ce.checkpoint_eval_id "
        "AND a.parent_experiment_id=e.experiment_id "
        "AND a.experiment_id=e.experiment_id "
        "AND a.model_id=ce.checkpoint_model_id "
        "AND a.checkpoint_epoch=ce.checkpoint_epoch "
        "AND a.symbol=e.symbol "
        "AND a.prediction_horizon=e.prediction_horizon "
        "ORDER BY ce.checkpoint_eval_id ASC;",
        pqxx::params{parentExperimentId});
    return {
        static_cast<int>(rows.size()),
        std::nullopt,
        CheckpointPolicyPopulationWatermark(rows)};
}

CheckpointPolicyPopulation
PostgresSchedulerRepository::loadCheckpointPolicyRankPopulation(
    const CheckpointEvaluationRecord& evaluation,
    const EA::ExperimentScheduler::CheckpointPolicyConfig& config)
{
    std::string sql =
        "SELECT ce.checkpoint_eval_id::text, a.analysis_id::text, "
        "ir.id::text, ce.checkpoint_epoch::text, a.leader_score::text, "
        "a.infer_accuracy::text, ce.status, ce.phase, a.analysis_status "
        "FROM experiment_checkpoint_eval ce JOIN experiment e "
        "ON e.experiment_id="
        "COALESCE(ce.parent_experiment_id,ce.experiment_id) "
        "JOIN experiment_analysis_result a ON a.analysis_id=ce.analysis_id "
        "LEFT JOIN inference_eval_result ir "
        "ON ir.checkpoint_eval_id=ce.checkpoint_eval_id "
        "AND ir.inference_scope='checkpoint' AND ir.status='completed' "
        "WHERE ce.status='completed' AND ce.phase='done' "
        "AND a.analysis_scope='checkpoint' "
        "AND a.analysis_status='completed' "
        "AND a.checkpoint_eval_id=ce.checkpoint_eval_id "
        "AND a.parent_experiment_id=e.experiment_id "
        "AND a.experiment_id=e.experiment_id "
        "AND a.model_id=ce.checkpoint_model_id "
        "AND a.checkpoint_epoch=ce.checkpoint_epoch "
        "AND a.symbol=e.symbol "
        "AND a.prediction_horizon=e.prediction_horizon ";
    pqxx::result rows;
    if (config.scope == "symbol_horizon")
    {
        sql += "AND a.symbol=$1 AND a.prediction_horizon=$2 ";
        sql += "ORDER BY a.leader_score DESC NULLS LAST, "
               "a.infer_accuracy DESC NULLS LAST, "
               "ce.checkpoint_epoch DESC, ce.checkpoint_eval_id ASC;";
        rows = transaction_.exec(
            sql,
            pqxx::params{evaluation.symbol, evaluation.predictionHorizon});
    }
    else if (config.scope == "horizon")
    {
        sql += "AND a.prediction_horizon=$1 ";
        sql += "ORDER BY a.leader_score DESC NULLS LAST, "
               "a.infer_accuracy DESC NULLS LAST, "
               "ce.checkpoint_epoch DESC, ce.checkpoint_eval_id ASC;";
        rows = transaction_.exec(
            sql, pqxx::params{evaluation.predictionHorizon});
    }
    else if (config.scope == "global")
    {
        sql += "ORDER BY a.leader_score DESC NULLS LAST, "
               "a.infer_accuracy DESC NULLS LAST, "
               "ce.checkpoint_epoch DESC, ce.checkpoint_eval_id ASC;";
        rows = transaction_.exec(sql);
    }
    else
        return {};

    CheckpointPolicyPopulation population{
        static_cast<int>(rows.size()),
        std::nullopt,
        CheckpointPolicyPopulationWatermark(rows)};
    for (pqxx::result::size_type index = 0; index < rows.size(); ++index)
    {
        if (rows[index][0].as<long long>() == evaluation.checkpointEvalId)
        {
            population.rankValue = static_cast<int>(index + 1);
            break;
        }
    }
    return population;
}

PersistedCheckpointPolicyDecision
PostgresSchedulerRepository::persistCheckpointPolicyDecision(
    const CheckpointEvaluationRecord& evaluation,
    const EA::ExperimentScheduler::CheckpointPolicyConfig& config,
    const EA::ExperimentScheduler::CheckpointPolicyDecision& decision,
    const ValidatedCheckpointPolicyEvidence& evidence,
    const EA::ExperimentScheduler::CheckpointPolicyEvidenceIdentity&
        evidenceIdentity)
{
    const pqxx::result mismatches = transaction_.exec(
        "SELECT parent_experiment_id,checkpoint_epoch,checkpoint_model_id "
        "FROM experiment_checkpoint_decision WHERE checkpoint_eval_id=$1 "
        "AND (parent_experiment_id<>$2 OR checkpoint_epoch<>$3 "
        "OR checkpoint_model_id<>$4) LIMIT 1 FOR UPDATE;",
        pqxx::params{
            evaluation.checkpointEvalId,
            evaluation.parentExperimentId,
            evaluation.checkpointEpoch,
            evaluation.checkpointModelId});
    if (!mismatches.empty())
        throw std::runtime_error(
            "checkpoint policy decision identity mismatch");

    const std::string policyHash =
        EA::ExperimentScheduler::CheckpointPolicySemanticHash(config);
    const std::string evidenceWatermark =
        EA::ExperimentScheduler::CheckpointPolicyEvidenceWatermark(
            evidenceIdentity);
    std::string identityStatus = "active";
    std::optional<std::string> supersededReason;
    const pqxx::result terminal = transaction_.exec(
        "SELECT checkpoint_policy_stop_decision_id FROM experiment "
        "WHERE experiment_id=$1;",
        pqxx::params{evaluation.parentExperimentId});
    if (!terminal.empty() && !terminal[0][0].is_null())
    {
        identityStatus = "superseded";
        supersededReason = "stop_action_already_applied";
    }
    else
    {
        const pqxx::result newer = transaction_.exec(
            "SELECT checkpoint_decision_id "
            "FROM experiment_checkpoint_decision "
            "WHERE parent_experiment_id=$1 "
            "AND identity_status IN ('active','action_applied') "
            "AND (checkpoint_epoch>$2 OR "
            "(checkpoint_epoch=$2 AND checkpoint_eval_id>$3)) "
            "ORDER BY checkpoint_epoch DESC,checkpoint_eval_id DESC,"
            "checkpoint_decision_id DESC LIMIT 1;",
            pqxx::params{
                evaluation.parentExperimentId,
                evaluation.checkpointEpoch,
                evaluation.checkpointEvalId});
        if (!newer.empty())
        {
            identityStatus = "superseded";
            supersededReason = "newer_checkpoint_decision_exists";
        }
    }

    const pqxx::result inserted = transaction_.exec(
        "INSERT INTO experiment_checkpoint_decision("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,"
        "checkpoint_model_id,analysis_id,inference_eval_result_id,"
        "policy_revision,policy_hash,evidence_watermark,"
        "rank_population_watermark,identity_status,superseded_at,"
        "superseded_reason,decision,reason,leader_score,infer_accuracy,"
        "rank_value,rank_scope,requested_stop_epoch) VALUES("
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,"
        "CASE WHEN $11='superseded' THEN clock_timestamp() ELSE NULL END,"
        "$12,$13,$14,$15,$16,$17,$18,$19) "
        "ON CONFLICT(checkpoint_eval_id,policy_revision,policy_hash,"
        "evidence_watermark) WHERE policy_revision IS NOT NULL "
        "AND policy_hash IS NOT NULL AND evidence_watermark IS NOT NULL "
        "DO NOTHING RETURNING checkpoint_decision_id,identity_status;",
        pqxx::params{
            evaluation.checkpointEvalId,
            evaluation.parentExperimentId,
            evaluation.checkpointEpoch,
            evaluation.checkpointModelId,
            evidence.analysisId,
            evidence.inferenceEvalResultId,
            config.policyRevision,
            policyHash,
            evidenceWatermark,
            evidenceIdentity.rankPopulationWatermark,
            identityStatus,
            supersededReason,
            decision.decision,
            decision.reason,
            evidence.leaderScore,
            evidence.inferenceAccuracy,
            decision.rankValue,
            config.scope,
            decision.requestedStopEpoch});

    PersistedCheckpointPolicyDecision persisted;
    if (!inserted.empty())
    {
        persisted.decisionId = inserted[0][0].as<long long>();
        persisted.identityStatus = inserted[0][1].as<std::string>();
    }
    else
    {
        const pqxx::result existing = transaction_.exec(
            "SELECT checkpoint_decision_id,identity_status "
            "FROM experiment_checkpoint_decision "
            "WHERE checkpoint_eval_id=$1 AND policy_revision=$2 "
            "AND policy_hash=$3 AND evidence_watermark=$4 FOR UPDATE;",
            pqxx::params{
                evaluation.checkpointEvalId,
                config.policyRevision,
                policyHash,
                evidenceWatermark});
        if (existing.size() != 1)
        {
            throw std::runtime_error(
                "checkpoint policy idempotent decision lookup failed");
        }
        persisted.decisionId = existing[0][0].as<long long>();
        persisted.identityStatus = existing[0][1].as<std::string>();
        persisted.reused = true;
    }

    if (!persisted.reused && persisted.identityStatus == "active")
    {
        transaction_.exec(
            "UPDATE experiment_checkpoint_decision SET "
            "identity_status='superseded',superseded_at=clock_timestamp(),"
            "superseded_reason=CASE WHEN checkpoint_eval_id=$1 THEN "
            "'policy_or_evidence_changed' ELSE "
            "'newer_checkpoint_decision_became_authoritative' END,"
            "superseded_by_decision_id=$2 WHERE parent_experiment_id=$3 "
            "AND checkpoint_decision_id<>$2 AND identity_status='active' "
            "AND (checkpoint_epoch<$4 OR "
            "(checkpoint_epoch=$4 AND checkpoint_eval_id<$1) OR "
            "checkpoint_eval_id=$1);",
            pqxx::params{
                evaluation.checkpointEvalId,
                persisted.decisionId,
                evaluation.parentExperimentId,
                evaluation.checkpointEpoch});
    }

    transaction_.exec(
        "UPDATE experiment SET checkpoint_policy_last_decision=$1,"
        "checkpoint_policy_last_decision_at=now(),"
        "checkpoint_policy_last_checkpoint_eval_id=$2,"
        "checkpoint_policy_last_reason=$3,"
        "checkpoint_policy_last_decision_id=$4,"
        "checkpoint_policy_hash=COALESCE(checkpoint_policy_hash,$5),"
        "updated_at=now() WHERE experiment_id=$6 AND $7='active';",
        pqxx::params{
            decision.decision,
            evaluation.checkpointEvalId,
            decision.reason,
            persisted.decisionId,
            policyHash,
            evaluation.parentExperimentId,
            persisted.identityStatus});
    return persisted;
}

std::string
PostgresSchedulerRepository::markCheckpointPolicyDecisionSuperseded(
    long long decisionId,
    std::string_view reason)
{
    transaction_.exec(
        "UPDATE experiment_checkpoint_decision SET "
        "identity_status='superseded',superseded_at=clock_timestamp(),"
        "superseded_reason=$1 WHERE checkpoint_decision_id=$2 "
        "AND identity_status='active';",
        pqxx::params{reason, decisionId});
    return std::string{reason};
}

std::string PostgresSchedulerRepository::applyCheckpointPolicyStopRequest(
    const CheckpointEvaluationRecord& evaluation,
    const EA::ExperimentScheduler::CheckpointPolicyConfig& config,
    const EA::ExperimentScheduler::CheckpointPolicyDecision& decision,
    const PersistedCheckpointPolicyDecision& persisted,
    const std::string& expectedEvidenceWatermark)
{
    if (!decision.requestedStopEpoch)
        return "no_requested_stop_epoch";
    if (persisted.identityStatus != "active")
        return "decision_superseded";

    const CheckpointPolicyEvidenceLoadResult currentEvidence =
        loadCheckpointPolicyEvidence(evaluation);
    if (!currentEvidence.evidence)
    {
        return markCheckpointPolicyDecisionSuperseded(
            persisted.decisionId,
            "stale_evidence:" + currentEvidence.rejectionReason);
    }
    CheckpointPolicyPopulation completedPopulation =
        loadCompletedCheckpointPolicyPopulation(
            evaluation.parentExperimentId);
    CheckpointPolicyPopulation rankPopulation =
        loadCheckpointPolicyRankPopulation(evaluation, config);
    const CheckpointPolicyDecisionContext currentContext =
        PlanCheckpointPolicyDecision(
            evaluation,
            config,
            *currentEvidence.evidence,
            std::move(completedPopulation),
            std::move(rankPopulation));
    const std::string currentEvidenceWatermark =
        EA::ExperimentScheduler::CheckpointPolicyEvidenceWatermark(
            MakeCheckpointPolicyEvidenceIdentity(
                evaluation,
                *currentEvidence.evidence,
                currentContext,
                config));
    if (currentEvidenceWatermark != expectedEvidenceWatermark)
    {
        return markCheckpointPolicyDecisionSuperseded(
            persisted.decisionId, "stale_evidence_watermark");
    }

    const pqxx::result state = transaction_.exec(
        "SELECT e.status,e.phase,e.current_epoch,e.target_epochs,"
        "e.checkpoint_policy_revision,e.checkpoint_policy_hash,"
        "e.stop_after_checkpoint_epoch,e.checkpoint_policy_stop_decision_id,"
        "e.active_scheduler_worker_attempt_id,d.identity_status,"
        "d.evidence_watermark,d.policy_hash,d.policy_revision,"
        "EXISTS(SELECT 1 FROM experiment_scheduler_worker_attempt a "
        "JOIN experiment_scheduler_lease l ON l.singleton "
        "WHERE a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
        "AND a.experiment_id=e.experiment_id "
        "AND a.worker_kind='experiment' AND a.lifecycle_phase='train' "
        "AND a.lifecycle_state IN ('reserved','spawned','running','observed') "
        "AND a.scheduler_invocation_id=l.owner_scheduler_invocation_id "
        "AND a.scheduler_fencing_token=l.fencing_token "
        "AND l.authority_state='active' "
        "AND l.expires_at>clock_timestamp()) FROM experiment e "
        "JOIN experiment_checkpoint_decision d "
        "ON d.checkpoint_decision_id=$1 WHERE e.experiment_id=$2 FOR UPDATE;",
        pqxx::params{persisted.decisionId, evaluation.parentExperimentId});
    if (state.size() != 1)
    {
        return markCheckpointPolicyDecisionSuperseded(
            persisted.decisionId, "parent_or_decision_missing");
    }
    const pqxx::row row = state[0];
    const std::string currentPolicyHash =
        EA::ExperimentScheduler::CheckpointPolicySemanticHash(config);
    std::string fenceReason;
    if (row[0].as<std::string>() != "running" ||
        row[1].as<std::string>() != "train")
    {
        fenceReason = "parent_not_running_train";
    }
    else if (row[4].as<long long>() != config.policyRevision ||
             row[5].is_null() ||
             row[5].as<std::string>() != currentPolicyHash ||
             row[11].as<std::string>() != currentPolicyHash ||
             row[12].as<long long>() != config.policyRevision)
    {
        fenceReason = "stale_policy_identity";
    }
    else if (row[9].as<std::string>() != "active")
        fenceReason = "decision_not_active";
    else if (row[10].as<std::string>() != expectedEvidenceWatermark)
        fenceReason = "stale_evidence_watermark";
    else if (!row[7].is_null())
        fenceReason = "terminal_stop_decision_already_applied";
    else if (!row[6].is_null())
        fenceReason = "conflicting_stop_request_already_present";
    else if (row[8].is_null() || !row[13].as<bool>())
        fenceReason =
            "active_training_attempt_not_scheduler_authoritative";
    else if (*decision.requestedStopEpoch <=
             (row[2].is_null()
                  ? evaluation.checkpointEpoch
                  : row[2].as<int>()))
    {
        fenceReason = "requested_stop_epoch_not_in_future";
    }
    else if (*decision.requestedStopEpoch >= row[3].as<int>())
        fenceReason = "requested_stop_epoch_not_before_target";
    if (!fenceReason.empty())
    {
        return markCheckpointPolicyDecisionSuperseded(
            persisted.decisionId, fenceReason);
    }

    const pqxx::result applied = transaction_.exec(
        "UPDATE experiment SET stop_after_checkpoint_epoch=$1,"
        "checkpoint_policy_stop_decision_id=$2,"
        "checkpoint_policy_last_decision_id=$2,updated_at=now() "
        "WHERE experiment_id=$3 AND status='running' AND phase='train' "
        "AND stop_after_checkpoint_epoch IS NULL "
        "AND checkpoint_policy_stop_decision_id IS NULL "
        "AND checkpoint_policy_revision=$4 AND checkpoint_policy_hash=$5 "
        "AND COALESCE(current_epoch,$7)<$1 AND target_epochs>$1 "
        "AND EXISTS(SELECT 1 FROM experiment_checkpoint_decision d "
        "WHERE d.checkpoint_decision_id=$2 AND d.parent_experiment_id=$3 "
        "AND d.identity_status='active' AND d.policy_revision=$4 "
        "AND d.policy_hash=$5 AND d.evidence_watermark=$6) "
        "AND NOT EXISTS(SELECT 1 FROM experiment_checkpoint_decision newer "
        "JOIN experiment_checkpoint_decision current_decision "
        "ON current_decision.checkpoint_decision_id=$2 "
        "WHERE newer.parent_experiment_id=$3 "
        "AND newer.identity_status IN ('active','action_applied') "
        "AND (newer.checkpoint_epoch>current_decision.checkpoint_epoch OR "
        "(newer.checkpoint_epoch=current_decision.checkpoint_epoch "
        "AND newer.checkpoint_eval_id>current_decision.checkpoint_eval_id))) "
        "AND EXISTS(SELECT 1 FROM experiment_scheduler_worker_attempt a "
        "JOIN experiment_scheduler_lease l ON l.singleton "
        "WHERE a.worker_attempt_id="
        "experiment.active_scheduler_worker_attempt_id "
        "AND a.experiment_id=experiment.experiment_id "
        "AND a.worker_kind='experiment' AND a.lifecycle_phase='train' "
        "AND a.lifecycle_state IN ('reserved','spawned','running','observed') "
        "AND a.scheduler_invocation_id=l.owner_scheduler_invocation_id "
        "AND a.scheduler_fencing_token=l.fencing_token "
        "AND l.authority_state='active' "
        "AND l.expires_at>clock_timestamp()) "
        "RETURNING active_scheduler_worker_attempt_id;",
        pqxx::params{
            decision.requestedStopEpoch,
            persisted.decisionId,
            evaluation.parentExperimentId,
            config.policyRevision,
            currentPolicyHash,
            expectedEvidenceWatermark,
            evaluation.checkpointEpoch});
    if (applied.size() != 1 || applied[0][0].is_null())
    {
        return markCheckpointPolicyDecisionSuperseded(
            persisted.decisionId, "atomic_stop_fence_failed");
    }
    const pqxx::result attributed = transaction_.exec(
        "UPDATE experiment_checkpoint_decision SET "
        "identity_status='action_applied',stop_request_applied=true,"
        "stop_request_applied_at=clock_timestamp(),"
        "stop_action_worker_attempt_id=$1 WHERE checkpoint_decision_id=$2 "
        "AND identity_status='active' RETURNING checkpoint_decision_id;",
        pqxx::params{applied[0][0].as<long long>(), persisted.decisionId});
    if (attributed.size() != 1)
    {
        throw std::runtime_error(
            "checkpoint policy stop decision attribution failed");
    }
    return "stop_request_applied";
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
