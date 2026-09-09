#pragma once

#include <pqxx/pqxx>

#include <optional>

namespace EA::Scheduler
{

struct AuthoritativeFinalInferenceResult
{
    long long resultId = 0;
    long long modelId = 0;
    bool forcedFinalInferenceRerun = false;
};

// This is the durable completion contract shared by scheduler-owned orphan
// recovery and the explicit administrative exact-attempt reconciliation path.
// A profitability observation or log marker is deliberately insufficient.
inline std::optional<AuthoritativeFinalInferenceResult>
FindAuthoritativeFinalInferenceResultForWorkerAttempt(
    pqxx::transaction_base& transaction,
    long long experimentId,
    long long workerAttemptId)
{
    const pqxx::row relation = transaction.exec(
        "SELECT to_regclass('inference_eval_result');").one_row();
    if (relation[0].is_null()) return std::nullopt;

    const pqxx::result rows = transaction.exec_params(
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
        "WHERE e.experiment_id=$1 AND e.status='running' "
        "AND e.phase='infer' "
        "AND e.active_scheduler_worker_attempt_id=a.worker_attempt_id "
        "ORDER BY r.completed_at DESC,r.id DESC LIMIT 1;",
        experimentId,
        workerAttemptId);
    if (rows.empty()) return std::nullopt;
    return AuthoritativeFinalInferenceResult{
        rows[0][0].as<long long>(),
        rows[0][1].as<long long>(),
        rows[0][2].as<bool>()};
}

} // namespace EA::Scheduler
