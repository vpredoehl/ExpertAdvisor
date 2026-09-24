#include "ManagedInferenceApplication.hpp"

#include "CanonicalSymbol.hpp"
#include "EconomicEventRepository.hpp"
#include "GlobalExperimentControl.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "SchedulerCore/SchedulerWorkerRegistration.hpp"
#include "WorkerLifecycleDiagnostics.hpp"

#include <cmath>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <pqxx/pqxx>

namespace EA::Inference
{
namespace
{
void LogManagedInferenceStage(const char* stage,
                              const ManagedInferenceRequest& request)
{
    std::cout << "MANAGED_INFERENCE_STAGE"
              << ",stage=" << stage
              << ",model_id=" << request.modelId
              << ",worker_attempt_id=" << request.workerAttemptId
              << std::endl;
}

struct Binding
{
    bool checkpoint = false;
    long long modelId = -1;
    std::optional<long long> finalExperimentId;
    std::optional<long long> checkpointEvalId;
    std::optional<long long> parentExperimentId;
    std::optional<long long> checkpointModelId;
    std::optional<int> checkpointEpoch;
};

struct Snapshot
{
    DBIO::PgModelIO::PersistedModelMaterialization model;
    Binding binding;
    std::string symbol;
    std::size_t horizon = 0;
    float threshold = 0.0f;
    std::size_t window = 0;
    std::size_t logicalOutputStartIndex = 0;
    EA::LSTM::TargetType targetType = EA::LSTM::TargetType::UpNeutralDownReturn;
    std::optional<std::size_t> completedEpochs;
};

std::string DatePrefix(const std::string& value)
{
    return value.substr(0, std::min<std::size_t>(10, value.size()));
}

bool IsIsoDate(const std::string& value)
{
    const std::string date = DatePrefix(value);
    if (date.size() != 10 || date[4] != '-' || date[7] != '-') return false;
    for (std::size_t index = 0; index < date.size(); ++index)
        if (index != 4 && index != 7 &&
            !std::isdigit(static_cast<unsigned char>(date[index])))
            return false;
    const int month = std::stoi(date.substr(5, 2));
    const int day = std::stoi(date.substr(8, 2));
    return month >= 1 && month <= 12 && day >= 1 && day <= 31;
}

template <typename... Args>
pqxx::result Execute(pqxx::work& transaction, const char* query,
                     Args&&... arguments)
{
    return transaction.exec(query,
                            pqxx::params{std::forward<Args>(arguments)...});
}

void ValidateRequest(const ManagedInferenceRequest& request)
{
    if (request.modelId <= 0) throw std::invalid_argument("managed_inference_invalid_model_id");
    if (request.workerAttemptId <= 0) throw std::invalid_argument("managed_inference_invalid_worker_attempt_id");
    if (request.finalExperimentId.has_value() == request.checkpointEvalId.has_value())
        throw std::invalid_argument("managed_inference_requires_exactly_one_scheduler_binding");
    if ((request.finalExperimentId && *request.finalExperimentId <= 0) ||
        (request.checkpointEvalId && *request.checkpointEvalId <= 0) ||
        !IsIsoDate(request.fromDate) || !IsIsoDate(request.toDate) ||
        DatePrefix(request.fromDate) > DatePrefix(request.toDate))
        throw std::invalid_argument("managed_inference_invalid_range_or_binding");
    if (request.database.forexConnectionString.empty() || request.database.lstmConnectionString.empty())
        throw std::invalid_argument("managed_inference_missing_database_settings");
}

void ValidateOptional(const ManagedInferenceRequest& request, const Snapshot& snapshot)
{
    const auto mismatch = [](const char* name) { throw std::runtime_error(std::string{"CONFIG_MISMATCH for "} + name); };
    if (request.requestedSymbol && EA::CanonicalSymbol::Normalize(*request.requestedSymbol) != snapshot.symbol) mismatch("--symbol");
    if (request.requestedPredictionHorizon && *request.requestedPredictionHorizon != snapshot.horizon) mismatch("--prediction-horizon");
    if (request.requestedThresholdLogret && std::fabs(*request.requestedThresholdLogret - snapshot.threshold) > 1e-7) mismatch("--threshold");
    if (request.requestedWindowSize && *request.requestedWindowSize != snapshot.window) mismatch("--window-size");
    if (request.requestedHiddenSize && *request.requestedHiddenSize != snapshot.model.modelMeta.hiddenSize) mismatch("--hidden-size");
    if (request.requestedDonchianLookback && *request.requestedDonchianLookback != snapshot.model.donchianLookback) mismatch("--donchian-lookback");
    if (request.requestedDonchian20Mode && *request.requestedDonchian20Mode != snapshot.model.donchian20Mode) mismatch("--donchian20-mode");
    if (request.requestedFeatureWarmupScope && *request.requestedFeatureWarmupScope != snapshot.model.featureWarmupScope) mismatch("--feature-warmup-scope");
}

Binding ReadBinding(pqxx::work& transaction, const ManagedInferenceRequest& request,
                    const std::string& symbol, std::size_t horizon, float threshold)
{
    Binding binding;
    binding.modelId = request.modelId;
    binding.checkpoint = request.checkpointEvalId.has_value();
    if (binding.checkpoint)
    {
        binding.checkpointEvalId = request.checkpointEvalId;
        const auto rows = Execute(transaction,
            "SELECT COALESCE(ce.parent_experiment_id,ce.experiment_id),ce.experiment_id,ce.checkpoint_model_id,ce.checkpoint_epoch,ce.status,ce.phase,e.symbol,e.prediction_horizon,e.c_next_threshold,e.infer_start::date::text,e.infer_end::date::text,m.experiment_id "
            "FROM experiment_checkpoint_eval ce JOIN experiment e ON e.experiment_id=COALESCE(ce.parent_experiment_id,ce.experiment_id) JOIN model m ON m.model_id=ce.checkpoint_model_id WHERE ce.checkpoint_eval_id=$1;",
            *request.checkpointEvalId);
        if (rows.size() != 1) throw std::runtime_error("checkpoint_eval_not_found");
        const auto row = rows.one_row();
        binding.parentExperimentId = row[0].as<long long>();
        binding.checkpointModelId = row[2].as<long long>();
        binding.checkpointEpoch = row[3].as<int>();
        if (*binding.checkpointModelId != request.modelId || row[1].as<long long>() != *binding.parentExperimentId ||
            (row[4].as<std::string>() != "pending" && row[4].as<std::string>() != "running") || row[5].as<std::string>() != "infer" ||
            EA::CanonicalSymbol::Normalize(row[6].as<std::string>()) != symbol || row[7].as<int>() != static_cast<int>(horizon) ||
            std::fabs(row[8].as<double>() - threshold) > 1e-7 || row[9].is_null() || row[10].is_null() ||
            row[9].as<std::string>() != DatePrefix(request.fromDate) || row[10].as<std::string>() != DatePrefix(request.toDate) ||
            row[11].is_null() || row[11].as<long long>() != *binding.parentExperimentId)
            throw std::runtime_error("checkpoint_scheduler_inference_binding_mismatch");
    }
    else
    {
        binding.finalExperimentId = request.finalExperimentId;
        const auto rows = Execute(transaction,
            "SELECT e.last_model_id,e.status,e.phase,e.symbol,e.prediction_horizon,e.c_next_threshold,e.infer_start::date::text,e.infer_end::date::text,m.experiment_id FROM experiment e JOIN model m ON m.model_id=$2 WHERE e.experiment_id=$1;",
            *request.finalExperimentId, request.modelId);
        if (rows.size() != 1) throw std::runtime_error("scheduler_final_experiment_not_found");
        const auto row = rows.one_row();
        if (row[0].is_null() || row[0].as<long long>() != request.modelId ||
            (row[1].as<std::string>() != "pending" && row[1].as<std::string>() != "running") || row[2].as<std::string>() != "infer" ||
            EA::CanonicalSymbol::Normalize(row[3].as<std::string>()) != symbol || row[4].as<int>() != static_cast<int>(horizon) ||
            std::fabs(row[5].as<double>() - threshold) > 1e-7 || row[6].is_null() || row[7].is_null() ||
            row[6].as<std::string>() != DatePrefix(request.fromDate) || row[7].as<std::string>() != DatePrefix(request.toDate) ||
            row[8].is_null() || row[8].as<long long>() != *request.finalExperimentId)
            throw std::runtime_error("scheduler_final_inference_binding_mismatch");
    }
    return binding;
}

Snapshot Materialize(pqxx::work& transaction, const ManagedInferenceRequest& request)
{
    Snapshot snapshot;
    snapshot.model = DBIO::PgModelIO::ReadPersistedModelMaterialization(
        transaction, request.modelId,
        {.afterStage = [&request](const char* stage)
         {
             LogManagedInferenceStage(stage, request);
         }});
    LogManagedInferenceStage("detached_materialization_state_transferred", request);
    if (!snapshot.model.trainConfigMeta || snapshot.model.trainConfigMeta->rows != 1 ||
        snapshot.model.trainConfigMeta->values.size() < DBIO::PgModelIO::kTrainConfigMetaFieldCount)
        throw std::runtime_error("managed_inference_train_config_meta_missing");
    const auto& values = snapshot.model.trainConfigMeta->values;
    if (std::llround(values[0]) != DBIO::PgModelIO::kTrainConfigMetaSchemaVersion ||
        std::llround(values[4]) != DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId)
        throw std::runtime_error("managed_inference_train_config_meta_unsupported");
    LogManagedInferenceStage("detached_materialization_train_config_validated", request);
    if (!snapshot.model.trainSymbol && !request.requestedSymbol)
        throw std::runtime_error("managed_inference_symbol_missing");
    // A scheduler-managed request intentionally omits --symbol when the
    // persisted model carries its immutable training symbol.  Do not evaluate
    // the optional CLI fallback unless it is actually needed.
    const std::string& symbol = snapshot.model.trainSymbol
        ? *snapshot.model.trainSymbol
        : *request.requestedSymbol;
    snapshot.symbol = EA::CanonicalSymbol::Normalize(symbol);
    snapshot.horizon = static_cast<std::size_t>(std::llround(values[1]));
    snapshot.threshold = static_cast<float>(values[2]);
    snapshot.window = static_cast<std::size_t>(std::llround(values[3]));
    snapshot.targetType = snapshot.model.targetMeta ? snapshot.model.targetMeta->targetType : EA::LSTM::TargetType::UpNeutralDownReturn;
    snapshot.completedEpochs = snapshot.model.completedEpoch;
    if (snapshot.horizon == 0 || snapshot.window == 0) throw std::runtime_error("managed_inference_invalid_persisted_runtime_config");
    ValidateOptional(request, snapshot);
    LogManagedInferenceStage("managed_request_persisted_config_validated", request);
    snapshot.binding = ReadBinding(transaction, request, snapshot.symbol, snapshot.horizon, snapshot.threshold);
    LogManagedInferenceStage("detached_materialization_snapshot_validated", request);
    return snapshot;
}

void RevalidateForWrite(pqxx::work& transaction, const ManagedInferenceRequest& request, const Snapshot& snapshot)
{
    EA::GlobalExperimentControl::AcquireCoordinationLock(transaction);
    if (snapshot.binding.checkpoint)
        Execute(transaction, "SELECT ce.checkpoint_eval_id FROM experiment_checkpoint_eval ce JOIN experiment e ON e.experiment_id=COALESCE(ce.parent_experiment_id,ce.experiment_id) JOIN model m ON m.model_id=ce.checkpoint_model_id WHERE ce.checkpoint_eval_id=$1 FOR UPDATE OF ce,e,m;", *snapshot.binding.checkpointEvalId);
    else
        Execute(transaction, "SELECT e.experiment_id FROM experiment e JOIN model m ON m.model_id=$2 WHERE e.experiment_id=$1 FOR UPDATE OF e,m;", *snapshot.binding.finalExperimentId, request.modelId);
    const Binding current = ReadBinding(transaction, request, snapshot.symbol, snapshot.horizon, snapshot.threshold);
    if (current.checkpoint != snapshot.binding.checkpoint || current.modelId != snapshot.binding.modelId || current.finalExperimentId != snapshot.binding.finalExperimentId || current.checkpointEvalId != snapshot.binding.checkpointEvalId || current.parentExperimentId != snapshot.binding.parentExperimentId || current.checkpointEpoch != snapshot.binding.checkpointEpoch)
        throw std::runtime_error("scheduler_inference_binding_stale");
}

long long PersistResult(pqxx::work& transaction, const ManagedInferenceRequest& request, const Snapshot& snapshot, const InferenceEvaluationFacts::EvaluationFacts& facts, bool& existing)
{
    const long long epochs = snapshot.completedEpochs ? static_cast<long long>(*snapshot.completedEpochs) : -1;
    const auto& a = facts.acceptance;
    pqxx::result row;
    if (snapshot.binding.checkpoint)
    {
        const auto old = Execute(transaction, "SELECT id,model_id,parent_experiment_id,checkpoint_epoch,status FROM inference_eval_result WHERE checkpoint_eval_id=$1 AND inference_scope='checkpoint' FOR UPDATE;", *snapshot.binding.checkpointEvalId);
        if (!old.empty() && old[0][4].as<std::string>() == "completed")
        {
            if (old[0][1].as<long long>() != request.modelId || old[0][2].is_null() || old[0][2].as<long long>() != *snapshot.binding.parentExperimentId || old[0][3].is_null() || old[0][3].as<int>() != *snapshot.binding.checkpointEpoch)
                throw std::runtime_error("existing_checkpoint_inference_result_identity_mismatch");
            existing = true;
            return old[0][0].as<long long>();
        }
        row = Execute(transaction, "INSERT INTO inference_eval_result (model_id,symbol,prediction_horizon,threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,completed_epochs,accuracy,accept_model,reject_reason,pred_down,pred_neutral,pred_up,status,completed_at,inference_scope,checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,producer_worker_attempt_id) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NULLIF($10::bigint,-1),$11,$12,$13,$14,$15,$16,'completed',now(),'checkpoint',$17,$18,$19,$20) ON CONFLICT (checkpoint_eval_id) WHERE inference_scope='checkpoint' DO UPDATE SET accuracy=EXCLUDED.accuracy,accept_model=EXCLUDED.accept_model,reject_reason=EXCLUDED.reject_reason,pred_down=EXCLUDED.pred_down,pred_neutral=EXCLUDED.pred_neutral,pred_up=EXCLUDED.pred_up,status='completed',completed_at=now(),producer_worker_attempt_id=EXCLUDED.producer_worker_attempt_id RETURNING id;", request.modelId,snapshot.symbol,static_cast<long long>(snapshot.horizon),snapshot.threshold,static_cast<long long>(snapshot.window),DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId,static_cast<int>(snapshot.targetType),DatePrefix(request.fromDate),DatePrefix(request.toDate),epochs,facts.accuracy,a.acceptModel,a.rejectReason,a.predFrac[0],a.predFrac[1],a.predFrac[2],*snapshot.binding.checkpointEvalId,*snapshot.binding.parentExperimentId,*snapshot.binding.checkpointEpoch,request.workerAttemptId);
    }
    else
        row = Execute(transaction, "INSERT INTO inference_eval_result (model_id,symbol,prediction_horizon,threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,completed_epochs,accuracy,accept_model,reject_reason,pred_down,pred_neutral,pred_up,status,completed_at,inference_scope,checkpoint_eval_id,parent_experiment_id,checkpoint_epoch,producer_worker_attempt_id) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NULLIF($10::bigint,-1),$11,$12,$13,$14,$15,$16,'completed',now(),'final',NULL,NULL,NULL,$17) ON CONFLICT (model_id,symbol,prediction_horizon,threshold_logret,window_size,label_rule_id,target_type,from_date,to_date) WHERE status='completed' AND inference_scope='final' DO UPDATE SET completed_epochs=EXCLUDED.completed_epochs,accuracy=EXCLUDED.accuracy,accept_model=EXCLUDED.accept_model,reject_reason=EXCLUDED.reject_reason,pred_down=EXCLUDED.pred_down,pred_neutral=EXCLUDED.pred_neutral,pred_up=EXCLUDED.pred_up,completed_at=now(),producer_worker_attempt_id=EXCLUDED.producer_worker_attempt_id RETURNING id;", request.modelId,snapshot.symbol,static_cast<long long>(snapshot.horizon),snapshot.threshold,static_cast<long long>(snapshot.window),DBIO::PgModelIO::kLookaheadHighLowFirstHitLabelRuleId,static_cast<int>(snapshot.targetType),DatePrefix(request.fromDate),DatePrefix(request.toDate),epochs,facts.accuracy,a.acceptModel,a.rejectReason,a.predFrac[0],a.predFrac[1],a.predFrac[2],request.workerAttemptId);
    if (row.size() != 1 || row.one_row()[0].is_null())
        throw std::runtime_error("managed_inference_result_not_returned");
    return row.one_row()[0].as<long long>();
}

void PersistProfitability(pqxx::work& transaction, long long resultId, const ManagedInferenceRequest& request, const Snapshot& snapshot, const InferenceEvaluationFacts::EvaluationFacts& facts)
{
    EA::InferenceProfitability::ObservationRequest observation;
    observation.provenance.experimentId = snapshot.binding.checkpoint ? snapshot.binding.parentExperimentId : snapshot.binding.finalExperimentId;
    observation.provenance.modelId = request.modelId;
    observation.provenance.inferenceEvalResultId = resultId;
    observation.provenance.scope = snapshot.binding.checkpoint ? EA::InferenceProfitability::Scope::checkpointInference : EA::InferenceProfitability::Scope::finalInference;
    observation.provenance.checkpointEvalId = snapshot.binding.checkpointEvalId;
    observation.provenance.inferenceStart = DatePrefix(request.fromDate);
    observation.provenance.inferenceEnd = DatePrefix(request.toDate);
    observation.statistics = facts.profitability;
    observation.sourceContentHash = facts.profitabilitySourceContentHash;
    (void)EA::InferenceProfitability::PersistObservationIdempotently(transaction, observation);
}
} // namespace

ManagedInferenceResult RunManagedInference(const ManagedInferenceRequest& request)
{
    LogManagedInferenceStage("managed_application_entry", request);
    ValidateRequest(request);
    const bool checkpoint = request.checkpointEvalId.has_value();
    if (!EA::SchedulerCore::RegisterSchedulerWorker({request.workerAttemptId, request.finalExperimentId, request.checkpointEvalId, checkpoint ? "checkpoint_infer" : "experiment", "infer"}))
        throw std::runtime_error("managed_inference_worker_registration_failed");
    if (checkpoint)
        EA::ExperimentScheduler::LogWorkerStarted(
            "CHECKPOINT_INFER_WORKER_STARTED", "checkpoint_infer",
            std::nullopt, request.modelId, request.checkpointEvalId);
    else
        EA::ExperimentScheduler::LogWorkerStarted(
            "INFERENCE_WORKER_STARTED", "infer", request.finalExperimentId,
            request.modelId);

    Snapshot snapshot;
    { // RR/RO admission must finish before market input or Tensor/LSTM work.
        pqxx::connection connection{request.database.lstmConnectionString};
        LogManagedInferenceStage("database_connection_established", request);
        pqxx::work read{connection};
        read.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
        LogManagedInferenceStage("repeatable_read_transaction_begun", request);
        LogManagedInferenceStage("detached_materialization_read_begun", request);
        snapshot = Materialize(read, request);
        LogManagedInferenceStage("detached_materialization_read_completed", request);
        read.commit();
        LogManagedInferenceStage("repeatable_read_transaction_committed", request);
    }
    std::optional<EA::EconomicCalendar::EconomicCalendarSnapshotIdentity> calendar;
    if (snapshot.model.identity.economicCalendarSnapshotId.has_value() !=
        snapshot.model.identity.economicCalendarSnapshotHash.has_value())
        throw std::runtime_error("economic_calendar_snapshot_identity_incomplete");
    if (snapshot.model.identity.economicCalendarSnapshotId)
        calendar = {*snapshot.model.identity.economicCalendarSnapshotId, *snapshot.model.identity.economicCalendarSnapshotHash};
    LogManagedInferenceStage("inference_input_tensor_preparation_begun", request);
    const auto input = PrepareInferenceInput({snapshot.symbol, request.fromDate, request.toDate, snapshot.model.featureWarmupScope, snapshot.model.donchian20Mode, snapshot.model.donchianLookback, calendar}, request.database);
    LogManagedInferenceStage("inference_input_tensor_preparation_completed", request);
    LogManagedInferenceStage("runtime_request_constructed", request);
    const RuntimeResult runtime = RunInferenceRuntime(
        {input.tensor, snapshot.model,
         {snapshot.window, snapshot.horizon, snapshot.threshold,
          input.logicalOutputStartIndex, snapshot.targetType, false},
         [&request](const char* stage)
         {
             LogManagedInferenceStage(stage, request);
         }});

    ManagedInferenceResult result;
    result.checkpoint = checkpoint;
    pqxx::connection connection{request.database.lstmConnectionString};
    pqxx::work write{connection};
    write.exec("SET TRANSACTION READ WRITE;");
    LogManagedInferenceStage("fresh_read_write_persistence_transaction_begun", request);
    RevalidateForWrite(write, request, snapshot);
    LogManagedInferenceStage("scheduler_identity_state_revalidation_completed", request);
    LogManagedInferenceStage("result_persistence_begun", request);
    result.inferenceEvalResultId = PersistResult(write, request, snapshot, runtime.evaluationFacts, result.idempotentExisting);
    LogManagedInferenceStage("result_persistence_completed", request);
    LogManagedInferenceStage("profitability_persistence_begun", request);
    PersistProfitability(write, result.inferenceEvalResultId, request, snapshot, runtime.evaluationFacts);
    LogManagedInferenceStage("profitability_persistence_completed", request);
    LogManagedInferenceStage("result_profitability_persistence_completed", request);
    write.commit();
    LogManagedInferenceStage("persistence_transaction_committed", request);
    LogManagedInferenceStage("managed_application_success_return", request);
    return result;
}
} // namespace EA::Inference
