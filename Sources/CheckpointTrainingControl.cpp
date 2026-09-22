#include "CheckpointTrainingControl.hpp"

#include "GlobalExperimentControl.hpp"
#include "ExperimentCurrentOperation.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <unistd.h>
#include <pqxx/pqxx>

namespace EA::CheckpointTrainingControl
{
namespace
{

std::string LstmDbConnectionString()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* database = std::getenv("LSTM_DB_NAME");
    return "hostaddr=" +
           std::string{
               host != nullptr && *host != '\0'
                   ? host
                   : "127.0.0.1"} +
           " gssencmode=disable user=pqxx dbname=" +
           std::string{
               database != nullptr && *database != '\0'
                   ? database
                   : "LSTM"};
}

bool SchedulerExperimentColumnExists(
    pqxx::work& w,
    const std::string& columnName)
{
    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' "
        "AND table_name = 'experiment' "
        "AND column_name = $1 LIMIT 1;",
        columnName);
    return !rows.empty();
}

bool SchedulerTableExists(
    pqxx::work& w,
    const std::string& tableName)
{
    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' "
        "AND table_name = $1 LIMIT 1;",
        tableName);
    return !rows.empty();
}

bool SchedulerColumnExists(
    pqxx::work& w,
    const std::string& tableName,
    const std::string& columnName)
{
    pqxx::result rows = w.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' "
        "AND table_name = $1 "
        "AND column_name = $2 LIMIT 1;",
        tableName,
        columnName);
    return !rows.empty();
}

std::optional<long long> ResolveSchedulerExperimentIdForCurrentProcess()
{
    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        pqxx::result rows = w.exec_params(
            "SELECT experiment_id "
            "FROM experiment "
            "WHERE worker_pid = $1 "
            "AND status = 'running' "
            "AND phase = 'train' "
            "ORDER BY updated_at DESC "
            "LIMIT 2;",
            static_cast<int>(::getpid()));
        w.commit();
        if (rows.size() == 1)
            return rows[0][0].as<long long>();
    }
    catch (const std::exception& e)
    {
        std::cerr << "SCHEDULER_PROGRESS_EXPERIMENT_LOOKUP_FAILED"
                  << ",pid=" << static_cast<int>(::getpid())
                  << ",error=" << e.what()
                  << std::endl;
    }
    return std::nullopt;
}


} // namespace

void UpdateSchedulerExperimentProgress(const std::optional<long long>& experimentId,
                                       int completedEpoch)
{
    const std::optional<long long> effectiveExperimentId =
        experimentId.has_value() ? experimentId : ResolveSchedulerExperimentIdForCurrentProcess();
    if (!effectiveExperimentId.has_value())
        return;
    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        w.exec("SET TRANSACTION READ WRITE;");
        w.exec_params(
            "UPDATE experiment "
            "SET current_epoch = $1, worker_pid = $2, "
            "current_operation = $3, updated_at = now() "
            "WHERE experiment_id = $4;",
            completedEpoch,
            static_cast<int>(::getpid()),
            std::string{EA::ExperimentLifecycle::kTrainOperation},
            *effectiveExperimentId);
        w.commit();
    }
    catch (const std::exception& e)
    {
        std::cerr << "SCHEDULER_PROGRESS_UPDATE_FAILED"
                  << ",experiment_id=" << *effectiveExperimentId
                  << ",epoch=" << completedEpoch
                  << ",error=" << e.what()
                  << std::endl;
    }
}


namespace
{

struct CheckpointInferConfig
{
    bool enabled = false;
    std::string symbol;
    int predictionHorizon = 0;
    std::optional<int> minEpoch;
    std::optional<int> interval;
};

std::optional<CheckpointInferConfig> LoadCheckpointInferConfig(const std::optional<long long>& experimentId,
                                                               int checkpointEpoch,
                                                               const std::optional<int>& launchCheckpointEvery)
{
    if (!experimentId.has_value())
        return std::nullopt;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        if (!SchedulerTableExists(w, "experiment") ||
            !SchedulerTableExists(w, "experiment_checkpoint_eval"))
        {
            w.commit();
            return std::nullopt;
        }

        const bool hasCheckpointInferEnabled =
            SchedulerExperimentColumnExists(w, "checkpoint_infer_enabled");
        const bool hasOpportunisticCheckpointInfer =
            SchedulerExperimentColumnExists(w, "opportunistic_checkpoint_infer");
        if (!hasCheckpointInferEnabled && !hasOpportunisticCheckpointInfer)
        {
            w.commit();
            return std::nullopt;
        }

        std::ostringstream sql;
        sql << "SELECT ";
        if (hasCheckpointInferEnabled && hasOpportunisticCheckpointInfer)
            sql << "(checkpoint_infer_enabled OR opportunistic_checkpoint_infer)";
        else if (hasCheckpointInferEnabled)
            sql << "checkpoint_infer_enabled";
        else
            sql << "opportunistic_checkpoint_infer";
        sql << ", symbol, prediction_horizon, checkpoint_infer_min_epoch, "
            << "COALESCE(checkpoint_infer_interval, NULLIF(checkpoint_interval, 0)) "
            << "FROM experiment WHERE experiment_id = $1;";

        pqxx::result rows = w.exec_params(sql.str(), *experimentId);
        w.commit();
        if (rows.empty())
            return std::nullopt;

        CheckpointInferConfig config;
        config.enabled = !rows[0][0].is_null() && rows[0][0].as<bool>();
        config.symbol = rows[0][1].as<std::string>();
        config.predictionHorizon = rows[0][2].as<int>();
        if (!rows[0][3].is_null())
            config.minEpoch = rows[0][3].as<int>();
        if (!rows[0][4].is_null())
            config.interval = rows[0][4].as<int>();
        else if (launchCheckpointEvery.has_value() && *launchCheckpointEvery > 0)
            config.interval = *launchCheckpointEvery;

        if (!config.enabled)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=disabled"
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }
        if (config.minEpoch.has_value() && checkpointEpoch < *config.minEpoch)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=below_min_epoch"
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }
        if (!config.interval.has_value() || *config.interval <= 0)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=missing_interval"
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }
        if (checkpointEpoch % *config.interval != 0)
        {
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *experimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " reason=interval_mismatch"
                      << " interval=" << *config.interval
                      << " symbol=" << config.symbol
                      << " horizon=" << config.predictionHorizon
                      << std::endl;
            return std::nullopt;
        }

        return config;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_INFER_SKIPPED"
                  << " experiment_id=" << *experimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " reason=config_load_failed"
                  << " error=" << e.what()
                  << std::endl;
        return std::nullopt;
    }
}


} // namespace

void QueueCheckpointInferenceIfEligible(const std::optional<long long>& schedulerExperimentId,
                                        const std::optional<int>& checkpointEvery,
                                        int checkpointEpoch,
                                        long long checkpointModelId)
{
    const std::optional<CheckpointInferConfig> config =
        LoadCheckpointInferConfig(schedulerExperimentId,
                                  checkpointEpoch,
                                  checkpointEvery);
    if (!config.has_value() || !schedulerExperimentId.has_value())
        return;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        w.exec("SET TRANSACTION READ WRITE;");
        EA::GlobalExperimentControl::AcquireCoordinationLock(w);
        if (!SchedulerTableExists(w, "experiment_checkpoint_eval"))
        {
            w.commit();
            return;
        }
        if (SchedulerExperimentColumnExists(w, "cancellation_request_id"))
        {
            const pqxx::result cancellation = w.exec(
                "SELECT cancellation_request_id FROM experiment "
                "WHERE experiment_id=$1;",
                pqxx::params{*schedulerExperimentId});
            if (!cancellation.empty() && !cancellation[0][0].is_null())
            {
                w.commit();
                std::cout << "CHECKPOINT_INFER_SKIPPED"
                          << " experiment_id=" << *schedulerExperimentId
                          << " checkpoint_epoch=" << checkpointEpoch
                          << " checkpoint_model_id=" << checkpointModelId
                          << " reason=global_cancellation"
                          << " symbol=" << config->symbol
                          << " horizon=" << config->predictionHorizon
                          << std::endl;
                return;
            }
        }

        pqxx::result modelRows = w.exec_params(
            "SELECT 1 FROM model WHERE model_id = $1 LIMIT 1;",
            checkpointModelId);
        if (modelRows.empty())
        {
            w.commit();
            std::cout << "CHECKPOINT_INFER_SKIPPED"
                      << " experiment_id=" << *schedulerExperimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " checkpoint_model_id=" << checkpointModelId
                      << " reason=model_not_found"
                      << " symbol=" << config->symbol
                      << " horizon=" << config->predictionHorizon
                      << std::endl;
            return;
        }

        std::cout << "CHECKPOINT_INFER_ELIGIBLE"
                  << " experiment_id=" << *schedulerExperimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " checkpoint_model_id=" << checkpointModelId
                  << " symbol=" << config->symbol
                  << " horizon=" << config->predictionHorizon
                  << std::endl;

        const bool hasParentExperimentId =
            SchedulerColumnExists(w, "experiment_checkpoint_eval", "parent_experiment_id");
        const bool hasSymbol =
            SchedulerColumnExists(w, "experiment_checkpoint_eval", "symbol");
        const bool hasPredictionHorizon =
            SchedulerColumnExists(w, "experiment_checkpoint_eval", "prediction_horizon");

        std::ostringstream sql;
        sql << "INSERT INTO experiment_checkpoint_eval (experiment_id";
        if (hasParentExperimentId)
            sql << ", parent_experiment_id";
        sql << ", checkpoint_epoch, checkpoint_model_id";
        if (hasSymbol)
            sql << ", symbol";
        if (hasPredictionHorizon)
            sql << ", prediction_horizon";
        sql << ") VALUES ($1";
        int param = 2;
        if (hasParentExperimentId)
            sql << ", $" << param++;
        sql << ", $" << param++ << ", $" << param++;
        if (hasSymbol)
            sql << ", $" << param++;
        if (hasPredictionHorizon)
            sql << ", $" << param++;
        sql << ") ON CONFLICT ";
        if (hasParentExperimentId)
            sql << "(parent_experiment_id, checkpoint_model_id, checkpoint_epoch) ";
        else
            sql << "(experiment_id, checkpoint_epoch, checkpoint_model_id) ";
        sql << "DO NOTHING RETURNING checkpoint_eval_id;";

        pqxx::result inserted;
        if (hasParentExperimentId && hasSymbol && hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol,
                                     config->predictionHorizon);
        }
        else if (hasParentExperimentId && hasSymbol)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol);
        }
        else if (hasParentExperimentId && hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->predictionHorizon);
        }
        else if (hasParentExperimentId)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId);
        }
        else if (hasSymbol && hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol,
                                     config->predictionHorizon);
        }
        else if (hasSymbol)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->symbol);
        }
        else if (hasPredictionHorizon)
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId,
                                     config->predictionHorizon);
        }
        else
        {
            inserted = w.exec_params(sql.str(),
                                     *schedulerExperimentId,
                                     checkpointEpoch,
                                     checkpointModelId);
        }
        w.commit();

        if (inserted.empty())
        {
            std::cout << "CHECKPOINT_INFER_DUPLICATE"
                      << " experiment_id=" << *schedulerExperimentId
                      << " checkpoint_epoch=" << checkpointEpoch
                      << " checkpoint_model_id=" << checkpointModelId
                      << " symbol=" << config->symbol
                      << " horizon=" << config->predictionHorizon
                      << std::endl;
            return;
        }

        std::cout << "CHECKPOINT_INFER_QUEUED"
                  << " experiment_id=" << *schedulerExperimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " checkpoint_model_id=" << checkpointModelId
                  << " symbol=" << config->symbol
                  << " horizon=" << config->predictionHorizon
                  << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_INFER_SKIPPED"
                  << " experiment_id=" << *schedulerExperimentId
                  << " checkpoint_epoch=" << checkpointEpoch
                  << " checkpoint_model_id=" << checkpointModelId
                  << " reason=queue_failed"
                  << " error=" << e.what()
                  << std::endl;
    }
}


std::optional<CheckpointStopConfig> LoadCheckpointStopConfig(const std::optional<long long>& experimentId,
                                                             int checkpointEpoch,
                                                             int targetEpochs,
                                                             const std::optional<int>& checkpointEvery)
{
    if (!experimentId.has_value() || !checkpointEvery.has_value() || *checkpointEvery <= 0)
        return std::nullopt;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        EA::GlobalExperimentControl::AcquireCoordinationLock(w);
        if (!SchedulerExperimentColumnExists(w, "stop_after_checkpoint_epoch"))
        {
            w.commit();
            return std::nullopt;
        }

        const bool hasCancellationControl =
            SchedulerExperimentColumnExists(w, "cancellation_request_id") &&
            SchedulerExperimentColumnExists(
                w, "last_checkpoint_stop_decision_epoch");
        pqxx::result rows = hasCancellationControl
            ? w.exec_params(
                  "SELECT stop_after_checkpoint_epoch,cancellation_request_id "
                  "FROM experiment WHERE experiment_id = $1 FOR UPDATE;",
                  *experimentId)
            : w.exec_params(
                  "SELECT stop_after_checkpoint_epoch,NULL::bigint "
                  "FROM experiment WHERE experiment_id = $1 FOR UPDATE;",
                  *experimentId);
        if (hasCancellationControl)
            w.exec_params(
                "UPDATE experiment "
                "SET last_checkpoint_stop_decision_epoch=$1,updated_at=now() "
                "WHERE experiment_id=$2;",
                checkpointEpoch,
                *experimentId);
        w.commit();
        if (rows.empty() || rows[0][0].is_null())
            return std::nullopt;

        const int requestedEpoch = rows[0][0].as<int>();
        if (requestedEpoch <= 0)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=invalid_requested_epoch"
                      << " experiment_id=" << *experimentId
                      << " requested_epoch=" << requestedEpoch
                      << std::endl;
            return std::nullopt;
        }
        if (requestedEpoch >= targetEpochs)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=requested_epoch_not_before_target"
                      << " experiment_id=" << *experimentId
                      << " requested_epoch=" << requestedEpoch
                      << " target_epochs=" << targetEpochs
                      << std::endl;
            return std::nullopt;
        }

        const int interval = *checkpointEvery;
        const int effectiveEpoch = ((requestedEpoch + interval - 1) / interval) * interval;
        if (effectiveEpoch >= targetEpochs)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=effective_epoch_not_before_target"
                      << " experiment_id=" << *experimentId
                      << " requested_epoch=" << requestedEpoch
                      << " effective_epoch=" << effectiveEpoch
                      << " target_epochs=" << targetEpochs
                      << std::endl;
            return std::nullopt;
        }
        if (checkpointEpoch < effectiveEpoch)
            return std::nullopt;

        std::cout << "CHECKPOINT_STOP_REQUESTED"
                  << " experiment_id=" << *experimentId
                  << " requested_epoch=" << requestedEpoch
                  << " effective_epoch=" << effectiveEpoch
                  << " checkpoint_epoch=" << checkpointEpoch
                  << std::endl;
        std::cout << "CHECKPOINT_STOP_LIVE_REQUEST_DETECTED"
                  << " experiment_id=" << *experimentId
                  << " requested_epoch=" << requestedEpoch
                  << " effective_epoch=" << effectiveEpoch
                  << " checkpoint_epoch=" << checkpointEpoch
                  << std::endl;
        return CheckpointStopConfig{
            requestedEpoch,
            effectiveEpoch,
            !rows[0][1].is_null()};
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_STOP_IGNORED"
                  << " reason=config_load_failed"
                  << " experiment_id=" << *experimentId
                  << " error=" << e.what()
                  << std::endl;
        return std::nullopt;
    }
}

bool RecordCheckpointStopReached(const std::optional<long long>& experimentId,
                                 const std::optional<long long>& workerAttemptId,
                                 int epoch,
                                 long long modelId)
{
    if (!experimentId.has_value() || !workerAttemptId.has_value())
        return false;

    try
    {
        pqxx::connection c{LstmDbConnectionString()};
        pqxx::work w{c};
        w.exec("SET TRANSACTION READ WRITE;");
        EA::GlobalExperimentControl::AcquireCoordinationLock(w);
        if (!SchedulerExperimentColumnExists(w, "stopped_at_checkpoint_epoch") ||
            !SchedulerExperimentColumnExists(w, "stopped_at_checkpoint_model_id") ||
            !SchedulerExperimentColumnExists(w, "cancellation_request_id"))
        {
            w.commit();
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=migration_required"
                      << " experiment_id=" << *experimentId
                      << std::endl;
            return false;
        }

        const EA::GlobalExperimentControl::CheckpointStopRecordResult result =
            EA::GlobalExperimentControl::RecordCheckpointStopReached(
                w, experimentId, *workerAttemptId, epoch, modelId);
        w.commit();
        if (!result.recorded)
        {
            std::cout << "CHECKPOINT_STOP_IGNORED"
                      << " reason=" << result.detail
                      << " experiment_id=" << *experimentId
                      << std::endl;
            return false;
        }

        std::cout << "CHECKPOINT_STOP_REACHED"
                  << " experiment_id=" << *experimentId
                  << " epoch=" << epoch
                  << " model_id=" << modelId
                  << std::endl;
        if (result.cancellationRequested)
            std::cout << "GLOBAL_CANCELLATION_CHECKPOINT_REACHED"
                      << " request_id=" << *result.cancellationRequestId
                      << " experiment_id=" << *experimentId
                      << " epoch=" << epoch
                      << " model_id=" << modelId
                      << " infer_before_cancel="
                      << (result.inferenceRequested ? "1" : "0")
                      << " detail=" << result.detail
                      << std::endl;
        else
            std::cout << "CHECKPOINT_STOP_ADVANCE_TO_INFER"
                      << " experiment_id=" << *experimentId
                      << " model_id=" << modelId
                      << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "CHECKPOINT_STOP_IGNORED"
                  << " reason=record_failed"
                  << " experiment_id=" << *experimentId
                  << " epoch=" << epoch
                  << " model_id=" << modelId
                  << " error=" << e.what()
                  << std::endl;
        return false;
    }
}


} // namespace EA::CheckpointTrainingControl
