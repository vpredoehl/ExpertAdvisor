#include "InferenceProfitabilityRepository.hpp"
#include "TrainingObjective.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace Objective = EA::TrainingObjective;
namespace Profitability = EA::InferenceProfitability;

namespace
{

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        throw std::runtime_error(std::string{"missing_environment:"} + name);
    return value;
}

void InsertMatrixRow(pqxx::transaction_base& transaction,
                     long long modelId,
                     const std::string& name,
                     const std::vector<double>& values)
{
    for (std::size_t index = 0; index < values.size(); ++index)
        transaction.exec(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,"
            "n_cols,value) VALUES($1,$2,0,$3,1,$4,$5);",
            pqxx::params{modelId, name, static_cast<int>(index),
                         static_cast<int>(values.size()), values[index]});
}

void InsertAscii(pqxx::transaction_base& transaction,
                 long long modelId,
                 const std::string& name,
                 const std::string& value)
{
    std::vector<double> encoded;
    encoded.reserve(value.size());
    for (const unsigned char character : value)
        encoded.push_back(static_cast<double>(character));
    InsertMatrixRow(transaction, modelId, name, encoded);
}

void InsertAnchor(pqxx::transaction_base& transaction,
                  long long experimentId,
                  const std::string& mask)
{
    const auto objective = Objective::Legacy();
    const std::string canonical = Objective::CanonicalText(objective);
    const std::string hash = Objective::Identity(objective);
    transaction.exec(
        "INSERT INTO experiment("
        "experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start,train_end,infer_start,infer_end,status,phase,"
        "duplicate_nonce,donchian20_mode,donchian_lookback,"
        "feature_warmup_scope,feature_ablation_mask,"
        "resume_expand_input_width,git_commit,git_branch,git_dirty,"
        "build_config,compiler_version,schema_version,scheduler_version,"
        "binary_name,invocation_mode,training_objective_id,"
        "training_objective_version,loss_definition_version,"
        "training_objective_canonical,training_objective_hash,"
        "auxiliary_loss_mode,auxiliary_loss_coefficient,"
        "target_clipping_definition,objective_normalization_identity,"
        "model_input_width,model_input_semantic_layout_version,"
        "economic_calendar_snapshot_id,economic_calendar_snapshot_hash,"
        "scheduler_priority) VALUES("
        "$1,'eurusdrmp',4,0.0008,119.75,25,80,20,"
        "'2010-01-01','2025-01-01','2025-01-01','2026-01-01',"
        "'pending','train',$1,'enabled',20,'full_history_warmup',$2,false,"
        "'fixture-commit','phase16-fixture',true,'Release','fixture-compiler',"
        "'fixture-schema','fixture-scheduler','LSTM_Release',"
        "'phase16_anchor_fixture',$3,1,1,$4,$5,'disabled',0,'none',"
        "'weighted_loss_sum_by_weight_sum_gradients__calculate_batch_return_"
        "by_example_count_v1',77,7,1,'fnv1a64:67610f94f5c8e7cc','high');",
        pqxx::params{experimentId, mask, objective.objectiveIdentifier,
                     canonical, hash});
}

void CompleteExperiment(pqxx::transaction_base& transaction,
                        long long experimentId,
                        double aggregateReturn)
{
    const pqxx::row experiment = transaction.exec(
        "SELECT symbol,prediction_horizon,c_next_threshold,target_epochs,"
        "core_lr_mult,head_lr_mult,train_start::date::text,"
        "train_end::date::text,model_input_width,"
        "model_input_semantic_layout_version,training_objective_canonical,"
        "training_objective_hash,economic_calendar_snapshot_id,"
        "economic_calendar_snapshot_hash FROM experiment "
        "WHERE experiment_id=$1;",
        pqxx::params{experimentId}).one_row();
    const std::string symbol = experiment["symbol"].as<std::string>();
    const int horizon = experiment["prediction_horizon"].as<int>();
    const double threshold = experiment["c_next_threshold"].as<double>();
    const int epochs = experiment["target_epochs"].as<int>();
    const double core = experiment["core_lr_mult"].as<double>();
    const double head = experiment["head_lr_mult"].as<double>();
    const int width = experiment["model_input_width"].as<int>();
    const int layout =
        experiment["model_input_semantic_layout_version"].as<int>();
    const std::string objectiveCanonical =
        experiment["training_objective_canonical"].as<std::string>();
    const std::string objectiveHash =
        experiment["training_objective_hash"].as<std::string>();
    const long long snapshotId =
        experiment["economic_calendar_snapshot_id"].as<long long>();
    const std::string snapshotHash =
        experiment["economic_calendar_snapshot_hash"].as<std::string>();

    const long long modelId = transaction.exec(
        "INSERT INTO model(name,comment,experiment_id,"
        "economic_calendar_snapshot_id,economic_calendar_snapshot_hash) "
        "VALUES($1,'final fixture model',$2,$3,$4) RETURNING model_id;",
        pqxx::params{"phase16-" + std::to_string(experimentId), experimentId,
                     snapshotId, snapshotHash}).one_row()[0].as<long long>();
    InsertMatrixRow(transaction, modelId, "train_config_meta", {
        1.0, static_cast<double>(horizon),
        static_cast<double>(static_cast<float>(threshold)), 64.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        static_cast<double>(epochs), core, head, head});
    InsertMatrixRow(transaction, modelId, "target_meta",
                    {0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    InsertMatrixRow(transaction, modelId, "model_meta",
                    {1.0, static_cast<double>(width), 64.0});
    InsertMatrixRow(transaction, modelId, "model_input_semantics_meta",
                    {1.0, static_cast<double>(layout)});
    InsertMatrixRow(transaction, modelId, "optimizer_meta",
                    {1.0, 1.0, 100.0, 0.0, 0.0});
    InsertAscii(transaction, modelId, "train_symbol_meta", symbol);
    InsertAscii(transaction, modelId, "train_range_meta",
                experiment["train_start"].as<std::string>() + "|" +
                experiment["train_end"].as<std::string>());
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,"
        "n_cols,value) VALUES($1,'param',0,0,$2,256,0);",
        pqxx::params{modelId, width + 64});
    InsertAscii(transaction, modelId, "training_objective_canonical_meta",
                objectiveCanonical);
    InsertAscii(transaction, modelId, "training_objective_hash_meta",
                objectiveHash);

    const long long inferenceId = transaction.exec(
        "INSERT INTO inference_eval_result("
        "model_id,status,inference_scope,symbol,prediction_horizon,"
        "threshold_logret,window_size,label_rule_id,target_type,from_date,"
        "to_date,completed_epochs,accuracy,accept_model,pred_down,"
        "pred_neutral,pred_up) VALUES($1,'completed','final',$2,$3,$4,64,"
        "1,0,'2025-01-01','2026-01-01',$5,0.60,true,0.25,0.35,0.40) "
        "RETURNING id;",
        pqxx::params{modelId, symbol, horizon,
                     static_cast<double>(static_cast<float>(threshold)),
                     epochs}).one_row()[0].as<long long>();
    transaction.exec(
        "INSERT INTO experiment_analysis_result("
        "experiment_id,model_id,symbol,prediction_horizon,target_epochs,"
        "completed_epochs,analysis_scope,analysis_status,infer_accuracy,"
        "accept_accuracy,accept_rate,leader_score,pred_down_count,"
        "pred_neutral_count,pred_up_count,accept_count) VALUES("
        "$1,$2,$3,$4,$5,$5,'final','completed',0.60,0.65,0.65,0.54,"
        "25,35,40,65);",
        pqxx::params{experimentId, modelId, symbol, horizon, epochs});

    Profitability::Accumulator accumulator;
    accumulator.Observe(Profitability::kNeutralClass, 100.0f, 100.0f);
    accumulator.Observe(
        Profitability::kUpClass, 100.0f,
        static_cast<float>(100.0 * std::exp(aggregateReturn)));
    Profitability::ObservationRequest request;
    request.provenance.experimentId = experimentId;
    request.provenance.modelId = modelId;
    request.provenance.inferenceEvalResultId = inferenceId;
    request.provenance.scope = Profitability::Scope::finalInference;
    request.provenance.inferenceStart = "2025-01-01";
    request.provenance.inferenceEnd = "2026-01-01";
    request.statistics = accumulator.statistics();
    request.sourceContentHash = accumulator.SourceContentHash();
    (void)Profitability::PersistObservationIdempotently(transaction, request);

    transaction.exec(
        "UPDATE experiment SET status='completed',phase='done',"
        "last_model_id=$2,current_epoch=target_epochs,current_operation=NULL,"
        "completed_at=clock_timestamp(),updated_at=clock_timestamp() "
        "WHERE experiment_id=$1;",
        pqxx::params{experimentId, modelId});
}

} // namespace

int main(int argc, char* argv[])
{
    const std::string database = RequiredEnvironment("LSTM_DB_NAME");
    const std::string host = RequiredEnvironment("LSTM_DB_HOST");
    const std::string port = RequiredEnvironment("LSTM_DB_PORT");
    const std::string user = RequiredEnvironment("LSTM_DB_USER");
    pqxx::connection connection{
        "hostaddr=" + host + " port=" + port +
        " user=" + user + " dbname=" + database};
    pqxx::work transaction{connection};
    if (argc == 2 && std::string{argv[1]} == "seed-anchor")
    {
        transaction.exec(
            "INSERT INTO economic_calendar_snapshot("
            "economic_calendar_snapshot_id,finalized_at,snapshot_state,"
            "content_hash,created_by,canonical_event_count,"
            "selected_consensus_count,release_actual_count,"
            "proven_first_release_actual_count,provenance_unavailable_count,"
            "ambiguous_first_release_count,source_family_counts) VALUES("
            "1,clock_timestamp(),'finalized','fnv1a64:67610f94f5c8e7cc',"
            "'phase16-fixture',0,0,0,0,0,0,'{}'::jsonb);");
        InsertAnchor(transaction, 624, "");
        InsertAnchor(transaction, 625,
            "causal_first_release_surprise_available,"
            "causal_first_release_surprise");
        CompleteExperiment(transaction, 624, 0.07);
        CompleteExperiment(transaction, 625, 0.04);
        transaction.exec(
            "SELECT setval('experiment_experiment_id_seq',625,true);");
    }
    else if (argc == 4 && std::string{argv[1]} == "complete-pair")
    {
        CompleteExperiment(transaction, std::stoll(argv[2]), 0.06);
        CompleteExperiment(transaction, std::stoll(argv[3]), 0.03);
    }
    else
    {
        throw std::invalid_argument(
            "usage: fixture seed-anchor | complete-pair CONTROL TREATMENT");
    }
    transaction.commit();
    std::cout << "Corrected causal-surprise fixture updated\n";
    return 0;
}
