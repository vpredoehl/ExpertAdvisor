#include "ControlledReplicationStudySpecificationService.hpp"

#include "PairedTrainingObjectiveEvaluationRepository.hpp"
#include "TrainingObjective.hpp"
#include "InferenceProfitability.hpp"
#include "InferenceProfitabilityRepository.hpp"

#include <pqxx/pqxx>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Study = EA::ControlledReplicationStudy;
namespace Objective = EA::TrainingObjective;

namespace
{

constexpr const char* kTreatment =
    "tg4_inner_break_any,tg4_source_tg3_structurally_eligible,tg4_source_tg3_confluent";
constexpr const char* kSourceCommit =
    "1111111111111111111111111111111111111111";
constexpr const char* kExecutableSha256 =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr const char* kRuntimeIdentity =
    "3333333333333333333333333333333333333333333333333333333333333333";

struct ArmIds
{
    long long experiment;
    long long model;
    long long trainAttempt;
    long long inferAttempt;
    long long inference;
    long long analysis;
};

std::string Environment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value) throw std::runtime_error(std::string("missing_") + name);
    return value;
}

void Matrix(pqxx::work& transaction, long long model, const std::string& name,
            const std::vector<double>& values, int rows = 1, int columns = 0)
{
    const int width = columns == 0 ? static_cast<int>(values.size()) : columns;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const int row = columns == 0 ? 0 : static_cast<int>(index) / width;
        const int column = columns == 0 ? static_cast<int>(index) :
            static_cast<int>(index) % width;
        transaction.exec(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,n_cols,value) "
            "VALUES($1,$2,$3,$4,$5,$6,$7);",
            pqxx::params{model, name, row, column, rows, width, values[index]});
    }
}

void Ascii(pqxx::work& transaction, long long model, const std::string& name,
           const std::string& value)
{
    std::vector<double> bytes;
    for (const unsigned char character : value) bytes.push_back(character);
    Matrix(transaction, model, name, bytes);
}

void InsertArm(pqxx::work& transaction, const ArmIds& ids,
               const std::string& symbol, unsigned int seed,
               const std::string& mask, double accuracy)
{
    const auto& objective = Objective::Legacy();
    const std::string canonical = Objective::CanonicalText(objective);
    const std::string hash = Objective::DeterministicHash(canonical);
    transaction.exec(
        "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,"
        "c_next_threshold,core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start,train_end,infer_start,infer_end,status,phase,last_model_id,"
        "resume_model_id,duplicate_nonce,donchian20_mode,donchian_lookback,"
        "feature_warmup_scope,feature_ablation_mask,resume_expand_input_width,"
        "git_commit,git_branch,git_dirty,build_config,compiler_version,schema_version,"
        "scheduler_version,binary_name,training_objective_id,training_objective_version,"
        "loss_definition_version,training_objective_canonical,training_objective_hash,"
        "auxiliary_loss_mode,auxiliary_loss_coefficient,regression_target_definition,"
        "regression_normalization_identity,robust_loss_definition,robust_loss_delta,"
        "target_clipping_definition,objective_normalization_identity) VALUES($1,$2,4,0.0008,120,25,80,20,"
        "'2020-01-01','2025-01-01','2025-01-02','2026-01-01','completed','done',$3,"
        "NULL,$5,'enabled',20,'full_history_warmup',$4,false,'fixture_commit',"
        "'fixture_branch',false,'Release','fixture_compiler','fixture_schema',"
        "'fixture_scheduler','LSTM_Release',$6,1,1,$7,$8,'disabled',0,NULL,NULL,NULL,NULL,"
        "'none','weighted_loss_sum_by_weight_sum_gradients__calculate_batch_return_by_example_count_v1');",
        pqxx::params{ids.experiment, symbol, ids.model, mask, ids.experiment,
                     objective.objectiveIdentifier, canonical, hash});

    transaction.exec(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "worker_attempt_id,experiment_id,worker_kind,lifecycle_phase,capacity_class,"
        "lifecycle_state,reserved_at,completed_at,exit_code,semantic_layout_version,"
        "model_input_width,semantic_worker_role,source_commit,executable_sha256,"
        "runtime_identity,canonical_manifest_path,canonical_executable_path) VALUES"
        "($1,$2,'experiment','train','train','completed','2025-01-01','2025-01-02',0,8,80,"
        "'train',$4,$5,$6,'fixture_manifest','fixture_train_executable'),"
        "($3,$2,'experiment','infer','infer','completed','2025-01-01','2025-01-02',0,8,80,"
        "'infer',$4,$5,$6,'fixture_infer_manifest','fixture_infer_executable');",
        pqxx::params{ids.trainAttempt, ids.experiment, ids.inferAttempt,
                     kSourceCommit, kExecutableSha256, kRuntimeIdentity});
    transaction.exec(
        "INSERT INTO model(model_id,experiment_id,parent_model_id,producer_worker_attempt_id) "
        "VALUES($1,$2,NULL,$3);",
        pqxx::params{ids.model, ids.experiment, ids.trainAttempt});

    Matrix(transaction, ids.model, "model_meta", {1, 80, 64});
    Matrix(transaction, ids.model, "param", {0}, 144, 256);
    Matrix(transaction, ids.model, "train_config_meta",
           {1, 4, 0.0008, 64, 1, 1, 1, 1, 1, 1, 80, 120, 25, 25});
    Matrix(transaction, ids.model, "target_meta", {0, 1, 0, 0, 0, 1});
    Matrix(transaction, ids.model, "optimizer_meta", {1, 1, 100, 0, 0});
    Matrix(transaction, ids.model, "model_input_semantics_meta", {1, 8});
    Ascii(transaction, ids.model, "train_symbol_meta", symbol);
    Ascii(transaction, ids.model, "train_range_meta", "2020-01-01|2025-01-01");
    Ascii(transaction, ids.model, "training_objective_canonical_meta", canonical);
    Ascii(transaction, ids.model, "training_objective_hash_meta", hash);
    Ascii(transaction, ids.model, "feature_warmup_scope_meta", "full_history_warmup");
    Ascii(transaction, ids.model, "donchian20_mode_meta", "enabled");
    Ascii(transaction, ids.model, "donchian_lookback_meta", "20");

    transaction.exec(
        "INSERT INTO inference_eval_result(id,model_id,status,inference_scope,"
        "checkpoint_eval_id,parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,completed_epochs,accuracy,"
        "accept_model,pred_down,pred_neutral,pred_up,producer_worker_attempt_id) VALUES"
        "($1,$2,'completed','final',NULL,NULL,$3,4,0.0008,64,1,0,'2025-01-02','2026-01-01',"
        "80,$4,true,0.2,0.3,0.5,$5);",
        pqxx::params{ids.inference, ids.model, symbol, accuracy, ids.inferAttempt});
    transaction.exec(
        "INSERT INTO experiment_analysis_result(analysis_id,experiment_id,model_id,"
        "analysis_scope,analysis_status,infer_accuracy,accept_accuracy,accept_rate,leader_score,"
        "pred_down_count,pred_neutral_count,pred_up_count,accept_count) VALUES"
        "($1,$2,$3,'final','completed',$4,0.75,0.70,0.63,20,30,50,40);",
        pqxx::params{ids.analysis, ids.experiment, ids.model, accuracy});
    EA::InferenceProfitability::ObservationRequest observation;
    observation.provenance.experimentId = ids.experiment;
    observation.provenance.modelId = ids.model;
    observation.provenance.inferenceEvalResultId = ids.inference;
    observation.provenance.scope = EA::InferenceProfitability::Scope::finalInference;
    observation.provenance.inferenceStart = "2025-01-02";
    observation.provenance.inferenceEnd = "2026-01-01";
    observation.statistics.predictionCount = 100;
    observation.statistics.actionableCount = 50;
    observation.statistics.winningActionableCount = 20;
    observation.statistics.losingActionableCount = 20;
    observation.statistics.grossPositiveTerminalHorizonLogReturnSum = 1.0;
    observation.statistics.grossNegativeTerminalHorizonLogReturnSum = -0.4;
    observation.statistics.aggregateTerminalHorizonLogReturnSum = 0.6;
    observation.metricDefinitionCanonical =
        EA::InferenceProfitability::kMetricDefinitionCanonical;
    observation.sourceContentHash = "fnv1a64:0000000000000000";
    (void)EA::InferenceProfitability::PersistObservationIdempotently(
        transaction, observation);
    (void)seed;
    transaction.exec("UPDATE experiment SET fresh_initialization_seed=$1 WHERE experiment_id=$2;",
                     pqxx::params{seed, ids.experiment});
}

Study::Specification Fixture()
{
    Study::Specification specification;
    specification.studyIdentifier = "postgres-study-fixture";
    specification.interventionField = "feature_ablation_mask";
    specification.controlSemantics = "empty_string";
    specification.controlValue = "";
    specification.treatmentValue = kTreatment;
    specification.replicationDimension = "fresh_initialization_seed";
    specification.allowedContextDimensions = {"symbol"};
    specification.requiredConfiguredIdentityFields = {
        "symbol", "prediction_horizon", "feature_ablation_mask",
        "fresh_initialization_seed", "training_objective_hash"};
    specification.requiredExecutionProvenanceFields = {
        "training_execution_identity", "inference_execution_identity",
        "producer_worker_attempt_id"};
    specification.aggregationPolicy = "unweighted_mean_of_context_family_means";
    specification.freezeTimestamp = "2026-09-28T10:00:00Z";
    specification.contexts = {
        {"CADCHF", {{"symbol", "CADCHF"}, {"prediction_horizon", "4"}},
         {{1001, 1002, 44}, {1003, 1004, 45}}},
        {"AUDCAD", {{"symbol", "AUDCAD"}, {"prediction_horizon", "4"}},
         {{1005, 1006, 44}, {1007, 1008, 45}}}};
    specification.identityHash = Study::IdentityHash(specification);
    return specification;
}

std::string ConnectionString()
{
    return "host=" + Environment("EA_STUDY_TEST_DB_HOST") +
        " port=" + Environment("EA_STUDY_TEST_DB_PORT") +
        " dbname=" + Environment("EA_STUDY_TEST_DB_NAME") + " user=pqxx";
}

void ExpectFailed(const std::string& path, const std::string& expected)
{
    std::ostringstream output;
    std::ostringstream errors;
    const int result = Study::RunCompareCommand(
        ConnectionString(), path, output, errors);
    assert(result == 3);
    assert(errors.str().find(expected) != std::string::npos);
}

} // namespace

int main()
{
    const std::string connectionString = ConnectionString();
    pqxx::connection connection{connectionString};
    const Study::Specification specification = Fixture();
    const std::filesystem::path artifact =
        std::filesystem::temp_directory_path() /
        "controlled_replication_study_postgres_fixture.txt";
    {
        std::ofstream file(artifact);
        file << Study::Render(specification);
    }

    {
        pqxx::work transaction{connection};
        InsertArm(transaction, {1001, 2001, 3001, 4001, 5001, 9001},
                  "CADCHF", 44, "", 0.70);
        InsertArm(transaction, {1002, 2002, 3002, 4002, 5002, 9002},
                  "CADCHF", 44, kTreatment, 0.72);
        InsertArm(transaction, {1003, 2003, 3003, 4003, 5003, 9003},
                  "CADCHF", 45, "", 0.68);
        InsertArm(transaction, {1004, 2004, 3004, 4004, 5004, 9004},
                  "CADCHF", 45, kTreatment, 0.69);
        InsertArm(transaction, {1005, 2005, 3005, 4005, 5005, 9005},
                  "AUDCAD", 44, "", 0.61);
        InsertArm(transaction, {1006, 2006, 3006, 4006, 5006, 9006},
                  "AUDCAD", 44, kTreatment, 0.65);
        InsertArm(transaction, {1007, 2007, 3007, 4007, 5007, 9007},
                  "AUDCAD", 45, "", 0.64);
        InsertArm(transaction, {1008, 2008, 3008, 4008, 5008, 9008},
                  "AUDCAD", 45, kTreatment, 0.63);
        transaction.commit();
    }

    {
        std::ostringstream output;
        std::ostringstream errors;
        assert(Study::RunCompareCommand(connectionString, artifact.string(),
                                         output, errors) == 0);
        const std::string rendered = output.str();
        assert(rendered.find("CONTROLLED_REPLICATION_STUDY_COMPARISON") !=
               std::string::npos);
        assert(rendered.find("family_count=2") != std::string::npos);
        assert(rendered.find("raw_pair_pooling=false") != std::string::npos);
        assert(rendered.find("unweighted_descriptive_family_means") !=
               std::string::npos);
        assert(rendered.find("subjective_winner=NONE") != std::string::npos);
        assert(rendered.find("read_only=true") != std::string::npos);
    }

    {
        pqxx::work transaction{connection};
        transaction.exec("UPDATE experiment SET feature_ablation_mask='wrong' WHERE experiment_id=1002;");
        transaction.commit();
        ExpectFailed(artifact.string(), "study_specification_intervention_mismatch");
        pqxx::work restore{connection};
        restore.exec("UPDATE experiment SET feature_ablation_mask=$1 WHERE experiment_id=1002;",
                     pqxx::params{kTreatment});
        restore.commit();
    }
    {
        pqxx::work transaction{connection};
        transaction.exec("UPDATE experiment SET fresh_initialization_seed=999 WHERE experiment_id=1004;");
        transaction.commit();
        ExpectFailed(artifact.string(), "study_specification_seed_mismatch");
        pqxx::work restore{connection};
        restore.exec("UPDATE experiment SET fresh_initialization_seed=45 WHERE experiment_id=1004;");
        restore.commit();
    }
    {
        pqxx::work transaction{connection};
        transaction.exec("UPDATE experiment SET symbol='USDCHF' WHERE experiment_id=1006;");
        transaction.commit();
        ExpectFailed(artifact.string(), "study_specification_context_identity_mismatch");
        pqxx::work restore{connection};
        restore.exec("UPDATE experiment SET symbol='AUDCAD' WHERE experiment_id=1006;");
        restore.commit();
    }
    {
        pqxx::work transaction{connection};
        transaction.exec("DELETE FROM experiment_analysis_result WHERE experiment_id=1008;");
        transaction.commit();
        std::ostringstream output;
        std::ostringstream errors;
        assert(Study::RunCompareCommand(connectionString, artifact.string(),
                                         output, errors) == 0);
        assert(output.str().find("cross_context_aggregation=suppressed") !=
               std::string::npos);
        pqxx::work restore{connection};
        restore.exec("INSERT INTO experiment_analysis_result(analysis_id,experiment_id,model_id,analysis_scope,analysis_status,infer_accuracy,accept_accuracy,accept_rate,leader_score,pred_down_count,pred_neutral_count,pred_up_count,accept_count) VALUES(9008,1008,2008,'final','completed',0.67,0.75,0.70,0.63,20,30,50,40);");
        restore.commit();
    }
    {
        pqxx::work transaction{connection};
        transaction.exec("UPDATE experiment_scheduler_worker_attempt SET runtime_identity='different_runtime' WHERE worker_attempt_id=3007;");
        transaction.commit();
        std::ostringstream output;
        std::ostringstream errors;
        assert(Study::RunCompareCommand(connectionString, artifact.string(),
                                         output, errors) == 0);
        assert(output.str().find("cross_context_aggregation=suppressed") !=
               std::string::npos);
        pqxx::work restore{connection};
        restore.exec("UPDATE experiment_scheduler_worker_attempt SET runtime_identity='fixture_runtime_identity' WHERE worker_attempt_id=3007;");
        restore.commit();
    }

    std::filesystem::remove(artifact);
    std::cout << "Controlled replication study PostgreSQL integration tests passed\n";
    return 0;
}
