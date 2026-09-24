#include "ExperimentReplicationMaterialization.hpp"
#include "ExperimentReplicationMaterializationRepository.hpp"

#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pqxx/pqxx>

namespace Materialization = EA::ExperimentReplicationMaterialization;
namespace Planning = EA::ExperimentReplicationPlanning;
namespace Objective = EA::TrainingObjective;

namespace
{

std::string Environment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value) throw std::runtime_error(name);
    return value;
}

Planning::ProposedExperimentSpecification Specification(
    long long source, unsigned int seed)
{
    Planning::ProposedExperimentSpecification result;
    result.experimentId = source;
    result.authoritativeSourceExperimentId = source;
    result.freshInitializationSeed = seed;
    return result;
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

void InsertAuthoritativeSource(pqxx::transaction_base& transaction,
                               long long experimentId,
                               long long modelId,
                               const std::string& mask)
{
    const auto objective = Objective::Legacy();
    const std::string canonical = Objective::CanonicalText(objective);
    const std::string hash = Objective::DeterministicHash(canonical);
    transaction.exec(
        "INSERT INTO experiment("
        "experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start,train_end,infer_start,infer_end,status,phase,last_model_id,"
        "resume_model_id,duplicate_nonce,donchian20_mode,donchian_lookback,"
        "feature_warmup_scope,feature_ablation_mask,fresh_initialization_seed,"
        "resume_expand_input_width,training_objective_id,"
        "training_objective_version,loss_definition_version,"
        "training_objective_canonical,training_objective_hash,"
        "auxiliary_loss_mode,auxiliary_loss_coefficient,"
        "target_clipping_definition,objective_normalization_identity,"
        "model_input_width,model_input_semantic_layout_version,"
        "git_commit,git_branch,git_dirty,build_config,compiler_version,"
        "schema_version,scheduler_version,binary_name) VALUES("
        "$1,'synthetic_materializer',4,0.0008,120,25,80,20,"
        "'2010-01-01','2025-01-01','2025-01-01','2026-01-01',"
        "'completed','done',$2,NULL,0,'enabled',20,'full_history_warmup',$3,"
        "43,false,$4,1,1,$5,$6,'disabled',0,'none','fixture_v1',80,8,"
        "'synthetic_commit','phase6',false,'Debug','AppleClang-test','test',"
        "'scheduler-test','LSTM_Debug');",
        pqxx::params{experimentId, modelId, mask,
                     objective.objectiveIdentifier, canonical, hash});
    transaction.exec(
        "INSERT INTO model(model_id,experiment_id,producer_worker_attempt_id) "
        "VALUES($1,$2,$3);",
        pqxx::params{modelId, experimentId, experimentId * 10 + 1});
    transaction.exec(
        "INSERT INTO experiment_scheduler_worker_attempt("
        "worker_attempt_id,experiment_id,worker_kind,lifecycle_phase,"
        "capacity_class,lifecycle_state,reserved_at,completed_at,exit_code,"
        "semantic_layout_version,model_input_width,semantic_worker_role,"
        "source_commit,executable_sha256,runtime_identity,"
        "canonical_executable_path) VALUES("
        "$1,$2,'experiment','train','train','completed','2025-01-01',"
        "'2025-02-01',0,8,80,'train','synthetic_commit',"
        "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',"
        "'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',"
        "'/synthetic/LSTM_Debug');",
        pqxx::params{experimentId * 10 + 1, experimentId});
    InsertMatrixRow(transaction, modelId, "train_config_meta",
                    {1, 4, 0.0008, 64, 1, 1, 1, 1, 1, 1, 80, 120, 25, 25});
    InsertMatrixRow(transaction, modelId, "target_meta", {0, 1, 0, 0, 0, 1});
    InsertMatrixRow(transaction, modelId, "model_meta", {1, 80, 64});
    InsertMatrixRow(transaction, modelId, "model_input_semantics_meta", {1, 8});
    InsertMatrixRow(transaction, modelId, "optimizer_meta", {1, 1, 100, 0, 0});
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,n_cols,"
        "value) VALUES($1,'param',0,0,144,256,0);",
        pqxx::params{modelId});
    InsertAscii(transaction, modelId, "train_symbol_meta",
                "synthetic_materializer");
    InsertAscii(transaction, modelId, "train_range_meta",
                "2010-01-01|2025-01-01");
    InsertAscii(transaction, modelId, "training_objective_canonical_meta",
                canonical);
    InsertAscii(transaction, modelId, "training_objective_hash_meta", hash);
}

long long CountSeed(pqxx::connection& connection, unsigned int seed)
{
    pqxx::read_transaction read{connection};
    return read.exec(
        "SELECT count(*) FROM experiment WHERE symbol='synthetic_materializer' "
        "AND fresh_initialization_seed=$1;", pqxx::params{seed})
        .one_row()[0].as<long long>();
}

int RunCommand(const std::string& connectionString,
               const std::vector<unsigned int>& seeds,
               std::string& output,
               std::string& errors)
{
    Materialization::MaterializationCommand command;
    command.sourceExperimentIds = {100, 101};
    command.requestedSeeds = seeds;
    std::ostringstream out;
    std::ostringstream err;
    const int result = Materialization::RunMaterializationCommand(
        connectionString, command, out, err);
    output = out.str();
    errors = err.str();
    return result;
}

} // namespace

int main()
{
    const std::string connectionString =
        "host=" + Environment("EA_REPLICATION_MATERIALIZER_DB_HOST") +
        " port=" + Environment("EA_REPLICATION_MATERIALIZER_DB_PORT") +
        " user=pqxx dbname=" +
        Environment("EA_REPLICATION_MATERIALIZER_DB_NAME");
    pqxx::connection connection{connectionString};

    std::vector<long long> createdIds;
    {
        pqxx::work transaction{connection};
        for (const unsigned int seed : {44U, 45U, 46U})
            for (const long long source : {10LL, 11LL})
                createdIds.push_back(
                    Materialization::InsertFreshPausedReplicationExperiment(
                        transaction, Specification(source, seed)));
        transaction.commit();
    }
    assert((createdIds == std::vector<long long>{12, 13, 14, 15, 16, 17}));
    {
        pqxx::read_transaction read{connection};
        const pqxx::row row = read.exec(
            "SELECT symbol,prediction_horizon,c_next_threshold,core_lr_mult,"
            "head_lr_mult,target_epochs,checkpoint_interval,train_start,"
            "train_end,infer_start,infer_end,feature_ablation_mask,"
            "fresh_initialization_seed,status,phase,invocation_mode,"
            "duplicate_nonce,resume_model_id,last_model_id,current_epoch,"
            "worker_pid,started_at,completed_at,parent_experiment_id,"
            "continuation_source_experiment_id,scheduler_priority,"
            "training_objective_hash,model_input_width,"
            "model_input_semantic_layout_version,checkpoint_policy_revision,"
            "train_log_path,infer_log_path,analysis_log_path,exit_code,"
            "error_message,current_operation,worker_started_at,"
            "active_scheduler_worker_attempt_id,"
            "operator_forced_final_inference_rerun_requested,resume_requested,"
            "scheduler_resume_origin,stop_after_checkpoint_epoch,"
            "stopped_at_checkpoint_epoch,stopped_at_checkpoint_model_id,"
            "continuation_source_model_id,continuation_source_epoch,"
            "continuation_decision_id,continuation_generation,git_commit,"
            "git_branch,git_dirty,build_config,compiler_version,schema_version,"
            "scheduler_version,binary_name "
            "FROM experiment WHERE experiment_id=$1;",
            pqxx::params{createdIds.front()}).one_row();
        assert(row["symbol"].as<std::string>() == "fixture");
        assert(row["prediction_horizon"].as<int>() == 4);
        assert(row["fresh_initialization_seed"].as<unsigned int>() == 44);
        assert(row["feature_ablation_mask"].as<std::string>() == "mask_a");
        assert(row["status"].as<std::string>() == "paused");
        assert(row["phase"].as<std::string>() == "train");
        assert(row["invocation_mode"].as<std::string>() ==
               "controlled_replication_materialization");
        // fresh_initialization_seed is intentionally not part of the legacy
        // production unique index. The serialized materializer assigns an
        // administrative nonce while scientific equivalence ignores it.
        assert(row["duplicate_nonce"].as<long long>() == 1);
        assert(row["resume_model_id"].is_null());
        assert(row["last_model_id"].is_null());
        assert(row["current_epoch"].is_null());
        assert(row["worker_pid"].is_null());
        assert(row["started_at"].is_null());
        assert(row["completed_at"].is_null());
        assert(row["parent_experiment_id"].is_null());
        assert(row["continuation_source_experiment_id"].is_null());
        assert(row["scheduler_priority"].as<std::string>() == "normal");
        assert(row["training_objective_hash"].as<std::string>() ==
               "fnv1a64:0000000000000001");
        assert(row["model_input_width"].as<int>() == 80);
        assert(row["model_input_semantic_layout_version"].as<int>() == 8);
        assert(row["checkpoint_policy_revision"].as<long long>() == 7);
        assert(row["train_log_path"].is_null());
        assert(row["infer_log_path"].is_null());
        assert(row["analysis_log_path"].is_null());
        assert(row["exit_code"].is_null());
        assert(row["error_message"].is_null());
        assert(row["current_operation"].is_null());
        assert(row["worker_started_at"].is_null());
        assert(row["active_scheduler_worker_attempt_id"].is_null());
        assert(!row["operator_forced_final_inference_rerun_requested"].as<bool>());
        assert(!row["resume_requested"].as<bool>());
        assert(row["scheduler_resume_origin"].as<std::string>() == "none");
        assert(row["stop_after_checkpoint_epoch"].is_null());
        assert(row["stopped_at_checkpoint_epoch"].is_null());
        assert(row["stopped_at_checkpoint_model_id"].is_null());
        assert(row["continuation_source_model_id"].is_null());
        assert(row["continuation_source_epoch"].is_null());
        assert(row["continuation_decision_id"].is_null());
        assert(row["continuation_generation"].as<int>() == 1);
        assert(row["git_commit"].is_null());
        assert(row["git_branch"].is_null());
        assert(row["git_dirty"].is_null());
        assert(row["build_config"].is_null());
        assert(row["compiler_version"].is_null());
        assert(row["schema_version"].is_null());
        assert(row["scheduler_version"].is_null());
        assert(row["binary_name"].is_null());
        const pqxx::result created = read.exec(
            "SELECT experiment_id,fresh_initialization_seed,"
            "feature_ablation_mask,duplicate_nonce FROM experiment "
            "WHERE experiment_id >= 12 ORDER BY experiment_id;");
        assert(created.size() == 6);
        for (int index = 0; index < created.size(); ++index)
        {
            assert(created[index]["experiment_id"].as<long long>() ==
                   12 + index);
            assert(created[index]["fresh_initialization_seed"].as<unsigned int>() ==
                   44U + static_cast<unsigned int>(index / 2));
            assert(created[index]["feature_ablation_mask"].as<std::string>() ==
                   (index % 2 == 0 ? "mask_a" : "mask_b"));
            assert(created[index]["duplicate_nonce"].as<long long>() ==
                   1 + index);
        }
    }

    // A later failure leaves the earlier insert uncommitted when the owning
    // transaction unwinds, which is the materializer's whole-wave behavior.
    try
    {
        pqxx::work transaction{connection};
        (void)Materialization::InsertFreshPausedReplicationExperiment(
            transaction, Specification(10, 47));
        (void)Materialization::InsertFreshPausedReplicationExperiment(
            transaction, Specification(999999, 47));
        transaction.commit();
        assert(false);
    }
    catch (const std::runtime_error&)
    {
    }
    {
        pqxx::read_transaction read{connection};
        assert(read.exec(
            "SELECT count(*) FROM experiment WHERE "
            "fresh_initialization_seed=47;").one_row()[0].as<int>() == 0);
    }

    // Command-level integration uses complete synthetic source evidence and
    // the real PostgreSQL loader/equivalence/transaction implementation.
    {
        pqxx::work transaction{connection};
        InsertAuthoritativeSource(transaction, 100, 1000, "");
        InsertAuthoritativeSource(transaction, 101, 1001, "mask_b");
        transaction.exec(
            "SELECT setval(pg_get_serial_sequence('experiment','experiment_id'),"
            "101,true);");
        transaction.commit();
    }

    std::string output;
    std::string errors;
    assert(RunCommand(connectionString, {44}, output, errors) == 0);
    assert(errors.empty());
    assert(output.find("state=materialized,pair_count=1,experiment_count=2") !=
           std::string::npos);
    assert(output.find("transaction=committed") != std::string::npos);
    assert(CountSeed(connection, 44) == 2);

    // An identical second command is fail-closed and creates no additional
    // rows even though the fresh rows do not yet have model evidence.
    assert(RunCommand(connectionString, {44}, output, errors) == 3);
    assert(output.find("state=materialized") == std::string::npos);
    assert(output.find("transaction=rolled_back") != std::string::npos);
    assert(CountSeed(connection, 44) == 2);

    // Partial preexistence aborts the complete requested pair.
    assert(RunCommand(connectionString, {45}, output, errors) == 0);
    {
        pqxx::work transaction{connection};
        transaction.exec(
            "DELETE FROM experiment WHERE symbol='synthetic_materializer' "
            "AND fresh_initialization_seed=45 AND feature_ablation_mask='mask_b';");
        transaction.commit();
    }
    assert(CountSeed(connection, 45) == 1);
    assert(RunCommand(connectionString, {45}, output, errors) == 3);
    assert(CountSeed(connection, 45) == 1);

    assert(RunCommand(connectionString, {46, 47, 48}, output, errors) == 0);
    assert(CountSeed(connection, 46) == 2);
    assert(CountSeed(connection, 47) == 2);
    assert(CountSeed(connection, 48) == 2);
    const auto seed46 = output.find("ordinal=1,requested_seed=46");
    const auto seed47 = output.find("ordinal=2,requested_seed=47");
    const auto seed48 = output.find("ordinal=3,requested_seed=48");
    assert(seed46 < seed47 && seed47 < seed48);

    // A later arm INSERT failure rolls back all earlier rows and never exposes
    // the staged success mapping.
    {
        pqxx::work transaction{connection};
        transaction.exec(R"sql(
            CREATE FUNCTION fail_materializer_insert() RETURNS trigger
            LANGUAGE plpgsql AS $$ BEGIN
              IF NEW.fresh_initialization_seed=50 AND
                 NEW.feature_ablation_mask='mask_b' THEN
                RAISE EXCEPTION 'synthetic later insert failure';
              END IF;
              RETURN NEW;
            END $$;
            CREATE TRIGGER fail_materializer_insert_trigger
            BEFORE INSERT ON experiment FOR EACH ROW
            EXECUTE FUNCTION fail_materializer_insert();
        )sql");
        transaction.commit();
    }
    assert(RunCommand(connectionString, {49, 50}, output, errors) == 2);
    assert(output.find("state=materialized") == std::string::npos);
    assert(errors.find("transaction=rolled_back") != std::string::npos);
    assert(CountSeed(connection, 49) == 0);
    assert(CountSeed(connection, 50) == 0);
    {
        pqxx::work transaction{connection};
        transaction.exec("DROP TRIGGER fail_materializer_insert_trigger ON experiment;");
        transaction.exec("DROP FUNCTION fail_materializer_insert();");
        transaction.commit();
    }

    // A deferred failure occurs during commit. Staged IDs and materialized
    // state are discarded, and the transaction is known not to have committed.
    {
        pqxx::work transaction{connection};
        transaction.exec(R"sql(
            CREATE FUNCTION fail_materializer_commit() RETURNS trigger
            LANGUAGE plpgsql AS $$ BEGIN
              RAISE EXCEPTION 'synthetic deferred commit failure';
            END $$;
            CREATE CONSTRAINT TRIGGER fail_materializer_commit_trigger
            AFTER INSERT ON experiment DEFERRABLE INITIALLY DEFERRED
            FOR EACH ROW WHEN (NEW.fresh_initialization_seed=51)
            EXECUTE FUNCTION fail_materializer_commit();
        )sql");
        transaction.commit();
    }
    assert(RunCommand(connectionString, {51}, output, errors) == 2);
    assert(output.empty());
    assert(errors.find("transaction=commit_failed") != std::string::npos);
    assert(errors.find("state=materialized") == std::string::npos);
    assert(CountSeed(connection, 51) == 0);
    {
        pqxx::work transaction{connection};
        transaction.exec("DROP TRIGGER fail_materializer_commit_trigger ON experiment;");
        transaction.exec("DROP FUNCTION fail_materializer_commit();");
        transaction.commit();
    }

    // Two actual command invocations contend at the table lock. Exactly one
    // creates the pair; the waiter reloads, rechecks, and aborts as equivalent.
    std::array<int, 2> concurrentResults{};
    std::array<std::string, 2> concurrentOutput;
    std::array<std::string, 2> concurrentErrors;
    std::array<std::thread, 2> threads;
    for (std::size_t index = 0; index < threads.size(); ++index)
        threads[index] = std::thread([&, index]
        {
            concurrentResults[index] = RunCommand(
                connectionString, {52}, concurrentOutput[index],
                concurrentErrors[index]);
        });
    for (std::thread& thread : threads) thread.join();
    std::sort(concurrentResults.begin(), concurrentResults.end());
    assert((concurrentResults == std::array<int, 2>{0, 3}));
    assert(CountSeed(connection, 52) == 2);

    return 0;
}
