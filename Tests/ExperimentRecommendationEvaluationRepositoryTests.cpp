#include <barrier>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <pqxx/pqxx>

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationEvaluationRepository.hpp"
#include "../Sources/ExperimentRecommendationEvaluationService.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::string ReadFile(const std::string& path)
{
    std::ifstream input{path};
    if (!input) throw std::runtime_error("unable_to_read_migration");
    return {std::istreambuf_iterator<char>{input},
            std::istreambuf_iterator<char>{}};
}

EffectiveExperimentConfiguration Configuration(double threshold,
                                                int targetEpochs)
{
    EffectiveExperimentConfiguration value;
    value.symbol = "phase4bfixture";
    value.predictionHorizon = 12;
    value.labelThreshold = threshold;
    value.coreLrMult = 1.0;
    value.headLrMult = 5.0;
    value.targetEpochs = targetEpochs;
    value.trainStartDate = "2010-01-01";
    value.trainEndDate = "2025-01-01";
    return value;
}

long long InsertRecommendation(pqxx::work& transaction, long long scanId,
                               long long sourceExperimentId,
                               long long sourceAnalysisId,
                               int ordinal,
                               const EffectiveExperimentConfiguration& candidate,
                               bool includeModel = true,
                               long long evidenceCount = 100)
{
    const auto identity = BuildRecommendationCandidateIdentity(candidate);
    return transaction.exec(
        "INSERT INTO experiment_recommendation (recommendation_scan_id,status,"
        "source_experiment_id,source_model_id,source_analysis_id,source_symbol,"
        "source_prediction_horizon,source_rank,source_leader_score,"
        "source_infer_accuracy,source_predicted_neutral_proportion,"
        "source_evidence_count,changed_parameter,source_value_canonical,"
        "proposed_value_canonical,absolute_delta,relative_delta,horizon_delta,"
        "generation_ordinal,structural_rank,semantic_configuration_canonical,"
        "semantic_hash,invocation_configuration_canonical,policy_canonical,"
        "policy_hash,duplicate_type) VALUES ($1,'proposed',$2,$3,$4,"
        "'phase4bfixture',12,1,0.75,0.70,0.30,$5,'core_lr_mult','1','1.25',"
        "0.25,0.25,NULL,$6,$6,$7,$8,$9,'policy_v1',$10,"
        "'no_duplicate') RETURNING recommendation_id;",
        pqxx::params{scanId, sourceExperimentId,
            includeModel ? std::optional<long long>{30} : std::nullopt,
            sourceAnalysisId, evidenceCount, ordinal,
            identity.canonicalText, identity.hash,
            "invocation_" + std::to_string(ordinal),
            RecommendationCanonicalHash("policy_v1")})
        .one_row()[0].as<long long>();
}

std::string Snapshot(pqxx::connection& connection, const std::string& schema)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
    return transaction.exec(
        "SELECT md5(coalesce(string_agg(concat_ws('|',experiment_id::text,"
        "status,phase,current_epoch::text,target_epochs::text,"
        "coalesce(worker_pid::text,'NULL'),coalesce(current_operation,'NULL'),"
        "continuation_policy_enabled::text,checkpoint_policy_enabled::text,"
        "marker),'#' ORDER BY experiment_id),'')) FROM experiment;")
        .one_row()[0].as<std::string>() + ":" +
        transaction.exec(
            "SELECT concat_ws(':',(SELECT md5(string_agg(marker,',' ORDER BY marker)) FROM model),"
            "(SELECT md5(string_agg(marker,',' ORDER BY marker)) FROM inference_eval_result),"
            "(SELECT md5(string_agg(marker,',' ORDER BY marker)) FROM experiment_analysis_result),"
            "(SELECT md5(string_agg(marker,',' ORDER BY marker)) FROM experiment_checkpoint_eval),"
            "(SELECT md5(string_agg(marker,',' ORDER BY marker)) FROM experiment_continuation_decision));")
        .one_row()[0].as<std::string>();
}

} // namespace

int main()
{
    const std::string database = EnvironmentOr("LSTM_DB_NAME", "LSTM");
    const std::string host = EnvironmentOr("LSTM_DB_HOST", "127.0.0.1");
    const std::string owner = EnvironmentOr(
        "LSTM_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema = "phase4b_eval_" + std::to_string(getpid());
    const std::string ownerConnectionString =
        "hostaddr=" + host + " user=" + owner + " dbname=" + database;
    const std::string runtimeConnectionString =
        "hostaddr=" + host + " user=pqxx dbname=" + database +
        " options='-c search_path=" + schema + "'";
    pqxx::connection ownerConnection{ownerConnectionString};

    {
        pqxx::work setup{ownerConnection};
        setup.exec("CREATE SCHEMA " + setup.quote_name(schema) + ";");
        setup.exec("SET LOCAL search_path TO " + setup.quote_name(schema) + ";");
        setup.exec(
            "CREATE TABLE experiment (experiment_id bigserial PRIMARY KEY,"
            "symbol text NOT NULL,prediction_horizon integer NOT NULL,"
            "c_next_threshold double precision NOT NULL,core_lr_mult double precision,"
            "head_lr_mult double precision,target_epochs integer NOT NULL,"
            "checkpoint_interval integer NOT NULL,train_start timestamptz NOT NULL,"
            "train_end timestamptz NOT NULL,infer_start timestamptz,infer_end timestamptz,"
            "resume_model_id bigint,last_model_id bigint,"
            "donchian20_mode text NOT NULL DEFAULT 'enabled',"
            "donchian_lookback integer NOT NULL DEFAULT 20,"
            "feature_warmup_scope text NOT NULL DEFAULT 'full_history_warmup',"
            "training_objective_canonical text NOT NULL DEFAULT "
            "'training_objective_configuration_v1;schema_version=1;"
            "objective_id=legacy_first_hit_weighted_ce_v1;"
            "objective_family=up_neutral_down_first_hit_classification;"
            "objective_version=1;loss_definition_version=1;"
            "mode=legacy_first_hit_classification;"
            "classification_loss=true_class_weighted_softmax_cross_entropy_v1;"
            "classification_target=up_neutral_down_return_high_low_first_hit_strict_threshold_up_tie_v1;"
            "class_index_order=down_0_neutral_1_up_2;"
            "class_weight_semantics=true_class_weight_multiplies_loss_and_all_logit_components_v1;"
            "class_weight_down=1;class_weight_neutral=1;class_weight_up=1;"
            "softmax_loss_probability_floor=1e-12;"
            "classification_logit_gradient_scale=0.1;"
            "shared_core_classification_gradient_scale=4;"
            "internal_loss_normalization=weighted_loss_sum_divided_by_true_class_weight_sum_v1;"
            "calculate_batch_return_normalization=weighted_loss_sum_divided_by_example_count_v1;"
            "gradient_normalization=all_calculate_batch_gradients_divided_by_true_class_weight_sum_v1;"
            "batch_window_boundary=overlapping_windows_do_not_cross_outer_tensor_batch_v1;"
            "optimizer_family=sgd;"
            "optimizer_update=parameter_minus_learning_rate_times_gradient_v1;"
            "learning_rate_contract=base_rate_and_parameter_group_multipliers_persisted_in_training_config_v1;"
            "gradient_clipping_mode=componentwise_after_normalization_before_update;"
            "gradient_clip_threshold=10;"
            "nonfinite_gradient_policy=skip_parameter_update_v1;"
            "weight_decay=none;"
            "gradient_accumulation_precision=core_gradient_accumulation_double_head_gradient_accumulation_float_loss_accumulation_double_v1;"
            "auxiliary_loss_mode=disabled;auxiliary_loss_coefficient=0;"
            "regression_target_definition=NULL;"
            "regression_normalization_identity=NULL;"
            "robust_loss_definition=NULL;robust_loss_delta=NULL;"
            "target_clipping_definition=none;"
            "shared_gradient_combination=classification_only_v1;',"
            "training_objective_hash text NOT NULL DEFAULT "
            "'fnv1a64:65818f2e1fa1a324',"
            "status text NOT NULL,"
            "phase text NOT NULL,current_epoch integer,worker_pid integer,"
            "current_operation text,continuation_policy_enabled boolean NOT NULL,"
            "checkpoint_policy_enabled boolean NOT NULL,marker text NOT NULL);");
        setup.exec("CREATE TABLE model(model_id bigint PRIMARY KEY,marker text NOT NULL);");
        setup.exec("CREATE TABLE inference_eval_result(id bigint PRIMARY KEY,marker text NOT NULL);");
        setup.exec(
            "CREATE TABLE experiment_analysis_result(analysis_id bigserial PRIMARY KEY,"
            "experiment_id bigint NOT NULL REFERENCES experiment,model_id bigint,"
            "analysis_status text,analysis_scope text NOT NULL,marker text NOT NULL);");
        setup.exec(
            "CREATE UNIQUE INDEX experiment_analysis_result_final_uidx "
            "ON experiment_analysis_result(experiment_id,model_id) "
            "WHERE analysis_scope='final';");
        setup.exec("CREATE TABLE experiment_checkpoint_eval(id bigint PRIMARY KEY,marker text NOT NULL);");
        setup.exec("CREATE TABLE experiment_continuation_decision(id bigint PRIMARY KEY,marker text NOT NULL);");
        setup.exec(
            "CREATE TABLE experiment_recommendation_scan("
            "recommendation_scan_id bigserial PRIMARY KEY,status text NOT NULL);");
        setup.exec(
            "CREATE TABLE experiment_recommendation("
            "recommendation_id bigserial PRIMARY KEY,recommendation_scan_id bigint NOT NULL REFERENCES experiment_recommendation_scan,"
            "status text NOT NULL,source_experiment_id bigint NOT NULL REFERENCES experiment,"
            "source_model_id bigint,source_analysis_id bigint REFERENCES experiment_analysis_result,"
            "source_symbol text NOT NULL,source_prediction_horizon integer NOT NULL,"
            "source_rank integer,source_leader_score double precision NOT NULL,"
            "source_infer_accuracy double precision NOT NULL,"
            "source_predicted_neutral_proportion double precision,source_evidence_count bigint NOT NULL,"
            "changed_parameter text NOT NULL,source_value_canonical text NOT NULL,"
            "proposed_value_canonical text NOT NULL,absolute_delta double precision NOT NULL,"
            "relative_delta double precision,horizon_delta integer,generation_ordinal integer NOT NULL,"
            "structural_rank integer NOT NULL,semantic_configuration_canonical text NOT NULL,"
            "semantic_hash text NOT NULL,invocation_configuration_canonical text NOT NULL,"
            "policy_canonical text NOT NULL,policy_hash text NOT NULL,duplicate_type text NOT NULL,"
            "final_profitability_provenance_version integer,"
            "source_final_inference_eval_result_id bigint,"
            "source_final_profitability_observation_id bigint,"
            "source_final_profitability_unavailable_reason text,"
            "source_final_profitability_inference_scope text,"
            "source_final_profitability_inference_start text,"
            "source_final_profitability_inference_end text,"
            "source_final_profitability_actionable_count bigint,"
            "source_final_profitability_aggregate_return double precision,"
            "source_final_profitability_average_return double precision,"
            "source_final_profitability_metric_definition_hash text,"
            "source_final_profitability_source_content_hash text,"
            "source_final_profitability_observation_identity_hash text);");
        setup.exec(ReadFile("Database/migrations/034_experiment_recommendation_evaluation.sql"));
        setup.exec(ReadFile("Database/migrations/035_experiment_recommendation_ranking.sql"));
        setup.exec(ReadFile(
            "Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql"));
        setup.exec(ReadFile(
            "Database/migrations/083_recommendation_evaluation_run_hash_identity.sql"));
        setup.exec(
            "ALTER TABLE experiment_recommendation_evaluation_result "
            "ADD final_profitability_provenance_version integer,"
            "ADD source_final_inference_eval_result_id bigint,"
            "ADD source_final_profitability_observation_id bigint,"
            "ADD source_final_profitability_unavailable_reason text,"
            "ADD source_final_profitability_inference_scope text,"
            "ADD source_final_profitability_inference_start text,"
            "ADD source_final_profitability_inference_end text,"
            "ADD source_final_profitability_actionable_count bigint,"
            "ADD source_final_profitability_aggregate_return double precision,"
            "ADD source_final_profitability_average_return double precision,"
            "ADD source_final_profitability_metric_definition_hash text,"
            "ADD source_final_profitability_source_content_hash text,"
            "ADD source_final_profitability_observation_identity_hash text,"
            "ADD profitability_evidence_canonical text,"
            "ADD profitability_evidence_hash text;");
        setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) + " TO pqxx;");
        setup.exec("GRANT SELECT ON experiment,model,inference_eval_result,"
                   "experiment_analysis_result,experiment_checkpoint_eval,"
                   "experiment_continuation_decision,experiment_recommendation_scan,"
                   "experiment_recommendation TO pqxx;");
        setup.commit();
    }

    try
    {
        long long scanId = -1;
        long long emptyScanId = -1;
        std::vector<long long> recommendationIds;
        {
            pqxx::work fixture{ownerConnection};
            fixture.exec("SET LOCAL search_path TO " + fixture.quote_name(schema) + ";");
            fixture.exec("INSERT INTO model VALUES (30,'model_unchanged'),"
                         "(31,'other_model_unchanged');");
            fixture.exec("INSERT INTO inference_eval_result VALUES (1,'inference_unchanged');");
            fixture.exec("INSERT INTO experiment_checkpoint_eval VALUES (1,'checkpoint_unchanged');");
            fixture.exec("INSERT INTO experiment_continuation_decision VALUES (1,'continuation_unchanged');");
            const long long sourceExperimentId = fixture.exec(
                "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,"
                "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,"
                "train_end,last_model_id,status,phase,current_epoch,worker_pid,current_operation,"
                "continuation_policy_enabled,checkpoint_policy_enabled,marker) VALUES ("
                "'phase4bfixture',12,0.001,1,5,120,20,'2010-01-01','2025-01-01',"
                "30,'completed','done',120,NULL,NULL,false,false,'source_unchanged') "
                "RETURNING experiment_id;").one_row()[0].as<long long>();
            const long long analysisId = fixture.exec(
                "INSERT INTO experiment_analysis_result(experiment_id,model_id,"
                "analysis_status,analysis_scope,marker) VALUES ($1,30,'completed',"
                "'final','analysis_unchanged') RETURNING analysis_id;",
                pqxx::params{sourceExperimentId}).one_row()[0].as<long long>();
            fixture.exec(
                "INSERT INTO experiment_analysis_result(experiment_id,model_id,"
                "analysis_status,analysis_scope,marker) VALUES "
                "($1,30,'completed','checkpoint','checkpoint_analysis_unchanged'),"
                "($1,31,'completed','final','other_model_analysis_unchanged');",
                pqxx::params{sourceExperimentId});
            scanId = fixture.exec(
                "INSERT INTO experiment_recommendation_scan(status) VALUES ('completed') "
                "RETURNING recommendation_scan_id;").one_row()[0].as<long long>();
            emptyScanId = fixture.exec(
                "INSERT INTO experiment_recommendation_scan(status) VALUES ('completed') "
                "RETURNING recommendation_scan_id;").one_row()[0].as<long long>();
            recommendationIds.push_back(InsertRecommendation(
                fixture, scanId, sourceExperimentId, analysisId, 1,
                Configuration(0.002, 121)));
            recommendationIds.push_back(InsertRecommendation(
                fixture, scanId, sourceExperimentId, analysisId, 2,
                Configuration(0.003, 122)));
            recommendationIds.push_back(InsertRecommendation(
                fixture, scanId, sourceExperimentId, analysisId, 3,
                Configuration(0.004, 123), false));
            const auto pending = Configuration(0.003, 122);
            fixture.exec(
                "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,"
                "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,"
                "train_end,status,phase,current_epoch,continuation_policy_enabled,"
                "checkpoint_policy_enabled,marker) VALUES ($1,$2,$3,$4,$5,$6,20,"
                "'2010-01-01','2025-01-01','pending','train',0,false,false,'duplicate_unchanged');",
                pqxx::params{pending.symbol,pending.predictionHorizon,pending.labelThreshold,
                    pending.coreLrMult,pending.headLrMult,pending.targetEpochs});
            fixture.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(RecommendationEvaluationSchemaExists(runtime));
        const auto rejectsInvalid = [](auto&& operation,
                                       const std::string& expected) {
            try { operation(); }
            catch (const std::invalid_argument& error) {
                return std::string{error.what()} == expected;
            }
            return false;
        };
        RecommendationEvaluationFilters invalidFilters;
        invalidFilters.limit = 0;
        assert(rejectsInvalid(
            [&] { (void)LoadRecommendationsForEvaluation(runtime, invalidFilters); },
            "recommendation_evaluation_limit_invalid"));
        invalidFilters.limit = 1001;
        assert(rejectsInvalid(
            [&] { (void)ListRecommendationEvaluations(runtime, invalidFilters); },
            "recommendation_evaluation_limit_invalid"));
        assert(rejectsInvalid(
            [&] { (void)FindRecommendationEvaluation(runtime, 0); },
            "recommendation_evaluation_result_id_invalid"));
        assert(rejectsInvalid(
            [&] { (void)FindRecommendationEvaluationRun(runtime, 0); },
            "recommendation_evaluation_run_id_invalid"));
        assert(rejectsInvalid(
            [&] { FailRecommendationEvaluationRun(
                runtime, 0, {}, "invalid_id"); },
            "recommendation_evaluation_run_id_invalid"));
        const std::string before = Snapshot(ownerConnection, schema);
        RecommendationEvaluationCommandRequest request;
        request.recommendationScanId = scanId;
        request.limit = 100;
        std::ostringstream output;
        std::ostringstream errors;
        const int evaluationExit = RunEvaluateExperimentRecommendationsCommand(
            runtimeConnectionString, request, output, errors);
        if (evaluationExit != 0)
            throw std::runtime_error("evaluation_failed:" + errors.str());
        assert(errors.str().empty());
        assert(output.str().find("disposition=advisory_ready") != std::string::npos);
        assert(output.str().find("disposition=blocked_pending_duplicate") != std::string::npos);
        assert(output.str().find("disposition=insufficient_evidence") != std::string::npos);
        assert(output.str().find("recommendation_semantic_hash=fnv1a64:") !=
               std::string::npos);
        assert(output.str().find("evaluation_identity_hash=fnv1a64:") !=
               std::string::npos);
        assert(output.str().find("recommendation_semantic_identity=") ==
               std::string::npos);
        assert(output.str().find("Positive components:") != std::string::npos);
        assert(output.str().find("Penalties:") != std::string::npos);
        assert(output.str().find("Advisory evaluation only.") != std::string::npos);
        assert(output.str().find("experiment_created=false,experiment_queued=false,scheduler_modified=false") != std::string::npos);
        assert(Snapshot(ownerConnection, schema) == before);

        RecommendationEvaluationFilters filters;
        filters.recommendationScanId = scanId;
        const auto results = ListRecommendationEvaluations(runtime, filters);
        assert(results.size() == 3);
        const auto ready = std::find_if(results.begin(), results.end(), [](const auto& row) {
            return row.disposition == RecommendationEvaluationDisposition::advisoryReady;
        });
        assert(ready != results.end() && ready->componentCount == 9);
        const auto detail = FindRecommendationEvaluation(runtime, ready->evaluationResultId);
        assert(detail && detail->components.size() == 9);
        assert(!ready->recommendationSemanticHash.empty());
        assert(detail->recommendationSemanticHash ==
               ready->recommendationSemanticHash);
        bool resultUpdateDenied = false;
        try
        {
            pqxx::work unauthorized{runtime};
            unauthorized.exec("SET TRANSACTION READ WRITE;");
            unauthorized.exec(
                "UPDATE experiment_recommendation_evaluation_result "
                "SET explanation='tampered' WHERE recommendation_evaluation_result_id=$1;",
                pqxx::params{ready->evaluationResultId});
            unauthorized.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            const std::string message = error.what();
            if (error.sqlstate() != "42501" &&
                message.find("permission denied") == std::string::npos)
                throw std::runtime_error(
                    "unexpected_result_update_error:" + message);
            resultUpdateDenied = true;
        }
        if (!resultUpdateDenied)
            throw std::runtime_error("runtime_result_update_was_permitted");

        bool immutableRunUpdateDenied = false;
        try
        {
            pqxx::work unauthorized{runtime};
            unauthorized.exec("SET TRANSACTION READ WRITE;");
            unauthorized.exec(
                "UPDATE experiment_recommendation_evaluation_run "
                "SET evaluation_policy_canonical='tampered' "
                "WHERE recommendation_evaluation_run_id=$1;",
                pqxx::params{ready->evaluationRunId});
            unauthorized.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            const std::string message = error.what();
            if (error.sqlstate() != "42501" &&
                message.find("permission denied") == std::string::npos)
                throw std::runtime_error(
                    "unexpected_run_update_error:" + message);
            immutableRunUpdateDenied = true;
        }
        if (!immutableRunUpdateDenied)
            throw std::runtime_error("runtime_immutable_run_update_was_permitted");

        std::ostringstream retryOutput;
        std::ostringstream retryErrors;
        assert(RunEvaluateExperimentRecommendationsCommand(
            runtimeConnectionString, request, retryOutput, retryErrors) == 0);
        assert(ListRecommendationEvaluations(runtime, filters).size() == 3);
        assert(ListRecommendationEvaluationRuns(runtime, 10).size() == 1);

        RecommendationEvaluationCommandRequest zeroRequest;
        zeroRequest.recommendationScanId = emptyScanId;
        std::ostringstream zeroOutput;
        std::ostringstream zeroErrors;
        assert(RunEvaluateExperimentRecommendationsCommand(
            runtimeConnectionString, zeroRequest, zeroOutput, zeroErrors) == 0);
        assert(zeroErrors.str().empty());
        assert(zeroOutput.str().find("count=0,persisted=true") !=
               std::string::npos);
        assert(ListRecommendationEvaluationRuns(runtime, 10).size() == 2);
        std::ostringstream zeroRetryOutput;
        std::ostringstream zeroRetryErrors;
        assert(RunEvaluateExperimentRecommendationsCommand(
            runtimeConnectionString, zeroRequest,
            zeroRetryOutput, zeroRetryErrors) == 0);
        assert(ListRecommendationEvaluationRuns(runtime, 10).size() == 2);

        RecommendationEvaluationCommandRequest concurrentRequest = request;
        concurrentRequest.policy.scoringPolicy.leaderScoreWeight = 0.30;

        struct ConcurrentEvaluationResult
        {
            int exitCode;
            std::string output;
            std::string errors;
        };

        std::barrier concurrentStart{3};
        const auto concurrentEvaluation = [&] {
            concurrentStart.arrive_and_wait();
            std::ostringstream concurrentOutput;
            std::ostringstream concurrentErrors;
            const int exitCode =
                RunEvaluateExperimentRecommendationsCommand(
                    runtimeConnectionString, concurrentRequest,
                    concurrentOutput, concurrentErrors);
            return ConcurrentEvaluationResult{
                exitCode,
                concurrentOutput.str(),
                concurrentErrors.str()};
        };

        auto firstConcurrent = std::async(
            std::launch::async, concurrentEvaluation);
        auto secondConcurrent = std::async(
            std::launch::async, concurrentEvaluation);
        concurrentStart.arrive_and_wait();

        const auto firstConcurrentResult = firstConcurrent.get();
        const auto secondConcurrentResult = secondConcurrent.get();

        if (firstConcurrentResult.exitCode != 0 ||
            secondConcurrentResult.exitCode != 0)
        {
            std::fprintf(
                stderr,
                "===== FIRST CONCURRENT EVALUATION =====\n"
                "EXIT_CODE=%d\n"
                "STDOUT:\n%s\n"
                "STDERR:\n%s\n"
                "===== SECOND CONCURRENT EVALUATION =====\n"
                "EXIT_CODE=%d\n"
                "STDOUT:\n%s\n"
                "STDERR:\n%s\n",
                firstConcurrentResult.exitCode,
                firstConcurrentResult.output.c_str(),
                firstConcurrentResult.errors.c_str(),
                secondConcurrentResult.exitCode,
                secondConcurrentResult.output.c_str(),
                secondConcurrentResult.errors.c_str());
        }

        assert(firstConcurrentResult.exitCode == 0);
        assert(secondConcurrentResult.exitCode == 0);
        assert(ListRecommendationEvaluationRuns(runtime, 10).size() == 3);
        assert(ListRecommendationEvaluations(runtime, filters).size() == 6);

        request.dryRun = true;
        std::ostringstream dryOutput;
        std::ostringstream dryErrors;
        assert(RunEvaluateExperimentRecommendationsCommand(
            runtimeConnectionString, request, dryOutput, dryErrors) == 0);
        assert(dryOutput.str().find("persisted=false") != std::string::npos);
        assert(ListRecommendationEvaluationRuns(runtime, 10).size() == 3);
        assert(Snapshot(ownerConnection, schema) == before);

        const auto run = ListRecommendationEvaluationRuns(runtime, 10).front();
        assert(run.status == "completed");
        auto loadFilters = filters;
        const auto loaded = LoadRecommendationsForEvaluation(runtime, loadFilters);
        const auto loadedReady = std::find_if(loaded.begin(), loaded.end(),
            [&](const auto& row) { return row.recommendationId == ready->recommendationId; });
        assert(loadedReady != loaded.end());
        auto mismatch = EvaluateExperimentRecommendation(
            concurrentRequest.policy, loadedReady->input);
        mismatch = RankRecommendationEvaluations({mismatch}).front();
        mismatch.explanation = "conflicting retry";
        bool retryMismatch = false;
        try
        {
            (void)PersistRecommendationEvaluation(runtime,
                {run.evaluationRunId, loadedReady->input, mismatch});
        }
        catch (const std::runtime_error& error)
        {
            retryMismatch = std::string{error.what()} ==
                "recommendation_evaluation_retry_mismatch";
        }
        assert(retryMismatch);
        assert(Snapshot(ownerConnection, schema) == before);

        long long additionalRecommendationId = -1;
        {
            pqxx::work additionalFixture{ownerConnection};
            additionalFixture.exec("SET LOCAL search_path TO " +
                additionalFixture.quote_name(schema) + ";");
            additionalRecommendationId = InsertRecommendation(
                additionalFixture, scanId,
                loadedReady->input.scoringInput.sourceExperimentId,
                *loadedReady->input.sourceAnalysisId, 4,
                Configuration(0.005, 124));
            additionalFixture.commit();
        }
        const auto expandedLoaded = LoadRecommendationsForEvaluation(
            runtime, loadFilters);
        const auto additionalLoaded = std::find_if(
            expandedLoaded.begin(), expandedLoaded.end(),
            [additionalRecommendationId](const auto& row) {
                return row.recommendationId == additionalRecommendationId;
            });
        assert(additionalLoaded != expandedLoaded.end());

        RecommendationEvaluationRunRequest conflictRunRequest;
        conflictRunRequest.policy = {};
        conflictRunRequest.filters = filters;
        conflictRunRequest.runIdentityCanonical = "conflict_target_run";
        conflictRunRequest.runIdentityHash = RecommendationEvaluationCanonicalHash(
            conflictRunRequest.runIdentityCanonical);
        conflictRunRequest.evidenceSnapshotCanonical =
            "conflict_target_snapshot";
        conflictRunRequest.evidenceSnapshotHash =
            RecommendationEvaluationCanonicalHash(
                conflictRunRequest.evidenceSnapshotCanonical);
        const auto conflictRun = BeginOrFindRecommendationEvaluationRun(
            runtime, conflictRunRequest);

        // Retry with identical hash/canonical identity must reuse the same run.
        const auto conflictRunRetry = BeginOrFindRecommendationEvaluationRun(
            runtime, conflictRunRequest);
        assert(!conflictRunRetry.created);
        assert(conflictRunRetry.evaluationRunId ==
               conflictRun.evaluationRunId);

        // Hash-based lookup is only an accelerator. Reusing the same valid
        // run identity with different valid persisted evidence must fail closed
        // after the existing row is located.
        auto retryMismatchRequest = conflictRunRequest;
        retryMismatchRequest.evidenceSnapshotCanonical =
            "different_conflict_target_snapshot";
        retryMismatchRequest.evidenceSnapshotHash =
            RecommendationEvaluationCanonicalHash(
                retryMismatchRequest.evidenceSnapshotCanonical);

        bool runRetryMismatchRejected = false;
        try
        {
            (void)BeginOrFindRecommendationEvaluationRun(
                runtime, retryMismatchRequest);
        }
        catch (const std::runtime_error& error)
        {
            runRetryMismatchRejected =
                std::string{error.what()} ==
                "recommendation_evaluation_run_retry_mismatch";
        }
        assert(runRetryMismatchRejected);

        auto firstConflictResult = RankRecommendationEvaluations({
            EvaluateExperimentRecommendation({}, loadedReady->input)}).front();
        const auto firstConflictPersisted = PersistRecommendationEvaluation(
            runtime, {conflictRun.evaluationRunId, loadedReady->input,
                      firstConflictResult});
        assert(firstConflictPersisted.created);

        auto secondConflictResult = RankRecommendationEvaluations({
            EvaluateExperimentRecommendation({}, additionalLoaded->input)}).front();
        bool duplicateOrdinalRejected = false;
        try
        {
            (void)PersistRecommendationEvaluation(
                runtime, {conflictRun.evaluationRunId,
                          additionalLoaded->input, secondConflictResult});
        }
        catch (const pqxx::unique_violation& error)
        {
            duplicateOrdinalRejected = error.sqlstate() == "23505";
        }
        assert(duplicateOrdinalRejected);

        secondConflictResult.rankingOrdinal = 2;
        secondConflictResult.evaluationIdentityCanonical =
            firstConflictResult.evaluationIdentityCanonical;
        secondConflictResult.evaluationIdentityHash =
            firstConflictResult.evaluationIdentityHash;
        bool duplicateIdentityRejected = false;
        try
        {
            (void)PersistRecommendationEvaluation(
                runtime, {conflictRun.evaluationRunId,
                          additionalLoaded->input, secondConflictResult});
        }
        catch (const pqxx::check_violation& error)
        {
            duplicateIdentityRejected = error.sqlstate() == "23514";
        }
        assert(duplicateIdentityRejected);

        secondConflictResult = RankRecommendationEvaluations({
            EvaluateExperimentRecommendation({}, additionalLoaded->input)}).front();
        secondConflictResult.rankingOrdinal = 2;
        secondConflictResult.components[1].componentName =
            secondConflictResult.components[0].componentName;
        bool unrelatedUniqueConflictRejected = false;
        try
        {
            (void)PersistRecommendationEvaluation(
                runtime, {conflictRun.evaluationRunId,
                          additionalLoaded->input, secondConflictResult});
        }
        catch (const pqxx::unique_violation& error)
        {
            unrelatedUniqueConflictRejected = error.sqlstate() == "23505";
        }
        assert(unrelatedUniqueConflictRejected);

        const auto exactConflictRetry = PersistRecommendationEvaluation(
            runtime, {conflictRun.evaluationRunId, loadedReady->input,
                      firstConflictResult});
        assert(!exactConflictRetry.created &&
               exactConflictRetry.evaluationResultId ==
                   firstConflictPersisted.evaluationResultId);

        RecommendationEvaluationRunCounters partialCounters;
        partialCounters.recommendationsConsidered = 2;
        partialCounters.recommendationsEvaluated = 1;
        partialCounters.recommendationsEligible = 1;
        partialCounters.evaluationErrors = 1;
        FailRecommendationEvaluationRun(runtime, conflictRun.evaluationRunId,
            partialCounters, "intentional_partial_run_failure");
        const auto failedRun = FindRecommendationEvaluationRun(
            runtime, conflictRun.evaluationRunId);
        assert(failedRun && failedRun->status == "failed" &&
               failedRun->counters.recommendationsEvaluated == 1);
        const auto failedRunExistingRetry = PersistRecommendationEvaluation(
            runtime, {conflictRun.evaluationRunId, loadedReady->input,
                      firstConflictResult});
        assert(!failedRunExistingRetry.created);
        bool failedRunNewResultRejected = false;
        secondConflictResult = RankRecommendationEvaluations({
            EvaluateExperimentRecommendation({}, additionalLoaded->input)}).front();
        secondConflictResult.rankingOrdinal = 2;
        try
        {
            (void)PersistRecommendationEvaluation(
                runtime, {conflictRun.evaluationRunId,
                          additionalLoaded->input, secondConflictResult});
        }
        catch (const std::runtime_error& error)
        {
            failedRunNewResultRejected = std::string{error.what()} ==
                "recommendation_evaluation_run_not_running";
        }
        assert(failedRunNewResultRejected);
        RecommendationEvaluationFilters additionalFilters;
        additionalFilters.recommendationId = additionalRecommendationId;
        assert(ListRecommendationEvaluations(runtime, additionalFilters).empty());
        const auto partialFilters = RecommendationEvaluationFilters{
            std::nullopt, loadedReady->recommendationId, std::nullopt, 100};
        const auto partialResults = ListRecommendationEvaluations(
            runtime, partialFilters);
        assert(std::any_of(
            partialResults.begin(), partialResults.end(),
            [&](const auto& value) {
                return value.evaluationRunId == conflictRun.evaluationRunId;
            }));
        assert(Snapshot(ownerConnection, schema) == before);

        RecommendationEvaluationRunRequest atomicRunRequest;
        atomicRunRequest.policy = {};
        atomicRunRequest.filters = filters;
        atomicRunRequest.runIdentityCanonical = "atomic_failure_run";
        atomicRunRequest.runIdentityHash = RecommendationEvaluationCanonicalHash(
            atomicRunRequest.runIdentityCanonical);
        atomicRunRequest.evidenceSnapshotCanonical = "atomic_failure_snapshot";
        atomicRunRequest.evidenceSnapshotHash = RecommendationEvaluationCanonicalHash(
            atomicRunRequest.evidenceSnapshotCanonical);
        const auto atomicRun = BeginOrFindRecommendationEvaluationRun(
            runtime, atomicRunRequest);
        auto invalidComponentResult = EvaluateExperimentRecommendation(
            {}, loadedReady->input);
        invalidComponentResult = RankRecommendationEvaluations(
            {invalidComponentResult}).front();
        invalidComponentResult.components.front().componentName =
            "invalid_component";
        bool partialFailureRolledBack = false;
        try
        {
            (void)PersistRecommendationEvaluation(runtime,
                {atomicRun.evaluationRunId, loadedReady->input,
                 invalidComponentResult});
        }
        catch (const pqxx::check_violation& error)
        {
            pqxx::read_transaction verify{runtime};
            partialFailureRolledBack = error.sqlstate() == "23514" &&
                verify.exec(
                    "SELECT count(*) FROM experiment_recommendation_evaluation_result "
                    "WHERE recommendation_evaluation_run_id=$1;",
                    pqxx::params{atomicRun.evaluationRunId})
                    .one_row()[0].as<int>() == 0;
        }
        assert(partialFailureRolledBack);
        RecommendationEvaluationRunCounters failedCounters;
        failedCounters.evaluationErrors = 1;
        FailRecommendationEvaluationRun(runtime, atomicRun.evaluationRunId,
            failedCounters, "intentional_atomicity_fixture");
        assert(Snapshot(ownerConnection, schema) == before);
    }
    catch (...)
    {
        pqxx::work cleanup{ownerConnection};
        cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE;");
        cleanup.commit();
        throw;
    }
    pqxx::work cleanup{ownerConnection};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE;");
    cleanup.commit();
    return 0;
}
