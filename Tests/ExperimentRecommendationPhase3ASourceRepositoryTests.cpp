#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <pqxx/pqxx>

#include "../Sources/ExperimentRecommendationRepository.hpp"
#include "../Sources/InferenceProfitability.hpp"

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
    std::ostringstream contents;
    contents << input.rdbuf();
    assert(input.good() || input.eof());
    return contents.str();
}

const RecommendationSourceLoadResult& Find(
    const std::vector<RecommendationSourceLoadResult>& values,
    long long experimentId)
{
    for (const auto& value : values)
        if (value.experimentId == experimentId) return value;
    assert(false && "missing recommendation source fixture");
    return values.front();
}

} // namespace

int main()
{
    const std::string host = EnvironmentOr("LSTM_DB_HOST", "127.0.0.1");
    const std::string database = EnvironmentOr("LSTM_DB_NAME", "LSTM");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "recommendation_phase3a_source_" + std::to_string(getpid());
    const std::string ownerConnection =
        "host=" + host + " user=" + ownerUser + " dbname=" + database;
    const std::string runtimeConnection =
        "host=" + host + " user=pqxx dbname=" + database +
        " options='-c search_path=" + schema + "'";

    pqxx::connection owner{ownerConnection};
    try
    {
        pqxx::work setup{owner};
        setup.exec("CREATE SCHEMA " + setup.quote_name(schema));
        setup.exec("SET LOCAL search_path TO " + setup.quote_name(schema));
        setup.exec(R"SQL(
CREATE TABLE experiment(
    experiment_id bigint PRIMARY KEY,symbol text,prediction_horizon integer,
    c_next_threshold double precision,core_lr_mult double precision,
    head_lr_mult double precision,target_epochs integer,
    checkpoint_interval integer,donchian20_mode text,train_start timestamptz,
    train_end timestamptz,infer_start timestamptz,infer_end timestamptz,
    resume_model_id bigint,last_model_id bigint,resume_expand_input_width boolean,
    donchian_lookback integer,feature_warmup_scope text,status text,phase text);
CREATE TABLE model(
    model_id bigint PRIMARY KEY,experiment_id bigint REFERENCES experiment);
CREATE TABLE matrix(
    model_id bigint REFERENCES model,param_name text,row_idx integer,
    col_idx integer,value double precision);
CREATE TABLE experiment_analysis_result(
    analysis_id bigint PRIMARY KEY,experiment_id bigint REFERENCES experiment,
    model_id bigint REFERENCES model,analysis_status text,analysis_scope text,
    leader_score double precision,infer_accuracy double precision,
    pred_down_count bigint,pred_neutral_count bigint,pred_up_count bigint);
CREATE TABLE experiment_checkpoint_eval(
    checkpoint_eval_id bigint PRIMARY KEY,
    parent_experiment_id bigint REFERENCES experiment,
    checkpoint_model_id bigint REFERENCES model);
CREATE TABLE inference_eval_result(
    id bigint PRIMARY KEY,model_id bigint REFERENCES model,symbol text,
    prediction_horizon integer,threshold_logret double precision,
    window_size bigint,label_rule_id integer,target_type integer,
    from_date text,to_date text,completed_epochs bigint,status text,
    inference_scope text,checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval,
    parent_experiment_id bigint REFERENCES experiment,accept_model boolean);
CREATE TABLE experiment_recommendation_scan(
    recommendation_scan_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation(
    recommendation_id bigint PRIMARY KEY);
)SQL");

        setup.exec(R"SQL(
INSERT INTO experiment VALUES
 (1,'eurusd',12,0.001,1,5,100,20,'enabled',
  '2024-01-01','2025-01-01','2025-01-01','2026-01-01',NULL,10,false,
  20,'full_history_warmup','completed','done'),
 (2,'gbpusd',12,0.001,1,5,100,20,'enabled',
  '2024-01-01','2025-01-01','2025-01-01','2026-01-01',NULL,20,false,
  20,'full_history_warmup','completed','done'),
 (3,'usdjpy',12,0.001,1,5,100,20,'enabled',
  '2024-01-01','2025-01-01','2025-01-01','2026-01-01',NULL,30,false,
  20,'full_history_warmup','completed','done');
INSERT INTO model VALUES (10,1),(11,1),(20,2),(30,3);
INSERT INTO matrix
SELECT model_id,'train_config_meta',0,col_idx,
       CASE col_idx WHEN 1 THEN 12 WHEN 2 THEN 0.001 WHEN 3 THEN 32
                    WHEN 4 THEN 1 WHEN 10 THEN 100 ELSE 0 END
FROM (VALUES (10),(11),(20),(30)) models(model_id)
CROSS JOIN generate_series(0,13) columns(col_idx);
INSERT INTO matrix VALUES
    (10,'target_meta',0,0,1),(11,'target_meta',0,0,1),
    (20,'target_meta',0,0,1),(30,'target_meta',0,0,1);
INSERT INTO experiment_analysis_result VALUES
    (10001,1,10,'completed','final',0.8,0.7,20,30,50),
    (10002,2,20,'completed','final',0.8,0.7,20,30,50),
    (10003,3,30,'completed','final',0.8,0.7,20,30,50);
INSERT INTO experiment_checkpoint_eval VALUES (200,1,10),(201,2,20);
INSERT INTO inference_eval_result VALUES
    (100,10,'eurusd',12,0.001,32,1,1,'2025-01-01','2026-01-01',100,
     'completed','final',NULL,NULL,true),
    (101,10,'eurusd',12,0.001,32,1,1,'2025-01-01','2026-01-01',100,
     'completed','checkpoint',200,1,true),
    (102,10,'eurusd',12,0.001,32,1,1,'2024-01-01','2024-12-31',100,
     'completed','final',NULL,NULL,true),
    (103,11,'eurusd',12,0.001,32,1,1,'2025-01-01','2026-01-01',100,
     'completed','final',NULL,NULL,true),
    (104,20,'gbpusd',12,0.001,32,1,1,'2025-01-01','2026-01-01',100,
     'completed','final',NULL,NULL,true),
    (105,20,'gbpusd',12,0.001,32,1,1,'2025-01-01','2026-01-01',100,
     'completed','checkpoint',201,2,true),
    (106,30,'usdjpy',12,0.001,32,1,1,'2025-01-01','2026-01-01',100,
     'completed','final',NULL,NULL,true),
    (107,30,'usdjpy',12,0.001,32,1,1,'2025-01-01','2026-01-01',100,
     'completed','final',NULL,NULL,true);
)SQL");
        setup.exec(ReadFile(
            "Database/migrations/073_inference_profitability_observation.sql"));
        setup.exec(R"SQL(
INSERT INTO inference_profitability_observation(
    profitability_observation_id,experiment_id,model_id,
    inference_eval_result_id,inference_scope,checkpoint_eval_id,
    inference_start,inference_end,prediction_count,actionable_count,
    winning_actionable_count,losing_actionable_count,
    gross_positive_terminal_horizon_log_return_sum,
    gross_negative_terminal_horizon_log_return_sum,
    aggregate_terminal_horizon_log_return_sum,
    average_terminal_horizon_log_return_per_actionable_prediction,
    metric_definition_canonical,metric_definition_hash,source_content_hash,
    observation_identity_canonical,observation_identity_hash)
VALUES
    (1000,1,10,100,'final',NULL,'2025-01-01','2026-01-01',100,80,
     60,20,12,-2,10,0.125,$1,$2,'fnv1a64:1111111111111111',
     'authoritative-final','fnv1a64:2222222222222222'),
    (1001,1,10,101,'checkpoint',200,'2025-01-01','2026-01-01',100,80,
     20,60,2,-12,-10,-0.125,$1,$2,'fnv1a64:3333333333333333',
     'checkpoint-not-fallback','fnv1a64:4444444444444444'),
    (1002,1,10,102,'final',NULL,'2024-01-01','2024-12-31',100,80,
     80,0,20,0,20,0.25,$1,$2,'fnv1a64:5555555555555555',
     'stale-final','fnv1a64:6666666666666666'),
    (1003,1,11,103,'final',NULL,'2025-01-01','2026-01-01',100,80,
     80,0,30,0,30,0.375,$1,$2,'fnv1a64:7777777777777777',
     'other-model','fnv1a64:8888888888888888'),
    (1004,2,20,105,'checkpoint',201,'2025-01-01','2026-01-01',100,80,
     80,0,40,0,40,0.5,$1,$2,'fnv1a64:9999999999999999',
     'missing-final-checkpoint-only','fnv1a64:aaaaaaaaaaaaaaaa');
)SQL", pqxx::params{
            EA::InferenceProfitability::kMetricDefinitionCanonical,
            EA::InferenceProfitability::MetricDefinitionHash()});
        setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                   " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                   setup.quote_name(schema) + " TO pqxx;");
        setup.commit();

        pqxx::connection runtime{runtimeConnection};
        const auto sources = LoadRecommendationSources(runtime, {});
        assert(sources.size() == 3);

        const auto& exact = Find(sources, 1);
        if (!exact.source) std::cerr << exact.skipReason << '\n';
        assert(exact.source);
        assert(exact.source->finalProfitabilityEvidence);
        const auto& exactEvidence =
            *exact.source->finalProfitabilityEvidence;
        assert(exactEvidence.Available());
        assert(exactEvidence.finalInferenceEvalResultId == 100);
        assert(exactEvidence.profitabilityObservationId == 1000);
        assert(exactEvidence.actionablePredictionCount == 80);
        assert(exactEvidence.aggregateTerminalHorizonLogReturnSum == 10.0);
        assert(exactEvidence.averageTerminalHorizonLogReturnPerActionablePrediction
               == 0.125);

        const auto& missing = Find(sources, 2);
        if (!missing.source) std::cerr << missing.skipReason << '\n';
        assert(missing.source);
        assert(missing.source->finalProfitabilityEvidence);
        const auto& missingEvidence =
            *missing.source->finalProfitabilityEvidence;
        assert(!missingEvidence.Available());
        assert(missingEvidence.finalInferenceEvalResultId == 104);
        assert(!missingEvidence.profitabilityObservationId);
        assert(missingEvidence.unavailableReason ==
               "no_profitability_observation");

        const auto& ambiguous = Find(sources, 3);
        if (!ambiguous.source) std::cerr << ambiguous.skipReason << '\n';
        assert(ambiguous.source);
        assert(ambiguous.source->finalProfitabilityEvidence);
        const auto& ambiguousEvidence =
            *ambiguous.source->finalProfitabilityEvidence;
        assert(!ambiguousEvidence.Available());
        assert(!ambiguousEvidence.finalInferenceEvalResultId);
        assert(ambiguousEvidence.unavailableReason ==
               "ambiguous_final_inference_result");
    }
    catch (...)
    {
        pqxx::work cleanup{owner};
        cleanup.exec("DROP SCHEMA IF EXISTS " +
                     cleanup.quote_name(schema) + " CASCADE");
        cleanup.commit();
        throw;
    }
    pqxx::work cleanup{owner};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE");
    cleanup.commit();
    return 0;
}
