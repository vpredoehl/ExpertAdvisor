#include <cassert>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <pqxx/pqxx>

#include "../Sources/ExperimentRecommendationEvaluationRepository.hpp"

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

RecommendationEvaluationInput Input(long long recommendationId)
{
    RecommendationEvaluationInput input;
    auto& score = input.scoringInput;
    score.recommendationId = recommendationId;
    score.sourceExperimentId = 1;
    score.sourcePredictionHorizon = 12;
    score.sourceRankWithinGroup = 1;
    score.sourceLeaderScore = 0.8;
    score.sourceInferenceAccuracy = 0.7;
    score.sourcePredictedNeutralProportion = 0.3;
    score.sourceEvidenceCount = 100;
    score.changedParameter = kCoreLrMult;
    score.sourceValueCanonical = "1";
    score.proposedValueCanonical = "1.25";
    score.absoluteDelta = 0.25;
    score.relativeDelta = 0.25;
    score.generationOrdinal = 1;
    score.structuralRank = 1;
    score.semanticCanonicalText =
        "semantic_" + std::to_string(recommendationId);
    score.invocationCanonicalText =
        "invocation_" + std::to_string(recommendationId);
    score.recommendationPolicyCanonicalText = "recommendation_policy";
    score.duplicateType = "no_duplicate";
    score.recommendationStatus = "proposed";
    input.recommendationSemanticHash =
        "semantic_hash_" + std::to_string(recommendationId);
    input.recommendationScanId = 1;
    input.sourceModelId = 1;
    input.sourceAnalysisId = 1;
    input.sourceSymbol = "eurusd";
    input.scanStatus = "completed";
    input.sourceExperimentStatus = "completed";
    input.sourceExperimentPhase = "done";
    input.currentSourceModelId = 1;
    input.currentSourceAnalysisId = 1;
    input.currentSourceAnalysisStatus = "completed";
    input.currentSourceAnalysisScope = "final";

    RecommendationSource::FinalProfitabilityEvidence evidence;
    evidence.finalInferenceEvalResultId = 100;
    evidence.profitabilityObservationId = 200;
    evidence.inferenceStart = "2025-01-01";
    evidence.inferenceEnd = "2026-01-01";
    evidence.actionablePredictionCount = 10;
    evidence.aggregateTerminalHorizonLogReturnSum = 1.0;
    evidence.averageTerminalHorizonLogReturnPerActionablePrediction = 0.1;
    evidence.metricDefinitionHash = "fnv1a64:1111111111111111";
    evidence.sourceContentHash = "fnv1a64:2222222222222222";
    evidence.observationIdentityHash = "fnv1a64:3333333333333333";
    input.finalProfitabilityEvidence = evidence;
    return input;
}

} // namespace

int main()
{
    const std::string host = EnvironmentOr("LSTM_DB_HOST", "127.0.0.1");
    const std::string database = EnvironmentOr("LSTM_DB_NAME", "LSTM");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "recommendation_phase3a_repository_" + std::to_string(getpid());
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
CREATE TABLE experiment(experiment_id bigint PRIMARY KEY);
CREATE TABLE model(model_id bigint PRIMARY KEY);
CREATE TABLE experiment_analysis_result(analysis_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation_scan(
    recommendation_scan_id bigint PRIMARY KEY);
CREATE TABLE experiment_recommendation(
    recommendation_id bigint PRIMARY KEY,
    recommendation_scan_id bigint REFERENCES experiment_recommendation_scan,
    source_experiment_id bigint REFERENCES experiment,
    source_model_id bigint REFERENCES model,
    source_analysis_id bigint REFERENCES experiment_analysis_result,
    source_symbol text,source_prediction_horizon integer,
    changed_parameter text,semantic_hash text,policy_hash text);
)SQL");
        setup.exec(ReadFile(
            "Database/migrations/034_experiment_recommendation_evaluation.sql"));
        setup.exec(R"SQL(
ALTER TABLE experiment_recommendation_evaluation_result
    ADD final_profitability_provenance_version integer,
    ADD source_final_inference_eval_result_id bigint,
    ADD source_final_profitability_observation_id bigint,
    ADD source_final_profitability_unavailable_reason text,
    ADD source_final_profitability_inference_scope text,
    ADD source_final_profitability_inference_start text,
    ADD source_final_profitability_inference_end text,
    ADD source_final_profitability_actionable_count bigint,
    ADD source_final_profitability_aggregate_return double precision,
    ADD source_final_profitability_average_return double precision,
    ADD source_final_profitability_metric_definition_hash text,
    ADD source_final_profitability_source_content_hash text,
    ADD source_final_profitability_observation_identity_hash text,
    ADD profitability_evidence_canonical text,
    ADD profitability_evidence_hash text;
)SQL");
        setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                   " TO pqxx; GRANT SELECT ON experiment_recommendation "
                   "TO pqxx;");
        setup.exec(R"SQL(
INSERT INTO experiment VALUES(1);
INSERT INTO model VALUES(1);
INSERT INTO experiment_analysis_result VALUES(1);
INSERT INTO experiment_recommendation_scan VALUES(1);
INSERT INTO experiment_recommendation VALUES(
    1,1,1,1,1,'eurusd',12,'core_lr_mult','semantic_hash_1',
    'recommendation_policy_hash');
)SQL");
        setup.commit();

        pqxx::connection runtime{runtimeConnection};
        RecommendationEvaluationRunRequest runRequest;
        runRequest.runIdentityCanonical = "phase3a_run";
        runRequest.runIdentityHash = RecommendationCanonicalHash("phase3a_run");
        runRequest.evidenceSnapshotCanonical = "decision_snapshot";
        runRequest.evidenceSnapshotHash =
            RecommendationCanonicalHash("decision_snapshot");
        const auto run = BeginOrFindRecommendationEvaluationRun(
            runtime, runRequest);
        assert(run.created);

        RecommendationEvaluationPersistenceRequest persistence;
        persistence.evaluationRunId = run.evaluationRunId;
        persistence.input = Input(1);
        persistence.result = EvaluateExperimentRecommendation(
            runRequest.policy, persistence.input);
        persistence.result.rankingOrdinal = 1;
        const auto inserted = PersistRecommendationEvaluation(
            runtime, persistence);
        assert(inserted.created);
        const auto retry = PersistRecommendationEvaluation(runtime, persistence);
        assert(!retry.created);
        assert(retry.evaluationResultId == inserted.evaluationResultId);

        const auto detail = FindRecommendationEvaluation(
            runtime, inserted.evaluationResultId);
        assert(detail);
        assert(detail->finalProfitabilityEvidence);
        assert(detail->finalProfitabilityEvidence->Available());
        assert(detail->finalProfitabilityEvidence
                   ->profitabilityObservationId == 200);
        assert(detail->profitabilityEvidenceCanonical ==
               persistence.result.profitabilityEvidenceCanonical);
        assert(detail->profitabilityEvidenceHash ==
               persistence.result.profitabilityEvidenceHash);
        assert(detail->finalScore == persistence.result.finalScore);

        // A later input cannot reinterpret the persisted evaluation.
        persistence.input.finalProfitabilityEvidence
            ->profitabilityObservationId = 201;
        persistence.result = EvaluateExperimentRecommendation(
            runRequest.policy, persistence.input);
        persistence.result.rankingOrdinal = 1;
        bool mismatch = false;
        try
        {
            (void)PersistRecommendationEvaluation(runtime, persistence);
        }
        catch (const std::runtime_error& error)
        {
            mismatch = std::string{error.what()} ==
                "recommendation_evaluation_retry_mismatch";
        }
        assert(mismatch);
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
