#include "../Sources/ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignPlanningService.hpp"
#include "../Sources/ExperimentRecommendationCampaignReviewService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <pqxx/pqxx>

using namespace EA::ExperimentRecommendation;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

bool SafeDisposableDatabaseName(const std::string& value)
{
    if (value.empty()) return false;
    std::string lower = value;
    std::transform(
        lower.begin(), lower.end(), lower.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lower == "lstm") return false;
    return std::all_of(
        value.begin(), value.end(),
        [](unsigned char ch)
        {
            return std::isalnum(ch) != 0 || ch == '_' || ch == '-';
        });
}

void InsertRecommendation(
    pqxx::transaction_base& transaction,
    long long id,
    const std::string& symbol,
    int horizon,
    double leader,
    double accuracy,
    std::optional<double> neutral)
{
    const std::string semantic = "campaign-semantic-" + std::to_string(id);
    const std::string invocation = "campaign-invocation-" + std::to_string(id);
    transaction.exec(R"SQL(
INSERT INTO experiment_recommendation(
    recommendation_id,source_experiment_id,source_symbol,
    source_prediction_horizon,changed_parameter,source_leader_score,
    source_infer_accuracy,source_predicted_neutral_proportion,
    semantic_configuration_canonical,semantic_hash,
    invocation_configuration_canonical,invocation_hash) VALUES(
    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12);
)SQL", pqxx::params{
        id, 500 + id, symbol, horizon, "core_lr_mult", leader, accuracy,
        neutral, semantic, RecommendationCanonicalHash(semantic), invocation,
        RecommendationCanonicalHash(invocation)});
}

void InsertMember(
    pqxx::transaction_base& transaction,
    long long memberId,
    long long recommendationId,
    long long sourceExperimentId,
    int ordinal,
    const std::string& bucket,
    std::optional<double> score,
    const std::string& symbol,
    int horizon)
{
    transaction.exec(R"SQL(
INSERT INTO experiment_recommendation_ranking_member VALUES(
    $1,7,$2,$3,$4,$5,$6,$7,$8,'core_lr_mult');
)SQL", pqxx::params{
        memberId, recommendationId, sourceExperimentId, ordinal, bucket,
        score, symbol, horizon});
}

} // namespace

int main()
{
    const char* testDatabase = std::getenv("LSTM_TEST_DB_NAME");
    if (testDatabase == nullptr || *testDatabase == '\0')
    {
        std::cerr << "LSTM_TEST_DB_NAME_required\n";
        return 2;
    }
    const std::string database = testDatabase;
    if (!SafeDisposableDatabaseName(database))
    {
        std::cerr << "active_LSTM_database_forbidden\n";
        return 2;
    }
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "phase4d_campaign_planning_" + std::to_string(getpid());
    const std::string ownerConnectionString =
        "host=" + host + " port=" + port + " user=" + ownerUser +
        " dbname=" + database;
    const std::string runtimeConnectionString =
        "host=" + host + " port=" + port + " user=pqxx dbname=" + database +
        " options='-c search_path=" + schema + "'";
    pqxx::connection owner{ownerConnectionString};

    try
    {
        {
            pqxx::work setup{owner};
            setup.exec("CREATE SCHEMA " + setup.quote_name(schema) + ";");
            setup.exec("SET LOCAL search_path TO " +
                       setup.quote_name(schema) + ";");
            setup.exec(R"SQL(
CREATE SEQUENCE campaign_read_sentinel;
CREATE TABLE experiment_recommendation_ranking_snapshot(
    recommendation_ranking_snapshot_id bigint PRIMARY KEY,
    status text NOT NULL,
    ranking_snapshot_identity_canonical text NOT NULL,
    ranking_snapshot_identity_hash text NOT NULL,
    ranking_policy_canonical text NOT NULL,
    ranking_policy_hash text NOT NULL,
    ranking_version integer NOT NULL,
    population_semantic_state text NOT NULL,
    scoring_semantic_canonical text,
    scoring_semantic_hash text,
    scoring_semantic_version integer,
    evaluation_semantic_canonical text,
    evaluation_semantic_hash text,
    evaluation_semantic_version integer,
    distinct_scoring_semantic_count integer NOT NULL,
    distinct_evaluation_semantic_count integer NOT NULL,
    homogeneity_validation_result text NOT NULL,
    member_count integer NOT NULL);
CREATE TABLE experiment_recommendation_ranking_member(
    recommendation_ranking_member_id bigint PRIMARY KEY,
    recommendation_ranking_snapshot_id bigint NOT NULL,
    recommendation_id bigint NOT NULL,
    source_experiment_id bigint NOT NULL,
    global_ordinal integer NOT NULL,
    bucket text NOT NULL,
    final_score double precision,
    symbol text NOT NULL,
    horizon integer NOT NULL,
    family text NOT NULL);
CREATE TABLE experiment_recommendation(
    recommendation_id bigint PRIMARY KEY,
    source_experiment_id bigint NOT NULL,
    source_symbol text NOT NULL,
    source_prediction_horizon integer NOT NULL,
    changed_parameter text NOT NULL,
    source_leader_score double precision NOT NULL,
    source_infer_accuracy double precision NOT NULL,
    source_predicted_neutral_proportion double precision,
    semantic_configuration_canonical text NOT NULL,
    semantic_hash text NOT NULL,
    invocation_configuration_canonical text NOT NULL,
    invocation_hash text NOT NULL,
    final_profitability_provenance_version integer,
    source_final_inference_eval_result_id bigint,
    source_final_profitability_observation_id bigint,
    source_final_profitability_unavailable_reason text,
    source_final_profitability_inference_scope text,
    source_final_profitability_inference_start text,
    source_final_profitability_inference_end text,
    source_final_profitability_actionable_count bigint,
    source_final_profitability_aggregate_return double precision,
    source_final_profitability_average_return double precision,
    source_final_profitability_metric_definition_hash text,
    source_final_profitability_source_content_hash text,
    source_final_profitability_observation_identity_hash text);
CREATE TABLE experiment(
    experiment_id bigint PRIMARY KEY,
    target_epochs integer,
    status text NOT NULL,
    phase text NOT NULL,
    worker_pid integer,
    current_operation text,
    current_epoch integer,
    updated_at timestamptz NOT NULL);
CREATE TABLE experiment_recommendation_conversion_proposal(
    recommendation_conversion_proposal_id bigint PRIMARY KEY,
    recommendation_id bigint NOT NULL,
    source_experiment_id bigint NOT NULL,
    conversion_contract_version integer NOT NULL,
    conversion_identity_canonical text NOT NULL,
    conversion_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL);
CREATE TABLE experiment_recommendation_conversion_review_decision(
    recommendation_conversion_review_decision_id bigint PRIMARY KEY,
    recommendation_conversion_proposal_id bigint NOT NULL,
    decision text NOT NULL,
    decided_at timestamptz NOT NULL);
CREATE TABLE experiment_recommendation_conversion_execution(
    recommendation_conversion_execution_id bigint PRIMARY KEY,
    recommendation_conversion_proposal_id bigint NOT NULL,
    recommendation_conversion_review_decision_id bigint NOT NULL,
    experiment_id bigint NOT NULL,
    execution_contract_version integer NOT NULL,
    authorization_decision text NOT NULL,
    execution_identity_canonical text NOT NULL,
    execution_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL);
)SQL");
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                       setup.quote_name(schema) + " TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(!RecommendationCampaignPlanningSchemasExist(runtime));

        {
            pqxx::work fixtures{owner};
            fixtures.exec("SET LOCAL search_path TO " +
                          fixtures.quote_name(schema) + ";");
            fixtures.exec(R"SQL(
CREATE TABLE experiment_recommendation_conversion_activation(
    recommendation_conversion_activation_id bigint PRIMARY KEY,
    recommendation_conversion_execution_id bigint NOT NULL,
    recommendation_conversion_proposal_id bigint NOT NULL,
    recommendation_conversion_review_decision_id bigint NOT NULL,
    experiment_id bigint NOT NULL,
    activation_contract_version integer NOT NULL,
    previous_status text NOT NULL,
    previous_phase text NOT NULL,
    resulting_status text NOT NULL,
    resulting_phase text NOT NULL,
    activation_identity_canonical text NOT NULL,
    activation_identity_hash text NOT NULL,
    created_at timestamptz NOT NULL);
)SQL");
            fixtures.exec(
                "GRANT SELECT ON experiment_recommendation_conversion_activation "
                "TO pqxx;");
            const std::string snapshotCanonical = "campaign-snapshot";
            const std::string rankingPolicyCanonical = "campaign-ranking-policy";
            const std::string scoringSemanticCanonical =
                "campaign-scoring-semantic";
            const std::string evaluationSemanticCanonical =
                "campaign-evaluation-semantic";
            fixtures.exec(R"SQL(
INSERT INTO experiment_recommendation_ranking_snapshot VALUES(
    7,'completed',$1,$2,$3,$4,1,'verified_homogeneous',$5,$6,1,$7,$8,1,
    1,1,'verified_homogeneous',4);
)SQL", pqxx::params{
                snapshotCanonical, RecommendationCanonicalHash(snapshotCanonical),
                rankingPolicyCanonical,
                RecommendationCanonicalHash(rankingPolicyCanonical),
                scoringSemanticCanonical,
                RecommendationCanonicalHash(scoringSemanticCanonical),
                evaluationSemanticCanonical,
                RecommendationCanonicalHash(evaluationSemanticCanonical)});

            for (long long id = 1; id <= 4; ++id)
            {
                fixtures.exec(
                    "INSERT INTO experiment VALUES($1,120,'completed','analyze',"
                    "NULL,NULL,NULL,'2026-07-01 00:00:00+00');",
                    pqxx::params{500 + id});
            }
            InsertRecommendation(fixtures, 1, "eurusd", 12, 0.90, 0.80, 0.20);
            InsertRecommendation(fixtures, 2, "eurusd", 12, 0.85, 0.75, 0.25);
            InsertRecommendation(fixtures, 3, "gbpusd", 24, 0.40, 0.70, 0.30);
            InsertRecommendation(fixtures, 4, "gbpusd", 24, 0.80, 0.70, 0.30);
            InsertMember(fixtures, 101, 1, 501, 1, "advisory_ready", 0.9,
                         "eurusd", 12);
            InsertMember(fixtures, 102, 2, 502, 2, "advisory_ready", 0.8,
                         "eurusd", 12);
            InsertMember(fixtures, 103, 3, 503, 3, "advisory_ready", 0.7,
                         "gbpusd", 24);
            InsertMember(fixtures, 104, 4, 504, 4, "blocked", std::nullopt,
                         "gbpusd", 24);

            const std::string proposalCanonical = "campaign-proposal-2";
            fixtures.exec(R"SQL(
INSERT INTO experiment_recommendation_conversion_proposal VALUES(
    200,2,502,1,$1,$2,now());
)SQL", pqxx::params{
                proposalCanonical, RecommendationCanonicalHash(proposalCanonical)});
            fixtures.commit();
        }

        assert(RecommendationCampaignPlanningSchemasExist(runtime));
        RecommendationCampaignPlanningPolicy policy;
        policy.enabled = true;
        policy.minimumLeaderScore = 0.5;
        RecommendationCampaignPlanningScope scope;
        scope.rankingSnapshotId = 7;

        const auto input = LoadRecommendationCampaignPlanInput(
            runtime, policy, scope);
        assert(input.candidates.size() == 4);
        assert(input.candidates[0].recommendationId == 1);
        assert(input.candidates[1].workflows.size() == 1);
        assert(input.candidates[1].workflows[0].state ==
               RecommendationConversionWorkflowState::pendingReview);
        const auto plan = PlanRecommendationCampaign(input);
        assert(plan.summary.candidateCount == 4);
        assert(plan.summary.selectedCount == 1);
        assert(plan.candidates[0].decision ==
               RecommendationCampaignDecision::include);

        scope.symbol = "eurusd";
        const auto filtered = LoadRecommendationCampaignPlanInput(
            runtime, policy, scope);
        assert(filtered.candidates.size() == 2);

        std::ostringstream output;
        std::ostringstream errors;
        assert(RunRecommendationCampaignPlanningCommand(
            runtimeConnectionString, policy, scope, output, errors) == 0);
        assert(output.str().find("RECOMMENDATION_CAMPAIGN_PLAN") !=
               std::string::npos);
        assert(output.str().find("read_only=true") != std::string::npos);
        assert(output.str().find("proposal_created=false") !=
               std::string::npos);
        assert(errors.str().empty());

        output.str({});
        output.clear();
        assert(RunRecommendationCampaignReviewCommand(
            runtimeConnectionString, policy, scope, output, errors) == 0);
        assert(output.str().find("RECOMMENDATION_CAMPAIGN_REVIEW") !=
               std::string::npos);
        assert(output.str().find(
            "deterministic_ordering_verified=true") != std::string::npos);
        assert(output.str().find(
            "RECOMMENDATION_CAMPAIGN_REVIEW_FAMILY") != std::string::npos);
        assert(output.str().find("read_only=true") != std::string::npos);
        assert(errors.str().empty());

        {
            pqxx::work invalidate{owner};
            invalidate.exec("SET LOCAL search_path TO " +
                            invalidate.quote_name(schema) + ";");
            invalidate.exec(
                "UPDATE experiment_recommendation_ranking_snapshot SET "
                "population_semantic_state='legacy_unverified',"
                "scoring_semantic_canonical=NULL,scoring_semantic_hash=NULL,"
                "scoring_semantic_version=NULL,evaluation_semantic_canonical=NULL,"
                "evaluation_semantic_hash=NULL,evaluation_semantic_version=NULL,"
                "distinct_scoring_semantic_count=0,"
                "distinct_evaluation_semantic_count=0,"
                "homogeneity_validation_result='legacy_unverified' "
                "WHERE recommendation_ranking_snapshot_id=7;");
            invalidate.commit();
        }
        bool unverifiedRejected = false;
        try
        {
            (void)LoadRecommendationCampaignPlanInput(runtime, policy, scope);
        }
        catch (const std::runtime_error& error)
        {
            unverifiedRejected = std::string{error.what()} ==
                "recommendation_campaign_ranking_semantics_not_verified";
        }
        assert(unverifiedRejected);

        scope.rankingSnapshotId = 999;
        output.str({});
        output.clear();
        assert(RunRecommendationCampaignPlanningCommand(
            runtimeConnectionString, policy, scope, output, errors) == 1);
        assert(errors.str().find(
            "RECOMMENDATION_CAMPAIGN_RANKING_SNAPSHOT_NOT_FOUND") !=
            std::string::npos);

        pqxx::read_transaction verify{owner};
        verify.exec("SET LOCAL search_path TO " + verify.quote_name(schema) + ";");
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_proposal;")
                   .one_row()[0].as<int>() == 1);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_review_decision;")
                   .one_row()[0].as<int>() == 0);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_execution;")
                   .one_row()[0].as<int>() == 0);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_activation;")
                   .one_row()[0].as<int>() == 0);
        assert(verify.exec("SELECT count(*) FROM experiment;")
                   .one_row()[0].as<int>() == 4);
        assert(verify.exec(
            "SELECT count(*) FROM experiment WHERE updated_at <> "
            "'2026-07-01 00:00:00+00'::timestamptz;")
                   .one_row()[0].as<int>() == 0);
        const pqxx::row sequence = verify.exec(
            "SELECT last_value,is_called FROM campaign_read_sentinel;").one_row();
        assert(sequence["last_value"].as<long long>() == 1);
        assert(!sequence["is_called"].as<bool>());
    }
    catch (...)
    {
        pqxx::work cleanup{owner};
        cleanup.exec("DROP SCHEMA IF EXISTS " + cleanup.quote_name(schema) +
                     " CASCADE;");
        cleanup.commit();
        throw;
    }

    pqxx::work cleanup{owner};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE;");
    cleanup.commit();
}
