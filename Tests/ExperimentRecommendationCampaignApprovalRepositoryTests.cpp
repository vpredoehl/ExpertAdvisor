#include "../Sources/ExperimentRecommendationCampaignApprovalRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignApprovalService.hpp"
#include "../Sources/ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignReview.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <algorithm>
#include <barrier>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
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

std::string ReadFile(const std::string& path)
{
    std::ifstream input{path};
    if (!input) throw std::runtime_error("unable_to_read_" + path);
    return {std::istreambuf_iterator<char>{input},
            std::istreambuf_iterator<char>{}};
}

bool SafeDisposableDatabaseName(const std::string& value)
{
    if (value.empty()) return false;
    std::string lower = value;
    std::transform(
        lower.begin(), lower.end(), lower.begin(),
        [](unsigned char value)
        { return static_cast<char>(std::tolower(value)); });
    if (lower == "lstm") return false;
    return std::all_of(
        value.begin(), value.end(),
        [](unsigned char value)
        { return std::isalnum(value) || value == '_' || value == '-'; });
}

bool PermissionDenied(const pqxx::sql_error& error)
{
    if (error.sqlstate() == "42501") return true;
    return error.sqlstate().empty() &&
           std::string(error.what()).find("permission denied") !=
               std::string::npos;
}

RecommendationCampaignPlanningPolicy Policy()
{
    RecommendationCampaignPlanningPolicy policy;
    policy.enabled = true;
    policy.maximumPerSourceExperiment.reset();
    return policy;
}

RecommendationCampaignReview LoadReview(
    pqxx::connection& connection,
    const RecommendationCampaignPlanningPolicy& policy)
{
    RecommendationCampaignPlanningScope scope;
    scope.rankingSnapshotId = 7;
    const auto input = LoadRecommendationCampaignPlanInput(
        connection, policy, scope);
    return ReviewRecommendationCampaignPlan(PlanRecommendationCampaign(input));
}

RecommendationCampaignApprovalRequest Request(
    const RecommendationCampaignReview& review,
    RecommendationCampaignApprovalDecision decision,
    const std::string& reviewer = "campaign-test-operator",
    const std::string& reason = "Explicit isolated campaign decision.")
{
    RecommendationCampaignApprovalRequest request;
    request.decision = decision;
    request.expectedCampaignReviewIdentityHash = review.identityHash;
    request.reviewerIdentity = reviewer;
    request.reasonText = reason;
    return request;
}

void InsertRecommendation(
    pqxx::transaction_base& transaction,
    long long id,
    double leader,
    int ordinal,
    const std::string& bucket,
    std::optional<double> score)
{
    const std::string semantic = "approval-semantic-" + std::to_string(id);
    const std::string invocation = "approval-invocation-" + std::to_string(id);
    transaction.exec(
        "INSERT INTO experiment_recommendation("
        "recommendation_id,source_experiment_id,source_symbol,"
        "source_prediction_horizon,changed_parameter,source_leader_score,"
        "source_infer_accuracy,source_predicted_neutral_proportion,"
        "semantic_configuration_canonical,semantic_hash,"
        "invocation_configuration_canonical,invocation_hash) VALUES("
        "$1,$2,'eurusd',12,'core_lr_mult',$3,0.8,0.2,$4,$5,$6,$7);",
        pqxx::params{
            id, 500 + id, leader, semantic,
            RecommendationCanonicalHash(semantic), invocation,
            RecommendationCanonicalHash(invocation)});
    transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_member VALUES("
        "$1,7,$2,$3,$4,$5,$6,'eurusd',12,'core_lr_mult');",
        pqxx::params{100 + id, id, 500 + id, ordinal, bucket, score});
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
        "phase4d_campaign_approval_" + std::to_string(getpid());
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
CREATE SEQUENCE campaign_approval_read_sentinel;
CREATE TABLE experiment_recommendation_ranking_snapshot(
    recommendation_ranking_snapshot_id bigint PRIMARY KEY,
    status text NOT NULL,
    ranking_snapshot_identity_canonical text NOT NULL,
    ranking_snapshot_identity_hash text NOT NULL,
    ranking_policy_canonical text NOT NULL,
    ranking_policy_hash text NOT NULL,
    ranking_version integer NOT NULL);
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
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                       setup.quote_name(schema) + " TO pqxx;");

            const std::string migration = ReadFile(
                "Database/migrations/040_experiment_recommendation_campaign_approval.sql");
            setup.exec(migration);
            setup.exec(migration);
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationCampaignApprovalMigrationTests.sql"));

            const std::string snapshotCanonical = "approval-snapshot";
            const std::string rankingPolicyCanonical = "approval-ranking-policy";
            setup.exec(
                "INSERT INTO experiment_recommendation_ranking_snapshot "
                "VALUES(7,'completed',$1,$2,$3,$4,1);",
                pqxx::params{
                    snapshotCanonical,
                    RecommendationCanonicalHash(snapshotCanonical),
                    rankingPolicyCanonical,
                    RecommendationCanonicalHash(rankingPolicyCanonical)});
            setup.exec(
                "INSERT INTO experiment VALUES"
                "(501,120,'completed','analyze',NULL,NULL,NULL,"
                "'2026-07-01 00:00:00+00'),"
                "(502,120,'completed','analyze',NULL,NULL,NULL,"
                "'2026-07-01 00:00:00+00');");
            InsertRecommendation(setup, 1, 0.90, 1, "advisory_ready", 0.9);
            InsertRecommendation(setup, 2, 0.85, 2, "blocked", std::nullopt);
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(RecommendationCampaignApprovalSchemaExists(runtime));
        RecommendationCampaignPlanningScope scope;
        scope.rankingSnapshotId = 7;
        const auto policy = Policy();
        const auto review = LoadReview(runtime, policy);
        assert(review.summary.selectedCount == 1);

        std::ostringstream output;
        std::ostringstream errors;
        const auto approved = Request(
            review, RecommendationCampaignApprovalDecision::approved);
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, policy, scope, approved,
            output, errors) == 0);
        assert(output.str().find("RECOMMENDATION_CAMPAIGN_APPROVAL_RECORDED") !=
               std::string::npos);
        assert(output.str().find("campaign_approval_recorded=true") !=
               std::string::npos);
        assert(output.str().find("experiment_created=false") !=
               std::string::npos);
        assert(errors.str().empty());

        const auto persisted = FindRecommendationCampaignApprovalByReviewIdentity(
            runtime, review.identityCanonical);
        assert(persisted);
        assert(persisted->evidence.campaignReviewIdentityCanonical ==
               review.identityCanonical);
        assert(persisted->evidence.campaignPlanIdentityCanonical ==
               review.campaignPlanIdentityCanonical);
        assert(persisted->evidence.summary.selectedCount == 1);

        output.str({});
        output.clear();
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, policy, scope, approved,
            output, errors) == 0);
        assert(output.str().find(
            "RECOMMENDATION_CAMPAIGN_APPROVAL_ALREADY_RECORDED") !=
               std::string::npos);
        assert(output.str().find("campaign_approval_recorded=false") !=
               std::string::npos);

        auto conflict = approved;
        conflict.reasonText = "A changed reason must conflict.";
        output.str({});
        output.clear();
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, policy, scope, conflict,
            output, errors) == 3);
        assert(errors.str().find("RECOMMENDATION_CAMPAIGN_APPROVAL_CONFLICT") !=
               std::string::npos);
        errors.str({});
        errors.clear();

        conflict = approved;
        conflict.reviewerIdentity = "different-operator";
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, policy, scope, conflict,
            output, errors) == 3);
        errors.str({});
        errors.clear();

        auto reversed = approved;
        reversed.decision = RecommendationCampaignApprovalDecision::rejected;
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, policy, scope, reversed,
            output, errors) == 3);
        errors.str({});
        errors.clear();

        auto stale = approved;
        stale.expectedCampaignReviewIdentityHash =
            "fnv1a64:0000000000000000";
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, policy, scope, stale,
            output, errors) == 1);
        assert(errors.str().find("RECOMMENDATION_CAMPAIGN_APPROVAL_STALE") !=
               std::string::npos);
        errors.str({});
        errors.clear();

        RecommendationCampaignPlanningScope missingScope = scope;
        missingScope.rankingSnapshotId = 999;
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, policy, missingScope, approved,
            output, errors) == 1);
        assert(errors.str().find(
            "RECOMMENDATION_CAMPAIGN_RANKING_SNAPSHOT_NOT_FOUND") !=
               std::string::npos);
        errors.str({});
        errors.clear();

        output.str({});
        output.clear();
        assert(RunShowRecommendationCampaignApprovalCommand(
            runtimeConnectionString, persisted->campaignApprovalId,
            output, errors) == 0);
        assert(output.str().find("read_only=true") != std::string::npos);
        output.str({});
        output.clear();
        assert(RunListRecommendationCampaignApprovalsCommand(
            runtimeConnectionString,
            RecommendationCampaignApprovalDecision::approved,
            10, output, errors) == 0);
        assert(output.str().find(
            "RECOMMENDATION_CAMPAIGN_APPROVAL_LIST_COMPLETE,count=1") !=
               std::string::npos);

        auto zeroPolicy = policy;
        zeroPolicy.minimumLeaderScore = 1.0;
        const auto zeroReview = LoadReview(runtime, zeroPolicy);
        assert(zeroReview.summary.selectedCount == 0);
        const auto zeroApproved = Request(
            zeroReview, RecommendationCampaignApprovalDecision::approved);
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, zeroPolicy, scope, zeroApproved,
            output, errors) == 1);
        errors.str({});
        errors.clear();
        const auto zeroRejected = Request(
            zeroReview, RecommendationCampaignApprovalDecision::rejected);
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, zeroPolicy, scope, zeroRejected,
            output, errors) == 0);

        auto rejectFirstPolicy = policy;
        rejectFirstPolicy.minimumInferenceAccuracy = 0.69;
        const auto rejectFirstReview = LoadReview(runtime, rejectFirstPolicy);
        const auto rejectFirst = Request(
            rejectFirstReview,
            RecommendationCampaignApprovalDecision::rejected,
            "reject-first-operator",
            "A deliberate rejection before attempted reversal.");
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, rejectFirstPolicy, scope, rejectFirst,
            output, errors) == 0);
        auto rejectFirstReversal = rejectFirst;
        rejectFirstReversal.decision =
            RecommendationCampaignApprovalDecision::approved;
        assert(RunRecordRecommendationCampaignApprovalCommand(
            runtimeConnectionString, rejectFirstPolicy, scope,
            rejectFirstReversal, output, errors) == 3);
        errors.str({});
        errors.clear();

        auto concurrentPolicy = policy;
        concurrentPolicy.maximumSelectedRecommendations = 1;
        concurrentPolicy.maximumCandidatesConsidered = 1;
        const auto concurrentReview = LoadReview(runtime, concurrentPolicy);
        const auto concurrentRequest = Request(
            concurrentReview,
            RecommendationCampaignApprovalDecision::approved,
            "concurrent-operator",
            "Concurrent identical request.");
        std::barrier start{2};
        int statusA = -1;
        int statusB = -1;
        std::string outputA;
        std::string outputB;
        auto runConcurrent = [&](int& status, std::string& text)
        {
            std::ostringstream localOutput;
            std::ostringstream localErrors;
            start.arrive_and_wait();
            status = RunRecordRecommendationCampaignApprovalCommand(
                runtimeConnectionString, concurrentPolicy, scope,
                concurrentRequest, localOutput, localErrors);
            text = localOutput.str() + localErrors.str();
        };
        std::thread first{runConcurrent, std::ref(statusA), std::ref(outputA)};
        std::thread second{runConcurrent, std::ref(statusB), std::ref(outputB)};
        first.join();
        second.join();
        assert(statusA == 0 && statusB == 0);
        const bool firstRecorded = outputA.find(
            "RECOMMENDATION_CAMPAIGN_APPROVAL_RECORDED,") !=
            std::string::npos;
        const bool secondRecorded = outputB.find(
            "RECOMMENDATION_CAMPAIGN_APPROVAL_RECORDED,") !=
            std::string::npos;
        const bool firstReplayed = outputA.find(
            "RECOMMENDATION_CAMPAIGN_APPROVAL_ALREADY_RECORDED,") !=
            std::string::npos;
        const bool secondReplayed = outputB.find(
            "RECOMMENDATION_CAMPAIGN_APPROVAL_ALREADY_RECORDED,") !=
            std::string::npos;
        assert((firstRecorded && secondReplayed) ||
               (secondRecorded && firstReplayed));

        {
            pqxx::work rollback{runtime};
            const auto rollbackPolicy = [&]
            {
                auto value = policy;
                value.maximumPredictedNeutralProportion = 0.7;
                return value;
            }();
            const auto input = LoadRecommendationCampaignPlanInput(
                rollback, rollbackPolicy, scope);
            const auto plan = PlanRecommendationCampaign(input);
            const auto rollbackReview = ReviewRecommendationCampaignPlan(plan);
            const auto evidence = BuildRecommendationCampaignApprovalEvidence(
                7, input.rankingSnapshotIdentityCanonical,
                input.rankingSnapshotIdentityHash, plan, rollbackReview,
                Request(
                    rollbackReview,
                    RecommendationCampaignApprovalDecision::approved,
                    "rollback-operator", "Rollback test."));
            (void)PersistRecommendationCampaignApproval(rollback, evidence);
        }

        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "UPDATE experiment_recommendation_campaign_approval "
                "SET reason_text='forged';");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "DELETE FROM experiment_recommendation_campaign_approval;");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "INSERT INTO experiment_recommendation_campaign_approval "
                "(recommendation_campaign_approval_id) VALUES(999);");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "INSERT INTO experiment_recommendation_campaign_approval "
                "(created_at) VALUES(now());");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }
        try
        {
            pqxx::work forbidden{runtime};
            forbidden.exec(
                "TRUNCATE experiment_recommendation_campaign_approval;");
            assert(false);
        }
        catch (const pqxx::sql_error& error)
        {
            assert(PermissionDenied(error));
        }

        pqxx::read_transaction verify{owner};
        verify.exec("SET LOCAL search_path TO " +
                    verify.quote_name(schema) + ";");
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_campaign_approval;")
                   .one_row()[0].as<int>() == 4);
        assert(verify.exec("SELECT count(*) FROM experiment;")
                   .one_row()[0].as<int>() == 2);
        assert(verify.exec(
            "SELECT count(*) FROM experiment WHERE updated_at <> "
            "'2026-07-01 00:00:00+00'::timestamptz;")
                   .one_row()[0].as<int>() == 0);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_proposal;")
                   .one_row()[0].as<int>() == 0);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_review_decision;")
                   .one_row()[0].as<int>() == 0);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_execution;")
                   .one_row()[0].as<int>() == 0);
        assert(verify.exec(
            "SELECT count(*) FROM experiment_recommendation_conversion_activation;")
                   .one_row()[0].as<int>() == 0);
        const pqxx::row sentinel = verify.exec(
            "SELECT last_value,is_called FROM campaign_approval_read_sentinel;")
            .one_row();
        assert(sentinel["last_value"].as<long long>() == 1);
        assert(!sentinel["is_called"].as<bool>());
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
    return 0;
}
