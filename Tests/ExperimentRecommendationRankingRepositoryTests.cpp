#include "../Sources/ExperimentRecommendationRankingRepository.hpp"
#include "../Sources/ExperimentRecommendationRankingService.hpp"

#include <cassert>
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

long long InsertEvaluation(pqxx::work& transaction,
                           long long runId,
                           long long recommendationId,
                           long long scanId,
                           long long experimentId,
                           long long analysisId,
                           int ordinal,
                           const std::string& disposition,
                           std::optional<double> score)
{
    const bool ready = score.has_value();
    const long long id = transaction.exec(
        "INSERT INTO experiment_recommendation_evaluation_result ("
        "recommendation_evaluation_run_id,recommendation_id,"
        "evaluation_identity_canonical,evaluation_identity_hash,"
        "recommendation_semantic_canonical,recommendation_semantic_hash,"
        "recommendation_policy_canonical,recommendation_policy_hash,"
        "recommendation_scan_id,source_experiment_id,source_model_id,"
        "source_analysis_id,evidence_canonical,evidence_hash,eligibility,"
        "disposition,reason_code,explanation,final_score,raw_positive_score,"
        "raw_penalty_score,raw_total_score,component_count,"
        "missing_evidence_count,ranking_ordinal) VALUES ($1,$2,$3,$4,$5,$6,"
        "'recommendation_policy','recommendation_policy_hash',$7,$8,1,$9,"
        "$10,$11,$12,$13,$13,'explanation',$14,$15,$16,$17,$18,$19,$20) "
        "RETURNING recommendation_evaluation_result_id;",
        pqxx::params{runId, recommendationId,
            "evaluation_identity_" + std::to_string(ordinal),
            "evaluation_hash_" + std::to_string(ordinal),
            "semantic_" + std::to_string(ordinal),
            "semantic_hash_" + std::to_string(ordinal), scanId, experimentId,
            analysisId, "evidence_" + std::to_string(ordinal),
            "evidence_hash_" + std::to_string(ordinal),
            ready ? "eligible" : "ineligible", disposition, score,
            ready ? std::optional<double>{1.0} : std::nullopt,
            ready ? std::optional<double>{0.2} : std::nullopt,
            ready ? score : std::nullopt, ready ? 2 : 0,
            disposition == "insufficient_evidence" ? 1 : 0, ordinal})
        .one_row()[0].as<long long>();
    if (ready)
    {
        transaction.exec(
            "INSERT INTO experiment_recommendation_evaluation_component ("
            "recommendation_evaluation_result_id,component_ordinal,"
            "component_name,reason_code,input_canonical,normalized_value,"
            "weight,weighted_contribution,is_penalty,is_missing,explanation) "
            "VALUES ($1,1,'leader_quality','quality','input',0.8,0.2,0.16,"
            "false,false,'Quality.'),($1,2,'horizon_change_penalty','penalty',"
            "'input',0.5,0.1,0.05,true,false,'Penalty.');",
            pqxx::params{id});
    }
    return id;
}

std::string SourceDigest(pqxx::connection& connection,
                         const std::string& schema)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
    return transaction.exec(
        "SELECT concat_ws(':',"
        "(SELECT md5(coalesce(string_agg(recommendation_evaluation_run_id::text"
        "||status||evaluation_run_identity_canonical,'|' ORDER BY "
        "recommendation_evaluation_run_id),'')) FROM "
        "experiment_recommendation_evaluation_run),"
        "(SELECT md5(coalesce(string_agg(recommendation_evaluation_result_id::text"
        "||evaluation_identity_canonical,'|' ORDER BY "
        "recommendation_evaluation_result_id),'')) FROM "
        "experiment_recommendation_evaluation_result),"
        "(SELECT md5(coalesce(string_agg(recommendation_evaluation_component_id::text"
        "||component_name,'|' ORDER BY recommendation_evaluation_component_id),''))"
        " FROM experiment_recommendation_evaluation_component),"
        "(SELECT md5(coalesce(string_agg(recommendation_id::text||status,'|' ORDER BY"
        " recommendation_id),'')) FROM experiment_recommendation),"
        "(SELECT md5(coalesce(string_agg(experiment_id::text||marker,'|' ORDER BY"
        " experiment_id),'')) FROM experiment),"
        "(SELECT md5(coalesce(string_agg(model_id::text||marker,'|' ORDER BY"
        " model_id),'')) FROM model),"
        "(SELECT md5(coalesce(string_agg(analysis_id::text||marker,'|' ORDER BY"
        " analysis_id),'')) FROM experiment_analysis_result),"
        "(SELECT md5(coalesce(string_agg(inference_id::text||marker,'|' ORDER BY"
        " inference_id),'')) FROM inference_evidence),"
        "(SELECT md5(coalesce(string_agg(checkpoint_id::text||marker,'|' ORDER BY"
        " checkpoint_id),'')) FROM checkpoint_evidence),"
        "(SELECT md5(coalesce(string_agg(continuation_id::text||marker,'|' ORDER BY"
        " continuation_id),'')) FROM continuation_evidence),"
        "(SELECT count(*)::text FROM experiment_recommendation_review_event));")
        .one_row()[0].as<std::string>();
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
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string owner = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema = "phase4b_rank_" + std::to_string(getpid());
    const std::string ownerConnectionString =
        "host=" + host + " port=" + port + " user=" + owner +
        " dbname=" + database;
    const std::string runtimeConnectionString =
        "host=" + host + " port=" + port + " user=pqxx dbname=" + database +
        " options='-c search_path=" + schema + "'";
    pqxx::connection ownerConnection{ownerConnectionString};

    try
    {
        {
            pqxx::work setup{ownerConnection};
            setup.exec("CREATE SCHEMA " + setup.quote_name(schema) + ";");
            setup.exec("SET LOCAL search_path TO " + setup.quote_name(schema) + ";");
            setup.exec("CREATE TABLE experiment(experiment_id bigserial PRIMARY KEY,"
                       "marker text NOT NULL);");
            setup.exec("CREATE TABLE model(model_id bigint PRIMARY KEY,marker text NOT NULL);");
            setup.exec("CREATE TABLE experiment_analysis_result(analysis_id bigint "
                       "PRIMARY KEY,marker text NOT NULL);");
            setup.exec("CREATE TABLE inference_evidence(inference_id bigint "
                       "PRIMARY KEY,marker text NOT NULL);");
            setup.exec("CREATE TABLE checkpoint_evidence(checkpoint_id bigint "
                       "PRIMARY KEY,marker text NOT NULL);");
            setup.exec("CREATE TABLE continuation_evidence(continuation_id bigint "
                       "PRIMARY KEY,marker text NOT NULL);");
            setup.exec("CREATE TABLE experiment_recommendation_scan("
                       "recommendation_scan_id bigserial PRIMARY KEY);");
            setup.exec("CREATE TABLE experiment_recommendation("
                       "recommendation_id bigserial PRIMARY KEY,"
                       "recommendation_scan_id bigint NOT NULL REFERENCES "
                       "experiment_recommendation_scan,source_experiment_id bigint "
                       "NOT NULL REFERENCES experiment,source_analysis_id bigint "
                       "REFERENCES experiment_analysis_result,status text NOT NULL,"
                       "source_symbol text NOT NULL,source_prediction_horizon integer "
                       "NOT NULL,changed_parameter text NOT NULL,"
                       "source_value_canonical text NOT NULL,"
                       "proposed_value_canonical text NOT NULL);");
            setup.exec("CREATE TABLE experiment_recommendation_review_event("
                       "recommendation_review_event_id bigserial PRIMARY KEY);");
            setup.exec(ReadFile(
                "Database/migrations/034_experiment_recommendation_evaluation.sql"));
            setup.exec(ReadFile(
                "Database/migrations/035_experiment_recommendation_ranking.sql"));
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx;");
            setup.exec("GRANT SELECT ON experiment,model,"
                       "experiment_analysis_result,inference_evidence,"
                       "checkpoint_evidence,continuation_evidence,"
                       "experiment_recommendation_scan,"
                       "experiment_recommendation,"
                       "experiment_recommendation_review_event TO pqxx;");
            setup.commit();
        }

        long long runId = -1;
        long long scanId = -1;
        std::vector<long long> evaluationIds;
        {
            pqxx::work fixture{ownerConnection};
            fixture.exec("SET LOCAL search_path TO " +
                         fixture.quote_name(schema) + ";");
            fixture.exec("INSERT INTO experiment(marker) VALUES ('unchanged');"
                         "INSERT INTO model VALUES (1,'unchanged');"
                         "INSERT INTO experiment_analysis_result VALUES "
                         "(1,'unchanged');"
                         "INSERT INTO inference_evidence VALUES (1,'unchanged');"
                         "INSERT INTO checkpoint_evidence VALUES (1,'unchanged');"
                         "INSERT INTO continuation_evidence VALUES "
                         "(1,'unchanged');");
            scanId = fixture.exec(
                "INSERT INTO experiment_recommendation_scan DEFAULT VALUES "
                "RETURNING recommendation_scan_id;").one_row()[0].as<long long>();
            runId = fixture.exec(
                "INSERT INTO experiment_recommendation_evaluation_run(status,"
                "evaluation_run_identity_canonical,evaluation_run_identity_hash,"
                "evaluation_policy_canonical,evaluation_policy_hash,"
                "evaluation_version,evaluator_version,scoring_policy_canonical,"
                "scoring_policy_hash,scoring_version,evidence_snapshot_canonical,"
                "evidence_snapshot_hash,completed_at) VALUES ('completed','run',"
                "'run_hash','evaluation_policy','evaluation_policy_hash',1,1,"
                "'scoring_policy','scoring_policy_hash',1,'evidence',"
                "'evidence_hash',now()) RETURNING recommendation_evaluation_run_id;")
                .one_row()[0].as<long long>();
            const char* families[] = {"core_lr_mult", "core_lr_mult",
                                      "core_lr_mult", "core_lr_mult",
                                      "head_lr_mult"};
            for (int index = 0; index < 5; ++index)
            {
                const long long recommendationId = fixture.exec(
                    "INSERT INTO experiment_recommendation("
                    "recommendation_scan_id,source_experiment_id,source_analysis_id,"
                    "status,source_symbol,source_prediction_horizon,changed_parameter,"
                    "source_value_canonical,proposed_value_canonical) VALUES "
                    "($1,1,1,'proposed',$2,12,$3,'1',$4) RETURNING recommendation_id;",
                    pqxx::params{scanId, index == 4 ? "GB,PUSD" : "EURUSD",
                                 families[index], std::to_string(index + 2)})
                    .one_row()[0].as<long long>();
                if (index == 0)
                    evaluationIds.push_back(InsertEvaluation(fixture, runId,
                        recommendationId, scanId, 1, 1, index + 1,
                        "advisory_ready", 0.9));
                else if (index == 1)
                    evaluationIds.push_back(InsertEvaluation(fixture, runId,
                        recommendationId, scanId, 1, 1, index + 1,
                        "advisory_ready", 0.8));
                else if (index == 2)
                    evaluationIds.push_back(InsertEvaluation(fixture, runId,
                        recommendationId, scanId, 1, 1, index + 1,
                        "blocked_pending_duplicate", std::nullopt));
                else if (index == 3)
                    evaluationIds.push_back(InsertEvaluation(fixture, runId,
                        recommendationId, scanId, 1, 1, index + 1,
                        "stale_source_evidence", std::nullopt));
                else
                    evaluationIds.push_back(InsertEvaluation(fixture, runId,
                        recommendationId, scanId, 1, 1, index + 1,
                        "advisory_ready", 0.7));
            }
            fixture.commit();
        }

        const std::string before = SourceDigest(ownerConnection, schema);
        pqxx::connection runtime{runtimeConnectionString};
        assert(RecommendationRankingSchemaExists(runtime));
        RecommendationRankingScope runScope;
        runScope.type = RecommendationRankingScopeType::evaluationRun;
        runScope.evaluationRunId = runId;
        auto evaluations = LoadEvaluationsForRanking(runtime, runScope);
        assert(evaluations.size() == 5);
        assert(evaluations.front().components.size() <= 2);
        for (const auto& evaluation : evaluations)
            assert(evaluation.componentCount ==
                   static_cast<int>(evaluation.components.size()));
        RecommendationRankingScope scanScope;
        scanScope.type = RecommendationRankingScopeType::recommendationScan;
        scanScope.recommendationScanId = scanId;
        assert(LoadEvaluationsForRanking(runtime, scanScope).size() == 5);
        RecommendationRankingScope symbolScope;
        symbolScope.type = RecommendationRankingScopeType::symbol;
        symbolScope.symbol = "EURUSD";
        assert(LoadEvaluationsForRanking(runtime, symbolScope).size() == 4);
        RecommendationRankingScope horizonScope;
        horizonScope.type = RecommendationRankingScopeType::horizon;
        horizonScope.horizon = 12;
        assert(LoadEvaluationsForRanking(runtime, horizonScope).size() == 5);
        RecommendationRankingScope symbolHorizonScope;
        symbolHorizonScope.type = RecommendationRankingScopeType::symbolHorizon;
        symbolHorizonScope.symbol = "EURUSD";
        symbolHorizonScope.horizon = 12;
        assert(LoadEvaluationsForRanking(runtime, symbolHorizonScope).size() == 4);
        RecommendationRankingScope familyScope;
        familyScope.type = RecommendationRankingScopeType::family;
        familyScope.family = "head_lr_mult";
        assert(LoadEvaluationsForRanking(runtime, familyScope).size() == 1);

        RecommendationRankingCommandRequest request;
        request.scope = runScope;
        request.limit = 100;
        request.dryRun = true;
        std::ostringstream dryOutput;
        std::ostringstream dryErrors;
        assert(RunRankExperimentRecommendationEvaluationsCommand(
            runtimeConnectionString, request, dryOutput, dryErrors) == 0);
        assert(dryOutput.str().find("persisted=false") != std::string::npos);
        assert(dryOutput.str().find("advisory_ready:") != std::string::npos);
        assert(dryOutput.str().find("blocked:") != std::string::npos);
        assert(dryOutput.str().find("non_actionable:") != std::string::npos);
        RecommendationRankingCommandRequest escapedRequest = request;
        escapedRequest.scope.type = RecommendationRankingScopeType::symbol;
        escapedRequest.scope.evaluationRunId.reset();
        escapedRequest.scope.symbol = "E,URUSD";
        std::ostringstream escapedOutput;
        std::ostringstream escapedErrors;
        assert(RunRankExperimentRecommendationEvaluationsCommand(
            runtimeConnectionString, escapedRequest, escapedOutput,
            escapedErrors) == 0);
        assert(escapedOutput.str().find("scope_value=E%2CURUSD") !=
               std::string::npos);
        assert(escapedOutput.str().find("Scope: symbol=E,URUSD") !=
               std::string::npos);
        {
            pqxx::read_transaction check{runtime};
            assert(check.exec("SELECT count(*) FROM "
                "experiment_recommendation_ranking_snapshot;")
                .one_row()[0].as<int>() == 0);
        }

        request.dryRun = false;
        std::ostringstream output;
        std::ostringstream errors;
        assert(RunRankExperimentRecommendationEvaluationsCommand(
            runtimeConnectionString, request, output, errors) == 0);
        assert(errors.str().empty());
        assert(output.str().find("experiment_created=false") != std::string::npos);
        assert(output.str().find("tie_break_3=") != std::string::npos);
        assert(output.str().find("symbol=GB%2CPUSD") != std::string::npos);
        assert(output.str().find("GB,PUSD | horizon") != std::string::npos);
        auto snapshots = ListRecommendationRankingSnapshots(runtime, 100);
        assert(snapshots.size() == 1);
        assert(snapshots.front().status == "completed");
        assert(snapshots.front().counts.memberCount == 5);
        assert(snapshots.front().counts.advisoryReadyCount == 3);
        assert(snapshots.front().counts.blockedCount == 1);
        assert(snapshots.front().counts.nonActionableCount == 1);
        const long long snapshotId = snapshots.front().snapshotId;
        auto members = ListRecommendationRankingMembers(
            runtime, snapshotId, std::nullopt, 100);
        assert(members.size() == 5);
        assert(members[0].member.evaluation.finalScore == 0.9);
        assert(members[2].member.bucket == RecommendationRankingBucket::advisoryReady);
        assert(members[3].member.bucket == RecommendationRankingBucket::blocked);
        assert(members[4].member.bucket == RecommendationRankingBucket::nonActionable);
        assert(FindRecommendationRankingSnapshot(runtime, snapshotId));
        assert(FindRecommendationRankingMember(runtime, members.front().memberId));
        std::ostringstream missingSnapshotOutput;
        assert(RunListExperimentRecommendationRankingMembersCommand(
            runtimeConnectionString, 9223372036854775807LL, std::nullopt, 100,
            missingSnapshotOutput) == 3);
        assert(missingSnapshotOutput.str().empty());

        std::ostringstream retryOutput;
        std::ostringstream retryErrors;
        assert(RunRankExperimentRecommendationEvaluationsCommand(
            runtimeConnectionString, request, retryOutput, retryErrors) == 0);
        assert(ListRecommendationRankingSnapshots(runtime, 100).size() == 1);
        assert(ListRecommendationRankingMembers(
            runtime, snapshotId, std::nullopt, 100).size() == 5);
        auto mismatchedRetryMembers = members;
        mismatchedRetryMembers.front().member.tieBreakPrimary = "score:changed";
        bool retryMismatch = false;
        try
        {
            std::vector<RecommendationRankingMember> changedMembers;
            for (const auto& item : mismatchedRetryMembers)
                changedMembers.push_back(item.member);
            PersistRecommendationRankingMembers(
                runtime, snapshotId, changedMembers);
        }
        catch (const std::exception& error)
        {
            retryMismatch = std::string{error.what()} ==
                "recommendation_ranking_member_retry_mismatch";
        }
        assert(retryMismatch);

        RecommendationRankingCommandRequest changed = request;
        changed.limit = 4;
        std::ostringstream changedOutput;
        std::ostringstream changedErrors;
        assert(RunRankExperimentRecommendationEvaluationsCommand(
            runtimeConnectionString, changed, changedOutput, changedErrors) == 0);
        assert(ListRecommendationRankingSnapshots(runtime, 100).size() == 2);

        const auto left = FindRankingEvaluation(runtime, evaluationIds[0]);
        const auto right = FindRankingEvaluation(runtime, evaluationIds[1]);
        assert(left && right);
        assert(CompareRecommendationEvaluations(*left, *right).state ==
               RecommendationComparisonState::comparable);
        std::ostringstream compareOutput;
        assert(RunCompareExperimentRecommendationEvaluationsCommand(
            runtimeConnectionString, {evaluationIds[0], evaluationIds[1]},
            compareOutput) == 0);
        assert(compareOutput.str().find("score_delta=NULL") == std::string::npos);
        std::ostringstream memberCompareOutput;
        assert(RunCompareExperimentRecommendationRankingMembersCommand(
            runtimeConnectionString,
            {members[0].memberId, members[1].memberId}, memberCompareOutput) == 0);
        assert(memberCompareOutput.str().find("left_ranked_higher") !=
               std::string::npos);

        std::ostringstream concurrentOutput1, concurrentOutput2;
        std::ostringstream concurrentErrors1, concurrentErrors2;
        int result1 = -1;
        int result2 = -1;
        RecommendationRankingCommandRequest concurrent = request;
        concurrent.scope = symbolScope;
        std::thread first([&] {
            result1 = RunRankExperimentRecommendationEvaluationsCommand(
                runtimeConnectionString, concurrent, concurrentOutput1,
                concurrentErrors1);
        });
        std::thread second([&] {
            result2 = RunRankExperimentRecommendationEvaluationsCommand(
                runtimeConnectionString, concurrent, concurrentOutput2,
                concurrentErrors2);
        });
        first.join();
        second.join();
        if (result1 != 0 || result2 != 0)
            throw std::runtime_error(
                "concurrent_ranking_failed:first=" +
                concurrentErrors1.str() + ":second=" +
                concurrentErrors2.str());
        assert(ListRecommendationRankingSnapshots(runtime, 100).size() == 3);

        RecommendationRankingCommandRequest unrelatedFirst = request;
        unrelatedFirst.scope = horizonScope;
        RecommendationRankingCommandRequest unrelatedSecond = request;
        unrelatedSecond.scope = familyScope;
        result1 = -1;
        result2 = -1;
        concurrentErrors1.str("");
        concurrentErrors1.clear();
        concurrentErrors2.str("");
        concurrentErrors2.clear();
        std::thread unrelatedThread1([&] {
            result1 = RunRankExperimentRecommendationEvaluationsCommand(
                runtimeConnectionString, unrelatedFirst, concurrentOutput1,
                concurrentErrors1);
        });
        std::thread unrelatedThread2([&] {
            result2 = RunRankExperimentRecommendationEvaluationsCommand(
                runtimeConnectionString, unrelatedSecond, concurrentOutput2,
                concurrentErrors2);
        });
        unrelatedThread1.join();
        unrelatedThread2.join();
        assert(result1 == 0 && result2 == 0);
        assert(ListRecommendationRankingSnapshots(runtime, 100).size() == 5);

        RecommendationRankingScope globalScope;
        globalScope.type = RecommendationRankingScopeType::global;
        const auto globalEvaluations = LoadEvaluationsForRanking(runtime, globalScope);
        const auto ranked = RankRecommendationEvaluationEvidence(
            RecommendationRankingPolicy{}, globalEvaluations, 100);
        const std::string identity = RecommendationRankingSnapshotIdentityCanonicalText(
            RecommendationRankingPolicy{}, globalScope, 100, globalEvaluations);
        const std::string membership =
            RecommendationRankingMembershipCanonicalText(globalEvaluations);
        RecommendationRankingSnapshotRequest malformedMembershipRequest{
            RecommendationRankingPolicy{}, globalScope, 100, "malformed_identity",
            "malformed_identity_hash", "not_canonical_membership",
            RecommendationRankingCanonicalHash("not_canonical_membership")};
        bool malformedMembershipRejected = false;
        try
        {
            (void)BeginOrFindRecommendationRankingSnapshot(
                runtime, malformedMembershipRequest);
        }
        catch (const std::invalid_argument& error)
        {
            malformedMembershipRejected = std::string{error.what()} ==
                "invalid_recommendation_ranking_membership_canonical";
        }
        assert(malformedMembershipRejected);
        RecommendationRankingSnapshotRequest invalidIdentityRequest{
            RecommendationRankingPolicy{}, globalScope, 100, identity,
            "incorrect_identity_hash", membership,
            RecommendationRankingCanonicalHash(membership)};
        bool invalidIdentityRejected = false;
        try
        {
            (void)BeginOrFindRecommendationRankingSnapshot(
                runtime, invalidIdentityRequest);
        }
        catch (const std::invalid_argument& error)
        {
            invalidIdentityRejected = std::string{error.what()} ==
                "invalid_recommendation_ranking_snapshot_identity";
        }
        assert(invalidIdentityRejected);
        auto atomicSnapshot = BeginOrFindRecommendationRankingSnapshot(runtime, {
            RecommendationRankingPolicy{}, globalScope, 100, identity,
            RecommendationRankingCanonicalHash(identity), membership,
            RecommendationRankingCanonicalHash(membership)});
        auto ownershipMismatch = ranked;
        ownershipMismatch.front().evaluation.recommendationId += 1000;
        bool ownershipRejected = false;
        try { PersistRecommendationRankingMembers(
            runtime, atomicSnapshot.snapshotId, ownershipMismatch); }
        catch (const std::exception&) { ownershipRejected = true; }
        assert(ownershipRejected);
        assert(ListRecommendationRankingMembers(
            runtime, atomicSnapshot.snapshotId, std::nullopt, 100).empty());
        auto invalidMembers = ranked;
        invalidMembers[1].globalOrdinal = invalidMembers[0].globalOrdinal;
        bool failed = false;
        try { PersistRecommendationRankingMembers(
            runtime, atomicSnapshot.snapshotId, invalidMembers); }
        catch (const std::exception&) { failed = true; }
        assert(failed);
        assert(ListRecommendationRankingMembers(
            runtime, atomicSnapshot.snapshotId, std::nullopt, 100).empty());
        FailRecommendationRankingSnapshot(
            runtime, atomicSnapshot.snapshotId, "intentional_atomicity_test");

        long long boundedRunId = -1;
        {
            pqxx::work fixture{ownerConnection};
            fixture.exec("SET LOCAL search_path TO " +
                         fixture.quote_name(schema) + ";");
            boundedRunId = fixture.exec(
                "INSERT INTO experiment_recommendation_evaluation_run(status,"
                "evaluation_run_identity_canonical,evaluation_run_identity_hash,"
                "evaluation_policy_canonical,evaluation_policy_hash,"
                "evaluation_version,evaluator_version,scoring_policy_canonical,"
                "scoring_policy_hash,scoring_version,evidence_snapshot_canonical,"
                "evidence_snapshot_hash,completed_at) VALUES ('completed',"
                "'bounded_run','bounded_run_hash','evaluation_policy',"
                "'evaluation_policy_hash',1,1,'scoring_policy',"
                "'scoring_policy_hash',1,'bounded_evidence',"
                "'bounded_evidence_hash',now()) RETURNING "
                "recommendation_evaluation_run_id;").one_row()[0].as<long long>();
            fixture.exec(
                "WITH recommendations AS (INSERT INTO "
                "experiment_recommendation(recommendation_scan_id,"
                "source_experiment_id,source_analysis_id,status,source_symbol,"
                "source_prediction_horizon,changed_parameter,"
                "source_value_canonical,proposed_value_canonical) SELECT $1,1,1,"
                "'proposed','BOUND',12,'core_lr_mult','1',series::text FROM "
                "generate_series(1,$3) series RETURNING recommendation_id), "
                "ordered AS (SELECT recommendation_id,row_number() OVER "
                "(ORDER BY recommendation_id) AS ordinal FROM recommendations) "
                "INSERT INTO experiment_recommendation_evaluation_result("
                "recommendation_evaluation_run_id,recommendation_id,"
                "evaluation_identity_canonical,evaluation_identity_hash,"
                "recommendation_semantic_canonical,recommendation_semantic_hash,"
                "recommendation_policy_canonical,recommendation_policy_hash,"
                "recommendation_scan_id,source_experiment_id,evidence_canonical,"
                "evidence_hash,eligibility,disposition,reason_code,explanation,"
                "component_count,missing_evidence_count,ranking_ordinal) SELECT "
                "$2,recommendation_id,'bounded_'||recommendation_id,"
                "'bounded_hash_'||recommendation_id,'semantic_'||recommendation_id,"
                "'semantic_hash_'||recommendation_id,'policy','policy_hash',$1,1,"
                "'evidence_'||recommendation_id,'evidence_hash_'||"
                "recommendation_id,'ineligible','insufficient_evidence',"
                "'insufficient','Insufficient.',0,1,ordinal FROM ordered;",
                pqxx::params{scanId, boundedRunId,
                             kMaximumRecommendationRankingInputs + 1});
            fixture.commit();
        }
        RecommendationRankingScope boundedScope;
        boundedScope.type = RecommendationRankingScopeType::evaluationRun;
        boundedScope.evaluationRunId = boundedRunId;
        bool boundedInputRejected = false;
        try { (void)LoadEvaluationsForRanking(runtime, boundedScope); }
        catch (const std::invalid_argument& error)
        {
            boundedInputRejected = std::string{error.what()} ==
                "recommendation_ranking_input_limit_exceeded";
        }
        assert(boundedInputRejected);
        {
            pqxx::work fixture{ownerConnection};
            fixture.exec("SET LOCAL search_path TO " +
                         fixture.quote_name(schema) + ";");
            fixture.exec("DELETE FROM experiment_recommendation_evaluation_result "
                         "WHERE recommendation_evaluation_run_id=$1;",
                         pqxx::params{boundedRunId});
            fixture.exec("DELETE FROM experiment_recommendation "
                         "WHERE source_symbol='BOUND';");
            fixture.exec("DELETE FROM experiment_recommendation_evaluation_run "
                         "WHERE recommendation_evaluation_run_id=$1;",
                         pqxx::params{boundedRunId});
            fixture.commit();
        }

        assert(before == SourceDigest(ownerConnection, schema));
    }
    catch (...)
    {
        pqxx::work cleanup{ownerConnection};
        cleanup.exec("DROP SCHEMA IF EXISTS " + cleanup.quote_name(schema) +
                     " CASCADE;");
        cleanup.commit();
        throw;
    }
    pqxx::work cleanup{ownerConnection};
    cleanup.exec("DROP SCHEMA " + cleanup.quote_name(schema) + " CASCADE;");
    cleanup.commit();
    return 0;
}
