#include <algorithm>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <future>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "../Sources/ExperimentRecommendationRepository.hpp"
#include "../Sources/ExperimentRecommendationService.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

long long InsertRecommendation(pqxx::work& tx, long long scanId,
                               long long experimentId, long long analysisId,
                               int ordinal)
{
    const std::string suffix = std::to_string(ordinal);
    return tx.exec(
        "INSERT INTO experiment_recommendation (recommendation_scan_id,status,"
        "source_experiment_id,source_analysis_id,source_symbol,"
        "source_prediction_horizon,source_leader_score,source_infer_accuracy,"
        "source_predicted_neutral_proportion,source_evidence_count,"
        "changed_parameter,source_value_canonical,proposed_value_canonical,"
        "absolute_delta,relative_delta,semantic_configuration_canonical,"
        "semantic_hash,invocation_configuration_canonical,invocation_hash,"
        "policy_canonical,policy_hash,source_rank,generation_ordinal,"
        "structural_rank,duplicate_type,reason) VALUES ("
        "$1,'proposed',$2,$3,'step5_review_fixture',12,0.75,0.70,0.30,100,"
        "'core_lr_mult','1','1.25',0.25,0.25,$4,$5,$6,$7,$8,$9,1,$10,$10,"
        "'no_duplicate','step5_review_fixture') RETURNING recommendation_id;",
        pqxx::params{
            scanId, experimentId, analysisId,
            "step5_review_semantic_" + suffix,
            "fnv1a64:step5_semantic_" + suffix,
            "step5_review_invocation_" + suffix,
            "fnv1a64:step5_invocation_" + suffix,
            "step5_review_policy_" + suffix,
            "fnv1a64:step5_policy_" + suffix, ordinal})
        .one_row()[0].as<long long>();
}

long long InsertScore(pqxx::work& tx, long long runId,
                      long long recommendationId, long long experimentId,
                      int ordinal)
{
    return tx.exec(
        "INSERT INTO experiment_recommendation_score ("
        "recommendation_score_run_id,recommendation_id,"
        "scoring_policy_canonical,scoring_policy_hash,scoring_version,"
        "recommendation_semantic_canonical,recommendation_policy_canonical,"
        "source_experiment_id,final_score,raw_positive_score,"
        "raw_penalty_score,raw_total_score,structural_distance,score_rank,"
        "tie_group,ranking_ordinal,score_status,reason_code,explanation) VALUES ("
        "$1,$2,'step5_scoring_policy','fnv1a64:step5_scoring',1,$3,$4,$5,"
        "0.75,0.80,0.05,0.75,0.25,$6,$6,$6,'scored','step5_fixture',"
        "'advisory fixture score') RETURNING recommendation_score_id;",
        pqxx::params{runId, recommendationId,
                     "step5_review_semantic_" + std::to_string(ordinal),
                     "step5_review_policy_" + std::to_string(ordinal),
                     experimentId, ordinal})
        .one_row()[0].as<long long>();
}

RecommendationReviewRequest Request(RecommendationReviewAction action)
{
    RecommendationReviewRequest request;
    request.action = action;
    if (action == RecommendationReviewAction::reject)
    {
        request.reasonCode = "operator_rejected";
        request.reasonText = "Concurrent rejection.";
    }
    else if (action == RecommendationReviewAction::expire)
    {
        request.reasonCode = "operator_expired";
        request.reasonText = "Concurrent expiration.";
    }
    return request;
}

std::string ReviewConcurrently(const std::string& connectionString,
                               std::barrier<>& start,
                               long long recommendationId,
                               RecommendationReviewAction action)
{
    try
    {
        pqxx::connection connection{connectionString};
        start.arrive_and_wait();
        ReviewRecommendation(connection,
            RecommendationReviewPersistenceRequest{
                recommendationId, Request(action)});
        return "success";
    }
    catch (const std::runtime_error& error)
    {
        return error.what();
    }
}

void RequireOneWinner(const std::string& first, const std::string& second)
{
    assert((first == "success") != (second == "success"));
    assert(first == "success" ||
           first == "recommendation_review_status_conflict");
    assert(second == "success" ||
           second == "recommendation_review_status_conflict");
}

void Cleanup(pqxx::connection& ownerConnection,
             const std::vector<long long>& recommendationIds,
             const std::vector<long long>& scoreIds,
             long long scoreRunId, long long scanId,
             long long analysisId, long long experimentId)
{
    pqxx::work tx{ownerConnection};
    tx.exec("SET TRANSACTION READ WRITE;");
    for (const long long recommendationId : recommendationIds)
        tx.exec("DELETE FROM experiment_recommendation_review_event "
                "WHERE recommendation_id=$1;",
                pqxx::params{recommendationId});
    for (const long long scoreId : scoreIds)
    {
        tx.exec("DELETE FROM experiment_recommendation_score_component "
                "WHERE recommendation_score_id=$1;", pqxx::params{scoreId});
        tx.exec("DELETE FROM experiment_recommendation_score "
                "WHERE recommendation_score_id=$1;", pqxx::params{scoreId});
    }
    tx.exec("DELETE FROM experiment_recommendation_score_run "
            "WHERE recommendation_score_run_id=$1;", pqxx::params{scoreRunId});
    for (const long long recommendationId : recommendationIds)
        tx.exec("DELETE FROM experiment_recommendation WHERE recommendation_id=$1;",
                pqxx::params{recommendationId});
    tx.exec("DELETE FROM experiment_recommendation_scan "
            "WHERE recommendation_scan_id=$1;", pqxx::params{scanId});
    tx.exec("DELETE FROM experiment_analysis_result WHERE analysis_id=$1;",
            pqxx::params{analysisId});
    tx.exec("DELETE FROM experiment WHERE experiment_id=$1;",
            pqxx::params{experimentId});
    tx.commit();
}

} // namespace

int main()
{
    const std::string connectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=pqxx dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    const std::string defaultOwnerUser = EnvironmentOr("USER", "vjp");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_DB_ADMIN_USER", defaultOwnerUser.c_str());
    const std::string ownerConnectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=" + ownerUser +
        " dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    pqxx::connection connection{connectionString};
    pqxx::connection ownerConnection{ownerConnectionString};
    assert(RecommendationReviewSchemaExists(connection));

    const auto rejectsInvalidArgument = [](auto&& operation,
                                           const std::string& expected) {
        try
        {
            operation();
        }
        catch (const std::invalid_argument& error)
        {
            return std::string{error.what()} == expected;
        }
        return false;
    };
    RecommendationReviewFilters invalidList;
    invalidList.limit = 0;
    assert(rejectsInvalidArgument(
        [&] { (void)ListRecommendationReviewEvents(connection, invalidList); },
        "recommendation_review_limit_invalid"));
    invalidList.limit = 1;
    invalidList.recommendationId = 0;
    assert(rejectsInvalidArgument(
        [&] { (void)ListRecommendationReviewEvents(connection, invalidList); },
        "recommendation_review_id_invalid"));
    assert(rejectsInvalidArgument(
        [&] { (void)FindRecommendationReviewEvent(connection, 0); },
        "recommendation_review_event_id_invalid"));
    assert(rejectsInvalidArgument(
        [&] { (void)ListReviewHistoryForRecommendation(connection, 0); },
        "recommendation_review_id_invalid"));

    long long experimentId = -1;
    long long analysisId = -1;
    long long scanId = -1;
    long long scoreRunId = -1;
    std::vector<long long> recommendationIds;
    std::vector<long long> scoreIds;
    {
        pqxx::work tx{connection};
        tx.exec("SET TRANSACTION READ WRITE;");
        experimentId = tx.exec(
            "INSERT INTO experiment (symbol,prediction_horizon,c_next_threshold,"
            "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
            "train_start,train_end,status,phase,duplicate_nonce) VALUES ("
            "'step5_review_fixture',12,0.001,1,5,120,20,"
            "'2010-01-01 America/Chicago'::timestamptz,"
            "'2025-01-01 America/Chicago'::timestamptz,'completed','done',"
            "floor(random()*900000000)::bigint+100000000) RETURNING experiment_id;")
            .one_row()[0].as<long long>();
        analysisId = tx.exec(
            "INSERT INTO experiment_analysis_result (experiment_id,model_id,"
            "symbol,prediction_horizon,target_epochs,infer_accuracy,leader_score,"
            "pred_down_count,pred_neutral_count,pred_up_count,analysis_status,"
            "analysis_scope) VALUES ($1,987654398,'step5_review_fixture',12,120,"
            "0.70,0.75,20,30,50,'completed','final') RETURNING analysis_id;",
            pqxx::params{experimentId}).one_row()[0].as<long long>();
        scanId = tx.exec(
            "INSERT INTO experiment_recommendation_scan (status,policy_canonical,"
            "policy_hash,policy_version,completed_at) VALUES ("
            "'completed','step5_review_scan_policy','fnv1a64:step5_scan',1,now()) "
            "RETURNING recommendation_scan_id;").one_row()[0].as<long long>();
        for (int ordinal = 1; ordinal <= 9; ++ordinal)
            recommendationIds.push_back(InsertRecommendation(
                tx, scanId, experimentId, analysisId, ordinal));
        scoreRunId = tx.exec(
            "INSERT INTO experiment_recommendation_score_run (status,"
            "scoring_policy_canonical,scoring_policy_hash,scoring_version,"
            "recommendation_status_filter,completed_at) VALUES ("
            "'completed','step5_scoring_policy','fnv1a64:step5_scoring',1,"
            "'proposed',now()) RETURNING recommendation_score_run_id;")
            .one_row()[0].as<long long>();
        scoreIds.push_back(InsertScore(
            tx, scoreRunId, recommendationIds[0], experimentId, 1));
        scoreIds.push_back(InsertScore(
            tx, scoreRunId, recommendationIds[1], experimentId, 2));
        tx.exec(
            "INSERT INTO experiment_recommendation_review_event ("
            "recommendation_id,action,previous_status,resulting_status,"
            "reason_code,recommendation_semantic_canonical,"
            "recommendation_semantic_hash,recommendation_policy_canonical,"
            "recommendation_policy_hash,recommendation_scan_id,"
            "source_experiment_id) SELECT recommendation_id,'approve',"
            "'proposed','approved','operator_approved',"
            "semantic_configuration_canonical,semantic_hash,policy_canonical,"
            "policy_hash,recommendation_scan_id,source_experiment_id "
            "FROM experiment_recommendation WHERE recommendation_id=$1;",
            pqxx::params{recommendationIds[8]});
        tx.commit();
    }

    try
    {
        std::string experimentBefore;
        std::string experimentTableBefore;
        std::string identitiesBefore;
        std::string scoresBefore;
        {
            pqxx::read_transaction snapshot{connection};
            experimentBefore = snapshot.exec(
                "SELECT status||'|'||phase||'|'||coalesce(worker_pid::text,'NULL')"
                "||'|'||coalesce(current_operation,'NULL')||'|'||updated_at::text "
                "FROM experiment WHERE experiment_id=$1;",
                pqxx::params{experimentId}).one_row()[0].as<std::string>();
            experimentTableBefore = snapshot.exec(
                "SELECT count(*)::text||':'||md5(coalesce(string_agg(concat_ws('|',"
                "experiment_id::text,status,phase,coalesce(worker_pid::text,'NULL'),"
                "coalesce(current_operation,'NULL'),updated_at::text),"
                "'#' ORDER BY experiment_id),'')) FROM experiment;")
                .one_row()[0].as<std::string>();
            identitiesBefore = snapshot.exec(
                "SELECT md5(string_agg(concat_ws('|',recommendation_id::text,"
                "semantic_configuration_canonical,semantic_hash,policy_canonical,"
                "policy_hash,recommendation_scan_id::text,source_experiment_id::text,"
                "changed_parameter,source_value_canonical,proposed_value_canonical),"
                "'#' ORDER BY recommendation_id)) FROM experiment_recommendation "
                "WHERE recommendation_scan_id=$1;", pqxx::params{scanId})
                .one_row()[0].as<std::string>();
            scoresBefore = snapshot.exec(
                "SELECT count(*)::text||':'||md5(coalesce(string_agg(concat_ws('|',"
                "recommendation_score_id::text,recommendation_id::text,"
                "final_score::text,score_rank::text,ranking_ordinal::text),"
                "'#' ORDER BY recommendation_score_id),'')) "
                "FROM experiment_recommendation_score "
                "WHERE recommendation_score_run_id=$1;",
                pqxx::params{scoreRunId}).one_row()[0].as<std::string>();
        }

        RecommendationReviewCommandRequest approval;
        approval.recommendationId = recommendationIds[0];
        approval.review.action = RecommendationReviewAction::approve;
        approval.review.recommendationScoreId = scoreIds[0];
        approval.review.reviewer = "NULL";
        approval.review.note = "checked, manually";
        std::ostringstream approvalOutput;
        std::ostringstream approvalErrors;
        assert(RunExperimentRecommendationReviewCommand(
            connectionString, approval, approvalOutput, approvalErrors) == 0);
        assert(approvalErrors.str().empty());
        assert(approvalOutput.str().find(
            "EXPERIMENT_RECOMMENDATION_REVIEW_APPROVED") != std::string::npos);
        assert(approvalOutput.str().find("reviewer=%4E%55%4C%4C") !=
               std::string::npos);
        assert(approvalOutput.str().find("note=checked%2C%20manually") !=
               std::string::npos);
        assert(approvalOutput.str().find("Reviewer: NULL\n") !=
               std::string::npos);
        assert(approvalOutput.str().find("Reviewer: %4E%55%4C%4C") ==
               std::string::npos);
        assert(approvalOutput.str().find(
            "Review is advisory only. No experiment was created or queued.") !=
               std::string::npos);

        const auto approvedHistory = ListReviewHistoryForRecommendation(
            connection, recommendationIds[0]);
        assert(approvedHistory.size() == 1);
        assert(approvedHistory.front().recommendationScoreId == scoreIds[0]);
        assert(approvedHistory.front().recommendationScanId == scanId);
        assert(approvedHistory.front().sourceExperimentId == experimentId);
        const auto approvedDetail = FindRecommendation(
            connection, recommendationIds[0]);
        assert(approvedDetail && approvedDetail->status == "approved");
        assert(approvedDetail->approvedAt);
        assert(!approvedDetail->approvedExperimentId);
        assert(!approvedDetail->rejectedAt && !approvedDetail->expiredAt);

        const std::string approvedMetadata = [&] {
            pqxx::read_transaction tx{connection};
            return tx.exec(
                "SELECT status||'|'||approved_at::text||'|'||updated_at::text "
                "FROM experiment_recommendation WHERE recommendation_id=$1;",
                pqxx::params{recommendationIds[0]})
                .one_row()[0].as<std::string>();
        }();
        std::ostringstream retryOutput;
        std::ostringstream retryErrors;
        assert(RunExperimentRecommendationReviewCommand(
            connectionString, approval, retryOutput, retryErrors) == 2);
        assert(retryErrors.str().find(
            "EXPERIMENT_RECOMMENDATION_REVIEW_CONFLICT") != std::string::npos);
        assert(ListReviewHistoryForRecommendation(
            connection, recommendationIds[0]).size() == 1);
        {
            pqxx::read_transaction tx{connection};
            assert(tx.exec(
                "SELECT status||'|'||approved_at::text||'|'||updated_at::text "
                "FROM experiment_recommendation WHERE recommendation_id=$1;",
                pqxx::params{recommendationIds[0]})
                .one_row()[0].as<std::string>() == approvedMetadata);
        }

        RecommendationReviewPersistenceRequest wrongScore{
            recommendationIds[1], Request(RecommendationReviewAction::approve)};
        wrongScore.review.recommendationScoreId = scoreIds[0];
        bool ownershipRejected = false;
        try { (void)ReviewRecommendation(connection, wrongScore); }
        catch (const std::runtime_error& error) {
            ownershipRejected = std::string{error.what()} ==
                "recommendation_review_score_ownership_mismatch";
        }
        assert(ownershipRejected);
        wrongScore.review.recommendationScoreId = 9223372036854770000LL;
        bool missingScoreRejected = false;
        try { (void)ReviewRecommendation(connection, wrongScore); }
        catch (const std::runtime_error& error) {
            missingScoreRejected = std::string{error.what()} ==
                "recommendation_review_score_not_found";
        }
        assert(missingScoreRejected);
        assert(ListReviewHistoryForRecommendation(
            connection, recommendationIds[1]).empty());

        RecommendationReviewCommandRequest rejection;
        rejection.recommendationId = recommendationIds[1];
        rejection.review.action = RecommendationReviewAction::reject;
        rejection.review.reasonCode = "insufficient_evidence";
        rejection.review.reasonText = "  Source evidence, not sufficient.  ";
        std::ostringstream rejectionOutput;
        std::ostringstream rejectionErrors;
        assert(RunExperimentRecommendationReviewCommand(
            connectionString, rejection, rejectionOutput, rejectionErrors) == 0);
        assert(rejectionOutput.str().find(
            "reason_text=Source%20evidence%2C%20not%20sufficient.") !=
               std::string::npos);
        assert(rejectionOutput.str().find(
            "Reason: insufficient_evidence\n"
            "Details: Source evidence, not sufficient.\n") !=
               std::string::npos);
        assert(rejectionOutput.str().find(
            "Details: Source%20evidence%2C%20not%20sufficient.") ==
               std::string::npos);
        const auto rejected = FindRecommendation(connection, recommendationIds[1]);
        assert(rejected && rejected->status == "rejected");
        assert(rejected->rejectedReason ==
               std::optional<std::string>{"Source evidence, not sufficient."});
        assert(rejected->rejectedAt && !rejected->approvedAt && !rejected->expiredAt);

        RecommendationReviewCommandRequest expiration;
        expiration.recommendationId = recommendationIds[2];
        expiration.review.action = RecommendationReviewAction::expire;
        expiration.review.reasonCode = "research_window_closed";
        expiration.review.reasonText = "Window closed.";
        std::ostringstream expirationOutput;
        std::ostringstream expirationErrors;
        assert(RunExperimentRecommendationReviewCommand(
            connectionString, expiration, expirationOutput, expirationErrors) == 0);
        const auto expired = FindRecommendation(connection, recommendationIds[2]);
        assert(expired && expired->status == "expired");
        assert(expired->expiredAt && !expired->approvedAt && !expired->rejectedAt);

        const auto runPair = [&](long long recommendationId,
                                 RecommendationReviewAction firstAction,
                                 RecommendationReviewAction secondAction) {
            std::barrier start{3};
            auto first = std::async(std::launch::async, [&] {
                return ReviewConcurrently(connectionString, start,
                                           recommendationId, firstAction);
            });
            auto second = std::async(std::launch::async, [&] {
                return ReviewConcurrently(connectionString, start,
                                           recommendationId, secondAction);
            });
            start.arrive_and_wait();
            const std::string firstResult = first.get();
            const std::string secondResult = second.get();
            RequireOneWinner(firstResult, secondResult);
            assert(ListReviewHistoryForRecommendation(
                connection, recommendationId).size() == 1);
        };
        runPair(recommendationIds[3], RecommendationReviewAction::approve,
                RecommendationReviewAction::approve);
        runPair(recommendationIds[4], RecommendationReviewAction::approve,
                RecommendationReviewAction::reject);
        runPair(recommendationIds[5], RecommendationReviewAction::reject,
                RecommendationReviewAction::expire);

        std::barrier unrelatedStart{3};
        auto unrelatedFirst = std::async(std::launch::async, [&] {
            return ReviewConcurrently(connectionString, unrelatedStart,
                recommendationIds[6], RecommendationReviewAction::approve);
        });
        auto unrelatedSecond = std::async(std::launch::async, [&] {
            return ReviewConcurrently(connectionString, unrelatedStart,
                recommendationIds[7], RecommendationReviewAction::expire);
        });
        unrelatedStart.arrive_and_wait();
        assert(unrelatedFirst.get() == "success");
        assert(unrelatedSecond.get() == "success");

        bool eventFailureRolledBack = false;
        try
        {
            (void)ReviewRecommendation(connection,
                RecommendationReviewPersistenceRequest{
                    recommendationIds[8],
                    Request(RecommendationReviewAction::approve)});
        }
        catch (const pqxx::unique_violation&)
        {
            pqxx::read_transaction verify{connection};
            eventFailureRolledBack = verify.exec(
                "SELECT status FROM experiment_recommendation "
                "WHERE recommendation_id=$1;",
                pqxx::params{recommendationIds[8]})
                .one_row()[0].as<std::string>() == "proposed";
        }
        assert(eventFailureRolledBack);

        RecommendationReviewListCommandRequest listRequest;
        listRequest.recommendationId = recommendationIds[0];
        listRequest.action = RecommendationReviewAction::approve;
        std::ostringstream listOutput;
        assert(RunListExperimentRecommendationReviewsCommand(
            connectionString, listRequest, listOutput) == 0);
        assert(listOutput.str().find(
            "EXPERIMENT_RECOMMENDATION_REVIEW_LIST_COMPLETE,count=1") !=
               std::string::npos);
        std::ostringstream detailOutput;
        assert(RunExperimentRecommendationReviewStatusCommand(
            connectionString,
            approvedHistory.front().recommendationReviewEventId,
            detailOutput) == 0);
        std::ostringstream historyOutput;
        assert(RunExperimentRecommendationReviewHistoryCommand(
            connectionString, recommendationIds[0], historyOutput) == 0);
        assert(historyOutput.str().find(
            "EXPERIMENT_RECOMMENDATION_REVIEW_HISTORY_COMPLETE") !=
               std::string::npos);

        const auto containsRecommendation = [](const auto& rows, long long id) {
            return std::any_of(rows.begin(), rows.end(), [id](const auto& row) {
                return row.recommendationId == id;
            });
        };
        RecommendationListFilters approvedFilter;
        approvedFilter.status = "approved";
        approvedFilter.recommendationScanId = scanId;
        assert(containsRecommendation(
            ListRecommendations(connection, approvedFilter), recommendationIds[0]));
        RecommendationListFilters rejectedFilter;
        rejectedFilter.status = "rejected";
        rejectedFilter.recommendationScanId = scanId;
        assert(containsRecommendation(
            ListRecommendations(connection, rejectedFilter), recommendationIds[1]));
        RecommendationListFilters expiredFilter;
        expiredFilter.status = "expired";
        expiredFilter.recommendationScanId = scanId;
        assert(containsRecommendation(
            ListRecommendations(connection, expiredFilter), recommendationIds[2]));

        pqxx::read_transaction after{connection};
        const std::string experimentAfter = after.exec(
            "SELECT status||'|'||phase||'|'||coalesce(worker_pid::text,'NULL')"
            "||'|'||coalesce(current_operation,'NULL')||'|'||updated_at::text "
            "FROM experiment WHERE experiment_id=$1;",
            pqxx::params{experimentId}).one_row()[0].as<std::string>();
        const std::string experimentTableAfter = after.exec(
            "SELECT count(*)::text||':'||md5(coalesce(string_agg(concat_ws('|',"
            "experiment_id::text,status,phase,coalesce(worker_pid::text,'NULL'),"
            "coalesce(current_operation,'NULL'),updated_at::text),"
            "'#' ORDER BY experiment_id),'')) FROM experiment;")
            .one_row()[0].as<std::string>();
        const std::string identitiesAfter = after.exec(
            "SELECT md5(string_agg(concat_ws('|',recommendation_id::text,"
            "semantic_configuration_canonical,semantic_hash,policy_canonical,"
            "policy_hash,recommendation_scan_id::text,source_experiment_id::text,"
            "changed_parameter,source_value_canonical,proposed_value_canonical),"
            "'#' ORDER BY recommendation_id)) FROM experiment_recommendation "
            "WHERE recommendation_scan_id=$1;", pqxx::params{scanId})
            .one_row()[0].as<std::string>();
        const std::string scoresAfter = after.exec(
            "SELECT count(*)::text||':'||md5(coalesce(string_agg(concat_ws('|',"
            "recommendation_score_id::text,recommendation_id::text,"
            "final_score::text,score_rank::text,ranking_ordinal::text),"
            "'#' ORDER BY recommendation_score_id),'')) "
            "FROM experiment_recommendation_score "
            "WHERE recommendation_score_run_id=$1;",
            pqxx::params{scoreRunId}).one_row()[0].as<std::string>();
        assert(experimentAfter == experimentBefore);
        assert(experimentTableAfter == experimentTableBefore);
        assert(identitiesAfter == identitiesBefore);
        assert(scoresAfter == scoresBefore);
    }
    catch (...)
    {
        Cleanup(ownerConnection, recommendationIds, scoreIds, scoreRunId,
                scanId, analysisId, experimentId);
        throw;
    }
    Cleanup(ownerConnection, recommendationIds, scoreIds, scoreRunId,
            scanId, analysisId, experimentId);
    return 0;
}
