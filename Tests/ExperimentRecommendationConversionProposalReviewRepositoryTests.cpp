#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <barrier>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

RecommendationConversionProposalReviewRequest Request(
    long long proposalId,
    RecommendationConversionProposalReviewDecision decision,
    const std::string& requestId)
{
    RecommendationConversionProposalReviewRequest request;
    request.proposalId = proposalId;
    request.decision = decision;
    request.requestId = requestId;
    request.operatorIdentity = "operator name";
    request.reasonText = "Manual decision reason.";
    return request;
}

bool PermissionDenied(const pqxx::sql_error& error)
{
    const std::string message = error.what();
    return error.sqlstate() == "42501" ||
           (error.sqlstate().empty() &&
            message.find("permission denied") != std::string::npos);
}

long long InsertProposal(pqxx::work& transaction,
                         long long recommendationId,
                         const std::string& hash,
                         int ordinal)
{
    return transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_proposal ("
        "recommendation_id,source_experiment_id,conversion_contract_version,"
        "changed_parameter,source_value_canonical,proposed_value_canonical,"
        "recommendation_semantic_hash,evaluation_identity_hash,"
        "evaluation_policy_hash,scoring_policy_hash,review_authorization_hash,"
        "proposed_symbol,proposed_prediction_horizon,"
        "proposed_label_threshold,proposed_target_epochs,"
        "proposed_train_start_date,proposed_train_end_date,"
        "proposed_checkpoint_interval,source_invocation_canonical,"
        "proposed_invocation_canonical,conversion_identity_canonical,"
        "conversion_identity_hash,conversion_hash_collision_ordinal) VALUES ("
        "$1,17,1,'core_lr_mult','1','1.25','semantic_hash',"
        "'evaluation_hash','evaluation_policy_hash','scoring_policy_hash',"
        "'review_authorization_hash','eurusd',12,0.001,120,'2010-01-01',"
        "'2025-01-01',15,'source_invocation','proposed_invocation',$2,$3,$4) "
        "RETURNING recommendation_conversion_proposal_id;",
        pqxx::params{recommendationId,
                     "conversion_identity_" + std::to_string(recommendationId),
                     hash, ordinal}).one_row()[0].as<long long>();
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
    if (database == "LSTM")
    {
        std::cerr << "active_LSTM_database_forbidden\n";
        return 2;
    }
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "phase4c_conversion_review_" + std::to_string(getpid());
    const std::string ownerConnectionString =
        "host=" + host + " port=" + port + " user=" + ownerUser +
        " dbname=" + database;
    const std::string runtimeConnectionString =
        "host=" + host + " port=" + port + " user=pqxx dbname=" + database +
        " options='-c search_path=" + schema + "'";
    pqxx::connection owner{ownerConnectionString};
    long long firstProposalId = -1;
    long long secondProposalId = -1;

    try
    {
        {
            pqxx::work setup{owner};
            setup.exec("CREATE SCHEMA " + setup.quote_name(schema) + ";");
            setup.exec("SET LOCAL search_path TO " + setup.quote_name(schema) +
                       ";");
            setup.exec("CREATE TABLE experiment(experiment_id bigint PRIMARY KEY,"
                       "status text NOT NULL,marker text NOT NULL);");
            setup.exec("CREATE TABLE model(model_id bigint PRIMARY KEY);");
            setup.exec("CREATE TABLE experiment_recommendation("
                       "recommendation_id bigint PRIMARY KEY,"
                       "source_experiment_id bigint NOT NULL REFERENCES "
                       "experiment(experiment_id),status text NOT NULL);");
            setup.exec("INSERT INTO experiment VALUES (17,'paused','unchanged');"
                       "INSERT INTO experiment_recommendation VALUES "
                       "(42,17,'approved'),(43,17,'approved');");
            const std::string proposalMigration = ReadFile(
                "Database/migrations/036_experiment_recommendation_conversion_proposal.sql");
            const std::string reviewMigration = ReadFile(
                "Database/migrations/037_experiment_recommendation_conversion_review.sql");
            setup.exec(proposalMigration);
            setup.exec(proposalMigration);
            firstProposalId = InsertProposal(setup, 42, "hash_one", 0);
            secondProposalId = InsertProposal(setup, 43, "hash_two", 0);
            setup.exec(reviewMigration);
            setup.exec(reviewMigration);
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationConversionProposalReviewMigrationTests.sql"));
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(RecommendationConversionProposalReviewSchemaExists(runtime));
        const auto pending = GetRecommendationConversionProposalCurrentReview(
            runtime, firstProposalId);
        assert(pending && pending->disposition ==
            RecommendationConversionProposalReviewDisposition::pendingReview);
        assert(!pending->latestDecision);

        const auto approvalRequest = Request(
            firstProposalId,
            RecommendationConversionProposalReviewDecision::approve,
            "approval-0001");
        const auto approval =
            RecordRecommendationConversionProposalReviewDecision(
                runtime, approvalRequest);
        assert(approval.outcome ==
            RecommendationConversionProposalReviewPersistOutcome::recorded);
        assert(approval.decision && approval.decision->reviewDecisionId > 0);
        assert(approval.decision->operatorIdentity ==
               std::optional<std::string>{"operator name"});
        assert(approval.decision->reasonText ==
               std::optional<std::string>{"Manual decision reason."});

        const auto found = FindRecommendationConversionProposalReviewDecision(
            runtime, approval.decision->reviewDecisionId);
        assert(found && found->requestId == "approval-0001");
        const auto replay = RecordRecommendationConversionProposalReviewDecision(
            runtime, approvalRequest);
        assert(replay.outcome ==
            RecommendationConversionProposalReviewPersistOutcome::
                existingIdentical);
        assert(replay.decision->reviewDecisionId ==
               approval.decision->reviewDecisionId);

        auto conflictingReplay = approvalRequest;
        conflictingReplay.decision =
            RecommendationConversionProposalReviewDecision::reject;
        bool conflictRejected = false;
        try
        {
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, conflictingReplay);
        }
        catch (const std::runtime_error& error)
        {
            conflictRejected = std::string{error.what()} ==
                "recommendation_conversion_proposal_review_request_conflict";
        }
        assert(conflictRejected);

        const auto rejection = RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Request(firstProposalId,
                    RecommendationConversionProposalReviewDecision::reject,
                    "rejection-0002"));
        assert(rejection.outcome ==
            RecommendationConversionProposalReviewPersistOutcome::recorded);
        auto current = GetRecommendationConversionProposalCurrentReview(
            runtime, firstProposalId);
        assert(current && current->disposition ==
            RecommendationConversionProposalReviewDisposition::rejected);

        const auto reversal = RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Request(firstProposalId,
                    RecommendationConversionProposalReviewDecision::approve,
                    "approval-0003"));
        current = GetRecommendationConversionProposalCurrentReview(
            runtime, firstProposalId);
        assert(current && current->disposition ==
            RecommendationConversionProposalReviewDisposition::approved);
        assert(current->latestDecision->reviewDecisionId ==
               reversal.decision->reviewDecisionId);
        const auto history = ListRecommendationConversionProposalReviewDecisions(
            runtime, firstProposalId);
        assert(history.size() == 3);
        assert(history[0].reviewDecisionId < history[1].reviewDecisionId);
        assert(history[1].reviewDecisionId < history[2].reviewDecisionId);

        auto optionalRequest = Request(
            secondProposalId,
            RecommendationConversionProposalReviewDecision::approve,
            "optional-0001");
        optionalRequest.operatorIdentity.reset();
        optionalRequest.reasonText.reset();
        const auto optional = RecordRecommendationConversionProposalReviewDecision(
            runtime, optionalRequest);
        assert(optional.decision && !optional.decision->operatorIdentity &&
               !optional.decision->reasonText);
        auto sameTokenOtherProposal = approvalRequest;
        sameTokenOtherProposal.proposalId = secondProposalId;
        const auto separateTokenScope =
            RecordRecommendationConversionProposalReviewDecision(
                runtime, sameTokenOtherProposal);
        assert(separateTokenScope.outcome ==
            RecommendationConversionProposalReviewPersistOutcome::recorded);

        const auto missing = RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Request(999999,
                    RecommendationConversionProposalReviewDecision::approve,
                    "missing-0001"));
        assert(missing.outcome ==
            RecommendationConversionProposalReviewPersistOutcome::proposalNotFound);
        assert(!GetRecommendationConversionProposalCurrentReview(runtime, 999999));

        const auto approved = ListRecommendationConversionProposalsByReviewDisposition(
            runtime,
            RecommendationConversionProposalReviewDisposition::approved);
        assert(approved.size() == 2);
        assert(ListRecommendationConversionProposalsByReviewDisposition(
            runtime,
            RecommendationConversionProposalReviewDisposition::pendingReview)
            .empty());

        // A rolled-back decision never changes current disposition.
        {
            pqxx::work rolledBack{owner};
            rolledBack.exec("SET LOCAL search_path TO " +
                            rolledBack.quote_name(schema) + ";");
            rolledBack.exec(
                "INSERT INTO experiment_recommendation_conversion_review_decision "
                "(recommendation_conversion_proposal_id,decision,"
                "decision_request_id) VALUES ($1,'reject','rolled-back');",
                pqxx::params{secondProposalId});
            rolledBack.abort();
        }
        current = GetRecommendationConversionProposalCurrentReview(
            runtime, secondProposalId);
        assert(current && current->disposition ==
            RecommendationConversionProposalReviewDisposition::approved);

        // Identical concurrent retries converge on one durable decision.
        const auto concurrentRequest = Request(
            firstProposalId,
            RecommendationConversionProposalReviewDecision::reject,
            "concurrent-identical");
        RecommendationConversionProposalReviewPersistResult firstConcurrent;
        RecommendationConversionProposalReviewPersistResult secondConcurrent;
        std::exception_ptr firstError;
        std::exception_ptr secondError;
        std::barrier identicalStart{3};
        std::thread first([&] {
            bool synchronized = false;
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                identicalStart.arrive_and_wait();
                synchronized = true;
                firstConcurrent =
                    RecordRecommendationConversionProposalReviewDecision(
                        connection, concurrentRequest);
            }
            catch (...)
            {
                if (!synchronized) identicalStart.arrive_and_drop();
                firstError = std::current_exception();
            }
        });
        std::thread second([&] {
            bool synchronized = false;
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                identicalStart.arrive_and_wait();
                synchronized = true;
                secondConcurrent =
                    RecordRecommendationConversionProposalReviewDecision(
                        connection, concurrentRequest);
            }
            catch (...)
            {
                if (!synchronized) identicalStart.arrive_and_drop();
                secondError = std::current_exception();
            }
        });
        identicalStart.arrive_and_wait();
        first.join();
        second.join();
        if (firstError) std::rethrow_exception(firstError);
        if (secondError) std::rethrow_exception(secondError);
        assert(firstConcurrent.decision->reviewDecisionId ==
               secondConcurrent.decision->reviewDecisionId);
        assert((firstConcurrent.outcome ==
                    RecommendationConversionProposalReviewPersistOutcome::recorded &&
                secondConcurrent.outcome ==
                    RecommendationConversionProposalReviewPersistOutcome::
                        existingIdentical) ||
               (secondConcurrent.outcome ==
                    RecommendationConversionProposalReviewPersistOutcome::recorded &&
                firstConcurrent.outcome ==
                    RecommendationConversionProposalReviewPersistOutcome::
                        existingIdentical));

        // Different request IDs both remain in history. The greatest database
        // decision ID, not timestamp or thread scheduling, defines current.
        RecommendationConversionProposalReviewPersistResult concurrentApprove;
        RecommendationConversionProposalReviewPersistResult concurrentReject;
        firstError = nullptr;
        secondError = nullptr;
        std::barrier conflictingStart{3};
        std::thread approveThread([&] {
            bool synchronized = false;
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                conflictingStart.arrive_and_wait();
                synchronized = true;
                concurrentApprove =
                    RecordRecommendationConversionProposalReviewDecision(
                        connection,
                        Request(firstProposalId,
                            RecommendationConversionProposalReviewDecision::approve,
                            "concurrent-approve"));
            }
            catch (...)
            {
                if (!synchronized) conflictingStart.arrive_and_drop();
                firstError = std::current_exception();
            }
        });
        std::thread rejectThread([&] {
            bool synchronized = false;
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                conflictingStart.arrive_and_wait();
                synchronized = true;
                concurrentReject =
                    RecordRecommendationConversionProposalReviewDecision(
                        connection,
                        Request(firstProposalId,
                            RecommendationConversionProposalReviewDecision::reject,
                            "concurrent-reject"));
            }
            catch (...)
            {
                if (!synchronized) conflictingStart.arrive_and_drop();
                secondError = std::current_exception();
            }
        });
        conflictingStart.arrive_and_wait();
        approveThread.join();
        rejectThread.join();
        if (firstError) std::rethrow_exception(firstError);
        if (secondError) std::rethrow_exception(secondError);
        assert(concurrentApprove.outcome ==
               RecommendationConversionProposalReviewPersistOutcome::recorded);
        assert(concurrentReject.outcome ==
               RecommendationConversionProposalReviewPersistOutcome::recorded);
        current = GetRecommendationConversionProposalCurrentReview(
            runtime, firstProposalId);
        const auto& expectedLatest =
            concurrentApprove.decision->reviewDecisionId >
                    concurrentReject.decision->reviewDecisionId
                ? *concurrentApprove.decision
                : *concurrentReject.decision;
        assert(current->latestDecision->reviewDecisionId ==
               expectedLatest.reviewDecisionId);
        assert(current->latestDecision->decision == expectedLatest.decision);

        // Database constraints reject invalid values and missing proposals.
        for (const std::pair<std::string, std::string>& invalid : {
                 std::pair<std::string, std::string>{
                     "INSERT INTO "
                     "experiment_recommendation_conversion_review_decision "
                     "(recommendation_conversion_proposal_id,decision,"
                     "decision_request_id) VALUES (" +
                         std::to_string(firstProposalId) +
                         ",'queued','invalid-decision');",
                     "23514"},
                 {"INSERT INTO "
                  "experiment_recommendation_conversion_review_decision "
                  "(recommendation_conversion_proposal_id,decision,"
                  "decision_request_id) VALUES (999999,'approve','bad-fk');",
                  "23503"},
                 {"INSERT INTO "
                  "experiment_recommendation_conversion_review_decision "
                  "(recommendation_conversion_proposal_id,decision,"
                  "decision_request_id,reason_text) VALUES (" +
                      std::to_string(firstProposalId) +
                      ",'approve','noncanonical-reason',' padded ');",
                  "23514"},
                 {"INSERT INTO "
                  "experiment_recommendation_conversion_review_decision "
                  "(recommendation_conversion_review_decision_id,"
                  "recommendation_conversion_proposal_id,decision,"
                  "decision_request_id) VALUES (-1," +
                      std::to_string(firstProposalId) +
                      ",'approve','negative-id');",
                  "23514"},
                 {"INSERT INTO "
                  "experiment_recommendation_conversion_review_decision "
                  "(recommendation_conversion_proposal_id,decision,"
                  "decision_request_id) VALUES (" +
                      std::to_string(firstProposalId) +
                      ",'approve','.invalid-request');",
                  "23514"},
                 {"INSERT INTO "
                  "experiment_recommendation_conversion_review_decision "
                  "(recommendation_conversion_proposal_id,decision,"
                  "decision_request_id,operator_identity) VALUES (" +
                      std::to_string(firstProposalId) +
                      ",'approve','noncanonical-operator',' padded ');",
                  "23514"}})
        {
            bool rejected = false;
            try
            {
                pqxx::work bad{owner};
                bad.exec("SET LOCAL search_path TO " + bad.quote_name(schema) +
                         ";");
                bad.exec(invalid.first);
                bad.commit();
            }
            catch (const pqxx::sql_error& error)
            {
                rejected = error.sqlstate() == invalid.second;
            }
            assert(rejected);
        }

        for (const std::string& statement : {
                 std::string{"UPDATE "
                             "experiment_recommendation_conversion_review_decision "
                             "SET reason_text='changed' WHERE "
                             "recommendation_conversion_review_decision_id="} +
                     std::to_string(approval.decision->reviewDecisionId),
                 std::string{"DELETE FROM "
                             "experiment_recommendation_conversion_review_decision "
                             "WHERE recommendation_conversion_review_decision_id="} +
                     std::to_string(approval.decision->reviewDecisionId),
                 std::string{"TRUNCATE "
                             "experiment_recommendation_conversion_review_decision"},
                 std::string{"INSERT INTO "
                             "experiment_recommendation_conversion_review_decision "
                             "(recommendation_conversion_review_decision_id,"
                             "recommendation_conversion_proposal_id,decision,"
                             "decision_request_id) VALUES (999999,"} +
                     std::to_string(firstProposalId) +
                     ",'approve','forged-order');",
                 std::string{"INSERT INTO "
                             "experiment_recommendation_conversion_review_decision "
                             "(recommendation_conversion_proposal_id,decision,"
                             "decision_request_id,decided_at) VALUES ("} +
                     std::to_string(firstProposalId) +
                     ",'approve','forged-decided-at',now());",
                 std::string{"INSERT INTO "
                             "experiment_recommendation_conversion_review_decision "
                             "(recommendation_conversion_proposal_id,decision,"
                             "decision_request_id,created_at) VALUES ("} +
                     std::to_string(firstProposalId) +
                     ",'approve','forged-created-at',now());",
                 std::string{"UPDATE "
                             "experiment_recommendation_conversion_proposal SET "
                             "source_value_canonical='changed' WHERE "
                             "recommendation_conversion_proposal_id="} +
                     std::to_string(firstProposalId)})
        {
            bool rejected = false;
            try
            {
                pqxx::work forbidden{runtime};
                forbidden.exec(statement);
                forbidden.commit();
            }
            catch (const pqxx::sql_error& error)
            {
                rejected = PermissionDenied(error);
            }
            assert(rejected);
        }

        // The owner cannot delete a proposal with append-only review history.
        bool restrictiveDeleteRejected = false;
        try
        {
            pqxx::work forbiddenDelete{owner};
            forbiddenDelete.exec("SET LOCAL search_path TO " +
                                 forbiddenDelete.quote_name(schema) + ";");
            forbiddenDelete.exec(
                "DELETE FROM experiment_recommendation_conversion_proposal "
                "WHERE recommendation_conversion_proposal_id=$1;",
                pqxx::params{firstProposalId});
            forbiddenDelete.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            restrictiveDeleteRejected = error.sqlstate() == "23503";
        }
        assert(restrictiveDeleteRejected);

        bool invalidLimitRejected = false;
        try
        {
            (void)ListRecommendationConversionProposalReviewDecisions(
                runtime, firstProposalId,
                kMaximumRecommendationConversionProposalReviewListLimit + 1);
        }
        catch (const std::invalid_argument&) { invalidLimitRejected = true; }
        assert(invalidLimitRejected);

        pqxx::read_transaction unchanged{owner};
        unchanged.exec("SET LOCAL search_path TO " +
                       unchanged.quote_name(schema) + ";");
        const pqxx::row state = unchanged.exec(
            "SELECT count(*) AS experiment_count,min(status) AS status,"
            "min(marker) AS marker FROM experiment;").one_row();
        assert(state["experiment_count"].as<int>() == 1);
        assert(state["status"].as<std::string>() == "paused");
        assert(state["marker"].as<std::string>() == "unchanged");
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
    std::cout
        << "ExperimentRecommendationConversionProposalReviewRepositoryTests "
           "passed\n";
}
