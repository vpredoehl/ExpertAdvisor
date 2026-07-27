#include "../Sources/ExperimentRecommendationConversionWorkflowRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionWorkflowService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
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

void InsertProposal(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    const std::string canonical =
        "workflow-proposal-" + std::to_string(proposalId);
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_proposal VALUES "
        "($1,42,17,1,$2,$3,now());",
        pqxx::params{
            proposalId, canonical, RecommendationCanonicalHash(canonical)});
}

void InsertReview(
    pqxx::transaction_base& transaction,
    long long reviewId,
    long long proposalId,
    const char* decision)
{
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_review_decision "
        "VALUES ($1,$2,$3,now());",
        pqxx::params{reviewId, proposalId, decision});
}

void InsertExecution(
    pqxx::transaction_base& transaction,
    long long executionId,
    long long proposalId,
    long long reviewId,
    long long experimentId)
{
    const std::string canonical =
        "workflow-execution-" + std::to_string(executionId);
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_execution VALUES "
        "($1,$2,$3,$4,1,'approve',$5,$6,now());",
        pqxx::params{
            executionId, proposalId, reviewId, experimentId, canonical,
            RecommendationCanonicalHash(canonical)});
}

void InsertActivation(
    pqxx::transaction_base& transaction,
    long long activationId,
    long long executionId,
    long long proposalId,
    long long reviewId,
    long long experimentId)
{
    const std::string canonical =
        "workflow-activation-" + std::to_string(activationId);
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_activation VALUES "
        "($1,$2,$3,$4,$5,1,'paused','train','pending','train',$6,$7,now());",
        pqxx::params{
            activationId, executionId, proposalId, reviewId, experimentId,
            canonical, RecommendationCanonicalHash(canonical)});
}

void InsertExperiment(
    pqxx::transaction_base& transaction,
    long long experimentId,
    const char* status,
    const char* phase,
    std::optional<int> workerPid = std::nullopt,
    std::optional<std::string> operation = std::nullopt)
{
    transaction.exec(
        "INSERT INTO experiment VALUES ($1,$2,$3,$4,$5,NULL,now());",
        pqxx::params{
            experimentId, status, phase, workerPid, operation});
}

void AssertState(
    pqxx::connection& connection,
    long long proposalId,
    RecommendationConversionWorkflowState state)
{
    const auto workflow = FindRecommendationConversionWorkflow(
        connection, proposalId);
    assert(workflow);
    assert(workflow->derivation.state == state);
    assert(workflow->derivation.integrity ==
           RecommendationConversionWorkflowIntegrity::consistent);
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
        "phase4c_conversion_workflow_" + std::to_string(getpid());
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
CREATE TABLE experiment(
    experiment_id bigint PRIMARY KEY,
    status text NOT NULL,
    phase text NOT NULL,
    worker_pid integer,
    current_operation text,
    current_epoch integer,
    updated_at timestamptz NOT NULL);
)SQL");
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                       setup.quote_name(schema) + " TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(!RecommendationConversionWorkflowSchemasExist(runtime));

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
                "GRANT SELECT ON "
                "experiment_recommendation_conversion_activation TO pqxx;");

            for (long long proposalId = 1; proposalId <= 11; ++proposalId)
                InsertProposal(fixtures, proposalId);

            InsertReview(fixtures, 20, 2, "approve");
            InsertReview(fixtures, 21, 2, "reject");
            InsertReview(fixtures, 30, 3, "approve");
            for (long long proposalId = 4; proposalId <= 11; ++proposalId)
                InsertReview(fixtures, 100 + proposalId, proposalId, "approve");

            long long executionId = 200;
            long long experimentId = 300;
            for (long long proposalId = 4; proposalId <= 9; ++proposalId)
            {
                InsertExperiment(
                    fixtures, experimentId,
                    proposalId == 4 ? "paused" :
                    proposalId == 5 ? "pending" :
                    proposalId == 6 ? "running" :
                    proposalId == 7 ? "completed" :
                    proposalId == 8 ? "failed" : "cancelled",
                    "train",
                    proposalId == 6 ? std::optional<int>{444} : std::nullopt,
                    proposalId == 6
                        ? std::optional<std::string>{"training"}
                        : std::nullopt);
                InsertExecution(
                    fixtures, executionId, proposalId, 100 + proposalId,
                    experimentId);
                if (proposalId != 4)
                    InsertActivation(
                        fixtures, 400 + proposalId, executionId, proposalId,
                        100 + proposalId, experimentId);
                ++executionId;
                ++experimentId;
            }

            InsertActivation(fixtures, 410, 999, 10, 110, 999);
            InsertExecution(fixtures, 211, 11, 111, 9999);
            fixtures.commit();
        }

        assert(RecommendationConversionWorkflowSchemasExist(runtime));
        assert(!FindRecommendationConversionWorkflow(runtime, 99999));
        AssertState(runtime, 1,
                    RecommendationConversionWorkflowState::pendingReview);
        AssertState(runtime, 2, RecommendationConversionWorkflowState::rejected);
        AssertState(
            runtime, 3,
            RecommendationConversionWorkflowState::approvedNotExecuted);
        AssertState(
            runtime, 4, RecommendationConversionWorkflowState::executedPaused);
        AssertState(
            runtime, 5,
            RecommendationConversionWorkflowState::activatedPending);
        AssertState(
            runtime, 6,
            RecommendationConversionWorkflowState::schedulerClaimedOrRunning);
        const auto legacyRunning =
            FindRecommendationConversionWorkflow(runtime, 6);
        assert(legacyRunning->experiment);
        assert(legacyRunning->experiment->currentOperation ==
               std::optional<std::string>{"train"});
        AssertState(runtime, 7, RecommendationConversionWorkflowState::completed);
        AssertState(runtime, 8, RecommendationConversionWorkflowState::failed);
        AssertState(runtime, 9, RecommendationConversionWorkflowState::cancelled);

        {
            pqxx::work laterReview{owner};
            laterReview.exec("SET LOCAL search_path TO " +
                             laterReview.quote_name(schema) + ";");
            InsertReview(laterReview, 999, 4, "reject");
            laterReview.commit();
        }
        AssertState(runtime, 4, RecommendationConversionWorkflowState::rejected);
        const auto rejectedAfterExecution =
            FindRecommendationConversionWorkflow(runtime, 4);
        assert(rejectedAfterExecution->execution);
        assert(rejectedAfterExecution->latestReview->reviewDecisionId == 999);

        const auto latest = FindRecommendationConversionWorkflow(runtime, 2);
        assert(latest->latestReview->reviewDecisionId == 21);
        assert(latest->latestReview->decision == "reject");

        const auto orphanActivation = FindRecommendationConversionWorkflow(
            runtime, 10);
        assert(orphanActivation->derivation.state ==
               RecommendationConversionWorkflowState::inconsistent);
        assert(orphanActivation->derivation.diagnosticCodes.front() ==
               "activation_without_execution");
        const auto missingExperiment = FindRecommendationConversionWorkflow(
            runtime, 11);
        assert(missingExperiment->derivation.state ==
               RecommendationConversionWorkflowState::inconsistent);
        assert(missingExperiment->derivation.diagnosticCodes.front() ==
               "experiment_missing");

        assert(FindRecommendationConversionWorkflowByExecution(runtime, 200));
        assert(FindRecommendationConversionWorkflowByExperiment(runtime, 300));
        assert(!FindRecommendationConversionWorkflowByExecution(runtime, 99999));

        const auto listed = ListRecommendationConversionWorkflows(runtime, 3);
        assert(listed.size() == 3);
        assert(listed[0].proposalId == 11);
        assert(listed[1].proposalId == 10);
        assert(listed[2].proposalId == 9);
        bool invalidLimit = false;
        try
        {
            (void)ListRecommendationConversionWorkflows(runtime, 0);
        }
        catch (const std::invalid_argument&)
        {
            invalidLimit = true;
        }
        assert(invalidLimit);

        std::ostringstream output;
        std::ostringstream errors;
        assert(RunRecommendationConversionWorkflowCommand(
            runtimeConnectionString, 5, output, errors) == 0);
        assert(output.str().find("workflow_state=activated_pending") !=
               std::string::npos);
        assert(output.str().find("read_only=true") != std::string::npos);
        assert(errors.str().empty());

        output.str({});
        output.clear();
        errors.str({});
        errors.clear();
        assert(RunRecommendationConversionWorkflowCommand(
            runtimeConnectionString, 10, output, errors) == 0);
        assert(output.str().find("integrity_status=inconsistent") !=
               std::string::npos);

        output.str({});
        output.clear();
        assert(RunRecommendationConversionWorkflowCommand(
            runtimeConnectionString, 99999, output, errors) == 1);
        assert(errors.str().find(
            "RECOMMENDATION_CONVERSION_WORKFLOW_NOT_FOUND") !=
            std::string::npos);

        output.str({});
        output.clear();
        errors.str({});
        errors.clear();
        assert(RunListRecommendationConversionWorkflowsCommand(
            runtimeConnectionString,
            RecommendationConversionWorkflowState::activatedPending,
            10, output, errors) == 0);
        assert(output.str().find("proposal_id=5") != std::string::npos);
        assert(output.str().find(
            "RECOMMENDATION_CONVERSION_WORKFLOW_LIST_COMPLETE,count=1") !=
            std::string::npos);

        pqxx::read_transaction verify{owner};
        verify.exec("SET LOCAL search_path TO " + verify.quote_name(schema) +
                    ";");
        assert(verify.exec(
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_proposal;")
                   .one_row()[0].as<int>() == 11);
        assert(verify.exec(
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_review_decision;")
                   .one_row()[0].as<int>() == 12);
        assert(verify.exec(
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution;")
                   .one_row()[0].as<int>() == 7);
        assert(verify.exec(
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_activation;")
                   .one_row()[0].as<int>() == 6);
        assert(verify.exec("SELECT count(*) FROM experiment;")
                   .one_row()[0].as<int>() == 6);
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
