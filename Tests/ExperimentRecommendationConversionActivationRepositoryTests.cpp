#include "../Sources/ExperimentRecommendationConversionActivationRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionActivationService.hpp"
#include "../Sources/ExperimentRecommendationConversionExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <barrier>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
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

ExperimentInvocationConfiguration SourceInvocation()
{
    ExperimentInvocationConfiguration invocation;
    invocation.configuration.symbol = "eurusd";
    invocation.configuration.predictionHorizon = 12;
    invocation.configuration.labelThreshold = 0.001;
    invocation.configuration.coreLrMult = 1.0;
    invocation.configuration.headLrMult = 5.0;
    invocation.configuration.targetEpochs = 120;
    invocation.configuration.trainStartDate = "2010-01-01";
    invocation.configuration.trainEndDate = "2025-01-01";
    invocation.configuration.inferStartDate = "2025-01-01";
    invocation.configuration.inferEndDate = "2026-01-01";
    invocation.checkpointInterval = 15;
    invocation.resumeModelId = 77;
    return invocation;
}

ProposedExperimentSpecification Proposal(double proposed, int variant)
{
    RecommendationConversionRequest request;
    request.recommendationExists = true;
    request.recommendationId = 42;
    request.recommendationStatus = RecommendationStatus::approved;
    request.sourceExperimentId = 17;
    request.recommendationSourceExperimentId = 17;
    request.sourceInvocation = SourceInvocation();
    request.mutations.push_back(
        {kCoreLrMult, "1", CanonicalRecommendationDouble(proposed)});
    request.reviewAuthorization.present = true;
    request.reviewAuthorization.recommendationId = 42;
    request.reviewAuthorization.latestAction = RecommendationReviewAction::approve;
    request.reviewAuthorization.resultingStatus = RecommendationStatus::approved;
    request.reviewAuthorization.latestActionEffective = true;
    request.reviewAuthorization.authorizationCanonical =
        "review_authorization_v1;variant=" + std::to_string(variant);
    request.reviewAuthorization.authorizationHash = RecommendationCanonicalHash(
        request.reviewAuthorization.authorizationCanonical);
    request.evaluation.state = RecommendationConversionEvidenceState::completed;
    request.evaluation.valid = true;
    request.evaluation.recommendationId = 42;
    request.evaluation.sourceExperimentId = 17;
    request.evaluation.eligibility = RecommendationEligibility::eligible;
    request.evaluation.disposition =
        RecommendationEvaluationDisposition::advisoryReady;
    request.evaluation.evaluationIdentityCanonical =
        "evaluation_identity_v1;variant=" + std::to_string(variant);
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    request.evaluation.evaluationPolicyCanonical = "evaluation_policy_v1";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);
    request.score.state = RecommendationConversionEvidenceState::completed;
    request.score.valid = true;
    request.score.recommendationId = 42;
    request.score.finalScore = 0.75;
    request.score.scoringPolicyCanonical = "scoring_policy_v1";
    request.score.scoringPolicyHash = RecommendationCanonicalHash(
        request.score.scoringPolicyCanonical);
    request.evaluation.scoringPolicyHash = request.score.scoringPolicyHash;
    auto proposedInvocation = request.sourceInvocation;
    proposedInvocation.configuration.coreLrMult = proposed;
    const auto semantic = BuildRecommendationCandidateIdentity(
        proposedInvocation.configuration);
    const auto invocation = BuildRecommendationInvocationIdentity(
        proposedInvocation);
    request.recommendationSemanticCanonical = semantic.canonicalText;
    request.recommendationSemanticHash = semantic.hash;
    request.recommendationInvocationCanonical = invocation.canonicalText;
    request.recommendationInvocationHash = invocation.hash;
    const auto result = BuildProposedExperimentSpecification(request);
    assert(result.eligibility.eligible && result.proposal);
    return *result.proposal;
}

RecommendationConversionProposalReviewRequest Approval(
    long long proposalId,
    int variant)
{
    RecommendationConversionProposalReviewRequest request;
    request.proposalId = proposalId;
    request.decision = RecommendationConversionProposalReviewDecision::approve;
    request.requestId = "activation-approval-" + std::to_string(variant);
    request.operatorIdentity = "isolated test operator";
    request.reasonText = "Explicit test approval.";
    return request;
}

PersistedRecommendationConversionExecution CreateExecution(
    pqxx::connection& runtime,
    double proposed,
    int variant)
{
    const auto persisted = PersistRecommendationConversionProposal(
        runtime, Proposal(proposed, variant));
    const auto review = RecordRecommendationConversionProposalReviewDecision(
        runtime, Approval(persisted.persisted.proposalId, variant));
    assert(review.decision);
    const auto execution = ExecuteApprovedRecommendationConversionProposal(
        runtime, persisted.persisted.proposalId);
    assert(execution.outcome == RecommendationConversionExecutionOutcome::created);
    assert(execution.execution);
    return *execution.execution;
}

void SetExperimentStatus(
    pqxx::connection& owner,
    const std::string& schema,
    long long experimentId,
    const std::string& status)
{
    pqxx::work transaction{owner};
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
    transaction.exec(
        "UPDATE experiment SET status=$1 WHERE experiment_id=$2;",
        pqxx::params{status, experimentId});
    transaction.commit();
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
        "phase4c_conversion_activation_" + std::to_string(getpid());
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
            setup.exec("SET LOCAL search_path TO " + setup.quote_name(schema) + ";");
            setup.exec(
                "CREATE TABLE experiment("
                "experiment_id bigserial PRIMARY KEY,symbol text NOT NULL,"
                "prediction_horizon integer NOT NULL,c_next_threshold double precision NOT NULL,"
                "core_lr_mult double precision,head_lr_mult double precision,"
                "target_epochs integer NOT NULL,checkpoint_interval integer NOT NULL,"
                "train_start timestamptz NOT NULL,train_end timestamptz NOT NULL,"
                "infer_start timestamptz,infer_end timestamptz,resume_model_id bigint,"
                "duplicate_nonce bigint NOT NULL DEFAULT 0,status text NOT NULL,"
                "phase text NOT NULL,invocation_mode text,updated_at timestamptz NOT NULL DEFAULT now(),"
                "worker_pid integer,current_operation text,current_epoch integer,"
                "worker_started_at timestamptz,started_at timestamptz,"
                "completed_at timestamptz,exit_code integer,error_message text,"
                "last_model_id bigint,marker text);"
                "CREATE UNIQUE INDEX experiment_unique_identity_uidx ON experiment("
                "symbol,prediction_horizon,c_next_threshold,"
                "coalesce(core_lr_mult,'-infinity'::double precision),"
                "coalesce(head_lr_mult,'-infinity'::double precision),target_epochs,"
                "checkpoint_interval,train_start,train_end,"
                "coalesce(infer_start,'-infinity'::timestamptz),"
                "coalesce(infer_end,'-infinity'::timestamptz),"
                "coalesce(resume_model_id,-1),duplicate_nonce) WHERE status<>'cancelled';"
                "CREATE TABLE model(model_id bigint PRIMARY KEY,marker text NOT NULL);"
                "CREATE TABLE experiment_recommendation("
                "recommendation_id bigint PRIMARY KEY,source_experiment_id bigint NOT NULL "
                "REFERENCES experiment(experiment_id),status text NOT NULL);"
                "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,"
                "c_next_threshold,core_lr_mult,head_lr_mult,target_epochs,"
                "checkpoint_interval,train_start,train_end,infer_start,infer_end,"
                "resume_model_id,status,phase,marker) VALUES (17,'eurusd',12,"
                "0.001,1,5,120,15,'2010-01-01 America/Chicago',"
                "'2025-01-01 America/Chicago','2025-01-01 America/Chicago',"
                "'2026-01-01 America/Chicago',77,'paused','train','unchanged');"
                "INSERT INTO model VALUES (77,'unchanged');"
                "INSERT INTO experiment_recommendation VALUES (42,17,'approved');");
            for (const char* migration : {
                     "Database/migrations/036_experiment_recommendation_conversion_proposal.sql",
                     "Database/migrations/037_experiment_recommendation_conversion_review.sql",
                     "Database/migrations/038_experiment_recommendation_conversion_execution.sql"})
            {
                setup.exec(ReadFile(migration));
            }
            setup.exec("GRANT SELECT,INSERT,UPDATE ON experiment TO pqxx;"
                       "GRANT USAGE ON SEQUENCE experiment_experiment_id_seq TO pqxx;"
                       "GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(!RecommendationConversionActivationSchemaExists(runtime));
        {
            pqxx::work migration{owner};
            migration.exec("SET LOCAL search_path TO " +
                           migration.quote_name(schema) + ";");
            const std::string sql = ReadFile(
                "Database/migrations/039_experiment_recommendation_conversion_activation.sql");
            migration.exec(sql);
            migration.exec(sql);
            migration.exec(ReadFile(
                "Tests/ExperimentRecommendationConversionActivationMigrationTests.sql"));
            migration.commit();
        }
        assert(RecommendationConversionActivationSchemaExists(runtime));

        const auto missing = ActivateRecommendationConversionExecution(
            runtime, 999999);
        assert(missing.outcome ==
               RecommendationConversionActivationOutcome::notFound);

        const auto firstExecution = CreateExecution(runtime, 1.25, 1);
        const auto activated = ActivateRecommendationConversionExecution(
            runtime, firstExecution.executionId);
        assert(activated.outcome ==
               RecommendationConversionActivationOutcome::activated);
        assert(activated.activation);
        assert(activated.activation->executionId == firstExecution.executionId);
        assert(activated.activation->proposalId == firstExecution.proposalId);
        assert(activated.activation->reviewDecisionId ==
               firstExecution.reviewDecisionId);
        assert(activated.activation->experimentId == firstExecution.experimentId);
        assert(FindRecommendationConversionActivation(
            runtime, activated.activation->activationId));
        assert(FindRecommendationConversionActivationByExecution(
            runtime, firstExecution.executionId));
        assert(FindRecommendationConversionActivationByExperiment(
            runtime, firstExecution.experimentId));

        std::string activatedUpdatedAt;
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " +
                        verify.quote_name(schema) + ";");
            const pqxx::row row = verify.exec(
                "SELECT status,phase,worker_pid,current_operation,current_epoch,"
                "updated_at::text AS updated_at FROM experiment WHERE "
                "experiment_id=$1;",
                pqxx::params{firstExecution.experimentId}).one_row();
            assert(row["status"].as<std::string>() == "pending");
            assert(row["phase"].as<std::string>() == "train");
            assert(row["worker_pid"].is_null());
            assert(row["current_operation"].is_null());
            assert(row["current_epoch"].is_null());
            activatedUpdatedAt = row["updated_at"].as<std::string>();
            assert(verify.exec("SELECT count(*) FROM experiment;")
                       .one_row()[0].as<int>() == 2);
            assert(verify.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_activation;")
                       .one_row()[0].as<int>() == 1);
        }

        const auto replay = ActivateRecommendationConversionExecution(
            runtime, firstExecution.executionId);
        assert(replay.outcome ==
               RecommendationConversionActivationOutcome::existingIdentical);
        assert(replay.activation->activationId == activated.activation->activationId);
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " +
                        verify.quote_name(schema) + ";");
            assert(verify.exec(
                "SELECT updated_at::text FROM experiment WHERE experiment_id=$1;",
                pqxx::params{firstExecution.experimentId})
                       .one_row()[0].as<std::string>() == activatedUpdatedAt);
        }

        const auto concurrentExecution = CreateExecution(runtime, 1.5, 2);
        RecommendationConversionActivationResult left;
        RecommendationConversionActivationResult right;
        std::exception_ptr leftError;
        std::exception_ptr rightError;
        std::barrier start{3};
        auto run = [&](RecommendationConversionActivationResult& result,
                       std::exception_ptr& error) {
            bool synchronized = false;
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                start.arrive_and_wait();
                synchronized = true;
                result = ActivateRecommendationConversionExecution(
                    connection, concurrentExecution.executionId);
            }
            catch (...)
            {
                if (!synchronized) start.arrive_and_drop();
                error = std::current_exception();
            }
        };
        std::thread leftThread{run, std::ref(left), std::ref(leftError)};
        std::thread rightThread{run, std::ref(right), std::ref(rightError)};
        start.arrive_and_wait();
        leftThread.join();
        rightThread.join();
        if (leftError) std::rethrow_exception(leftError);
        if (rightError) std::rethrow_exception(rightError);
        assert(left.activation->activationId == right.activation->activationId);
        assert((left.outcome ==
                    RecommendationConversionActivationOutcome::activated &&
                right.outcome == RecommendationConversionActivationOutcome::
                    existingIdentical) ||
               (right.outcome ==
                    RecommendationConversionActivationOutcome::activated &&
                left.outcome == RecommendationConversionActivationOutcome::
                    existingIdentical));

        int variant = 10;
        double proposed = 2.0;
        long long invalidStateExecutionId = 0;
        for (const std::string& status : {
                 "pending", "running", "completed", "failed", "cancelled"})
        {
            const auto execution = CreateExecution(
                runtime, proposed, variant);
            SetExperimentStatus(
                owner, schema, execution.experimentId, status);
            const auto result = ActivateRecommendationConversionExecution(
                runtime, execution.executionId);
            if (invalidStateExecutionId == 0)
                invalidStateExecutionId = execution.executionId;
            assert(result.outcome ==
                   RecommendationConversionActivationOutcome::invalidState);
            assert(result.reason == "activation_requires_paused_status");
            assert(!FindRecommendationConversionActivationByExecution(
                runtime, execution.executionId));
            proposed += 0.25;
            ++variant;
        }

        const auto changedIdentityExecution = CreateExecution(runtime, 3.25, 25);
        {
            pqxx::work changed{owner};
            changed.exec("SET LOCAL search_path TO " +
                         changed.quote_name(schema) + ";");
            changed.exec(
                "UPDATE experiment SET duplicate_nonce=1 WHERE "
                "experiment_id=$1;",
                pqxx::params{changedIdentityExecution.experimentId});
            changed.commit();
        }
        const auto changedIdentity = ActivateRecommendationConversionExecution(
            runtime, changedIdentityExecution.executionId);
        assert(changedIdentity.outcome ==
               RecommendationConversionActivationOutcome::invalidState);
        assert(changedIdentity.reason ==
               "activation_requires_zero_duplicate_nonce");
        assert(!FindRecommendationConversionActivationByExecution(
            runtime, changedIdentityExecution.executionId));

        const auto auditFailureExecution = CreateExecution(runtime, 3.5, 30);
        {
            pqxx::work trigger{owner};
            trigger.exec("SET LOCAL search_path TO " +
                         trigger.quote_name(schema) + ";");
            trigger.exec(
                "CREATE FUNCTION reject_activation_insert() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'forced audit "
                "failure'; END $$; CREATE TRIGGER reject_activation_insert "
                "BEFORE INSERT ON "
                "experiment_recommendation_conversion_activation FOR EACH "
                "ROW EXECUTE FUNCTION reject_activation_insert();");
            trigger.commit();
        }
        bool auditFailure = false;
        try
        {
            (void)ActivateRecommendationConversionExecution(
                runtime, auditFailureExecution.executionId);
        }
        catch (const pqxx::sql_error&)
        {
            auditFailure = true;
        }
        assert(auditFailure);
        {
            pqxx::work dropTrigger{owner};
            dropTrigger.exec("SET LOCAL search_path TO " +
                             dropTrigger.quote_name(schema) + ";");
            dropTrigger.exec(
                "DROP TRIGGER reject_activation_insert ON "
                "experiment_recommendation_conversion_activation;"
                "DROP FUNCTION reject_activation_insert();");
            dropTrigger.commit();
        }
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " +
                        verify.quote_name(schema) + ";");
            assert(verify.exec(
                "SELECT status FROM experiment WHERE experiment_id=$1;",
                pqxx::params{auditFailureExecution.experimentId})
                       .one_row()[0].as<std::string>() == "paused");
            assert(!FindRecommendationConversionActivationByExecution(
                runtime, auditFailureExecution.executionId));
        }

        const auto lifecycleFailureExecution = CreateExecution(runtime, 3.75, 31);
        {
            pqxx::work trigger{owner};
            trigger.exec("SET LOCAL search_path TO " +
                         trigger.quote_name(schema) + ";");
            trigger.exec(
                "CREATE FUNCTION reject_activation_update() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN IF NEW.experiment_id=" +
                std::to_string(lifecycleFailureExecution.experimentId) +
                " THEN RAISE EXCEPTION 'forced lifecycle failure'; END IF; "
                "RETURN NEW; END $$; CREATE TRIGGER reject_activation_update "
                "BEFORE UPDATE ON experiment FOR EACH ROW EXECUTE FUNCTION "
                "reject_activation_update();");
            trigger.commit();
        }
        bool lifecycleFailure = false;
        try
        {
            (void)ActivateRecommendationConversionExecution(
                runtime, lifecycleFailureExecution.executionId);
        }
        catch (const pqxx::sql_error&)
        {
            lifecycleFailure = true;
        }
        assert(lifecycleFailure);
        {
            pqxx::work dropTrigger{owner};
            dropTrigger.exec("SET LOCAL search_path TO " +
                             dropTrigger.quote_name(schema) + ";");
            dropTrigger.exec(
                "DROP TRIGGER reject_activation_update ON experiment;"
                "DROP FUNCTION reject_activation_update();");
            dropTrigger.commit();
        }
        assert(!FindRecommendationConversionActivationByExecution(
            runtime, lifecycleFailureExecution.executionId));

        bool mismatchedProvenanceRejected = false;
        try
        {
            pqxx::work invalid{owner};
            invalid.exec("SET LOCAL search_path TO " +
                         invalid.quote_name(schema) + ";");
            invalid.exec(
                "INSERT INTO experiment_recommendation_conversion_activation ("
                "recommendation_conversion_execution_id,"
                "recommendation_conversion_proposal_id,"
                "recommendation_conversion_review_decision_id,experiment_id,"
                "activation_contract_version,previous_status,previous_phase,"
                "resulting_status,resulting_phase,activation_identity_canonical,"
                "activation_identity_hash) VALUES ($1,$2,$3,17,1,'paused',"
                "'train','pending','train','invalid','invalid');",
                pqxx::params{
                    auditFailureExecution.executionId,
                    auditFailureExecution.proposalId,
                    auditFailureExecution.reviewDecisionId});
            invalid.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            mismatchedProvenanceRejected = error.sqlstate() == "23503";
        }
        assert(mismatchedProvenanceRejected);

        for (const std::string& statement : {
                 "UPDATE experiment_recommendation_conversion_activation SET "
                 "activation_contract_version=1",
                 "DELETE FROM experiment_recommendation_conversion_activation",
                 "TRUNCATE experiment_recommendation_conversion_activation"})
        {
            bool denied = false;
            try
            {
                pqxx::work forbidden{runtime};
                forbidden.exec(statement);
                forbidden.commit();
            }
            catch (const pqxx::sql_error& error)
            {
                denied = error.sqlstate() == "42501" ||
                    (error.sqlstate().empty() &&
                     std::string{error.what()}.find("permission denied") !=
                         std::string::npos);
            }
            assert(denied);
        }

        std::ostringstream output;
        std::ostringstream errors;
        assert(RunActivateRecommendationConversionExecutionCommand(
            runtimeConnectionString, firstExecution.executionId,
            output, errors) == 0);
        assert(output.str().find(
            "RECOMMENDATION_CONVERSION_ALREADY_ACTIVATED") !=
            std::string::npos);
        assert(output.str().find("worker_started=false") != std::string::npos);
        assert(errors.str().empty());

        output.str({});
        output.clear();
        assert(RunRecommendationConversionActivationStatusCommand(
            runtimeConnectionString, activated.activation->activationId,
            output) == 0);
        assert(output.str().find("RECOMMENDATION_CONVERSION_ACTIVATION") !=
               std::string::npos);

        output.str({});
        output.clear();
        assert(RunRecommendationConversionActivationStatusCommand(
            runtimeConnectionString, 999999, output) == 1);
        assert(output.str().find(
            "RECOMMENDATION_CONVERSION_ACTIVATION_NOT_FOUND") !=
            std::string::npos);

        output.str({});
        output.clear();
        errors.str({});
        errors.clear();
        assert(RunActivateRecommendationConversionExecutionCommand(
            runtimeConnectionString, 999999, output, errors) == 1);
        assert(errors.str().find(
            "RECOMMENDATION_CONVERSION_EXECUTION_NOT_FOUND") !=
            std::string::npos);

        output.str({});
        output.clear();
        errors.str({});
        errors.clear();
        assert(RunActivateRecommendationConversionExecutionCommand(
            runtimeConnectionString, invalidStateExecutionId,
            output, errors) == 2);
        assert(errors.str().find(
            "RECOMMENDATION_CONVERSION_ACTIVATION_INVALID_STATE") !=
            std::string::npos);
        assert(output.str().empty());

        pqxx::read_transaction unchanged{owner};
        unchanged.exec("SET LOCAL search_path TO " +
                       unchanged.quote_name(schema) + ";");
        const pqxx::row source = unchanged.exec(
            "SELECT status,phase,marker FROM experiment WHERE experiment_id=17;")
            .one_row();
        assert(source["status"].as<std::string>() == "paused");
        assert(source["phase"].as<std::string>() == "train");
        assert(source["marker"].as<std::string>() == "unchanged");
        assert(unchanged.exec(
            "SELECT status FROM experiment_recommendation WHERE "
            "recommendation_id=42;").one_row()[0].as<std::string>() ==
               "approved");
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
    std::cout << "ExperimentRecommendationConversionActivationRepositoryTests "
                 "passed\n";
}
