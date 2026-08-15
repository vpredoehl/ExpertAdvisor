#include "../Sources/ExperimentRecommendationConversionExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionExecutionService.hpp"
#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"
#include <exception>

#include <barrier>
#include <cassert>
#include <cstdlib>
#include <functional>
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

ProposedExperimentSpecification Proposal(
    double proposed,
    int variant,
    Donchian20Mode donchian20Mode = Donchian20Mode::Enabled)
{
    RecommendationConversionRequest request;
    request.recommendationExists = true;
    request.recommendationId = 42;
    request.recommendationStatus = RecommendationStatus::approved;
    request.sourceExperimentId = 17;
    request.recommendationSourceExperimentId = 17;
    request.sourceInvocation = SourceInvocation();
    request.sourceInvocation.configuration.donchian20Mode = donchian20Mode;
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
    request.evaluation.disposition = RecommendationEvaluationDisposition::advisoryReady;
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
    ExperimentInvocationConfiguration proposedInvocation = request.sourceInvocation;
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

RecommendationConversionProposalReviewRequest Review(
    long long proposalId,
    RecommendationConversionProposalReviewDecision decision,
    const std::string& requestId)
{
    RecommendationConversionProposalReviewRequest request;
    request.proposalId = proposalId;
    request.decision = decision;
    request.requestId = requestId;
    request.operatorIdentity = "test operator";
    request.reasonText = "Explicit isolated test decision.";
    return request;
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
        "phase4c_conversion_execution_" + std::to_string(getpid());
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
                "donchian20_mode text NOT NULL DEFAULT 'enabled',"
                "donchian_lookback integer NOT NULL DEFAULT 20,"
                "feature_warmup_scope text NOT NULL DEFAULT 'legacy_cold_boundary',"
                "duplicate_nonce bigint NOT NULL DEFAULT 0,status text NOT NULL,"
                "phase text NOT NULL,invocation_mode text,updated_at timestamptz NOT NULL DEFAULT now(),"
                "worker_pid integer,current_operation text,current_epoch integer,marker text);"
                "CREATE UNIQUE INDEX experiment_unique_identity_uidx ON experiment("
                "symbol,prediction_horizon,c_next_threshold,"
                "coalesce(core_lr_mult,'-infinity'::double precision),"
                "coalesce(head_lr_mult,'-infinity'::double precision),target_epochs,"
                "checkpoint_interval,train_start,train_end,"
                "coalesce(infer_start,'-infinity'::timestamptz),"
                "coalesce(infer_end,'-infinity'::timestamptz),"
                "coalesce(resume_model_id,-1),donchian20_mode,donchian_lookback,"
                "feature_warmup_scope,duplicate_nonce) WHERE status<>'cancelled';"
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
                const std::string sql = ReadFile(migration);
                setup.exec(sql);
                setup.exec(sql);
            }
            setup.exec(ReadFile(
                "Tests/ExperimentRecommendationConversionExecutionMigrationTests.sql"));
            setup.exec("GRANT SELECT,INSERT ON experiment TO pqxx;"
                       "GRANT USAGE ON SEQUENCE experiment_experiment_id_seq "
                       "TO pqxx;");
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        assert(RecommendationConversionExecutionSchemaExists(runtime));
        const auto firstProposal = PersistRecommendationConversionProposal(
            runtime, Proposal(1.25, 1, Donchian20Mode::ZeroAblation));
        const long long firstId = firstProposal.persisted.proposalId;

        const auto missing = ExecuteApprovedRecommendationConversionProposal(
            runtime, 999999);
        assert(missing.outcome ==
               RecommendationConversionExecutionOutcome::proposalNotFound);
        const auto pending = ExecuteApprovedRecommendationConversionProposal(
            runtime, firstId);
        assert(pending.outcome ==
               RecommendationConversionExecutionOutcome::pendingReview);
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunExecuteApprovedRecommendationConversionProposalCommand(
                runtimeConnectionString, firstId, output, errors) == 2);
            assert(errors.str().find("CONVERSION_PROPOSAL_NOT_APPROVED") !=
                   std::string::npos);
            assert(errors.str().find("review_disposition=pending_review") !=
                   std::string::npos);
            assert(output.str().empty());
        }
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " +
                        verify.quote_name(schema) + ";");
            assert(verify.exec("SELECT count(*) FROM experiment;")
                       .one_row()[0].as<int>() == 1);
        }
        const auto rejection = RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Review(firstId,
                   RecommendationConversionProposalReviewDecision::reject,
                   "reject-first"));
        assert(rejection.outcome ==
               RecommendationConversionProposalReviewPersistOutcome::recorded);
        assert(rejection.decision);
        // The database provenance link itself rejects a non-approving review;
        // repository validation is not the only authorization boundary.
        bool rejectedAuthorizationDenied = false;
        try
        {
            pqxx::work invalidAuthorization{owner};
            invalidAuthorization.exec(
                "SET LOCAL search_path TO " +
                invalidAuthorization.quote_name(schema) + ";");
            invalidAuthorization.exec(
                "INSERT INTO experiment_recommendation_conversion_execution ("
                "recommendation_conversion_proposal_id,"
                "recommendation_conversion_review_decision_id,experiment_id,"
                "execution_contract_version,authorization_decision,"
                "execution_identity_canonical,execution_identity_hash) VALUES "
                "($1,$2,17,1,'approve','invalid','invalid');",
                pqxx::params{firstId, rejection.decision->reviewDecisionId});
            invalidAuthorization.commit();
        }
        catch (const pqxx::sql_error& error)
        {
            rejectedAuthorizationDenied = error.sqlstate() == "23503";
        }
        assert(rejectedAuthorizationDenied);
        const auto rejected = ExecuteApprovedRecommendationConversionProposal(
            runtime, firstId);
        assert(rejected.outcome ==
               RecommendationConversionExecutionOutcome::rejected);
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunExecuteApprovedRecommendationConversionProposalCommand(
                runtimeConnectionString, firstId, output, errors) == 2);
            assert(errors.str().find("review_disposition=rejected") !=
                   std::string::npos);
            assert(output.str().empty());
        }
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " +
                        verify.quote_name(schema) + ";");
            assert(verify.exec("SELECT count(*) FROM experiment;")
                       .one_row()[0].as<int>() == 1);
        }
        const auto approval = RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Review(firstId,
                   RecommendationConversionProposalReviewDecision::approve,
                   "approve-first"));
        assert(approval.decision);

        const auto created = ExecuteApprovedRecommendationConversionProposal(
            runtime, firstId);
        assert(created.outcome == RecommendationConversionExecutionOutcome::created);
        assert(created.execution && created.execution->proposalId == firstId);
        assert(created.execution->reviewDecisionId ==
               approval.decision->reviewDecisionId);
        assert(created.execution->experimentId > 0);
        const auto replay = ExecuteApprovedRecommendationConversionProposal(
            runtime, firstId);
        assert(replay.outcome ==
               RecommendationConversionExecutionOutcome::existingIdentical);
        assert(replay.execution->experimentId == created.execution->experimentId);
        assert(FindRecommendationConversionExecution(
            runtime, created.execution->executionId));
        assert(FindRecommendationConversionExecutionByProposal(runtime, firstId));

        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " + verify.quote_name(schema) + ";");
            const pqxx::row row = verify.exec(
                "SELECT status,phase,worker_pid,current_operation,current_epoch,"
                "invocation_mode,marker,donchian20_mode,donchian_lookback,feature_warmup_scope "
                "FROM experiment WHERE experiment_id=$1;",
                pqxx::params{created.execution->experimentId}).one_row();
            assert(row["status"].as<std::string>() == "paused");
            assert(row["phase"].as<std::string>() == "train");
            assert(row["worker_pid"].is_null());
            assert(row["current_operation"].is_null());
            assert(row["current_epoch"].is_null());
            assert(row["invocation_mode"].as<std::string>() ==
                   "recommendation_conversion");
            assert(row["marker"].is_null());
            assert(row["donchian20_mode"].as<std::string>() ==
                   "zero_ablation");
            assert(row["donchian_lookback"].as<int>() == 20);
            assert(row["feature_warmup_scope"].as<std::string>() ==
                   "full_history_warmup");
        }

        // A later review reversal does not duplicate or erase a completed
        // conversion; retries return the original durable execution.
        (void)RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Review(firstId,
                   RecommendationConversionProposalReviewDecision::reject,
                   "reject-after-conversion"));
        const auto replayAfterRejection =
            ExecuteApprovedRecommendationConversionProposal(runtime, firstId);
        assert(replayAfterRejection.outcome ==
               RecommendationConversionExecutionOutcome::existingIdentical);

        const auto concurrentProposal = PersistRecommendationConversionProposal(
            runtime, Proposal(1.5, 2));
        const long long concurrentId = concurrentProposal.persisted.proposalId;
        (void)RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Review(concurrentId,
                   RecommendationConversionProposalReviewDecision::approve,
                   "approve-concurrent"));
        RecommendationConversionExecutionResult left;
        RecommendationConversionExecutionResult right;
        std::exception_ptr leftError;
        std::exception_ptr rightError;
        std::barrier start{3};
        auto run = [&](RecommendationConversionExecutionResult& result,
                       std::exception_ptr& error) {
            bool synchronized = false;
            try
            {
                pqxx::connection connection{runtimeConnectionString};
                start.arrive_and_wait();
                synchronized = true;
                result = ExecuteApprovedRecommendationConversionProposal(
                    connection, concurrentId);
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
        assert(left.execution->experimentId == right.execution->experimentId);
        assert((left.outcome == RecommendationConversionExecutionOutcome::created &&
                right.outcome ==
                    RecommendationConversionExecutionOutcome::existingIdentical) ||
               (right.outcome == RecommendationConversionExecutionOutcome::created &&
                left.outcome ==
                    RecommendationConversionExecutionOutcome::existingIdentical));

        // A distinct proposal for an already represented invocation fails
        // atomically rather than linking unrelated existing work.
        const auto conflictingProposal = PersistRecommendationConversionProposal(
            runtime, Proposal(1.25, 99, Donchian20Mode::ZeroAblation));
        (void)RecordRecommendationConversionProposalReviewDecision(
            runtime,
            Review(conflictingProposal.persisted.proposalId,
                   RecommendationConversionProposalReviewDecision::approve,
                   "approve-conflict"));
        bool identityConflict = false;
        try
        {
            (void)ExecuteApprovedRecommendationConversionProposal(
                runtime, conflictingProposal.persisted.proposalId);
        }
        catch (const std::runtime_error& error)
        {
            identityConflict = std::string{error.what()} ==
                "recommendation_conversion_execution_experiment_identity_conflict";
        }
        assert(identityConflict);
        assert(!FindRecommendationConversionExecutionByProposal(
            runtime, conflictingProposal.persisted.proposalId));
        {
            pqxx::read_transaction verify{owner};
            verify.exec("SET LOCAL search_path TO " +
                        verify.quote_name(schema) + ";");
            assert(verify.exec("SELECT count(*) FROM experiment;")
                       .one_row()[0].as<int>() == 3);
        }

        std::ostringstream output;
        std::ostringstream errors;
        assert(RunExecuteApprovedRecommendationConversionProposalCommand(
            runtimeConnectionString, concurrentId, output, errors) == 0);
        assert(output.str().find(
            "CONVERSION_PROPOSAL_EXPERIMENT_ALREADY_CREATED") !=
            std::string::npos);
        assert(output.str().find("experiment_created=false") !=
               std::string::npos);
        assert(output.str().find("review_decision=approve") !=
               std::string::npos);
        assert(output.str().find("experiment_queued=false") !=
               std::string::npos);
        assert(errors.str().empty());
        {
            std::ostringstream statusOutput;
            assert(RunRecommendationConversionExecutionStatusCommand(
                runtimeConnectionString, concurrentId, statusOutput) == 0);
            assert(statusOutput.str().find("CONVERSION_PROPOSAL_EXECUTION") !=
                   std::string::npos);
            assert(statusOutput.str().find(
                "experiment_id=" + std::to_string(left.execution->experimentId)) !=
                   std::string::npos);
        }

        // Runtime cannot rewrite or remove conversion audit history.
        for (const std::string& statement : {
                 "UPDATE experiment_recommendation_conversion_execution SET "
                 "execution_contract_version=1",
                 "DELETE FROM experiment_recommendation_conversion_execution",
                 "TRUNCATE experiment_recommendation_conversion_execution"})
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
        assert(unchanged.exec(
            "SELECT count(*) FROM "
            "experiment_recommendation_conversion_execution;")
                   .one_row()[0].as<int>() == 2);
        assert(unchanged.exec("SELECT count(*) FROM experiment;")
                   .one_row()[0].as<int>() == 3);
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
    std::cout << "ExperimentRecommendationConversionExecutionRepositoryTests "
                 "passed\n";
}
