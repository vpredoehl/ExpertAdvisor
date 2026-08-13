#include "../Sources/ExperimentRecommendationCampaignExecutionRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignExecutionService.hpp"
#include "../Sources/ExperimentRecommendationCampaignHandoffRepository.hpp"
#include "../Sources/ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <cassert>
#include <barrier>
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

void SetSearchPath(
    pqxx::transaction_base& transaction,
    const std::string& schema)
{
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
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
    const std::string& requestId,
    RecommendationConversionProposalReviewDecision decision =
        RecommendationConversionProposalReviewDecision::approve)
{
    RecommendationConversionProposalReviewRequest request;
    request.proposalId = proposalId;
    request.decision = decision;
    request.requestId = requestId;
    request.operatorIdentity = "phase5 test operator";
    request.reasonText = "Explicit isolated Phase 5 execution test.";
    return request;
}

std::size_t CountOccurrences(
    const std::string& text,
    const std::string& needle)
{
    std::size_t count = 0;
    std::size_t offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos)
    {
        ++count;
        offset += needle.size();
    }
    return count;
}

void InsertMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId,
    const std::vector<PersistedRecommendationConversionProposal>& proposals)
{
    const std::string canonical =
        "phase5-campaign-materialization-" + std::to_string(materializationId);
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization("
        "recommendation_campaign_materialization_id,"
        "recommendation_campaign_approval_id,materialization_contract_version,"
        "approval_identity_hash,recommendation_ranking_snapshot_id,"
        "ranking_snapshot_identity_hash,planning_policy_hash,"
        "campaign_plan_identity_hash,campaign_review_identity_hash,"
        "materialized_by,materialization_reason_text,selected_member_count,"
        "initially_created_proposal_count,initially_reused_proposal_count,"
        "materialization_identity_canonical,materialization_identity_hash) "
        "VALUES ($1::bigint,1000::bigint+$1::bigint,1,'approval-hash',"
        "2000::bigint+$1::bigint,'ranking-hash',"
        "'policy-hash','plan-hash','review-hash','materializer','reason',$2,$2,"
        "0,$3,$4);",
        pqxx::params{materializationId, static_cast<int>(proposals.size()),
                     canonical, RecommendationCanonicalHash(canonical)});
    for (std::size_t index = 0; index < proposals.size(); ++index)
    {
        const auto& proposal = proposals[index];
        const int ordinal = static_cast<int>(index) + 1;
        const std::string selected = canonical + ";member=" +
            std::to_string(ordinal);
        transaction.exec(
            "INSERT INTO experiment_recommendation_campaign_materialization_member("
            "recommendation_campaign_materialization_id,member_ordinal,"
            "recommendation_ranking_member_id,recommendation_id,"
            "source_experiment_id,ranking_position,"
            "selected_member_identity_canonical,selected_member_identity_hash,"
            "recommendation_conversion_proposal_id,proposal_identity_canonical,"
            "proposal_identity_hash) VALUES ($1,$2,3000+$3,$4,$5,$2,$6,$7,$3,$8,$9);",
            pqxx::params{
                materializationId, ordinal, proposal.proposalId,
                proposal.proposal.recommendationId,
                proposal.proposal.sourceExperimentId, selected,
                RecommendationCanonicalHash(selected),
                proposal.proposal.conversionIdentityCanonical,
                proposal.proposal.conversionIdentityHash});
    }
}

struct RunResult
{
    RecommendationCampaignExecutionPlan plan;
    std::vector<PersistedRecommendationConversionExecution> executions;
};

RunResult Run(
    const std::string& connectionString,
    long long materializationId)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    const auto materialization = LoadRecommendationCampaignExecutionMaterialization(
        transaction, materializationId);
    assert(materialization);
    LockRecommendationCampaignExecutions(transaction, *materialization);
    const auto input = LoadRecommendationCampaignExecutionInput(
        transaction, *materialization);
    RunResult result;
    result.plan = BuildRecommendationCampaignExecutionPlan(
        {materializationId, false}, input);
    if (result.plan.state == RecommendationCampaignExecutionPlanState::ready)
        result.executions = PersistRecommendationCampaignExecutions(
            transaction, result.plan);
    transaction.commit();
    return result;
}

long long Scalar(
    pqxx::connection& connection,
    const std::string& schema,
    const std::string& sql)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec(sql).one_row()[0].as<long long>();
}

long long SequenceValue(
    pqxx::connection& connection,
    const std::string& schema,
    const std::string& table,
    const std::string& column)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    const std::string sequence = transaction.exec(
        "SELECT pg_get_serial_sequence($1,$2);",
        pqxx::params{table, column}).one_row()[0].as<std::string>();
    return transaction.exec("SELECT last_value FROM " + sequence + ";")
        .one_row()[0].as<long long>();
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
    {
        std::ostringstream output;
        std::ostringstream errors;
        assert(RunRecommendationCampaignExecutionCommand(
            "host=invalid.invalid dbname=must_not_connect", {0, false},
            output, errors) == 1);
        assert(output.str().empty());
        assert(errors.str().find("campaign_execution_materialization_id_invalid") !=
               std::string::npos);
        assert(CountOccurrences(errors.str(), "experiments_created=") == 1);
    }

    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "phase5_campaign_execution_" + std::to_string(getpid());
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
            SetSearchPath(setup, schema);
            setup.exec(
                "CREATE TABLE experiment("
                "experiment_id bigserial PRIMARY KEY,symbol text NOT NULL,"
                "prediction_horizon integer NOT NULL,c_next_threshold double precision NOT NULL,"
                "core_lr_mult double precision,head_lr_mult double precision,"
                "target_epochs integer NOT NULL,checkpoint_interval integer NOT NULL,"
                "train_start timestamptz NOT NULL,train_end timestamptz NOT NULL,"
                "infer_start timestamptz,infer_end timestamptz,resume_model_id bigint,"
                "duplicate_nonce bigint NOT NULL DEFAULT 0,status text NOT NULL,"
                "phase text NOT NULL,invocation_mode text,"
                "donchian20_mode text NOT NULL DEFAULT 'enabled' CHECK "
                "(donchian20_mode IN ('enabled','zero_ablation')),"
                "updated_at timestamptz NOT NULL DEFAULT now(),"
                "worker_pid integer,current_operation text,current_epoch integer,marker text);"
                "CREATE UNIQUE INDEX experiment_unique_identity_uidx ON experiment("
                "symbol,prediction_horizon,c_next_threshold,"
                "coalesce(core_lr_mult,'-infinity'::double precision),"
                "coalesce(head_lr_mult,'-infinity'::double precision),target_epochs,"
                "checkpoint_interval,train_start,train_end,"
                "coalesce(infer_start,'-infinity'::timestamptz),"
                "coalesce(infer_end,'-infinity'::timestamptz),"
                "coalesce(resume_model_id,-1),donchian20_mode,duplicate_nonce) "
                "WHERE status<>'cancelled';"
                "CREATE TABLE model(model_id bigint PRIMARY KEY,marker text NOT NULL);"
                "CREATE TABLE experiment_recommendation("
                "recommendation_id bigint PRIMARY KEY,source_experiment_id bigint NOT NULL "
                "REFERENCES experiment(experiment_id),status text NOT NULL);"
                "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,"
                "c_next_threshold,core_lr_mult,head_lr_mult,target_epochs,"
                "checkpoint_interval,train_start,train_end,infer_start,infer_end,"
                "resume_model_id,status,phase,marker,updated_at) VALUES (17,'eurusd',12,"
                "0.001,1,5,120,15,'2010-01-01 America/Chicago',"
                "'2025-01-01 America/Chicago','2025-01-01 America/Chicago',"
                "'2026-01-01 America/Chicago',77,'paused','train','unchanged',"
                "'2026-07-19 12:34:56+00');"
                "INSERT INTO model VALUES (77,'unchanged');"
                "INSERT INTO experiment_recommendation VALUES (42,17,'approved');");
            for (const char* migration : {
                     "Database/migrations/036_experiment_recommendation_conversion_proposal.sql",
                     "Database/migrations/037_experiment_recommendation_conversion_review.sql",
                     "Database/migrations/038_experiment_recommendation_conversion_execution.sql",
                     "Database/migrations/039_experiment_recommendation_conversion_activation.sql"})
            {
                const std::string sql = ReadFile(migration);
                setup.exec(sql);
                setup.exec(sql);
            }
            setup.exec(R"SQL(
CREATE TABLE experiment_recommendation_campaign_materialization(
 recommendation_campaign_materialization_id bigint PRIMARY KEY,
 recommendation_campaign_approval_id bigint NOT NULL,
 materialization_contract_version integer NOT NULL,
 approval_identity_hash text NOT NULL,
 recommendation_ranking_snapshot_id bigint NOT NULL,
 ranking_snapshot_identity_hash text NOT NULL,
 planning_policy_hash text NOT NULL,
 campaign_plan_identity_hash text NOT NULL,
 campaign_review_identity_hash text NOT NULL,
 materialized_by text NOT NULL,
 materialization_reason_text text NOT NULL,
 selected_member_count integer NOT NULL,
 initially_created_proposal_count integer NOT NULL,
 initially_reused_proposal_count integer NOT NULL,
 materialization_identity_canonical text NOT NULL,
 materialization_identity_hash text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_campaign_materialization_member(
 recommendation_campaign_materialization_member_id bigserial PRIMARY KEY,
 recommendation_campaign_materialization_id bigint NOT NULL,
 member_ordinal integer NOT NULL,
 recommendation_ranking_member_id bigint NOT NULL,
 recommendation_id bigint NOT NULL,
 source_experiment_id bigint NOT NULL,
 ranking_position integer NOT NULL,
 selected_member_identity_canonical text NOT NULL,
 selected_member_identity_hash text NOT NULL,
 recommendation_conversion_proposal_id bigint NOT NULL,
 proposal_identity_canonical text NOT NULL,
 proposal_identity_hash text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
)SQL");
            setup.exec("GRANT SELECT,INSERT ON experiment TO pqxx;"
                       "GRANT USAGE ON SEQUENCE experiment_experiment_id_seq TO pqxx;"
                       "GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx; GRANT SELECT ON "
                       "experiment_recommendation_campaign_materialization,"
                       "experiment_recommendation_campaign_materialization_member "
                       "TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        const std::string sourceBefore = [&] {
            pqxx::read_transaction read{owner};
            SetSearchPath(read, schema);
            return read.exec("SELECT row_to_json(e)::text FROM experiment e "
                             "WHERE experiment_id=17;").one_row()[0].as<std::string>();
        }();

        std::vector<PersistedRecommendationConversionProposal> success;
        success.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(1.25, 1)).persisted);
        success.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(1.5, 2)).persisted);
        for (const auto& proposal : success)
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, Review(proposal.proposalId,
                                "phase5-success-" + std::to_string(proposal.proposalId)));
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 1, success);
            setup.commit();
        }

        const auto first = Run(runtimeConnectionString, 1);
        assert(first.plan.state == RecommendationCampaignExecutionPlanState::ready);
        assert(first.executions.size() == 2);
        assert(first.executions[0].proposalId == success[0].proposalId);
        assert(first.executions[1].proposalId == success[1].proposalId);
        assert(first.executions[0].experimentId != first.executions[1].experimentId);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_execution;") == 2);
        assert(Scalar(owner, schema, "SELECT count(*) FROM experiment;") == 3);
        const auto retry = Run(runtimeConnectionString, 1);
        assert(retry.plan.state ==
               RecommendationCampaignExecutionPlanState::alreadySatisfied);
        assert(retry.executions.empty());

        {
            pqxx::read_transaction read{owner};
            SetSearchPath(read, schema);
            const auto handoff = FindRecommendationCampaignHandoff(read, 1);
            assert(handoff && handoff->state ==
                RecommendationCampaignHandoffState::readyForPhase4cActivation);
            assert(handoff->summary.executionsPresent == 2);
        }

        const auto reversal =
            RecordRecommendationConversionProposalReviewDecision(
                runtime,
                Review(success[0].proposalId, "phase5-reversal",
                       RecommendationConversionProposalReviewDecision::reject));
        assert(reversal.decision);
        assert(reversal.decision->reviewDecisionId !=
               first.executions[0].reviewDecisionId);
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunRecommendationCampaignExecutionCommand(
                runtimeConnectionString, {1, false}, output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("result=already_satisfied") !=
                   std::string::npos);
            assert(output.str().find(
                "authorization_review_decision_id=" +
                std::to_string(first.executions[0].reviewDecisionId)) !=
                std::string::npos);
            assert(output.str().find(
                "authorization_review_decision_id=" +
                std::to_string(reversal.decision->reviewDecisionId)) ==
                std::string::npos);
        }

        std::vector<PersistedRecommendationConversionProposal> dry;
        dry.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(1.75, 3)).persisted);
        for (const auto& proposal : dry)
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, Review(proposal.proposalId, "phase5-dry"));
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 2, dry);
            setup.commit();
        }
        const long long experimentSequenceBefore = SequenceValue(
            owner, schema, "experiment", "experiment_id");
        const long long executionSequenceBefore = SequenceValue(
            owner, schema, "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id");
        {
            std::ostringstream output;
            std::ostringstream errors;
            assert(RunRecommendationCampaignExecutionCommand(
                runtimeConnectionString, {2, true}, output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("result=validated") != std::string::npos);
            assert(output.str().find("read_only=true") != std::string::npos);
            assert(output.str().find("experiments_created=0") != std::string::npos);
        }
        assert(SequenceValue(owner, schema, "experiment", "experiment_id") ==
               experimentSequenceBefore);
        assert(SequenceValue(
            owner, schema, "experiment_recommendation_conversion_execution",
            "recommendation_conversion_execution_id") == executionSequenceBefore);
        assert(!FindRecommendationConversionExecutionByProposal(
            runtime, dry[0].proposalId));

        std::vector<PersistedRecommendationConversionProposal> rollback;
        rollback.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(2.0, 4)).persisted);
        rollback.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(2.0, 5)).persisted);
        for (const auto& proposal : rollback)
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, Review(proposal.proposalId,
                                "phase5-rollback-" + std::to_string(proposal.proposalId)));
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 3, rollback);
            setup.commit();
        }
        const long long experimentsBeforeRollback = Scalar(
            owner, schema, "SELECT count(*) FROM experiment;");
        bool rolledBack = false;
        try
        {
            (void)Run(runtimeConnectionString, 3);
        }
        catch (const std::runtime_error& error)
        {
            rolledBack = std::string{error.what()} ==
                "campaign_execution_experiment_identity_conflict";
        }
        assert(rolledBack);
        assert(!FindRecommendationConversionExecutionByProposal(
            runtime, rollback[0].proposalId));
        assert(!FindRecommendationConversionExecutionByProposal(
            runtime, rollback[1].proposalId));
        assert(Scalar(owner, schema, "SELECT count(*) FROM experiment;") ==
               experimentsBeforeRollback);

        std::vector<PersistedRecommendationConversionProposal> concurrent;
        concurrent.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(2.25, 6)).persisted);
        concurrent.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(2.5, 7)).persisted);
        for (const auto& proposal : concurrent)
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, Review(proposal.proposalId,
                                "phase5-concurrent-" + std::to_string(proposal.proposalId)));
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 4, concurrent);
            setup.commit();
        }
        RunResult left;
        RunResult right;
        std::exception_ptr leftError;
        std::exception_ptr rightError;
        std::barrier start{3};
        auto invoke = [&](RunResult& result, std::exception_ptr& error) {
            try
            {
                start.arrive_and_wait();
                result = Run(runtimeConnectionString, 4);
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread leftThread{invoke, std::ref(left), std::ref(leftError)};
        std::thread rightThread{invoke, std::ref(right), std::ref(rightError)};
        start.arrive_and_wait();
        leftThread.join();
        rightThread.join();
        if (leftError) std::rethrow_exception(leftError);
        if (rightError) std::rethrow_exception(rightError);
        assert((left.executions.size() == 2 && right.executions.empty()) ||
               (right.executions.size() == 2 && left.executions.empty()));
        assert(left.plan.state != right.plan.state);

        std::vector<PersistedRecommendationConversionProposal> partial;
        partial.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(2.75, 8)).persisted);
        partial.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(3.0, 9)).persisted);
        for (const auto& proposal : partial)
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, Review(proposal.proposalId,
                                "phase5-partial-" + std::to_string(proposal.proposalId)));
        (void)ExecuteApprovedRecommendationConversionProposal(
            runtime, partial[0].proposalId);
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 5, partial);
            setup.commit();
        }
        bool partialConflict = false;
        try
        {
            (void)Run(runtimeConnectionString, 5);
        }
        catch (const std::invalid_argument& error)
        {
            partialConflict = std::string{error.what()} ==
                "campaign_execution_partial_operation_conflict";
        }
        assert(partialConflict);
        assert(!FindRecommendationConversionExecutionByProposal(
            runtime, partial[1].proposalId));

        std::vector<PersistedRecommendationConversionProposal> overlapping;
        overlapping.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(3.25, 10)).persisted);
        overlapping.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(3.5, 11)).persisted);
        overlapping.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(3.75, 12)).persisted);
        for (const auto& proposal : overlapping)
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, Review(proposal.proposalId,
                                "phase5-overlap-" +
                                    std::to_string(proposal.proposalId)));
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(
                setup, 6, {overlapping[0], overlapping[1]});
            InsertMaterialization(
                setup, 7, {overlapping[1], overlapping[2]});
            setup.commit();
        }
        RunResult overlapLeft;
        RunResult overlapRight;
        std::exception_ptr overlapLeftError;
        std::exception_ptr overlapRightError;
        std::barrier overlapStart{3};
        auto invokeOverlap = [&](long long materializationId, RunResult& result,
                                 std::exception_ptr& error) {
            try
            {
                overlapStart.arrive_and_wait();
                result = Run(runtimeConnectionString, materializationId);
            }
            catch (...)
            {
                error = std::current_exception();
            }
        };
        std::thread overlapLeftThread{
            invokeOverlap, 6, std::ref(overlapLeft),
            std::ref(overlapLeftError)};
        std::thread overlapRightThread{
            invokeOverlap, 7, std::ref(overlapRight),
            std::ref(overlapRightError)};
        overlapStart.arrive_and_wait();
        overlapLeftThread.join();
        overlapRightThread.join();
        assert(static_cast<bool>(overlapLeftError) !=
               static_cast<bool>(overlapRightError));
        const auto overlapError = overlapLeftError
            ? overlapLeftError : overlapRightError;
        bool overlapPartialConflict = false;
        try
        {
            std::rethrow_exception(overlapError);
        }
        catch (const std::invalid_argument& error)
        {
            overlapPartialConflict = std::string{error.what()} ==
                "campaign_execution_partial_operation_conflict";
        }
        assert(overlapPartialConflict);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_execution "
            "WHERE recommendation_conversion_proposal_id IN (" +
            std::to_string(overlapping[0].proposalId) + "," +
            std::to_string(overlapping[1].proposalId) + "," +
            std::to_string(overlapping[2].proposalId) + ");") == 2);

        std::vector<PersistedRecommendationConversionProposal> directRace;
        directRace.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(4.0, 13)).persisted);
        (void)RecordRecommendationConversionProposalReviewDecision(
            runtime, Review(directRace[0].proposalId, "phase5-direct-race"));
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 8, directRace);
            setup.commit();
        }
        RunResult campaignRace;
        RecommendationConversionExecutionResult directRaceResult;
        std::exception_ptr campaignRaceError;
        std::exception_ptr directRaceError;
        std::barrier directStart{3};
        std::thread campaignRaceThread{[&] {
            try
            {
                directStart.arrive_and_wait();
                campaignRace = Run(runtimeConnectionString, 8);
            }
            catch (...)
            {
                campaignRaceError = std::current_exception();
            }
        }};
        std::thread directRaceThread{[&] {
            try
            {
                pqxx::connection direct{runtimeConnectionString};
                directStart.arrive_and_wait();
                directRaceResult = ExecuteApprovedRecommendationConversionProposal(
                    direct, directRace[0].proposalId);
            }
            catch (...)
            {
                directRaceError = std::current_exception();
            }
        }};
        directStart.arrive_and_wait();
        campaignRaceThread.join();
        directRaceThread.join();
        if (campaignRaceError) std::rethrow_exception(campaignRaceError);
        if (directRaceError) std::rethrow_exception(directRaceError);
        assert(directRaceResult.execution);
        assert(campaignRace.plan.state ==
                   RecommendationCampaignExecutionPlanState::ready ||
               campaignRace.plan.state ==
                   RecommendationCampaignExecutionPlanState::alreadySatisfied);
        assert(Scalar(owner, schema,
            "SELECT count(*) FROM experiment_recommendation_conversion_execution "
            "WHERE recommendation_conversion_proposal_id=" +
            std::to_string(directRace[0].proposalId) + ";") == 1);

        std::vector<PersistedRecommendationConversionProposal> reviewRace;
        reviewRace.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(4.25, 14)).persisted);
        reviewRace.push_back(PersistRecommendationConversionProposal(
            runtime, Proposal(4.5, 15)).persisted);
        for (const auto& proposal : reviewRace)
            (void)RecordRecommendationConversionProposalReviewDecision(
                runtime, Review(proposal.proposalId,
                                "phase5-review-race-approve-" +
                                    std::to_string(proposal.proposalId)));
        {
            pqxx::work setup{owner};
            SetSearchPath(setup, schema);
            InsertMaterialization(setup, 9, reviewRace);
            setup.commit();
        }
        RunResult reviewRaceCampaign;
        RecommendationConversionProposalReviewPersistResult reviewRaceResult;
        std::exception_ptr reviewRaceCampaignError;
        std::exception_ptr reviewRaceReviewError;
        std::barrier reviewStart{3};
        std::thread reviewRaceCampaignThread{[&] {
            try
            {
                reviewStart.arrive_and_wait();
                reviewRaceCampaign = Run(runtimeConnectionString, 9);
            }
            catch (...)
            {
                reviewRaceCampaignError = std::current_exception();
            }
        }};
        std::thread reviewRaceReviewThread{[&] {
            try
            {
                pqxx::connection direct{runtimeConnectionString};
                reviewStart.arrive_and_wait();
                reviewRaceResult =
                    RecordRecommendationConversionProposalReviewDecision(
                        direct,
                        Review(reviewRace[1].proposalId,
                               "phase5-review-race-reject",
                               RecommendationConversionProposalReviewDecision::reject));
            }
            catch (...)
            {
                reviewRaceReviewError = std::current_exception();
            }
        }};
        reviewStart.arrive_and_wait();
        reviewRaceCampaignThread.join();
        reviewRaceReviewThread.join();
        if (reviewRaceReviewError)
            std::rethrow_exception(reviewRaceReviewError);
        assert(reviewRaceResult.decision);
        if (reviewRaceCampaignError)
        {
            bool notApproved = false;
            try
            {
                std::rethrow_exception(reviewRaceCampaignError);
            }
            catch (const std::invalid_argument& error)
            {
                notApproved = std::string{error.what()} ==
                    "campaign_execution_proposal_not_approved";
            }
            assert(notApproved);
            assert(!FindRecommendationConversionExecutionByProposal(
                runtime, reviewRace[0].proposalId));
            assert(!FindRecommendationConversionExecutionByProposal(
                runtime, reviewRace[1].proposalId));
        }
        else
        {
            assert(reviewRaceCampaign.executions.size() == 2);
            assert(FindRecommendationConversionExecutionByProposal(
                runtime, reviewRace[0].proposalId));
            assert(FindRecommendationConversionExecutionByProposal(
                runtime, reviewRace[1].proposalId));
        }

        {
            pqxx::read_transaction read{owner};
            SetSearchPath(read, schema);
            const std::string sourceAfter = read.exec(
                "SELECT row_to_json(e)::text FROM experiment e WHERE "
                "experiment_id=17;").one_row()[0].as<std::string>();
            assert(sourceAfter == sourceBefore);
            assert(read.exec(
                "SELECT count(*) FROM experiment WHERE status<>'paused' OR "
                "phase<>'train' OR worker_pid IS NOT NULL OR "
                "current_operation IS NOT NULL OR current_epoch IS NOT NULL;")
                .one_row()[0].as<long long>() == 0);
        }
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
    std::cout << "Experiment recommendation campaign execution repository tests passed\n";
    return 0;
}
