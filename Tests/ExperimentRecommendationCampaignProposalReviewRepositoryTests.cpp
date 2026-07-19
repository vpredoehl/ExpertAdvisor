#include "../Sources/ExperimentRecommendationCampaignProposalReviewRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignProposalReviewService.hpp"

#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/ExperimentRecommendationCampaignHandoffRepository.hpp"

#include <cassert>
#include <cstdlib>
#include <future>
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

void SetSearchPath(pqxx::transaction_base& transaction,
                   const std::string& schema)
{
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
}

void InsertProposal(
    pqxx::transaction_base& transaction,
    long long proposalId,
    long long recommendationId,
    long long sourceExperimentId)
{
    const std::string canonical =
        "campaign-review-proposal-" + std::to_string(proposalId);
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_proposal("
        "recommendation_conversion_proposal_id,recommendation_id,"
        "source_experiment_id,conversion_contract_version,"
        "conversion_identity_canonical,conversion_identity_hash) "
        "VALUES ($1,$2,$3,1,$4,$5);",
        pqxx::params{proposalId, recommendationId, sourceExperimentId,
                     canonical, RecommendationCanonicalHash(canonical)});
}

void InsertMaterialization(
    pqxx::transaction_base& transaction,
    long long materializationId,
    long long firstProposalId,
    int memberCount,
    bool insertProposals = true)
{
    const std::string canonical =
        "campaign-review-materialization-" +
        std::to_string(materializationId);
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
        pqxx::params{materializationId, memberCount, canonical,
                     RecommendationCanonicalHash(canonical)});
    for (int ordinal = 1; ordinal <= memberCount; ++ordinal)
    {
        const long long proposalId = firstProposalId + ordinal - 1;
        const long long recommendationId = 10000 + proposalId;
        const long long sourceExperimentId = 20000 + proposalId;
        if (insertProposals)
            InsertProposal(
                transaction, proposalId, recommendationId, sourceExperimentId);
        const std::string selected =
            "campaign-review-member-" + std::to_string(materializationId) +
            "-" + std::to_string(ordinal);
        const std::string proposal =
            "campaign-review-proposal-" + std::to_string(proposalId);
        transaction.exec(
            "INSERT INTO "
            "experiment_recommendation_campaign_materialization_member("
            "recommendation_campaign_materialization_id,member_ordinal,"
            "recommendation_ranking_member_id,recommendation_id,"
            "source_experiment_id,ranking_position,"
            "selected_member_identity_canonical,selected_member_identity_hash,"
            "recommendation_conversion_proposal_id,proposal_identity_canonical,"
            "proposal_identity_hash) VALUES ($1,$2,30000+$3,$4,$5,$2,$6,$7,"
            "$3,$8,$9);",
            pqxx::params{materializationId, ordinal, proposalId,
                         recommendationId, sourceExperimentId, selected,
                         RecommendationCanonicalHash(selected), proposal,
                         RecommendationCanonicalHash(proposal)});
    }
}

RecommendationCampaignProposalReviewRequest Request(
    long long materializationId,
    RecommendationConversionProposalReviewDecision decision,
    bool dryRun = false)
{
    RecommendationCampaignProposalReviewRequest request;
    request.materializationId = materializationId;
    request.decision = decision;
    request.operatorIdentity = "operator@example";
    request.reasonText = "review exact persisted campaign";
    request.dryRun = dryRun;
    return request;
}

struct RunResult
{
    RecommendationCampaignProposalReviewPlan plan;
    std::vector<PersistedRecommendationConversionProposalReviewDecision> rows;
};

RunResult Run(
    const std::string& connectionString,
    const RecommendationCampaignProposalReviewRequest& request)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    assert(RecommendationCampaignProposalReviewSchemasExist(transaction));
    const auto materialization =
        LoadRecommendationCampaignProposalReviewMaterialization(
            transaction, request.materializationId);
    if (!materialization)
        throw std::runtime_error("materialization_not_found");
    LockRecommendationCampaignProposalReviews(transaction, *materialization);
    const auto input = LoadRecommendationCampaignProposalReviewInput(
        transaction, *materialization);
    RunResult result;
    result.plan = BuildRecommendationCampaignProposalReviewPlan(request, input);
    if (result.plan.state == RecommendationCampaignProposalReviewPlanState::ready)
        result.rows = PersistRecommendationCampaignProposalReviews(
            transaction, result.plan);
    transaction.commit();
    return result;
}

RunResult DryRun(
    const std::string& connectionString,
    const RecommendationCampaignProposalReviewRequest& request)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    assert(RecommendationCampaignProposalReviewSchemasExist(transaction));
    const auto materialization =
        LoadRecommendationCampaignProposalReviewMaterialization(
            transaction, request.materializationId);
    assert(materialization);
    const auto input = LoadRecommendationCampaignProposalReviewInput(
        transaction, *materialization);
    return {BuildRecommendationCampaignProposalReviewPlan(request, input), {}};
}

long long Count(
    pqxx::connection& connection,
    const std::string& schema,
    const char* table)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec("SELECT count(*) FROM " + std::string{table} + ";")
        .one_row()[0].as<long long>();
}

long long ReviewSequenceValue(
    pqxx::connection& connection,
    const std::string& schema)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec(R"SQL(
SELECT last_value
FROM pg_sequences
WHERE schemaname=current_schema()
  AND sequencename=regexp_replace(
      pg_get_serial_sequence(
        'experiment_recommendation_conversion_review_decision',
        'recommendation_conversion_review_decision_id'),
      '^.*\.', '');
)SQL").one_row()[0].as<long long>();
}

std::string ExperimentFingerprint(
    pqxx::connection& connection,
    const std::string& schema,
    long long experimentId)
{
    pqxx::read_transaction transaction{connection};
    SetSearchPath(transaction, schema);
    return transaction.exec(
        "SELECT row_to_json(e)::text FROM experiment e WHERE "
        "experiment_id=$1;",
        pqxx::params{experimentId}).one_row()[0].as<std::string>();
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
        auto invalid = Request(
            1, RecommendationConversionProposalReviewDecision::approve);
        invalid.decision =
            static_cast<RecommendationConversionProposalReviewDecision>(999);
        std::ostringstream output;
        std::ostringstream errors;
        assert(RunRecommendationCampaignProposalReviewCommand(
            "host=invalid.invalid dbname=must_not_connect", invalid,
            output, errors) == 1);
        assert(output.str().empty());
        assert(errors.str().find("campaign_decision=invalid") !=
               std::string::npos);
    }
    const std::string host = EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1");
    const std::string port = EnvironmentOr("LSTM_TEST_DB_PORT", "5432");
    const std::string ownerUser = EnvironmentOr(
        "LSTM_TEST_DB_ADMIN_USER", EnvironmentOr("USER", "vjp").c_str());
    const std::string schema =
        "phase4d_campaign_proposal_review_" + std::to_string(getpid());
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
            setup.exec(R"SQL(
CREATE TABLE experiment_recommendation_conversion_proposal(
 recommendation_conversion_proposal_id bigint PRIMARY KEY,
 recommendation_id bigint NOT NULL, source_experiment_id bigint NOT NULL,
 conversion_contract_version integer NOT NULL,
 conversion_identity_canonical text NOT NULL,
 conversion_identity_hash text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_conversion_review_decision(
 recommendation_conversion_review_decision_id bigserial PRIMARY KEY,
 recommendation_conversion_proposal_id bigint NOT NULL,
 decision text NOT NULL CHECK (decision IN ('approve','reject')),
 decision_request_id text NOT NULL,
 operator_identity text, reason_text text,
 decided_at timestamptz NOT NULL DEFAULT now(),
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(recommendation_conversion_proposal_id,decision_request_id));
CREATE TABLE experiment_recommendation_conversion_execution(
 recommendation_conversion_execution_id bigint PRIMARY KEY,
 recommendation_conversion_proposal_id bigint NOT NULL,
 recommendation_conversion_review_decision_id bigint NOT NULL,
 experiment_id bigint NOT NULL, execution_contract_version integer NOT NULL,
 authorization_decision text NOT NULL,
 execution_identity_canonical text NOT NULL,
 execution_identity_hash text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_conversion_activation(
 recommendation_conversion_activation_id bigint PRIMARY KEY,
 recommendation_conversion_execution_id bigint NOT NULL,
 recommendation_conversion_proposal_id bigint NOT NULL,
 recommendation_conversion_review_decision_id bigint NOT NULL,
 experiment_id bigint NOT NULL, activation_contract_version integer NOT NULL,
 previous_status text NOT NULL, previous_phase text NOT NULL,
 resulting_status text NOT NULL, resulting_phase text NOT NULL,
 activation_identity_canonical text NOT NULL,
 activation_identity_hash text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment(
 experiment_id bigint PRIMARY KEY, status text NOT NULL, phase text NOT NULL,
 worker_pid integer, current_operation text, current_epoch integer,
 updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE experiment_recommendation_campaign_materialization(
 recommendation_campaign_materialization_id bigint PRIMARY KEY,
 recommendation_campaign_approval_id bigint NOT NULL,
 materialization_contract_version integer NOT NULL,
 approval_identity_hash text NOT NULL,
 recommendation_ranking_snapshot_id bigint NOT NULL,
 ranking_snapshot_identity_hash text NOT NULL,
 planning_policy_hash text NOT NULL, campaign_plan_identity_hash text NOT NULL,
 campaign_review_identity_hash text NOT NULL, materialized_by text NOT NULL,
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
 recommendation_id bigint NOT NULL, source_experiment_id bigint NOT NULL,
 ranking_position integer NOT NULL,
 selected_member_identity_canonical text NOT NULL,
 selected_member_identity_hash text NOT NULL,
 recommendation_conversion_proposal_id bigint NOT NULL,
 proposal_identity_canonical text NOT NULL,
 proposal_identity_hash text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
)SQL");
            InsertMaterialization(setup, 1, 1, 2);
            InsertMaterialization(setup, 2, 3, 2);
            InsertMaterialization(setup, 3, 5, 2);
            InsertMaterialization(setup, 4, 7, 1, false);
            InsertMaterialization(setup, 5, 8, 1);
            setup.exec(
                "UPDATE experiment_recommendation_campaign_materialization "
                "SET selected_member_count=2 WHERE "
                "recommendation_campaign_materialization_id=5;");
            InsertMaterialization(setup, 6, 9, 2);
            InsertMaterialization(setup, 7, 11, 2);
            InsertMaterialization(setup, 8, 13, 1);
            InsertMaterialization(setup, 9, 14, 2);
            InsertMaterialization(setup, 10, 16, 1);
            setup.exec(
                "INSERT INTO experiment(experiment_id,status,phase,worker_pid,"
                "current_operation,current_epoch,updated_at) VALUES "
                "(900000,'paused','train',NULL,NULL,NULL,"
                "'2026-07-19 12:34:56+00'::timestamptz);");
            // A non-member proposal with the missing member's recommendation
            // and source IDs proves that lookup never infers membership from
            // those attributes.
            InsertProposal(setup, 70, 10007, 20007);
            setup.exec(
                "UPDATE experiment_recommendation_conversion_proposal SET "
                "conversion_identity_canonical='corrupt-proposal',"
                "conversion_identity_hash=$1 WHERE "
                "recommendation_conversion_proposal_id=13;",
                pqxx::params{RecommendationCanonicalHash("corrupt-proposal")});
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                       setup.quote_name(schema) +
                       " TO pqxx; GRANT INSERT ON "
                       "experiment_recommendation_conversion_review_decision "
                       "TO pqxx; GRANT USAGE ON ALL SEQUENCES IN SCHEMA " +
                       setup.quote_name(schema) + " TO pqxx;");
            setup.commit();
        }
        const std::string sentinelExperimentBefore =
            ExperimentFingerprint(owner, schema, 900000);

        auto approved = Run(
            runtimeConnectionString,
            Request(1, RecommendationConversionProposalReviewDecision::approve));
        assert(approved.rows.size() == 2);
        assert(approved.rows[0].proposalId == 1);
        assert(approved.rows[1].proposalId == 2);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 2);
        {
            pqxx::read_transaction read{owner};
            SetSearchPath(read, schema);
            const auto handoff = FindRecommendationCampaignHandoff(read, 1);
            assert(handoff);
            assert(handoff->state ==
                   RecommendationCampaignHandoffState::readyForPhase4cExecution);
        }

        const auto retried = Run(
            runtimeConnectionString,
            Request(1, RecommendationConversionProposalReviewDecision::approve));
        assert(retried.plan.state ==
               RecommendationCampaignProposalReviewPlanState::alreadySatisfied);
        assert(retried.rows.empty());
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 2);
        const auto retriedAgain = Run(
            runtimeConnectionString,
            Request(1, RecommendationConversionProposalReviewDecision::approve));
        assert(retriedAgain.plan.state ==
               RecommendationCampaignProposalReviewPlanState::alreadySatisfied);
        assert(retriedAgain.rows.empty());

        const auto rejected = Run(
            runtimeConnectionString,
            Request(2, RecommendationConversionProposalReviewDecision::reject));
        assert(rejected.rows.size() == 2);
        {
            pqxx::read_transaction read{owner};
            SetSearchPath(read, schema);
            const auto handoff = FindRecommendationCampaignHandoff(read, 2);
            assert(handoff && handoff->state ==
                RecommendationCampaignHandoffState::reviewRejected);
        }

        const long long beforeDryRun = Count(
            owner, schema,
            "experiment_recommendation_conversion_review_decision");
        const long long reviewSequenceBefore =
            ReviewSequenceValue(owner, schema);
        const auto dryRun = DryRun(
            runtimeConnectionString,
            Request(3, RecommendationConversionProposalReviewDecision::approve,
                    true));
        assert(dryRun.plan.state ==
               RecommendationCampaignProposalReviewPlanState::ready);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") ==
               beforeDryRun);
        assert(ReviewSequenceValue(owner, schema) == reviewSequenceBefore);
        {
            std::ostringstream output;
            std::ostringstream errors;
            const long long sequenceBeforeSatisfiedDryRun =
                ReviewSequenceValue(owner, schema);
            assert(RunRecommendationCampaignProposalReviewCommand(
                runtimeConnectionString,
                Request(1,
                    RecommendationConversionProposalReviewDecision::approve,
                    true),
                output, errors) == 0);
            assert(errors.str().empty());
            assert(output.str().find("result=already_satisfied") !=
                   std::string::npos);
            assert(ReviewSequenceValue(owner, schema) ==
                   sequenceBeforeSatisfiedDryRun);
        }

        {
            pqxx::connection runtime{runtimeConnectionString};
            RecommendationConversionProposalReviewRequest unrelated;
            unrelated.proposalId = 16;
            unrelated.decision =
                RecommendationConversionProposalReviewDecision::approve;
            unrelated.requestId = "unrelated-manual-review";
            unrelated.operatorIdentity = "operator@example";
            unrelated.reasonText = "review exact persisted campaign";
            const auto recorded =
                RecordRecommendationConversionProposalReviewDecision(
                    runtime, unrelated);
            assert(recorded.outcome ==
                   RecommendationConversionProposalReviewPersistOutcome::recorded);
        }
        try
        {
            (void)Run(runtimeConnectionString,
                Request(10,
                    RecommendationConversionProposalReviewDecision::approve));
            assert(false);
        }
        catch (const std::invalid_argument& error)
        {
            assert(std::string{error.what()} ==
                   "campaign_proposal_review_existing_decision_conflict");
        }

        try
        {
            (void)Run(runtimeConnectionString,
                Request(1,
                    RecommendationConversionProposalReviewDecision::reject));
            assert(false);
        }
        catch (const std::invalid_argument& error)
        {
            assert(std::string{error.what()} ==
                   "campaign_proposal_review_opposite_decision_conflict");
        }
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 5);

        try
        {
            (void)Run(runtimeConnectionString,
                Request(4,
                    RecommendationConversionProposalReviewDecision::approve));
            assert(false);
        }
        catch (const std::invalid_argument& error)
        {
            assert(std::string{error.what()} ==
                   "campaign_proposal_review_workflow_evidence_invalid");
        }
        try
        {
            (void)Run(runtimeConnectionString,
                Request(5,
                    RecommendationConversionProposalReviewDecision::approve));
            assert(false);
        }
        catch (const std::runtime_error& error)
        {
            assert(std::string{error.what()} ==
                   "incomplete_persisted_recommendation_campaign_materialization");
        }
        try
        {
            (void)Run(runtimeConnectionString,
                Request(8,
                    RecommendationConversionProposalReviewDecision::approve));
            assert(false);
        }
        catch (const std::invalid_argument& error)
        {
            assert(std::string{error.what()} ==
                   "campaign_proposal_review_workflow_evidence_invalid");
        }
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 5);

        {
            pqxx::work trigger{owner};
            SetSearchPath(trigger, schema);
            trigger.exec(R"SQL(
CREATE FUNCTION fail_second_campaign_review() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.recommendation_conversion_proposal_id=10 THEN
  RAISE EXCEPTION 'injected_campaign_review_failure';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER fail_second_campaign_review_trigger
BEFORE INSERT ON experiment_recommendation_conversion_review_decision
FOR EACH ROW EXECUTE FUNCTION fail_second_campaign_review();
)SQL");
            trigger.commit();
        }
        try
        {
            (void)Run(runtimeConnectionString,
                Request(6,
                    RecommendationConversionProposalReviewDecision::approve));
            assert(false);
        }
        catch (const pqxx::sql_error&) {}
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 5);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_execution") == 0);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_activation") == 0);
        assert(Count(owner, schema, "experiment") == 1);
        assert(ExperimentFingerprint(owner, schema, 900000) ==
               sentinelExperimentBefore);

        {
            pqxx::work dropTrigger{owner};
            SetSearchPath(dropTrigger, schema);
            dropTrigger.exec(
                "DROP TRIGGER fail_second_campaign_review_trigger ON "
                "experiment_recommendation_conversion_review_decision; "
                "DROP FUNCTION fail_second_campaign_review();");
            dropTrigger.commit();
        }

        auto left = std::async(std::launch::async, [&] {
            try
            {
                return Run(runtimeConnectionString,
                    Request(6,
                        RecommendationConversionProposalReviewDecision::approve))
                    .rows.size() == 2;
            }
            catch (...) { return false; }
        });
        auto right = std::async(std::launch::async, [&] {
            try
            {
                return Run(runtimeConnectionString,
                    Request(6,
                        RecommendationConversionProposalReviewDecision::reject))
                    .rows.size() == 2;
            }
            catch (...) { return false; }
        });
        const bool leftWon = left.get();
        const bool rightWon = right.get();
        assert(leftWon != rightWon);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 7);

        auto sameLeft = std::async(std::launch::async, [&] {
            return Run(runtimeConnectionString,
                Request(9,
                    RecommendationConversionProposalReviewDecision::approve));
        });
        auto sameRight = std::async(std::launch::async, [&] {
            return Run(runtimeConnectionString,
                Request(9,
                    RecommendationConversionProposalReviewDecision::approve));
        });
        const auto sameLeftResult = sameLeft.get();
        const auto sameRightResult = sameRight.get();
        assert(sameLeftResult.rows.size() + sameRightResult.rows.size() == 2);
        assert((sameLeftResult.plan.state ==
                    RecommendationCampaignProposalReviewPlanState::alreadySatisfied) !=
               (sameRightResult.plan.state ==
                    RecommendationCampaignProposalReviewPlanState::alreadySatisfied));
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 9);

        std::ostringstream serviceOutput;
        std::ostringstream serviceErrors;
        assert(RunRecommendationCampaignProposalReviewCommand(
            runtimeConnectionString,
            Request(7, RecommendationConversionProposalReviewDecision::approve),
            serviceOutput, serviceErrors) == 0);
        assert(serviceErrors.str().empty());
        assert(serviceOutput.str().find(
            "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW,materialization_id=7") !=
               std::string::npos);
        assert(serviceOutput.str().find("result=reviewed") !=
               std::string::npos);
        assert(serviceOutput.str().find(
            "phase4c_review_decisions_created=2") != std::string::npos);
        assert(serviceOutput.str().find("proposals_executed=false") !=
               std::string::npos);
        assert(serviceOutput.str().find("scheduler_started=false") !=
               std::string::npos);
        assert(serviceOutput.str().find("workers_started=false") !=
               std::string::npos);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 11);
        std::ostringstream conflictOutput;
        std::ostringstream conflictErrors;
        assert(RunRecommendationCampaignProposalReviewCommand(
            runtimeConnectionString,
            Request(7, RecommendationConversionProposalReviewDecision::reject),
            conflictOutput, conflictErrors) == 2);
        assert(conflictOutput.str().empty());
        assert(conflictErrors.str().find(
            "RECOMMENDATION_CAMPAIGN_PROPOSAL_REVIEW_CONFLICT") !=
               std::string::npos);
        assert(conflictErrors.str().find("conflict_count=1") !=
               std::string::npos);
        assert(conflictErrors.str().find("selected_member_count=2") !=
               std::string::npos);
        assert(conflictErrors.str().find("proposals_validated=2") !=
               std::string::npos);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") == 11);

        long long executedReviewId = -1;
        long long activatedReviewId = -1;
        {
            pqxx::work progressed{owner};
            SetSearchPath(progressed, schema);
            InsertMaterialization(progressed, 11, 80, 1);
            InsertMaterialization(progressed, 12, 81, 1);
            executedReviewId = progressed.exec(
                "INSERT INTO "
                "experiment_recommendation_conversion_review_decision("
                "recommendation_conversion_proposal_id,decision,"
                "decision_request_id,operator_identity,reason_text) VALUES "
                "(80,'approve','prior-executed-review','prior-operator',"
                "'prior reason') RETURNING "
                "recommendation_conversion_review_decision_id;")
                .one_row()[0].as<long long>();
            activatedReviewId = progressed.exec(
                "INSERT INTO "
                "experiment_recommendation_conversion_review_decision("
                "recommendation_conversion_proposal_id,decision,"
                "decision_request_id,operator_identity,reason_text) VALUES "
                "(81,'approve','prior-activated-review','prior-operator',"
                "'prior reason') RETURNING "
                "recommendation_conversion_review_decision_id;")
                .one_row()[0].as<long long>();
            progressed.exec(
                "INSERT INTO experiment(experiment_id,status,phase,worker_pid,"
                "current_operation,current_epoch,updated_at) VALUES "
                "(900001,'paused','train',NULL,NULL,NULL,"
                "'2026-07-19 12:35:56+00'::timestamptz),"
                "(900002,'pending','train',NULL,NULL,NULL,"
                "'2026-07-19 12:36:56+00'::timestamptz);");
            const std::string executedCanonical = "executed-workflow";
            const std::string activatedExecutionCanonical =
                "activated-execution-workflow";
            progressed.exec(
                "INSERT INTO experiment_recommendation_conversion_execution("
                "recommendation_conversion_execution_id,"
                "recommendation_conversion_proposal_id,"
                "recommendation_conversion_review_decision_id,experiment_id,"
                "execution_contract_version,authorization_decision,"
                "execution_identity_canonical,execution_identity_hash) VALUES "
                "(11001,80,$1,900001,1,'approve',$2,$3),"
                "(12001,81,$4,900002,1,'approve',$5,$6);",
                pqxx::params{
                    executedReviewId, executedCanonical,
                    RecommendationCanonicalHash(executedCanonical),
                    activatedReviewId, activatedExecutionCanonical,
                    RecommendationCanonicalHash(activatedExecutionCanonical)});
            const std::string activationCanonical = "activated-workflow";
            progressed.exec(
                "INSERT INTO experiment_recommendation_conversion_activation("
                "recommendation_conversion_activation_id,"
                "recommendation_conversion_execution_id,"
                "recommendation_conversion_proposal_id,"
                "recommendation_conversion_review_decision_id,experiment_id,"
                "activation_contract_version,previous_status,previous_phase,"
                "resulting_status,resulting_phase,activation_identity_canonical,"
                "activation_identity_hash) VALUES "
                "(12002,12001,81,$1,900002,1,'paused','train','pending',"
                "'train',$2,$3);",
                pqxx::params{activatedReviewId, activationCanonical,
                             RecommendationCanonicalHash(activationCanonical)});
            progressed.commit();
        }
        const std::string executedExperimentBefore =
            ExperimentFingerprint(owner, schema, 900001);
        const std::string activatedExperimentBefore =
            ExperimentFingerprint(owner, schema, 900002);
        const long long progressedReviewCount = Count(
            owner, schema,
            "experiment_recommendation_conversion_review_decision");
        for (const long long materializationId : {11LL, 12LL})
        {
            try
            {
                (void)Run(runtimeConnectionString,
                    Request(materializationId,
                        RecommendationConversionProposalReviewDecision::approve));
                assert(false);
            }
            catch (const std::invalid_argument& error)
            {
                assert(std::string{error.what()} ==
                       "campaign_proposal_review_existing_decision_conflict");
            }
        }
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_review_decision") ==
               progressedReviewCount);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_execution") == 2);
        assert(Count(owner, schema,
            "experiment_recommendation_conversion_activation") == 1);
        assert(ExperimentFingerprint(owner, schema, 900001) ==
               executedExperimentBefore);
        assert(ExperimentFingerprint(owner, schema, 900002) ==
               activatedExperimentBefore);
        assert(Count(owner, schema,
            "experiment_recommendation_campaign_materialization") == 12);
        assert(Count(owner, schema,
            "experiment_recommendation_campaign_materialization_member") == 18);
        assert(Count(owner, schema, "experiment") == 3);
        assert(ExperimentFingerprint(owner, schema, 900000) ==
               sentinelExperimentBefore);
        {
            pqxx::read_transaction read{owner};
            SetSearchPath(read, schema);
            assert(read.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_review_decision WHERE "
                "recommendation_conversion_proposal_id=70;")
                .one_row()[0].as<int>() == 0);
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
}
