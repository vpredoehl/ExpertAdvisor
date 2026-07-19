#include "../Sources/ExperimentRecommendationCampaignHandoffRepository.hpp"
#include "../Sources/ExperimentRecommendationCampaignHandoffService.hpp"

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

void SetSearchPath(pqxx::transaction_base& transaction,
                   const std::string& schema)
{
    transaction.exec("SET LOCAL search_path TO " +
                     transaction.quote_name(schema) + ";");
}

void InsertProposal(pqxx::transaction_base& transaction, long long id)
{
    const std::string canonical = "handoff-proposal-" + std::to_string(id);
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_proposal VALUES "
        "($1::bigint,$1::bigint,100::bigint+$1::bigint,1,$2,$3,now());",
        pqxx::params{id, canonical, RecommendationCanonicalHash(canonical)});
}

void InsertReview(pqxx::transaction_base& transaction,
                  long long id,
                  long long proposalId,
                  const char* decision)
{
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_review_decision "
        "VALUES ($1,$2,$3,now());",
        pqxx::params{id, proposalId, decision});
}

void InsertExecution(pqxx::transaction_base& transaction,
                     long long id,
                     long long proposalId,
                     long long reviewId,
                     long long experimentId)
{
    const std::string canonical = "handoff-execution-" + std::to_string(id);
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_execution VALUES "
        "($1,$2,$3,$4,1,'approve',$5,$6,now());",
        pqxx::params{id, proposalId, reviewId, experimentId, canonical,
                     RecommendationCanonicalHash(canonical)});
}

void InsertActivation(pqxx::transaction_base& transaction,
                      long long id,
                      long long executionId,
                      long long proposalId,
                      long long reviewId,
                      long long experimentId)
{
    const std::string canonical = "handoff-activation-" + std::to_string(id);
    transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_activation VALUES "
        "($1,$2,$3,$4,$5,1,'paused','train','pending','train',$6,$7,now());",
        pqxx::params{id, executionId, proposalId, reviewId, experimentId,
                     canonical, RecommendationCanonicalHash(canonical)});
}

void AssertState(pqxx::transaction_base& transaction,
                 RecommendationCampaignHandoffState state)
{
    const auto handoff = FindRecommendationCampaignHandoff(transaction, 1);
    assert(handoff);
    assert(handoff->state == state);
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
        "phase4d_campaign_handoff_" + std::to_string(getpid());
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
CREATE SEQUENCE handoff_read_sentinel;
SELECT nextval('handoff_read_sentinel');
CREATE TABLE experiment_recommendation_conversion_proposal(
 recommendation_conversion_proposal_id bigint PRIMARY KEY,
 recommendation_id bigint NOT NULL, source_experiment_id bigint NOT NULL,
 conversion_contract_version integer NOT NULL,
 conversion_identity_canonical text NOT NULL,
 conversion_identity_hash text NOT NULL, created_at timestamptz NOT NULL);
CREATE TABLE experiment_recommendation_conversion_review_decision(
 recommendation_conversion_review_decision_id bigint PRIMARY KEY,
 recommendation_conversion_proposal_id bigint NOT NULL,
 decision text NOT NULL, decided_at timestamptz NOT NULL);
CREATE TABLE experiment_recommendation_conversion_execution(
 recommendation_conversion_execution_id bigint PRIMARY KEY,
 recommendation_conversion_proposal_id bigint NOT NULL,
 recommendation_conversion_review_decision_id bigint NOT NULL,
 experiment_id bigint NOT NULL, execution_contract_version integer NOT NULL,
 authorization_decision text NOT NULL,
 execution_identity_canonical text NOT NULL,
 execution_identity_hash text NOT NULL, created_at timestamptz NOT NULL);
CREATE TABLE experiment(
 experiment_id bigint PRIMARY KEY, status text NOT NULL, phase text NOT NULL,
 worker_pid integer, current_operation text, current_epoch integer,
 updated_at timestamptz NOT NULL);
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
 materialization_identity_hash text NOT NULL, created_at timestamptz NOT NULL);
CREATE TABLE experiment_recommendation_campaign_materialization_member(
 recommendation_campaign_materialization_member_id bigint PRIMARY KEY,
 recommendation_campaign_materialization_id bigint NOT NULL,
 member_ordinal integer NOT NULL, recommendation_ranking_member_id bigint NOT NULL,
 recommendation_id bigint NOT NULL, source_experiment_id bigint NOT NULL,
 ranking_position integer NOT NULL,
 selected_member_identity_canonical text NOT NULL,
 selected_member_identity_hash text NOT NULL,
 recommendation_conversion_proposal_id bigint NOT NULL,
 proposal_identity_canonical text NOT NULL,
 proposal_identity_hash text NOT NULL, created_at timestamptz NOT NULL);
)SQL");
            setup.exec("GRANT USAGE ON SCHEMA " + setup.quote_name(schema) +
                       " TO pqxx; GRANT SELECT ON ALL TABLES IN SCHEMA " +
                       setup.quote_name(schema) + " TO pqxx;");
            setup.commit();
        }

        pqxx::connection runtime{runtimeConnectionString};
        {
            pqxx::read_transaction beforeActivation{runtime};
            assert(!RecommendationCampaignHandoffSchemasExist(
                beforeActivation));
        }

        {
            pqxx::work fixtures{owner};
            SetSearchPath(fixtures, schema);
            fixtures.exec(R"SQL(
CREATE TABLE experiment_recommendation_conversion_activation(
 recommendation_conversion_activation_id bigint PRIMARY KEY,
 recommendation_conversion_execution_id bigint NOT NULL,
 recommendation_conversion_proposal_id bigint NOT NULL,
 recommendation_conversion_review_decision_id bigint NOT NULL,
 experiment_id bigint NOT NULL, activation_contract_version integer NOT NULL,
 previous_status text NOT NULL, previous_phase text NOT NULL,
 resulting_status text NOT NULL, resulting_phase text NOT NULL,
 activation_identity_canonical text NOT NULL,
 activation_identity_hash text NOT NULL, created_at timestamptz NOT NULL);
GRANT SELECT ON experiment_recommendation_conversion_activation TO pqxx;
)SQL");
            InsertProposal(fixtures, 1);
            InsertProposal(fixtures, 2);
            const std::string materialization = "handoff-materialization";
            fixtures.exec(
                "INSERT INTO experiment_recommendation_campaign_materialization "
                "VALUES (1,50,1,'approval-hash',60,'ranking-hash',"
                "'policy-hash','plan-hash','review-hash','operator','reason',"
                "2,2,0,$1,$2,now());",
                pqxx::params{materialization,
                    RecommendationCanonicalHash(materialization)});
            for (int ordinal = 1; ordinal <= 2; ++ordinal)
            {
                const std::string selected =
                    "handoff-member-" + std::to_string(ordinal);
                const std::string proposal =
                    "handoff-proposal-" + std::to_string(ordinal);
                fixtures.exec(
                    "INSERT INTO "
                    "experiment_recommendation_campaign_materialization_member "
                    "VALUES ($1::bigint,1,$1::integer,"
                    "100::bigint+$1::bigint,$1::bigint,"
                    "100::bigint+$1::bigint,$1::integer,$2,$3,"
                    "$1::bigint,$4,$5,now());",
                    pqxx::params{ordinal, selected,
                        RecommendationCanonicalHash(selected), proposal,
                        RecommendationCanonicalHash(proposal)});
            }
            InsertProposal(fixtures, 3);
            const std::string secondMaterialization =
                "handoff-materialization-2";
            fixtures.exec(
                "INSERT INTO experiment_recommendation_campaign_materialization "
                "VALUES (2,51,1,'approval-hash-2',61,'ranking-hash-2',"
                "'policy-hash-2','plan-hash-2','review-hash-2','operator',"
                "'reason',1,1,0,$1,$2,now());",
                pqxx::params{secondMaterialization,
                    RecommendationCanonicalHash(secondMaterialization)});
            const std::string selected = "handoff-member-3";
            const std::string proposal = "handoff-proposal-3";
            fixtures.exec(
                "INSERT INTO "
                "experiment_recommendation_campaign_materialization_member "
                "VALUES (3,2,1,103,3,103,3,$1,$2,3,$3,$4,now());",
                pqxx::params{selected, RecommendationCanonicalHash(selected),
                    proposal, RecommendationCanonicalHash(proposal)});
            fixtures.commit();
        }

        {
            pqxx::read_transaction read{runtime};
            assert(RecommendationCampaignHandoffSchemasExist(read));
            AssertState(read,
                RecommendationCampaignHandoffState::awaitingPhase4cReview);
            assert(!FindRecommendationCampaignHandoff(read, 999));
        }

        {
            pqxx::work missingProposal{owner};
            SetSearchPath(missingProposal, schema);
            missingProposal.exec(
                "DELETE FROM experiment_recommendation_conversion_proposal "
                "WHERE recommendation_conversion_proposal_id=2;");
            missingProposal.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::inconsistent);
            assert(handoff.summary.proposalsPresent == 1);
            assert(handoff.members[1].diagnosticCodes[0] ==
                   "linked_proposal_missing");
        }
        {
            pqxx::work restoreProposal{owner};
            SetSearchPath(restoreProposal, schema);
            InsertProposal(restoreProposal, 2);
            restoreProposal.commit();
        }

        {
            pqxx::work reviews{owner};
            SetSearchPath(reviews, schema);
            InsertReview(reviews, 10, 1, "reject");
            InsertReview(reviews, 20, 1, "approve");
            reviews.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            AssertState(read,
                RecommendationCampaignHandoffState::partiallyReviewed);
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.members[0].reviewDecisionId == 20);
            assert(handoff.members[0].reviewStatus ==
                   RecommendationCampaignHandoffReviewStatus::approved);
        }

        {
            pqxx::work review{owner};
            SetSearchPath(review, schema);
            InsertReview(review, 25, 2, "reject");
            review.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            AssertState(read,
                RecommendationCampaignHandoffState::reviewRejected);
        }
        {
            pqxx::work review{owner};
            SetSearchPath(review, schema);
            InsertReview(review, 30, 2, "approve");
            review.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            AssertState(read,
                RecommendationCampaignHandoffState::readyForPhase4cExecution);
        }

        {
            pqxx::work execution{owner};
            SetSearchPath(execution, schema);
            execution.exec(
                "INSERT INTO experiment VALUES "
                "(101,'paused','train',NULL,NULL,NULL,now()),"
                "(102,'paused','train',NULL,NULL,NULL,now());");
            InsertExecution(execution, 41, 1, 20, 101);
            execution.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            AssertState(read,
                RecommendationCampaignHandoffState::partiallyExecuted);
        }
        {
            pqxx::work duplicateExecution{owner};
            SetSearchPath(duplicateExecution, schema);
            duplicateExecution.exec(
                "INSERT INTO experiment VALUES "
                "(103,'paused','train',NULL,NULL,NULL,now());");
            InsertExecution(duplicateExecution, 43, 1, 20, 103);
            duplicateExecution.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::inconsistent);
            assert(handoff.members[0].diagnosticCodes[0] ==
                   "execution_cardinality_invalid");
        }
        {
            pqxx::work removeDuplicateExecution{owner};
            SetSearchPath(removeDuplicateExecution, schema);
            removeDuplicateExecution.exec(
                "DELETE FROM experiment_recommendation_conversion_execution "
                "WHERE recommendation_conversion_execution_id=43;"
                "DELETE FROM experiment WHERE experiment_id=103;");
            removeDuplicateExecution.commit();
        }
        {
            pqxx::work missingExperiment{owner};
            SetSearchPath(missingExperiment, schema);
            missingExperiment.exec(
                "DELETE FROM experiment WHERE experiment_id=101;");
            missingExperiment.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::inconsistent);
            assert(handoff.members[0].diagnosticCodes[0] ==
                   "experiment_missing");
        }
        {
            pqxx::work restoreExperiment{owner};
            SetSearchPath(restoreExperiment, schema);
            restoreExperiment.exec(
                "INSERT INTO experiment VALUES "
                "(101,'paused','train',NULL,NULL,NULL,now());");
            restoreExperiment.commit();
        }
        {
            pqxx::work orphanActivation{owner};
            SetSearchPath(orphanActivation, schema);
            InsertActivation(orphanActivation, 52, 999, 2, 30, 102);
            orphanActivation.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::inconsistent);
            assert(handoff.members[1].diagnosticCodes[0] ==
                   "activation_without_execution");
        }
        {
            pqxx::work removeOrphan{owner};
            SetSearchPath(removeOrphan, schema);
            removeOrphan.exec(
                "DELETE FROM experiment_recommendation_conversion_activation "
                "WHERE recommendation_conversion_activation_id=52;");
            removeOrphan.commit();
        }
        {
            pqxx::work invalidAuthorization{owner};
            SetSearchPath(invalidAuthorization, schema);
            InsertExecution(invalidAuthorization, 42, 2, 25, 102);
            invalidAuthorization.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::inconsistent);
            assert(handoff.members[1].diagnosticCodes[0] ==
                   "execution_review_invalid");
        }
        {
            pqxx::work replaceExecution{owner};
            SetSearchPath(replaceExecution, schema);
            replaceExecution.exec(
                "DELETE FROM experiment_recommendation_conversion_execution "
                "WHERE recommendation_conversion_execution_id=42;");
            InsertExecution(replaceExecution, 42, 2, 30, 102);
            replaceExecution.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            AssertState(read, RecommendationCampaignHandoffState::
                readyForPhase4cActivation);
        }

        {
            pqxx::work activation{owner};
            SetSearchPath(activation, schema);
            InsertActivation(activation, 51, 41, 1, 20, 101);
            activation.exec(
                "UPDATE experiment SET status='pending' WHERE experiment_id=101;");
            activation.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            AssertState(read,
                RecommendationCampaignHandoffState::partiallyActivated);
        }
        {
            pqxx::work activation{owner};
            SetSearchPath(activation, schema);
            InsertActivation(activation, 52, 42, 2, 30, 102);
            activation.exec(
                "UPDATE experiment SET status='pending' WHERE experiment_id=102;");
            activation.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            AssertState(read,
                RecommendationCampaignHandoffState::fullyActivated);
            const auto listed = ListRecommendationCampaignHandoffs(read, 1);
            assert(listed.size() == 1);
            assert(listed.front().members[0].memberOrdinal == 1);
            assert(listed.front().members[1].memberOrdinal == 2);
            const auto all = ListRecommendationCampaignHandoffs(read, 2);
            assert(all.size() == 2);
            assert(all[0].materializationId == 1);
            assert(all[1].materializationId == 2);
        }
        {
            pqxx::work duplicateActivation{owner};
            SetSearchPath(duplicateActivation, schema);
            InsertActivation(duplicateActivation, 53, 41, 1, 20, 101);
            duplicateActivation.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::inconsistent);
            assert(handoff.members[0].diagnosticCodes[0] ==
                   "activation_cardinality_invalid");
        }
        {
            pqxx::work removeDuplicateActivation{owner};
            SetSearchPath(removeDuplicateActivation, schema);
            removeDuplicateActivation.exec(
                "DELETE FROM experiment_recommendation_conversion_activation "
                "WHERE recommendation_conversion_activation_id=53;");
            removeDuplicateActivation.commit();
        }
        {
            pqxx::work laterRejection{owner};
            SetSearchPath(laterRejection, schema);
            InsertReview(laterRejection, 40, 1, "reject");
            laterRejection.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::reviewRejected);
            assert(handoff.members[0].reviewDecisionId == 40);
            assert(handoff.members[0].executionId == 41);
            assert(handoff.members[0].activationId == 51);
        }
        {
            pqxx::work laterApproval{owner};
            SetSearchPath(laterApproval, schema);
            InsertReview(laterApproval, 50, 1, "approve");
            laterApproval.commit();
        }

        std::string experimentUpdatedAtBeforeRead;
        {
            pqxx::read_transaction beforeRead{owner};
            SetSearchPath(beforeRead, schema);
            experimentUpdatedAtBeforeRead = beforeRead.exec(
                "SELECT string_agg(experiment_id::text || ':' || "
                "updated_at::text, ',' ORDER BY experiment_id) FROM experiment;")
                .one_row()[0].as<std::string>();
        }

        std::ostringstream first;
        std::ostringstream firstErrors;
        assert(RunShowRecommendationCampaignHandoffCommand(
            runtimeConnectionString, 1, first, firstErrors) == 0);
        assert(firstErrors.str().empty());
        assert(first.str().find("handoff_state=fully_activated") !=
               std::string::npos);
        assert(first.str().find("phase4c_workflow_state=activated_pending") !=
               std::string::npos);
        assert(first.str().find("read_only=true") != std::string::npos);
        assert(first.str().find(
            "campaign_materializations_created=false") != std::string::npos);
        assert(first.str().find("scheduler_started=false") !=
               std::string::npos);
        assert(first.str().find("workers_started=false") !=
               std::string::npos);
        std::ostringstream second;
        std::ostringstream secondErrors;
        assert(RunShowRecommendationCampaignHandoffCommand(
            runtimeConnectionString, 1, second, secondErrors) == 0);
        assert(first.str() == second.str());
        assert(secondErrors.str().empty());

        std::ostringstream listOutput;
        std::ostringstream listErrors;
        assert(RunListRecommendationCampaignHandoffsCommand(
            runtimeConnectionString, 1, listOutput, listErrors) == 0);
        assert(listErrors.str().empty());
        assert(listOutput.str().find(
            "RECOMMENDATION_CAMPAIGN_HANDOFF_LIST_COMPLETE,count=1") !=
            std::string::npos);
        std::ostringstream allListOutput;
        std::ostringstream allListErrors;
        assert(RunListRecommendationCampaignHandoffsCommand(
            runtimeConnectionString, 2, allListOutput, allListErrors) == 0);
        assert(allListErrors.str().empty());
        const auto firstSummary = allListOutput.str().find(
            "RECOMMENDATION_CAMPAIGN_HANDOFF,materialization_id=1");
        const auto secondSummary = allListOutput.str().find(
            "RECOMMENDATION_CAMPAIGN_HANDOFF,materialization_id=2");
        assert(firstSummary != std::string::npos);
        assert(secondSummary != std::string::npos);
        assert(firstSummary < secondSummary);
        assert(allListOutput.str().find(
            "RECOMMENDATION_CAMPAIGN_HANDOFF_LIST_COMPLETE,count=2") !=
            std::string::npos);
        std::ostringstream missingOutput;
        std::ostringstream missingErrors;
        assert(RunShowRecommendationCampaignHandoffCommand(
            runtimeConnectionString, 999, missingOutput, missingErrors) == 1);
        assert(missingErrors.str().find(
            "RECOMMENDATION_CAMPAIGN_HANDOFF_NOT_FOUND") !=
            std::string::npos);

        {
            pqxx::work malformed{owner};
            SetSearchPath(malformed, schema);
            malformed.exec(
                "UPDATE experiment_recommendation_campaign_materialization "
                "SET selected_member_count=3 WHERE "
                "recommendation_campaign_materialization_id=1;");
            malformed.commit();
        }
        std::ostringstream malformedOutput;
        std::ostringstream malformedErrors;
        assert(RunShowRecommendationCampaignHandoffCommand(
            runtimeConnectionString, 1, malformedOutput, malformedErrors) == 2);
        assert(malformedErrors.str().find(
            "incomplete_persisted_recommendation_campaign_materialization") !=
            std::string::npos);
        {
            pqxx::work restore{owner};
            SetSearchPath(restore, schema);
            restore.exec(
                "UPDATE experiment_recommendation_campaign_materialization "
                "SET selected_member_count=2 WHERE "
                "recommendation_campaign_materialization_id=1;");
            restore.commit();
        }

        {
            pqxx::work corrupt{owner};
            SetSearchPath(corrupt, schema);
            const std::string changed = "different-proposal";
            corrupt.exec(
                "UPDATE experiment_recommendation_conversion_proposal SET "
                "conversion_identity_canonical=$1,conversion_identity_hash=$2 "
                "WHERE recommendation_conversion_proposal_id=1;",
                pqxx::params{changed, RecommendationCanonicalHash(changed)});
            corrupt.commit();
        }
        {
            pqxx::read_transaction read{runtime};
            const auto handoff = *FindRecommendationCampaignHandoff(read, 1);
            assert(handoff.state ==
                   RecommendationCampaignHandoffState::inconsistent);
            assert(handoff.members[0].diagnosticCodes[0] ==
                   "proposal_identity_canonical_mismatch");
        }

        {
            pqxx::read_transaction verify{owner};
            SetSearchPath(verify, schema);
            assert(verify.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_campaign_materialization;")
                       .one_row()[0].as<int>() == 2);
            assert(verify.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_campaign_materialization_member;")
                       .one_row()[0].as<int>() == 3);
            assert(verify.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_proposal;")
                       .one_row()[0].as<int>() == 3);
            assert(verify.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_review_decision;")
                       .one_row()[0].as<int>() == 6);
            assert(verify.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_execution;")
                       .one_row()[0].as<int>() == 2);
            assert(verify.exec(
                "SELECT count(*) FROM "
                "experiment_recommendation_conversion_activation;")
                       .one_row()[0].as<int>() == 2);
            assert(verify.exec("SELECT last_value FROM handoff_read_sentinel;")
                       .one_row()[0].as<long long>() == 1);
            assert(verify.exec(
                "SELECT string_agg(experiment_id::text || ':' || "
                "updated_at::text, ',' ORDER BY experiment_id) FROM experiment;")
                       .one_row()[0].as<std::string>() ==
                   experimentUpdatedAtBeforeRead);
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
