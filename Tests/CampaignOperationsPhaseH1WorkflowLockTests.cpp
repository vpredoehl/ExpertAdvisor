#include "CampaignOperationsCompletionService.hpp"
#include "CampaignOperationsControlService.hpp"
#include "CampaignOperationsRepository.hpp"
#include "CampaignOperationsService.hpp"
#include "ExperimentRecommendation.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <pqxx/pqxx>

namespace CO = EA::CampaignOperations;

namespace
{

constexpr long long kPrimaryCampaignId = 71;

void SeedOneFixture(pqxx::transaction_base& transaction, long long id)
{
    const std::string suffix = std::to_string(id);
    const std::string materializationCanonical =
        "h1-lock-materialization-" + suffix;
    const std::string materializationHash =
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            materializationCanonical);
    const auto hash = [](const std::string& value)
    {
        return EA::ExperimentRecommendation::RecommendationCanonicalHash(value);
    };
    const std::string approvalCanonical = "h1-lock-approval-" + suffix;
    const std::string rankingCanonical = "h1-lock-ranking-" + suffix;
    const std::string planningCanonical = "h1-lock-planning-" + suffix;
    const std::string planCanonical = "h1-lock-plan-" + suffix;
    const std::string reviewCanonical = "h1-lock-review-" + suffix;
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization ("
        "recommendation_campaign_materialization_id,"
        "recommendation_campaign_approval_id,materialization_contract_version,"
        "approval_identity_canonical,approval_identity_hash,"
        "recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,"
        "ranking_snapshot_identity_hash,planning_policy_canonical,"
        "planning_policy_hash,planning_scope_canonical,"
        "campaign_plan_identity_canonical,campaign_plan_identity_hash,"
        "campaign_review_identity_canonical,campaign_review_identity_hash,"
        "approval_decision,approval_reviewer_identity,approval_reason_text,"
        "materialized_by,materialization_reason_text,selected_member_count,"
        "initially_created_proposal_count,initially_reused_proposal_count,"
        "materialization_identity_canonical,materialization_identity_hash) "
        "VALUES($1,$2,1,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,"
        "'approved','assurance@example.test','H1 lock fixture approval.',"
        "'assurance@example.test','H1 lock fixture materialization.',1,1,0,"
        "$15,$16);",
        pqxx::params{id, id + 1000, approvalCanonical, hash(approvalCanonical),
            id + 2000, rankingCanonical, hash(rankingCanonical),
            planningCanonical, hash(planningCanonical),
            "h1-lock-planning-scope-" + suffix, planCanonical,
            hash(planCanonical), reviewCanonical, hash(reviewCanonical),
            materializationCanonical, materializationHash});
    const std::string selectedCanonical = "h1-lock-selected-" + suffix;
    const std::string proposalCanonical = "h1-lock-proposal-" + suffix;
    transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_materialization_member ("
        "recommendation_campaign_materialization_member_id,"
        "recommendation_campaign_materialization_id,member_ordinal,"
        "recommendation_ranking_member_id,recommendation_id,"
        "source_experiment_id,ranking_position,"
        "selected_member_identity_canonical,selected_member_identity_hash,"
        "recommendation_conversion_proposal_id,proposal_identity_canonical,"
        "proposal_identity_hash) VALUES($1,$2,1,$3,$4,$5,1,$6,$7,$8,$9,$10);",
        pqxx::params{id * 100 + 1, id, id * 1000 + 1, id * 2000 + 1,
            id * 3000 + 1, selectedCanonical, hash(selectedCanonical),
            id * 4000 + 1, proposalCanonical, hash(proposalCanonical)});

    const CO::OperationalCampaign campaign = CO::BuildOperationalCampaign(
        id, 1, materializationCanonical, materializationHash, 1);
    const CO::OperationalCampaignId campaignId(id);
    const CO::OperationalAuthorizationEvent authorization =
        CO::BuildOperationalAuthorizationEvent(campaignId,
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, CO::AuthorizationEventKind::granted,
            CO::OperationalActionKind::dispatchFullMaterialization, 1,
            CO::ScopeKind::completeMaterialization, 1,
            CO::PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt, std::nullopt,
            CO::kCampaignOperationsAuthorizationRole,
            CO::ActorIdentity("assurance@example.test"),
            CO::Reason("H1 lock fixture authorization."),
            CO::UtcTimestamp("2026-08-01T00:00:00.000000Z"), std::nullopt);
    const CO::BudgetLedgerEntry budget = CO::BuildBudgetLedgerEntry(campaignId,
        campaign.identity.canonicalText(), std::nullopt, std::nullopt,
        std::nullopt, 1, CO::BudgetLedgerEntryKind::grant,
        CO::BudgetLedgerStatus::active,
        CO::BudgetUnit::materializedMemberDispatch, 1, 0, 1,
        CO::ActorIdentity("assurance@example.test"),
        CO::Reason("H1 lock fixture budget."));
    const CO::LogicalOperation operation =
        CO::BuildLogicalOperation(campaignId, campaign);
    const CO::Reservation reservation = CO::BuildReservation(operation,
        CO::AuthorizationEventId(id), authorization.identity.canonicalText(),
        authorization.identity.hash(), CO::BudgetLedgerEntryId(id), 1,
        budget.identity.canonicalText(), budget.identity.hash(), 1, 1,
        CO::BudgetUnit::materializedMemberDispatch, std::nullopt);
    const CO::OperationalRequest request = CO::BuildOperationalRequest(
        operation, CO::AuthorizationEventId(id),
        authorization.identity.canonicalText(), authorization.identity.hash(),
        CO::ReservationId(id), reservation.identity.canonicalText(),
        reservation.identity.hash(), 1, materializationHash,
        CO::ActorIdentity("assurance@example.test"),
        CO::Reason("H1 lock fixture request."),
        CO::PrerequisitePolicy::phase4dMaterializationOnlyV1,
        std::nullopt, std::nullopt);
    const CO::ReservationEvent acquisition =
        CO::BuildReservationAcquisitionEvent(CO::ReservationId(id),
            reservation.identity.canonicalText(), CO::OperationalRequestId(id),
            request.identity.canonicalText(), 1);

    transaction.exec(
        "INSERT INTO campaign_operations_campaign (operational_campaign_id,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,materialization_member_count,origin_kind,"
        "action_kind,action_contract_version,scope_kind,scope_contract_version,"
        "campaign_contract_version,campaign_identity_canonical,"
        "campaign_identity_hash) VALUES($1,$1,1,$2,$3,1,"
        "'phase4d_materialization_v1','dispatch_full_materialization',1,"
        "'complete_materialization',1,1,$4,$5);",
        pqxx::params{id, materializationCanonical, materializationHash,
            campaign.identity.canonicalText(), campaign.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_authorization_event ("
        "authorization_event_id,operational_campaign_id,"
        "campaign_identity_canonical,chain_version,event_kind,action_kind,"
        "action_contract_version,scope_kind,scope_contract_version,"
        "prerequisite_policy,authorization_role,actor_identity,reason,"
        "not_before,authorization_contract_version,"
        "authorization_identity_canonical,authorization_identity_hash) "
        "VALUES($1,$1,$2,1,'granted','dispatch_full_materialization',1,"
        "'complete_materialization',1,'phase4d_materialization_only_v1',"
        "$3,'assurance@example.test','H1 lock fixture authorization.',"
        "'2026-08-01T00:00:00.000000Z',1,$4,$5);",
        pqxx::params{id, campaign.identity.canonicalText(),
            CO::kCampaignOperationsAuthorizationRole,
            authorization.identity.canonicalText(), authorization.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_budget_ledger_entry ("
        "budget_ledger_entry_id,operational_campaign_id,"
        "campaign_identity_canonical,ledger_version,entry_kind,ledger_status,"
        "budget_unit,delta,prior_total,resulting_total,administrator_identity,"
        "reason,budget_contract_version,budget_identity_canonical,"
        "budget_identity_hash) VALUES($1,$1,$2,1,'grant','active',"
        "'materialized_member_dispatch',1,0,1,'assurance@example.test',"
        "'H1 lock fixture budget.',1,$3,$4);",
        pqxx::params{id, campaign.identity.canonicalText(),
            budget.identity.canonicalText(), budget.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_reservation (reservation_id,"
        "operational_campaign_id,campaign_identity_canonical,"
        "logical_operation_contract_version,logical_operation_canonical,"
        "logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "budget_ledger_entry_id,budget_ledger_version,budget_identity_canonical,"
        "budget_identity_hash,action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,scope_kind,scope_contract_version,"
        "materialization_member_count,amount,budget_unit,"
        "reservation_contract_version,reservation_identity_canonical,"
        "reservation_identity_hash,reservation_state,state_version) "
        "VALUES($1,$1,$2,1,$3,$4,$1,$5,$6,$1,1,$7,$8,"
        "'dispatch_full_materialization',1,$1,1,$9,$10,"
        "'complete_materialization',1,1,1,'materialized_member_dispatch',"
        "1,$11,$12,'held',1);",
        pqxx::params{id, campaign.identity.canonicalText(),
            operation.identity.canonicalText(), operation.identity.hash(),
            authorization.identity.canonicalText(), authorization.identity.hash(),
            budget.identity.canonicalText(), budget.identity.hash(),
            materializationCanonical, materializationHash,
            reservation.identity.canonicalText(), reservation.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_operational_request ("
        "operational_request_id,operational_campaign_id,"
        "campaign_identity_canonical,logical_operation_contract_version,"
        "logical_operation_canonical,logical_operation_hash,authorization_event_id,"
        "authorization_identity_canonical,authorization_identity_hash,"
        "reservation_id,reservation_identity_canonical,reservation_identity_hash,"
        "action_kind,action_contract_version,"
        "recommendation_campaign_materialization_id,"
        "materialization_contract_version,materialization_identity_canonical,"
        "materialization_identity_hash,ordered_scope_digest,"
        "materialization_member_count,accepting_actor_identity,reason,"
        "prerequisite_policy,request_contract_version,request_identity_canonical,"
        "request_identity_hash,request_state,state_version,"
        "production_dispatch_enabled) VALUES($1,$1,$2,1,$3,$4,$1,$5,$6,$1,"
        "$7,$8,'dispatch_full_materialization',1,$1,1,$9,$10,$10,1,"
        "'assurance@example.test','H1 lock fixture request.',"
        "'phase4d_materialization_only_v1',1,$11,$12,'ready',3,false);",
        pqxx::params{id, campaign.identity.canonicalText(),
            operation.identity.canonicalText(), operation.identity.hash(),
            authorization.identity.canonicalText(), authorization.identity.hash(),
            reservation.identity.canonicalText(), reservation.identity.hash(),
            materializationCanonical, materializationHash,
            request.identity.canonicalText(), request.identity.hash()});
    transaction.exec(
        "INSERT INTO campaign_operations_reservation_event ("
        "reservation_event_id,reservation_id,reservation_identity_canonical,"
        "transition_kind,expected_state,resulting_state,expected_version,"
        "resulting_version,operational_request_id,request_identity_canonical,"
        "amount,reservation_event_contract_version,"
        "reservation_event_identity_canonical,reservation_event_identity_hash) "
        "VALUES($1,$1,$2,'acquired',NULL,'held',0,1,$1,$3,1,1,$4,$5);",
        pqxx::params{id, reservation.identity.canonicalText(),
            request.identity.canonicalText(),
            acquisition.identity.canonicalText(), acquisition.identity.hash()});
}

void SeedFixtures(const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    transaction.exec("SET LOCAL session_replication_role=replica;");
    SeedOneFixture(transaction, kPrimaryCampaignId);
    SeedOneFixture(transaction, 72);
    transaction.commit();
}

void Pause(pqxx::transaction_base& transaction)
{
    transaction.exec("SELECT pg_sleep(8);");
}

void RunCompletion(const std::string& connectionString)
{
    const CO::CompleteIfSettledRequest request{
        kPrimaryCampaignId, "h1-lock-completion", "assurance@example.test",
        "H1 complete-workflow lock assurance."};
    const auto factory = [&connectionString]
    {
        return std::make_unique<pqxx::connection>(connectionString);
    };
    (void)CO::CompleteCampaignIfSettled(factory, request,
        [](CO::CompletionTestInjectionPoint point)
        {
            if (point == CO::CompletionTestInjectionPoint::
                    afterLocksBeforeEvidence)
                std::this_thread::sleep_for(std::chrono::seconds(4));
        });
}

void RunCancellation(const std::string& connectionString,
    const std::string& operationKey)
{
    const CO::CampaignCancellationCommandRequest request{
        kPrimaryCampaignId, kPrimaryCampaignId, 3, operationKey,
        "assurance@example.test",
        "H1 complete-workflow lock assurance."};
    (void)CO::CancelCampaign(connectionString, request,
        [](CO::CampaignOperationsControlTestInjectionPoint point,
           pqxx::transaction_base& transaction)
        {
            if (point == CO::CampaignOperationsControlTestInjectionPoint::
                    afterCancellationIntentInsertion)
                Pause(transaction);
        });
}

void RunReconciliation(const std::string& connectionString, bool recover)
{
    const CO::ReconciliationObserveRequest request{
        recover ? "h1-lock-recovery" : "h1-lock-reconciliation",
        0, 100, recover};
    (void)CO::ObserveAndRecoverCampaignOperations(connectionString, request,
        [recover](CO::CampaignOperationsControlTestInjectionPoint point,
                  pqxx::transaction_base& transaction)
        {
            if ((!recover && point == CO::CampaignOperationsControlTestInjectionPoint::
                    afterReconciliationCampaignLocksBeforeRequestLocks) ||
                (recover && point == CO::CampaignOperationsControlTestInjectionPoint::
                    beforeReconciliationRecoveryCommit))
                Pause(transaction);
        });
}

void RunBudgetMutation(const std::string& connectionString)
{
    pqxx::connection connection{connectionString};
    const CO::BudgetAdministrationRequest request{
        kPrimaryCampaignId, 1, CO::BudgetLedgerEntryKind::amend, 1,
        "assurance@example.test", "H1 complete-workflow lock assurance."};
    (void)CO::AdministerCampaignBudget(connection, request,
        [](CO::BudgetAdministrationTestInjectionPoint point)
        {
            if (point == CO::BudgetAdministrationTestInjectionPoint::
                    afterDomainLocksBeforePersistence)
                std::this_thread::sleep_for(std::chrono::seconds(4));
        });
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: CampaignOperationsPhaseH1WorkflowLockTests "
                     "CONNECTION MODE\n";
        return 64;
    }
    try
    {
        const std::string connectionString = argv[1];
        const std::string mode = argv[2];
        if (mode == "fixture-setup") SeedFixtures(connectionString);
        else if (mode == "completion") RunCompletion(connectionString);
        else if (mode == "cancellation")
            RunCancellation(connectionString, "h1-lock-cancellation");
        else if (mode == "reservation-release")
            RunCancellation(connectionString, "h1-lock-reservation-release");
        else if (mode == "reconciliation")
            RunReconciliation(connectionString, false);
        else if (mode == "recovery") RunReconciliation(connectionString, true);
        else if (mode == "budget-mutation") RunBudgetMutation(connectionString);
        else throw std::invalid_argument("unknown workflow mode");
        std::cout << "workflow=" << mode << ",outcome=completed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "workflow_error=" << error.what() << '\n';
        return 1;
    }
}
