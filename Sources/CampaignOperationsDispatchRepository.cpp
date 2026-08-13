#include "CampaignOperationsDispatchRepository.hpp"

#include "CampaignOperationsManager.hpp"
#include "CampaignOperationsProductionAdmission.hpp"
#include "ExperimentRecommendationConversionWorkflowRepository.hpp"
#include "CampaignOperationsProductionAdmissionRepository.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace EA::CampaignOperations
{
namespace
{

void LockAuthorizationDomains(pqxx::transaction_base& transaction,
    const std::string& campaignCanonical)
{
    std::array<std::string, 2> domains{
        campaignCanonical +
            ";action=adopt_existing_pending_and_control"
            ";scope=complete_materialization",
        campaignCanonical +
            ";action=dispatch_full_materialization"
            ";scope=complete_materialization"};
    std::sort(domains.begin(), domains.end());
    for (const auto& domain : domains)
        transaction.exec(
            "SELECT pg_advisory_xact_lock(hashtextextended($1,"
            "1179402835030003));", pqxx::params{domain});
}

pqxx::row LockRequestRow(pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    return transaction.exec(
        "SELECT operational_request_id,operational_campaign_id,"
        "campaign_identity_canonical,authorization_event_id,reservation_id,"
        "recommendation_campaign_materialization_id,"
        "materialization_member_count,request_identity_canonical,"
        "request_state,state_version,lease_token_hash,"
        "lease_expires_at::text,dispatcher_identity,"
        "production_dispatch_enabled "
        "FROM lock_campaign_operations_request($1);",
        pqxx::params{requestId.value()}).one_row();
}

void ValidateDispatchAuthorization(pqxx::transaction_base& transaction,
    long long campaignId, long long exactAuthorizationId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT authorization_event_id FROM "
        "lock_campaign_operations_authorization_head("
        "$1,'dispatch_full_materialization');",
        pqxx::params{campaignId});
    if (rows.empty() || rows.one_row()[0].is_null() ||
        rows.one_row()[0].as<long long>() != exactAuthorizationId)
        throw std::runtime_error(
            "campaign_operations_dispatch_authorization_not_head");
    const auto active = transaction.exec(
        "SELECT event_kind='granted' AND "
        "not_before<=transaction_timestamp() AND "
        "(expires_at IS NULL OR expires_at>transaction_timestamp()) "
        "FROM campaign_operations_authorization_event "
        "WHERE authorization_event_id=$1;",
        pqxx::params{exactAuthorizationId}).one_row()[0].as<bool>();
    if (!active)
        throw std::runtime_error(
            "campaign_operations_dispatch_authorization_inactive");
}

std::optional<long long> ValidateAdoptionAuthorization(
    pqxx::transaction_base& transaction, long long campaignId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT authorization_event_id,event_kind='granted' AND "
        "not_before<=transaction_timestamp() AND "
        "(expires_at IS NULL OR expires_at>transaction_timestamp()) "
        "FROM lock_campaign_operations_authorization_head("
        "$1,'adopt_existing_pending_and_control');",
        pqxx::params{campaignId});
    if (rows.empty() || rows.one_row()[0].is_null() ||
        rows.one_row()[1].is_null() || !rows.one_row()[1].as<bool>())
        return std::nullopt;
    return rows.one_row()[0].as<long long>();
}

void ValidateBudgetHead(pqxx::transaction_base& transaction,
    long long campaignId, long long reservationBudgetId,
    int reservationBudgetVersion)
{
    const pqxx::result rows = transaction.exec(
        "SELECT budget_ledger_entry_id,ledger_version,ledger_status "
        "FROM lock_campaign_operations_budget_head($1);",
        pqxx::params{campaignId});
    if (rows.empty() || rows.one_row()[0].is_null() ||
        rows.one_row()[0].as<long long>() !=
            reservationBudgetId ||
        rows.one_row()[1].as<int>() != reservationBudgetVersion ||
        rows.one_row()[2].as<std::string>() != "active")
        throw std::runtime_error(
            "campaign_operations_dispatch_budget_inactive");
}

void ValidateCampaignControlGate(
    pqxx::transaction_base& transaction, long long campaignId)
{
    const bool controlSchema = transaction.exec(
        "SELECT to_regclass('campaign_operations_control_event') "
        "IS NOT NULL AND to_regclass("
        "'campaign_operations_cancellation_request') IS NOT NULL;")
        .one_row()[0].as<bool>();
    if (!controlSchema) return;
    const auto allowed = transaction.exec(
        "SELECT campaign_operations_future_actions_allowed($1);",
        pqxx::params{campaignId}).one_row()[0].as<bool>();
    if (!allowed)
        throw std::runtime_error(
            "campaign_operations_dispatch_control_blocked");
}

DispatchAttemptRecord MapAttempt(const pqxx::row& row)
{
    const DispatchAttemptId attemptId(row[0].as<long long>());
    auto acquisition = BuildDispatchAttemptAcquisition(
        OperationalRequestId(row[1].as<long long>()),
        row[2].as<std::string>(), row[3].as<int>(), row[4].as<int>(),
        row[5].as<int>(), LeaseTokenDigest::Hydrate(row[6].as<std::string>()),
        UtcTimestamp(row[7].as<std::string>()),
        ActorIdentity(row[8].as<std::string>()));
    if (acquisition.identity.canonicalText() != row[9].as<std::string>() ||
        acquisition.identity.hash() != row[10].as<std::string>())
        throw std::runtime_error(
            "campaign_operations_dispatch_attempt_corrupt");
    return {attemptId, std::move(acquisition)};
}

void ValidateAttemptAudit(pqxx::transaction_base& transaction,
    const DispatchAttemptRecord& attempt)
{
    const pqxx::result rows = transaction.exec(
        "SELECT operational_request_id,dispatch_attempt_outcome_id,"
        "cause_kind,actor_identity,capability,prior_version,"
        "resulting_version,outcome,diagnostic_code "
        "FROM campaign_operations_dispatch_audit_reference_event "
        "WHERE dispatch_attempt_id=$1 "
        "AND cause_kind='dispatch_lease_acquired';",
        pqxx::params{attempt.attemptId.value()});
    if (rows.size() != 1)
        throw std::runtime_error(
            "campaign_operations_dispatch_acquisition_audit_corrupt");
    const auto& row = rows.one_row();
    if (row[0].as<long long>() !=
            attempt.acquisition.requestId.value() ||
        !row[1].is_null() ||
        row[2].as<std::string>() != "dispatch_lease_acquired" ||
        row[3].as<std::string>() !=
            attempt.acquisition.dispatcher.value() ||
        row[4].as<std::string>() !=
            kCampaignOperationsDispatcherRole ||
        row[5].as<int>() !=
            attempt.acquisition.expectedRequestVersion ||
        row[6].as<int>() !=
            attempt.acquisition.resultingRequestVersion ||
        row[7].as<std::string>() != "recorded" ||
        row[8].as<std::string>() != "dispatch_lease_acquired")
        throw std::runtime_error(
            "campaign_operations_dispatch_acquisition_audit_corrupt");
}

} // namespace

bool DispatchSchemaExists(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('campaign_operations_dispatch_attempt') "
        "IS NOT NULL AND "
        "to_regclass('campaign_operations_request_binding') IS NOT NULL "
        "AND to_regclass("
        "'campaign_operations_dispatch_attempt_outcome') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

std::vector<OperationalRequestId> SelectDispatchCandidatesForIsolatedTest(
    pqxx::transaction_base& transaction, int limit)
{
    if (limit <= 0 || limit > kCampaignOperationsMaximumDispatchBatchSize)
        throw std::invalid_argument("campaign_operations_dispatch_limit");
    std::vector<OperationalRequestId> result;
    const bool controlSchema = transaction.exec(
        "SELECT to_regclass('campaign_operations_control_event') "
        "IS NOT NULL AND to_regclass("
        "'campaign_operations_cancellation_request') IS NOT NULL;")
        .one_row()[0].as<bool>();
    const auto rows = controlSchema
        ? transaction.exec(
            "SELECT request.operational_request_id "
            "FROM campaign_operations_operational_request request "
            "WHERE request.request_state='ready' "
            "AND request.production_dispatch_enabled=false "
            "AND campaign_operations_future_actions_allowed("
            " request.operational_campaign_id) "
            "ORDER BY request.operational_request_id LIMIT $1;",
            pqxx::params{limit})
        : transaction.exec(
            "SELECT operational_request_id "
            "FROM campaign_operations_operational_request "
            "WHERE request_state='ready' "
            "AND production_dispatch_enabled=false "
            "ORDER BY operational_request_id LIMIT $1;",
            pqxx::params{limit});
    result.reserve(rows.size());
    for (const auto& row : rows)
        result.emplace_back(row[0].as<long long>());
    return result;
}

std::vector<DispatchCandidate> SelectDispatchCandidatesForManager(
    pqxx::connection& connection, int limit)
{
    if (limit <= 0 || limit > kCampaignOperationsManagerMaximumRunOnceLimit)
        throw std::invalid_argument("campaign_operations_manager_run_once_limit");

    pqxx::read_transaction transaction{connection};
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    // The Phase 5 capability already owns the exact read surface used by the
    // common Phase E engine, including the authoritative future-action gate.
    // This transaction remains READ ONLY; selecting the capability does not
    // grant the Manager any additional mutation path.
    transaction.exec("SET LOCAL ROLE " +
        transaction.quote_name(kProductionPhase5TransactionalRole) + ";");
    const auto rows = transaction.exec(
        "SELECT request.operational_request_id,"
        "request.request_identity_canonical,request.state_version "
        "FROM campaign_operations_operational_request request "
        "WHERE request.request_state='ready' "
        "AND request.production_dispatch_enabled=false "
        "AND campaign_operations_future_actions_allowed("
        " request.operational_campaign_id) "
        "ORDER BY request.operational_request_id LIMIT $1;",
        pqxx::params{limit});
    std::vector<DispatchCandidate> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
        result.push_back({OperationalRequestId(row[0].as<long long>()),
            row[1].as<std::string>(), row[2].as<int>()});
    transaction.commit();
    return result;
}

void RequireExactManagerOperationSourceEvidence(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId,
    OperationalRequestId requestId, const std::string& operationKey,
    const std::string& requestIdentityCanonical, int expectedRequestVersion,
    const std::string& expectedSourceCanonical)
{
    const auto expected = BuildManagerRequestOperationIdentity(
        requestIdentityCanonical, expectedRequestVersion);
    if (expected.source.canonicalText() != expectedSourceCanonical ||
        expected.operationKey != operationKey)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_manager_operation_conflicting_replay");
    const auto rows = transaction.exec(
        "SELECT operational_request_id,operation_key,request_identity_canonical,"
        "expected_request_version,source_canonical,source_hash,contract_version "
        "FROM campaign_operations_dispatch_manager_operation "
        "WHERE dispatch_attempt_id=$1;", pqxx::params{attemptId.value()});
    if (rows.empty())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_manager_operation_evidence_missing");
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_manager_operation_evidence_corrupt");
    const auto& row = rows.one_row();
    if (row[0].is_null() || row[1].is_null() || row[2].is_null() ||
        row[3].is_null() || row[4].is_null() || row[5].is_null() ||
        row[6].is_null() || row[0].as<long long>() != requestId.value() ||
        row[1].as<std::string>() != operationKey ||
        row[2].as<std::string>() != requestIdentityCanonical ||
        row[3].as<int>() != expectedRequestVersion ||
        row[4].as<std::string>() != expectedSourceCanonical ||
        row[5].as<std::string>() != expected.source.hash() ||
        row[6].as<int>() != kCampaignOperationsManagerOperationContractVersion)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_manager_operation_conflicting_replay");
    (void)CanonicalIdentity::Hydrate(1, row[4].as<std::string>(),
        row[5].as<std::string>());
}

void PersistManagerOperationSourceCanonical(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId,
    OperationalRequestId requestId, const std::string& operationKey,
    const std::string& requestIdentityCanonical, int expectedRequestVersion,
    const std::string& sourceCanonical, const std::string& sourceHash)
{
    const auto expected = BuildManagerRequestOperationIdentity(
        requestIdentityCanonical, expectedRequestVersion);
    if (expected.source.canonicalText() != sourceCanonical ||
        expected.source.hash() != sourceHash ||
        expected.operationKey != operationKey)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_manager_operation_conflicting_replay");
    const auto rows = transaction.exec(
        "INSERT INTO campaign_operations_dispatch_manager_operation("
        "dispatch_attempt_id,operational_request_id,operation_key,"
        "request_identity_canonical,expected_request_version,source_canonical,"
        "source_hash,contract_version) VALUES($1,$2,$3,$4,$5,$6,$7,$8) "
        "ON CONFLICT DO NOTHING "
        "RETURNING dispatch_attempt_id;",
        pqxx::params{attemptId.value(), requestId.value(), operationKey,
            requestIdentityCanonical, expectedRequestVersion, sourceCanonical,
            sourceHash, kCampaignOperationsManagerOperationContractVersion});
    (void)rows;
    RequireExactManagerOperationSourceEvidence(transaction, attemptId,
        requestId, operationKey, requestIdentityCanonical,
        expectedRequestVersion, sourceCanonical);
}

bool IsExactGrandfatheredH2ManagerOperation(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    const std::string& operationKey)
{
    // Before 058 there was no reserved Manager namespace.  Retaining that
    // grammar for a pre-058 executable/schema pair is necessary for the
    // upgrade regression; once the inventory relation exists, only its exact
    // immutable rows may pass this gate.
    if (!transaction.exec(
            "SELECT to_regclass("
            "'campaign_operations_h2_manager_key_compatibility') IS NOT NULL;")
             .one_row()[0].as<bool>())
        return true;

    const auto rows = transaction.exec(
        "SELECT compatibility.dispatch_attempt_id,"
        "compatibility.request_identity_canonical,"
        "compatibility.expected_request_version,"
        "compatibility.requesting_actor,"
        "compatibility.approved_build_contract_canonical,"
        "compatibility.attempt_identity_canonical,"
        "compatibility.attempt_identity_hash "
        "FROM campaign_operations_h2_manager_key_compatibility compatibility "
        "WHERE compatibility.operational_request_id=$1 "
        "AND compatibility.operation_key=$2;",
        pqxx::params{requestId.value(), operationKey});
    if (rows.empty()) return false;
    if (rows.size() != 1U)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_h2_manager_key_compatibility_corrupt");

    const auto& row = rows.one_row();
    const DispatchAttemptId attemptId(row[0].as<long long>());
    const auto persisted = FindProductionDispatchAttemptV2(transaction,
        attemptId);
    if (!persisted || persisted->attempt.requestId != requestId ||
        persisted->attempt.operationKey != operationKey ||
        persisted->attempt.requestIdentityCanonical != row[1].as<std::string>() ||
        persisted->attempt.expectedRequestVersion != row[2].as<int>() ||
        persisted->attempt.requestingActor.value() != row[3].as<std::string>() ||
        persisted->attempt.approvedBuildContract.identity.canonicalText() !=
            row[4].as<std::string>() ||
        persisted->attempt.identity.canonicalText() != row[5].as<std::string>() ||
        persisted->attempt.identity.hash() != row[6].as<std::string>() ||
        persisted->attempt.requestingActor.value() ==
            "campaign_operations_manager" ||
        transaction.exec(
            "SELECT EXISTS(SELECT 1 FROM "
            "campaign_operations_dispatch_manager_operation "
            "WHERE dispatch_attempt_id=$1);",
            pqxx::params{attemptId.value()}).one_row()[0].as<bool>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_h2_manager_key_compatibility_corrupt");
    return true;
}

DispatchLease AcquireDispatchLeaseInTransaction(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    const ActorIdentity& dispatcher, DispatchTestHook testHook)
{
    const pqxx::row advisory = transaction.exec(
        "SELECT operational_campaign_id,campaign_identity_canonical "
        "FROM campaign_operations_operational_request "
        "WHERE operational_request_id=$1;",
        pqxx::params{requestId.value()}).one_row();
    const auto campaignId = advisory[0].as<long long>();
    const auto campaignCanonical = advisory[1].as<std::string>();
    LockAuthorizationDomains(transaction, campaignCanonical);
    ValidateDispatchAuthorization(transaction, campaignId,
        transaction.exec(
            "SELECT authorization_event_id FROM "
            "campaign_operations_operational_request "
            "WHERE operational_request_id=$1;",
            pqxx::params{requestId.value()}).one_row()[0].as<long long>());
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        "1179402835030004));", pqxx::params{campaignCanonical});
    const pqxx::row reservationPreview = transaction.exec(
        "SELECT reservation_id,budget_ledger_entry_id,budget_ledger_version "
        "FROM campaign_operations_reservation WHERE reservation_id=("
        "SELECT reservation_id FROM campaign_operations_operational_request "
        "WHERE operational_request_id=$1);",
        pqxx::params{requestId.value()}).one_row();
    ValidateBudgetHead(transaction, campaignId,
        reservationPreview[1].as<long long>(),
        reservationPreview[2].as<int>());
    (void)LockOperationalCampaign(
        transaction, OperationalCampaignId(campaignId));
    ValidateCampaignControlGate(transaction, campaignId);
    const pqxx::row reservation = transaction.exec(
        "SELECT reservation_id,reservation_state,state_version,amount,"
        "recommendation_campaign_materialization_id,"
        "materialization_member_count,expires_at::text "
        "FROM lock_campaign_operations_reservation($1);",
        pqxx::params{reservationPreview[0].as<long long>()}).one_row();
    if (reservation[1].as<std::string>() != "held" ||
        (reservation[6].is_null() == false &&
         !transaction.exec(
             "SELECT $1::timestamptz > transaction_timestamp();",
             pqxx::params{reservation[6].as<std::string>()})
              .one_row()[0].as<bool>()))
        throw std::runtime_error(
            "campaign_operations_dispatch_reservation_unavailable");
    const pqxx::row request = LockRequestRow(transaction, requestId);
    if (request[8].as<std::string>() != "ready" ||
        request[9].as<int>() != expectedRequestVersion ||
        !request[10].is_null() || !request[11].is_null() ||
        !request[12].is_null() || request[13].as<bool>() ||
        transaction.exec(
            "SELECT EXISTS(SELECT 1 FROM "
            "campaign_operations_request_binding "
            "WHERE operational_request_id=$1);",
            pqxx::params{requestId.value()}).one_row()[0].as<bool>())
        throw std::runtime_error(
            "campaign_operations_dispatch_request_unavailable");

    const int ordinal = transaction.exec(
        "SELECT coalesce(max(attempt_ordinal),0)+1 "
        "FROM campaign_operations_dispatch_attempt "
        "WHERE operational_request_id=$1;",
        pqxx::params{requestId.value()}).one_row()[0].as<int>();
    if (testHook)
        testHook(
            DispatchTestInjectionPoint::beforeRequestLeaseTransition);
    const pqxx::row changed = transaction.exec(
        "SELECT operational_request_id,operational_campaign_id,"
        "reservation_id,recommendation_campaign_materialization_id,"
        "materialization_member_count,request_identity_canonical,"
        "state_version,to_char(lease_expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') "
        "FROM transition_campaign_operations_request_dispatching("
        "$1,$2,$3,$4);",
        pqxx::params{requestId.value(), expectedRequestVersion,
            leaseTokenDigest.value(), dispatcher.value()}).one_row();
    if (testHook)
        testHook(DispatchTestInjectionPoint::
            afterRequestLeaseTransitionBeforeAttempt);
    DispatchAttemptAcquisition acquisition =
        BuildDispatchAttemptAcquisition(requestId,
            changed[5].as<std::string>(), ordinal, expectedRequestVersion,
            changed[6].as<int>(), leaseTokenDigest,
            UtcTimestamp(changed[7].as<std::string>()), dispatcher);
    const long long attemptId = transaction.exec(
        "INSERT INTO campaign_operations_dispatch_attempt("
        "operational_request_id,request_identity_canonical,attempt_ordinal,"
        "expected_request_version,resulting_request_version,"
        "lease_token_digest,lease_expires_at,dispatcher_identity,"
        "attempt_contract_version,attempt_identity_canonical,"
        "attempt_identity_hash) VALUES($1,$2,$3,$4,$5,$6,$7::timestamptz,"
        "$8,1,$9,$10) RETURNING dispatch_attempt_id;",
        pqxx::params{requestId.value(), acquisition.requestCanonicalText,
            ordinal, acquisition.expectedRequestVersion,
            acquisition.resultingRequestVersion,
            acquisition.leaseTokenDigest.value(),
            acquisition.leaseExpiresAt.value(),
            acquisition.dispatcher.value(),
            acquisition.identity.canonicalText(),
            acquisition.identity.hash()}).one_row()[0].as<long long>();
    if (testHook)
        testHook(
            DispatchTestInjectionPoint::afterAttemptBeforeAcquisitionAudit);
    transaction.exec(
        "INSERT INTO campaign_operations_dispatch_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "dispatch_attempt_id,cause_kind,actor_identity,capability,"
        "prior_version,resulting_version,outcome,replay_disposition,"
        "diagnostic_code) VALUES($1,$2,$3,'dispatch_lease_acquired',$4,"
        "'campaign_operations_dispatcher',$5,$6,'recorded',"
        "'new_operation','dispatch_lease_acquired');",
        pqxx::params{campaignId, requestId.value(), attemptId,
            dispatcher.value(), expectedRequestVersion,
            acquisition.resultingRequestVersion});
    return {DispatchAttemptId(attemptId), std::move(acquisition),
        ReservationId(reservation[0].as<long long>()),
        reservation[2].as<int>(), reservation[4].as<long long>(),
        reservation[5].as<int>(), false, {}, {}};
}

DispatchLease AcquireProductionDispatchLeaseInTransaction(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    const std::string& operationKey, const ActorIdentity& requestingActor,
    const std::string& approvedBuildContractCanonical,
    const std::optional<std::string>& managerSourceCanonical)
{
    if (!IsValidProductionOperationKey(operationKey))
        throw std::invalid_argument(
            "campaign_operations_production_operation_key_invalid");
    bool managerAttemptExistedBeforeAcquisition = false;
    if (managerSourceCanonical)
    {
        // A Manager caller may create its source row only for the Attempt V2
        // it is acquiring now.  An existing row is recovery/replay evidence,
        // so it must already be complete; never repair an orphaned attempt by
        // backfilling source evidence.
        transaction.exec("SET LOCAL ROLE " +
            transaction.quote_name(kProductionPhase5TransactionalRole) + ";");
        managerAttemptExistedBeforeAcquisition = static_cast<bool>(
            FindProductionDispatchAttemptV2(transaction, requestId,
                operationKey));
        transaction.exec("SET LOCAL ROLE " +
            transaction.quote_name(kProductionDispatchServiceRole) + ";");
    }
    const auto leaseExpiresAt = transaction.exec(
        "SELECT (transaction_timestamp() + make_interval(secs => $1))::text;",
        pqxx::params{kCampaignOperationsDispatchLeaseSeconds})
        .one_row()[0].as<std::string>();
    const auto rows = transaction.exec(
        "SELECT * FROM campaign_operations_production_dispatch_authorized_v3("
        "$1,$2,$3,$4::timestamptz,$5,$6,$7);",
        pqxx::params{requestId.value(), expectedRequestVersion,
            leaseTokenDigest.value(), leaseExpiresAt,
            operationKey, requestingActor.value(),
            approvedBuildContractCanonical});
    if (rows.empty())
        throw std::runtime_error(
            "campaign_operations_production_dispatch_attempt_missing");

    // The fixed transition has completed under the production dispatcher.
    // Immutable attempt hydration and the common handoff use the accepted
    // Phase 5 transactional capability and its established Phase E reads.
    transaction.exec("SET LOCAL ROLE " +
        transaction.quote_name(kProductionPhase5TransactionalRole) + ";");
    const DispatchAttemptId attemptId(rows.one_row()[0].as<long long>());
    const auto persisted = FindProductionDispatchAttemptV2(
        transaction, attemptId);
    if (!persisted || persisted->attempt.requestId != requestId ||
        persisted->attempt.operationKey != operationKey ||
        persisted->attempt.requestingActor != requestingActor ||
        persisted->attempt.leaseTokenDigest != leaseTokenDigest ||
        persisted->attempt.approvedBuildContract.identity.canonicalText() !=
            approvedBuildContractCanonical)
        throw std::runtime_error(
            "campaign_operations_production_dispatch_attempt_corrupt");
    if (managerSourceCanonical)
    {
        const auto expectedManagerIdentity =
            BuildManagerRequestOperationIdentity(
                persisted->attempt.requestIdentityCanonical,
                expectedRequestVersion);
        if (expectedManagerIdentity.source.canonicalText() !=
                *managerSourceCanonical ||
            expectedManagerIdentity.operationKey != operationKey)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_manager_operation_conflicting_replay");
        if (managerAttemptExistedBeforeAcquisition)
        {
            RequireExactManagerOperationSourceEvidence(transaction, attemptId,
                requestId, operationKey,
                persisted->attempt.requestIdentityCanonical,
                expectedRequestVersion, *managerSourceCanonical);
        }
        else
            PersistManagerOperationSourceCanonical(transaction, attemptId,
                requestId, operationKey,
                persisted->attempt.requestIdentityCanonical,
                expectedRequestVersion, *managerSourceCanonical,
                expectedManagerIdentity.source.hash());
        RequireExactManagerOperationSourceEvidence(transaction, attemptId,
            requestId, operationKey,
            persisted->attempt.requestIdentityCanonical,
            expectedRequestVersion, *managerSourceCanonical);
    }
    const auto request = transaction.exec(
        "SELECT reservation_id,"
        "recommendation_campaign_materialization_id,"
        "materialization_member_count,state_version,request_state,"
        "lease_expires_at>transaction_timestamp() "
        "FROM campaign_operations_operational_request "
        "WHERE operational_request_id=$1;",
        pqxx::params{requestId.value()}).one_row();
    if (request[4].as<std::string>() != "dispatching" ||
        request[3].as<int>() != persisted->attempt.resultingRequestVersion ||
        !request[5].as<bool>())
        throw std::runtime_error(
            "campaign_operations_production_dispatch_lease_invalid");
    const auto reservation = transaction.exec(
        "SELECT state_version FROM campaign_operations_reservation "
        "WHERE reservation_id=$1 AND reservation_state='held';",
        pqxx::params{request[0].as<long long>()});
    if (reservation.empty())
        throw std::runtime_error(
            "campaign_operations_production_dispatch_reservation_invalid");
    return {attemptId,
        BuildDispatchAttemptAcquisition(
            persisted->attempt.requestId,
            persisted->attempt.requestIdentityCanonical,
            persisted->attempt.attemptOrdinal,
            persisted->attempt.expectedRequestVersion,
            persisted->attempt.resultingRequestVersion,
            persisted->attempt.leaseTokenDigest,
            persisted->attempt.leaseExpiresAt,
            persisted->attempt.requestingActor),
        ReservationId(request[0].as<long long>()),
        reservation.one_row()[0].as<int>(),
        request[1].as<long long>(), request[2].as<int>(), true,
        operationKey, approvedBuildContractCanonical};
}

DispatchLockedAuthority LockAndRevalidateDispatchAuthority(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    bool lockAdoptionAuthorization, bool productionDispatch,
    const std::string& productionOperationKey,
    const std::string& approvedBuildContractCanonical
#if defined(CAMPAIGN_OPERATIONS_H2_TESTING)
    , DispatchTestHook testHook
#endif
    )
{
    // Phase H's 0a/0b locks precede every Phase E domain lock.  The fixed H1
    // acquisition transition takes these locks inside its SECURITY DEFINER
    // body; handoff must reacquire them before entering the common engine.
    if (productionDispatch)
    {
        (void)LockSchedulerProtocolEvidence(transaction);
#if defined(CAMPAIGN_OPERATIONS_H2_TESTING)
        if (testHook)
            testHook(DispatchTestInjectionPoint::beforeProductionHandoffGate);
#endif
        transaction.exec(
            "SELECT pg_advisory_xact_lock_shared(19055,1);");
    }
    const pqxx::row advisory = transaction.exec(
        "SELECT operational_campaign_id,campaign_identity_canonical,"
        "authorization_event_id,reservation_id "
        "FROM campaign_operations_operational_request "
        "WHERE operational_request_id=$1;",
        pqxx::params{requestId.value()}).one_row();
    const long long campaignId = advisory[0].as<long long>();
    const auto campaignCanonical = advisory[1].as<std::string>();
    LockAuthorizationDomains(transaction, campaignCanonical);
    ValidateDispatchAuthorization(transaction, campaignId,
        advisory[2].as<long long>());
    const auto adoption = ValidateAdoptionAuthorization(
        transaction, campaignId);
    if (lockAdoptionAuthorization && !adoption)
        throw std::runtime_error(
            "campaign_operations_adoption_authorization_required");
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        "1179402835030004));", pqxx::params{campaignCanonical});
    const pqxx::row reservationPreview = transaction.exec(
        "SELECT budget_ledger_entry_id,budget_ledger_version "
        "FROM campaign_operations_reservation WHERE reservation_id=$1;",
        pqxx::params{advisory[3].as<long long>()}).one_row();
    ValidateBudgetHead(transaction, campaignId,
        reservationPreview[0].as<long long>(),
        reservationPreview[1].as<int>());
    (void)LockOperationalCampaign(
        transaction, OperationalCampaignId(campaignId));
    ValidateCampaignControlGate(transaction, campaignId);
    const pqxx::row reservation = transaction.exec(
        "SELECT reservation_id,reservation_identity_canonical,"
        "reservation_state,state_version,amount,"
        "recommendation_campaign_materialization_id,"
        "materialization_member_count,expires_at::text "
        "FROM lock_campaign_operations_reservation($1);",
        pqxx::params{advisory[3].as<long long>()}).one_row();
    if (reservation[2].as<std::string>() != "held" ||
        (!reservation[7].is_null() &&
         !transaction.exec(
             "SELECT $1::timestamptz > transaction_timestamp();",
             pqxx::params{reservation[7].as<std::string>()})
              .one_row()[0].as<bool>()))
        throw std::runtime_error(
            "campaign_operations_dispatch_reservation_unavailable");
    const pqxx::row request = LockRequestRow(transaction, requestId);
    if (request[8].as<std::string>() != "dispatching" ||
        request[9].as<int>() != expectedRequestVersion ||
        request[10].is_null() ||
        request[10].as<std::string>() != leaseTokenDigest.value() ||
        request[11].is_null() ||
        !transaction.exec(
            "SELECT $1::timestamptz > transaction_timestamp();",
            pqxx::params{request[11].as<std::string>()})
             .one_row()[0].as<bool>() ||
        (!productionDispatch && request[13].as<bool>()))
        throw std::runtime_error(
            "campaign_operations_dispatch_lease_invalid");
    if (productionDispatch)
    {
        if (productionOperationKey.empty() ||
            approvedBuildContractCanonical.empty())
            throw std::runtime_error(
                "campaign_operations_production_dispatch_identity_missing");
        const auto attempt = FindProductionDispatchAttemptV2(
            transaction, requestId, productionOperationKey);
        const auto current = FindCurrentProductionEnablementHead(transaction);
        if (!attempt || !current ||
            current->kind != ProductionEnablementEventKind::enable ||
            attempt->attempt.resultingRequestVersion !=
                expectedRequestVersion ||
            attempt->attempt.leaseTokenDigest != leaseTokenDigest ||
            attempt->attempt.approvedBuildContract.identity.canonicalText() !=
                approvedBuildContractCanonical ||
            attempt->authorizingEnablement.eventId != current->eventId ||
            !current->approvedBuildContract ||
            current->approvedBuildContract->identity.canonicalText() !=
                approvedBuildContractCanonical)
            throw std::runtime_error(
                "campaign_operations_production_dispatch_authority_invalid");
    }
    return {requestId, OperationalCampaignId(campaignId),
        ReservationId(reservation[0].as<long long>()),
        AuthorizationEventId(advisory[2].as<long long>()),
        adoption ? std::optional<AuthorizationEventId>(
                       AuthorizationEventId(*adoption)) : std::nullopt,
        request[9].as<int>(), reservation[3].as<int>(),
        reservation[5].as<long long>(), reservation[6].as<int>(),
        reservation[4].as<long long>(), request[7].as<std::string>(),
        reservation[1].as<std::string>(), leaseTokenDigest.value()};
}

std::optional<DispatchAttemptRecord> FindDispatchAttempt(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId)
{
    const auto rows = transaction.exec(
        "SELECT dispatch_attempt_id,operational_request_id,"
        "request_identity_canonical,attempt_ordinal,"
        "expected_request_version,resulting_request_version,"
        "lease_token_digest,to_char(lease_expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'),dispatcher_identity,"
        "attempt_identity_canonical,attempt_identity_hash "
        "FROM campaign_operations_dispatch_attempt "
        "WHERE dispatch_attempt_id=$1;",
        pqxx::params{attemptId.value()});
    if (rows.empty()) return std::nullopt;
    auto attempt = MapAttempt(rows.one_row());
    ValidateAttemptAudit(transaction, attempt);
    return attempt;
}

std::optional<DispatchAttemptRecord> FindLatestDispatchAttempt(
    pqxx::transaction_base& transaction, OperationalRequestId requestId)
{
    const auto rows = transaction.exec(
        "SELECT dispatch_attempt_id,operational_request_id,"
        "request_identity_canonical,attempt_ordinal,"
        "expected_request_version,resulting_request_version,"
        "lease_token_digest,to_char(lease_expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'),dispatcher_identity,"
        "attempt_identity_canonical,attempt_identity_hash "
        "FROM campaign_operations_dispatch_attempt "
        "WHERE operational_request_id=$1 "
        "ORDER BY attempt_ordinal DESC LIMIT 1;",
        pqxx::params{requestId.value()});
    if (rows.empty()) return std::nullopt;
    auto attempt = MapAttempt(rows.one_row());
    ValidateAttemptAudit(transaction, attempt);
    return attempt;
}

std::optional<DispatchLease> FindRecoverableDispatchLease(
    pqxx::transaction_base& transaction, OperationalRequestId requestId)
{
    const auto latest = FindLatestDispatchAttempt(transaction, requestId);
    if (!latest) return std::nullopt;
    const pqxx::result rows = transaction.exec(
        "SELECT request.reservation_id,reservation.state_version,"
        "request.recommendation_campaign_materialization_id,"
        "request.materialization_member_count "
        "FROM campaign_operations_operational_request request "
        "JOIN campaign_operations_reservation reservation "
        "ON reservation.reservation_id=request.reservation_id "
        "WHERE request.operational_request_id=$1 "
        "AND request.request_state='dispatching' "
        "AND request.state_version=$2 "
        "AND request.lease_token_hash=$3 "
        "AND request.lease_expires_at>transaction_timestamp() "
        "AND request.production_dispatch_enabled=false "
        "AND reservation.reservation_state='held' "
        "AND NOT EXISTS(SELECT 1 FROM "
        "campaign_operations_dispatch_attempt_outcome outcome "
        "WHERE outcome.dispatch_attempt_id=$4);",
        pqxx::params{requestId.value(),
            latest->acquisition.resultingRequestVersion,
            latest->acquisition.leaseTokenDigest.value(),
            latest->attemptId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    return DispatchLease{latest->attemptId, latest->acquisition,
        ReservationId(row[0].as<long long>()), row[1].as<int>(),
        row[2].as<long long>(), row[3].as<int>(), false, {}, {}};
}

std::optional<DispatchLease> FindRecoverableProductionDispatchLease(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    const std::string& operationKey,
    const std::optional<std::string>& managerSourceCanonical)
{
    const auto persisted = FindProductionDispatchAttemptV2(
        transaction, requestId, operationKey);
    if (!persisted) return std::nullopt;
    if (managerSourceCanonical)
        RequireExactManagerOperationSourceEvidence(transaction,
            persisted->attemptId, requestId, operationKey,
            persisted->attempt.requestIdentityCanonical,
            persisted->attempt.expectedRequestVersion,
            *managerSourceCanonical);
    const auto rows = transaction.exec(
        "SELECT request.reservation_id,reservation.state_version,"
        "request.recommendation_campaign_materialization_id,"
        "request.materialization_member_count "
        "FROM campaign_operations_operational_request request "
        "JOIN campaign_operations_reservation reservation "
        "ON reservation.reservation_id=request.reservation_id "
        "WHERE request.operational_request_id=$1 "
        "AND request.request_state='dispatching' "
        "AND request.state_version=$2 "
        "AND request.lease_token_hash=$3 "
        "AND request.lease_expires_at>transaction_timestamp() "
        "AND request.production_dispatch_enabled=true "
        "AND reservation.reservation_state='held' "
        "AND NOT EXISTS(SELECT 1 FROM "
        "campaign_operations_dispatch_attempt_outcome outcome "
        "WHERE outcome.dispatch_attempt_id=$4);",
        pqxx::params{requestId.value(),
            persisted->attempt.resultingRequestVersion,
            persisted->attempt.leaseTokenDigest.value(),
            persisted->attemptId.value()});
    if (rows.empty()) return std::nullopt;
    const auto& row = rows.one_row();
    return DispatchLease{
        persisted->attemptId,
        BuildDispatchAttemptAcquisition(
            persisted->attempt.requestId,
            persisted->attempt.requestIdentityCanonical,
            persisted->attempt.attemptOrdinal,
            persisted->attempt.expectedRequestVersion,
            persisted->attempt.resultingRequestVersion,
            persisted->attempt.leaseTokenDigest,
            persisted->attempt.leaseExpiresAt,
            persisted->attempt.requestingActor),
        ReservationId(row[0].as<long long>()), row[1].as<int>(),
        row[2].as<long long>(), row[3].as<int>(), true, operationKey,
        persisted->attempt.approvedBuildContract.identity.canonicalText()};
}

DownstreamEvidenceClassification ClassifyDownstreamEvidence(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority)
{
    const int bindingCount = transaction.exec(
        "SELECT count(*) FROM campaign_operations_request_binding "
        "WHERE operational_request_id=$1;",
        pqxx::params{authority.requestId.value()})
            .one_row()[0].as<int>();
    // A validated complete binding is returned by the service before this
    // classifier.  Any remaining binding evidence is incomplete or
    // contradictory and must never authorize downstream adoption.
    if (bindingCount != 0)
        return DownstreamEvidenceClassification::causallyAmbiguous;

    const pqxx::result memberRows = transaction.exec(
        "SELECT member_ordinal,recommendation_conversion_proposal_id,"
        "proposal_identity_canonical,proposal_identity_hash "
        "FROM experiment_recommendation_campaign_materialization_member "
        "WHERE recommendation_campaign_materialization_id=$1 "
        "ORDER BY member_ordinal;",
        pqxx::params{authority.materializationId});
    if (static_cast<int>(memberRows.size()) != authority.memberCount)
        return DownstreamEvidenceClassification::causallyAmbiguous;

    std::vector<long long> proposalIds;
    proposalIds.reserve(memberRows.size());
    for (const auto& member : memberRows)
        proposalIds.push_back(member[1].as<long long>());
    const auto workflows = ExperimentRecommendation::
        ListRecommendationConversionWorkflowsForProposals(
            transaction, proposalIds);
    if (workflows.size() !=
        static_cast<std::size_t>(memberRows.size()))
        return DownstreamEvidenceClassification::causallyAmbiguous;

    int withoutExecution = 0;
    int paused = 0;
    int pending = 0;
    int progressed = 0;
    for (const auto& member : memberRows)
    {
        const long long proposalId = member[1].as<long long>();
        const auto found = std::lower_bound(workflows.begin(), workflows.end(),
            proposalId, [](const auto& workflow, long long id)
            {
                return workflow.proposalId < id;
            });
        if (found == workflows.end() || found->proposalId != proposalId)
            return DownstreamEvidenceClassification::causallyAmbiguous;
        const auto& workflow = *found;
        if (workflow.proposalId != proposalId ||
            workflow.proposalIdentityCanonical !=
                member[2].as<std::string>() ||
            workflow.proposalIdentityHash != member[3].as<std::string>() ||
            workflow.derivation.integrity != ExperimentRecommendation::
                RecommendationConversionWorkflowIntegrity::consistent)
            return DownstreamEvidenceClassification::causallyAmbiguous;
        using State = ExperimentRecommendation::
            RecommendationConversionWorkflowState;
        switch (workflow.derivation.state)
        {
            case State::proposed:
            case State::pendingReview:
            case State::rejected:
            case State::approvedNotExecuted:
                ++withoutExecution;
                break;
            case State::executedPaused:
                ++paused;
                break;
            case State::activatedPending:
                ++pending;
                break;
            case State::schedulerClaimedOrRunning:
            case State::completed:
            case State::failed:
            case State::cancelled:
                ++progressed;
                break;
            case State::inconsistent:
                return DownstreamEvidenceClassification::causallyAmbiguous;
        }
    }
    const int total = static_cast<int>(workflows.size());
    if (withoutExecution == total)
        return DownstreamEvidenceClassification::noPhase5Evidence;
    if (pending == total)
        return DownstreamEvidenceClassification::exactCompletePendingTrain;
    if (paused == total)
        return DownstreamEvidenceClassification::pausedOnlyEvidence;
    if (progressed != 0)
        return DownstreamEvidenceClassification::progressedUnboundEvidence;
    return DownstreamEvidenceClassification::partialPhase5Evidence;
}

bool HasDownstreamControlOwnerCollision(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority)
{
    return transaction.exec(
        "SELECT EXISTS("
        "SELECT 1 "
        "FROM experiment_recommendation_campaign_materialization_member member "
        "JOIN experiment_recommendation_conversion_execution execution "
        "ON execution.recommendation_conversion_proposal_id="
        "member.recommendation_conversion_proposal_id "
        "JOIN campaign_operations_downstream_control_owner owner "
        "ON owner.experiment_id=execution.experiment_id "
        "WHERE member.recommendation_campaign_materialization_id=$1 "
        "AND owner.operational_request_id<>$2);",
        pqxx::params{authority.materializationId,
            authority.requestId.value()}).one_row()[0].as<bool>();
}

} // namespace EA::CampaignOperations
