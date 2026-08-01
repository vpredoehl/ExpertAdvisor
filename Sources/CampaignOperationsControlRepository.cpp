#include "CampaignOperationsControlRepository.hpp"

#include "CampaignOperationsDispatch.hpp"
#include "CampaignOperationsRepository.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

std::string Framed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

CanonicalIdentity Identity(const std::string& canonical)
{
    return CanonicalIdentity::Create(
        kCampaignOperationsControlContractVersion, canonical);
}

PersistedCampaignControlEvent MapControl(const pqxx::row& row)
{
    const auto previousId = row[3].is_null()
        ? std::nullopt
        : std::optional<ControlEventId>(
              ControlEventId(row[3].as<long long>()));
    const auto previousCanonical = row[4].is_null()
        ? std::nullopt
        : std::optional<std::string>(row[4].as<std::string>());
    auto event = BuildCampaignControlEvent(
        OperationalCampaignId(row[1].as<long long>()),
        row[2].as<std::string>(), previousId, previousCanonical,
        row[5].as<int>(),
        ControlEventKindFromText(row[6].as<std::string>()),
        ActorIdentity(row[7].as<std::string>()),
        Reason(row[8].as<std::string>()));
    if (event.identity.canonicalText() != row[9].as<std::string>() ||
        event.identity.hash() != row[10].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_control_event_corrupt");
    return {ControlEventId(row[0].as<long long>()), std::move(event),
        row[11].as<std::string>()};
}

PersistedCampaignCancellationRequest MapCancellationRequest(
    const pqxx::row& row)
{
    const auto requestId = row[3].is_null()
        ? std::nullopt
        : std::optional<OperationalRequestId>(
              OperationalRequestId(row[3].as<long long>()));
    const auto requestCanonical = row[4].is_null()
        ? std::nullopt
        : std::optional<std::string>(row[4].as<std::string>());
    const auto expectedState = row[5].is_null()
        ? std::nullopt
        : std::optional<RequestState>(
              RequestStateFromText(row[5].as<std::string>()));
    const auto expectedVersion = row[6].is_null()
        ? std::nullopt
        : std::optional<int>(row[6].as<int>());
    auto request = BuildCampaignCancellationRequest(
        OperationalCampaignId(row[1].as<long long>()),
        row[2].as<std::string>(), requestId, requestCanonical,
        expectedState, expectedVersion, row[7].as<std::string>(),
        ActorIdentity(row[8].as<std::string>()),
        Reason(row[9].as<std::string>()));
    if (request.identity.canonicalText() != row[10].as<std::string>() ||
        request.identity.hash() != row[11].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_cancellation_request_corrupt");
    return {CancellationRequestId(row[0].as<long long>()),
        std::move(request), row[12].as<std::string>()};
}

PersistedCampaignCancellationSettlement MapSettlement(
    const pqxx::row& row)
{
    const auto reservationEventId = row[3].is_null()
        ? std::nullopt
        : std::optional<ReservationEventId>(
              ReservationEventId(row[3].as<long long>()));
    const auto reservationCanonical = row[4].is_null()
        ? std::nullopt
        : std::optional<std::string>(row[4].as<std::string>());
    const auto requestVersion = row[5].is_null()
        ? std::nullopt
        : std::optional<int>(row[5].as<int>());
    const auto lifecycleCanonical = row[6].is_null()
        ? std::nullopt
        : std::optional<std::string>(row[6].as<std::string>());
    const auto lifecycleHash = row[7].is_null()
        ? std::nullopt
        : std::optional<std::string>(row[7].as<std::string>());
    auto settlement = BuildCampaignCancellationSettlement(
        CancellationRequestId(row[1].as<long long>()),
        row[2].as<std::string>(),
        CancellationSettlementDispositionFromText(
            row[8].as<std::string>()),
        reservationEventId, reservationCanonical, requestVersion,
        lifecycleCanonical, lifecycleHash);
    if (settlement.identity.canonicalText() != row[9].as<std::string>() ||
        settlement.identity.hash() != row[10].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_cancellation_settlement_corrupt");
    return {CancellationSettlementId(row[0].as<long long>()),
        std::move(settlement), row[11].as<std::string>()};
}

PersistedReconciliationObservation MapObservation(const pqxx::row& row)
{
    auto observation = BuildReconciliationObservation(
        row[1].as<std::string>(),
        OperationalCampaignId(row[2].as<long long>()),
        OperationalRequestId(row[3].as<long long>()),
        row[4].as<std::string>(),
        RequestStateFromText(row[5].as<std::string>()),
        row[6].as<int>(),
        ReconciliationReasonFromText(row[7].as<std::string>()),
        row[8].as<std::string>(), row[10].as<std::string>(),
        row[11].as<std::string>(), row[12].as<std::string>());
    if (observation.evidenceIdentityHash != row[9].as<std::string>() ||
        observation.identity.canonicalText() !=
            row[13].as<std::string>() ||
        observation.identity.hash() != row[14].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_reconciliation_observation_corrupt");
    return {ReconciliationObservationId(row[0].as<long long>()),
        std::move(observation), row[15].as<std::string>()};
}

PersistedReconciliationResolution MapResolution(const pqxx::row& row)
{
    auto resolution = BuildReconciliationResolution(
        ReconciliationObservationId(row[1].as<long long>()),
        row[2].as<std::string>(), row[3].as<std::string>(),
        row[4].as<std::string>(), row[5].as<std::string>(),
        row[6].as<std::string>(), row[7].as<std::string>());
    if (resolution.identity.canonicalText() != row[8].as<std::string>() ||
        resolution.identity.hash() != row[9].as<std::string>())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_reconciliation_resolution_corrupt");
    return {ReconciliationResolutionId(row[0].as<long long>()),
        std::move(resolution), row[10].as<std::string>()};
}

void LockCancellationDomainsInOrder(pqxx::transaction_base& transaction,
    const ReconciliationCandidate& candidate)
{
    const auto campaign = transaction.exec(
        "SELECT campaign_identity_canonical FROM "
        "campaign_operations_campaign WHERE operational_campaign_id=$1;",
        pqxx::params{candidate.campaignId.value()}).one_row();
    const auto canonical = campaign[0].as<std::string>();
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        "1179402835030004));", pqxx::params{canonical});
    transaction.exec(
        "SELECT lock_campaign_operations_budget_head($1);",
        pqxx::params{candidate.campaignId.value()});
    transaction.exec(
        "SELECT lock_campaign_operations_campaign($1);",
        pqxx::params{candidate.campaignId.value()});
    transaction.exec(
        "SELECT lock_campaign_operations_reservation($1);",
        pqxx::params{candidate.reservationId.value()});
    transaction.exec(
        "SELECT lock_campaign_operations_request($1);",
        pqxx::params{candidate.requestId.value()});
}

ReconciliationCandidate LoadLockedCandidate(
    pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    const pqxx::row row = transaction.exec(
        "SELECT request.operational_campaign_id,"
        "request.operational_request_id,request.reservation_id,"
        "request.request_identity_canonical,request.request_state,"
        "request.state_version,reservation.reservation_state,"
        "reservation.state_version,"
        "CASE WHEN reservation.expires_at IS NULL THEN NULL ELSE "
        "to_char(reservation.expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') END,"
        "(reservation.expires_at IS NOT NULL AND "
        " reservation.expires_at<=transaction_timestamp()),"
        "CASE WHEN request.lease_expires_at IS NULL THEN NULL ELSE "
        "to_char(request.lease_expires_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') END,"
        "(request.lease_expires_at IS NOT NULL AND "
        " request.lease_expires_at<=transaction_timestamp()),"
        "(SELECT count(*) FROM campaign_operations_request_binding binding "
        " WHERE binding.operational_request_id=request.operational_request_id),"
        "(SELECT count(*) FROM "
        " experiment_recommendation_campaign_materialization_member member "
        " JOIN experiment_recommendation_conversion_execution execution "
        " ON execution.recommendation_conversion_proposal_id="
        " member.recommendation_conversion_proposal_id "
        " WHERE member.recommendation_campaign_materialization_id="
        " request.recommendation_campaign_materialization_id),"
        "(SELECT count(*) FROM campaign_operations_dispatch_attempt attempt "
        " JOIN campaign_operations_dispatch_attempt_outcome outcome "
        " ON outcome.dispatch_attempt_id=attempt.dispatch_attempt_id "
        " WHERE attempt.operational_request_id=request.operational_request_id "
        " AND attempt.attempt_ordinal=("
        "  SELECT max(latest.attempt_ordinal) "
        "  FROM campaign_operations_dispatch_attempt latest "
        "  WHERE latest.operational_request_id="
        "   request.operational_request_id)),"
        "cancellation.cancellation_request_id,"
        "cancellation.cancellation_identity_canonical "
        "FROM campaign_operations_operational_request request "
        "JOIN campaign_operations_reservation reservation "
        "ON reservation.reservation_id=request.reservation_id "
        "LEFT JOIN LATERAL ("
        " SELECT candidate.cancellation_request_id,"
        " candidate.cancellation_identity_canonical "
        " FROM campaign_operations_cancellation_request candidate "
        " LEFT JOIN campaign_operations_cancellation_settlement settlement "
        " ON settlement.cancellation_request_id="
        " candidate.cancellation_request_id "
        " WHERE candidate.operational_request_id="
        " request.operational_request_id "
        " AND settlement.cancellation_settlement_id IS NULL "
        " ORDER BY candidate.cancellation_request_id "
        " LIMIT 1) cancellation ON true "
        "WHERE request.operational_request_id=$1;",
        pqxx::params{requestId.value()}).one_row();
    return {
        OperationalCampaignId(row[0].as<long long>()),
        OperationalRequestId(row[1].as<long long>()),
        ReservationId(row[2].as<long long>()),
        row[3].as<std::string>(),
        RequestStateFromText(row[4].as<std::string>()),
        row[5].as<int>(),
        ReservationStateFromText(row[6].as<std::string>()),
        row[7].as<int>(),
        row[8].is_null() ? std::nullopt
                         : std::optional<std::string>(
                               row[8].as<std::string>()),
        row[9].as<bool>(),
        row[10].is_null() ? std::nullopt
                          : std::optional<std::string>(
                                row[10].as<std::string>()),
        row[11].as<bool>(),
        row[12].as<int>(), row[13].as<int>(), row[14].as<int>(),
        row[15].is_null()
            ? std::nullopt
            : std::optional<CancellationRequestId>(
                  CancellationRequestId(row[15].as<long long>())),
        row[16].is_null()
            ? std::nullopt
            : std::optional<std::string>(row[16].as<std::string>())};
}

std::pair<DispatchAttemptId, std::string> LatestAttempt(
    pqxx::transaction_base& transaction, OperationalRequestId requestId)
{
    const auto row = transaction.exec(
        "SELECT dispatch_attempt_id,attempt_identity_canonical "
        "FROM campaign_operations_dispatch_attempt "
        "WHERE operational_request_id=$1 "
        "ORDER BY attempt_ordinal DESC LIMIT 1;",
        pqxx::params{requestId.value()}).one_row();
    return {DispatchAttemptId(row[0].as<long long>()),
        row[1].as<std::string>()};
}

DispatchAttemptOutcomeEvidence PersistRecoveryOutcome(
    pqxx::transaction_base& transaction,
    const ReconciliationCandidate& candidate,
    const std::string& diagnosticCode, int resultingRequestVersion,
    int resultingReservationVersion, const ActorIdentity& actor,
    const char* capability)
{
    const auto [attemptId, attemptCanonical] =
        LatestAttempt(transaction, candidate.requestId);
    const auto existing = transaction.exec(
        "SELECT outcome_identity_canonical,outcome_identity_hash "
        "FROM campaign_operations_dispatch_attempt_outcome "
        "WHERE dispatch_attempt_id=$1;",
        pqxx::params{attemptId.value()});
    auto outcome = BuildDispatchAttemptOutcomeEvidence(
        attemptId, attemptCanonical,
        DispatchResultClassification::rejected,
        DownstreamEvidenceClassification::noPhase5Evidence,
        SemanticConflictClassification::none,
        UncertainCommitRecoveryClassification::provenNoCommit,
        diagnosticCode, candidate.requestVersion,
        resultingRequestVersion, candidate.reservationVersion,
        resultingReservationVersion, std::nullopt, std::nullopt);
    if (!existing.empty())
    {
        if (existing.one_row()[0].as<std::string>() !=
                outcome.identity.canonicalText() ||
            existing.one_row()[1].as<std::string>() !=
                outcome.identity.hash())
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_dispatch_recovery_outcome_conflict");
        return outcome;
    }
    const long long outcomeId = transaction.exec(
        "INSERT INTO campaign_operations_dispatch_attempt_outcome("
        "dispatch_attempt_id,attempt_identity_canonical,"
        "result_classification,downstream_evidence_classification,"
        "semantic_conflict_classification,"
        "uncertain_commit_recovery_classification,diagnostic_code,"
        "expected_request_version,resulting_request_version,"
        "expected_reservation_version,resulting_reservation_version,"
        "outcome_contract_version,outcome_identity_canonical,"
        "outcome_identity_hash) VALUES($1,$2,'rejected',"
        "'no_phase5_evidence','none','proven_no_commit',$3,$4,$5,$6,$7,"
        "1,$8,$9) RETURNING dispatch_attempt_outcome_id;",
        pqxx::params{attemptId.value(), attemptCanonical, diagnosticCode,
            candidate.requestVersion, resultingRequestVersion,
            candidate.reservationVersion, resultingReservationVersion,
            outcome.identity.canonicalText(), outcome.identity.hash()})
        .one_row()[0].as<long long>();
    transaction.exec(
        "INSERT INTO campaign_operations_dispatch_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "dispatch_attempt_id,dispatch_attempt_outcome_id,cause_kind,"
        "actor_identity,capability,prior_version,resulting_version,outcome,"
        "replay_disposition,diagnostic_code) VALUES($1,$2,$3,$4,"
        "'dispatch_lease_recovered',$5,$6,$7,$8,'recorded',"
        "'proven_absent',$9);",
        pqxx::params{candidate.campaignId.value(),
            candidate.requestId.value(), attemptId.value(), outcomeId,
            actor.value(), capability, candidate.requestVersion,
            resultingRequestVersion, diagnosticCode});
    return outcome;
}

PersistedReconciliationResolution AppendReconciliationResolution(
    pqxx::transaction_base& transaction,
    const PersistedReconciliationObservation& persistedObservation,
    const std::string& owningService, const char* owningCapability,
    const std::string& transitionCanonicalText,
    const std::string& transitionIdentityHash,
    const std::string& disposition, int priorVersion,
    int resultingVersion)
{
    if (const auto existing = FindReconciliationResolution(
            transaction, persistedObservation.observationId))
        return *existing;
    const auto& observation = persistedObservation.observation;
    auto resolution = BuildReconciliationResolution(
        persistedObservation.observationId,
        observation.identity.canonicalText(), owningService,
        owningCapability, transitionCanonicalText,
        transitionIdentityHash, disposition);
    pqxx::row row;
    if (resolution.owningCapability ==
            kCampaignOperationsCancellationRole)
        row = transaction.exec(
            "SELECT reconciliation_resolution_id,created_at::text "
            "FROM append_campaign_operations_cancellation_resolution("
            "$1,$2,$3,$4,$5);",
            pqxx::params{persistedObservation.observationId.value(),
                resolution.transitionCanonicalText,
                resolution.transitionIdentityHash,
                resolution.identity.canonicalText(),
                resolution.identity.hash()}).one_row();
    else if (resolution.owningCapability ==
             kCampaignOperationsRecoveryRole)
        row = transaction.exec(
            "SELECT reconciliation_resolution_id,created_at::text "
            "FROM append_campaign_operations_recovery_resolution("
            "$1,$2,$3,$4,$5,$6);",
            pqxx::params{persistedObservation.observationId.value(),
                resolution.transitionCanonicalText,
                resolution.transitionIdentityHash,
                resolution.disposition,
                resolution.identity.canonicalText(),
                resolution.identity.hash()}).one_row();
    else
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_reconciliation_capability_invalid");
    const auto resolutionId =
        ReconciliationResolutionId(row[0].as<long long>());
    transaction.exec(
        "INSERT INTO campaign_operations_control_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "reconciliation_observation_id,reconciliation_resolution_id,"
        "cause_kind,actor_identity,capability,reason,prior_version,"
        "resulting_version,outcome,diagnostic_code) VALUES($1,$2,$3,$4,"
        "'reconciliation_resolved',$5,$5,$6,$7,$8,'recorded',$6);",
        pqxx::params{observation.campaignId.value(),
            observation.requestId.value(),
            persistedObservation.observationId.value(),
            resolutionId.value(), owningCapability,
            observation.diagnosticCode, priorVersion, resultingVersion});
    return {resolutionId, std::move(resolution),
        row[1].as<std::string>()};
}

void ResolveExactCancellationObservations(
    pqxx::transaction_base& transaction,
    const PersistedCampaignCancellationRequest& cancellation,
    const ReconciliationCandidate& candidate,
    const PersistedCampaignCancellationSettlement& settlement)
{
    if (!cancellation.request.requestId ||
        !cancellation.request.requestCanonicalText ||
        !cancellation.request.expectedRequestState ||
        !cancellation.request.expectedRequestVersion ||
        candidate.campaignId != cancellation.request.campaignId ||
        candidate.requestId != *cancellation.request.requestId ||
        candidate.requestCanonicalText !=
            *cancellation.request.requestCanonicalText ||
        candidate.requestState !=
            *cancellation.request.expectedRequestState ||
        candidate.requestVersion !=
            *cancellation.request.expectedRequestVersion ||
        !candidate.cancellationRequestId ||
        *candidate.cancellationRequestId !=
            cancellation.cancellationRequestId ||
        !candidate.cancellationCanonicalText ||
        *candidate.cancellationCanonicalText !=
            cancellation.request.identity.canonicalText())
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_cancellation_resolution_evidence_changed");

    std::vector<std::string> acceptedEvidence;
    const auto appendEvidence = [&](const ReconciliationCandidate& evidence)
    {
        const auto canonical =
            BuildReconciliationEvidenceCanonical(evidence);
        if (std::find(acceptedEvidence.begin(), acceptedEvidence.end(),
                canonical) == acceptedEvidence.end())
            acceptedEvidence.push_back(canonical);
    };
    appendEvidence(candidate);
    if (candidate.leaseExpired)
    {
        auto activeLeaseEvidence = candidate;
        activeLeaseEvidence.leaseExpired = false;
        appendEvidence(activeLeaseEvidence);
    }
    if (candidate.reservationExpired)
    {
        auto activeReservationEvidence = candidate;
        activeReservationEvidence.reservationExpired = false;
        appendEvidence(activeReservationEvidence);
        if (candidate.leaseExpired)
        {
            activeReservationEvidence.leaseExpired = false;
            appendEvidence(activeReservationEvidence);
        }
    }
    const auto rows = transaction.exec(
        "SELECT observation.reconciliation_observation_id,"
        "observation.run_key,observation.operational_campaign_id,"
        "observation.operational_request_id,"
        "observation.request_identity_canonical,"
        "observation.expected_request_state,"
        "observation.expected_request_version,observation.reason_code,"
        "observation.evidence_identity_canonical,"
        "observation.evidence_identity_hash,"
        "observation.recommended_service,"
        "observation.recommended_action,observation.diagnostic_code,"
        "observation.observation_identity_canonical,"
        "observation.observation_identity_hash,"
        "observation.observed_at::text "
        "FROM campaign_operations_reconciliation_observation observation "
        "LEFT JOIN campaign_operations_reconciliation_resolution resolution "
        "ON resolution.reconciliation_observation_id="
        "observation.reconciliation_observation_id "
        "WHERE resolution.reconciliation_resolution_id IS NULL "
        "AND observation.operational_campaign_id=$1 "
        "AND observation.operational_request_id=$2 "
        "AND observation.request_identity_canonical=$3 "
        "AND observation.expected_request_state=$4 "
        "AND observation.expected_request_version=$5 "
        "AND observation.reason_code='cancellation_settlement_pending' "
        "AND observation.recommended_service="
        "'campaign_operations_cancellation_coordinator' "
        "AND observation.recommended_action='replay_cancellation' "
        "AND observation.diagnostic_code='cancellation_settlement_pending' "
        "ORDER BY observation.reconciliation_observation_id;",
        pqxx::params{candidate.campaignId.value(),
            candidate.requestId.value(), candidate.requestCanonicalText,
            ToText(candidate.requestState), candidate.requestVersion});
    const int resultingVersion =
        settlement.settlement.resultingRequestVersion.value_or(
            candidate.requestVersion);
    for (const auto& row : rows)
    {
        const auto observation = MapObservation(row);
        if (std::find(acceptedEvidence.begin(), acceptedEvidence.end(),
                observation.observation.evidenceCanonicalText) ==
                acceptedEvidence.end())
            continue;
        const auto evidenceHash = CanonicalIdentity::Create(
            kCampaignOperationsControlContractVersion,
            observation.observation.evidenceCanonicalText).hash();
        if (evidenceHash !=
            observation.observation.evidenceIdentityHash)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_cancellation_observation_corrupt");
        (void)AppendReconciliationResolution(transaction, observation,
            "campaign_operations_cancellation_coordinator",
            kCampaignOperationsCancellationRole,
            settlement.settlement.identity.canonicalText(),
            settlement.settlement.identity.hash(), "cancellation_settled",
            observation.observation.expectedRequestVersion,
            resultingVersion);
    }
}

} // namespace

bool ControlSchemaExists(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('campaign_operations_control_event') "
        "IS NOT NULL AND to_regclass("
        "'campaign_operations_cancellation_request') IS NOT NULL AND "
        "to_regclass('campaign_operations_reconciliation_observation') "
        "IS NOT NULL;").one_row()[0].as<bool>();
}

std::optional<PersistedCampaignControlEvent> FindCampaignControlHead(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId)
{
    const auto rows = transaction.exec(
        "SELECT control_event_id,operational_campaign_id,"
        "campaign_identity_canonical,previous_control_event_id,"
        "previous_control_event_identity_canonical,control_version,"
        "event_kind,actor_identity,reason,control_identity_canonical,"
        "control_identity_hash,created_at::text "
        "FROM campaign_operations_control_event "
        "WHERE operational_campaign_id=$1 "
        "ORDER BY control_version DESC LIMIT 1;",
        pqxx::params{campaignId.value()});
    if (rows.empty()) return std::nullopt;
    return MapControl(rows.one_row());
}

PersistedCampaignControlEvent PersistCampaignControlEvent(
    pqxx::transaction_base& transaction,
    const CampaignControlEvent& event)
{
    ValidateCampaignControlEvent(event);
    const auto existing = transaction.exec(
        "SELECT control_event_id,operational_campaign_id,"
        "campaign_identity_canonical,previous_control_event_id,"
        "previous_control_event_identity_canonical,control_version,"
        "event_kind,actor_identity,reason,control_identity_canonical,"
        "control_identity_hash,created_at::text "
        "FROM campaign_operations_control_event "
        "WHERE operational_campaign_id=$1 AND control_version=$2;",
        pqxx::params{event.campaignId.value(), event.controlVersion});
    if (!existing.empty())
    {
        auto persisted = MapControl(existing.one_row());
        if (persisted.event != event)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_control_event_conflict");
        return persisted;
    }
    const std::optional<long long> previousEventId =
        event.previousEventId
        ? std::optional<long long>(event.previousEventId->value())
        : std::nullopt;
    const auto row = transaction.exec(
        "INSERT INTO campaign_operations_control_event("
        "operational_campaign_id,campaign_identity_canonical,"
        "previous_control_event_id,"
        "previous_control_event_identity_canonical,control_version,"
        "event_kind,actor_identity,capability,reason,"
        "control_contract_version,control_identity_canonical,"
        "control_identity_hash) VALUES($1,$2,$3,$4,$5,$6,$7,"
        "'campaign_operations_controller',$8,1,$9,$10) "
        "RETURNING control_event_id,created_at::text;",
        pqxx::params{event.campaignId.value(),
            event.campaignCanonicalText,
            previousEventId,
            event.previousEventCanonicalText, event.controlVersion,
            ToText(event.eventKind), event.actor.value(),
            event.reason.value(), event.identity.canonicalText(),
            event.identity.hash()}).one_row();
    const auto id = ControlEventId(row[0].as<long long>());
    transaction.exec(
        "INSERT INTO campaign_operations_control_audit_reference_event("
        "operational_campaign_id,control_event_id,cause_kind,"
        "actor_identity,capability,reason,prior_version,"
        "resulting_version,outcome,diagnostic_code) VALUES($1,$2,$3,$4,"
        "'campaign_operations_controller',$5,$6,$7,'recorded',$3);",
        pqxx::params{event.campaignId.value(), id.value(),
            event.eventKind == ControlEventKind::pause
                ? "campaign_paused" : "campaign_resumed",
            event.actor.value(), event.reason.value(),
            event.controlVersion - 1, event.controlVersion});
    return {id, event, row[1].as<std::string>()};
}

std::optional<PersistedCampaignCancellationRequest>
FindCampaignCancellationRequestByOperation(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId, const std::string& operationKey)
{
    const auto rows = transaction.exec(
        "SELECT cancellation_request_id,operational_campaign_id,"
        "campaign_identity_canonical,operational_request_id,"
        "request_identity_canonical,expected_request_state,"
        "expected_request_version,operation_key,actor_identity,reason,"
        "cancellation_identity_canonical,cancellation_identity_hash,"
        "created_at::text "
        "FROM campaign_operations_cancellation_request "
        "WHERE operational_campaign_id=$1 "
        "AND cancellation_scope='complete_materialization' "
        "AND operation_key=$2;",
        pqxx::params{campaignId.value(), operationKey});
    if (rows.empty()) return std::nullopt;
    return MapCancellationRequest(rows.one_row());
}

std::optional<PersistedCampaignCancellationRequest>
FindCampaignCancellationRequestByTarget(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId,
    std::optional<OperationalRequestId> requestId)
{
    const auto rows = transaction.exec(
        "SELECT cancellation_request_id,operational_campaign_id,"
        "campaign_identity_canonical,operational_request_id,"
        "request_identity_canonical,expected_request_state,"
        "expected_request_version,operation_key,actor_identity,reason,"
        "cancellation_identity_canonical,cancellation_identity_hash,"
        "created_at::text "
        "FROM campaign_operations_cancellation_request "
        "WHERE operational_campaign_id=$1 "
        "AND operational_request_id IS NOT DISTINCT FROM $2 "
        "ORDER BY cancellation_request_id LIMIT 1;",
        pqxx::params{campaignId.value(),
            requestId
                ? std::optional<long long>(requestId->value())
                : std::nullopt});
    if (rows.empty()) return std::nullopt;
    return MapCancellationRequest(rows.one_row());
}

PersistedCampaignCancellationRequest PersistCampaignCancellationRequest(
    pqxx::transaction_base& transaction,
    const CampaignCancellationRequest& request)
{
    ValidateCampaignCancellationRequest(request);
    if (const auto existing = FindCampaignCancellationRequestByOperation(
            transaction, request.campaignId, request.operationKey))
    {
        if (existing->request != request)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_cancellation_request_conflict");
        return *existing;
    }
    const auto row = transaction.exec(
        "INSERT INTO campaign_operations_cancellation_request("
        "operational_campaign_id,campaign_identity_canonical,"
        "operational_request_id,request_identity_canonical,"
        "expected_request_state,expected_request_version,"
        "cancellation_scope,operation_key,actor_identity,capability,reason,"
        "cancellation_contract_version,cancellation_identity_canonical,"
        "cancellation_identity_hash) VALUES($1,$2,$3,$4,$5,$6,"
        "'complete_materialization',$7,$8,"
        "'campaign_operations_cancellation_coordinator',$9,1,$10,$11) "
        "RETURNING cancellation_request_id,created_at::text;",
        pqxx::params{request.campaignId.value(),
            request.campaignCanonicalText,
            request.requestId
                ? std::optional<long long>(request.requestId->value())
                : std::nullopt,
            request.requestCanonicalText,
            request.expectedRequestState
                ? std::optional<std::string>(
                      ToText(*request.expectedRequestState))
                : std::nullopt,
            request.expectedRequestVersion, request.operationKey,
            request.actor.value(), request.reason.value(),
            request.identity.canonicalText(), request.identity.hash()})
        .one_row();
    const auto id = CancellationRequestId(row[0].as<long long>());
    transaction.exec(
        "INSERT INTO campaign_operations_control_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "cancellation_request_id,cause_kind,actor_identity,capability,"
        "reason,prior_version,resulting_version,outcome,diagnostic_code) "
        "VALUES($1,$2,$3,'cancellation_requested',$4,"
        "'campaign_operations_cancellation_coordinator',$5,$6,$6,"
        "'recorded','cancellation_requested');",
        pqxx::params{request.campaignId.value(),
            request.requestId
                ? std::optional<long long>(request.requestId->value())
                : std::nullopt,
            id.value(), request.actor.value(), request.reason.value(),
            request.expectedRequestVersion});
    return {id, request, row[1].as<std::string>()};
}

std::optional<PersistedCampaignCancellationSettlement>
FindCampaignCancellationSettlement(
    pqxx::transaction_base& transaction,
    CancellationRequestId cancellationRequestId)
{
    const auto rows = transaction.exec(
        "SELECT cancellation_settlement_id,cancellation_request_id,"
        "cancellation_request_identity_canonical,reservation_event_id,"
        "reservation_event_identity_canonical,resulting_request_version,"
        "lifecycle_evidence_identity_canonical,"
        "lifecycle_evidence_identity_hash,disposition,"
        "settlement_identity_canonical,settlement_identity_hash,"
        "created_at::text "
        "FROM campaign_operations_cancellation_settlement "
        "WHERE cancellation_request_id=$1;",
        pqxx::params{cancellationRequestId.value()});
    if (rows.empty()) return std::nullopt;
    return MapSettlement(rows.one_row());
}

PersistedCampaignCancellationSettlement
SettleUnboundCancellationInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedCampaignCancellationRequest& cancellation,
    const ReconciliationCandidate& snapshot,
    std::optional<ReconciliationObservationId> observationId)
{
    if (const auto existing = FindCampaignCancellationSettlement(
            transaction, cancellation.cancellationRequestId))
        return *existing;
    LockCancellationDomains(transaction, snapshot);
    if (const auto existing = FindCampaignCancellationSettlement(
            transaction, cancellation.cancellationRequestId))
        return *existing;
    const auto candidate =
        LoadLockedCandidate(transaction, snapshot.requestId);
    if (candidate.requestState != snapshot.requestState ||
        candidate.requestVersion != snapshot.requestVersion ||
        candidate.reservationState != ReservationState::held ||
        candidate.reservationVersion != snapshot.reservationVersion ||
        candidate.bindingCount != 0 ||
        candidate.downstreamExecutionCount != 0 ||
        (candidate.requestState == RequestState::dispatching &&
            candidate.dispatchOutcomeCount != 0) ||
        (candidate.requestState == RequestState::dispatching &&
            transaction.exec(
                "SELECT $1::timestamptz > transaction_timestamp();",
                pqxx::params{*candidate.leaseExpiresAt})
                .one_row()[0].as<bool>()))
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_unbound_cancellation_evidence_changed");

    const int resultingRequestVersion = candidate.requestVersion + 1;
    const int resultingReservationVersion =
        candidate.reservationVersion + 1;
    transaction.exec(
        "SELECT transition_campaign_operations_request_cancelled($1,$2);",
        pqxx::params{candidate.requestId.value(),
            candidate.requestVersion});
    transaction.exec(
        "SELECT transition_campaign_operations_reservation_released($1,$2);",
        pqxx::params{candidate.reservationId.value(),
            candidate.reservationVersion});

    if (candidate.requestState == RequestState::dispatching)
        (void)PersistRecoveryOutcome(transaction, candidate,
            "cancellation_recovered_expired_dispatch",
            resultingRequestVersion, resultingReservationVersion,
            cancellation.request.actor,
            kCampaignOperationsCancellationRole);

    const auto reservation = transaction.exec(
        "SELECT reservation_identity_canonical,amount "
        "FROM campaign_operations_reservation WHERE reservation_id=$1;",
        pqxx::params{candidate.reservationId.value()}).one_row();
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_reservation_event_v1"
        << ";reservation_id=" << candidate.reservationId.value()
        << ";reservation_canonical="
        << Framed(reservation[0].as<std::string>())
        << ";transition_kind=released"
        << ";expected_state=held;resulting_state=released"
        << ";expected_version=" << candidate.reservationVersion
        << ";resulting_version=" << resultingReservationVersion
        << ";request_id=" << candidate.requestId.value()
        << ";request_canonical=" << Framed(candidate.requestCanonicalText)
        << ";amount=" << reservation[1].as<long long>()
        << ";cancellation_request_id="
        << cancellation.cancellationRequestId.value()
        << ";reconciliation_observation_id="
        << (observationId
                ? std::to_string(observationId->value()) : "none");
    const auto eventIdentity = Identity(canonical.str());
    const auto eventRow = transaction.exec(
        "INSERT INTO campaign_operations_reservation_event("
        "reservation_id,reservation_identity_canonical,transition_kind,"
        "expected_state,resulting_state,expected_version,resulting_version,"
        "operational_request_id,request_identity_canonical,amount,"
        "reservation_event_contract_version,"
        "reservation_event_identity_canonical,"
        "reservation_event_identity_hash,cancellation_request_id,"
        "reconciliation_observation_id) VALUES($1,$2,'released','held',"
        "'released',$3,$4,$5,$6,$7,1,$8,$9,$10,$11) "
        "RETURNING reservation_event_id;",
        pqxx::params{candidate.reservationId.value(),
            reservation[0].as<std::string>(),
            candidate.reservationVersion, resultingReservationVersion,
            candidate.requestId.value(), candidate.requestCanonicalText,
            reservation[1].as<long long>(),
            eventIdentity.canonicalText(), eventIdentity.hash(),
            cancellation.cancellationRequestId.value(),
            observationId
                ? std::optional<long long>(observationId->value())
                : std::nullopt}).one_row();
    const auto reservationEventId =
        ReservationEventId(eventRow[0].as<long long>());
    auto settlement = BuildCampaignCancellationSettlement(
        cancellation.cancellationRequestId,
        cancellation.request.identity.canonicalText(),
        CancellationSettlementDisposition::unboundCancelled,
        reservationEventId, eventIdentity.canonicalText(),
        resultingRequestVersion, std::nullopt, std::nullopt);
    const auto settlementRow = transaction.exec(
        "INSERT INTO campaign_operations_cancellation_settlement("
        "cancellation_request_id,"
        "cancellation_request_identity_canonical,disposition,"
        "reservation_event_id,reservation_event_identity_canonical,"
        "resulting_request_version,settlement_contract_version,"
        "settlement_identity_canonical,settlement_identity_hash) "
        "VALUES($1,$2,'unbound_cancelled',$3,$4,$5,1,$6,$7) "
        "RETURNING cancellation_settlement_id,created_at::text;",
        pqxx::params{cancellation.cancellationRequestId.value(),
            cancellation.request.identity.canonicalText(),
            reservationEventId.value(), eventIdentity.canonicalText(),
            resultingRequestVersion, settlement.identity.canonicalText(),
            settlement.identity.hash()}).one_row();
    const auto settlementId =
        CancellationSettlementId(settlementRow[0].as<long long>());
    transaction.exec(
        "INSERT INTO campaign_operations_control_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "cancellation_request_id,cancellation_settlement_id,cause_kind,"
        "actor_identity,capability,reason,prior_version,resulting_version,"
        "outcome,diagnostic_code) VALUES($1,$2,$3,$4,"
        "'cancellation_settled',$5,"
        "'campaign_operations_cancellation_coordinator',$6,$7,$8,"
        "'recorded','unbound_cancelled');",
        pqxx::params{candidate.campaignId.value(),
            candidate.requestId.value(),
            cancellation.cancellationRequestId.value(),
            settlementId.value(), cancellation.request.actor.value(),
            cancellation.request.reason.value(), candidate.requestVersion,
            resultingRequestVersion});
    PersistedCampaignCancellationSettlement persistedSettlement{
        settlementId, std::move(settlement),
        settlementRow[1].as<std::string>()};
    ResolveExactCancellationObservations(transaction, cancellation,
        candidate, persistedSettlement);
    return persistedSettlement;
}

void LockCancellationDomains(pqxx::transaction_base& transaction,
    const ReconciliationCandidate& candidate)
{
    LockCancellationDomainsInOrder(transaction, candidate);
}

std::vector<std::pair<DownstreamControlOwnerId, long long>>
LoadDownstreamControlOwners(pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    const auto rows = transaction.exec(
        "SELECT downstream_control_owner_id,experiment_id "
        "FROM campaign_operations_downstream_control_owner "
        "WHERE operational_request_id=$1 ORDER BY experiment_id;",
        pqxx::params{requestId.value()});
    std::vector<std::pair<DownstreamControlOwnerId, long long>> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
        result.emplace_back(
            DownstreamControlOwnerId(row[0].as<long long>()),
            row[1].as<long long>());
    return result;
}

LifecycleCancellationEvidence ApplyLifecycleCancellationInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedCampaignCancellationRequest& cancellation,
    DownstreamControlOwnerId controlOwnerId, long long experimentId)
{
    std::string status;
    std::string phase;
    std::string canonical;
    std::string hash;
    const auto existing = transaction.exec(
        "SELECT expected_status,expected_phase,event_identity_canonical,"
        "event_identity_hash FROM experiment_lifecycle_cancellation_event "
        "WHERE cancellation_request_id=$1 "
        "AND downstream_control_owner_id=$2 AND experiment_id=$3;",
        pqxx::params{cancellation.cancellationRequestId.value(),
            controlOwnerId.value(), experimentId});
    if (!existing.empty())
    {
        status = existing.one_row()[0].as<std::string>();
        phase = existing.one_row()[1].as<std::string>();
        canonical = existing.one_row()[2].as<std::string>();
        hash = existing.one_row()[3].as<std::string>();
    }
    else
    {
        const auto current = transaction.exec(
            "SELECT status,phase FROM experiment WHERE experiment_id=$1;",
            pqxx::params{experimentId}).one_row();
        status = current[0].as<std::string>();
        phase = current[1].as<std::string>();
        const LifecycleCancellationDisposition disposition =
            status == "pending" || status == "paused"
                ? LifecycleCancellationDisposition::accepted
                : status == "running"
                    ? LifecycleCancellationDisposition::runningNotSupported
                    : LifecycleCancellationDisposition::alreadyTerminal;
        const auto resulting =
            disposition == LifecycleCancellationDisposition::accepted
                ? "cancelled" : status;
        std::ostringstream candidate;
        candidate.imbue(std::locale::classic());
        candidate << "experiment_lifecycle_cancellation_event_v1"
            << ";cancellation_request_id="
            << cancellation.cancellationRequestId.value()
            << ";cancellation_request_canonical="
            << Framed(cancellation.request.identity.canonicalText())
            << ";control_owner_id=" << controlOwnerId.value()
            << ";experiment_id=" << experimentId
            << ";expected_status=" << status
            << ";expected_phase=" << phase
            << ";observed_status=" << status
            << ";observed_phase=" << phase
            << ";resulting_status=" << resulting
            << ";disposition=" << ToText(disposition)
            << ";actor=" << Framed(cancellation.request.actor.value());
        const auto identity = Identity(candidate.str());
        canonical = identity.canonicalText();
        hash = identity.hash();
    }
    const auto row = transaction.exec(
        "SELECT lifecycle_cancellation_event_id,cancellation_request_id,"
        "downstream_control_owner_id,experiment_id,observed_status,"
        "observed_phase,resulting_status,disposition,"
        "event_identity_canonical,event_identity_hash "
        "FROM apply_experiment_lifecycle_cancellation("
        "$1,$2,$3,$4,$5,$6,$7,$8);",
        pqxx::params{cancellation.cancellationRequestId.value(),
            controlOwnerId.value(), experimentId, status, phase,
            cancellation.request.actor.value(), canonical, hash}).one_row();
    const auto persistedIdentity = CanonicalIdentity::Hydrate(
        1, row[8].as<std::string>(), row[9].as<std::string>());
    return {row[0].as<long long>(),
        CancellationRequestId(row[1].as<long long>()),
        DownstreamControlOwnerId(row[2].as<long long>()),
        row[3].as<long long>(), row[4].as<std::string>(),
        row[5].as<std::string>(), row[6].as<std::string>(),
        row[7].as<std::string>() == "accepted"
            ? LifecycleCancellationDisposition::accepted
            : row[7].as<std::string>() == "already_terminal"
                ? LifecycleCancellationDisposition::alreadyTerminal
                : LifecycleCancellationDisposition::runningNotSupported,
        persistedIdentity};
}

std::vector<LifecycleCancellationEvidence>
LoadLifecycleCancellationEvidence(pqxx::transaction_base& transaction,
    CancellationRequestId cancellationRequestId)
{
    const auto rows = transaction.exec(
        "SELECT lifecycle_cancellation_event_id,cancellation_request_id,"
        "downstream_control_owner_id,experiment_id,observed_status,"
        "observed_phase,resulting_status,disposition,"
        "event_identity_canonical,event_identity_hash "
        "FROM experiment_lifecycle_cancellation_event "
        "WHERE cancellation_request_id=$1 ORDER BY experiment_id;",
        pqxx::params{cancellationRequestId.value()});
    std::vector<LifecycleCancellationEvidence> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
    {
        const auto identity = CanonicalIdentity::Hydrate(
            1, row[8].as<std::string>(), row[9].as<std::string>());
        result.push_back({row[0].as<long long>(),
            CancellationRequestId(row[1].as<long long>()),
            DownstreamControlOwnerId(row[2].as<long long>()),
            row[3].as<long long>(), row[4].as<std::string>(),
            row[5].as<std::string>(), row[6].as<std::string>(),
            row[7].as<std::string>() == "accepted"
                ? LifecycleCancellationDisposition::accepted
                : row[7].as<std::string>() == "already_terminal"
                    ? LifecycleCancellationDisposition::alreadyTerminal
                    : LifecycleCancellationDisposition::
                        runningNotSupported,
            identity});
    }
    return result;
}

PersistedCampaignCancellationSettlement
SettleBoundCancellationInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedCampaignCancellationRequest& cancellation,
    const std::vector<LifecycleCancellationEvidence>& evidence)
{
    if (const auto existing = FindCampaignCancellationSettlement(
            transaction, cancellation.cancellationRequestId))
        return *existing;
    if (evidence.empty())
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_lifecycle_cancellation_evidence_missing");
    if (!cancellation.request.requestId)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_bound_cancellation_target_missing");
    const auto snapshot = LoadLockedCandidate(
        transaction, *cancellation.request.requestId);
    transaction.exec(
        "SELECT lock_campaign_operations_campaign($1);",
        pqxx::params{snapshot.campaignId.value()});
    transaction.exec(
        "SELECT lock_campaign_operations_reservation($1);",
        pqxx::params{snapshot.reservationId.value()});
    transaction.exec(
        "SELECT lock_campaign_operations_request($1);",
        pqxx::params{snapshot.requestId.value()});
    if (const auto existing = FindCampaignCancellationSettlement(
            transaction, cancellation.cancellationRequestId))
        return *existing;
    const auto candidate = LoadLockedCandidate(
        transaction, *cancellation.request.requestId);
    if (candidate.requestState != RequestState::bound ||
        candidate.reservationState != ReservationState::committed ||
        candidate.bindingCount != static_cast<int>(evidence.size()))
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_bound_cancellation_evidence_changed");
    bool anyAccepted = false;
    bool anyRunning = false;
    std::ostringstream setCanonical;
    setCanonical.imbue(std::locale::classic());
    setCanonical
        << "campaign_operations_lifecycle_cancellation_evidence_set_v1"
        << ";count=" << evidence.size();
    for (const auto& item : evidence)
    {
        if (item.cancellationRequestId !=
            cancellation.cancellationRequestId)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_lifecycle_cancellation_cause_mismatch");
        anyAccepted |= item.disposition ==
            LifecycleCancellationDisposition::accepted;
        anyRunning |= item.disposition ==
            LifecycleCancellationDisposition::runningNotSupported;
        setCanonical << ";item=" << Framed(
            item.identity.canonicalText());
    }
    const auto evidenceIdentity = Identity(setCanonical.str());
    const auto disposition = anyRunning
        ? CancellationSettlementDisposition::
            runningCancellationNotSupported
        : anyAccepted
            ? CancellationSettlementDisposition::lifecycleRequestAccepted
            : CancellationSettlementDisposition::alreadyTerminal;
    auto settlement = BuildCampaignCancellationSettlement(
        cancellation.cancellationRequestId,
        cancellation.request.identity.canonicalText(), disposition,
        std::nullopt, std::nullopt, std::nullopt,
        evidenceIdentity.canonicalText(), evidenceIdentity.hash());
    const auto row = transaction.exec(
        "INSERT INTO campaign_operations_cancellation_settlement("
        "cancellation_request_id,"
        "cancellation_request_identity_canonical,disposition,"
        "lifecycle_evidence_identity_canonical,"
        "lifecycle_evidence_identity_hash,settlement_contract_version,"
        "settlement_identity_canonical,settlement_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,1,$6,$7) "
        "RETURNING cancellation_settlement_id,created_at::text;",
        pqxx::params{cancellation.cancellationRequestId.value(),
            cancellation.request.identity.canonicalText(),
            ToText(disposition), evidenceIdentity.canonicalText(),
            evidenceIdentity.hash(), settlement.identity.canonicalText(),
            settlement.identity.hash()}).one_row();
    const auto settlementId =
        CancellationSettlementId(row[0].as<long long>());
    transaction.exec(
        "INSERT INTO campaign_operations_control_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "cancellation_request_id,cancellation_settlement_id,cause_kind,"
        "actor_identity,capability,reason,prior_version,resulting_version,"
        "outcome,diagnostic_code) VALUES($1,$2,$3,$4,"
        "'cancellation_settled',$5,"
        "'campaign_operations_cancellation_coordinator',$6,$7,$7,"
        "'recorded',$8);",
        pqxx::params{cancellation.request.campaignId.value(),
            cancellation.request.requestId
                ? std::optional<long long>(
                      cancellation.request.requestId->value())
                : std::nullopt,
            cancellation.cancellationRequestId.value(),
            settlementId.value(), cancellation.request.actor.value(),
            cancellation.request.reason.value(),
            cancellation.request.expectedRequestVersion,
            ToText(disposition)});
    PersistedCampaignCancellationSettlement persistedSettlement{
        settlementId, std::move(settlement), row[1].as<std::string>()};
    ResolveExactCancellationObservations(transaction, cancellation,
        candidate, persistedSettlement);
    return persistedSettlement;
}

std::vector<ReconciliationCandidate> SelectReconciliationCandidates(
    pqxx::transaction_base& transaction, long long afterRequestId,
    int limit)
{
    if (afterRequestId < 0 || limit <= 0 ||
        limit > kCampaignOperationsMaximumReconciliationBatchSize)
        throw std::invalid_argument(
            "campaign_operations_reconciliation_cursor_invalid");
    const auto rows = transaction.exec(
        "SELECT request.operational_request_id "
        "FROM campaign_operations_operational_request request "
        "WHERE request.operational_request_id>$1 AND ("
        "(request.request_state='dispatching' AND "
        " request.lease_expires_at<=transaction_timestamp()) OR "
        "EXISTS(SELECT 1 FROM campaign_operations_reservation reservation "
        " WHERE reservation.reservation_id=request.reservation_id "
        " AND reservation.reservation_state='held' "
        " AND reservation.expires_at IS NOT NULL "
        " AND reservation.expires_at<=transaction_timestamp()) OR "
        "request.request_state='reconciliation_required' OR EXISTS("
        " SELECT 1 FROM campaign_operations_cancellation_request cancellation "
        " LEFT JOIN campaign_operations_cancellation_settlement settlement "
        " ON settlement.cancellation_request_id="
        " cancellation.cancellation_request_id "
        " WHERE cancellation.operational_request_id="
        " request.operational_request_id "
        " AND settlement.cancellation_settlement_id IS NULL)) "
        "ORDER BY request.operational_request_id LIMIT $2;",
        pqxx::params{afterRequestId, limit});
    std::vector<ReconciliationCandidate> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
        result.push_back(LoadLockedCandidate(
            transaction,
            OperationalRequestId(row[0].as<long long>())));
    return result;
}

ReconciliationCandidate LoadReconciliationCandidate(
    pqxx::transaction_base& transaction,
    OperationalRequestId requestId)
{
    return LoadLockedCandidate(transaction, requestId);
}

std::string BuildReconciliationEvidenceCanonical(
    const ReconciliationCandidate& candidate)
{
    std::ostringstream result;
    result.imbue(std::locale::classic());
    result << "campaign_operations_reconciliation_evidence_v1"
        << ";campaign_id=" << candidate.campaignId.value()
        << ";request_id=" << candidate.requestId.value()
        << ";request_canonical=" << Framed(candidate.requestCanonicalText)
        << ";request_state=" << ToText(candidate.requestState)
        << ";request_version=" << candidate.requestVersion
        << ";reservation_id=" << candidate.reservationId.value()
        << ";reservation_state=" << ToText(candidate.reservationState)
        << ";reservation_version=" << candidate.reservationVersion
        << ";reservation_expires_at="
        << candidate.reservationExpiresAt.value_or("none")
        << ";reservation_expired="
        << (candidate.reservationExpired ? "true" : "false")
        << ";lease_expires_at="
        << candidate.leaseExpiresAt.value_or("none")
        << ";lease_expired="
        << (candidate.leaseExpired ? "true" : "false")
        << ";binding_count=" << candidate.bindingCount
        << ";downstream_execution_count="
        << candidate.downstreamExecutionCount
        << ";dispatch_outcome_count=" << candidate.dispatchOutcomeCount
        << ";cancellation_request_id="
        << (candidate.cancellationRequestId
                ? std::to_string(candidate.cancellationRequestId->value())
                : "none")
        << ";cancellation_canonical="
        << (candidate.cancellationCanonicalText
                ? Framed(*candidate.cancellationCanonicalText)
                : "none");
    return result.str();
}

PersistedReconciliationObservation InsertReconciliationObservation(
    pqxx::transaction_base& transaction,
    long long reconciliationCursorEventId,
    const ReconciliationObservation& observation)
{
    ValidateReconciliationObservation(observation);
    transaction.exec(
        "SELECT lock_campaign_operations_campaign($1);",
        pqxx::params{observation.campaignId.value()});
    const auto target = LoadLockedCandidate(
        transaction, observation.requestId);
    if (target.campaignId != observation.campaignId ||
        target.requestCanonicalText != observation.requestCanonicalText ||
        target.requestState != observation.expectedRequestState ||
        target.requestVersion != observation.expectedRequestVersion ||
        BuildReconciliationEvidenceCanonical(target) !=
            observation.evidenceCanonicalText)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_reconciliation_target_changed");
    const auto row = transaction.exec(
        "INSERT INTO campaign_operations_reconciliation_observation("
        "reconciliation_cursor_event_id,run_key,"
        "operational_campaign_id,operational_request_id,"
        "request_identity_canonical,expected_request_state,"
        "expected_request_version,reason_code,evidence_identity_canonical,"
        "evidence_identity_hash,recommended_service,recommended_action,"
        "diagnostic_code,observation_contract_version,"
        "observation_identity_canonical,observation_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,1,$14,$15) "
        "RETURNING reconciliation_observation_id,observed_at::text;",
        pqxx::params{reconciliationCursorEventId, observation.runKey,
            observation.campaignId.value(),
            observation.requestId.value(),
            observation.requestCanonicalText,
            ToText(observation.expectedRequestState),
            observation.expectedRequestVersion,
            ToText(observation.reason),
            observation.evidenceCanonicalText,
            observation.evidenceIdentityHash,
            observation.recommendedService,
            observation.recommendedAction,
            observation.diagnosticCode,
            observation.identity.canonicalText(),
            observation.identity.hash()}).one_row();
    const auto id =
        ReconciliationObservationId(row[0].as<long long>());
    transaction.exec(
        "INSERT INTO campaign_operations_control_audit_reference_event("
        "operational_campaign_id,operational_request_id,"
        "reconciliation_observation_id,cause_kind,actor_identity,"
        "capability,reason,prior_version,resulting_version,outcome,"
        "diagnostic_code) VALUES($1,$2,$3,'reconciliation_observed',"
        "'campaign_operations_reconciler',"
        "'campaign_operations_reconciler',$4,$5,$5,'recorded',$4);",
        pqxx::params{observation.campaignId.value(),
            observation.requestId.value(), id.value(),
            observation.diagnosticCode,
            observation.expectedRequestVersion});
    return {id, observation, row[1].as<std::string>()};
}

std::optional<PersistedReconciliationCursor> FindReconciliationCursor(
    pqxx::transaction_base& transaction, const std::string& runKey,
    long long priorTargetId)
{
    const auto rows = transaction.exec(
        "SELECT reconciliation_cursor_event_id,run_key,prior_target_id,"
        "last_target_id,requested_limit,selected_count "
        "FROM campaign_operations_reconciliation_cursor_event "
        "WHERE run_key=$1 AND prior_target_id=$2;",
        pqxx::params{runKey, priorTargetId});
    if (rows.empty()) return std::nullopt;
    const auto row = rows.one_row();
    return PersistedReconciliationCursor{row[0].as<long long>(),
        row[1].as<std::string>(), row[2].as<long long>(),
        row[3].as<long long>(), row[4].as<int>(), row[5].as<int>()};
}

std::vector<PersistedReconciliationObservation>
LoadReconciliationObservationsForCursor(
    pqxx::transaction_base& transaction,
    long long reconciliationCursorEventId)
{
    const auto rows = transaction.exec(
        "SELECT reconciliation_observation_id,run_key,"
        "operational_campaign_id,operational_request_id,"
        "request_identity_canonical,expected_request_state,"
        "expected_request_version,reason_code,evidence_identity_canonical,"
        "evidence_identity_hash,recommended_service,recommended_action,"
        "diagnostic_code,observation_identity_canonical,"
        "observation_identity_hash,observed_at::text "
        "FROM campaign_operations_reconciliation_observation "
        "WHERE reconciliation_cursor_event_id=$1 "
        "ORDER BY operational_request_id;",
        pqxx::params{reconciliationCursorEventId});
    std::vector<PersistedReconciliationObservation> result;
    result.reserve(rows.size());
    for (const auto& row : rows) result.push_back(MapObservation(row));
    return result;
}

PersistedReconciliationBatch PersistReconciliationBatch(
    pqxx::transaction_base& transaction,
    const std::string& runKey, long long priorTargetId,
    int requestedLimit,
    const std::vector<ReconciliationObservation>& observations,
    CampaignOperationsControlTestHook testHook)
{
    if (priorTargetId < 0 || requestedLimit <= 0 ||
        requestedLimit > kCampaignOperationsMaximumReconciliationBatchSize ||
        observations.size() > static_cast<std::size_t>(requestedLimit))
        throw std::invalid_argument(
            "campaign_operations_reconciliation_cursor_invalid");
    const auto cursorKey = runKey + ":" + std::to_string(priorTargetId);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,"
        "1179402835030007));", pqxx::params{cursorKey});
    if (const auto existing = FindReconciliationCursor(
            transaction, runKey, priorTargetId))
    {
        if (existing->requestedLimit != requestedLimit)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_reconciliation_cursor_conflict");
        auto persisted = LoadReconciliationObservationsForCursor(
            transaction, existing->reconciliationCursorEventId);
        if (static_cast<int>(persisted.size()) != existing->selectedCount)
            throw Error(ErrorCode::persistenceCorruption,
                "campaign_operations_reconciliation_cursor_incomplete");
        return {*existing, std::move(persisted), true};
    }

    std::vector<long long> campaignIds;
    std::vector<long long> requestIds;
    campaignIds.reserve(observations.size());
    requestIds.reserve(observations.size());
    long long lastTargetId = priorTargetId;
    for (const auto& observation : observations)
    {
        ValidateReconciliationObservation(observation);
        if (observation.runKey != runKey ||
            observation.requestId.value() <= priorTargetId)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_reconciliation_batch_membership_invalid");
        campaignIds.push_back(observation.campaignId.value());
        requestIds.push_back(observation.requestId.value());
        lastTargetId = std::max(
            lastTargetId, observation.requestId.value());
    }
    std::sort(campaignIds.begin(), campaignIds.end());
    campaignIds.erase(
        std::unique(campaignIds.begin(), campaignIds.end()),
        campaignIds.end());
    std::sort(requestIds.begin(), requestIds.end());
    if (std::adjacent_find(requestIds.begin(), requestIds.end()) !=
        requestIds.end())
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_reconciliation_batch_membership_invalid");

    for (const auto campaignId : campaignIds)
        transaction.exec(
            "SELECT lock_campaign_operations_campaign($1);",
            pqxx::params{campaignId});
    if (testHook)
        testHook(CampaignOperationsControlTestInjectionPoint::
                afterReconciliationCampaignLocksBeforeRequestLocks,
            transaction);
    for (const auto requestId : requestIds)
        transaction.exec(
            "SELECT lock_campaign_operations_request($1);",
            pqxx::params{requestId});

    const auto cursorRow = transaction.exec(
        "INSERT INTO campaign_operations_reconciliation_cursor_event("
        "run_key,prior_target_id,last_target_id,requested_limit,"
        "selected_count) VALUES($1,$2,$3,$4,$5) "
        "RETURNING reconciliation_cursor_event_id;",
        pqxx::params{runKey, priorTargetId, lastTargetId,
            requestedLimit, static_cast<int>(observations.size())})
        .one_row();
    PersistedReconciliationCursor cursor{
        cursorRow[0].as<long long>(), runKey, priorTargetId,
        lastTargetId, requestedLimit,
        static_cast<int>(observations.size())};
    std::vector<PersistedReconciliationObservation> persisted;
    persisted.reserve(observations.size());
    for (const auto& observation : observations)
        persisted.push_back(InsertReconciliationObservation(
            transaction, cursor.reconciliationCursorEventId,
            observation));
    return {std::move(cursor), std::move(persisted), false};
}

std::optional<PersistedReconciliationResolution>
FindReconciliationResolution(pqxx::transaction_base& transaction,
    ReconciliationObservationId observationId)
{
    const auto rows = transaction.exec(
        "SELECT reconciliation_resolution_id,"
        "reconciliation_observation_id,observation_identity_canonical,"
        "owning_service,owning_capability,transition_identity_canonical,"
        "transition_identity_hash,resolution_disposition,"
        "resolution_identity_canonical,resolution_identity_hash,"
        "created_at::text "
        "FROM campaign_operations_reconciliation_resolution "
        "WHERE reconciliation_observation_id=$1;",
        pqxx::params{observationId.value()});
    if (rows.empty()) return std::nullopt;
    return MapResolution(rows.one_row());
}

PersistedReconciliationResolution RecoverExpiredDispatchLeaseInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedReconciliationObservation& persistedObservation)
{
    if (const auto existing = FindReconciliationResolution(
            transaction, persistedObservation.observationId))
        return *existing;
    const auto& observation = persistedObservation.observation;
    if (observation.reason != ReconciliationReason::
            dispatchLeaseExpiredNoDownstreamEvidence ||
        observation.recommendedAction != "clear_stale_dispatch_lease")
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_recovery_observation_not_actionable");
    transaction.exec(
        "SELECT lock_campaign_operations_campaign($1);",
        pqxx::params{observation.campaignId.value()});
    if (const auto existing = FindReconciliationResolution(
            transaction, persistedObservation.observationId))
        return *existing;
    const auto candidate = LoadLockedCandidate(
        transaction, observation.requestId);
    transaction.exec(
        "SELECT lock_campaign_operations_reservation($1);",
        pqxx::params{candidate.reservationId.value()});
    transaction.exec(
        "SELECT lock_campaign_operations_request($1);",
        pqxx::params{candidate.requestId.value()});
    const auto locked = LoadLockedCandidate(
        transaction, observation.requestId);
    const bool exactActionableEvidence =
        locked.campaignId == observation.campaignId &&
        locked.requestCanonicalText == observation.requestCanonicalText &&
        locked.requestState == RequestState::dispatching &&
        locked.requestVersion == observation.expectedRequestVersion &&
        locked.reservationState == ReservationState::held &&
        !locked.reservationExpired && locked.leaseExpired &&
        locked.bindingCount == 0 &&
        locked.downstreamExecutionCount == 0 &&
        locked.dispatchOutcomeCount == 0 &&
        BuildReconciliationEvidenceCanonical(locked) ==
            observation.evidenceCanonicalText &&
        locked.leaseExpiresAt.has_value();
    if (!exactActionableEvidence)
    {
        const bool recoveredState =
            locked.campaignId == observation.campaignId &&
            locked.requestCanonicalText ==
                observation.requestCanonicalText &&
            locked.requestState == RequestState::ready &&
            locked.requestVersion ==
                observation.expectedRequestVersion + 1 &&
            locked.reservationState == ReservationState::held &&
            !locked.leaseExpiresAt &&
            locked.bindingCount == 0 &&
            locked.downstreamExecutionCount == 0 &&
            locked.dispatchOutcomeCount == 1;
        if (!recoveredState)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_recovery_evidence_changed");
        const auto accepted = transaction.exec(
                "SELECT DISTINCT outcome.outcome_identity_canonical,"
                "outcome.outcome_identity_hash "
                "FROM campaign_operations_dispatch_attempt attempt "
                "JOIN campaign_operations_dispatch_attempt_outcome outcome "
                "ON outcome.dispatch_attempt_id=attempt.dispatch_attempt_id "
                "JOIN campaign_operations_dispatch_audit_reference_event audit "
                "ON audit.dispatch_attempt_id=attempt.dispatch_attempt_id "
                "AND audit.dispatch_attempt_outcome_id="
                "outcome.dispatch_attempt_outcome_id "
                "JOIN campaign_operations_reconciliation_resolution resolution "
                "ON resolution.transition_identity_canonical="
                "outcome.outcome_identity_canonical "
                "AND resolution.transition_identity_hash="
                "outcome.outcome_identity_hash "
                "JOIN campaign_operations_reconciliation_observation equivalent "
                "ON equivalent.reconciliation_observation_id="
                "resolution.reconciliation_observation_id "
                "WHERE attempt.operational_request_id=$1 "
                "AND attempt.attempt_ordinal=("
                " SELECT max(latest.attempt_ordinal) "
                " FROM campaign_operations_dispatch_attempt latest "
                " WHERE latest.operational_request_id=$1) "
                "AND outcome.result_classification='rejected' "
                "AND outcome.downstream_evidence_classification="
                "'no_phase5_evidence' "
                "AND outcome.semantic_conflict_classification='none' "
                "AND outcome.uncertain_commit_recovery_classification="
                "'proven_no_commit' "
                "AND outcome.diagnostic_code="
                "'dispatch_lease_expired_no_downstream_evidence' "
                "AND outcome.expected_request_version=$2 "
                "AND outcome.resulting_request_version=$3 "
                "AND outcome.expected_reservation_version=$4 "
                "AND outcome.resulting_reservation_version=$4 "
                "AND audit.cause_kind='dispatch_lease_recovered' "
                "AND audit.capability='campaign_operations_recovery' "
                "AND audit.prior_version=$2 "
                "AND audit.resulting_version=$3 "
                "AND audit.outcome='recorded' "
                "AND equivalent.operational_campaign_id=$5 "
                "AND equivalent.operational_request_id=$1 "
                "AND equivalent.request_identity_canonical=$6 "
                "AND equivalent.expected_request_state='dispatching' "
                "AND equivalent.expected_request_version=$2 "
                "AND equivalent.reason_code="
                "'dispatch_lease_expired_no_downstream_evidence' "
                "AND equivalent.evidence_identity_canonical=$7 "
                "AND equivalent.evidence_identity_hash=$8 "
                "AND equivalent.recommended_service="
                "'campaign_operations_dispatch_recovery' "
                "AND equivalent.recommended_action="
                "'clear_stale_dispatch_lease' "
                "AND equivalent.diagnostic_code="
                "'dispatch_lease_expired_no_downstream_evidence' "
                "AND resolution.owning_service="
                "'campaign_operations_dispatch_recovery' "
                "AND resolution.owning_capability="
                "'campaign_operations_recovery' "
                "AND resolution.resolution_disposition="
                "'request_returned_ready';",
                pqxx::params{observation.requestId.value(),
                    observation.expectedRequestVersion,
                    observation.expectedRequestVersion + 1,
                    locked.reservationVersion,
                    observation.campaignId.value(),
                    observation.requestCanonicalText,
                    observation.evidenceCanonicalText,
                    observation.evidenceIdentityHash});
        if (accepted.size() != 1)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_recovery_evidence_changed");
        return AppendReconciliationResolution(transaction,
            persistedObservation,
            "campaign_operations_dispatch_recovery",
            kCampaignOperationsRecoveryRole,
            accepted.one_row()[0].as<std::string>(),
            accepted.one_row()[1].as<std::string>(),
            "already_resolved", observation.expectedRequestVersion,
            observation.expectedRequestVersion + 1);
    }
    transaction.exec(
        "SELECT transition_campaign_operations_request_ready_recovered("
        "$1,$2);", pqxx::params{locked.requestId.value(),
            locked.requestVersion});
    auto outcome = PersistRecoveryOutcome(transaction, locked,
        "dispatch_lease_expired_no_downstream_evidence",
        locked.requestVersion + 1, locked.reservationVersion,
        ActorIdentity(kCampaignOperationsRecoveryRole),
        kCampaignOperationsRecoveryRole);
    return AppendReconciliationResolution(transaction,
        persistedObservation, "campaign_operations_dispatch_recovery",
        kCampaignOperationsRecoveryRole,
        outcome.identity.canonicalText(), outcome.identity.hash(),
        "request_returned_ready", locked.requestVersion,
        locked.requestVersion + 1);
}

CampaignControlStatus LoadCampaignControlStatusProjection(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId,
    CampaignOperationsControlTestHook testHook)
{
    const auto exists = transaction.exec(
        "SELECT 1 FROM campaign_operations_campaign "
        "WHERE operational_campaign_id=$1;",
        pqxx::params{campaignId.value()});
    if (exists.empty())
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_campaign_not_found");
    CampaignControlStatus result{
        campaignId, false, 0, std::nullopt, false, false,
        std::nullopt, std::nullopt, 0};
    if (const auto control = FindCampaignControlHead(
            transaction, campaignId))
    {
        result.paused =
            control->event.eventKind == ControlEventKind::pause;
        result.controlVersion = control->event.controlVersion;
        result.controlEventId.emplace(control->controlEventId);
    }
    if (testHook)
        testHook(CampaignOperationsControlTestInjectionPoint::
                afterStatusControlReadBeforeCancellationRead,
            transaction);
    const auto cancellation = transaction.exec(
        "SELECT request.cancellation_request_id,"
        "settlement.disposition "
        "FROM campaign_operations_cancellation_request request "
        "LEFT JOIN campaign_operations_cancellation_settlement settlement "
        "ON settlement.cancellation_request_id="
        "request.cancellation_request_id "
        "WHERE request.operational_campaign_id=$1 "
        "ORDER BY request.cancellation_request_id LIMIT 1;",
        pqxx::params{campaignId.value()});
    if (!cancellation.empty())
    {
        result.cancellationRequested = true;
        result.cancellationRequestId.emplace(
            cancellation.one_row()[0].as<long long>());
        if (!cancellation.one_row()[1].is_null())
        {
            result.cancellationSettled = true;
            result.cancellationDisposition =
                CancellationSettlementDispositionFromText(
                    cancellation.one_row()[1].as<std::string>());
        }
    }
    result.unresolvedObservationCount = transaction.exec(
        "SELECT count(*) FROM "
        "campaign_operations_reconciliation_observation observation "
        "LEFT JOIN campaign_operations_reconciliation_resolution resolution "
        "ON resolution.reconciliation_observation_id="
        "observation.reconciliation_observation_id "
        "WHERE observation.operational_campaign_id=$1 "
        "AND resolution.reconciliation_resolution_id IS NULL;",
        pqxx::params{campaignId.value()}).one_row()[0].as<int>();
    return result;
}

} // namespace EA::CampaignOperations
