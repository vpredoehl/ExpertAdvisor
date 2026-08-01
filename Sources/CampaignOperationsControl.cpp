#include "CampaignOperationsControl.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

std::string Framed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string OptionalFramed(const std::optional<std::string>& value)
{
    return value ? Framed(*value) : "none";
}

std::string OptionalId(const std::optional<OperationalRequestId>& value)
{
    return value ? std::to_string(value->value()) : "none";
}

std::string OptionalControlId(const std::optional<ControlEventId>& value)
{
    return value ? std::to_string(value->value()) : "none";
}

std::string OptionalReservationEventId(
    const std::optional<ReservationEventId>& value)
{
    return value ? std::to_string(value->value()) : "none";
}

std::string OptionalVersion(const std::optional<int>& value)
{
    return value ? std::to_string(*value) : "none";
}

std::string OptionalRequestState(
    const std::optional<RequestState>& value)
{
    return value ? ToText(*value) : "none";
}

void RequireCanonical(const std::string& value, ErrorCode code,
    const char* diagnostic)
{
    if (value.empty() ||
        value.size() > kCampaignOperationsCanonicalMaximumBytes)
        throw Error(code, diagnostic);
}

void RequireToken(const std::string& value, ErrorCode code,
    const char* diagnostic)
{
    if (value.empty() || value.size() > 128U ||
        !std::all_of(value.begin(), value.end(), [](unsigned char c)
        {
            return (c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                c == '.' || c == ':' || c == '/';
        }))
        throw Error(code, diagnostic);
}

CanonicalIdentity Identity(const std::string& canonical)
{
    return CanonicalIdentity::Create(
        kCampaignOperationsControlContractVersion, canonical);
}

} // namespace

std::string ToText(ControlReplayDisposition value)
{
    switch (value)
    {
        case ControlReplayDisposition::recorded: return "recorded";
        case ControlReplayDisposition::existingIdentical:
            return "existing_identical";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_control_replay_invalid");
}

std::string ToText(CancellationProgress value)
{
    switch (value)
    {
        case CancellationProgress::settled: return "settled";
        case CancellationProgress::waitingForLeaseExpiry:
            return "waiting_for_lease_expiry";
        case CancellationProgress::lifecycleCoordinationRequired:
            return "lifecycle_coordination_required";
        case CancellationProgress::reconciliationRequired:
            return "reconciliation_required";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_cancellation_progress_invalid");
}

std::string ToText(LifecycleCancellationDisposition value)
{
    switch (value)
    {
        case LifecycleCancellationDisposition::accepted: return "accepted";
        case LifecycleCancellationDisposition::alreadyTerminal:
            return "already_terminal";
        case LifecycleCancellationDisposition::runningNotSupported:
            return "running_not_supported";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_lifecycle_cancellation_invalid");
}

CampaignControlEvent BuildCampaignControlEvent(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::optional<ControlEventId> previousEventId,
    std::optional<std::string> previousEventCanonicalText,
    int controlVersion, ControlEventKind eventKind, ActorIdentity actor,
    Reason reason)
{
    RequireCanonical(campaignCanonicalText, ErrorCode::invalidControlEvent,
        "campaign_operations_control_campaign_canonical_invalid");
    if (controlVersion <= 0 ||
        (controlVersion == 1) != !previousEventId ||
        previousEventId.has_value() !=
            previousEventCanonicalText.has_value() ||
        (controlVersion == 1 &&
            eventKind != ControlEventKind::pause))
        throw Error(ErrorCode::invalidControlEvent,
            "campaign_operations_control_chain_invalid");
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_control_event_v1"
        << ";campaign_id=" << campaignId.value()
        << ";campaign_canonical=" << Framed(campaignCanonicalText)
        << ";control_version=" << controlVersion
        << ";previous_event_id=" << OptionalControlId(previousEventId)
        << ";previous_event_canonical="
        << OptionalFramed(previousEventCanonicalText)
        << ";event_kind=" << ToText(eventKind)
        << ";expected_paused="
        << (eventKind == ControlEventKind::resume ? "true" : "false")
        << ";actor=" << Framed(actor.value())
        << ";capability=" << kCampaignOperationsControllerRole
        << ";reason=" << Framed(reason.value());
    return {Identity(canonical.str()), campaignId,
        std::move(campaignCanonicalText), std::move(previousEventId),
        std::move(previousEventCanonicalText), controlVersion, eventKind,
        std::move(actor), std::move(reason)};
}

void ValidateCampaignControlEvent(const CampaignControlEvent& event)
{
    if (BuildCampaignControlEvent(event.campaignId,
            event.campaignCanonicalText, event.previousEventId,
            event.previousEventCanonicalText, event.controlVersion,
            event.eventKind, event.actor, event.reason) != event)
        throw Error(ErrorCode::invalidControlEvent,
            "campaign_operations_control_identity_mismatch");
}

CampaignCancellationRequest BuildCampaignCancellationRequest(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::optional<OperationalRequestId> requestId,
    std::optional<std::string> requestCanonicalText,
    std::optional<RequestState> expectedRequestState,
    std::optional<int> expectedRequestVersion, std::string operationKey,
    ActorIdentity actor, Reason reason)
{
    RequireCanonical(campaignCanonicalText,
        ErrorCode::invalidCancellationRequest,
        "campaign_operations_cancellation_campaign_canonical_invalid");
    RequireToken(operationKey, ErrorCode::invalidCancellationRequest,
        "campaign_operations_cancellation_operation_key_invalid");
    const bool targeted = requestId.has_value();
    if (targeted != requestCanonicalText.has_value() ||
        targeted != expectedRequestState.has_value() ||
        targeted != expectedRequestVersion.has_value() ||
        (expectedRequestVersion && *expectedRequestVersion <= 0))
        throw Error(ErrorCode::invalidCancellationRequest,
            "campaign_operations_cancellation_target_shape_invalid");
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_cancellation_request_v1"
        << ";campaign_id=" << campaignId.value()
        << ";campaign_canonical=" << Framed(campaignCanonicalText)
        << ";scope=complete_materialization"
        << ";request_id=" << OptionalId(requestId)
        << ";request_canonical=" << OptionalFramed(requestCanonicalText)
        << ";expected_request_state="
        << OptionalRequestState(expectedRequestState)
        << ";expected_request_version="
        << OptionalVersion(expectedRequestVersion)
        << ";operation_key=" << Framed(operationKey)
        << ";actor=" << Framed(actor.value())
        << ";capability=" << kCampaignOperationsCancellationRole
        << ";reason=" << Framed(reason.value());
    return {Identity(canonical.str()), campaignId,
        std::move(campaignCanonicalText), std::move(requestId),
        std::move(requestCanonicalText), std::move(expectedRequestState),
        std::move(expectedRequestVersion), std::move(operationKey),
        std::move(actor), std::move(reason)};
}

void ValidateCampaignCancellationRequest(
    const CampaignCancellationRequest& request)
{
    if (BuildCampaignCancellationRequest(request.campaignId,
            request.campaignCanonicalText, request.requestId,
            request.requestCanonicalText, request.expectedRequestState,
            request.expectedRequestVersion, request.operationKey,
            request.actor, request.reason) != request)
        throw Error(ErrorCode::invalidCancellationRequest,
            "campaign_operations_cancellation_request_identity_mismatch");
}

CampaignCancellationSettlement BuildCampaignCancellationSettlement(
    CancellationRequestId cancellationRequestId,
    std::string cancellationRequestCanonicalText,
    CancellationSettlementDisposition disposition,
    std::optional<ReservationEventId> reservationEventId,
    std::optional<std::string> reservationEventCanonicalText,
    std::optional<int> resultingRequestVersion,
    std::optional<std::string> lifecycleEvidenceCanonicalText,
    std::optional<std::string> lifecycleEvidenceIdentityHash)
{
    RequireCanonical(cancellationRequestCanonicalText,
        ErrorCode::invalidCancellationSettlement,
        "campaign_operations_cancellation_settlement_request_invalid");
    if (reservationEventId.has_value() !=
            reservationEventCanonicalText.has_value() ||
        lifecycleEvidenceCanonicalText.has_value() !=
            lifecycleEvidenceIdentityHash.has_value() ||
        (resultingRequestVersion && *resultingRequestVersion <= 0))
        throw Error(ErrorCode::invalidCancellationSettlement,
            "campaign_operations_cancellation_settlement_shape_invalid");
    const bool unbound =
        disposition == CancellationSettlementDisposition::unboundCancelled;
    if (unbound != reservationEventId.has_value() ||
        (unbound && !resultingRequestVersion) ||
        (unbound && lifecycleEvidenceCanonicalText) ||
        (!unbound && reservationEventId) ||
        (!unbound && !lifecycleEvidenceCanonicalText))
        throw Error(ErrorCode::invalidCancellationSettlement,
            "campaign_operations_cancellation_settlement_evidence_invalid");
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_cancellation_settlement_v1"
        << ";cancellation_request_id=" << cancellationRequestId.value()
        << ";cancellation_request_canonical="
        << Framed(cancellationRequestCanonicalText)
        << ";disposition=" << ToText(disposition)
        << ";reservation_event_id="
        << OptionalReservationEventId(reservationEventId)
        << ";reservation_event_canonical="
        << OptionalFramed(reservationEventCanonicalText)
        << ";resulting_request_version="
        << OptionalVersion(resultingRequestVersion)
        << ";lifecycle_evidence_canonical="
        << OptionalFramed(lifecycleEvidenceCanonicalText)
        << ";lifecycle_evidence_hash="
        << OptionalFramed(lifecycleEvidenceIdentityHash);
    return {Identity(canonical.str()), cancellationRequestId,
        std::move(cancellationRequestCanonicalText), disposition,
        std::move(reservationEventId),
        std::move(reservationEventCanonicalText),
        std::move(resultingRequestVersion),
        std::move(lifecycleEvidenceCanonicalText),
        std::move(lifecycleEvidenceIdentityHash)};
}

void ValidateCampaignCancellationSettlement(
    const CampaignCancellationSettlement& settlement)
{
    if (BuildCampaignCancellationSettlement(
            settlement.cancellationRequestId,
            settlement.cancellationRequestCanonicalText,
            settlement.disposition, settlement.reservationEventId,
            settlement.reservationEventCanonicalText,
            settlement.resultingRequestVersion,
            settlement.lifecycleEvidenceCanonicalText,
            settlement.lifecycleEvidenceIdentityHash) != settlement)
        throw Error(ErrorCode::invalidCancellationSettlement,
            "campaign_operations_cancellation_settlement_identity_mismatch");
}

ReconciliationObservation BuildReconciliationObservation(
    std::string runKey, OperationalCampaignId campaignId,
    OperationalRequestId requestId, std::string requestCanonicalText,
    RequestState expectedRequestState, int expectedRequestVersion,
    ReconciliationReason reason, std::string evidenceCanonicalText,
    std::string recommendedService, std::string recommendedAction,
    std::string diagnosticCode)
{
    RequireToken(runKey, ErrorCode::invalidReconciliationObservation,
        "campaign_operations_reconciliation_run_key_invalid");
    RequireCanonical(requestCanonicalText,
        ErrorCode::invalidReconciliationObservation,
        "campaign_operations_reconciliation_request_canonical_invalid");
    RequireCanonical(evidenceCanonicalText,
        ErrorCode::invalidReconciliationObservation,
        "campaign_operations_reconciliation_evidence_invalid");
    RequireToken(recommendedService,
        ErrorCode::invalidReconciliationObservation,
        "campaign_operations_reconciliation_service_invalid");
    RequireToken(recommendedAction,
        ErrorCode::invalidReconciliationObservation,
        "campaign_operations_reconciliation_action_invalid");
    if (diagnosticCode.empty() || diagnosticCode.size() > 128U ||
        !std::all_of(diagnosticCode.begin(), diagnosticCode.end(),
            [](unsigned char c)
            {
                return (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '_';
            }))
        throw Error(ErrorCode::invalidReconciliationObservation,
            "campaign_operations_reconciliation_diagnostic_invalid");
    if (expectedRequestVersion <= 0)
        throw Error(ErrorCode::invalidReconciliationObservation,
            "campaign_operations_reconciliation_version_invalid");
    const auto evidence = Identity(evidenceCanonicalText);
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_reconciliation_observation_v1"
        << ";run_key=" << Framed(runKey)
        << ";campaign_id=" << campaignId.value()
        << ";request_id=" << requestId.value()
        << ";request_canonical=" << Framed(requestCanonicalText)
        << ";expected_request_state=" << ToText(expectedRequestState)
        << ";expected_request_version=" << expectedRequestVersion
        << ";reason=" << ToText(reason)
        << ";evidence_canonical=" << Framed(evidenceCanonicalText)
        << ";evidence_hash=" << Framed(evidence.hash())
        << ";recommended_service=" << recommendedService
        << ";recommended_action=" << recommendedAction
        << ";diagnostic_code=" << diagnosticCode;
    return {Identity(canonical.str()), std::move(runKey), campaignId,
        requestId, std::move(requestCanonicalText), expectedRequestState,
        expectedRequestVersion, reason, std::move(evidenceCanonicalText),
        evidence.hash(), std::move(recommendedService),
        std::move(recommendedAction), std::move(diagnosticCode)};
}

void ValidateReconciliationObservation(
    const ReconciliationObservation& observation)
{
    if (BuildReconciliationObservation(observation.runKey,
            observation.campaignId, observation.requestId,
            observation.requestCanonicalText,
            observation.expectedRequestState,
            observation.expectedRequestVersion, observation.reason,
            observation.evidenceCanonicalText,
            observation.recommendedService,
            observation.recommendedAction,
            observation.diagnosticCode) != observation)
        throw Error(ErrorCode::invalidReconciliationObservation,
            "campaign_operations_reconciliation_observation_identity_mismatch");
}

ReconciliationResolution BuildReconciliationResolution(
    ReconciliationObservationId observationId,
    std::string observationCanonicalText, std::string owningService,
    std::string owningCapability, std::string transitionCanonicalText,
    std::string transitionIdentityHash, std::string disposition)
{
    RequireCanonical(observationCanonicalText,
        ErrorCode::invalidReconciliationResolution,
        "campaign_operations_resolution_observation_invalid");
    RequireCanonical(transitionCanonicalText,
        ErrorCode::invalidReconciliationResolution,
        "campaign_operations_resolution_transition_invalid");
    RequireToken(owningService, ErrorCode::invalidReconciliationResolution,
        "campaign_operations_resolution_service_invalid");
    RequireToken(owningCapability,
        ErrorCode::invalidReconciliationResolution,
        "campaign_operations_resolution_capability_invalid");
    RequireToken(disposition, ErrorCode::invalidReconciliationResolution,
        "campaign_operations_resolution_disposition_invalid");
    if (owningCapability != kCampaignOperationsRecoveryRole &&
        owningCapability != kCampaignOperationsCancellationRole)
        throw Error(ErrorCode::invalidReconciliationResolution,
            "campaign_operations_reconciliation_resolution_owner_invalid");
    if (disposition != "request_returned_ready" &&
        disposition != "cancellation_settled" &&
        disposition != "already_resolved")
        throw Error(ErrorCode::invalidReconciliationResolution,
            "campaign_operations_resolution_disposition_invalid");
    if (CanonicalIdentity::Create(1, transitionCanonicalText).hash() !=
            transitionIdentityHash)
        throw Error(ErrorCode::invalidReconciliationResolution,
            "campaign_operations_resolution_transition_hash_invalid");
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_reconciliation_resolution_v1"
        << ";observation_id=" << observationId.value()
        << ";observation_canonical=" << Framed(observationCanonicalText)
        << ";owning_service=" << owningService
        << ";owning_capability=" << owningCapability
        << ";transition_canonical=" << Framed(transitionCanonicalText)
        << ";transition_hash=" << Framed(transitionIdentityHash)
        << ";disposition=" << disposition;
    return {Identity(canonical.str()), observationId,
        std::move(observationCanonicalText), std::move(owningService),
        std::move(owningCapability), std::move(transitionCanonicalText),
        std::move(transitionIdentityHash), std::move(disposition)};
}

void ValidateReconciliationResolution(
    const ReconciliationResolution& resolution)
{
    if (BuildReconciliationResolution(resolution.observationId,
            resolution.observationCanonicalText, resolution.owningService,
            resolution.owningCapability,
            resolution.transitionCanonicalText,
            resolution.transitionIdentityHash,
            resolution.disposition) != resolution)
        throw Error(ErrorCode::invalidReconciliationResolution,
            "campaign_operations_reconciliation_resolution_identity_mismatch");
}

} // namespace EA::CampaignOperations
