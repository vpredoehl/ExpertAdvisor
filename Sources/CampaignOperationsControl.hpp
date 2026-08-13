#pragma once

#include "CampaignOperations.hpp"

#include <optional>
#include <string>

namespace EA::CampaignOperations
{

inline constexpr int kCampaignOperationsControlContractVersion = 1;
inline constexpr int kCampaignOperationsMaximumReconciliationBatchSize = 1000;
inline constexpr char kCampaignOperationsControllerRole[] =
    "campaign_operations_controller";
inline constexpr char kCampaignOperationsCancellationRole[] =
    "campaign_operations_cancellation_coordinator";
inline constexpr char kCampaignOperationsReconcilerRole[] =
    "campaign_operations_reconciler";
inline constexpr char kCampaignOperationsRecoveryRole[] =
    "campaign_operations_recovery";
inline constexpr char kCampaignOperationsReaderRole[] =
    "campaign_operations_reader";
inline constexpr char kExperimentLifecycleCancellationRole[] =
    "experiment_lifecycle_cancellation";

enum class ControlReplayDisposition
{
    recorded,
    existingIdentical
};

enum class CancellationProgress
{
    settled,
    waitingForLeaseExpiry,
    lifecycleCoordinationRequired,
    reconciliationRequired
};

enum class LifecycleCancellationDisposition
{
    accepted,
    alreadyTerminal,
    runningNotSupported
};

std::string ToText(ControlReplayDisposition value);
std::string ToText(CancellationProgress value);
std::string ToText(LifecycleCancellationDisposition value);

struct CampaignControlEvent final
{
    CanonicalIdentity identity;
    OperationalCampaignId campaignId;
    std::string campaignCanonicalText;
    std::optional<ControlEventId> previousEventId;
    std::optional<std::string> previousEventCanonicalText;
    int controlVersion;
    ControlEventKind eventKind;
    ActorIdentity actor;
    Reason reason;

    bool operator==(const CampaignControlEvent&) const = default;
};

CampaignControlEvent BuildCampaignControlEvent(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::optional<ControlEventId> previousEventId,
    std::optional<std::string> previousEventCanonicalText,
    int controlVersion, ControlEventKind eventKind, ActorIdentity actor,
    Reason reason);
void ValidateCampaignControlEvent(const CampaignControlEvent& event);

struct CampaignCancellationRequest final
{
    CanonicalIdentity identity;
    OperationalCampaignId campaignId;
    std::string campaignCanonicalText;
    std::optional<OperationalRequestId> requestId;
    std::optional<std::string> requestCanonicalText;
    std::optional<RequestState> expectedRequestState;
    std::optional<int> expectedRequestVersion;
    std::string operationKey;
    ActorIdentity actor;
    Reason reason;

    bool operator==(const CampaignCancellationRequest&) const = default;
};

CampaignCancellationRequest BuildCampaignCancellationRequest(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::optional<OperationalRequestId> requestId,
    std::optional<std::string> requestCanonicalText,
    std::optional<RequestState> expectedRequestState,
    std::optional<int> expectedRequestVersion, std::string operationKey,
    ActorIdentity actor, Reason reason);
void ValidateCampaignCancellationRequest(
    const CampaignCancellationRequest& request);

struct CampaignCancellationSettlement final
{
    CanonicalIdentity identity;
    CancellationRequestId cancellationRequestId;
    std::string cancellationRequestCanonicalText;
    CancellationSettlementDisposition disposition;
    std::optional<ReservationEventId> reservationEventId;
    std::optional<std::string> reservationEventCanonicalText;
    std::optional<int> resultingRequestVersion;
    std::optional<std::string> lifecycleEvidenceCanonicalText;
    std::optional<std::string> lifecycleEvidenceIdentityHash;

    bool operator==(const CampaignCancellationSettlement&) const = default;
};

CampaignCancellationSettlement BuildCampaignCancellationSettlement(
    CancellationRequestId cancellationRequestId,
    std::string cancellationRequestCanonicalText,
    CancellationSettlementDisposition disposition,
    std::optional<ReservationEventId> reservationEventId,
    std::optional<std::string> reservationEventCanonicalText,
    std::optional<int> resultingRequestVersion,
    std::optional<std::string> lifecycleEvidenceCanonicalText,
    std::optional<std::string> lifecycleEvidenceIdentityHash);
void ValidateCampaignCancellationSettlement(
    const CampaignCancellationSettlement& settlement);

struct ReconciliationObservation final
{
    CanonicalIdentity identity;
    std::string runKey;
    OperationalCampaignId campaignId;
    OperationalRequestId requestId;
    std::string requestCanonicalText;
    RequestState expectedRequestState;
    int expectedRequestVersion;
    ReconciliationReason reason;
    std::string evidenceCanonicalText;
    std::string evidenceIdentityHash;
    std::string recommendedService;
    std::string recommendedAction;
    std::string diagnosticCode;

    bool operator==(const ReconciliationObservation&) const = default;
};

ReconciliationObservation BuildReconciliationObservation(
    std::string runKey, OperationalCampaignId campaignId,
    OperationalRequestId requestId, std::string requestCanonicalText,
    RequestState expectedRequestState, int expectedRequestVersion,
    ReconciliationReason reason, std::string evidenceCanonicalText,
    std::string recommendedService, std::string recommendedAction,
    std::string diagnosticCode);
void ValidateReconciliationObservation(
    const ReconciliationObservation& observation);

struct ReconciliationResolution final
{
    CanonicalIdentity identity;
    ReconciliationObservationId observationId;
    std::string observationCanonicalText;
    std::string owningService;
    std::string owningCapability;
    std::string transitionCanonicalText;
    std::string transitionIdentityHash;
    std::string disposition;

    bool operator==(const ReconciliationResolution&) const = default;
};

ReconciliationResolution BuildReconciliationResolution(
    ReconciliationObservationId observationId,
    std::string observationCanonicalText, std::string owningService,
    std::string owningCapability, std::string transitionCanonicalText,
    std::string transitionIdentityHash, std::string disposition);
void ValidateReconciliationResolution(
    const ReconciliationResolution& resolution);

} // namespace EA::CampaignOperations
