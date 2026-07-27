#pragma once

#include "CampaignOperationsControl.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::CampaignOperations
{

struct PersistedCampaignControlEvent final
{
    ControlEventId controlEventId;
    CampaignControlEvent event;
    std::string createdAt;
};

struct PersistedCampaignCancellationRequest final
{
    CancellationRequestId cancellationRequestId;
    CampaignCancellationRequest request;
    std::string createdAt;
};

struct PersistedCampaignCancellationSettlement final
{
    CancellationSettlementId cancellationSettlementId;
    CampaignCancellationSettlement settlement;
    std::string createdAt;
};

struct LifecycleCancellationEvidence final
{
    long long lifecycleCancellationEventId = 0;
    CancellationRequestId cancellationRequestId;
    DownstreamControlOwnerId controlOwnerId;
    long long experimentId = 0;
    std::string observedStatus;
    std::string observedPhase;
    std::string resultingStatus;
    LifecycleCancellationDisposition disposition;
    CanonicalIdentity identity;
};

struct PersistedReconciliationObservation final
{
    ReconciliationObservationId observationId;
    ReconciliationObservation observation;
    std::string observedAt;
};

struct PersistedReconciliationResolution final
{
    ReconciliationResolutionId resolutionId;
    ReconciliationResolution resolution;
    std::string createdAt;
};

struct PersistedReconciliationCursor final
{
    long long reconciliationCursorEventId = 0;
    std::string runKey;
    long long priorTargetId = 0;
    long long lastTargetId = 0;
    int requestedLimit = 0;
    int selectedCount = 0;
};

enum class CampaignOperationsControlTestInjectionPoint
{
    afterCancellationIntentInsertion,
    afterReconciliationCampaignLocksBeforeRequestLocks,
    beforeReconciliationBatchCommit,
    afterReconciliationBatchCommitBeforeResponse,
    beforeReconciliationRecoveryCommit,
    afterReconciliationRecoveryCommitBeforeResponse,
    afterStatusControlReadBeforeCancellationRead
};

// Verification hooks run only when explicitly supplied by an isolated test.
// Production callers use the default-empty hook.
using CampaignOperationsControlTestHook = std::function<void(
    CampaignOperationsControlTestInjectionPoint,
    pqxx::transaction_base&)>;

struct PersistedReconciliationBatch final
{
    PersistedReconciliationCursor cursor;
    std::vector<PersistedReconciliationObservation> observations;
    bool replay = false;
};

struct ReconciliationCandidate final
{
    OperationalCampaignId campaignId;
    OperationalRequestId requestId;
    ReservationId reservationId;
    std::string requestCanonicalText;
    RequestState requestState;
    int requestVersion = 0;
    ReservationState reservationState;
    int reservationVersion = 0;
    std::optional<std::string> reservationExpiresAt;
    bool reservationExpired = false;
    std::optional<std::string> leaseExpiresAt;
    bool leaseExpired = false;
    int bindingCount = 0;
    int downstreamExecutionCount = 0;
    int dispatchOutcomeCount = 0;
    std::optional<CancellationRequestId> cancellationRequestId;
    std::optional<std::string> cancellationCanonicalText;
};

struct CampaignControlStatus final
{
    OperationalCampaignId campaignId;
    bool paused = false;
    int controlVersion = 0;
    std::optional<ControlEventId> controlEventId;
    bool cancellationRequested = false;
    bool cancellationSettled = false;
    std::optional<CancellationRequestId> cancellationRequestId;
    std::optional<CancellationSettlementDisposition>
        cancellationDisposition;
    int unresolvedObservationCount = 0;
};

bool ControlSchemaExists(pqxx::transaction_base& transaction);

std::optional<PersistedCampaignControlEvent> FindCampaignControlHead(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId);
PersistedCampaignControlEvent PersistCampaignControlEvent(
    pqxx::transaction_base& transaction,
    const CampaignControlEvent& event);

std::optional<PersistedCampaignCancellationRequest>
FindCampaignCancellationRequestByOperation(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId, const std::string& operationKey);
std::optional<PersistedCampaignCancellationRequest>
FindCampaignCancellationRequestByTarget(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId,
    std::optional<OperationalRequestId> requestId);
PersistedCampaignCancellationRequest PersistCampaignCancellationRequest(
    pqxx::transaction_base& transaction,
    const CampaignCancellationRequest& request);
std::optional<PersistedCampaignCancellationSettlement>
FindCampaignCancellationSettlement(
    pqxx::transaction_base& transaction,
    CancellationRequestId cancellationRequestId);

PersistedCampaignCancellationSettlement
SettleUnboundCancellationInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedCampaignCancellationRequest& cancellation,
    const ReconciliationCandidate& candidate,
    std::optional<ReconciliationObservationId> observationId);
void LockCancellationDomains(pqxx::transaction_base& transaction,
    const ReconciliationCandidate& candidate);

std::vector<std::pair<DownstreamControlOwnerId, long long>>
LoadDownstreamControlOwners(pqxx::transaction_base& transaction,
    OperationalRequestId requestId);
LifecycleCancellationEvidence ApplyLifecycleCancellationInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedCampaignCancellationRequest& cancellation,
    DownstreamControlOwnerId controlOwnerId, long long experimentId);
std::vector<LifecycleCancellationEvidence>
LoadLifecycleCancellationEvidence(pqxx::transaction_base& transaction,
    CancellationRequestId cancellationRequestId);
PersistedCampaignCancellationSettlement
SettleBoundCancellationInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedCampaignCancellationRequest& cancellation,
    const std::vector<LifecycleCancellationEvidence>& evidence);

std::vector<ReconciliationCandidate> SelectReconciliationCandidates(
    pqxx::transaction_base& transaction, long long afterRequestId,
    int limit);
ReconciliationCandidate LoadReconciliationCandidate(
    pqxx::transaction_base& transaction,
    OperationalRequestId requestId);
std::string BuildReconciliationEvidenceCanonical(
    const ReconciliationCandidate& candidate);
std::optional<PersistedReconciliationCursor> FindReconciliationCursor(
    pqxx::transaction_base& transaction, const std::string& runKey,
    long long priorTargetId);
std::vector<PersistedReconciliationObservation>
LoadReconciliationObservationsForCursor(
    pqxx::transaction_base& transaction,
    long long reconciliationCursorEventId);
PersistedReconciliationBatch PersistReconciliationBatch(
    pqxx::transaction_base& transaction,
    const std::string& runKey, long long priorTargetId,
    int requestedLimit,
    const std::vector<ReconciliationObservation>& observations,
    CampaignOperationsControlTestHook testHook = {});
std::optional<PersistedReconciliationResolution>
FindReconciliationResolution(pqxx::transaction_base& transaction,
    ReconciliationObservationId observationId);
PersistedReconciliationResolution RecoverExpiredDispatchLeaseInTransaction(
    pqxx::transaction_base& transaction,
    const PersistedReconciliationObservation& observation);

CampaignControlStatus LoadCampaignControlStatusProjection(
    pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId,
    CampaignOperationsControlTestHook testHook = {});

} // namespace EA::CampaignOperations
