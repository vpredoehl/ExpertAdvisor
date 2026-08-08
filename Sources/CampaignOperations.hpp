#pragma once

#include <compare>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace EA::CampaignOperations
{

inline constexpr int kCampaignOperationsContractVersion = 1;
inline constexpr int kCampaignOperationsActionContractVersion = 1;
inline constexpr int kCampaignOperationsScopeContractVersion = 1;
inline constexpr char kCampaignOperationsAuthorizationRole[] =
    "campaign_operations_authorizer";
inline constexpr std::size_t kCampaignOperationsActorMaximumBytes = 128U;
inline constexpr std::size_t kCampaignOperationsReasonMaximumBytes = 4096U;
inline constexpr std::size_t kCampaignOperationsCanonicalMaximumBytes =
    128U * 1024U * 1024U;

enum class ErrorCode
{
    invalidIdentifier,
    invalidCanonicalText,
    invalidCanonicalHash,
    unsupportedContractVersion,
    invalidActorIdentity,
    invalidReason,
    invalidTimestamp,
    invalidMaterialization,
    invalidProvenance,
    invalidAuthorizationEvent,
    invalidBudgetLedgerEntry,
    invalidLogicalOperation,
    invalidReservation,
    invalidReservationEvent,
    invalidOperationalRequest,
    invalidControlEvent,
    invalidCancellationRequest,
    invalidCancellationSettlement,
    invalidReconciliationObservation,
    invalidReconciliationResolution,
    invalidBudgetAccounting,
    invalidCompletionEvidence,
    invalidEnumText,
    authorizationDenied,
    budgetDenied,
    persistenceConflict,
    persistenceCorruption
};

class Error final : public std::runtime_error
{
public:
    Error(ErrorCode code, std::string reason);

    ErrorCode code() const noexcept;

private:
    const ErrorCode code_;
};

template <typename Tag>
class ImmutableIdentifier
{
public:
    explicit ImmutableIdentifier(long long value)
        : value_(value)
    {
        if (value <= 0)
            throw Error(ErrorCode::invalidIdentifier,
                "campaign_operations_identifier_invalid");
    }

    long long value() const noexcept { return value_; }
    bool operator==(const ImmutableIdentifier&) const = default;
    auto operator<=>(const ImmutableIdentifier&) const = default;

private:
    const long long value_;
};

struct OperationalCampaignIdTag;
struct GovernanceProvenanceEventIdTag;
struct AuthorizationEventIdTag;
struct BudgetLedgerEntryIdTag;
struct ReservationIdTag;
struct ReservationEventIdTag;
struct OperationalRequestIdTag;
struct DispatchAttemptIdTag;
struct DispatchAttemptOutcomeIdTag;
struct RequestBindingIdTag;
struct DownstreamControlOwnerIdTag;
struct ControlEventIdTag;
struct CancellationRequestIdTag;
struct CancellationSettlementIdTag;
struct ReconciliationObservationIdTag;
struct ReconciliationResolutionIdTag;
struct CompletionEventIdTag;
struct AuditReferenceEventIdTag;
struct ProductionEnablementEventIdTag;
struct RequestProductionAdmissionIdTag;

using OperationalCampaignId = ImmutableIdentifier<OperationalCampaignIdTag>;
using GovernanceProvenanceEventId =
    ImmutableIdentifier<GovernanceProvenanceEventIdTag>;
using AuthorizationEventId = ImmutableIdentifier<AuthorizationEventIdTag>;
using BudgetLedgerEntryId = ImmutableIdentifier<BudgetLedgerEntryIdTag>;
using ReservationId = ImmutableIdentifier<ReservationIdTag>;
using ReservationEventId = ImmutableIdentifier<ReservationEventIdTag>;
using OperationalRequestId = ImmutableIdentifier<OperationalRequestIdTag>;
using DispatchAttemptId = ImmutableIdentifier<DispatchAttemptIdTag>;
using DispatchAttemptOutcomeId =
    ImmutableIdentifier<DispatchAttemptOutcomeIdTag>;
using RequestBindingId = ImmutableIdentifier<RequestBindingIdTag>;
using DownstreamControlOwnerId =
    ImmutableIdentifier<DownstreamControlOwnerIdTag>;
using ControlEventId = ImmutableIdentifier<ControlEventIdTag>;
using CancellationRequestId = ImmutableIdentifier<CancellationRequestIdTag>;
using CancellationSettlementId =
    ImmutableIdentifier<CancellationSettlementIdTag>;
using ReconciliationObservationId =
    ImmutableIdentifier<ReconciliationObservationIdTag>;
using ReconciliationResolutionId =
    ImmutableIdentifier<ReconciliationResolutionIdTag>;
using CompletionEventId = ImmutableIdentifier<CompletionEventIdTag>;
using AuditReferenceEventId =
    ImmutableIdentifier<AuditReferenceEventIdTag>;
using ProductionEnablementEventId =
    ImmutableIdentifier<ProductionEnablementEventIdTag>;
using RequestProductionAdmissionId =
    ImmutableIdentifier<RequestProductionAdmissionIdTag>;

class CanonicalIdentity final
{
public:
    static CanonicalIdentity Create(int contractVersion,
        std::string canonicalText);
    static CanonicalIdentity Hydrate(int contractVersion,
        std::string canonicalText, std::string hash);

    CanonicalIdentity(const CanonicalIdentity&) = default;
    CanonicalIdentity(CanonicalIdentity&&) = default;
    CanonicalIdentity& operator=(const CanonicalIdentity&) = delete;
    CanonicalIdentity& operator=(CanonicalIdentity&&) = delete;

    int contractVersion() const noexcept;
    const std::string& canonicalText() const noexcept;
    const std::string& hash() const noexcept;
    bool operator==(const CanonicalIdentity&) const = default;

private:
    CanonicalIdentity(int contractVersion, std::string canonicalText,
        std::string hash);

    const int contractVersion_;
    const std::string canonicalText_;
    const std::string hash_;
};

class ActorIdentity final
{
public:
    explicit ActorIdentity(std::string value);

    const std::string& value() const noexcept;
    bool operator==(const ActorIdentity&) const = default;

private:
    const std::string value_;
};

class Reason final
{
public:
    explicit Reason(std::string value);

    const std::string& value() const noexcept;
    bool operator==(const Reason&) const = default;

private:
    const std::string value_;
};

class UtcTimestamp final
{
public:
    explicit UtcTimestamp(std::string value);

    const std::string& value() const noexcept;
    bool operator==(const UtcTimestamp&) const = default;
    auto operator<=>(const UtcTimestamp&) const = default;

private:
    const std::string value_;
};

enum class CampaignOriginKind
{
    phase4dMaterializationV1
};

enum class OperationalActionKind
{
    dispatchFullMaterialization,
    adoptExistingPendingAndControl
};

enum class ScopeKind
{
    completeMaterialization
};

enum class PrerequisitePolicy
{
    phase4dMaterializationOnlyV1,
    phase4dMaterializationPlusExactPhase6dRatificationV1
};

enum class AuthorizationEventKind
{
    granted,
    revoked,
    expiryObserved
};

enum class AdministrativeCampaignState
{
    awaitingOperationalAuthorization,
    authorized,
    budgeted,
    reserving,
    ready,
    dispatching,
    active,
    paused,
    cancellationRequested,
    cancelling,
    terminalCompleted,
    terminalCancelled,
    terminalFailed,
    inconsistent,
    reconciliationRequired
};

enum class MemberOperationalState
{
    notRequested,
    reserved,
    requestReady,
    dispatching,
    boundPending,
    claimed,
    running,
    terminalCompleted,
    terminalFailed,
    terminalCancelled,
    inconsistent,
    reconciliationRequired
};

enum class BudgetLedgerEntryKind
{
    grant,
    amend,
    revoke,
    supersede
};

enum class BudgetLedgerStatus
{
    active,
    revoked
};

enum class BudgetUnit
{
    materializedMemberDispatch
};

enum class ReservationState
{
    held,
    committed,
    released,
    expired,
    reconciliationRequired
};

enum class ReservationEventKind
{
    acquired,
    committed,
    released,
    expired,
    reconciliationRequired
};

enum class RequestState
{
    ready,
    dispatching,
    bound,
    permanentlyFailed,
    cancelled,
    reconciliationRequired
};

enum class ControlEventKind
{
    pause,
    resume
};

enum class DispatchOutcome
{
    createdAndBound,
    adoptedExistingPendingAndBound,
    existingIdentical,
    rejected,
    conflict,
    reconciliationRequired,
    noDownstreamCommit
};

enum class BindingDisposition
{
    created,
    adoptedExistingPending
};

enum class DownstreamControlMode
{
    createdControl,
    authorizedAdoptionControl
};

enum class CancellationSettlementDisposition
{
    unboundCancelled,
    lifecycleRequestAccepted,
    alreadyTerminal,
    runningCancellationNotSupported,
    inconsistent
};

enum class ReconciliationReason
{
    readyRequestNotDispatched,
    dispatchLeaseExpiredNoDownstreamEvidence,
    dispatchOutcomeUnknown,
    bindingProjectionMissing,
    reservationProjectionMissingCommit,
    heldReservationTerminalUnboundRequest,
    reservationExpiredNoDownstreamEvidence,
    cancellationSettlementPending,
    terminalLifecycleCompletionReady,
    progressedUnboundEvidence,
    partialDownstreamEvidence,
    bindingCardinalityMismatch,
    controlOwnerConflict,
    budgetAccountingMismatch,
    postCompletionLifecycleChanged,
    causalityAmbiguous
};

enum class CompletionClassification
{
    operationalRequestFailed,
    mixedTerminalOutcomes,
    downstreamFailure,
    terminalPartialCompletion,
    allScopeCancelled,
    allDownstreamCompleted
};

enum class ReplayDisposition
{
    recorded,
    existingIdentical,
    conflict,
    rejected,
    repaired,
    noChange
};

std::string ToText(CampaignOriginKind value);
std::string ToText(OperationalActionKind value);
std::string ToText(ScopeKind value);
std::string ToText(PrerequisitePolicy value);
std::string ToText(AuthorizationEventKind value);
std::string ToText(AdministrativeCampaignState value);
std::string ToText(MemberOperationalState value);
std::string ToText(BudgetLedgerEntryKind value);
std::string ToText(BudgetLedgerStatus value);
std::string ToText(BudgetUnit value);
std::string ToText(ReservationState value);
std::string ToText(ReservationEventKind value);
std::string ToText(RequestState value);
std::string ToText(ControlEventKind value);
std::string ToText(DispatchOutcome value);
std::string ToText(BindingDisposition value);
std::string ToText(DownstreamControlMode value);
std::string ToText(CancellationSettlementDisposition value);
std::string ToText(ReconciliationReason value);
std::string ToText(CompletionClassification value);
std::string ToText(ReplayDisposition value);

CampaignOriginKind CampaignOriginKindFromText(const std::string& text);
OperationalActionKind OperationalActionKindFromText(const std::string& text);
ScopeKind ScopeKindFromText(const std::string& text);
PrerequisitePolicy PrerequisitePolicyFromText(const std::string& text);
AuthorizationEventKind AuthorizationEventKindFromText(
    const std::string& text);
AdministrativeCampaignState AdministrativeCampaignStateFromText(
    const std::string& text);
BudgetLedgerEntryKind BudgetLedgerEntryKindFromText(const std::string& text);
BudgetLedgerStatus BudgetLedgerStatusFromText(const std::string& text);
BudgetUnit BudgetUnitFromText(const std::string& text);
ReservationState ReservationStateFromText(const std::string& text);
ReservationEventKind ReservationEventKindFromText(const std::string& text);
RequestState RequestStateFromText(const std::string& text);
ControlEventKind ControlEventKindFromText(const std::string& text);
CancellationSettlementDisposition
CancellationSettlementDispositionFromText(const std::string& text);
ReconciliationReason ReconciliationReasonFromText(
    const std::string& text);

struct OperationalCampaign final
{
    static constexpr bool persistent = true;
    static constexpr bool mutableStatusPresent = false;
    static constexpr bool readinessEventPresent = false;
    static constexpr bool schedulerAuthorityGranted = false;
    static constexpr bool lifecycleAuthorityGranted = false;

    const CanonicalIdentity identity;
    const long long materializationId;
    const int materializationContractVersion;
    const std::string materializationCanonicalText;
    const std::string materializationIdentityHash;
    const int memberCount;
    const CampaignOriginKind originKind;
    const OperationalActionKind actionKind;
    const int actionContractVersion;
    const ScopeKind scopeKind;
    const int scopeContractVersion;

    OperationalCampaign(const OperationalCampaign&) = default;
    OperationalCampaign(OperationalCampaign&&) = default;
    OperationalCampaign& operator=(const OperationalCampaign&) = delete;
    OperationalCampaign& operator=(OperationalCampaign&&) = delete;
    bool operator==(const OperationalCampaign&) const = default;

private:
    OperationalCampaign(CanonicalIdentity identity, long long materializationId,
        int materializationContractVersion,
        std::string materializationCanonicalText,
        std::string materializationIdentityHash, int memberCount,
        CampaignOriginKind originKind, OperationalActionKind actionKind,
        int actionContractVersion, ScopeKind scopeKind,
        int scopeContractVersion);
    friend OperationalCampaign BuildOperationalCampaign(long long, int,
        std::string, std::string, int, CampaignOriginKind,
        OperationalActionKind, int, ScopeKind, int);
};

OperationalCampaign BuildOperationalCampaign(long long materializationId,
    int materializationContractVersion,
    std::string materializationCanonicalText,
    std::string materializationIdentityHash, int memberCount,
    CampaignOriginKind originKind = CampaignOriginKind::phase4dMaterializationV1,
    OperationalActionKind actionKind =
        OperationalActionKind::dispatchFullMaterialization,
    int actionContractVersion = kCampaignOperationsActionContractVersion,
    ScopeKind scopeKind = ScopeKind::completeMaterialization,
    int scopeContractVersion = kCampaignOperationsScopeContractVersion);

void ValidateOperationalCampaign(const OperationalCampaign& campaign);
AdministrativeCampaignState InitialAdministrativeState(
    const OperationalCampaign& campaign);

struct GovernanceProvenanceEvent final
{
    const CanonicalIdentity identity;
    const OperationalCampaignId campaignId;
    const std::string campaignCanonicalText;
    const long long ratificationEventId;
    const int ratificationContractVersion;
    const std::string ratificationCanonicalText;
    const std::string ratificationIdentityHash;
    const long long reviewEventId;
    const int reviewContractVersion;
    const std::string reviewCanonicalText;
    const std::string reviewIdentityHash;
    const long long proposalId;
    const int proposalContractVersion;
    const std::string proposalCanonicalText;
    const std::string proposalIdentityHash;
    const PrerequisitePolicy prerequisitePolicy;

    GovernanceProvenanceEvent(const GovernanceProvenanceEvent&) = default;
    GovernanceProvenanceEvent(GovernanceProvenanceEvent&&) = default;
    GovernanceProvenanceEvent& operator=(
        const GovernanceProvenanceEvent&) = delete;
    GovernanceProvenanceEvent& operator=(
        GovernanceProvenanceEvent&&) = delete;
    bool operator==(const GovernanceProvenanceEvent&) const = default;

private:
    GovernanceProvenanceEvent(CanonicalIdentity identity,
        OperationalCampaignId campaignId, std::string campaignCanonicalText,
        long long ratificationEventId, int ratificationContractVersion,
        std::string ratificationCanonicalText,
        std::string ratificationIdentityHash, long long reviewEventId,
        int reviewContractVersion, std::string reviewCanonicalText,
        std::string reviewIdentityHash, long long proposalId,
        int proposalContractVersion, std::string proposalCanonicalText,
        std::string proposalIdentityHash,
        PrerequisitePolicy prerequisitePolicy);
    friend GovernanceProvenanceEvent BuildGovernanceProvenanceEvent(
        OperationalCampaignId, std::string, long long, int, std::string,
        std::string, long long, int, std::string, std::string, long long, int,
        std::string, std::string, PrerequisitePolicy);
};

GovernanceProvenanceEvent BuildGovernanceProvenanceEvent(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    long long ratificationEventId, int ratificationContractVersion,
    std::string ratificationCanonicalText,
    std::string ratificationIdentityHash, long long reviewEventId,
    int reviewContractVersion, std::string reviewCanonicalText,
    std::string reviewIdentityHash, long long proposalId,
    int proposalContractVersion, std::string proposalCanonicalText,
    std::string proposalIdentityHash, PrerequisitePolicy prerequisitePolicy);

void ValidateGovernanceProvenanceEvent(
    const GovernanceProvenanceEvent& event);

struct OperationalAuthorizationEvent final
{
    const CanonicalIdentity identity;
    const OperationalCampaignId campaignId;
    const std::string campaignCanonicalText;
    const std::optional<AuthorizationEventId> previousEventId;
    const std::optional<std::string> previousEventCanonicalText;
    const std::optional<std::string> previousEventIdentityHash;
    const int chainVersion;
    const AuthorizationEventKind eventKind;
    const OperationalActionKind actionKind;
    const int actionContractVersion;
    const ScopeKind scopeKind;
    const int scopeContractVersion;
    const PrerequisitePolicy prerequisitePolicy;
    const std::optional<GovernanceProvenanceEventId> provenanceEventId;
    const std::optional<std::string> provenanceCanonicalText;
    const std::optional<std::string> provenanceIdentityHash;
    const std::string authorizationRole;
    const ActorIdentity actor;
    const Reason reason;
    const UtcTimestamp notBefore;
    const std::optional<UtcTimestamp> expiresAt;

    OperationalAuthorizationEvent(const OperationalAuthorizationEvent&) =
        default;
    OperationalAuthorizationEvent(OperationalAuthorizationEvent&&) = default;
    OperationalAuthorizationEvent& operator=(
        const OperationalAuthorizationEvent&) = delete;
    OperationalAuthorizationEvent& operator=(
        OperationalAuthorizationEvent&&) = delete;
    bool operator==(const OperationalAuthorizationEvent&) const = default;

private:
    OperationalAuthorizationEvent(CanonicalIdentity identity,
        OperationalCampaignId campaignId, std::string campaignCanonicalText,
        std::optional<AuthorizationEventId> previousEventId,
        std::optional<std::string> previousEventCanonicalText,
        std::optional<std::string> previousEventIdentityHash, int chainVersion,
        AuthorizationEventKind eventKind, OperationalActionKind actionKind,
        int actionContractVersion, ScopeKind scopeKind,
        int scopeContractVersion, PrerequisitePolicy prerequisitePolicy,
        std::optional<GovernanceProvenanceEventId> provenanceEventId,
        std::optional<std::string> provenanceCanonicalText,
        std::optional<std::string> provenanceIdentityHash,
        std::string authorizationRole, ActorIdentity actor, Reason reason,
        UtcTimestamp notBefore, std::optional<UtcTimestamp> expiresAt);
    friend OperationalAuthorizationEvent BuildOperationalAuthorizationEvent(
        OperationalCampaignId, std::string,
        std::optional<AuthorizationEventId>, std::optional<std::string>,
        std::optional<std::string>, int, AuthorizationEventKind,
        OperationalActionKind, int, ScopeKind, int, PrerequisitePolicy,
        std::optional<GovernanceProvenanceEventId>,
        std::optional<std::string>, std::optional<std::string>, std::string,
        ActorIdentity, Reason, UtcTimestamp, std::optional<UtcTimestamp>);
};

OperationalAuthorizationEvent BuildOperationalAuthorizationEvent(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::optional<AuthorizationEventId> previousEventId,
    std::optional<std::string> previousEventCanonicalText,
    std::optional<std::string> previousEventIdentityHash, int chainVersion,
    AuthorizationEventKind eventKind, OperationalActionKind actionKind,
    int actionContractVersion, ScopeKind scopeKind, int scopeContractVersion,
    PrerequisitePolicy prerequisitePolicy,
    std::optional<GovernanceProvenanceEventId> provenanceEventId,
    std::optional<std::string> provenanceCanonicalText,
    std::optional<std::string> provenanceIdentityHash,
    std::string authorizationRole, ActorIdentity actor, Reason reason,
    UtcTimestamp notBefore, std::optional<UtcTimestamp> expiresAt);

void ValidateOperationalAuthorizationEvent(
    const OperationalAuthorizationEvent& event);
bool IsAuthorizationEffectiveAt(
    const OperationalAuthorizationEvent& head, const UtcTimestamp& databaseTime);

struct BudgetLedgerEntry final
{
    const CanonicalIdentity identity;
    const OperationalCampaignId campaignId;
    const std::string campaignCanonicalText;
    const std::optional<BudgetLedgerEntryId> previousEntryId;
    const std::optional<std::string> previousEntryCanonicalText;
    const std::optional<std::string> previousEntryIdentityHash;
    const int ledgerVersion;
    const BudgetLedgerEntryKind entryKind;
    const BudgetLedgerStatus status;
    const BudgetUnit unit;
    const long long delta;
    const long long priorTotal;
    const long long resultingTotal;
    const ActorIdentity administrator;
    const Reason reason;

    BudgetLedgerEntry(const BudgetLedgerEntry&) = default;
    BudgetLedgerEntry(BudgetLedgerEntry&&) = default;
    BudgetLedgerEntry& operator=(const BudgetLedgerEntry&) = delete;
    BudgetLedgerEntry& operator=(BudgetLedgerEntry&&) = delete;
    bool operator==(const BudgetLedgerEntry&) const = default;

private:
    BudgetLedgerEntry(CanonicalIdentity identity,
        OperationalCampaignId campaignId, std::string campaignCanonicalText,
        std::optional<BudgetLedgerEntryId> previousEntryId,
        std::optional<std::string> previousEntryCanonicalText,
        std::optional<std::string> previousEntryIdentityHash,
        int ledgerVersion, BudgetLedgerEntryKind entryKind,
        BudgetLedgerStatus status, BudgetUnit unit, long long delta,
        long long priorTotal, long long resultingTotal,
        ActorIdentity administrator, Reason reason);
    friend BudgetLedgerEntry BuildBudgetLedgerEntry(OperationalCampaignId,
        std::string, std::optional<BudgetLedgerEntryId>,
        std::optional<std::string>, std::optional<std::string>, int,
        BudgetLedgerEntryKind, BudgetLedgerStatus, BudgetUnit, long long,
        long long, long long, ActorIdentity, Reason);
};

BudgetLedgerEntry BuildBudgetLedgerEntry(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::optional<BudgetLedgerEntryId> previousEntryId,
    std::optional<std::string> previousEntryCanonicalText,
    std::optional<std::string> previousEntryIdentityHash, int ledgerVersion,
    BudgetLedgerEntryKind entryKind, BudgetLedgerStatus status,
    BudgetUnit unit, long long delta, long long priorTotal,
    long long resultingTotal, ActorIdentity administrator, Reason reason);
void ValidateBudgetLedgerEntry(const BudgetLedgerEntry& entry);

struct LogicalOperation final
{
    const CanonicalIdentity identity;
    const OperationalCampaignId campaignId;
    const std::string campaignCanonicalText;
    const OperationalActionKind actionKind;
    const int actionContractVersion;
    const long long materializationId;
    const int materializationContractVersion;
    const std::string materializationCanonicalText;
    const std::string materializationIdentityHash;
    const ScopeKind scopeKind;
    const int scopeContractVersion;

    LogicalOperation(const LogicalOperation&) = default;
    LogicalOperation(LogicalOperation&&) = default;
    LogicalOperation& operator=(const LogicalOperation&) = delete;
    LogicalOperation& operator=(LogicalOperation&&) = delete;
    bool operator==(const LogicalOperation&) const = default;

private:
    LogicalOperation(CanonicalIdentity identity,
        OperationalCampaignId campaignId, std::string campaignCanonicalText,
        OperationalActionKind actionKind, int actionContractVersion,
        long long materializationId, int materializationContractVersion,
        std::string materializationCanonicalText,
        std::string materializationIdentityHash, ScopeKind scopeKind,
        int scopeContractVersion);
    friend LogicalOperation BuildLogicalOperation(OperationalCampaignId,
        std::string, OperationalActionKind, int, long long, int, std::string,
        std::string, ScopeKind, int);
};

LogicalOperation BuildLogicalOperation(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    OperationalActionKind actionKind, int actionContractVersion,
    long long materializationId, int materializationContractVersion,
    std::string materializationCanonicalText,
    std::string materializationIdentityHash, ScopeKind scopeKind,
    int scopeContractVersion);
LogicalOperation BuildLogicalOperation(
    OperationalCampaignId campaignId, const OperationalCampaign& campaign);
void ValidateLogicalOperation(const LogicalOperation& operation);

struct Reservation final
{
    const CanonicalIdentity identity;
    const LogicalOperation logicalOperation;
    const AuthorizationEventId acceptingAuthorizationEventId;
    const std::string acceptingAuthorizationCanonicalText;
    const std::string acceptingAuthorizationIdentityHash;
    const BudgetLedgerEntryId budgetLedgerEntryId;
    const int budgetLedgerVersion;
    const std::string budgetLedgerCanonicalText;
    const std::string budgetLedgerIdentityHash;
    const int memberCount;
    const long long amount;
    const BudgetUnit unit;
    const std::optional<UtcTimestamp> expiresAt;

    Reservation(const Reservation&) = default;
    Reservation(Reservation&&) = default;
    Reservation& operator=(const Reservation&) = delete;
    Reservation& operator=(Reservation&&) = delete;
    bool operator==(const Reservation&) const = default;

private:
    Reservation(CanonicalIdentity identity, LogicalOperation logicalOperation,
        AuthorizationEventId acceptingAuthorizationEventId,
        std::string acceptingAuthorizationCanonicalText,
        std::string acceptingAuthorizationIdentityHash,
        BudgetLedgerEntryId budgetLedgerEntryId, int budgetLedgerVersion,
        std::string budgetLedgerCanonicalText,
        std::string budgetLedgerIdentityHash, int memberCount,
        long long amount, BudgetUnit unit,
        std::optional<UtcTimestamp> expiresAt);
    friend Reservation BuildReservation(LogicalOperation,
        AuthorizationEventId, std::string, std::string, BudgetLedgerEntryId,
        int, std::string, std::string, int, long long, BudgetUnit,
        std::optional<UtcTimestamp>);
};

Reservation BuildReservation(LogicalOperation logicalOperation,
    AuthorizationEventId acceptingAuthorizationEventId,
    std::string acceptingAuthorizationCanonicalText,
    std::string acceptingAuthorizationIdentityHash,
    BudgetLedgerEntryId budgetLedgerEntryId, int budgetLedgerVersion,
    std::string budgetLedgerCanonicalText,
    std::string budgetLedgerIdentityHash, int memberCount, long long amount,
    BudgetUnit unit, std::optional<UtcTimestamp> expiresAt);
void ValidateReservation(const Reservation& reservation);

struct OperationalRequest final
{
    static constexpr bool dispatchAuthorityGranted = false;
    static constexpr bool schedulerAuthorityGranted = false;
    static constexpr bool lifecycleAuthorityGranted = false;

    const CanonicalIdentity identity;
    const LogicalOperation logicalOperation;
    const AuthorizationEventId acceptingAuthorizationEventId;
    const std::string acceptingAuthorizationCanonicalText;
    const std::string acceptingAuthorizationIdentityHash;
    const ReservationId reservationId;
    const std::string reservationCanonicalText;
    const std::string reservationIdentityHash;
    const int memberCount;
    const std::string orderedScopeDigest;
    const ActorIdentity acceptingActor;
    const Reason reason;
    const PrerequisitePolicy prerequisitePolicy;
    const std::optional<std::string> provenanceCanonicalText;
    const std::optional<std::string> provenanceIdentityHash;

    OperationalRequest(const OperationalRequest&) = default;
    OperationalRequest(OperationalRequest&&) = default;
    OperationalRequest& operator=(const OperationalRequest&) = delete;
    OperationalRequest& operator=(OperationalRequest&&) = delete;
    bool operator==(const OperationalRequest&) const = default;

private:
    OperationalRequest(CanonicalIdentity identity,
        LogicalOperation logicalOperation,
        AuthorizationEventId acceptingAuthorizationEventId,
        std::string acceptingAuthorizationCanonicalText,
        std::string acceptingAuthorizationIdentityHash,
        ReservationId reservationId, std::string reservationCanonicalText,
        std::string reservationIdentityHash, int memberCount,
        std::string orderedScopeDigest, ActorIdentity acceptingActor,
        Reason reason, PrerequisitePolicy prerequisitePolicy,
        std::optional<std::string> provenanceCanonicalText,
        std::optional<std::string> provenanceIdentityHash);
    friend OperationalRequest BuildOperationalRequest(LogicalOperation,
        AuthorizationEventId, std::string, std::string, ReservationId,
        std::string, std::string, int, std::string, ActorIdentity, Reason,
        PrerequisitePolicy, std::optional<std::string>,
        std::optional<std::string>);
};

OperationalRequest BuildOperationalRequest(
    LogicalOperation logicalOperation,
    AuthorizationEventId acceptingAuthorizationEventId,
    std::string acceptingAuthorizationCanonicalText,
    std::string acceptingAuthorizationIdentityHash,
    ReservationId reservationId, std::string reservationCanonicalText,
    std::string reservationIdentityHash, int memberCount,
    std::string orderedScopeDigest, ActorIdentity acceptingActor,
    Reason reason, PrerequisitePolicy prerequisitePolicy,
    std::optional<std::string> provenanceCanonicalText,
    std::optional<std::string> provenanceIdentityHash);
void ValidateOperationalRequest(const OperationalRequest& request);

struct ReservationEvent final
{
    const CanonicalIdentity identity;
    const ReservationId reservationId;
    const std::string reservationCanonicalText;
    const ReservationEventKind eventKind;
    const std::optional<ReservationState> expectedState;
    const ReservationState resultingState;
    const int expectedVersion;
    const int resultingVersion;
    const OperationalRequestId requestId;
    const std::string requestCanonicalText;
    const long long amount;

    ReservationEvent(const ReservationEvent&) = default;
    ReservationEvent(ReservationEvent&&) = default;
    ReservationEvent& operator=(const ReservationEvent&) = delete;
    ReservationEvent& operator=(ReservationEvent&&) = delete;
    bool operator==(const ReservationEvent&) const = default;

private:
    ReservationEvent(CanonicalIdentity identity, ReservationId reservationId,
        std::string reservationCanonicalText, ReservationEventKind eventKind,
        std::optional<ReservationState> expectedState,
        ReservationState resultingState, int expectedVersion,
        int resultingVersion, OperationalRequestId requestId,
        std::string requestCanonicalText, long long amount);
    friend ReservationEvent BuildReservationAcquisitionEvent(ReservationId,
        std::string, OperationalRequestId, std::string, long long);
};

ReservationEvent BuildReservationAcquisitionEvent(
    ReservationId reservationId, std::string reservationCanonicalText,
    OperationalRequestId requestId, std::string requestCanonicalText,
    long long amount);
void ValidateReservationEvent(const ReservationEvent& event);

struct BudgetAccounting final
{
    const long long granted;
    const long long everReserved;
    const long long committed;
    const long long releasedOrExpired;
    const long long held;
    const long long arithmeticallyUnallocated;
    const long long reservable;
    const BudgetLedgerStatus status;

    bool operator==(const BudgetAccounting&) const = default;
};

BudgetAccounting CalculateBudgetAccounting(long long granted,
    long long everReserved, long long committed,
    long long releasedOrExpired, BudgetLedgerStatus status);

struct CompletionEvidence final
{
    int scopeMemberCount = 0;
    int completedMemberCount = 0;
    int failedMemberCount = 0;
    int cancelledOrNeverDispatchedMemberCount = 0;
    bool unboundRequestPermanentlyFailed = false;
    bool anyMemberBound = false;
    bool allObligationsSettled = false;
};

CompletionClassification ClassifyCompletion(
    const CompletionEvidence& evidence);

} // namespace EA::CampaignOperations
