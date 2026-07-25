#pragma once

#include "CampaignOperations.hpp"

#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace EA::CampaignOperations
{

inline constexpr int kCampaignOperationsDispatchContractVersion = 1;
inline constexpr int kCampaignOperationsMaximumDispatchBatchSize = 100;
inline constexpr int kCampaignOperationsDispatchLeaseSeconds = 300;
inline constexpr char kCampaignOperationsDispatcherRole[] =
    "campaign_operations_dispatcher";
inline constexpr char kCampaignOperationsPhase5TransactionalRole[] =
    "campaign_operations_phase5_transactional";

class LeaseTokenDigest final
{
public:
    static LeaseTokenDigest Derive(const std::string& opaqueToken);
    static LeaseTokenDigest Hydrate(std::string value);

    const std::string& value() const noexcept;
    bool operator==(const LeaseTokenDigest&) const = default;

private:
    explicit LeaseTokenDigest(std::string value);
    const std::string value_;
};

enum class DownstreamEvidenceClassification
{
    noPhase5Evidence,
    exactCompletePendingTrain,
    partialPhase5Evidence,
    pausedOnlyEvidence,
    progressedUnboundEvidence,
    causallyAmbiguous,
    completeCampaignOperationsBinding
};

enum class DispatchResultClassification
{
    createdAndBound,
    adoptedExistingPendingAndBound,
    existingIdentical,
    rejected,
    semanticConflict,
    reconciliationRequired
};

enum class ExactReplayDisposition
{
    newOperation,
    authoritativeExisting,
    provenAbsent,
    changedPayloadConflict,
    reconciliationRequired
};

enum class SemanticConflictClassification
{
    none,
    stateVersionMismatch,
    leaseUnavailable,
    authorizationInactive,
    budgetInactive,
    reservationMismatch,
    requestMismatch,
    controlOwnerCollision,
    partialDownstreamEvidence,
    pausedOnlyEvidence,
    progressedUnboundEvidence,
    causalityMismatch,
    bindingMismatch
};

enum class UncertainCommitRecoveryClassification
{
    completeAuthoritativeBinding,
    provenNoCommit,
    ambiguousEvidence
};

enum class Phase5ExecutionDisposition
{
    created,
    reused
};

enum class Phase5ActivationDisposition
{
    created,
    reused
};

// Verification hooks are consumed only by the isolated Phase 3 adapter.  The
// production path supplies no hook, and the adapter's exact database-prefix
// and acknowledgement gates are validated before any hook can run.
enum class DispatchTestInjectionPoint
{
    beforeRequestLeaseTransition,
    afterRequestLeaseTransitionBeforeAttempt,
    afterAttemptBeforeAcquisitionAudit,
    beforeAcquisitionCommit,
    beforeDownstreamEvidenceClassification,
    beforePhase5Invocation,
    afterPhase5ExecutionMutation,
    afterPhase5ActivationMutation,
    afterExperimentCreationOrReuseMutation,
    duringBindingInsertion,
    afterBindingSubset,
    duringControlOwnerInsertion,
    afterControlOwnerSubset,
    beforeReservationCommitmentEventInsertion,
    afterReservationCommitmentEventBeforeProjection,
    afterReservationProjectionBeforeRequestTransition,
    afterRequestTransitionBeforeAttemptOutcome,
    afterAttemptOutcomeBeforeAudit,
    beforeHandoffCommit,
    afterSuccessfulCommitBeforeResponse
};

using DispatchTestHook =
    std::function<void(DispatchTestInjectionPoint)>;

class DispatchTestSqlState final : public std::exception
{
public:
    explicit DispatchTestSqlState(std::string sqlState);
    const char* what() const noexcept override;
    const std::string& sqlState() const noexcept;

private:
    const std::string sqlState_;
};

std::string ToText(DownstreamEvidenceClassification value);
std::string ToText(DispatchResultClassification value);
std::string ToText(ExactReplayDisposition value);
std::string ToText(SemanticConflictClassification value);
std::string ToText(UncertainCommitRecoveryClassification value);
std::string ToText(Phase5ExecutionDisposition value);
std::string ToText(Phase5ActivationDisposition value);

DownstreamEvidenceClassification DownstreamEvidenceClassificationFromText(
    const std::string& text);
DispatchResultClassification DispatchResultClassificationFromText(
    const std::string& text);
ExactReplayDisposition ExactReplayDispositionFromText(
    const std::string& text);
SemanticConflictClassification SemanticConflictClassificationFromText(
    const std::string& text);
UncertainCommitRecoveryClassification
UncertainCommitRecoveryClassificationFromText(const std::string& text);
Phase5ExecutionDisposition Phase5ExecutionDispositionFromText(
    const std::string& text);
Phase5ActivationDisposition Phase5ActivationDispositionFromText(
    const std::string& text);

struct DispatchAttemptAcquisition final
{
    const CanonicalIdentity identity;
    const OperationalRequestId requestId;
    const std::string requestCanonicalText;
    const int attemptOrdinal;
    const int expectedRequestVersion;
    const int resultingRequestVersion;
    const LeaseTokenDigest leaseTokenDigest;
    const UtcTimestamp leaseExpiresAt;
    const ActorIdentity dispatcher;

    DispatchAttemptAcquisition(const DispatchAttemptAcquisition&) = default;
    DispatchAttemptAcquisition(DispatchAttemptAcquisition&&) = default;
    DispatchAttemptAcquisition& operator=(
        const DispatchAttemptAcquisition&) = delete;
    DispatchAttemptAcquisition& operator=(
        DispatchAttemptAcquisition&&) = delete;
    bool operator==(const DispatchAttemptAcquisition&) const = default;

private:
    DispatchAttemptAcquisition(CanonicalIdentity identity,
        OperationalRequestId requestId, std::string requestCanonicalText,
        int attemptOrdinal, int expectedRequestVersion,
        int resultingRequestVersion, LeaseTokenDigest leaseTokenDigest,
        UtcTimestamp leaseExpiresAt, ActorIdentity dispatcher);
    friend DispatchAttemptAcquisition BuildDispatchAttemptAcquisition(
        OperationalRequestId, std::string, int, int, int, LeaseTokenDigest,
        UtcTimestamp, ActorIdentity);
};

DispatchAttemptAcquisition BuildDispatchAttemptAcquisition(
    OperationalRequestId requestId, std::string requestCanonicalText,
    int attemptOrdinal, int expectedRequestVersion,
    int resultingRequestVersion, LeaseTokenDigest leaseTokenDigest,
    UtcTimestamp leaseExpiresAt, ActorIdentity dispatcher);
void ValidateDispatchAttemptAcquisition(
    const DispatchAttemptAcquisition& acquisition);

struct RequestMemberBinding final
{
    const CanonicalIdentity identity;
    const OperationalRequestId requestId;
    const std::string requestCanonicalText;
    const long long materializationId;
    const std::string materializationCanonicalText;
    const long long materializationMemberId;
    const int memberOrdinal;
    const std::string selectedMemberCanonicalText;
    const std::string selectedMemberIdentityHash;
    const long long proposalId;
    const std::string proposalCanonicalText;
    const std::string proposalIdentityHash;
    const long long reviewDecisionId;
    const long long executionId;
    const std::string executionCanonicalText;
    const std::string executionIdentityHash;
    const long long activationId;
    const std::string activationCanonicalText;
    const std::string activationIdentityHash;
    const long long experimentId;
    const BindingDisposition bindingDisposition;
    const Phase5ExecutionDisposition executionDisposition;
    const Phase5ActivationDisposition activationDisposition;

    RequestMemberBinding(const RequestMemberBinding&) = default;
    RequestMemberBinding(RequestMemberBinding&&) = default;
    RequestMemberBinding& operator=(const RequestMemberBinding&) = delete;
    RequestMemberBinding& operator=(RequestMemberBinding&&) = delete;
    bool operator==(const RequestMemberBinding&) const = default;

private:
    RequestMemberBinding(CanonicalIdentity identity,
        OperationalRequestId requestId, std::string requestCanonicalText,
        long long materializationId,
        std::string materializationCanonicalText,
        long long materializationMemberId, int memberOrdinal,
        std::string selectedMemberCanonicalText,
        std::string selectedMemberIdentityHash, long long proposalId,
        std::string proposalCanonicalText, std::string proposalIdentityHash,
        long long reviewDecisionId, long long executionId,
        std::string executionCanonicalText,
        std::string executionIdentityHash, long long activationId,
        std::string activationCanonicalText,
        std::string activationIdentityHash, long long experimentId,
        BindingDisposition bindingDisposition,
        Phase5ExecutionDisposition executionDisposition,
        Phase5ActivationDisposition activationDisposition);
    friend RequestMemberBinding BuildRequestMemberBinding(
        OperationalRequestId, std::string, long long, std::string, long long,
        int, std::string, std::string, long long, std::string, std::string,
        long long, long long, std::string, std::string, long long, std::string,
        std::string, long long, BindingDisposition,
        Phase5ExecutionDisposition, Phase5ActivationDisposition);
};

RequestMemberBinding BuildRequestMemberBinding(
    OperationalRequestId requestId, std::string requestCanonicalText,
    long long materializationId, std::string materializationCanonicalText,
    long long materializationMemberId, int memberOrdinal,
    std::string selectedMemberCanonicalText,
    std::string selectedMemberIdentityHash, long long proposalId,
    std::string proposalCanonicalText, std::string proposalIdentityHash,
    long long reviewDecisionId, long long executionId,
    std::string executionCanonicalText, std::string executionIdentityHash,
    long long activationId, std::string activationCanonicalText,
    std::string activationIdentityHash, long long experimentId,
    BindingDisposition bindingDisposition,
    Phase5ExecutionDisposition executionDisposition,
    Phase5ActivationDisposition activationDisposition);
void ValidateRequestMemberBinding(const RequestMemberBinding& binding);

struct RequestBindingSet final
{
    const CanonicalIdentity identity;
    const OperationalRequestId requestId;
    const std::string requestCanonicalText;
    const int memberCount;
    const std::vector<RequestMemberBinding> members;

    RequestBindingSet(const RequestBindingSet&) = default;
    RequestBindingSet(RequestBindingSet&&) = default;
    RequestBindingSet& operator=(const RequestBindingSet&) = delete;
    RequestBindingSet& operator=(RequestBindingSet&&) = delete;
    bool operator==(const RequestBindingSet&) const = default;

private:
    RequestBindingSet(CanonicalIdentity identity,
        OperationalRequestId requestId, std::string requestCanonicalText,
        int memberCount, std::vector<RequestMemberBinding> members);
    friend RequestBindingSet BuildRequestBindingSet(
        OperationalRequestId, std::string,
        std::vector<RequestMemberBinding>);
};

RequestBindingSet BuildRequestBindingSet(OperationalRequestId requestId,
    std::string requestCanonicalText,
    std::vector<RequestMemberBinding> members);
void ValidateRequestBindingSet(const RequestBindingSet& bindingSet);

struct DownstreamControlOwner final
{
    const CanonicalIdentity identity;
    const OperationalRequestId requestId;
    const std::string requestCanonicalText;
    const std::string bindingCanonicalText;
    const long long experimentId;
    const DownstreamControlMode mode;
    const std::optional<AuthorizationEventId> adoptionAuthorizationEventId;
    const std::optional<std::string> adoptionAuthorizationCanonicalText;
    const std::optional<std::string> adoptionAuthorizationIdentityHash;

    DownstreamControlOwner(const DownstreamControlOwner&) = default;
    DownstreamControlOwner(DownstreamControlOwner&&) = default;
    DownstreamControlOwner& operator=(const DownstreamControlOwner&) = delete;
    DownstreamControlOwner& operator=(DownstreamControlOwner&&) = delete;
    bool operator==(const DownstreamControlOwner&) const = default;

private:
    DownstreamControlOwner(CanonicalIdentity identity,
        OperationalRequestId requestId, std::string requestCanonicalText,
        std::string bindingCanonicalText, long long experimentId,
        DownstreamControlMode mode,
        std::optional<AuthorizationEventId> adoptionAuthorizationEventId,
        std::optional<std::string> adoptionAuthorizationCanonicalText,
        std::optional<std::string> adoptionAuthorizationIdentityHash);
    friend DownstreamControlOwner BuildDownstreamControlOwner(
        OperationalRequestId, std::string, std::string, long long,
        DownstreamControlMode, std::optional<AuthorizationEventId>,
        std::optional<std::string>, std::optional<std::string>);
};

DownstreamControlOwner BuildDownstreamControlOwner(
    OperationalRequestId requestId, std::string requestCanonicalText,
    std::string bindingCanonicalText, long long experimentId,
    DownstreamControlMode mode,
    std::optional<AuthorizationEventId> adoptionAuthorizationEventId,
    std::optional<std::string> adoptionAuthorizationCanonicalText,
    std::optional<std::string> adoptionAuthorizationIdentityHash);
void ValidateDownstreamControlOwner(const DownstreamControlOwner& owner);

struct ReservationCommitment final
{
    const CanonicalIdentity identity;
    const ReservationId reservationId;
    const std::string reservationCanonicalText;
    const OperationalRequestId requestId;
    const std::string requestCanonicalText;
    const std::string bindingSetCanonicalText;
    const std::string bindingSetIdentityHash;
    const int expectedReservationVersion;
    const int resultingReservationVersion;
    const long long amount;

    ReservationCommitment(const ReservationCommitment&) = default;
    ReservationCommitment(ReservationCommitment&&) = default;
    ReservationCommitment& operator=(const ReservationCommitment&) = delete;
    ReservationCommitment& operator=(ReservationCommitment&&) = delete;
    bool operator==(const ReservationCommitment&) const = default;

private:
    ReservationCommitment(CanonicalIdentity identity,
        ReservationId reservationId, std::string reservationCanonicalText,
        OperationalRequestId requestId, std::string requestCanonicalText,
        std::string bindingSetCanonicalText,
        std::string bindingSetIdentityHash, int expectedReservationVersion,
        int resultingReservationVersion, long long amount);
    friend ReservationCommitment BuildReservationCommitment(
        ReservationId, std::string, OperationalRequestId, std::string,
        std::string, std::string, int, int, long long);
};

ReservationCommitment BuildReservationCommitment(
    ReservationId reservationId, std::string reservationCanonicalText,
    OperationalRequestId requestId, std::string requestCanonicalText,
    std::string bindingSetCanonicalText, std::string bindingSetIdentityHash,
    int expectedReservationVersion, int resultingReservationVersion,
    long long amount);
void ValidateReservationCommitment(const ReservationCommitment& commitment);

struct DispatchAttemptOutcomeEvidence final
{
    const CanonicalIdentity identity;
    const DispatchAttemptId attemptId;
    const std::string attemptCanonicalText;
    const DispatchResultClassification result;
    const DownstreamEvidenceClassification downstreamEvidence;
    const SemanticConflictClassification conflict;
    const UncertainCommitRecoveryClassification recovery;
    const std::string diagnosticCode;
    const int expectedRequestVersion;
    const int resultingRequestVersion;
    const int expectedReservationVersion;
    const int resultingReservationVersion;
    const std::optional<std::string> bindingSetCanonicalText;
    const std::optional<std::string> bindingSetIdentityHash;

    DispatchAttemptOutcomeEvidence(
        const DispatchAttemptOutcomeEvidence&) = default;
    DispatchAttemptOutcomeEvidence(
        DispatchAttemptOutcomeEvidence&&) = default;
    DispatchAttemptOutcomeEvidence& operator=(
        const DispatchAttemptOutcomeEvidence&) = delete;
    DispatchAttemptOutcomeEvidence& operator=(
        DispatchAttemptOutcomeEvidence&&) = delete;
    bool operator==(const DispatchAttemptOutcomeEvidence&) const = default;

private:
    DispatchAttemptOutcomeEvidence(CanonicalIdentity identity,
        DispatchAttemptId attemptId, std::string attemptCanonicalText,
        DispatchResultClassification result,
        DownstreamEvidenceClassification downstreamEvidence,
        SemanticConflictClassification conflict,
        UncertainCommitRecoveryClassification recovery,
        std::string diagnosticCode, int expectedRequestVersion,
        int resultingRequestVersion, int expectedReservationVersion,
        int resultingReservationVersion,
        std::optional<std::string> bindingSetCanonicalText,
        std::optional<std::string> bindingSetIdentityHash);
    friend DispatchAttemptOutcomeEvidence BuildDispatchAttemptOutcomeEvidence(
        DispatchAttemptId, std::string, DispatchResultClassification,
        DownstreamEvidenceClassification, SemanticConflictClassification,
        UncertainCommitRecoveryClassification, std::string, int, int, int,
        int, std::optional<std::string>, std::optional<std::string>);
};

DispatchAttemptOutcomeEvidence BuildDispatchAttemptOutcomeEvidence(
    DispatchAttemptId attemptId, std::string attemptCanonicalText,
    DispatchResultClassification result,
    DownstreamEvidenceClassification downstreamEvidence,
    SemanticConflictClassification conflict,
    UncertainCommitRecoveryClassification recovery,
    std::string diagnosticCode, int expectedRequestVersion,
    int resultingRequestVersion, int expectedReservationVersion,
    int resultingReservationVersion,
    std::optional<std::string> bindingSetCanonicalText,
    std::optional<std::string> bindingSetIdentityHash);
void ValidateDispatchAttemptOutcomeEvidence(
    const DispatchAttemptOutcomeEvidence& outcome);

} // namespace EA::CampaignOperations
