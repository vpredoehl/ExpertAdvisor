#pragma once

#include "CampaignOperationsDispatch.hpp"
#include "CampaignOperationsRepository.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::CampaignOperations
{

struct DispatchLease final
{
    DispatchAttemptId attemptId;
    DispatchAttemptAcquisition acquisition;
    ReservationId reservationId;
    int reservationVersion = 0;
    long long materializationId = 0;
    int memberCount = 0;
};

struct DispatchAttemptRecord final
{
    DispatchAttemptId attemptId;
    DispatchAttemptAcquisition acquisition;
};

struct DispatchLockedAuthority final
{
    OperationalRequestId requestId;
    OperationalCampaignId campaignId;
    ReservationId reservationId;
    AuthorizationEventId dispatchAuthorizationId;
    std::optional<AuthorizationEventId> adoptionAuthorizationId;
    int requestVersion = 0;
    int reservationVersion = 0;
    long long materializationId = 0;
    int memberCount = 0;
    long long amount = 0;
    std::string requestCanonicalText;
    std::string reservationCanonicalText;
    std::string leaseTokenDigest;
};

bool DispatchSchemaExists(pqxx::transaction_base& transaction);

std::vector<OperationalRequestId> SelectDispatchCandidatesForIsolatedTest(
    pqxx::transaction_base& transaction, int limit);

DispatchLease AcquireDispatchLeaseInTransaction(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    const ActorIdentity& dispatcher, DispatchTestHook testHook = {});

DispatchLockedAuthority LockAndRevalidateDispatchAuthority(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    bool lockAdoptionAuthorization);

std::optional<DispatchAttemptRecord> FindDispatchAttempt(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId);
std::optional<DispatchAttemptRecord> FindLatestDispatchAttempt(
    pqxx::transaction_base& transaction, OperationalRequestId requestId);
std::optional<DispatchLease> FindRecoverableDispatchLease(
    pqxx::transaction_base& transaction, OperationalRequestId requestId);

DownstreamEvidenceClassification ClassifyDownstreamEvidence(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority);
bool HasDownstreamControlOwnerCollision(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority);

} // namespace EA::CampaignOperations
