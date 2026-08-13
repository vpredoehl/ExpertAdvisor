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
    bool production = false;
    std::string operationKey;
    std::string approvedBuildContractCanonical;
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

struct DispatchCandidate final
{
    OperationalRequestId requestId;
    std::string requestIdentityCanonical;
    int expectedRequestVersion = 0;
};

bool DispatchSchemaExists(pqxx::transaction_base& transaction);

std::vector<OperationalRequestId> SelectDispatchCandidatesForIsolatedTest(
    pqxx::transaction_base& transaction, int limit);
std::vector<DispatchCandidate> SelectDispatchCandidatesForManager(
    pqxx::connection& connection, int limit);

void RequireExactManagerOperationSourceEvidence(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId,
    OperationalRequestId requestId, const std::string& operationKey,
    const std::string& requestIdentityCanonical, int expectedRequestVersion,
    const std::string& expectedSourceCanonical);
void PersistManagerOperationSourceCanonical(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId,
    OperationalRequestId requestId, const std::string& operationKey,
    const std::string& requestIdentityCanonical, int expectedRequestVersion,
    const std::string& sourceCanonical, const std::string& sourceHash);

// Migration 058 records the finite set of pre-existing H2 caller-keyed
// mgr-v1 operations.  This is intentionally an exact historical lookup, not
// a namespace exception for new caller-keyed acquisition.
bool IsExactGrandfatheredH2ManagerOperation(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    const std::string& operationKey);

DispatchLease AcquireDispatchLeaseInTransaction(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    const ActorIdentity& dispatcher, DispatchTestHook testHook = {});

DispatchLease AcquireProductionDispatchLeaseInTransaction(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    const std::string& operationKey, const ActorIdentity& requestingActor,
    const std::string& approvedBuildContractCanonical,
    const std::optional<std::string>& managerSourceCanonical = std::nullopt);

DispatchLockedAuthority LockAndRevalidateDispatchAuthority(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    int expectedRequestVersion, const LeaseTokenDigest& leaseTokenDigest,
    bool lockAdoptionAuthorization, bool productionDispatch = false,
    const std::string& productionOperationKey = {},
    const std::string& approvedBuildContractCanonical = {}
#if defined(CAMPAIGN_OPERATIONS_H2_TESTING)
    , DispatchTestHook testHook = {}
#endif
    );

std::optional<DispatchAttemptRecord> FindDispatchAttempt(
    pqxx::transaction_base& transaction, DispatchAttemptId attemptId);
std::optional<DispatchAttemptRecord> FindLatestDispatchAttempt(
    pqxx::transaction_base& transaction, OperationalRequestId requestId);
std::optional<DispatchLease> FindRecoverableDispatchLease(
    pqxx::transaction_base& transaction, OperationalRequestId requestId);
std::optional<DispatchLease> FindRecoverableProductionDispatchLease(
    pqxx::transaction_base& transaction, OperationalRequestId requestId,
    const std::string& operationKey,
    const std::optional<std::string>& managerSourceCanonical = std::nullopt);

DownstreamEvidenceClassification ClassifyDownstreamEvidence(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority);
bool HasDownstreamControlOwnerCollision(
    pqxx::transaction_base& transaction,
    const DispatchLockedAuthority& authority);

} // namespace EA::CampaignOperations
