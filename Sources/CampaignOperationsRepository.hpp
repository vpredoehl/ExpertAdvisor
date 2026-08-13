#pragma once

#include "CampaignOperations.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pqxx/pqxx>

namespace EA::CampaignOperations
{

inline constexpr int kCampaignOperationsRepositoryMaximumListLimit = 1000;

enum class PersistOutcome
{
    recorded,
    existingIdentical
};

std::string ToText(PersistOutcome outcome);

struct PersistedOperationalCampaign final
{
    const OperationalCampaignId campaignId;
    const OperationalCampaign campaign;
    const std::string createdAt;

    PersistedOperationalCampaign(OperationalCampaignId campaignId,
        OperationalCampaign campaign, std::string createdAt);
    PersistedOperationalCampaign(const PersistedOperationalCampaign&) = default;
    PersistedOperationalCampaign(PersistedOperationalCampaign&&) = default;
    PersistedOperationalCampaign& operator=(
        const PersistedOperationalCampaign&) = delete;
    PersistedOperationalCampaign& operator=(
        PersistedOperationalCampaign&&) = delete;
};

struct PersistedGovernanceProvenanceEvent final
{
    const GovernanceProvenanceEventId provenanceEventId;
    const GovernanceProvenanceEvent event;
    const std::string createdAt;

    PersistedGovernanceProvenanceEvent(
        GovernanceProvenanceEventId provenanceEventId,
        GovernanceProvenanceEvent event, std::string createdAt);
    PersistedGovernanceProvenanceEvent(
        const PersistedGovernanceProvenanceEvent&) = default;
    PersistedGovernanceProvenanceEvent(
        PersistedGovernanceProvenanceEvent&&) = default;
    PersistedGovernanceProvenanceEvent& operator=(
        const PersistedGovernanceProvenanceEvent&) = delete;
    PersistedGovernanceProvenanceEvent& operator=(
        PersistedGovernanceProvenanceEvent&&) = delete;
};

struct PersistedOperationalAuthorizationEvent final
{
    const AuthorizationEventId authorizationEventId;
    const OperationalAuthorizationEvent event;
    const std::string createdAt;

    PersistedOperationalAuthorizationEvent(
        AuthorizationEventId authorizationEventId,
        OperationalAuthorizationEvent event, std::string createdAt);
    PersistedOperationalAuthorizationEvent(
        const PersistedOperationalAuthorizationEvent&) = default;
    PersistedOperationalAuthorizationEvent(
        PersistedOperationalAuthorizationEvent&&) = default;
    PersistedOperationalAuthorizationEvent& operator=(
        const PersistedOperationalAuthorizationEvent&) = delete;
    PersistedOperationalAuthorizationEvent& operator=(
        PersistedOperationalAuthorizationEvent&&) = delete;
};

struct PersistedBudgetLedgerEntry final
{
    const BudgetLedgerEntryId budgetLedgerEntryId;
    const BudgetLedgerEntry entry;
    const std::string createdAt;

    PersistedBudgetLedgerEntry(BudgetLedgerEntryId budgetLedgerEntryId,
        BudgetLedgerEntry entry, std::string createdAt);
    PersistedBudgetLedgerEntry(const PersistedBudgetLedgerEntry&) = default;
    PersistedBudgetLedgerEntry(PersistedBudgetLedgerEntry&&) = default;
    PersistedBudgetLedgerEntry& operator=(
        const PersistedBudgetLedgerEntry&) = delete;
    PersistedBudgetLedgerEntry& operator=(
        PersistedBudgetLedgerEntry&&) = delete;
};

struct PersistedReservation final
{
    const ReservationId reservationId;
    const Reservation reservation;
    const ReservationState state;
    const int stateVersion;
    const std::string createdAt;
    const std::string updatedAt;

    PersistedReservation(ReservationId reservationId,
        Reservation reservation, ReservationState state, int stateVersion,
        std::string createdAt, std::string updatedAt);
    PersistedReservation(const PersistedReservation&) = default;
    PersistedReservation(PersistedReservation&&) = default;
    PersistedReservation& operator=(const PersistedReservation&) = delete;
    PersistedReservation& operator=(PersistedReservation&&) = delete;
};

struct PersistedOperationalRequest final
{
    const OperationalRequestId requestId;
    const OperationalRequest request;
    const RequestState state;
    const int stateVersion;
    const bool productionDispatchEnabled;
    const std::string createdAt;
    const std::string updatedAt;

    PersistedOperationalRequest(OperationalRequestId requestId,
        OperationalRequest request, RequestState state, int stateVersion,
        bool productionDispatchEnabled, std::string createdAt,
        std::string updatedAt);
    PersistedOperationalRequest(const PersistedOperationalRequest&) = default;
    PersistedOperationalRequest(PersistedOperationalRequest&&) = default;
    PersistedOperationalRequest& operator=(
        const PersistedOperationalRequest&) = delete;
    PersistedOperationalRequest& operator=(
        PersistedOperationalRequest&&) = delete;
};

struct PersistedReservationEvent final
{
    const ReservationEventId reservationEventId;
    const ReservationEvent event;
    const std::string createdAt;

    PersistedReservationEvent(ReservationEventId reservationEventId,
        ReservationEvent event, std::string createdAt);
    PersistedReservationEvent(const PersistedReservationEvent&) = default;
    PersistedReservationEvent(PersistedReservationEvent&&) = default;
    PersistedReservationEvent& operator=(
        const PersistedReservationEvent&) = delete;
    PersistedReservationEvent& operator=(
        PersistedReservationEvent&&) = delete;
};

struct AcceptedOperationalRequest final
{
    const PersistedReservation reservation;
    const PersistedOperationalRequest request;
    const PersistedReservationEvent acquisitionEvent;

    AcceptedOperationalRequest(PersistedReservation reservation,
        PersistedOperationalRequest request,
        PersistedReservationEvent acquisitionEvent);
    AcceptedOperationalRequest(const AcceptedOperationalRequest&) = default;
    AcceptedOperationalRequest(AcceptedOperationalRequest&&) = default;
    AcceptedOperationalRequest& operator=(
        const AcceptedOperationalRequest&) = delete;
    AcceptedOperationalRequest& operator=(
        AcceptedOperationalRequest&&) = delete;
};

struct CampaignBudgetStatusProjection final
{
    long long campaignId = 0;
    std::optional<long long> budgetLedgerEntryId;
    std::optional<int> ledgerVersion;
    std::optional<BudgetLedgerStatus> status;
    long long granted = 0;
    long long everReserved = 0;
    long long committed = 0;
    long long releasedOrExpired = 0;
    long long held = 0;
    long long unallocated = 0;
    long long reservable = 0;
    std::optional<std::string> budgetIdentityHash;
    bool accountingConsistent = false;
};

struct OperationalRequestStatusProjection final
{
    long long requestId = 0;
    long long campaignId = 0;
    RequestState requestState = RequestState::ready;
    int requestStateVersion = 0;
    std::string requestIdentityHash;
    bool productionDispatchEnabled = false;
    long long reservationId = 0;
    ReservationState reservationState = ReservationState::held;
    int reservationStateVersion = 0;
    long long amount = 0;
    BudgetUnit budgetUnit = BudgetUnit::materializedMemberDispatch;
    std::optional<std::string> expiresAt;
    std::string reservationIdentityHash;
    long long authorizationEventId = 0;
    long long budgetLedgerEntryId = 0;
    int budgetLedgerVersion = 0;
    bool evidenceConsistent = false;
};

template <typename Persisted>
struct PersistResult final
{
    const PersistOutcome outcome;
    const Persisted persisted;

    PersistResult(PersistOutcome outcomeValue, Persisted persistedValue)
        : outcome(outcomeValue), persisted(std::move(persistedValue))
    {
    }

    PersistResult(const PersistResult&) = default;
    PersistResult(PersistResult&&) = default;
    PersistResult& operator=(const PersistResult&) = delete;
    PersistResult& operator=(PersistResult&&) = delete;
};

bool SchemaExists(pqxx::connection& connection);
bool SchemaExists(pqxx::transaction_base& transaction);
bool BudgetRequestSchemaExists(pqxx::transaction_base& transaction);

PersistResult<PersistedOperationalCampaign> PersistOperationalCampaign(
    pqxx::transaction_base& transaction,
    const OperationalCampaign& campaign, const ActorIdentity& actor,
    const Reason& reason);

std::optional<PersistedOperationalCampaign> FindOperationalCampaign(
    pqxx::connection& connection, OperationalCampaignId campaignId);
std::optional<PersistedOperationalCampaign> FindOperationalCampaign(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);
std::optional<PersistedOperationalCampaign>
FindOperationalCampaignByMaterialization(pqxx::transaction_base& transaction,
    long long materializationId);
std::vector<PersistedOperationalCampaign> ListOperationalCampaigns(
    pqxx::transaction_base& transaction, int limit);

PersistResult<PersistedGovernanceProvenanceEvent>
PersistGovernanceProvenanceEvent(pqxx::transaction_base& transaction,
    const GovernanceProvenanceEvent& event, const ActorIdentity& actor,
    const Reason& reason);

std::optional<PersistedGovernanceProvenanceEvent>
FindGovernanceProvenanceEvent(pqxx::transaction_base& transaction,
    GovernanceProvenanceEventId provenanceEventId);
std::optional<PersistedGovernanceProvenanceEvent>
FindGovernanceProvenanceEventByRatification(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    long long ratificationEventId);

PersistResult<PersistedOperationalAuthorizationEvent>
PersistOperationalAuthorizationEvent(pqxx::transaction_base& transaction,
    const OperationalAuthorizationEvent& event);

std::optional<PersistedOperationalAuthorizationEvent>
FindOperationalAuthorizationEvent(pqxx::transaction_base& transaction,
    AuthorizationEventId authorizationEventId);
std::optional<PersistedOperationalAuthorizationEvent>
FindOperationalAuthorizationHead(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId, OperationalActionKind actionKind,
    int actionContractVersion, ScopeKind scopeKind, int scopeContractVersion);
std::vector<PersistedOperationalAuthorizationEvent>
LoadOperationalAuthorizationChain(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId, OperationalActionKind actionKind,
    int actionContractVersion, ScopeKind scopeKind, int scopeContractVersion,
    int limit);

void LockOperationalAuthorizationDomain(
    pqxx::transaction_base& transaction,
    const OperationalCampaign& campaign);
void LockBudgetDomain(pqxx::transaction_base& transaction,
    const OperationalCampaign& campaign);
PersistedOperationalCampaign LockOperationalCampaign(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);
UtcTimestamp CurrentDatabaseTime(pqxx::transaction_base& transaction);

PersistResult<PersistedBudgetLedgerEntry> PersistBudgetLedgerEntry(
    pqxx::transaction_base& transaction, const BudgetLedgerEntry& entry);
std::optional<PersistedBudgetLedgerEntry> FindBudgetLedgerEntry(
    pqxx::transaction_base& transaction,
    BudgetLedgerEntryId budgetLedgerEntryId);
std::optional<PersistedBudgetLedgerEntry> FindBudgetLedgerHead(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);
std::optional<PersistedBudgetLedgerEntry> FindBudgetLedgerEntryByVersion(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    int ledgerVersion);
BudgetAccounting LoadBudgetAccounting(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);

PersistResult<AcceptedOperationalRequest> PersistAcceptedOperationalRequest(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    const ActorIdentity& actor, const Reason& reason,
    const std::optional<UtcTimestamp>& expiresAt);
std::optional<PersistedReservation> FindReservation(
    pqxx::transaction_base& transaction, ReservationId reservationId);
std::optional<PersistedOperationalRequest> FindOperationalRequest(
    pqxx::transaction_base& transaction,
    OperationalRequestId requestId);
std::optional<PersistedOperationalRequest>
FindOperationalRequestByCampaignAction(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    OperationalActionKind actionKind, int actionContractVersion);
std::optional<PersistedReservationEvent> FindReservationEvent(
    pqxx::transaction_base& transaction,
    ReservationEventId reservationEventId);
std::optional<PersistedReservationEvent> FindReservationAcquisitionEvent(
    pqxx::transaction_base& transaction, ReservationId reservationId);
std::optional<CampaignBudgetStatusProjection>
FindCampaignBudgetStatusProjection(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId);
std::optional<OperationalRequestStatusProjection>
FindOperationalRequestStatusProjection(pqxx::transaction_base& transaction,
    OperationalRequestId requestId);

} // namespace EA::CampaignOperations
