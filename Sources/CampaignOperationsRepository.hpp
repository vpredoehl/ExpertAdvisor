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

} // namespace EA::CampaignOperations
