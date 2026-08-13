#pragma once

#include "CampaignOperationsCompletion.hpp"
#include "CampaignOperationsRepository.hpp"

#include <optional>
#include <vector>

#include <pqxx/pqxx>

namespace EA::CampaignOperations
{

struct PersistedCompletionEvent final
{
    CompletionEventId completionEventId;
    CompletionEvent event;
    std::string recordedAt;
};

struct CompletionStatus final
{
    OperationalCampaignId campaignId;
    AdministrativeCampaignState currentOperationalState;
    bool completionRecorded = false;
    std::optional<PersistedCompletionEvent> completion;
    std::vector<CompletionBlocker> blockers;
    bool postCompletionLifecycleChanged = false;
    std::string currentLifecycleEvidence;
    std::string cancellationEvidence;
    std::string reconciliationEvidence;
};

bool CompletionSchemaExists(pqxx::transaction_base& transaction);
PersistedOperationalCampaign LockCompletionDomains(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);
std::optional<PersistedCompletionEvent> FindCompletionEvent(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);
std::vector<CompletionBlocker> LoadCompletionBlockers(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);
CompletionEvent BuildCompletionEventFromAuthority(
    pqxx::transaction_base& transaction,
    const PersistedOperationalCampaign& campaign,
    const std::string& operationKey, const ActorIdentity& actor,
    const Reason& reason);
PersistedCompletionEvent PersistCompletionEvent(
    pqxx::transaction_base& transaction, const CompletionEvent& event);
CompletionStatus LoadCompletionStatusProjection(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId);

} // namespace EA::CampaignOperations
