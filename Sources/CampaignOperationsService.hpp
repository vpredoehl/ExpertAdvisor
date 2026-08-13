#pragma once

#include "CampaignOperationsRepository.hpp"

#include <functional>

#include <iosfwd>
#include <optional>
#include <string>

namespace EA::CampaignOperations
{

inline constexpr char kCampaignOperationsBudgetAdministratorRole[] =
    "campaign_operations_budget_administrator";
inline constexpr char kCampaignOperationsRequestAcceptorRole[] =
    "campaign_operations_request_acceptor";

struct BudgetAdministrationRequest
{
    long long campaignId = 0;
    int expectedLedgerVersion = 0;
    BudgetLedgerEntryKind kind = BudgetLedgerEntryKind::grant;
    std::optional<long long> value;
    std::string actorIdentity;
    std::string reason;
};

enum class BudgetAdministrationTestInjectionPoint
{
    afterDomainLocksBeforePersistence
};

using BudgetAdministrationTestInjection =
    std::function<void(BudgetAdministrationTestInjectionPoint)>;

struct OperationalRequestAcceptanceRequest
{
    long long campaignId = 0;
    std::string actorIdentity;
    std::string reason;
    std::optional<std::string> expiresAt;
};

using CampaignBudgetStatus = CampaignBudgetStatusProjection;
using OperationalRequestStatus = OperationalRequestStatusProjection;

std::string CampaignOperationsMachineText(const std::string& value);

BudgetAdministrationRequest ValidateBudgetAdministrationRequest(
    const BudgetAdministrationRequest& request);
OperationalRequestAcceptanceRequest
ValidateOperationalRequestAcceptanceRequest(
    const OperationalRequestAcceptanceRequest& request);

PersistResult<PersistedBudgetLedgerEntry> AdministerCampaignBudget(
    pqxx::connection& connection,
    const BudgetAdministrationRequest& request,
    const BudgetAdministrationTestInjection& testInjection = {});
PersistResult<AcceptedOperationalRequest> AcceptOperationalRequest(
    pqxx::connection& connection,
    const OperationalRequestAcceptanceRequest& request);
CampaignBudgetStatus LoadCampaignBudgetStatus(
    pqxx::connection& connection, OperationalCampaignId campaignId);
OperationalRequestStatus LoadOperationalRequestStatus(
    pqxx::connection& connection, OperationalRequestId requestId);

int RunCampaignBudgetAdministrationCommand(
    const std::string& connectionString,
    const BudgetAdministrationRequest& request, std::ostream& output,
    std::ostream& errors);
int RunCampaignOperationalRequestAcceptanceCommand(
    const std::string& connectionString,
    const OperationalRequestAcceptanceRequest& request,
    std::ostream& output, std::ostream& errors);
int RunCampaignBudgetStatusCommand(const std::string& connectionString,
    OperationalCampaignId campaignId, std::ostream& output,
    std::ostream& errors);
int RunCampaignOperationalRequestStatusCommand(
    const std::string& connectionString, OperationalRequestId requestId,
    std::ostream& output, std::ostream& errors);

} // namespace EA::CampaignOperations
