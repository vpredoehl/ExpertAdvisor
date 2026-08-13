#include "CampaignOperationsService.hpp"

#include <chrono>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <thread>
#include <utility>

namespace EA::CampaignOperations
{
std::string CampaignOperationsMachineText(const std::string& value)
{
    std::ostringstream escaped;
    escaped.imbue(std::locale::classic());
    escaped << std::uppercase << std::hex;
    for (const unsigned char character : value)
    {
        const bool alphanumeric =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9');
        const bool safe = alphanumeric || character == '-' ||
            character == '_' || character == '.' || character == ':' ||
            character == '/' || character == ';';
        if (safe)
            escaped << static_cast<char>(character);
        else
            escaped << '%' << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(character);
    }
    return escaped.str() == "NULL" ? "%4E%55%4C%4C" : escaped.str();
}

namespace
{

long long CheckedAdd(long long left, long long right)
{
    if ((right > 0 &&
            left > std::numeric_limits<long long>::max() - right) ||
        (right < 0 &&
            left < std::numeric_limits<long long>::min() - right))
        throw Error(ErrorCode::invalidBudgetLedgerEntry,
            "campaign_operations_budget_arithmetic_overflow");
    return left + right;
}

int CommandFailureCode(const std::exception& error)
{
    const auto* domainError = dynamic_cast<const Error*>(&error);
    if (!domainError) return 1;
    if (domainError->code() == ErrorCode::persistenceConflict)
        return 2;
    if (domainError->code() == ErrorCode::authorizationDenied ||
        domainError->code() == ErrorCode::budgetDenied)
        return 3;
    return 1;
}

template <typename Function>
int RunCommandWithRetry(Function&& function, std::ostream& errors,
    const char* failurePrefix)
{
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
        try
        {
            return function();
        }
        catch (const pqxx::sql_error& error)
        {
            const bool transient = error.sqlstate() == "40001" ||
                error.sqlstate() == "40P01";
            if (transient && attempt < 3)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(attempt * 5));
                continue;
            }
            errors << failurePrefix
                   << ",status=failed,diagnostic_code=postgresql_"
                   << CampaignOperationsMachineText(error.sqlstate())
                   << ",diagnostic_detail="
                   << CampaignOperationsMachineText(error.what())
                   << '\n';
            return 1;
        }
        catch (const std::exception& error)
        {
            const auto* domainError = dynamic_cast<const Error*>(&error);
            errors << failurePrefix << ",status=failed,diagnostic_code="
                   << CampaignOperationsMachineText(domainError
                           ? error.what()
                           : "campaign_operations_unexpected_exception");
            if (!domainError)
                errors << ",diagnostic_detail="
                       << CampaignOperationsMachineText(error.what());
            errors << '\n';
            return CommandFailureCode(error);
        }
    }
    return 1;
}

} // namespace

BudgetAdministrationRequest ValidateBudgetAdministrationRequest(
    const BudgetAdministrationRequest& request)
{
    (void)OperationalCampaignId(request.campaignId);
    (void)ActorIdentity(request.actorIdentity);
    (void)Reason(request.reason);
    const bool isGrant = request.kind == BudgetLedgerEntryKind::grant;
    if ((isGrant && request.expectedLedgerVersion != 0) ||
        (!isGrant && request.expectedLedgerVersion <= 0))
        throw Error(ErrorCode::invalidBudgetLedgerEntry,
            "campaign_operations_budget_expected_version_invalid");
    if ((request.kind == BudgetLedgerEntryKind::revoke &&
            request.value.has_value()) ||
        (request.kind != BudgetLedgerEntryKind::revoke &&
            !request.value.has_value()) ||
        (request.kind == BudgetLedgerEntryKind::grant &&
            *request.value <= 0) ||
        (request.kind == BudgetLedgerEntryKind::amend &&
            *request.value == 0) ||
        (request.kind == BudgetLedgerEntryKind::supersede &&
            *request.value < 0))
        throw Error(ErrorCode::invalidBudgetLedgerEntry,
            "campaign_operations_budget_value_invalid");
    return request;
}

OperationalRequestAcceptanceRequest
ValidateOperationalRequestAcceptanceRequest(
    const OperationalRequestAcceptanceRequest& request)
{
    (void)OperationalCampaignId(request.campaignId);
    (void)ActorIdentity(request.actorIdentity);
    (void)Reason(request.reason);
    if (request.expiresAt) (void)UtcTimestamp(*request.expiresAt);
    return request;
}

PersistResult<PersistedBudgetLedgerEntry> AdministerCampaignBudget(
    pqxx::connection& connection,
    const BudgetAdministrationRequest& request,
    const BudgetAdministrationTestInjection& testInjection)
{
    const auto validated = ValidateBudgetAdministrationRequest(request);
    pqxx::work transaction{connection};
    if (!SchemaExists(transaction) ||
        !BudgetRequestSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_budget_request_schema_required");
    const OperationalCampaignId campaignId(validated.campaignId);
    const auto campaign = FindOperationalCampaign(transaction, campaignId);
    if (!campaign)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_campaign_not_found");
    LockBudgetDomain(transaction, campaign->campaign);
    if (testInjection)
        testInjection(BudgetAdministrationTestInjectionPoint::
            afterDomainLocksBeforePersistence);

    std::optional<PersistedBudgetLedgerEntry> predecessor;
    if (validated.expectedLedgerVersion > 0)
    {
        auto found = FindBudgetLedgerEntryByVersion(transaction, campaignId,
            validated.expectedLedgerVersion);
        if (!found)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_budget_expected_head_missing");
        predecessor.emplace(std::move(*found));
    }
    const long long priorTotal =
        predecessor ? predecessor->entry.resultingTotal : 0;
    long long delta = 0;
    long long resultingTotal = 0;
    BudgetLedgerStatus status = BudgetLedgerStatus::active;
    switch (validated.kind)
    {
        case BudgetLedgerEntryKind::grant:
            resultingTotal = *validated.value;
            delta = resultingTotal;
            break;
        case BudgetLedgerEntryKind::amend:
            delta = *validated.value;
            resultingTotal = CheckedAdd(priorTotal, delta);
            break;
        case BudgetLedgerEntryKind::revoke:
        {
            const auto accounting =
                LoadBudgetAccounting(transaction, campaignId);
            resultingTotal = accounting.committed + accounting.held;
            delta = resultingTotal - priorTotal;
            status = BudgetLedgerStatus::revoked;
            break;
        }
        case BudgetLedgerEntryKind::supersede:
            resultingTotal = *validated.value;
            delta = resultingTotal - priorTotal;
            break;
    }
    BudgetLedgerEntry entry = BuildBudgetLedgerEntry(campaignId,
        campaign->campaign.identity.canonicalText(),
        predecessor
            ? std::optional<BudgetLedgerEntryId>(
                  predecessor->budgetLedgerEntryId)
            : std::nullopt,
        predecessor
            ? std::optional<std::string>(
                  predecessor->entry.identity.canonicalText())
            : std::nullopt,
        predecessor
            ? std::optional<std::string>(
                  predecessor->entry.identity.hash())
            : std::nullopt,
        validated.expectedLedgerVersion + 1, validated.kind, status,
        BudgetUnit::materializedMemberDispatch, delta, priorTotal,
        resultingTotal, ActorIdentity(validated.actorIdentity),
        Reason(validated.reason));
    auto result = PersistBudgetLedgerEntry(transaction, entry);
    transaction.commit();
    return result;
}

PersistResult<AcceptedOperationalRequest> AcceptOperationalRequest(
    pqxx::connection& connection,
    const OperationalRequestAcceptanceRequest& request)
{
    const auto validated =
        ValidateOperationalRequestAcceptanceRequest(request);
    pqxx::work transaction{connection};
    if (!SchemaExists(transaction) ||
        !BudgetRequestSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_budget_request_schema_required");
    const bool controlSchema = transaction.exec(
        "SELECT to_regclass('campaign_operations_control_event') "
        "IS NOT NULL AND to_regclass("
        "'campaign_operations_cancellation_request') IS NOT NULL;")
        .one_row()[0].as<bool>();
    if (controlSchema)
    {
        const auto allowed = transaction.exec(
            "SELECT campaign_operations_future_actions_allowed($1);",
            pqxx::params{validated.campaignId}).one_row()[0].as<bool>();
        if (!allowed)
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_request_control_blocked");
    }
    auto result = PersistAcceptedOperationalRequest(transaction,
        OperationalCampaignId(validated.campaignId),
        ActorIdentity(validated.actorIdentity), Reason(validated.reason),
        validated.expiresAt
            ? std::optional<UtcTimestamp>(
                  UtcTimestamp(*validated.expiresAt))
            : std::nullopt);
    transaction.commit();
    return result;
}

CampaignBudgetStatus LoadCampaignBudgetStatus(
    pqxx::connection& connection, OperationalCampaignId campaignId)
{
    pqxx::transaction<pqxx::isolation_level::repeatable_read,
        pqxx::write_policy::read_only> transaction{connection};
    if (!SchemaExists(transaction) ||
        !BudgetRequestSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_budget_request_schema_required");
    auto status =
        FindCampaignBudgetStatusProjection(transaction, campaignId);
    if (!status)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_campaign_not_found");
    if (!status->accountingConsistent)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_reservation_reconciliation_required");
    return std::move(*status);
}

OperationalRequestStatus LoadOperationalRequestStatus(
    pqxx::connection& connection, OperationalRequestId requestId)
{
    pqxx::transaction<pqxx::isolation_level::repeatable_read,
        pqxx::write_policy::read_only> transaction{connection};
    if (!SchemaExists(transaction) ||
        !BudgetRequestSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_budget_request_schema_required");
    auto status =
        FindOperationalRequestStatusProjection(transaction, requestId);
    if (!status)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_request_not_found");
    if (!status->evidenceConsistent)
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_reservation_reconciliation_required");
    return std::move(*status);
}

int RunCampaignBudgetAdministrationCommand(
    const std::string& connectionString,
    const BudgetAdministrationRequest& request, std::ostream& output,
    std::ostream& errors)
{
    return RunCommandWithRetry([&]
    {
        pqxx::connection connection{connectionString};
        const auto result = AdministerCampaignBudget(connection, request);
        output << "CAMPAIGN_OPERATIONS_BUDGET"
               << ",campaign_id=" << result.persisted.entry.campaignId.value()
               << ",budget_ledger_entry_id="
               << result.persisted.budgetLedgerEntryId.value()
               << ",ledger_version=" << result.persisted.entry.ledgerVersion
               << ",entry_kind="
               << CampaignOperationsMachineText(
                      ToText(result.persisted.entry.entryKind))
               << ",status="
               << CampaignOperationsMachineText(
                      ToText(result.persisted.entry.status))
               << ",resulting_total="
               << result.persisted.entry.resultingTotal
               << ",identity_hash="
               << CampaignOperationsMachineText(
                      result.persisted.entry.identity.hash())
               << ",replay_disposition="
               << CampaignOperationsMachineText(ToText(result.outcome))
               << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_BUDGET");
}

int RunCampaignOperationalRequestAcceptanceCommand(
    const std::string& connectionString,
    const OperationalRequestAcceptanceRequest& request,
    std::ostream& output, std::ostream& errors)
{
    return RunCommandWithRetry([&]
    {
        pqxx::connection connection{connectionString};
        const auto result = AcceptOperationalRequest(connection, request);
        output << "CAMPAIGN_OPERATIONS_REQUEST_ACCEPTED"
               << ",operational_campaign_id="
               << result.persisted.request.request.logicalOperation
                      .campaignId.value()
               << ",authorization_event_id="
               << result.persisted.request.request
                      .acceptingAuthorizationEventId.value()
               << ",budget_ledger_entry_id="
               << result.persisted.reservation.reservation
                      .budgetLedgerEntryId.value()
               << ",budget_ledger_version="
               << result.persisted.reservation.reservation
                      .budgetLedgerVersion
               << ",reservation_id="
               << result.persisted.reservation.reservationId.value()
               << ",reservation_event_id="
               << result.persisted.acquisitionEvent.reservationEventId.value()
               << ",operational_request_id="
               << result.persisted.request.requestId.value()
               << ",request_state="
               << CampaignOperationsMachineText(
                      ToText(result.persisted.request.state))
               << ",request_state_version="
               << result.persisted.request.stateVersion
               << ",reservation_state="
               << CampaignOperationsMachineText(
                      ToText(result.persisted.reservation.state))
               << ",reservation_state_version="
               << result.persisted.reservation.stateVersion
               << ",amount="
               << result.persisted.reservation.reservation.amount
               << ",production_dispatch_enabled="
               << (result.persisted.request.productionDispatchEnabled
                       ? "true"
                       : "false")
               << ",request_identity_hash="
               << CampaignOperationsMachineText(
                      result.persisted.request.request.identity.hash())
               << ",reservation_identity_hash="
               << CampaignOperationsMachineText(
                      result.persisted.reservation.reservation.identity.hash())
               << ",acquisition_event_identity_hash="
               << CampaignOperationsMachineText(
                      result.persisted.acquisitionEvent.event.identity.hash())
               << ",replay_disposition="
               << CampaignOperationsMachineText(ToText(result.outcome))
               << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_REQUEST");
}

int RunCampaignBudgetStatusCommand(const std::string& connectionString,
    OperationalCampaignId campaignId, std::ostream& output,
    std::ostream& errors)
{
    return RunCommandWithRetry([&]
    {
        pqxx::connection connection{connectionString};
        const auto status = LoadCampaignBudgetStatus(connection, campaignId);
        output << "CAMPAIGN_OPERATIONS_BUDGET_STATUS"
               << ",campaign_id=" << campaignId.value();
        if (!status.ledgerVersion)
            output << ",ledger_version=null,status=unbudgeted"
                   << ",granted=0,ever_reserved=0,committed=0,held=0,"
                      "released_or_expired=0,reservable=0";
        else
            output << ",ledger_version=" << *status.ledgerVersion
                   << ",status="
                   << CampaignOperationsMachineText(ToText(*status.status))
                   << ",granted=" << status.granted
                   << ",ever_reserved=" << status.everReserved
                   << ",committed=" << status.committed
                   << ",held=" << status.held
                   << ",released_or_expired="
                   << status.releasedOrExpired
                   << ",reservable=" << status.reservable;
        output << ",accounting_consistent="
               << (status.accountingConsistent ? "true" : "false") << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_BUDGET_STATUS");
}

int RunCampaignOperationalRequestStatusCommand(
    const std::string& connectionString, OperationalRequestId requestId,
    std::ostream& output, std::ostream& errors)
{
    return RunCommandWithRetry([&]
    {
        pqxx::connection connection{connectionString};
        const auto status =
            LoadOperationalRequestStatus(connection, requestId);
        output << "CAMPAIGN_OPERATIONS_REQUEST_STATUS"
               << ",operational_request_id=" << status.requestId
               << ",operational_campaign_id=" << status.campaignId
               << ",request_state="
               << CampaignOperationsMachineText(ToText(status.requestState))
               << ",request_state_version=" << status.requestStateVersion
               << ",production_dispatch_enabled="
               << (status.productionDispatchEnabled ? "true" : "false")
               << ",request_identity_hash="
               << CampaignOperationsMachineText(status.requestIdentityHash)
               << ",authorization_event_id="
               << status.authorizationEventId
               << ",budget_ledger_entry_id="
               << status.budgetLedgerEntryId
               << ",budget_ledger_version="
               << status.budgetLedgerVersion
               << ",reservation_id=" << status.reservationId
               << ",reservation_state="
               << CampaignOperationsMachineText(
                      ToText(status.reservationState))
               << ",reservation_state_version="
               << status.reservationStateVersion
               << ",reservation_amount=" << status.amount
               << ",reservation_budget_unit="
               << CampaignOperationsMachineText(ToText(status.budgetUnit))
               << ",reservation_expires_at="
               << (status.expiresAt
                       ? CampaignOperationsMachineText(*status.expiresAt)
                       : "NULL")
               << ",reservation_identity_hash="
               << CampaignOperationsMachineText(
                      status.reservationIdentityHash)
               << ",evidence_consistent="
               << (status.evidenceConsistent ? "true" : "false") << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_REQUEST_STATUS");
}

} // namespace EA::CampaignOperations
