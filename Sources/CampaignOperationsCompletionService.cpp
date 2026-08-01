#include "CampaignOperationsCompletionService.hpp"

#include "CampaignOperationsControl.hpp"
#include "CampaignOperationsService.hpp"

#include <algorithm>
#include <chrono>
#include <ostream>
#include <sstream>
#include <thread>

namespace EA::CampaignOperations
{
namespace
{

void SetRole(pqxx::transaction_base& transaction, const char* role)
{
    transaction.exec(
        "SET LOCAL ROLE " + transaction.quote_name(role) + ";");
}

void RequireCompletionSchema(pqxx::transaction_base& transaction)
{
    if (!CompletionSchemaExists(transaction))
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_completion_schema_required");
}

bool ValidOperationKey(const std::string& value)
{
    return !value.empty() && value.size() <= 128U &&
        std::all_of(value.begin(), value.end(), [](unsigned char c)
        {
            return (c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                c == '.' || c == ':' || c == '/';
        });
}

int FailureCode(const std::exception& error)
{
    const auto* domain = dynamic_cast<const Error*>(&error);
    if (domain && domain->code() == ErrorCode::persistenceConflict)
        return 2;
    return 1;
}

template <typename Function>
int RunCommand(Function&& function, std::ostream& errors,
    const char* prefix)
{
    try
    {
        return function();
    }
    catch (const pqxx::sql_error& error)
    {
        errors << prefix
               << ",status=failed,diagnostic_code=postgresql_"
               << CampaignOperationsMachineText(error.sqlstate())
               << ",diagnostic_detail="
               << CampaignOperationsMachineText(error.what()) << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        errors << prefix << ",status=failed,diagnostic_code="
               << CampaignOperationsMachineText(error.what()) << '\n';
        return FailureCode(error);
    }
}

std::string BlockerText(const std::vector<CompletionBlocker>& blockers)
{
    if (blockers.empty()) return "none";
    std::ostringstream output;
    bool first = true;
    for (const auto& blocker : blockers)
    {
        if (!first) output << ';';
        first = false;
        output << blocker.code << ':' << blocker.detail << ':'
               << blocker.blockerClass;
    }
    return output.str();
}

CompleteIfSettledResult CompleteCampaignIfSettledOnce(
    pqxx::connection& connection, const CompleteIfSettledRequest& validated,
    std::optional<CompletionEvent>* attemptedEvent,
    const CompletionTestHook& testHook)
{
    // PostgreSQL SERIALIZABLE includes the accepted repeatable-read snapshot
    // and additionally rejects child-row write skew (for example a
    // cancellation intent or reconciliation observation inserted while this
    // transaction waits on the campaign lock).
    pqxx::transaction<pqxx::isolation_level::serializable> transaction{
        connection};
    SetRole(transaction, kCampaignOperationsCompletionWriterRole);
    RequireCompletionSchema(transaction);
    const OperationalCampaignId campaignId(validated.campaignId);
    const auto campaign = LockCompletionDomains(transaction, campaignId);
    if (testHook)
        testHook(CompletionTestInjectionPoint::afterLocksBeforeEvidence);
    const auto existing = FindCompletionEvent(transaction, campaignId);
    auto blockers = LoadCompletionBlockers(transaction, campaignId);
    if (existing)
    {
        if (!blockers.empty())
        {
            transaction.commit();
            return {CompletionAttemptDisposition::conflictingReplay,
                existing, std::move(blockers)};
        }
        const auto candidate = BuildCompletionEventFromAuthority(transaction,
            campaign, validated.operationKey,
            ActorIdentity(validated.actorIdentity), Reason(validated.reason));
        const bool identical = existing->event == candidate;
        transaction.commit();
        return {identical
                    ? CompletionAttemptDisposition::existingIdentical
                    : CompletionAttemptDisposition::conflictingReplay,
            existing, {}};
    }
    if (!blockers.empty())
    {
        transaction.commit();
        return {CompletionAttemptDisposition::blocked, std::nullopt,
            std::move(blockers)};
    }
    const auto event = BuildCompletionEventFromAuthority(transaction,
        campaign, validated.operationKey,
        ActorIdentity(validated.actorIdentity), Reason(validated.reason));
    if (attemptedEvent) attemptedEvent->emplace(event);
    auto persisted = PersistCompletionEvent(transaction, event);
    if (testHook)
        testHook(CompletionTestInjectionPoint::beforeCommit);
    transaction.commit();
    if (testHook)
        testHook(CompletionTestInjectionPoint::afterCommitBeforeResponse);
    return {CompletionAttemptDisposition::recorded,
        std::move(persisted), {}};
}

std::optional<CompleteIfSettledResult> LookupCompletionOutcome(
    pqxx::connection& connection, const CompleteIfSettledRequest& validated,
    const std::optional<CompletionEvent>& attemptedEvent,
    const CompletionTestHook& testHook)
{
    pqxx::transaction<pqxx::isolation_level::serializable> transaction{
        connection};
    SetRole(transaction, kCampaignOperationsCompletionWriterRole);
    RequireCompletionSchema(transaction);
    const OperationalCampaignId campaignId(validated.campaignId);
    const auto campaign = LockCompletionDomains(transaction, campaignId);
    if (testHook)
        testHook(CompletionTestInjectionPoint::duringOutcomeLookup);
    const auto existing = FindCompletionEvent(
        transaction, campaignId);
    if (!existing)
    {
        transaction.commit();
        return std::nullopt;
    }
    auto blockers = LoadCompletionBlockers(transaction, campaignId);
    bool identical = false;
    if (blockers.empty())
    {
        const auto candidate = BuildCompletionEventFromAuthority(
            transaction, campaign, validated.operationKey,
            ActorIdentity(validated.actorIdentity),
            Reason(validated.reason));
        identical = existing->event == candidate &&
            (!attemptedEvent || existing->event == *attemptedEvent);
    }
    CompleteIfSettledResult result{
        identical ? CompletionAttemptDisposition::existingIdentical
                  : CompletionAttemptDisposition::conflictingReplay,
        existing, std::move(blockers)};
    transaction.commit();
    return result;
}

bool RetryableSqlState(const pqxx::sql_error& error)
{
    return error.sqlstate() == "40001" || error.sqlstate() == "40P01" ||
        error.sqlstate() == "23505";
}

} // namespace

CompleteIfSettledRequest ValidateCompleteIfSettledRequest(
    const CompleteIfSettledRequest& request)
{
    (void)OperationalCampaignId(request.campaignId);
    (void)ActorIdentity(request.actorIdentity);
    (void)Reason(request.reason);
    if (!ValidOperationKey(request.operationKey))
        throw Error(ErrorCode::invalidCompletionEvidence,
            "campaign_operations_completion_operation_key_invalid");
    return request;
}

CompleteIfSettledResult CompleteCampaignIfSettled(
    pqxx::connection& connection, const CompleteIfSettledRequest& request)
{
    const auto validated = ValidateCompleteIfSettledRequest(request);
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
        try
        {
            return CompleteCampaignIfSettledOnce(
                connection, validated, nullptr, {});
        }
        catch (const pqxx::in_doubt_error&)
        {
            throw Error(ErrorCode::persistenceConflict,
                "campaign_operations_completion_outcome_ambiguous");
        }
        catch (const pqxx::sql_error& error)
        {
            if (!RetryableSqlState(error)) throw;
            if (error.sqlstate() == "23505")
            {
                if (const auto recovered = LookupCompletionOutcome(
                        connection, validated, std::nullopt, {}))
                    return *recovered;
            }
            if (attempt == 3) throw;
            std::this_thread::sleep_for(
                std::chrono::milliseconds(attempt * 5));
        }
    }
    throw Error(ErrorCode::persistenceConflict,
        "campaign_operations_completion_retry_exhausted");
}

CompleteIfSettledResult CompleteCampaignIfSettled(
    const CompletionConnectionFactory& connectionFactory,
    const CompleteIfSettledRequest& request, CompletionTestHook testHook)
{
    if (!connectionFactory)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_completion_connection_factory_required");
    const auto validated = ValidateCompleteIfSettledRequest(request);
    std::optional<CompletionEvent> attemptedEvent;
    bool outcomeMustBeProven = false;
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
        try
        {
            auto connection = connectionFactory();
            if (!connection || !connection->is_open())
                throw pqxx::broken_connection(
                    "campaign operations completion connection unavailable");
            if (const auto recovered = LookupCompletionOutcome(
                    *connection, validated, attemptedEvent, testHook))
                return *recovered;
            // A successful lookup proves that no prior append committed.  A
            // complete new transaction must rebuild every canonical from its
            // new serializable snapshot before another append attempt.
            attemptedEvent.reset();
            outcomeMustBeProven = false;
            return CompleteCampaignIfSettledOnce(
                *connection, validated, &attemptedEvent, testHook);
        }
        catch (const pqxx::in_doubt_error&)
        {
            // libpqxx uses this distinct exception for a commit whose durable
            // outcome is unknown.  Discard the connection and require a new
            // connection to prove the canonical row present or absent before
            // any complete append transaction can be retried.
            outcomeMustBeProven = true;
            if (attempt != 3)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(attempt * 5));
        }
        catch (const pqxx::broken_connection&)
        {
            outcomeMustBeProven = true;
            if (attempt != 3)
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(attempt * 5));
        }
        catch (const pqxx::sql_error& error)
        {
            if (!RetryableSqlState(error)) throw;
            if (error.sqlstate() == "23505")
                outcomeMustBeProven = true;
            else if (attempt == 3)
                throw;
            std::this_thread::sleep_for(
                std::chrono::milliseconds(attempt * 5));
        }
    }
    if (outcomeMustBeProven)
    {
        try
        {
            auto connection = connectionFactory();
            if (connection && connection->is_open())
            {
                if (const auto recovered = LookupCompletionOutcome(
                        *connection, validated, attemptedEvent, testHook))
                    return *recovered;
                throw Error(ErrorCode::persistenceConflict,
                    "campaign_operations_completion_retry_exhausted");
            }
        }
        catch (const Error&)
        {
            throw;
        }
        catch (const pqxx::in_doubt_error&)
        {
        }
        catch (const pqxx::broken_connection&)
        {
        }
        catch (const pqxx::sql_error& error)
        {
            if (!RetryableSqlState(error)) throw;
        }
    }
    throw Error(ErrorCode::persistenceConflict,
        outcomeMustBeProven
            ? "campaign_operations_completion_outcome_ambiguous"
            : "campaign_operations_completion_retry_exhausted");
}

CompletionStatus LoadCampaignCompletionStatus(
    pqxx::connection& connection, OperationalCampaignId campaignId)
{
    pqxx::read_transaction transaction{connection};
    transaction.exec(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    SetRole(transaction, kCampaignOperationsReaderRole);
    RequireCompletionSchema(transaction);
    return LoadCompletionStatusProjection(transaction, campaignId);
}

int RunCompleteCampaignIfSettledCommand(
    const std::string& connectionString,
    const CompleteIfSettledRequest& request, std::ostream& output,
    std::ostream& errors)
{
    return RunCommand([&]
    {
        const auto result = CompleteCampaignIfSettled(
            [&connectionString]
            {
                return std::make_unique<pqxx::connection>(connectionString);
            },
            request);
        output << "CAMPAIGN_OPERATIONS_COMPLETION"
               << ",operational_campaign_id=" << request.campaignId
               << ",disposition="
               << CampaignOperationsMachineText(ToText(result.disposition))
               << ",completion_event_id="
               << (result.completion
                       ? std::to_string(
                             result.completion->completionEventId.value())
                       : "NULL")
               << ",terminal_state="
               << (result.completion
                       ? CampaignOperationsMachineText(ToText(
                             result.completion->event.terminalState))
                       : "NULL")
               << ",classification="
               << (result.completion
                       ? CampaignOperationsMachineText(ToText(
                             result.completion->event.classification))
                       : "NULL")
               << ",blockers="
               << CampaignOperationsMachineText(BlockerText(result.blockers))
               << '\n';
        if (result.disposition == CompletionAttemptDisposition::blocked)
            return 3;
        return result.disposition ==
                CompletionAttemptDisposition::conflictingReplay
            ? 4 : 0;
    }, errors, "CAMPAIGN_OPERATIONS_COMPLETION");
}

int RunCampaignCompletionStatusCommand(
    const std::string& connectionString, OperationalCampaignId campaignId,
    std::ostream& output, std::ostream& errors)
{
    return RunCommand([&]
    {
        pqxx::connection connection{connectionString};
        const auto status =
            LoadCampaignCompletionStatus(connection, campaignId);
        output << "CAMPAIGN_OPERATIONS_COMPLETION_STATUS"
               << ",operational_campaign_id=" << campaignId.value()
               << ",operational_state="
               << CampaignOperationsMachineText(
                      ToText(status.currentOperationalState))
               << ",completion_recorded="
               << (status.completionRecorded ? "true" : "false")
               << ",logically_archived="
               << (status.completionRecorded ? "true" : "false")
               << ",completion_event_id="
               << (status.completion
                       ? std::to_string(
                             status.completion->completionEventId.value())
                       : "NULL")
               << ",recorded_at="
               << (status.completion
                       ? CampaignOperationsMachineText(
                             status.completion->recordedAt)
                       : "NULL")
               << ",recorded_terminal_state="
               << (status.completion
                       ? CampaignOperationsMachineText(ToText(
                             status.completion->event.terminalState))
                       : "NULL")
               << ",completion_classification="
               << (status.completion
                       ? CampaignOperationsMachineText(ToText(
                             status.completion->event.classification))
                       : "NULL")
               << ",completion_identity_hash="
               << (status.completion
                       ? status.completion->event.identity.hash()
                       : "NULL")
               << ",post_completion_lifecycle_changed="
               << (status.postCompletionLifecycleChanged
                       ? "true" : "false")
               << ",blockers="
               << CampaignOperationsMachineText(
                      BlockerText(status.blockers))
               << ",current_lifecycle_evidence="
               << CampaignOperationsMachineText(
                      status.currentLifecycleEvidence)
               << ",cancellation_evidence="
               << CampaignOperationsMachineText(
                      status.cancellationEvidence)
               << ",reconciliation_evidence="
               << CampaignOperationsMachineText(
                      status.reconciliationEvidence)
               << ",scientific_outcome=NOT_AUTHORITATIVE_NOT_EVALUATED"
               << '\n';
        return 0;
    }, errors, "CAMPAIGN_OPERATIONS_COMPLETION_STATUS");
}

} // namespace EA::CampaignOperations
