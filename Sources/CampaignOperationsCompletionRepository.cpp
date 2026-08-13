#include "CampaignOperationsCompletionRepository.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

CompletionClassification ParseCompletionClassification(
    const std::string& value)
{
    for (const auto candidate : {
             CompletionClassification::operationalRequestFailed,
             CompletionClassification::mixedTerminalOutcomes,
             CompletionClassification::downstreamFailure,
             CompletionClassification::terminalPartialCompletion,
             CompletionClassification::allScopeCancelled,
             CompletionClassification::allDownstreamCompleted})
        if (ToText(candidate) == value) return candidate;
    throw Error(ErrorCode::persistenceCorruption,
        "campaign_operations_completion_classification_corrupt");
}

CanonicalIdentity EvidenceIdentity(const pqxx::row& row,
    pqxx::row::size_type canonicalColumn, pqxx::row::size_type hashColumn)
{
    return CanonicalIdentity::Hydrate(
        kCampaignOperationsCompletionContractVersion,
        row[canonicalColumn].as<std::string>(),
        row[hashColumn].as<std::string>());
}

CompletionEvent MapCompletionEvent(const pqxx::row& row)
{
    auto event = BuildCompletionEvent(
        OperationalCampaignId(row[1].as<long long>()),
        row[2].as<std::string>(), row[3].as<std::string>(),
        AdministrativeCampaignStateFromText(row[4].as<std::string>()),
        ParseCompletionClassification(row[5].as<std::string>()),
        BudgetLedgerEntryId(row[6].as<long long>()), row[7].as<int>(),
        row[8].as<long long>(), row[9].as<long long>(),
        row[10].as<long long>(), row[11].as<long long>(),
        row[12].as<long long>(), row[13].as<long long>(),
        row[14].as<int>(), row[15].as<int>(), row[16].as<int>(),
        row[17].as<int>(), row[18].as<int>(), row[19].as<int>(),
        row[20].as<int>(), row[21].as<int>(), row[22].as<int>(),
        row[23].as<int>(), row[24].as<int>(),
        EvidenceIdentity(row, 25, 26), EvidenceIdentity(row, 27, 28),
        EvidenceIdentity(row, 29, 30), EvidenceIdentity(row, 31, 32),
        EvidenceIdentity(row, 33, 34), EvidenceIdentity(row, 35, 36),
        EvidenceIdentity(row, 37, 38), EvidenceIdentity(row, 39, 40),
        ActorIdentity(row[41].as<std::string>()),
        Reason(row[42].as<std::string>()));
    if (row[43].as<std::string>() !=
            kCampaignOperationsCompletionWriterRole ||
        row[44].as<int>() != kCampaignOperationsCompletionContractVersion ||
        row[45].as<std::string>() != event.identity.canonicalText() ||
        row[46].as<std::string>() != event.identity.hash())
        throw Error(ErrorCode::persistenceCorruption,
            "campaign_operations_completion_event_corrupt");
    return event;
}

std::string CompletionColumns()
{
    return
        "completion_event_id,operational_campaign_id,"
        "campaign_identity_canonical,operation_key,"
        "administrative_terminal_state,completion_classification,"
        "budget_ledger_entry_id,budget_ledger_version,"
        "budget_resulting_total,budget_ever_reserved,budget_committed,"
        "budget_released_or_expired,budget_held,budget_unallocated,"
        "scope_member_count,completed_member_count,failed_member_count,"
        "cancelled_or_never_dispatched_member_count,reservation_count,"
        "request_count,binding_count,control_owner_count,"
        "cancellation_request_count,cancellation_settlement_count,"
        "unresolved_blocking_observation_count,"
        "authorization_evidence_canonical,authorization_evidence_hash,"
        "budget_evidence_canonical,budget_evidence_hash,"
        "reservation_evidence_canonical,reservation_evidence_hash,"
        "request_evidence_canonical,request_evidence_hash,"
        "binding_evidence_canonical,binding_evidence_hash,"
        "lifecycle_evidence_canonical,lifecycle_evidence_hash,"
        "cancellation_evidence_canonical,cancellation_evidence_hash,"
        "reconciliation_evidence_canonical,reconciliation_evidence_hash,"
        "actor_identity,reason,capability,completion_contract_version,"
        "completion_identity_canonical,completion_identity_hash,"
        "to_char(recorded_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"')";
}

std::string LoadEvidenceText(pqxx::transaction_base& transaction,
    OperationalCampaignId campaignId, const char* kind)
{
    return transaction.exec(
        "SELECT campaign_operations_completion_evidence_text($1,$2);",
        pqxx::params{campaignId.value(), kind})
        .one_row()[0].as<std::string>();
}

AdministrativeCampaignState DeriveNonterminalState(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId,
    const std::vector<CompletionBlocker>& blockers)
{
    if (std::any_of(blockers.begin(), blockers.end(),
            [](const CompletionBlocker& blocker)
            {
                return blocker.blockerClass == "inconsistent";
            }))
        return AdministrativeCampaignState::inconsistent;
    if (std::any_of(blockers.begin(), blockers.end(),
            [](const CompletionBlocker& blocker)
            {
                return blocker.blockerClass == "reconciliation_required";
            }))
        return AdministrativeCampaignState::reconciliationRequired;
    const pqxx::row row = transaction.exec(
        "SELECT "
        "COALESCE((SELECT event_kind FROM "
        "campaign_operations_control_event WHERE operational_campaign_id=$1 "
        "ORDER BY control_version DESC LIMIT 1),'none'),"
        "(SELECT count(*) FROM campaign_operations_cancellation_request "
        "WHERE operational_campaign_id=$1),"
        "(SELECT count(*) FROM campaign_operations_cancellation_request c "
        "LEFT JOIN campaign_operations_cancellation_settlement s "
        "USING(cancellation_request_id) "
        "WHERE c.operational_campaign_id=$1 "
        "AND s.cancellation_settlement_id IS NULL),"
        "COALESCE((SELECT request_state FROM "
        "campaign_operations_operational_request "
        "WHERE operational_campaign_id=$1 "
        "ORDER BY operational_request_id LIMIT 1),'none'),"
        "(SELECT count(*) FROM campaign_operations_budget_ledger_entry "
        "WHERE operational_campaign_id=$1),"
        "(SELECT count(*) FROM campaign_operations_authorization_event "
        "WHERE operational_campaign_id=$1);",
        pqxx::params{campaignId.value()}).one_row();
    const int cancellations = row[1].as<int>();
    if (cancellations > 0)
        return row[2].as<int>() > 0
            ? AdministrativeCampaignState::cancelling
            : AdministrativeCampaignState::cancellationRequested;
    if (row[0].as<std::string>() == "pause")
        return AdministrativeCampaignState::paused;
    const std::string requestState = row[3].as<std::string>();
    if (requestState == "dispatching")
        return AdministrativeCampaignState::dispatching;
    if (requestState == "bound")
        return AdministrativeCampaignState::active;
    if (requestState == "ready")
        return AdministrativeCampaignState::ready;
    if (row[4].as<int>() > 0)
        return AdministrativeCampaignState::budgeted;
    if (row[5].as<int>() > 0)
        return AdministrativeCampaignState::authorized;
    return AdministrativeCampaignState::awaitingOperationalAuthorization;
}

} // namespace

bool CompletionSchemaExists(pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('campaign_operations_completion_event') "
        "IS NOT NULL AND to_regclass("
        "'campaign_operations_completion_audit_reference_event') "
        "IS NOT NULL AND to_regclass("
        "'campaign_operations_completion_status_v1') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

PersistedOperationalCampaign LockCompletionDomains(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    const auto campaign = FindOperationalCampaign(transaction, campaignId);
    if (!campaign)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_campaign_not_found");
    std::array<std::string, 2> authorizationActions{
        "adopt_existing_pending_and_control",
        "dispatch_full_materialization"};
    std::sort(authorizationActions.begin(), authorizationActions.end());
    for (const auto& action : authorizationActions)
    {
        const std::string domain = campaign->campaign.identity.canonicalText() +
            ";action=" + action + ";scope=" +
            ToText(campaign->campaign.scopeKind);
        transaction.exec(
            "SELECT pg_advisory_xact_lock(hashtextextended($1,"
            "1179402835030003));", pqxx::params{domain});
        transaction.exec(
            "SELECT lock_campaign_operations_authorization_head($1,$2);",
            pqxx::params{campaignId.value(), action});
    }
    LockBudgetDomain(transaction, campaign->campaign);
    transaction.exec(
        "SELECT lock_campaign_operations_budget_head($1);",
        pqxx::params{campaignId.value()});
    auto lockedCampaign = LockOperationalCampaign(transaction, campaignId);
    const auto reservations = transaction.exec(
        "SELECT reservation_id FROM campaign_operations_reservation "
        "WHERE operational_campaign_id=$1 ORDER BY reservation_id;",
        pqxx::params{campaignId.value()});
    for (const auto& row : reservations)
        transaction.exec(
            "SELECT lock_campaign_operations_reservation($1);",
            pqxx::params{row[0].as<long long>()});
    const auto requests = transaction.exec(
        "SELECT operational_request_id FROM "
        "campaign_operations_operational_request "
        "WHERE operational_campaign_id=$1 ORDER BY operational_request_id;",
        pqxx::params{campaignId.value()});
    for (const auto& row : requests)
        transaction.exec(
            "SELECT lock_campaign_operations_request($1);",
            pqxx::params{row[0].as<long long>()});
    return lockedCampaign;
}

std::optional<PersistedCompletionEvent> FindCompletionEvent(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    const auto rows = transaction.exec(
        "SELECT " + CompletionColumns() +
        " FROM campaign_operations_completion_event "
        "WHERE operational_campaign_id=$1;",
        pqxx::params{campaignId.value()});
    if (rows.empty()) return std::nullopt;
    return PersistedCompletionEvent{
        CompletionEventId(rows.one_row()[0].as<long long>()),
        MapCompletionEvent(rows.one_row()),
        rows.one_row()[47].as<std::string>()};
}

std::vector<CompletionBlocker> LoadCompletionBlockers(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    const auto rows = transaction.exec(
        "SELECT blocker_code,blocker_detail,blocker_class "
        "FROM campaign_operations_completion_blockers($1) "
        "ORDER BY blocker_class,blocker_code,blocker_detail;",
        pqxx::params{campaignId.value()});
    std::vector<CompletionBlocker> result;
    result.reserve(rows.size());
    for (const auto& row : rows)
        result.push_back({row[0].as<std::string>(),
            row[1].as<std::string>(), row[2].as<std::string>()});
    return result;
}

CompletionEvent BuildCompletionEventFromAuthority(
    pqxx::transaction_base& transaction,
    const PersistedOperationalCampaign& campaign,
    const std::string& operationKey, const ActorIdentity& actor,
    const Reason& reason)
{
    const auto classified = transaction.exec(
        "SELECT * FROM campaign_operations_completion_classification($1);",
        pqxx::params{campaign.campaignId.value()}).one_row();
    const auto budget = transaction.exec(
        "SELECT head.budget_ledger_entry_id,head.ledger_version,"
        "head.resulting_total,"
        "COALESCE(sum(reservation.amount),0)::bigint,"
        "COALESCE(sum(reservation.amount) FILTER (WHERE "
        "reservation.reservation_state='committed'),0)::bigint,"
        "COALESCE(sum(reservation.amount) FILTER (WHERE "
        "reservation.reservation_state IN ('released','expired')),0)::bigint,"
        "COALESCE(sum(reservation.amount) FILTER (WHERE "
        "reservation.reservation_state='held'),0)::bigint "
        "FROM LATERAL (SELECT * FROM "
        "campaign_operations_budget_ledger_entry "
        "WHERE operational_campaign_id=$1 "
        "ORDER BY ledger_version DESC LIMIT 1) head "
        "LEFT JOIN campaign_operations_reservation reservation "
        "ON reservation.operational_campaign_id=$1 "
        "GROUP BY head.budget_ledger_entry_id,head.ledger_version,"
        "head.resulting_total;",
        pqxx::params{campaign.campaignId.value()}).one_row();
    const long long held = budget[6].as<long long>();
    const long long committed = budget[4].as<long long>();
    const long long total = budget[2].as<long long>();
    const auto evidence = [&](const char* kind)
    {
        return CanonicalIdentity::Create(
            kCampaignOperationsCompletionContractVersion,
            LoadEvidenceText(transaction, campaign.campaignId, kind));
    };
    return BuildCompletionEvent(campaign.campaignId,
        campaign.campaign.identity.canonicalText(), operationKey,
        AdministrativeCampaignStateFromText(
            classified[0].as<std::string>()),
        ParseCompletionClassification(classified[1].as<std::string>()),
        BudgetLedgerEntryId(budget[0].as<long long>()),
        budget[1].as<int>(), total, budget[3].as<long long>(),
        committed, budget[5].as<long long>(), held,
        total - committed - held, classified[2].as<int>(),
        classified[3].as<int>(), classified[4].as<int>(),
        classified[5].as<int>(), classified[6].as<int>(),
        classified[7].as<int>(), classified[8].as<int>(),
        classified[9].as<int>(), classified[10].as<int>(),
        classified[11].as<int>(), classified[12].as<int>(),
        evidence("authorization"), evidence("budget"),
        evidence("reservation"), evidence("request"),
        evidence("binding"), evidence("lifecycle"),
        evidence("cancellation"), evidence("reconciliation"),
        actor, reason);
}

PersistedCompletionEvent PersistCompletionEvent(
    pqxx::transaction_base& transaction, const CompletionEvent& event)
{
    ValidateCompletionEvent(event);
    const auto rows = transaction.exec(
        "INSERT INTO campaign_operations_completion_event("
        "operational_campaign_id,campaign_identity_canonical,operation_key,"
        "administrative_terminal_state,completion_classification,"
        "budget_ledger_entry_id,budget_ledger_version,"
        "budget_resulting_total,budget_ever_reserved,budget_committed,"
        "budget_released_or_expired,budget_held,budget_unallocated,"
        "scope_member_count,completed_member_count,failed_member_count,"
        "cancelled_or_never_dispatched_member_count,reservation_count,"
        "request_count,binding_count,control_owner_count,"
        "cancellation_request_count,cancellation_settlement_count,"
        "unresolved_blocking_observation_count,"
        "authorization_evidence_canonical,authorization_evidence_hash,"
        "budget_evidence_canonical,"
        "budget_evidence_hash,reservation_evidence_canonical,"
        "reservation_evidence_hash,request_evidence_canonical,"
        "request_evidence_hash,binding_evidence_canonical,"
        "binding_evidence_hash,lifecycle_evidence_canonical,"
        "lifecycle_evidence_hash,cancellation_evidence_canonical,"
        "cancellation_evidence_hash,reconciliation_evidence_canonical,"
        "reconciliation_evidence_hash,actor_identity,capability,reason,"
        "completion_contract_version,completion_identity_canonical,"
        "completion_identity_hash) VALUES("
        "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,"
        "$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,"
        "$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,1,$44,$45) "
        "RETURNING completion_event_id,"
        "to_char(recorded_at AT TIME ZONE 'UTC',"
        "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"');",
        pqxx::params{
            event.campaignId.value(), event.campaignCanonicalText,
            event.operationKey, ToText(event.terminalState),
            ToText(event.classification), event.budgetLedgerEntryId.value(),
            event.budgetLedgerVersion, event.budgetResultingTotal,
            event.budgetEverReserved, event.budgetCommitted,
            event.budgetReleasedOrExpired, event.budgetHeld,
            event.budgetUnallocated, event.scopeMemberCount,
            event.completedMemberCount, event.failedMemberCount,
            event.cancelledOrNeverDispatchedMemberCount,
            event.reservationCount, event.requestCount, event.bindingCount,
            event.controlOwnerCount, event.cancellationRequestCount,
            event.cancellationSettlementCount,
            event.unresolvedBlockingObservationCount,
            event.authorizationEvidence.canonicalText(),
            event.authorizationEvidence.hash(),
            event.budgetEvidence.canonicalText(),
            event.budgetEvidence.hash(),
            event.reservationEvidence.canonicalText(),
            event.reservationEvidence.hash(),
            event.requestEvidence.canonicalText(),
            event.requestEvidence.hash(),
            event.bindingEvidence.canonicalText(),
            event.bindingEvidence.hash(),
            event.lifecycleEvidence.canonicalText(),
            event.lifecycleEvidence.hash(),
            event.cancellationEvidence.canonicalText(),
            event.cancellationEvidence.hash(),
            event.reconciliationEvidence.canonicalText(),
            event.reconciliationEvidence.hash(), event.actor.value(),
            kCampaignOperationsCompletionWriterRole, event.reason.value(),
            event.identity.canonicalText(), event.identity.hash()});
    const CompletionEventId eventId(rows.one_row()[0].as<long long>());
    transaction.exec(
        "INSERT INTO campaign_operations_completion_audit_reference_event("
        "operational_campaign_id,completion_event_id,actor_identity,"
        "capability,reason,outcome,replay_disposition,diagnostic_code) "
        "VALUES($1,$2,$3,$4,$5,'recorded','recorded',"
        "'all_completion_prerequisites_proven');",
        pqxx::params{event.campaignId.value(), eventId.value(),
            event.actor.value(), kCampaignOperationsCompletionWriterRole,
            event.reason.value()});
    return {eventId, event, rows.one_row()[1].as<std::string>()};
}

CompletionStatus LoadCompletionStatusProjection(
    pqxx::transaction_base& transaction, OperationalCampaignId campaignId)
{
    const auto campaign = FindOperationalCampaign(transaction, campaignId);
    if (!campaign)
        throw Error(ErrorCode::persistenceConflict,
            "campaign_operations_campaign_not_found");
    auto completion = FindCompletionEvent(transaction, campaignId);
    auto blockers = LoadCompletionBlockers(transaction, campaignId);
    const std::string lifecycle =
        LoadEvidenceText(transaction, campaignId, "lifecycle");
    const bool lifecycleChanged = completion &&
        completion->event.lifecycleEvidence.canonicalText() != lifecycle;
    const bool inconsistent = std::any_of(
        blockers.begin(), blockers.end(), [](const CompletionBlocker& blocker)
        {
            return blocker.blockerClass == "inconsistent";
        });
    const bool reconciliationRequired = std::any_of(
        blockers.begin(), blockers.end(), [](const CompletionBlocker& blocker)
        {
            return blocker.blockerClass == "reconciliation_required";
        });
    AdministrativeCampaignState state;
    if (inconsistent)
        state = AdministrativeCampaignState::inconsistent;
    else if (reconciliationRequired || lifecycleChanged)
        state = AdministrativeCampaignState::reconciliationRequired;
    else if (completion)
        state = completion->event.terminalState;
    else
        state = DeriveNonterminalState(transaction, campaignId, blockers);
    return {campaignId, state, completion.has_value(),
        std::move(completion), std::move(blockers), lifecycleChanged,
        lifecycle, LoadEvidenceText(transaction, campaignId, "cancellation"),
        LoadEvidenceText(transaction, campaignId, "reconciliation")};
}

} // namespace EA::CampaignOperations
