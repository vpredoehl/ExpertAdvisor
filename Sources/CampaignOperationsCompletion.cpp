#include "CampaignOperationsCompletion.hpp"

#include <algorithm>
#include <locale>
#include <sstream>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

std::string Framed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
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

bool IsTerminalState(AdministrativeCampaignState value)
{
    return value == AdministrativeCampaignState::terminalCompleted ||
        value == AdministrativeCampaignState::terminalCancelled ||
        value == AdministrativeCampaignState::terminalFailed;
}

} // namespace

std::string ToText(CompletionAttemptDisposition value)
{
    switch (value)
    {
        case CompletionAttemptDisposition::recorded:
            return "recorded";
        case CompletionAttemptDisposition::existingIdentical:
            return "existing_identical";
        case CompletionAttemptDisposition::conflictingReplay:
            return "conflicting_replay";
        case CompletionAttemptDisposition::blocked:
            return "blocked";
    }
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_completion_disposition_invalid");
}

CompletionEvent BuildCompletionEvent(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::string operationKey, AdministrativeCampaignState terminalState,
    CompletionClassification classification,
    BudgetLedgerEntryId budgetLedgerEntryId, int budgetLedgerVersion,
    long long budgetResultingTotal, long long budgetEverReserved,
    long long budgetCommitted, long long budgetReleasedOrExpired,
    long long budgetHeld, long long budgetUnallocated, int scopeMemberCount,
    int completedMemberCount, int failedMemberCount,
    int cancelledOrNeverDispatchedMemberCount, int reservationCount,
    int requestCount, int bindingCount, int controlOwnerCount,
    int cancellationRequestCount, int cancellationSettlementCount,
    int unresolvedBlockingObservationCount,
    CanonicalIdentity authorizationEvidence,
    CanonicalIdentity budgetEvidence,
    CanonicalIdentity reservationEvidence,
    CanonicalIdentity requestEvidence, CanonicalIdentity bindingEvidence,
    CanonicalIdentity lifecycleEvidence,
    CanonicalIdentity cancellationEvidence,
    CanonicalIdentity reconciliationEvidence, ActorIdentity actor,
    Reason reason)
{
    if (campaignCanonicalText.empty() ||
        campaignCanonicalText.size() >
            kCampaignOperationsCanonicalMaximumBytes ||
        !ValidOperationKey(operationKey) || !IsTerminalState(terminalState) ||
        budgetLedgerVersion <= 0 || budgetResultingTotal < 0 ||
        budgetEverReserved < 0 || budgetCommitted < 0 ||
        budgetReleasedOrExpired < 0 || budgetHeld != 0 ||
        budgetUnallocated < 0 || scopeMemberCount <= 0 ||
        completedMemberCount < 0 || failedMemberCount < 0 ||
        cancelledOrNeverDispatchedMemberCount < 0 ||
        completedMemberCount + failedMemberCount +
                cancelledOrNeverDispatchedMemberCount !=
            scopeMemberCount ||
        reservationCount < 0 || requestCount < 0 || bindingCount < 0 ||
        controlOwnerCount < 0 || cancellationRequestCount < 0 ||
        cancellationSettlementCount < 0 ||
        unresolvedBlockingObservationCount != 0 ||
        budgetEverReserved !=
            budgetCommitted + budgetReleasedOrExpired + budgetHeld ||
        budgetResultingTotal !=
            budgetCommitted + budgetHeld + budgetUnallocated)
        throw Error(ErrorCode::invalidCompletionEvidence,
            "campaign_operations_completion_event_invalid");

    const CompletionEvidence classificationEvidence{
        scopeMemberCount, completedMemberCount, failedMemberCount,
        cancelledOrNeverDispatchedMemberCount,
        classification ==
            CompletionClassification::operationalRequestFailed,
        bindingCount > 0, true};
    if (ClassifyCompletion(classificationEvidence) != classification)
        throw Error(ErrorCode::invalidCompletionEvidence,
            "campaign_operations_completion_classification_mismatch");
    const AdministrativeCampaignState expectedState =
        classification == CompletionClassification::operationalRequestFailed ||
            classification == CompletionClassification::mixedTerminalOutcomes ||
            classification == CompletionClassification::downstreamFailure
        ? AdministrativeCampaignState::terminalFailed
        : (classification == CompletionClassification::allScopeCancelled
              ? AdministrativeCampaignState::terminalCancelled
              : AdministrativeCampaignState::terminalCompleted);
    if (terminalState != expectedState)
        throw Error(ErrorCode::invalidCompletionEvidence,
            "campaign_operations_completion_terminal_state_mismatch");

    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "campaign_operations_completion_v1"
        << ";campaign=" << Framed(campaignCanonicalText)
        << ";operation_key=" << Framed(operationKey)
        << ";terminal_state=" << ToText(terminalState)
        << ";classification=" << ToText(classification)
        << ";budget_ledger_entry_id=" << budgetLedgerEntryId.value()
        << ";budget_ledger_version=" << budgetLedgerVersion
        << ";budget_resulting_total=" << budgetResultingTotal
        << ";budget_ever_reserved=" << budgetEverReserved
        << ";budget_committed=" << budgetCommitted
        << ";budget_released_or_expired=" << budgetReleasedOrExpired
        << ";budget_held=" << budgetHeld
        << ";budget_unallocated=" << budgetUnallocated
        << ";scope_member_count=" << scopeMemberCount
        << ";completed_member_count=" << completedMemberCount
        << ";failed_member_count=" << failedMemberCount
        << ";cancelled_or_never_dispatched_member_count="
        << cancelledOrNeverDispatchedMemberCount
        << ";reservation_count=" << reservationCount
        << ";request_count=" << requestCount
        << ";binding_count=" << bindingCount
        << ";control_owner_count=" << controlOwnerCount
        << ";cancellation_request_count=" << cancellationRequestCount
        << ";cancellation_settlement_count=" << cancellationSettlementCount
        << ";unresolved_blocking_observation_count="
        << unresolvedBlockingObservationCount
        << ";authorization_evidence="
        << Framed(authorizationEvidence.canonicalText())
        << ";budget_evidence=" << Framed(budgetEvidence.canonicalText())
        << ";reservation_evidence="
        << Framed(reservationEvidence.canonicalText())
        << ";request_evidence=" << Framed(requestEvidence.canonicalText())
        << ";binding_evidence=" << Framed(bindingEvidence.canonicalText())
        << ";lifecycle_evidence="
        << Framed(lifecycleEvidence.canonicalText())
        << ";cancellation_evidence="
        << Framed(cancellationEvidence.canonicalText())
        << ";reconciliation_evidence="
        << Framed(reconciliationEvidence.canonicalText())
        << ";actor=" << Framed(actor.value())
        << ";capability=" << kCampaignOperationsCompletionWriterRole
        << ";reason=" << Framed(reason.value());
    return {
        CanonicalIdentity::Create(
            kCampaignOperationsCompletionContractVersion, canonical.str()),
        campaignId, std::move(campaignCanonicalText),
        std::move(operationKey), terminalState, classification,
        budgetLedgerEntryId, budgetLedgerVersion, budgetResultingTotal,
        budgetEverReserved, budgetCommitted, budgetReleasedOrExpired,
        budgetHeld, budgetUnallocated, scopeMemberCount,
        completedMemberCount, failedMemberCount,
        cancelledOrNeverDispatchedMemberCount, reservationCount,
        requestCount, bindingCount, controlOwnerCount,
        cancellationRequestCount, cancellationSettlementCount,
        unresolvedBlockingObservationCount,
        std::move(authorizationEvidence), std::move(budgetEvidence),
        std::move(reservationEvidence), std::move(requestEvidence),
        std::move(bindingEvidence), std::move(lifecycleEvidence),
        std::move(cancellationEvidence),
        std::move(reconciliationEvidence), std::move(actor),
        std::move(reason)};
}

void ValidateCompletionEvent(const CompletionEvent& event)
{
    const auto rebuilt = BuildCompletionEvent(event.campaignId,
        event.campaignCanonicalText, event.operationKey, event.terminalState,
        event.classification, event.budgetLedgerEntryId,
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
        event.authorizationEvidence, event.budgetEvidence,
        event.reservationEvidence, event.requestEvidence,
        event.bindingEvidence, event.lifecycleEvidence,
        event.cancellationEvidence, event.reconciliationEvidence,
        event.actor, event.reason);
    if (rebuilt != event)
        throw Error(ErrorCode::invalidCompletionEvidence,
            "campaign_operations_completion_identity_mismatch");
}

} // namespace EA::CampaignOperations
