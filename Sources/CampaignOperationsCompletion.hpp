#pragma once

#include "CampaignOperations.hpp"

#include <optional>
#include <string>
#include <vector>

namespace EA::CampaignOperations
{

inline constexpr int kCampaignOperationsCompletionContractVersion = 1;
inline constexpr char kCampaignOperationsCompletionWriterRole[] =
    "campaign_operations_completion_writer";

enum class CompletionAttemptDisposition
{
    recorded,
    existingIdentical,
    conflictingReplay,
    blocked
};

std::string ToText(CompletionAttemptDisposition value);

struct CompletionBlocker final
{
    std::string code;
    std::string detail;
    std::string blockerClass;

    bool operator==(const CompletionBlocker&) const = default;
};

struct CompletionEvent final
{
    CanonicalIdentity identity;
    OperationalCampaignId campaignId;
    std::string campaignCanonicalText;
    std::string operationKey;
    AdministrativeCampaignState terminalState;
    CompletionClassification classification;
    BudgetLedgerEntryId budgetLedgerEntryId;
    int budgetLedgerVersion = 0;
    long long budgetResultingTotal = 0;
    long long budgetEverReserved = 0;
    long long budgetCommitted = 0;
    long long budgetReleasedOrExpired = 0;
    long long budgetHeld = 0;
    long long budgetUnallocated = 0;
    int scopeMemberCount = 0;
    int completedMemberCount = 0;
    int failedMemberCount = 0;
    int cancelledOrNeverDispatchedMemberCount = 0;
    int reservationCount = 0;
    int requestCount = 0;
    int bindingCount = 0;
    int controlOwnerCount = 0;
    int cancellationRequestCount = 0;
    int cancellationSettlementCount = 0;
    int unresolvedBlockingObservationCount = 0;
    CanonicalIdentity authorizationEvidence;
    CanonicalIdentity budgetEvidence;
    CanonicalIdentity reservationEvidence;
    CanonicalIdentity requestEvidence;
    CanonicalIdentity bindingEvidence;
    CanonicalIdentity lifecycleEvidence;
    CanonicalIdentity cancellationEvidence;
    CanonicalIdentity reconciliationEvidence;
    ActorIdentity actor;
    Reason reason;

    bool operator==(const CompletionEvent&) const = default;
};

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
    Reason reason);
void ValidateCompletionEvent(const CompletionEvent& event);

} // namespace EA::CampaignOperations
