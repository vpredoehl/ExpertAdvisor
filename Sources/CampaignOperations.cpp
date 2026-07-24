#include "CampaignOperations.hpp"

#include "ExperimentRecommendation.hpp"

#include <array>
#include <cctype>
#include <locale>
#include <sstream>
#include <utility>

namespace EA::CampaignOperations
{
namespace
{

std::string LengthPrefixed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

bool IsAsciiAlphanumeric(unsigned char character)
{
    return (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9');
}

bool IsValidUtf8(const std::string& value)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(value.data());
    std::size_t index = 0U;
    while (index < value.size())
    {
        const unsigned char first = bytes[index];
        if (first <= 0x7fU)
        {
            ++index;
            continue;
        }
        std::size_t continuationCount = 0U;
        unsigned int codePoint = 0U;
        if (first >= 0xc2U && first <= 0xdfU)
        {
            continuationCount = 1U;
            codePoint = first & 0x1fU;
        }
        else if (first >= 0xe0U && first <= 0xefU)
        {
            continuationCount = 2U;
            codePoint = first & 0x0fU;
        }
        else if (first >= 0xf0U && first <= 0xf4U)
        {
            continuationCount = 3U;
            codePoint = first & 0x07U;
        }
        else return false;
        if (index + continuationCount >= value.size()) return false;
        for (std::size_t offset = 1U; offset <= continuationCount; ++offset)
        {
            const unsigned char next = bytes[index + offset];
            if ((next & 0xc0U) != 0x80U) return false;
            codePoint = (codePoint << 6U) | (next & 0x3fU);
        }
        if ((continuationCount == 2U && codePoint < 0x800U) ||
            (continuationCount == 3U && codePoint < 0x10000U) ||
            (codePoint >= 0xd800U && codePoint <= 0xdfffU) ||
            codePoint > 0x10ffffU)
            return false;
        index += continuationCount + 1U;
    }
    return true;
}

bool IsValidFramedText(const std::string& value, std::size_t maximum)
{
    if (value.empty() || value.size() > maximum || !IsValidUtf8(value))
        return false;
    for (const unsigned char character : value)
        if (character == 0U || character == 0x7fU ||
            (character < 0x20U && character != '\t' && character != '\n' &&
                character != '\r'))
            return false;
    return true;
}

bool IsValidActor(const std::string& value)
{
    if (value.empty() || value.size() > kCampaignOperationsActorMaximumBytes ||
        !IsAsciiAlphanumeric(static_cast<unsigned char>(value.front())))
        return false;
    for (const unsigned char character : value)
        if (!IsAsciiAlphanumeric(character) && character != '.' &&
            character != '_' && character != '@' && character != ':' &&
            character != '/' && character != '+' && character != '-')
            return false;
    return true;
}

bool IsLeapYear(int year)
{
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int ParseDigits(const std::string& value, std::size_t start,
    std::size_t count)
{
    int result = 0;
    for (std::size_t index = start; index < start + count; ++index)
    {
        const unsigned char character =
            static_cast<unsigned char>(value[index]);
        if (!std::isdigit(character)) return -1;
        result = result * 10 + static_cast<int>(character - '0');
    }
    return result;
}

bool IsNormalizedUtcTimestamp(const std::string& value)
{
    if (value.size() != 27U || value[4] != '-' || value[7] != '-' ||
        value[10] != 'T' || value[13] != ':' || value[16] != ':' ||
        value[19] != '.' || value[26] != 'Z')
        return false;
    const int year = ParseDigits(value, 0U, 4U);
    const int month = ParseDigits(value, 5U, 2U);
    const int day = ParseDigits(value, 8U, 2U);
    const int hour = ParseDigits(value, 11U, 2U);
    const int minute = ParseDigits(value, 14U, 2U);
    const int second = ParseDigits(value, 17U, 2U);
    if (year < 1 || month < 1 || month > 12 || hour < 0 || hour > 23 ||
        minute < 0 || minute > 59 || second < 0 || second > 59 ||
        ParseDigits(value, 20U, 6U) < 0)
        return false;
    constexpr std::array<int, 12> daysInMonth{
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const int maximumDay = month == 2 && IsLeapYear(year)
        ? 29 : daysInMonth[static_cast<std::size_t>(month - 1)];
    return day >= 1 && day <= maximumDay;
}

void ValidateCanonicalComponent(const std::string& canonical,
    const std::string& hash, const char* reason)
{
    if (!IsValidFramedText(canonical,
            kCampaignOperationsCanonicalMaximumBytes))
        throw Error(ErrorCode::invalidCanonicalText, reason);
    if (hash != ExperimentRecommendation::RecommendationCanonicalHash(canonical))
        throw Error(ErrorCode::invalidCanonicalHash, reason);
}

template <typename Enum>
Enum InvalidEnumText()
{
    throw Error(ErrorCode::invalidEnumText,
        "campaign_operations_enum_text_invalid");
}

std::string OptionalCanonical(
    const std::optional<std::string>& value)
{
    return value ? LengthPrefixed(*value) : "none";
}

std::string OptionalTimestamp(const std::optional<UtcTimestamp>& value)
{
    return value ? value->value() : "none";
}

} // namespace

Error::Error(ErrorCode codeValue, std::string reason)
    : std::runtime_error(std::move(reason)), code_(codeValue)
{
}

ErrorCode Error::code() const noexcept { return code_; }

CanonicalIdentity CanonicalIdentity::Create(
    int contractVersion, std::string canonicalText)
{
    return Hydrate(contractVersion, canonicalText,
        ExperimentRecommendation::RecommendationCanonicalHash(canonicalText));
}

CanonicalIdentity CanonicalIdentity::Hydrate(int contractVersion,
    std::string canonicalText, std::string hash)
{
    if (contractVersion != kCampaignOperationsContractVersion)
        throw Error(ErrorCode::unsupportedContractVersion,
            "campaign_operations_contract_unsupported");
    ValidateCanonicalComponent(canonicalText, hash,
        "campaign_operations_identity_invalid");
    return CanonicalIdentity(contractVersion, std::move(canonicalText),
        std::move(hash));
}

CanonicalIdentity::CanonicalIdentity(int contractVersion,
    std::string canonicalText, std::string hash)
    : contractVersion_(contractVersion),
      canonicalText_(std::move(canonicalText)), hash_(std::move(hash))
{
}

int CanonicalIdentity::contractVersion() const noexcept
{
    return contractVersion_;
}

const std::string& CanonicalIdentity::canonicalText() const noexcept
{
    return canonicalText_;
}

const std::string& CanonicalIdentity::hash() const noexcept { return hash_; }

ActorIdentity::ActorIdentity(std::string value) : value_(std::move(value))
{
    if (!IsValidActor(value_))
        throw Error(ErrorCode::invalidActorIdentity,
            "campaign_operations_actor_identity_invalid");
}

const std::string& ActorIdentity::value() const noexcept { return value_; }

Reason::Reason(std::string value) : value_(std::move(value))
{
    if (!IsValidFramedText(value_, kCampaignOperationsReasonMaximumBytes))
        throw Error(ErrorCode::invalidReason,
            "campaign_operations_reason_invalid");
    bool nonWhitespace = false;
    for (const unsigned char character : value_)
        if (character >= 0x80U ||
            (character != ' ' && character != '\t' && character != '\n' &&
                character != '\r'))
            nonWhitespace = true;
    if (!nonWhitespace)
        throw Error(ErrorCode::invalidReason,
            "campaign_operations_reason_invalid");
}

const std::string& Reason::value() const noexcept { return value_; }

UtcTimestamp::UtcTimestamp(std::string value) : value_(std::move(value))
{
    if (!IsNormalizedUtcTimestamp(value_))
        throw Error(ErrorCode::invalidTimestamp,
            "campaign_operations_timestamp_invalid");
}

const std::string& UtcTimestamp::value() const noexcept { return value_; }

#define EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(value, text) \
    case value: return text

std::string ToText(CampaignOriginKind value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CampaignOriginKind::phase4dMaterializationV1,
            "phase4d_materialization_v1");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(OperationalActionKind value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            OperationalActionKind::dispatchFullMaterialization,
            "dispatch_full_materialization");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            OperationalActionKind::adoptExistingPendingAndControl,
            "adopt_existing_pending_and_control");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(ScopeKind value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ScopeKind::completeMaterialization,
            "complete_materialization");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(PrerequisitePolicy value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            "phase4d_materialization_only_v1");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1,
            "phase4d_materialization_plus_exact_phase6d_ratification_v1");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(AuthorizationEventKind value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AuthorizationEventKind::granted, "granted");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AuthorizationEventKind::revoked, "revoked");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AuthorizationEventKind::expiryObserved, "expiry_observed");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(AdministrativeCampaignState value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::awaitingOperationalAuthorization,
            "awaiting_operational_authorization");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::authorized, "authorized");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::budgeted, "budgeted");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::reserving, "reserving");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::ready, "ready");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::dispatching, "dispatching");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::active, "active");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::paused, "paused");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::cancellationRequested,
            "cancellation_requested");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::cancelling, "cancelling");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::terminalCompleted,
            "terminal_completed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::terminalCancelled,
            "terminal_cancelled");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::terminalFailed, "terminal_failed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::inconsistent, "inconsistent");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            AdministrativeCampaignState::reconciliationRequired,
            "reconciliation_required");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(MemberOperationalState value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::notRequested, "not_requested");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::reserved, "reserved");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::requestReady, "request_ready");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::dispatching, "dispatching");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::boundPending, "bound_pending");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::claimed, "claimed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::running, "running");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::terminalCompleted, "terminal_completed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::terminalFailed, "terminal_failed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::terminalCancelled, "terminal_cancelled");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::inconsistent, "inconsistent");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            MemberOperationalState::reconciliationRequired,
            "reconciliation_required");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(BudgetLedgerEntryKind value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(BudgetLedgerEntryKind::grant,
            "grant");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(BudgetLedgerEntryKind::amend,
            "amend");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(BudgetLedgerEntryKind::revoke,
            "revoke");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(BudgetLedgerEntryKind::supersede,
            "supersede");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(BudgetLedgerStatus value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(BudgetLedgerStatus::active,
            "active");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(BudgetLedgerStatus::revoked,
            "revoked");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(BudgetUnit value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            BudgetUnit::materializedMemberDispatch,
            "materialized_member_dispatch");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(ReservationState value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReservationState::held, "held");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReservationState::committed,
            "committed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReservationState::released,
            "released");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReservationState::expired,
            "expired");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReservationState::reconciliationRequired,
            "reconciliation_required");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(RequestState value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(RequestState::ready, "ready");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(RequestState::dispatching,
            "dispatching");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(RequestState::bound, "bound");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(RequestState::permanentlyFailed,
            "permanently_failed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(RequestState::cancelled,
            "cancelled");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            RequestState::reconciliationRequired, "reconciliation_required");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(ControlEventKind value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ControlEventKind::pause, "pause");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ControlEventKind::resume,
            "resume");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(DispatchOutcome value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(DispatchOutcome::createdAndBound,
            "created_and_bound");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            DispatchOutcome::adoptedExistingPendingAndBound,
            "adopted_existing_pending_and_bound");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            DispatchOutcome::existingIdentical, "existing_identical");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(DispatchOutcome::rejected,
            "rejected");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(DispatchOutcome::conflict,
            "conflict");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            DispatchOutcome::reconciliationRequired,
            "reconciliation_required");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            DispatchOutcome::noDownstreamCommit, "no_downstream_commit");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(BindingDisposition value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(BindingDisposition::created,
            "created");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            BindingDisposition::adoptedExistingPending,
            "adopted_existing_pending");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(DownstreamControlMode value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            DownstreamControlMode::createdControl, "created_control");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            DownstreamControlMode::authorizedAdoptionControl,
            "authorized_adoption_control");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(CancellationSettlementDisposition value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CancellationSettlementDisposition::unboundCancelled,
            "unbound_cancelled");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CancellationSettlementDisposition::lifecycleRequestAccepted,
            "lifecycle_request_accepted");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CancellationSettlementDisposition::alreadyTerminal,
            "already_terminal");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CancellationSettlementDisposition::
                runningCancellationNotSupported,
            "running_cancellation_not_supported");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CancellationSettlementDisposition::inconsistent,
            "inconsistent");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(ReconciliationReason value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::readyRequestNotDispatched,
            "ready_request_not_dispatched");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::dispatchLeaseExpiredNoDownstreamEvidence,
            "dispatch_lease_expired_no_downstream_evidence");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::dispatchOutcomeUnknown,
            "dispatch_outcome_unknown");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::bindingProjectionMissing,
            "binding_projection_missing");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::reservationProjectionMissingCommit,
            "reservation_projection_missing_commit");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::heldReservationTerminalUnboundRequest,
            "held_reservation_terminal_unbound_request");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::reservationExpiredNoDownstreamEvidence,
            "reservation_expired_no_downstream_evidence");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::cancellationSettlementPending,
            "cancellation_settlement_pending");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::terminalLifecycleCompletionReady,
            "terminal_lifecycle_completion_ready");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::progressedUnboundEvidence,
            "progressed_unbound_evidence");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::partialDownstreamEvidence,
            "partial_downstream_evidence");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::bindingCardinalityMismatch,
            "binding_cardinality_mismatch");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::controlOwnerConflict,
            "control_owner_conflict");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::budgetAccountingMismatch,
            "budget_accounting_mismatch");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::postCompletionLifecycleChanged,
            "post_completion_lifecycle_changed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReconciliationReason::causalityAmbiguous,
            "causality_ambiguous");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(CompletionClassification value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CompletionClassification::operationalRequestFailed,
            "operational_request_failed");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CompletionClassification::mixedTerminalOutcomes,
            "mixed_terminal_outcomes");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CompletionClassification::downstreamFailure,
            "downstream_failure");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CompletionClassification::terminalPartialCompletion,
            "terminal_partial_completion");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CompletionClassification::allScopeCancelled,
            "all_scope_cancelled");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            CompletionClassification::allDownstreamCompleted,
            "all_downstream_completed");
    }
    return InvalidEnumText<std::string>();
}

std::string ToText(ReplayDisposition value)
{
    switch (value)
    {
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReplayDisposition::recorded,
            "recorded");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(
            ReplayDisposition::existingIdentical, "existing_identical");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReplayDisposition::conflict,
            "conflict");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReplayDisposition::rejected,
            "rejected");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReplayDisposition::repaired,
            "repaired");
        EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE(ReplayDisposition::noChange,
            "no_change");
    }
    return InvalidEnumText<std::string>();
}

#undef EA_CAMPAIGN_OPERATIONS_TO_TEXT_CASE

CampaignOriginKind CampaignOriginKindFromText(const std::string& text)
{
    if (text == "phase4d_materialization_v1")
        return CampaignOriginKind::phase4dMaterializationV1;
    return InvalidEnumText<CampaignOriginKind>();
}

OperationalActionKind OperationalActionKindFromText(const std::string& text)
{
    if (text == "dispatch_full_materialization")
        return OperationalActionKind::dispatchFullMaterialization;
    if (text == "adopt_existing_pending_and_control")
        return OperationalActionKind::adoptExistingPendingAndControl;
    return InvalidEnumText<OperationalActionKind>();
}

ScopeKind ScopeKindFromText(const std::string& text)
{
    if (text == "complete_materialization")
        return ScopeKind::completeMaterialization;
    return InvalidEnumText<ScopeKind>();
}

PrerequisitePolicy PrerequisitePolicyFromText(const std::string& text)
{
    if (text == "phase4d_materialization_only_v1")
        return PrerequisitePolicy::phase4dMaterializationOnlyV1;
    if (text ==
        "phase4d_materialization_plus_exact_phase6d_ratification_v1")
        return PrerequisitePolicy::
            phase4dMaterializationPlusExactPhase6dRatificationV1;
    return InvalidEnumText<PrerequisitePolicy>();
}

AuthorizationEventKind AuthorizationEventKindFromText(
    const std::string& text)
{
    if (text == "granted") return AuthorizationEventKind::granted;
    if (text == "revoked") return AuthorizationEventKind::revoked;
    if (text == "expiry_observed")
        return AuthorizationEventKind::expiryObserved;
    return InvalidEnumText<AuthorizationEventKind>();
}

AdministrativeCampaignState AdministrativeCampaignStateFromText(
    const std::string& text)
{
    constexpr std::array<AdministrativeCampaignState, 15> values{
        AdministrativeCampaignState::awaitingOperationalAuthorization,
        AdministrativeCampaignState::authorized,
        AdministrativeCampaignState::budgeted,
        AdministrativeCampaignState::reserving,
        AdministrativeCampaignState::ready,
        AdministrativeCampaignState::dispatching,
        AdministrativeCampaignState::active,
        AdministrativeCampaignState::paused,
        AdministrativeCampaignState::cancellationRequested,
        AdministrativeCampaignState::cancelling,
        AdministrativeCampaignState::terminalCompleted,
        AdministrativeCampaignState::terminalCancelled,
        AdministrativeCampaignState::terminalFailed,
        AdministrativeCampaignState::inconsistent,
        AdministrativeCampaignState::reconciliationRequired};
    for (const auto value : values)
        if (ToText(value) == text) return value;
    return InvalidEnumText<AdministrativeCampaignState>();
}

OperationalCampaign::OperationalCampaign(CanonicalIdentity identityValue,
    long long materializationIdValue, int materializationContractVersionValue,
    std::string materializationCanonicalTextValue,
    std::string materializationIdentityHashValue, int memberCountValue,
    CampaignOriginKind originKindValue,
    OperationalActionKind actionKindValue, int actionContractVersionValue,
    ScopeKind scopeKindValue, int scopeContractVersionValue)
    : identity(std::move(identityValue)),
      materializationId(materializationIdValue),
      materializationContractVersion(materializationContractVersionValue),
      materializationCanonicalText(
          std::move(materializationCanonicalTextValue)),
      materializationIdentityHash(
          std::move(materializationIdentityHashValue)),
      memberCount(memberCountValue), originKind(originKindValue),
      actionKind(actionKindValue),
      actionContractVersion(actionContractVersionValue),
      scopeKind(scopeKindValue),
      scopeContractVersion(scopeContractVersionValue)
{
}

OperationalCampaign BuildOperationalCampaign(long long materializationId,
    int materializationContractVersion,
    std::string materializationCanonicalText,
    std::string materializationIdentityHash, int memberCount,
    CampaignOriginKind originKind, OperationalActionKind actionKind,
    int actionContractVersion, ScopeKind scopeKind,
    int scopeContractVersion)
{
    if (materializationId <= 0 || materializationContractVersion != 1 ||
        memberCount <= 0)
        throw Error(ErrorCode::invalidMaterialization,
            "campaign_operations_materialization_invalid");
    ValidateCanonicalComponent(materializationCanonicalText,
        materializationIdentityHash,
        "campaign_operations_materialization_identity_invalid");
    if (originKind != CampaignOriginKind::phase4dMaterializationV1 ||
        actionKind != OperationalActionKind::dispatchFullMaterialization ||
        actionContractVersion != kCampaignOperationsActionContractVersion ||
        scopeKind != ScopeKind::completeMaterialization ||
        scopeContractVersion != kCampaignOperationsScopeContractVersion)
        throw Error(ErrorCode::unsupportedContractVersion,
            "campaign_operations_campaign_contract_unsupported");
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "campaign_operations_campaign_v1"
        << ";materialization_id=" << materializationId
        << ";materialization_contract_version="
        << materializationContractVersion
        << ";materialization_canonical="
        << LengthPrefixed(materializationCanonicalText)
        << ";materialization_identity_hash="
        << LengthPrefixed(materializationIdentityHash)
        << ";member_count=" << memberCount
        << ";origin_kind=" << ToText(originKind)
        << ";action_kind=" << ToText(actionKind)
        << ";action_contract_version=" << actionContractVersion
        << ";scope_kind=" << ToText(scopeKind)
        << ";scope_contract_version=" << scopeContractVersion;
    OperationalCampaign campaign(
        CanonicalIdentity::Create(kCampaignOperationsContractVersion,
            output.str()),
        materializationId, materializationContractVersion,
        std::move(materializationCanonicalText),
        std::move(materializationIdentityHash), memberCount, originKind,
        actionKind, actionContractVersion, scopeKind, scopeContractVersion);
    return campaign;
}

void ValidateOperationalCampaign(const OperationalCampaign& campaign)
{
    const OperationalCampaign rebuilt = BuildOperationalCampaign(
        campaign.materializationId, campaign.materializationContractVersion,
        campaign.materializationCanonicalText,
        campaign.materializationIdentityHash, campaign.memberCount,
        campaign.originKind, campaign.actionKind,
        campaign.actionContractVersion, campaign.scopeKind,
        campaign.scopeContractVersion);
    if (rebuilt != campaign)
        throw Error(ErrorCode::invalidMaterialization,
            "campaign_operations_campaign_identity_mismatch");
}

AdministrativeCampaignState InitialAdministrativeState(
    const OperationalCampaign& campaign)
{
    ValidateOperationalCampaign(campaign);
    return AdministrativeCampaignState::awaitingOperationalAuthorization;
}

GovernanceProvenanceEvent::GovernanceProvenanceEvent(
    CanonicalIdentity identityValue, OperationalCampaignId campaignIdValue,
    std::string campaignCanonicalTextValue, long long ratificationEventIdValue,
    int ratificationContractVersionValue,
    std::string ratificationCanonicalTextValue,
    std::string ratificationIdentityHashValue, long long reviewEventIdValue,
    int reviewContractVersionValue, std::string reviewCanonicalTextValue,
    std::string reviewIdentityHashValue, long long proposalIdValue,
    int proposalContractVersionValue, std::string proposalCanonicalTextValue,
    std::string proposalIdentityHashValue,
    PrerequisitePolicy prerequisitePolicyValue)
    : identity(std::move(identityValue)), campaignId(campaignIdValue),
      campaignCanonicalText(std::move(campaignCanonicalTextValue)),
      ratificationEventId(ratificationEventIdValue),
      ratificationContractVersion(ratificationContractVersionValue),
      ratificationCanonicalText(std::move(ratificationCanonicalTextValue)),
      ratificationIdentityHash(std::move(ratificationIdentityHashValue)),
      reviewEventId(reviewEventIdValue),
      reviewContractVersion(reviewContractVersionValue),
      reviewCanonicalText(std::move(reviewCanonicalTextValue)),
      reviewIdentityHash(std::move(reviewIdentityHashValue)),
      proposalId(proposalIdValue),
      proposalContractVersion(proposalContractVersionValue),
      proposalCanonicalText(std::move(proposalCanonicalTextValue)),
      proposalIdentityHash(std::move(proposalIdentityHashValue)),
      prerequisitePolicy(prerequisitePolicyValue)
{
}

GovernanceProvenanceEvent BuildGovernanceProvenanceEvent(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    long long ratificationEventId, int ratificationContractVersion,
    std::string ratificationCanonicalText,
    std::string ratificationIdentityHash, long long reviewEventId,
    int reviewContractVersion, std::string reviewCanonicalText,
    std::string reviewIdentityHash, long long proposalId,
    int proposalContractVersion, std::string proposalCanonicalText,
    std::string proposalIdentityHash, PrerequisitePolicy prerequisitePolicy)
{
    if (ratificationEventId <= 0 || reviewEventId <= 0 || proposalId <= 0 ||
        ratificationContractVersion != 1 || reviewContractVersion != 1 ||
        proposalContractVersion != 1 ||
        (prerequisitePolicy !=
                PrerequisitePolicy::phase4dMaterializationOnlyV1 &&
            prerequisitePolicy != PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1))
        throw Error(ErrorCode::invalidProvenance,
            "campaign_operations_governance_provenance_invalid");
    ValidateCanonicalComponent(campaignCanonicalText,
        ExperimentRecommendation::RecommendationCanonicalHash(
            campaignCanonicalText),
        "campaign_operations_campaign_identity_invalid");
    ValidateCanonicalComponent(ratificationCanonicalText,
        ratificationIdentityHash,
        "campaign_operations_ratification_identity_invalid");
    ValidateCanonicalComponent(reviewCanonicalText, reviewIdentityHash,
        "campaign_operations_review_identity_invalid");
    ValidateCanonicalComponent(proposalCanonicalText, proposalIdentityHash,
        "campaign_operations_proposal_identity_invalid");
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "campaign_operations_governance_provenance_v1"
        << ";campaign_canonical=" << LengthPrefixed(campaignCanonicalText)
        << ";ratification_event_id=" << ratificationEventId
        << ";ratification_contract_version=" << ratificationContractVersion
        << ";ratification_canonical="
        << LengthPrefixed(ratificationCanonicalText)
        << ";ratification_identity_hash="
        << LengthPrefixed(ratificationIdentityHash)
        << ";review_event_id=" << reviewEventId
        << ";review_contract_version=" << reviewContractVersion
        << ";review_canonical=" << LengthPrefixed(reviewCanonicalText)
        << ";review_identity_hash=" << LengthPrefixed(reviewIdentityHash)
        << ";proposal_id=" << proposalId
        << ";proposal_contract_version=" << proposalContractVersion
        << ";proposal_canonical=" << LengthPrefixed(proposalCanonicalText)
        << ";proposal_identity_hash=" << LengthPrefixed(proposalIdentityHash)
        << ";prerequisite_policy=" << ToText(prerequisitePolicy);
    GovernanceProvenanceEvent event(
        CanonicalIdentity::Create(kCampaignOperationsContractVersion,
            output.str()),
        campaignId, std::move(campaignCanonicalText), ratificationEventId,
        ratificationContractVersion, std::move(ratificationCanonicalText),
        std::move(ratificationIdentityHash), reviewEventId,
        reviewContractVersion, std::move(reviewCanonicalText),
        std::move(reviewIdentityHash), proposalId, proposalContractVersion,
        std::move(proposalCanonicalText), std::move(proposalIdentityHash),
        prerequisitePolicy);
    return event;
}

void ValidateGovernanceProvenanceEvent(
    const GovernanceProvenanceEvent& event)
{
    const GovernanceProvenanceEvent rebuilt = BuildGovernanceProvenanceEvent(
        event.campaignId, event.campaignCanonicalText,
        event.ratificationEventId, event.ratificationContractVersion,
        event.ratificationCanonicalText, event.ratificationIdentityHash,
        event.reviewEventId, event.reviewContractVersion,
        event.reviewCanonicalText, event.reviewIdentityHash, event.proposalId,
        event.proposalContractVersion, event.proposalCanonicalText,
        event.proposalIdentityHash, event.prerequisitePolicy);
    if (rebuilt != event)
        throw Error(ErrorCode::invalidProvenance,
            "campaign_operations_governance_provenance_identity_mismatch");
}

OperationalAuthorizationEvent::OperationalAuthorizationEvent(
    CanonicalIdentity identityValue, OperationalCampaignId campaignIdValue,
    std::string campaignCanonicalTextValue,
    std::optional<AuthorizationEventId> previousEventIdValue,
    std::optional<std::string> previousEventCanonicalTextValue,
    std::optional<std::string> previousEventIdentityHashValue,
    int chainVersionValue, AuthorizationEventKind eventKindValue,
    OperationalActionKind actionKindValue, int actionContractVersionValue,
    ScopeKind scopeKindValue, int scopeContractVersionValue,
    PrerequisitePolicy prerequisitePolicyValue,
    std::optional<GovernanceProvenanceEventId> provenanceEventIdValue,
    std::optional<std::string> provenanceCanonicalTextValue,
    std::optional<std::string> provenanceIdentityHashValue,
    std::string authorizationRoleValue, ActorIdentity actorValue,
    Reason reasonValue, UtcTimestamp notBeforeValue,
    std::optional<UtcTimestamp> expiresAtValue)
    : identity(std::move(identityValue)), campaignId(campaignIdValue),
      campaignCanonicalText(std::move(campaignCanonicalTextValue)),
      previousEventId(std::move(previousEventIdValue)),
      previousEventCanonicalText(
          std::move(previousEventCanonicalTextValue)),
      previousEventIdentityHash(std::move(previousEventIdentityHashValue)),
      chainVersion(chainVersionValue), eventKind(eventKindValue),
      actionKind(actionKindValue),
      actionContractVersion(actionContractVersionValue),
      scopeKind(scopeKindValue), scopeContractVersion(scopeContractVersionValue),
      prerequisitePolicy(prerequisitePolicyValue),
      provenanceEventId(std::move(provenanceEventIdValue)),
      provenanceCanonicalText(std::move(provenanceCanonicalTextValue)),
      provenanceIdentityHash(std::move(provenanceIdentityHashValue)),
      authorizationRole(std::move(authorizationRoleValue)),
      actor(std::move(actorValue)), reason(std::move(reasonValue)),
      notBefore(std::move(notBeforeValue)),
      expiresAt(std::move(expiresAtValue))
{
}

OperationalAuthorizationEvent BuildOperationalAuthorizationEvent(
    OperationalCampaignId campaignId, std::string campaignCanonicalText,
    std::optional<AuthorizationEventId> previousEventId,
    std::optional<std::string> previousEventCanonicalText,
    std::optional<std::string> previousEventIdentityHash, int chainVersion,
    AuthorizationEventKind eventKind, OperationalActionKind actionKind,
    int actionContractVersion, ScopeKind scopeKind, int scopeContractVersion,
    PrerequisitePolicy prerequisitePolicy,
    std::optional<GovernanceProvenanceEventId> provenanceEventId,
    std::optional<std::string> provenanceCanonicalText,
    std::optional<std::string> provenanceIdentityHash,
    std::string authorizationRole, ActorIdentity actor, Reason reason,
    UtcTimestamp notBefore, std::optional<UtcTimestamp> expiresAt)
{
    if (chainVersion <= 0 ||
        actionContractVersion != kCampaignOperationsActionContractVersion ||
        scopeContractVersion != kCampaignOperationsScopeContractVersion ||
        authorizationRole != kCampaignOperationsAuthorizationRole)
        throw Error(ErrorCode::invalidAuthorizationEvent,
            "campaign_operations_authorization_event_invalid");
    const bool hasPrevious = previousEventId.has_value() &&
        previousEventCanonicalText.has_value() &&
        previousEventIdentityHash.has_value();
    if ((chainVersion == 1 && hasPrevious) ||
        (chainVersion > 1 && !hasPrevious) ||
        previousEventId.has_value() != previousEventCanonicalText.has_value() ||
        previousEventId.has_value() != previousEventIdentityHash.has_value())
        throw Error(ErrorCode::invalidAuthorizationEvent,
            "campaign_operations_authorization_predecessor_invalid");
    if (chainVersion == 1 && eventKind != AuthorizationEventKind::granted)
        throw Error(ErrorCode::invalidAuthorizationEvent,
            "campaign_operations_authorization_initial_event_invalid");
    ValidateCanonicalComponent(campaignCanonicalText,
        ExperimentRecommendation::RecommendationCanonicalHash(
            campaignCanonicalText),
        "campaign_operations_campaign_identity_invalid");
    if (hasPrevious)
        ValidateCanonicalComponent(*previousEventCanonicalText,
            *previousEventIdentityHash,
            "campaign_operations_authorization_predecessor_invalid");
    const bool hasProvenance = provenanceEventId.has_value() &&
        provenanceCanonicalText.has_value() &&
        provenanceIdentityHash.has_value();
    if (provenanceEventId.has_value() != provenanceCanonicalText.has_value() ||
        provenanceEventId.has_value() != provenanceIdentityHash.has_value() ||
        (prerequisitePolicy == PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1 &&
            !hasProvenance))
        throw Error(ErrorCode::invalidAuthorizationEvent,
            "campaign_operations_authorization_provenance_invalid");
    if (hasProvenance)
        ValidateCanonicalComponent(*provenanceCanonicalText,
            *provenanceIdentityHash,
            "campaign_operations_authorization_provenance_invalid");
    if (expiresAt && !(notBefore < *expiresAt))
        throw Error(ErrorCode::invalidAuthorizationEvent,
            "campaign_operations_authorization_validity_invalid");
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "campaign_operations_authorization_event_v1"
        << ";campaign_canonical=" << LengthPrefixed(campaignCanonicalText)
        << ";chain_version=" << chainVersion
        << ";previous_event_canonical="
        << OptionalCanonical(previousEventCanonicalText)
        << ";previous_event_identity_hash="
        << OptionalCanonical(previousEventIdentityHash)
        << ";event_kind=" << ToText(eventKind)
        << ";action_kind=" << ToText(actionKind)
        << ";action_contract_version=" << actionContractVersion
        << ";scope_kind=" << ToText(scopeKind)
        << ";scope_contract_version=" << scopeContractVersion
        << ";prerequisite_policy=" << ToText(prerequisitePolicy)
        << ";provenance_canonical="
        << OptionalCanonical(provenanceCanonicalText)
        << ";provenance_identity_hash="
        << OptionalCanonical(provenanceIdentityHash)
        << ";authorization_role=" << LengthPrefixed(authorizationRole)
        << ";actor=" << LengthPrefixed(actor.value())
        << ";reason=" << LengthPrefixed(reason.value())
        << ";not_before=" << notBefore.value()
        << ";expires_at=" << OptionalTimestamp(expiresAt);
    OperationalAuthorizationEvent event(
        CanonicalIdentity::Create(kCampaignOperationsContractVersion,
            output.str()),
        campaignId, std::move(campaignCanonicalText),
        std::move(previousEventId), std::move(previousEventCanonicalText),
        std::move(previousEventIdentityHash), chainVersion, eventKind,
        actionKind, actionContractVersion, scopeKind, scopeContractVersion,
        prerequisitePolicy, std::move(provenanceEventId),
        std::move(provenanceCanonicalText),
        std::move(provenanceIdentityHash), std::move(authorizationRole),
        std::move(actor), std::move(reason), std::move(notBefore),
        std::move(expiresAt));
    return event;
}

void ValidateOperationalAuthorizationEvent(
    const OperationalAuthorizationEvent& event)
{
    const OperationalAuthorizationEvent rebuilt =
        BuildOperationalAuthorizationEvent(event.campaignId,
            event.campaignCanonicalText, event.previousEventId,
            event.previousEventCanonicalText, event.previousEventIdentityHash,
            event.chainVersion, event.eventKind, event.actionKind,
            event.actionContractVersion, event.scopeKind,
            event.scopeContractVersion, event.prerequisitePolicy,
            event.provenanceEventId, event.provenanceCanonicalText,
            event.provenanceIdentityHash, event.authorizationRole,
            event.actor, event.reason, event.notBefore, event.expiresAt);
    if (rebuilt != event)
        throw Error(ErrorCode::invalidAuthorizationEvent,
            "campaign_operations_authorization_identity_mismatch");
}

bool IsAuthorizationEffectiveAt(
    const OperationalAuthorizationEvent& head, const UtcTimestamp& databaseTime)
{
    ValidateOperationalAuthorizationEvent(head);
    return head.eventKind == AuthorizationEventKind::granted &&
        !(databaseTime < head.notBefore) &&
        (!head.expiresAt || databaseTime < *head.expiresAt);
}

BudgetAccounting CalculateBudgetAccounting(long long granted,
    long long everReserved, long long committed,
    long long releasedOrExpired, BudgetLedgerStatus status)
{
    if ((status != BudgetLedgerStatus::active &&
            status != BudgetLedgerStatus::revoked) ||
        granted < 0 || everReserved < 0 || committed < 0 ||
        releasedOrExpired < 0 || committed > everReserved ||
        releasedOrExpired > everReserved - committed)
        throw Error(ErrorCode::invalidBudgetAccounting,
            "campaign_operations_budget_accounting_invalid");
    const long long held =
        everReserved - committed - releasedOrExpired;
    if (committed > granted || held > granted - committed)
        throw Error(ErrorCode::invalidBudgetAccounting,
            "campaign_operations_budget_accounting_invalid");
    const long long unallocated = granted - committed - held;
    return {granted, everReserved, committed, releasedOrExpired, held,
        unallocated,
        status == BudgetLedgerStatus::active ? unallocated : 0, status};
}

CompletionClassification ClassifyCompletion(
    const CompletionEvidence& evidence)
{
    const long long classifiedMemberCount =
        static_cast<long long>(evidence.completedMemberCount) +
        static_cast<long long>(evidence.failedMemberCount) +
        static_cast<long long>(
            evidence.cancelledOrNeverDispatchedMemberCount);
    if (!evidence.allObligationsSettled || evidence.scopeMemberCount <= 0 ||
        evidence.completedMemberCount < 0 || evidence.failedMemberCount < 0 ||
        evidence.cancelledOrNeverDispatchedMemberCount < 0 ||
        classifiedMemberCount != evidence.scopeMemberCount ||
        (!evidence.anyMemberBound &&
            (evidence.completedMemberCount > 0 ||
                evidence.failedMemberCount > 0)) ||
        (evidence.unboundRequestPermanentlyFailed && evidence.anyMemberBound))
        throw Error(ErrorCode::invalidCompletionEvidence,
            "campaign_operations_completion_evidence_invalid");
    if (evidence.unboundRequestPermanentlyFailed)
        return CompletionClassification::operationalRequestFailed;
    if (evidence.failedMemberCount > 0 &&
        (evidence.completedMemberCount > 0 ||
            evidence.cancelledOrNeverDispatchedMemberCount > 0))
        return CompletionClassification::mixedTerminalOutcomes;
    if (evidence.failedMemberCount == evidence.scopeMemberCount)
        return CompletionClassification::downstreamFailure;
    if (evidence.failedMemberCount == 0 &&
        evidence.completedMemberCount > 0 &&
        evidence.cancelledOrNeverDispatchedMemberCount > 0)
        return CompletionClassification::terminalPartialCompletion;
    if (evidence.cancelledOrNeverDispatchedMemberCount ==
        evidence.scopeMemberCount)
        return CompletionClassification::allScopeCancelled;
    if (evidence.completedMemberCount == evidence.scopeMemberCount &&
        evidence.anyMemberBound)
        return CompletionClassification::allDownstreamCompleted;
    throw Error(ErrorCode::invalidCompletionEvidence,
        "campaign_operations_completion_evidence_invalid");
}

} // namespace EA::CampaignOperations
