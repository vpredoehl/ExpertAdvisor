#include "../Sources/CampaignOperations.hpp"
#include "../Sources/CampaignOperationsCompletion.hpp"
#include "../Sources/CampaignOperationsControl.hpp"
#include "../Sources/CampaignOperationsDispatch.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>

using namespace EA::CampaignOperations;

namespace
{

template <typename Function>
void AssertError(Function&& function, ErrorCode expectedCode,
    const std::string& expectedReason)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const Error& error)
    {
        threw = true;
        assert(error.code() == expectedCode);
        assert(error.what() == expectedReason);
    }
    assert(threw);
}

OperationalCampaign BuildCampaign()
{
    const std::string canonical = "phase4d-materialization-canonical";
    return BuildOperationalCampaign(41, 1, canonical,
        EA::ExperimentRecommendation::RecommendationCanonicalHash(canonical),
        3);
}

} // namespace

int main()
{
    static_assert(!std::is_default_constructible_v<OperationalCampaignId>);
    static_assert(!std::is_assignable_v<OperationalCampaignId&,
        OperationalCampaignId>);
    static_assert(!std::is_copy_assignable_v<CanonicalIdentity>);
    static_assert(!std::is_copy_assignable_v<OperationalCampaign>);
    static_assert(!std::is_copy_assignable_v<GovernanceProvenanceEvent>);
    static_assert(!std::is_copy_assignable_v<
        OperationalAuthorizationEvent>);
    static_assert(!OperationalCampaign::mutableStatusPresent);
    static_assert(!OperationalCampaign::readinessEventPresent);
    static_assert(!OperationalCampaign::schedulerAuthorityGranted);
    static_assert(!OperationalCampaign::lifecycleAuthorityGranted);
    static_assert(!OperationalRequest::dispatchAuthorityGranted);
    static_assert(!OperationalRequest::schedulerAuthorityGranted);
    static_assert(!OperationalRequest::lifecycleAuthorityGranted);

    const OperationalCampaign campaign = BuildCampaign();
    const std::string expectedCampaignCanonical =
        "campaign_operations_campaign_v1"
        ";materialization_id=41"
        ";materialization_contract_version=1"
        ";materialization_canonical=33:phase4d-materialization-canonical"
        ";materialization_identity_hash=24:fnv1a64:c2eadfe30d7191a9"
        ";member_count=3"
        ";origin_kind=phase4d_materialization_v1"
        ";action_kind=dispatch_full_materialization"
        ";action_contract_version=1"
        ";scope_kind=complete_materialization"
        ";scope_contract_version=1";
    assert(campaign.identity.canonicalText() == expectedCampaignCanonical);
    assert(campaign.identity.hash() ==
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            expectedCampaignCanonical));
    assert(campaign.identity.hash() == "fnv1a64:fd300019cb593389");
    assert(InitialAdministrativeState(campaign) ==
        AdministrativeCampaignState::awaitingOperationalAuthorization);
    assert(ToText(InitialAdministrativeState(campaign)) ==
        "awaiting_operational_authorization");
    assert(ToText(campaign.originKind) == "phase4d_materialization_v1");
    assert(ToText(campaign.actionKind) == "dispatch_full_materialization");
    assert(ToText(campaign.scopeKind) == "complete_materialization");
    const std::string twoByteUtf8("\xC3\xA9", 2);
    const OperationalCampaign utf8Campaign = BuildOperationalCampaign(42, 1,
        twoByteUtf8,
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            twoByteUtf8),
        1);
    assert(utf8Campaign.identity.canonicalText().find(
        ";materialization_canonical=2:") != std::string::npos);

    AssertError([]
    {
        (void)AdministrativeCampaignStateFromText("drafted");
    }, ErrorCode::invalidEnumText, "campaign_operations_enum_text_invalid");
    AssertError([]
    {
        (void)AuthorizationEventKindFromText("superseded");
    }, ErrorCode::invalidEnumText, "campaign_operations_enum_text_invalid");
    assert(AuthorizationEventKindFromText("granted") ==
        AuthorizationEventKind::granted);
    assert(AuthorizationEventKindFromText("revoked") ==
        AuthorizationEventKind::revoked);
    assert(AuthorizationEventKindFromText("expiry_observed") ==
        AuthorizationEventKind::expiryObserved);

    AssertError([]
    {
        (void)BuildOperationalCampaign(0, 1, "x",
            EA::ExperimentRecommendation::RecommendationCanonicalHash("x"), 1);
    }, ErrorCode::invalidMaterialization,
        "campaign_operations_materialization_invalid");
    AssertError([]
    {
        (void)BuildOperationalCampaign(1, 1, "x",
            "fnv1a64:0000000000000000", 1);
    }, ErrorCode::invalidCanonicalHash,
        "campaign_operations_materialization_identity_invalid");
    AssertError([]
    {
        (void)ActorIdentity("actor with spaces");
    }, ErrorCode::invalidActorIdentity,
        "campaign_operations_actor_identity_invalid");
    AssertError([]
    {
        (void)Reason(" \t\n");
    }, ErrorCode::invalidReason, "campaign_operations_reason_invalid");
    AssertError([]
    {
        (void)UtcTimestamp("2026-07-22T12:00:00Z");
    }, ErrorCode::invalidTimestamp,
        "campaign_operations_timestamp_invalid");
    AssertError([]
    {
        (void)UtcTimestamp("2025-02-29T12:00:00.000000Z");
    }, ErrorCode::invalidTimestamp,
        "campaign_operations_timestamp_invalid");
    assert(UtcTimestamp("2024-02-29T12:00:00.000000Z").value() ==
        "2024-02-29T12:00:00.000000Z");

    const std::string ratificationCanonical = "phase6d-ratification";
    const std::string reviewCanonical = "phase6c-review";
    const std::string proposalCanonical = "phase6b-proposal";
    const GovernanceProvenanceEvent provenance =
        BuildGovernanceProvenanceEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), 73, 1, ratificationCanonical,
            EA::ExperimentRecommendation::RecommendationCanonicalHash(
                ratificationCanonical),
            72, 1, reviewCanonical,
            EA::ExperimentRecommendation::RecommendationCanonicalHash(
                reviewCanonical),
            71, 1, proposalCanonical,
            EA::ExperimentRecommendation::RecommendationCanonicalHash(
                proposalCanonical),
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1);
    assert(provenance.identity.canonicalText().starts_with(
        "campaign_operations_governance_provenance_v1;"));
    assert(provenance.identity.hash() ==
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            provenance.identity.canonicalText()));
    assert(provenance.identity.hash() == "fnv1a64:d3fbd06c3a047144");
    const GovernanceProvenanceEvent optionalProvenance =
        BuildGovernanceProvenanceEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), 73, 1, ratificationCanonical,
            EA::ExperimentRecommendation::RecommendationCanonicalHash(
                ratificationCanonical),
            72, 1, reviewCanonical,
            EA::ExperimentRecommendation::RecommendationCanonicalHash(
                reviewCanonical),
            71, 1, proposalCanonical,
            EA::ExperimentRecommendation::RecommendationCanonicalHash(
                proposalCanonical),
            PrerequisitePolicy::phase4dMaterializationOnlyV1);
    assert(optionalProvenance.prerequisitePolicy ==
        PrerequisitePolicy::phase4dMaterializationOnlyV1);

    const OperationalAuthorizationEvent grant =
        BuildOperationalAuthorizationEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::granted,
            OperationalActionKind::dispatchFullMaterialization, 1,
            ScopeKind::completeMaterialization, 1,
            PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
            std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole,
            ActorIdentity("authorizer@example.test"),
            Reason("Explicit foundation authorization evidence."),
            UtcTimestamp("2026-07-22T12:00:00.000000Z"), std::nullopt);
    assert(grant.identity.canonicalText().ends_with(
        ";not_before=2026-07-22T12:00:00.000000Z;expires_at=none"));
    assert(grant.identity.hash() == "fnv1a64:11a524cbf6d63d93");
    assert(IsAuthorizationEffectiveAt(grant,
        UtcTimestamp("2026-07-22T12:00:00.000000Z")));
    assert(IsAuthorizationEffectiveAt(grant,
        UtcTimestamp("2027-01-01T00:00:00.000000Z")));

    const OperationalAuthorizationEvent successor =
        BuildOperationalAuthorizationEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), AuthorizationEventId(11),
            grant.identity.canonicalText(), grant.identity.hash(), 2,
            AuthorizationEventKind::granted,
            OperationalActionKind::dispatchFullMaterialization, 1,
            ScopeKind::completeMaterialization, 1,
            PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
            std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole,
            ActorIdentity("successor@example.test"),
            Reason("One-event supersession successor."),
            UtcTimestamp("2026-07-23T00:00:00.000000Z"),
            UtcTimestamp("2026-08-01T00:00:00.000000Z"));
    assert(successor.eventKind == AuthorizationEventKind::granted);
    assert(successor.chainVersion == grant.chainVersion + 1);
    assert(successor.previousEventCanonicalText ==
        grant.identity.canonicalText());
    assert(!IsAuthorizationEffectiveAt(successor,
        UtcTimestamp("2026-07-22T23:59:59.999999Z")));
    assert(IsAuthorizationEffectiveAt(successor,
        UtcTimestamp("2026-07-23T00:00:00.000000Z")));
    assert(!IsAuthorizationEffectiveAt(successor,
        UtcTimestamp("2026-08-01T00:00:00.000000Z")));

    const OperationalAuthorizationEvent revoked =
        BuildOperationalAuthorizationEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), AuthorizationEventId(12),
            successor.identity.canonicalText(), successor.identity.hash(), 3,
            AuthorizationEventKind::revoked,
            OperationalActionKind::dispatchFullMaterialization, 1,
            ScopeKind::completeMaterialization, 1,
            PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
            std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole,
            ActorIdentity("authorizer@example.test"),
            Reason("Explicit revocation."),
            UtcTimestamp("2026-07-24T00:00:00.000000Z"), std::nullopt);
    assert(!IsAuthorizationEffectiveAt(revoked,
        UtcTimestamp("2026-07-24T00:00:00.000000Z")));

    AssertError([&]
    {
        (void)BuildOperationalAuthorizationEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 2, AuthorizationEventKind::granted,
            OperationalActionKind::dispatchFullMaterialization, 1,
            ScopeKind::completeMaterialization, 1,
            PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
            std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole, ActorIdentity("actor"),
            Reason("reason"),
            UtcTimestamp("2026-07-22T00:00:00.000000Z"), std::nullopt);
    }, ErrorCode::invalidAuthorizationEvent,
        "campaign_operations_authorization_predecessor_invalid");

    AssertError([&]
    {
        (void)BuildOperationalAuthorizationEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::revoked,
            OperationalActionKind::dispatchFullMaterialization, 1,
            ScopeKind::completeMaterialization, 1,
            PrerequisitePolicy::phase4dMaterializationOnlyV1, std::nullopt,
            std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole, ActorIdentity("actor"),
            Reason("reason"),
            UtcTimestamp("2026-07-22T00:00:00.000000Z"), std::nullopt);
    }, ErrorCode::invalidAuthorizationEvent,
        "campaign_operations_authorization_initial_event_invalid");

    AssertError([&]
    {
        (void)BuildOperationalAuthorizationEvent(OperationalCampaignId(7),
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 1, AuthorizationEventKind::granted,
            OperationalActionKind::dispatchFullMaterialization, 1,
            ScopeKind::completeMaterialization, 1,
            PrerequisitePolicy::
                phase4dMaterializationPlusExactPhase6dRatificationV1,
            std::nullopt, std::nullopt, std::nullopt,
            kCampaignOperationsAuthorizationRole, ActorIdentity("actor"),
            Reason("reason"),
            UtcTimestamp("2026-07-22T00:00:00.000000Z"), std::nullopt);
    }, ErrorCode::invalidAuthorizationEvent,
        "campaign_operations_authorization_provenance_invalid");

    const BudgetAccounting active = CalculateBudgetAccounting(
        10, 8, 3, 2, BudgetLedgerStatus::active);
    assert(active.held == 3);
    assert(active.arithmeticallyUnallocated == 4);
    assert(active.reservable == 4);
    const BudgetAccounting inactive = CalculateBudgetAccounting(
        10, 8, 3, 2, BudgetLedgerStatus::revoked);
    assert(inactive.arithmeticallyUnallocated == 4);
    assert(inactive.reservable == 0);
    AssertError([]
    {
        (void)CalculateBudgetAccounting(
            3, 5, 3, 0, BudgetLedgerStatus::active);
    }, ErrorCode::invalidBudgetAccounting,
        "campaign_operations_budget_accounting_invalid");

    const LogicalOperation operation =
        BuildLogicalOperation(OperationalCampaignId(7), campaign);
    assert(operation.identity.canonicalText().starts_with(
        "campaign_operations_logical_operation_v1;"));
    assert(operation.identity.hash() ==
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            operation.identity.canonicalText()));
    assert(operation.identity.hash() == "fnv1a64:f254a98743e91370");
    const BudgetLedgerEntry budget = BuildBudgetLedgerEntry(
        OperationalCampaignId(7), campaign.identity.canonicalText(),
        std::nullopt, std::nullopt, std::nullopt, 1,
        BudgetLedgerEntryKind::grant, BudgetLedgerStatus::active,
        BudgetUnit::materializedMemberDispatch, 3, 0, 3,
        ActorIdentity("budget.admin@example.test"),
        Reason("Grant one complete three-member operation."));
    assert(budget.identity.canonicalText().starts_with(
        "campaign_operations_budget_ledger_entry_v1;"));
    assert(budget.identity.hash() ==
        EA::ExperimentRecommendation::RecommendationCanonicalHash(
            budget.identity.canonicalText()));
    assert(budget.identity.hash() == "fnv1a64:5258a9d40a8c3c38");
    const Reservation reservation = BuildReservation(operation,
        AuthorizationEventId(19), grant.identity.canonicalText(),
        grant.identity.hash(), BudgetLedgerEntryId(23), 1,
        budget.identity.canonicalText(), budget.identity.hash(), 3, 3,
        BudgetUnit::materializedMemberDispatch,
        UtcTimestamp("2026-08-01T00:00:00.000000Z"));
    assert(reservation.identity.canonicalText().starts_with(
        "campaign_operations_reservation_v1;"));
    assert(reservation.identity.hash() == "fnv1a64:edf40f5e202de730");
    assert(reservation.amount == reservation.memberCount);
    const OperationalRequest request = BuildOperationalRequest(operation,
        AuthorizationEventId(19), grant.identity.canonicalText(),
        grant.identity.hash(), ReservationId(29),
        reservation.identity.canonicalText(), reservation.identity.hash(), 3,
        campaign.materializationIdentityHash,
        ActorIdentity("requester@example.test"),
        Reason("Accept one durable operation."),
        PrerequisitePolicy::phase4dMaterializationOnlyV1,
        std::nullopt, std::nullopt);
    assert(request.identity.canonicalText().starts_with(
        "campaign_operations_request_v1;"));
    assert(request.identity.hash() == "fnv1a64:2c6c3922481b636c");
    assert(request.logicalOperation.identity ==
        reservation.logicalOperation.identity);
    const ReservationEvent acquisition =
        BuildReservationAcquisitionEvent(ReservationId(29),
            reservation.identity.canonicalText(), OperationalRequestId(31),
            request.identity.canonicalText(), 3);
    assert(acquisition.identity.canonicalText().starts_with(
        "campaign_operations_reservation_event_v1;"));
    assert(acquisition.eventKind == ReservationEventKind::acquired);
    assert(!acquisition.expectedState);
    assert(acquisition.resultingState == ReservationState::held);
    assert(acquisition.expectedVersion == 0);
    assert(acquisition.resultingVersion == 1);
    assert(acquisition.identity.hash() == "fnv1a64:20459a7ac198ce34");

    AssertError([&]
    {
        (void)BuildBudgetLedgerEntry(OperationalCampaignId(7),
            campaign.identity.canonicalText(), std::nullopt, std::nullopt,
            std::nullopt, 2, BudgetLedgerEntryKind::amend,
            BudgetLedgerStatus::active,
            BudgetUnit::materializedMemberDispatch, 1, 3, 4,
            ActorIdentity("budget.admin@example.test"), Reason("invalid"));
    }, ErrorCode::invalidBudgetLedgerEntry,
        "campaign_operations_budget_ledger_entry_invalid");
    AssertError([&]
    {
        (void)BuildReservation(operation, AuthorizationEventId(19),
            grant.identity.canonicalText(), grant.identity.hash(),
            BudgetLedgerEntryId(23), 1, budget.identity.canonicalText(),
            budget.identity.hash(), 3, 2,
            BudgetUnit::materializedMemberDispatch, std::nullopt);
    }, ErrorCode::invalidReservation,
        "campaign_operations_reservation_invalid");
    AssertError([&]
    {
        (void)BuildOperationalRequest(operation, AuthorizationEventId(19),
            grant.identity.canonicalText(), grant.identity.hash(),
            ReservationId(29), reservation.identity.canonicalText(),
            reservation.identity.hash(), 3,
            "fnv1a64:0000000000000000",
            ActorIdentity("requester@example.test"), Reason("invalid"),
            PrerequisitePolicy::phase4dMaterializationOnlyV1,
            std::nullopt, std::nullopt);
    }, ErrorCode::invalidOperationalRequest,
        "campaign_operations_request_invalid");
    AssertError([]
    {
        (void)CalculateBudgetAccounting(3, 0, 0, 0,
            static_cast<BudgetLedgerStatus>(99));
    }, ErrorCode::invalidBudgetAccounting,
        "campaign_operations_budget_accounting_invalid");
    AssertError([]
    {
        (void)CalculateBudgetAccounting(
            std::numeric_limits<long long>::max(),
            std::numeric_limits<long long>::max(),
            std::numeric_limits<long long>::max(), 1,
            BudgetLedgerStatus::active);
    }, ErrorCode::invalidBudgetAccounting,
        "campaign_operations_budget_accounting_invalid");

    assert(ClassifyCompletion({3, 0, 0, 3, false, false, true}) ==
        CompletionClassification::allScopeCancelled);
    assert(ClassifyCompletion({3, 3, 0, 0, false, true, true}) ==
        CompletionClassification::allDownstreamCompleted);
    assert(ClassifyCompletion({3, 1, 1, 1, false, true, true}) ==
        CompletionClassification::mixedTerminalOutcomes);
    assert(ClassifyCompletion({3, 0, 3, 0, false, true, true}) ==
        CompletionClassification::downstreamFailure);
    assert(ClassifyCompletion({3, 2, 0, 1, false, true, true}) ==
        CompletionClassification::terminalPartialCompletion);
    assert(ClassifyCompletion({3, 0, 0, 3, true, false, true}) ==
        CompletionClassification::operationalRequestFailed);
    // Exhaustive precedence pairs: failed evidence dominates every mixed
    // terminal alternative, while completed plus cancelled/never-dispatched
    // remains a successful partial operational completion.
    assert(ClassifyCompletion({2, 1, 1, 0, false, true, true}) ==
        CompletionClassification::mixedTerminalOutcomes);
    assert(ClassifyCompletion({2, 0, 1, 1, false, true, true}) ==
        CompletionClassification::mixedTerminalOutcomes);
    assert(ClassifyCompletion({3, 0, 1, 2, false, true, true}) ==
        CompletionClassification::mixedTerminalOutcomes);
    assert(ClassifyCompletion({2, 1, 0, 1, false, true, true}) ==
        CompletionClassification::terminalPartialCompletion);
    assert(ClassifyCompletion({3, 1, 0, 2, false, true, true}) ==
        CompletionClassification::terminalPartialCompletion);
    assert(ClassifyCompletion({2, 0, 0, 2, false, false, true}) ==
        CompletionClassification::allScopeCancelled);
    assert(ClassifyCompletion({2, 0, 0, 2, true, false, true}) ==
        CompletionClassification::operationalRequestFailed);
    AssertError([]
    {
        (void)ClassifyCompletion({2, 0, 0, 2, true, true, true});
    }, ErrorCode::invalidCompletionEvidence,
        "campaign_operations_completion_evidence_invalid");
    AssertError([]
    {
        (void)ClassifyCompletion({2, 1, 1, 0, false, true, false});
    }, ErrorCode::invalidCompletionEvidence,
        "campaign_operations_completion_evidence_invalid");
    AssertError([]
    {
        (void)ClassifyCompletion({3, 3, 0, 0, false, true, false});
    }, ErrorCode::invalidCompletionEvidence,
        "campaign_operations_completion_evidence_invalid");
    const auto evidenceIdentity = [](const std::string& text)
    {
        return CanonicalIdentity::Create(
            kCampaignOperationsCompletionContractVersion, text);
    };
    const auto completion = BuildCompletionEvent(
        OperationalCampaignId(7), campaign.identity.canonicalText(),
        "complete-all-cancelled",
        AdministrativeCampaignState::terminalCancelled,
        CompletionClassification::allScopeCancelled,
        BudgetLedgerEntryId(23), 1, 3, 3, 0, 3, 0, 3,
        3, 0, 0, 3, 1, 1, 0, 0, 1, 1, 0,
        evidenceIdentity("authorization-evidence"),
        evidenceIdentity("budget-evidence"),
        evidenceIdentity("reservation-evidence"),
        evidenceIdentity("request-evidence"),
        evidenceIdentity("binding-evidence"),
        evidenceIdentity("lifecycle-evidence"),
        evidenceIdentity("cancellation-evidence"),
        evidenceIdentity("reconciliation-evidence"),
        ActorIdentity("completion.operator@example.test"),
        Reason("Record exact settled operational evidence."));
    ValidateCompletionEvent(completion);
    assert(completion.identity.canonicalText().rfind(
        "campaign_operations_completion_v1;", 0) == 0);
    const auto changedCompletion = BuildCompletionEvent(
        OperationalCampaignId(7), campaign.identity.canonicalText(),
        "different-replay",
        AdministrativeCampaignState::terminalCancelled,
        CompletionClassification::allScopeCancelled,
        BudgetLedgerEntryId(23), 1, 3, 3, 0, 3, 0, 3,
        3, 0, 0, 3, 1, 1, 0, 0, 1, 1, 0,
        evidenceIdentity("authorization-evidence"),
        evidenceIdentity("budget-evidence"),
        evidenceIdentity("reservation-evidence"),
        evidenceIdentity("request-evidence"),
        evidenceIdentity("binding-evidence"),
        evidenceIdentity("lifecycle-evidence"),
        evidenceIdentity("cancellation-evidence"),
        evidenceIdentity("reconciliation-evidence"),
        ActorIdentity("completion.operator@example.test"),
        Reason("Record exact settled operational evidence."));
    assert(completion.identity != changedCompletion.identity);
    const auto changedEvidenceCompletion = BuildCompletionEvent(
        OperationalCampaignId(7), campaign.identity.canonicalText(),
        "complete-all-cancelled",
        AdministrativeCampaignState::terminalCancelled,
        CompletionClassification::allScopeCancelled,
        BudgetLedgerEntryId(23), 1, 3, 3, 0, 3, 0, 3,
        3, 0, 0, 3, 1, 1, 0, 0, 1, 1, 0,
        evidenceIdentity("authorization-evidence"),
        evidenceIdentity("budget-evidence"),
        evidenceIdentity("reservation-evidence"),
        evidenceIdentity("changed-request-evidence"),
        evidenceIdentity("binding-evidence"),
        evidenceIdentity("lifecycle-evidence"),
        evidenceIdentity("cancellation-evidence"),
        evidenceIdentity("reconciliation-evidence"),
        ActorIdentity("completion.operator@example.test"),
        Reason("Record exact settled operational evidence."));
    assert(completion.identity != changedEvidenceCompletion.identity);
    assert(completion.requestEvidence.canonicalText() !=
        changedEvidenceCompletion.requestEvidence.canonicalText());
    const auto rebuildCompletion = [&](const std::string& operationKey,
                                       CanonicalIdentity cancellationEvidence,
                                       CanonicalIdentity reconciliationEvidence,
                                       ActorIdentity actor, Reason reason)
    {
        return BuildCompletionEvent(completion.campaignId,
            completion.campaignCanonicalText, operationKey,
            completion.terminalState, completion.classification,
            completion.budgetLedgerEntryId, completion.budgetLedgerVersion,
            completion.budgetResultingTotal, completion.budgetEverReserved,
            completion.budgetCommitted, completion.budgetReleasedOrExpired,
            completion.budgetHeld, completion.budgetUnallocated,
            completion.scopeMemberCount, completion.completedMemberCount,
            completion.failedMemberCount,
            completion.cancelledOrNeverDispatchedMemberCount,
            completion.reservationCount, completion.requestCount,
            completion.bindingCount, completion.controlOwnerCount,
            completion.cancellationRequestCount,
            completion.cancellationSettlementCount,
            completion.unresolvedBlockingObservationCount,
            completion.authorizationEvidence, completion.budgetEvidence,
            completion.reservationEvidence, completion.requestEvidence,
            completion.bindingEvidence, completion.lifecycleEvidence,
            std::move(cancellationEvidence),
            std::move(reconciliationEvidence), std::move(actor),
            std::move(reason));
    };
    assert(rebuildCompletion(completion.operationKey,
               evidenceIdentity("changed-cancellation-evidence"),
               completion.reconciliationEvidence, completion.actor,
               completion.reason).identity != completion.identity);
    assert(rebuildCompletion(completion.operationKey,
               completion.cancellationEvidence,
               evidenceIdentity("changed-reconciliation-evidence"),
               completion.actor, completion.reason).identity !=
        completion.identity);
    assert(rebuildCompletion("changed-operation-key",
               completion.cancellationEvidence,
               completion.reconciliationEvidence, completion.actor,
               completion.reason).identity != completion.identity);
    assert(rebuildCompletion(completion.operationKey,
               completion.cancellationEvidence,
               completion.reconciliationEvidence,
               ActorIdentity("changed.completer@example.test"),
               completion.reason).identity != completion.identity);
    assert(rebuildCompletion(completion.operationKey,
               completion.cancellationEvidence,
               completion.reconciliationEvidence, completion.actor,
               Reason("Changed completion reason.")).identity !=
        completion.identity);
    const auto canonicalLeft = evidenceIdentity("canonical-left");
    const auto canonicalRight = evidenceIdentity("canonical-right");
    AssertError([&]
    {
        (void)CanonicalIdentity::Hydrate(
            kCampaignOperationsCompletionContractVersion,
            canonicalLeft.canonicalText(), canonicalRight.hash());
    }, ErrorCode::invalidCanonicalHash,
        "campaign_operations_identity_invalid");
    AssertError([&]
    {
        (void)BuildCompletionEvent(
            OperationalCampaignId(7), campaign.identity.canonicalText(),
            "paused-is-not-complete", AdministrativeCampaignState::paused,
            CompletionClassification::allScopeCancelled,
            BudgetLedgerEntryId(23), 1, 3, 3, 0, 3, 0, 3,
            3, 0, 0, 3, 1, 1, 0, 0, 1, 1, 0,
            evidenceIdentity("authorization-evidence"),
            evidenceIdentity("budget-evidence"),
            evidenceIdentity("reservation-evidence"),
            evidenceIdentity("request-evidence"),
            evidenceIdentity("binding-evidence"),
            evidenceIdentity("lifecycle-evidence"),
            evidenceIdentity("cancellation-evidence"),
            evidenceIdentity("reconciliation-evidence"),
            ActorIdentity("completion.operator@example.test"),
            Reason("Pause cannot masquerade as completion."));
    }, ErrorCode::invalidCompletionEvidence,
        "campaign_operations_completion_event_invalid");

    const LeaseTokenDigest leaseDigest = LeaseTokenDigest::Derive(
        "0123456789abcdef0123456789abcdef");
    const auto attempt = BuildDispatchAttemptAcquisition(
        OperationalRequestId(31), request.identity.canonicalText(), 1, 1, 2,
        leaseDigest, UtcTimestamp("2026-07-25T12:00:00.000000Z"),
        ActorIdentity("phase3.dispatcher@example.test"));
    const auto firstBinding = BuildRequestMemberBinding(
        OperationalRequestId(31), request.identity.canonicalText(), 41,
        "materialization-canonical", 4101, 1, "selected-one",
        "fnv1a64:0000000000000001", 4201, "proposal-one",
        "fnv1a64:0000000000000002", 4301, 4401, "execution-one",
        "fnv1a64:0000000000000003", 4501, "activation-one",
        "fnv1a64:0000000000000004", 4601, BindingDisposition::created,
        Phase5ExecutionDisposition::created,
        Phase5ActivationDisposition::created);
    const auto secondBinding = BuildRequestMemberBinding(
        OperationalRequestId(31), request.identity.canonicalText(), 41,
        "materialization-canonical", 4102, 2, "selected-two",
        "fnv1a64:0000000000000005", 4202, "proposal-two",
        "fnv1a64:0000000000000006", 4302, 4402, "execution-two",
        "fnv1a64:0000000000000007", 4502, "activation-two",
        "fnv1a64:0000000000000008", 4602,
        BindingDisposition::created,
        Phase5ExecutionDisposition::created,
        Phase5ActivationDisposition::created);
    AssertError([&]
    {
        (void)BuildRequestMemberBinding(
            OperationalRequestId(31), request.identity.canonicalText(), 41,
            "materialization-canonical", 4103, 3, "selected-three",
            "fnv1a64:000000000000000a", 4203, "proposal-three",
            "fnv1a64:000000000000000b", 4303, 4403,
            "execution-three", "fnv1a64:000000000000000c", 4503,
            "activation-three", "fnv1a64:000000000000000d", 4603,
            BindingDisposition::created,
            Phase5ExecutionDisposition::reused,
            Phase5ActivationDisposition::created);
    }, ErrorCode::invalidOperationalRequest,
        "campaign_operations_binding_disposition_invalid");
    const auto bindingSet = BuildRequestBindingSet(
        OperationalRequestId(31), request.identity.canonicalText(),
        {firstBinding, secondBinding});
    const auto owner = BuildDownstreamControlOwner(
        OperationalRequestId(31), request.identity.canonicalText(),
        secondBinding.identity.canonicalText(), 4602,
        DownstreamControlMode::authorizedAdoptionControl,
        AuthorizationEventId(47), "adoption-authorization",
        "fnv1a64:0000000000000009");
    const auto commitment = BuildReservationCommitment(
        ReservationId(29), reservation.identity.canonicalText(),
        OperationalRequestId(31), request.identity.canonicalText(),
        bindingSet.identity.canonicalText(), bindingSet.identity.hash(),
        1, 2, 3);
    const auto outcome = BuildDispatchAttemptOutcomeEvidence(
        DispatchAttemptId(51), attempt.identity.canonicalText(),
        DispatchResultClassification::createdAndBound,
        DownstreamEvidenceClassification::noPhase5Evidence,
        SemanticConflictClassification::none,
        UncertainCommitRecoveryClassification::provenNoCommit,
        "dispatch_created_and_bound", 2, 3, 1, 2,
        bindingSet.identity.canonicalText(), bindingSet.identity.hash());
    assert(leaseDigest.value() == "fnv1a64:01527c9731f0ff55");
    assert(attempt.identity.hash() == "fnv1a64:8a2997f32b017e15");
    assert(firstBinding.identity.hash() == "fnv1a64:5a5d3859fafb3d66");
    assert(secondBinding.identity.hash() == "fnv1a64:478d91a756482217");
    assert(bindingSet.identity.hash() == "fnv1a64:3a32410aa5f73f02");
    assert(owner.identity.hash() == "fnv1a64:337c848f545dad02");
    assert(commitment.identity.hash() == "fnv1a64:dd92c2c4cd231c39");
    assert(outcome.identity.hash() == "fnv1a64:b9275d5d0e0f49f8");

    const auto pause = BuildCampaignControlEvent(
        OperationalCampaignId(17), "campaign-canonical", std::nullopt,
        std::nullopt, 1, ControlEventKind::pause,
        ActorIdentity("phase4.operator@example.test"),
        Reason("Pause future campaign operations."));
    const auto pauseReplay = BuildCampaignControlEvent(
        OperationalCampaignId(17), "campaign-canonical", std::nullopt,
        std::nullopt, 1, ControlEventKind::pause,
        ActorIdentity("phase4.operator@example.test"),
        Reason("Pause future campaign operations."));
    assert(pause == pauseReplay);
    assert(pause.identity.canonicalText().rfind(
        "campaign_operations_control_event_v1;", 0) == 0);
    const auto resume = BuildCampaignControlEvent(
        OperationalCampaignId(17), "campaign-canonical",
        ControlEventId(71), pause.identity.canonicalText(), 2,
        ControlEventKind::resume,
        ActorIdentity("phase4.operator@example.test"),
        Reason("Resume future campaign operations."));
    assert(resume.identity != pause.identity);
    AssertError([&]
    {
        (void)BuildCampaignControlEvent(
            OperationalCampaignId(17), "campaign-canonical",
            std::nullopt, std::nullopt, 1, ControlEventKind::resume,
            ActorIdentity("phase4.operator@example.test"),
            Reason("A control chain cannot begin with resume."));
    }, ErrorCode::invalidControlEvent,
        "campaign_operations_control_chain_invalid");

    const auto cancellation = BuildCampaignCancellationRequest(
        OperationalCampaignId(17), "campaign-canonical",
        OperationalRequestId(31), request.identity.canonicalText(),
        RequestState::dispatching, 2, "operator-cancel-31",
        ActorIdentity("phase4.operator@example.test"),
        Reason("Stop future work and coordinate cancellation."));
    assert(cancellation.identity.canonicalText().rfind(
        "campaign_operations_cancellation_request_v1;", 0) == 0);
    assert(BuildCampaignCancellationRequest(
        OperationalCampaignId(17), "campaign-canonical",
        OperationalRequestId(31), request.identity.canonicalText(),
        RequestState::dispatching, 2, "operator-cancel-31",
        ActorIdentity("phase4.operator@example.test"),
        Reason("Stop future work and coordinate cancellation.")) ==
        cancellation);
    AssertError([&]
    {
        (void)BuildCampaignCancellationRequest(
            OperationalCampaignId(17), "campaign-canonical",
            OperationalRequestId(31), std::nullopt,
            RequestState::ready, 1, "operator-cancel-31",
            ActorIdentity("phase4.operator@example.test"),
            Reason("Incomplete target evidence must fail."));
    }, ErrorCode::invalidCancellationRequest,
        "campaign_operations_cancellation_target_shape_invalid");

    const auto unboundSettlement =
        BuildCampaignCancellationSettlement(
            CancellationRequestId(81),
            cancellation.identity.canonicalText(),
            CancellationSettlementDisposition::unboundCancelled,
            ReservationEventId(91), "reservation-release-canonical", 3,
            std::nullopt, std::nullopt);
    assert(unboundSettlement.identity.canonicalText().rfind(
        "campaign_operations_cancellation_settlement_v1;", 0) == 0);
    const auto lifecycleSettlement =
        BuildCampaignCancellationSettlement(
            CancellationRequestId(82),
            cancellation.identity.canonicalText(),
            CancellationSettlementDisposition::alreadyTerminal,
            std::nullopt, std::nullopt, std::nullopt,
            "lifecycle-evidence-canonical",
            "fnv1a64:0000000000000001");
    assert(lifecycleSettlement.disposition ==
        CancellationSettlementDisposition::alreadyTerminal);
    AssertError([&]
    {
        (void)BuildCampaignCancellationSettlement(
            CancellationRequestId(82),
            cancellation.identity.canonicalText(),
            CancellationSettlementDisposition::unboundCancelled,
            std::nullopt, std::nullopt, 3, std::nullopt, std::nullopt);
    }, ErrorCode::invalidCancellationSettlement,
        "campaign_operations_cancellation_settlement_evidence_invalid");

    const auto observation = BuildReconciliationObservation(
        "restart-run-1", OperationalCampaignId(17),
        OperationalRequestId(31), request.identity.canonicalText(),
        RequestState::dispatching, 2,
        ReconciliationReason::
            dispatchLeaseExpiredNoDownstreamEvidence,
        "lease-expired-no-downstream-evidence",
        "campaign_operations_dispatch_recovery",
        "clear_stale_dispatch_lease",
        "dispatch_lease_expired_no_downstream_evidence");
    assert(observation.identity.canonicalText().rfind(
        "campaign_operations_reconciliation_observation_v1;", 0) == 0);
    assert(BuildReconciliationObservation(
        "restart-run-1", OperationalCampaignId(17),
        OperationalRequestId(31), request.identity.canonicalText(),
        RequestState::dispatching, 2,
        ReconciliationReason::
            dispatchLeaseExpiredNoDownstreamEvidence,
        "lease-expired-no-downstream-evidence",
        "campaign_operations_dispatch_recovery",
        "clear_stale_dispatch_lease",
        "dispatch_lease_expired_no_downstream_evidence") ==
        observation);
    AssertError([&]
    {
        (void)BuildReconciliationObservation(
            "restart-run-1", OperationalCampaignId(17),
            OperationalRequestId(31), request.identity.canonicalText(),
            RequestState::dispatching, 2,
            ReconciliationReason::
                dispatchLeaseExpiredNoDownstreamEvidence,
            "lease-expired-no-downstream-evidence",
            "campaign_operations_dispatch_recovery",
            "clear_stale_dispatch_lease", "Uppercase_Diagnostic");
    }, ErrorCode::invalidReconciliationObservation,
        "campaign_operations_reconciliation_diagnostic_invalid");
    const auto resolution = BuildReconciliationResolution(
        ReconciliationObservationId(101),
        observation.identity.canonicalText(),
        "campaign_operations_dispatch_recovery",
        kCampaignOperationsRecoveryRole,
        "dispatch-recovery-transition",
        CanonicalIdentity::Create(
            1, "dispatch-recovery-transition").hash(),
        "request_returned_ready");
    assert(resolution.identity.canonicalText().rfind(
        "campaign_operations_reconciliation_resolution_v1;", 0) == 0);
    AssertError([&]
    {
        (void)BuildReconciliationResolution(
            ReconciliationObservationId(101),
            observation.identity.canonicalText(),
            "campaign_operations_dispatch_recovery",
            kCampaignOperationsReconcilerRole,
            "dispatch-recovery-transition",
            CanonicalIdentity::Create(
                1, "dispatch-recovery-transition").hash(),
            "request_returned_ready");
    }, ErrorCode::invalidReconciliationResolution,
        "campaign_operations_reconciliation_resolution_owner_invalid");

    return 0;
}
