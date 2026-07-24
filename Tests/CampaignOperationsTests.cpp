#include "../Sources/CampaignOperations.hpp"

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
    AssertError([]
    {
        (void)ClassifyCompletion({3, 3, 0, 0, false, true, false});
    }, ErrorCode::invalidCompletionEvidence,
        "campaign_operations_completion_evidence_invalid");

    return 0;
}
