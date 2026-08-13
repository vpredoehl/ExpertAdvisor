#include "../Sources/CampaignOperationsProductionAdmission.hpp"
#include "../Sources/CampaignOperationsProductionAdmissionService.hpp"

#include <cassert>
#include <functional>
#include <string>
#include <type_traits>

using namespace EA::CampaignOperations;

namespace
{

template <typename Function>
void AssertError(Function&& function, ErrorCode code)
{
    bool threw = false;
    try { function(); }
    catch (const Error& error)
    {
        threw = true;
        assert(error.code() == code);
    }
    assert(threw);
}

void AssertContains(const std::string& value, const std::string& expected)
{
    assert(value.find(expected) != std::string::npos);
}

} // namespace

int main()
{
    assert(IsValidProductionOperationKey("canary-001/attempt:1"));
    assert(IsValidProductionOperationKey(std::string(128, 'a')));
    assert(!IsValidProductionOperationKey(""));
    assert(!IsValidProductionOperationKey("1 canary"));
    assert(!IsValidProductionOperationKey("_leading"));
    assert(!IsValidProductionOperationKey(std::string(129, 'a')));

    static_assert(!std::is_copy_assignable_v<SchedulerProtocolEvidence>);
    static_assert(!std::is_copy_assignable_v<ManagerBuildContract>);
    static_assert(!std::is_copy_assignable_v<ProductionEnableEvent>);
    static_assert(!std::is_copy_assignable_v<RequestProductionAdmission>);
    static_assert(!std::is_copy_assignable_v<ProductionDispatchAttemptV2>);

    const auto scheduler = BuildSchedulerProtocolEvidence(52, "complete",
        UtcTimestamp("2026-07-31T12:34:56.123456Z"), "scheduler.owner",
        "/Applications/LSTM_Release", "cutover-process-evidence");
    const std::string expectedScheduler =
        "campaign_operations_scheduler_protocol_evidence_v1"
        ";required_generation=52;cutover_state=complete"
        ";migration_contract=61:migration-052-scheduler-generation-52-exact-attempt-authority"
        ";protocol_contract=50:scheduler-generation-52-exact-attempt-authority-v1"
        ";cutover_completed_at=27:2026-07-31T12:34:56.123456Z"
        ";cutover_completed_by=15:scheduler.owner"
        ";cutover_executable_path=26:/Applications/LSTM_Release"
        ";cutover_process_evidence=24:cutover-process-evidence";
    assert(scheduler.identity.canonicalText() == expectedScheduler);
    assert(scheduler.identity.hash() == "fnv1a64:af614995e691e378");

    const auto build = BuildManagerBuildContract(kManagerServiceContract,
        std::string(40, 'a'), "Apple clang 21.0.0",
        "sha256:" + std::string(64, 'b'));
    const std::string expectedBuild =
        "campaign_operations_manager_build_v1"
        ";manager_service_contract=63:campaign-operations-production-dispatch-and-manager-run-once-v1"
        ";source_commit=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ";source_tree_state=clean;build_configuration=Release"
        ";compiler_contract=18:Apple clang 21.0.0"
        ";executable_sha256=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ";build_contract_version=1";
    assert(build.identity.canonicalText() == expectedBuild);
    assert(build.identity.hash() == "fnv1a64:9afc5f65cbaaece3");

    const auto enable = BuildProductionEnableEvent("enable-001",
        std::nullopt, "", 0, 1, scheduler,
        "cee://phase-h/generation-52",
        ActorIdentity("operator@example.test"), kManagerServiceContract,
        build, Reason("Enable verified H1 evidence."));
    assert(enable.identity.canonicalText().size() == 1399U);
    assert(enable.identity.hash() == "fnv1a64:05e8f676db9ba009");
    assert(enable.identity.canonicalText().find(
        ";predecessor_event_canonical=0:") != std::string::npos);

    const auto disable = BuildProductionDisableEvent("disable-001",
        ProductionEnablementEventId(1), enable.identity.canonicalText(), 1, 2,
        ActorIdentity("operator@example.test"),
        Reason("Disable verified H1 evidence."));
    assert(disable.identity.hash() == "fnv1a64:03730d0ccf99db85");

    const auto admission = BuildRequestProductionAdmission(
        OperationalRequestId(7), "request-canonical-v1", 3, "dispatch-001",
        ProductionEnablementEventId(1), enable.identity.canonicalText(),
        ActorIdentity("manager@example.test"), "campaign_manager_login",
        build);
    assert(admission.identity.canonicalText().size() == 2244U);
    assert(admission.identity.hash() == "fnv1a64:674a534f93f43c48");

    const auto attempt = BuildProductionDispatchAttemptV2(
        OperationalRequestId(7), "request-canonical-v1", admission,
        ProductionEnablementEventId(1), enable.identity.canonicalText(),
        "dispatch-001", 1, 3, 4,
        LeaseTokenDigest::Hydrate("fnv1a64:0123456789abcdef"),
        UtcTimestamp("2026-07-31T12:39:56.123456Z"),
        ActorIdentity("manager@example.test"), "campaign_manager_login",
        build);
    assert(attempt.identity.hash() == "fnv1a64:17a81d69b45dc93a");
    assert(attempt.identity.canonicalText().find(
        ";request_production_admission_canonical=2244:") !=
        std::string::npos);
    assert(attempt.identity.canonicalText().find(
        ";enable_event_canonical=1399:") != std::string::npos);

    const auto utf8Scheduler = BuildSchedulerProtocolEvidence(52, "complete",
        UtcTimestamp("2026-07-31T12:34:56.123456Z"),
        std::string("\xC3\xA9", 2), "/x", "p");
    assert(utf8Scheduler.identity.canonicalText().find(
        ";cutover_completed_by=2:") != std::string::npos);

    AssertError([&]
    {
        (void)BuildSchedulerProtocolEvidence(53, "complete",
            UtcTimestamp("2026-07-31T12:34:56.123456Z"), "a", "/x", "p");
    }, ErrorCode::unsupportedContractVersion);
    AssertError([&]
    {
        (void)BuildManagerBuildContract(kManagerServiceContract,
            std::string(39, 'a'), "compiler", "sha256:" +
                std::string(64, 'b'));
    }, ErrorCode::invalidCanonicalHash);
    AssertError([&]
    {
        (void)BuildProductionEnableEvent("bad key", std::nullopt, "", 0, 1,
            scheduler, "verification", ActorIdentity("operator"),
            kManagerServiceContract, build, Reason("reason"));
    }, ErrorCode::invalidCanonicalText);
    for (const char leading : std::string("._:/-"))
        AssertError([&]
        {
            (void)BuildProductionEnableEvent(
                std::string(1, leading) + "bad", std::nullopt, "", 0, 1,
                scheduler, "verification", ActorIdentity("operator"),
                kManagerServiceContract, build, Reason("reason"));
        }, ErrorCode::invalidCanonicalText);
    for (const std::string& validKey : {
             std::string("A"), std::string("z._:/-09"),
             std::string("0") + std::string(127, '-')})
        (void)BuildProductionEnableEvent(validKey, std::nullopt, "", 0, 1,
            scheduler, "verification", ActorIdentity("operator"),
            kManagerServiceContract, build, Reason("reason"));
    AssertError([&]
    {
        (void)BuildProductionEnableEvent(
            std::string("A") + std::string(128, '-'), std::nullopt, "", 0, 1,
            scheduler, "verification", ActorIdentity("operator"),
            kManagerServiceContract, build, Reason("reason"));
    }, ErrorCode::invalidCanonicalText);
    AssertError([&]
    {
        (void)BuildProductionEnableEvent(
            std::string("A\xC3\xA9", 3), std::nullopt, "", 0, 1,
            scheduler, "verification", ActorIdentity("operator"),
            kManagerServiceContract, build, Reason("reason"));
    }, ErrorCode::invalidCanonicalText);
    AssertError([&]
    {
        (void)BuildProductionDispatchAttemptV2(OperationalRequestId(8),
            "request-canonical-v1", admission,
            ProductionEnablementEventId(1), enable.identity.canonicalText(),
            "dispatch-001", 1, 3, 4,
            LeaseTokenDigest::Hydrate("fnv1a64:0123456789abcdef"),
            UtcTimestamp("2026-07-31T12:39:56.123456Z"),
            ActorIdentity("manager@example.test"),
            "campaign_manager_login", build);
    }, ErrorCode::invalidCanonicalText);
    const auto mismatchedKeyAdmission = BuildRequestProductionAdmission(
        OperationalRequestId(7), "request-canonical-v1", 3, "dispatch-002",
        ProductionEnablementEventId(1), enable.identity.canonicalText(),
        ActorIdentity("manager@example.test"), "campaign_manager_login", build);
    const auto laterKeyAttempt = BuildProductionDispatchAttemptV2(
        OperationalRequestId(7), "request-canonical-v1",
        mismatchedKeyAdmission, ProductionEnablementEventId(1),
        enable.identity.canonicalText(), "dispatch-001", 2, 5, 6,
        LeaseTokenDigest::Hydrate("fnv1a64:0123456789abcdef"),
        UtcTimestamp("2026-07-31T12:39:56.123456Z"),
        ActorIdentity("other.manager@example.test"),
        "other_campaign_manager_login", build);
    assert(laterKeyAttempt.admission.dispatchOperationKey == "dispatch-002");
    assert(laterKeyAttempt.operationKey == "dispatch-001");
    const auto mismatchedVersionAdmission = BuildRequestProductionAdmission(
        OperationalRequestId(7), "request-canonical-v1", 2, "dispatch-001",
        ProductionEnablementEventId(1), enable.identity.canonicalText(),
        ActorIdentity("manager@example.test"), "campaign_manager_login", build);
    const auto laterVersionAttempt = BuildProductionDispatchAttemptV2(
        OperationalRequestId(7), "request-canonical-v1",
        mismatchedVersionAdmission, ProductionEnablementEventId(1),
        enable.identity.canonicalText(), "dispatch-003", 3, 7, 8,
        LeaseTokenDigest::Hydrate("fnv1a64:0123456789abcdef"),
        UtcTimestamp("2026-07-31T12:39:56.123456Z"),
        ActorIdentity("manager@example.test"),
        "campaign_manager_login", build);
    assert(laterVersionAttempt.admission.expectedRequestVersion == 2);
    assert(laterVersionAttempt.expectedRequestVersion == 7);

    ProductionReadinessSnapshot readySnapshot;
    readySnapshot.migrationVersion = "055";
    readySnapshot.migrationFilename =
        kProductionAdmissionMigrationFilename;
    readySnapshot.migrationChecksum = kProductionAdmissionMigrationChecksum;
    readySnapshot.schedulerEvidenceContractVersion = "1";
    readySnapshot.managerBuildContractVersion = "1";
    readySnapshot.enablementContractVersion = "1";
    readySnapshot.admissionContractVersion = "1";
    readySnapshot.productionAttemptContractVersion = "2";
    readySnapshot.admissionEvidenceCount = 1;
    readySnapshot.productionAttemptEvidenceCount = 1;
    readySnapshot.schedulerGeneration = 52;
    readySnapshot.schedulerCutoverState = "complete";
    readySnapshot.schedulerEvidenceComplete = true;
    readySnapshot.schedulerEvidence.emplace(scheduler);
    readySnapshot.enablementHead.emplace(PersistedProductionEnablementHead{
        ProductionEnablementEventId(1),
        ProductionEnablementEventKind::enable, enable.identity, 1,
        scheduler, build, ProductionEnablementAuditEvidence{
            1, ProductionEnablementEventId(1), "enable-001",
            ProductionEnablementEventKind::enable,
            ActorIdentity("operator@example.test"), kProductionEnablerRole,
            Reason("Enable verified H1 evidence."), "recorded",
            "new_operation", "immutable_enablement_event_recorded"}});
    readySnapshot.independentVerificationReference =
        "cee://phase-h/generation-52";
    readySnapshot.sessionPrincipal = "campaign_manager_login";
    readySnapshot.currentPrincipal = "campaign_manager_login";
    readySnapshot.readerMember = true;
    readySnapshot.dispatcherMember = true;
    readySnapshot.phase5TransactionalMember = true;
    readySnapshot.schedulerEvidenceReaderMember = true;
    readySnapshot.enablementEffective = true;
    readySnapshot.completionNestedV2ProofVersion = "1";
    readySnapshot.completionNestedV2ProofValid = true;
    const auto ready = EvaluateProductionReadiness(
        std::move(readySnapshot), build);
    assert(ready.ready);
    assert(ready.blockers.empty());
    const auto readinessOutput = RenderProductionReadiness(ready);
    for (const std::string& field : {
             "migration_version=055",
             "migration_filename=055_campaign_operations_production_admission_foundation.sql",
             "migration_checksum=1b13d3a64336d7cbd55c935396ec42c4c06320105677829f0cf405733e5715fe",
             "scheduler_contract_version=1",
             "scheduler_generation=52",
             "scheduler_evidence_canonical=campaign_operations_scheduler_protocol_evidence_v1",
             "enablement_contract_version=1",
             "manager_build_contract_version=1",
             "manager_service_contract=campaign-operations-production-dispatch-and-manager-run-once-v1",
             "admission_contract_version=1",
             "production_attempt_contract_version=2",
             "approved_build_canonical=campaign_operations_manager_build_v1",
             "approved_build_hash=fnv1a64:9afc5f65cbaaece3",
             "actual_running_build_canonical=campaign_operations_manager_build_v1",
             "actual_running_build_hash=fnv1a64:9afc5f65cbaaece3",
             "completion_nested_v2_proof_version=1",
             "completion_nested_v2_proof=valid",
             "reader_member=true",
             "dispatcher_member=true",
             "phase5_transactional_member=true",
             "scheduler_evidence_reader_member=true",
             "enabler_member=false",
             "disabler_member=false",
             "prohibited_test_dispatcher_member=false",
             "prohibited_test_phase5_member=false",
             "scheduler_cutover_state=complete",
             "enablement_effective=true",
             "ready_admitted=0",
             "ready_unadmitted=0",
             "current_event_leases=0",
             "old_event_blocked_leases=0",
             "reconciliation_required=0",
             "blockers=none"})
        AssertContains(readinessOutput, field);

    auto genesisSnapshot = ready.snapshot;
    genesisSnapshot.admissionContractVersion.reset();
    genesisSnapshot.productionAttemptContractVersion.reset();
    genesisSnapshot.admissionEvidenceCount = 0;
    genesisSnapshot.productionAttemptEvidenceCount = 0;
    const auto genesis = EvaluateProductionReadiness(
        std::move(genesisSnapshot), build);
    assert(genesis.ready);
    assert(genesis.blockers.empty());
    AssertContains(RenderProductionReadiness(genesis),
        "admission_contract_version=genesis-empty,");
    AssertContains(RenderProductionReadiness(genesis),
        "production_attempt_contract_version=genesis-empty,");

    auto admissionOnlySnapshot = ready.snapshot;
    admissionOnlySnapshot.productionAttemptContractVersion.reset();
    admissionOnlySnapshot.productionAttemptEvidenceCount = 0;
    const auto admissionOnly = EvaluateProductionReadiness(
        std::move(admissionOnlySnapshot), build);
    assert(admissionOnly.ready);

    auto attemptOnlySnapshot = ready.snapshot;
    attemptOnlySnapshot.admissionContractVersion.reset();
    attemptOnlySnapshot.admissionEvidenceCount = 0;
    const auto attemptOnly = EvaluateProductionReadiness(
        std::move(attemptOnlySnapshot), build);
    assert(attemptOnly.ready);

    auto wrongAdmissionSnapshot = ready.snapshot;
    wrongAdmissionSnapshot.admissionContractVersion = "2";
    const auto wrongAdmission = EvaluateProductionReadiness(
        std::move(wrongAdmissionSnapshot), build);
    assert(!wrongAdmission.ready);
    assert(wrongAdmission.blockers == std::vector<std::string>{
        "canonical_contract_versions"});

    auto wrongAttemptSnapshot = ready.snapshot;
    wrongAttemptSnapshot.productionAttemptContractVersion = "3";
    const auto wrongAttempt = EvaluateProductionReadiness(
        std::move(wrongAttemptSnapshot), build);
    assert(!wrongAttempt.ready);
    assert(wrongAttempt.blockers == std::vector<std::string>{
        "canonical_contract_versions"});

    auto mixedSnapshot = ready.snapshot;
    mixedSnapshot.admissionContractVersion = "1|2";
    mixedSnapshot.admissionEvidenceCount = 2;
    const auto mixed = EvaluateProductionReadiness(
        std::move(mixedSnapshot), build);
    assert(!mixed.ready);
    assert(mixed.blockers == std::vector<std::string>{
        "canonical_contract_versions"});

    auto unknownPresenceSnapshot = ready.snapshot;
    unknownPresenceSnapshot.admissionEvidenceCount.reset();
    const auto unknownPresence = EvaluateProductionReadiness(
        std::move(unknownPresenceSnapshot), build);
    assert(!unknownPresence.ready);
    assert(unknownPresence.blockers == std::vector<std::string>{
        "canonical_contract_versions"});

    auto missingChecksumSnapshot = ready.snapshot;
    missingChecksumSnapshot.migrationChecksum.reset();
    const auto missingChecksum = EvaluateProductionReadiness(
        std::move(missingChecksumSnapshot), build);
    assert(!missingChecksum.ready);
    assert(missingChecksum.blockers == std::vector<std::string>{
        "migration_055_identity_or_checksum"});
    AssertContains(RenderProductionReadiness(missingChecksum),
        "migration_checksum=missing,"
        "scheduler_contract_version=1");
    AssertContains(RenderProductionReadiness(missingChecksum),
        "blockers=migration_055_identity_or_checksum");

    auto contradictoryContractSnapshot = ready.snapshot;
    contradictoryContractSnapshot.schedulerEvidenceContractVersion = "77";
    const auto contradictoryContract = EvaluateProductionReadiness(
        std::move(contradictoryContractSnapshot), build);
    assert(!contradictoryContract.ready);
    assert(contradictoryContract.blockers == std::vector<std::string>{
        "canonical_contract_versions"});
    AssertContains(RenderProductionReadiness(contradictoryContract),
        "scheduler_contract_version=77");
    AssertContains(RenderProductionReadiness(contradictoryContract),
        "blockers=canonical_contract_versions");

    ProductionReadinessSnapshot snapshot;
    snapshot.migrationVersion = "055";
    snapshot.schedulerGeneration = 52;
    snapshot.schedulerCutoverState = "complete";
    snapshot.completionNestedV2ProofVersion = "1";
    snapshot.completionNestedV2ProofValid = true;
    const auto readiness = EvaluateProductionReadiness(std::move(snapshot));
    assert(!readiness.ready);
    assert(!readiness.blockers.empty());
}
