#pragma once

#include "CampaignOperationsBindingRepository.hpp"
#include "CampaignOperationsProductionAdmission.hpp"

#include <string>

namespace EA::CampaignOperations
{

inline constexpr int kCampaignOperationsDispatchMaximumTransactionRetries = 3;
inline constexpr char kCampaignOperationsPhase3TestDatabasePrefix[] =
    "expertadvisor_campaign_operations_phase3_test_";
inline constexpr char kCampaignOperationsPhase3TestAcknowledgement[] =
    "I_UNDERSTAND_PHASE3_TEST_ONLY";

struct IsolatedDispatchSafetyGate final
{
    std::string expectedDatabase;
    std::string acknowledgement;
};

enum class DispatchServiceFailureClassification
{
    none,
    transientDatabaseRetryExhausted,
    commitOutcomeUnknown
};

struct DispatchServiceResult final
{
    DispatchResultClassification classification;
    DownstreamEvidenceClassification downstreamEvidence;
    ExactReplayDisposition replayDisposition;
    UncertainCommitRecoveryClassification recovery;
    OperationalRequestId requestId;
    std::string bindingSetIdentityHash;
    int transactionAttempts = 0;
    DispatchServiceFailureClassification failure =
        DispatchServiceFailureClassification::none;
    std::string diagnosticCode;
};

struct ProductionDispatchRequest final
{
    OperationalRequestId requestId;
    int expectedRequestVersion = 0;
    std::string operationKey;
    ActorIdentity requestingActor;
    ManagerBuildContract executingBuild;
    bool acknowledged = false;
};

// This is the only Phase 3 executable adapter.  It is deliberately a
// one-request test adapter: no polling, scheduler, worker or production mode.
DispatchServiceResult DispatchOneRequestForIsolatedTest(
    const std::string& connectionString, OperationalRequestId requestId,
    int expectedRequestVersion, const ActorIdentity& dispatcher,
    const IsolatedDispatchSafetyGate& safetyGate,
    DispatchTestHook testHook = {});

// H2's exact production adapter.  It accepts one caller-named request and
// uses the same Phase E handoff engine as the isolated adapter.  No batch,
// polling, manager-run-once, or continuous-operation surface is exposed.
DispatchServiceResult DispatchOneRequestForProduction(
    const std::string& connectionString, const ProductionDispatchRequest&,
    const std::string& executablePath);

DispatchServiceResult DispatchOneRequestForProduction(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest&, const std::string& executablePath);

// H3's bounded Manager adapter.  It supplies only the frozen deterministic
// source canonical to the shared Phase E engine; it exposes no test hook.
DispatchServiceResult DispatchOneRequestForProductionManager(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest&,
    const std::string& managerSourceCanonical,
    const std::string& executablePath);

#if defined(CAMPAIGN_OPERATIONS_H3_TESTING)
// Test-only build injection for the disposable H3 harness.  It enters the
// same Manager production adapter and readiness/authority path while avoiding
// a dependency on the dirty development worktree's executable contract.
using ManagerAdapterTestHook = std::function<void(DispatchTestInjectionPoint)>;

DispatchServiceResult DispatchOneRequestForProductionManagerWithFixture(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest&,
    const std::string& managerSourceCanonical, const ManagerBuildContract&,
    ManagerAdapterTestHook testHook = {});
#endif

#if defined(CAMPAIGN_OPERATIONS_H2_TESTING)
// Test-only production adapter entry point.  It retains the production role
// switching and handoff path while accepting a fixture-owned build contract;
// the production CLI remains bound to CaptureActualManagerBuildContract.
DispatchServiceResult DispatchOneRequestForProductionForTest(
    const std::string& connectionString, const ProductionDispatchRequest&,
    DispatchTestHook testHook = {});

DispatchServiceResult DispatchOneRequestForProductionForTest(
    const std::string& managerConnectionString,
    const std::string& dispatchServiceConnectionString,
    const ProductionDispatchRequest&, DispatchTestHook testHook = {});
#endif

} // namespace EA::CampaignOperations
