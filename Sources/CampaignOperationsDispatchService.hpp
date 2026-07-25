#pragma once

#include "CampaignOperationsBindingRepository.hpp"

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

// This is the only Phase 3 executable adapter.  It is deliberately a
// one-request test adapter: no polling, scheduler, worker or production mode.
DispatchServiceResult DispatchOneRequestForIsolatedTest(
    const std::string& connectionString, OperationalRequestId requestId,
    int expectedRequestVersion, const ActorIdentity& dispatcher,
    const IsolatedDispatchSafetyGate& safetyGate,
    DispatchTestHook testHook = {});

} // namespace EA::CampaignOperations
