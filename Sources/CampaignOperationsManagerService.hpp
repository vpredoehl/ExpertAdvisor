#pragma once

#include "CampaignOperationsDispatchRepository.hpp"
#include "CampaignOperationsDispatchService.hpp"

#include <iosfwd>
#include <functional>
#include <string>
#include <vector>

namespace EA::CampaignOperations
{

enum class ManagerRequestOutcomeClassification
{
    dispatchResult,
    requestLocalSemanticFailure
};

enum class ManagerGlobalStopReason
{
    none,
    productionDisabled,
    schedulerProtocolIneffective,
    managerBuildNotReady,
    privilegeFailure,
    databaseFailure
};

struct ManagerRequestResult final
{
    OperationalRequestId requestId;
    int expectedRequestVersion = 0;
    std::string operationKey;
    ManagerRequestOutcomeClassification outcome =
        ManagerRequestOutcomeClassification::dispatchResult;
    std::string dispatchClassification;
    std::string replayDisposition;
    bool newlyCommitted = false;
    bool exactReplay = false;
    std::string diagnosticCode;
};

struct ManagerRunOnceResult final
{
    int dispatchLimit = 0;
    std::vector<DispatchCandidate> candidates;
    std::vector<ManagerRequestResult> requests;
    bool stoppedEarly = false;
    ManagerGlobalStopReason stopReason = ManagerGlobalStopReason::none;
    std::string stopDiagnostic;
};

#if defined(CAMPAIGN_OPERATIONS_H3_TESTING)
// Test-only synchronization is outside the production command surface.  The
// callback is invoked after the snapshot transaction has committed and around
// each sequential candidate attempt so disposable harnesses can coordinate
// state changes without adding a Manager batch identity or a production hook.
enum class ManagerTestInjectionPoint
{
    afterCandidateSnapshot,
    beforeCandidateProcessing,
    afterCandidateProcessing
};

using ManagerTestHook = std::function<void(
    ManagerTestInjectionPoint, int, OperationalRequestId)>;

ManagerRunOnceResult RunCampaignOperationsManagerOnceForTest(
    const std::string& connectionString, int dispatchLimit,
    const ManagerBuildContract&, const ManagerTestHook&);
#endif

std::string ToText(ManagerRequestOutcomeClassification value);
std::string ToText(ManagerGlobalStopReason value);

ManagerRunOnceResult RunCampaignOperationsManagerOnce(
    const std::string& connectionString, int dispatchLimit,
    const std::string& executablePath);

int RunCampaignOperationsManagerOnceCommand(
    const std::string& connectionString, int dispatchLimit,
    const std::string& executablePath, std::ostream& output,
    std::ostream& errors);

} // namespace EA::CampaignOperations
