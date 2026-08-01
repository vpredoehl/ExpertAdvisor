#pragma once

#include "CampaignOperationsControlRepository.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace EA::CampaignOperations
{

inline constexpr int
    kCampaignOperationsReconciliationMaximumTransactionRetries = 3;

struct CampaignControlRequest final
{
    long long campaignId = 0;
    int expectedControlVersion = 0;
    ControlEventKind action = ControlEventKind::pause;
    std::string actorIdentity;
    std::string reason;
};

struct CampaignCancellationCommandRequest final
{
    long long campaignId = 0;
    std::optional<long long> requestId;
    std::optional<int> expectedRequestVersion;
    std::string operationKey;
    std::string actorIdentity;
    std::string reason;
};

struct ReconciliationObserveRequest final
{
    std::string runKey;
    long long afterRequestId = 0;
    int limit = 100;
    bool resolveSafeTransitions = false;
};

struct CampaignControlResult final
{
    ControlReplayDisposition replay;
    PersistedCampaignControlEvent persisted;
};

struct CampaignCancellationResult final
{
    ControlReplayDisposition replay;
    CancellationProgress progress;
    PersistedCampaignCancellationRequest request;
    std::optional<PersistedCampaignCancellationSettlement> settlement;
};

struct ReconciliationBatchResult final
{
    std::string runKey;
    long long priorTargetId = 0;
    long long lastTargetId = 0;
    int selectedCount = 0;
    int observationCount = 0;
    int resolutionCount = 0;
};

CampaignControlRequest ValidateCampaignControlRequest(
    const CampaignControlRequest& request);
CampaignCancellationCommandRequest ValidateCampaignCancellationRequest(
    const CampaignCancellationCommandRequest& request);
ReconciliationObserveRequest ValidateReconciliationObserveRequest(
    const ReconciliationObserveRequest& request);

CampaignControlResult ControlCampaign(
    pqxx::connection& connection, const CampaignControlRequest& request);
CampaignCancellationResult CancelCampaign(
    const std::string& connectionString,
    const CampaignCancellationCommandRequest& request,
    CampaignOperationsControlTestHook testHook = {});
ReconciliationBatchResult ObserveAndRecoverCampaignOperations(
    const std::string& connectionString,
    const ReconciliationObserveRequest& request,
    CampaignOperationsControlTestHook testHook = {});
CampaignControlStatus LoadCampaignControlStatus(
    pqxx::connection& connection, OperationalCampaignId campaignId,
    CampaignOperationsControlTestHook testHook = {});

int RunCampaignControlCommand(const std::string& connectionString,
    const CampaignControlRequest& request, std::ostream& output,
    std::ostream& errors);
int RunCampaignCancellationCommand(const std::string& connectionString,
    const CampaignCancellationCommandRequest& request,
    std::ostream& output, std::ostream& errors);
int RunCampaignReconciliationCommand(const std::string& connectionString,
    const ReconciliationObserveRequest& request, std::ostream& output,
    std::ostream& errors);
int RunCampaignControlStatusCommand(const std::string& connectionString,
    OperationalCampaignId campaignId, std::ostream& output,
    std::ostream& errors);

} // namespace EA::CampaignOperations
