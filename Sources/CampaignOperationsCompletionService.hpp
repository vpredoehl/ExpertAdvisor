#pragma once

#include "CampaignOperationsCompletionRepository.hpp"

#include <iosfwd>
#include <functional>
#include <memory>
#include <string>

namespace EA::CampaignOperations
{

struct CompleteIfSettledRequest final
{
    long long campaignId = 0;
    std::string operationKey;
    std::string actorIdentity;
    std::string reason;
};

struct CompleteIfSettledResult final
{
    CompletionAttemptDisposition disposition;
    std::optional<PersistedCompletionEvent> completion;
    std::vector<CompletionBlocker> blockers;
};

enum class CompletionTestInjectionPoint
{
    afterLocksBeforeEvidence,
    beforeCommit,
    duringOutcomeLookup,
    afterCommitBeforeResponse
};

using CompletionConnectionFactory =
    std::function<std::unique_ptr<pqxx::connection>()>;
using CompletionTestHook =
    std::function<void(CompletionTestInjectionPoint)>;

CompleteIfSettledRequest ValidateCompleteIfSettledRequest(
    const CompleteIfSettledRequest& request);
CompleteIfSettledResult CompleteCampaignIfSettled(
    pqxx::connection& connection, const CompleteIfSettledRequest& request);
CompleteIfSettledResult CompleteCampaignIfSettled(
    const CompletionConnectionFactory& connectionFactory,
    const CompleteIfSettledRequest& request,
    CompletionTestHook testHook = {});
CompletionStatus LoadCampaignCompletionStatus(
    pqxx::connection& connection, OperationalCampaignId campaignId);

int RunCompleteCampaignIfSettledCommand(
    const std::string& connectionString,
    const CompleteIfSettledRequest& request, std::ostream& output,
    std::ostream& errors);
int RunCampaignCompletionStatusCommand(
    const std::string& connectionString, OperationalCampaignId campaignId,
    std::ostream& output, std::ostream& errors);

} // namespace EA::CampaignOperations
