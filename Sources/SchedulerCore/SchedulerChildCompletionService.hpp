#pragma once

#include "SchedulerChildStatus.hpp"

#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include <sys/types.h>

namespace EA::SchedulerCore
{

struct SchedulerOwnedChild
{
    pid_t pid = -1;
    std::optional<long long> workerAttemptId;
    long long experimentId = -1;
    std::string phase;
    std::string operation;
    std::string commandLine;
    std::optional<long long> expectedModelId;
    std::optional<long long> checkpointEvalId;
    std::string logPath;
    std::string launchedAt;
    double launchedEpoch = 0.0;
    std::optional<EA::ExperimentScheduler::ObservedChildStatus> observedStatus;
    bool terminationObserved = false;
    bool exitDiagnosticEmitted = false;
};

using SchedulerOwnedChildren = std::map<pid_t, SchedulerOwnedChild>;

struct SchedulerChildCompletionEvidence
{
    int exitCode = -1;
    std::optional<int> signalNumber;
    bool coreDumped = false;
    std::string error;
};

// Adapter boundary for authority-bound exact-attempt checks and durable
// completion handling. The service owns observation, classification, retry
// caching, diagnostics, ordering, and child-bookkeeping decisions.
struct SchedulerChildCompletionOperations
{
    std::function<EA::ExperimentScheduler::ObservedChildStatus(pid_t)>
        observeChild;
    std::function<bool(
        const SchedulerOwnedChild&,
        const EA::ExperimentScheduler::ObservedChildStatus&)>
        observeTerminalCheckpointStop;
    std::function<void(const SchedulerOwnedChild&)>
        injectStaleReaperReplacementForTest;
    std::function<bool(const SchedulerOwnedChild&)> verifyExactActiveAttempt;
    std::function<bool()> staleReaperFailpointEnabled;
    std::function<void()> requestSchedulerStop;
    std::function<void(const SchedulerOwnedChild&, int rawStatus)>
        persistUnexpectedStatus;
    std::function<void(
        const SchedulerOwnedChild&,
        const SchedulerChildCompletionEvidence&)>
        persistCheckpointCompletion;
    std::function<void(
        const SchedulerOwnedChild&,
        const SchedulerChildCompletionEvidence&)>
        persistExperimentCompletion;
    std::function<void(
        const SchedulerOwnedChild&,
        int exitCode,
        const std::optional<int>& signalNumber,
        std::string_view diagnostic)>
        finalizeWorkerAttempt;
};

class SchedulerChildCompletionService final
{
public:
    SchedulerChildCompletionService(
        SchedulerChildCompletionOperations operations,
        std::ostream& diagnostics);

    void reap(SchedulerOwnedChildren& children);

private:
    SchedulerChildCompletionOperations operations_;
    std::ostream& diagnostics_;
};

} // namespace EA::SchedulerCore
