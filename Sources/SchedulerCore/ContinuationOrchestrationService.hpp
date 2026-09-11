#pragma once

#include "ContinuationPolicy.hpp"

#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace EA::SchedulerCore
{

struct ContinuationAutomationOptions
{
    bool autoQueue = false;
    int scanSeconds = 300;
    int maxQueuesPerScan = 1;
    bool dryRun = false;
};

struct ContinuationAutomationCandidate
{
    long long sourceExperimentId = -1;
    EA::ExperimentScheduler::ContinuationPolicyConfig config;
    EA::ExperimentScheduler::ContinuationEvaluation evaluation;
};

struct ContinuationAutomationPreflight
{
    bool alreadySatisfied = false;
    std::string satisfiedDiagnosticFields;
};

struct ContinuationAutomationCounts
{
    int candidates = 0;
    int evaluated = 0;
    int alreadySatisfied = 0;
    int eligible = 0;
    int queued = 0;
    int errors = 0;
};

enum class ContinuationQueueResult
{
    Queued,
    AlreadyQueued,
    Failed
};

// Adapter boundary for authoritative persistence and compatibility workflows.
// Implementations retain their existing transaction and fencing semantics.
class ContinuationOrchestrationPort
{
public:
    virtual ~ContinuationOrchestrationPort() = default;

    virtual bool automationAllowed() = 0;
    virtual bool refreshAuthority() = 0;
    virtual std::vector<long long> loadCandidateIds() = 0;
    virtual ContinuationAutomationPreflight preflight(
        long long sourceExperimentId,
        bool dryRun) = 0;
    virtual ContinuationAutomationCandidate evaluate(
        long long sourceExperimentId,
        bool persist) = 0;
    virtual ContinuationQueueResult queue(
        long long sourceExperimentId) = 0;
    virtual void refreshQueuedIdentity(
        ContinuationAutomationCandidate& candidate) = 0;
};

bool BetterContinuationAutomationCandidate(
    const ContinuationAutomationCandidate& lhs,
    const ContinuationAutomationCandidate& rhs);

class ContinuationOrchestrationService final
{
public:
    ContinuationOrchestrationService(
        ContinuationOrchestrationPort& port,
        std::ostream& output,
        std::ostream& errors);

    ContinuationAutomationCounts runAutomaticScan(
        const ContinuationAutomationOptions& options);

private:
    void printEvaluation(
        const std::string& marker,
        const ContinuationAutomationCandidate& candidate,
        bool dryRun,
        const std::optional<std::string>& action = std::nullopt);

    ContinuationOrchestrationPort& port_;
    std::ostream& output_;
    std::ostream& errors_;
};

} // namespace EA::SchedulerCore
