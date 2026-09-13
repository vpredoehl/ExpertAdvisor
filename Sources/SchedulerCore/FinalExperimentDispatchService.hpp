#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string_view>
#include <vector>

namespace EA::SchedulerCore
{

enum class FinalExperimentPhase
{
    Train,
    Infer,
    Analyze
};

struct FinalExperimentDispatchCandidate
{
    std::size_t sourceIndex = 0;
    long long experimentId = -1;
    bool hasActiveWorkerAttempt = false;
    bool hasLastModel = false;
};

struct FinalExperimentDispatchBatch
{
    bool launchAllowed = false;
    int freeSlots = 0;
    std::vector<FinalExperimentDispatchCandidate> candidates;
};

enum class FinalExperimentStoppedAdmission
{
    NotApplicable,
    Admitted,
    MissingProcessFallbackReady,
    DeferredNoCapacity,
    DeferredUnsafe
};

enum class FinalExperimentEligibility
{
    Eligible,
    Skipped,
    Failed
};

struct FinalExperimentDispatchStats
{
    FinalExperimentPhase phase = FinalExperimentPhase::Train;
    int examined = 0;
    int skipped = 0;
    int launched = 0;
    int freeSlots = 0;
};

struct FinalExperimentDispatchConfiguration
{
    bool dryRun = false;
    int maxTrainProcesses = 0;
    int maxInferProcesses = 0;
    int maxAnalyzeProcesses = 0;
};

// Adapter boundary for database-backed selection and eligibility, authority
// validation, admission/preemption, worker-attempt lifecycle, command
// construction, process launch, persistence, and diagnostics.
struct FinalExperimentDispatchOperations
{
    std::function<FinalExperimentDispatchBatch(
        FinalExperimentPhase, bool cancellationOnly)> load;
    std::function<void()> ensureLogDirectory;
    std::function<bool(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase)> semanticPreflight;
    std::function<void(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase,
        int maximumCapacity)> preemptOneLowerPriorityWorker;
    std::function<FinalExperimentStoppedAdmission(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase,
        int maximumCapacity)> admitStoppedWorker;
    std::function<FinalExperimentEligibility(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase)> evaluateEligibility;
    std::function<void(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase)> emitDryRunCommand;
    std::function<std::optional<long long>(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase,
        int maximumCapacity,
        bool cancellationOnly)> reserveWorkerAttempt;
    std::function<int(FinalExperimentPhase)> capacityUsed;
    std::function<void(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase,
        long long workerAttemptId)> prepareReservedLaunch;
    std::function<void(
        const FinalExperimentDispatchCandidate&,
        FinalExperimentPhase,
        long long workerAttemptId)> launchPreparedWorker;
    std::function<void(
        FinalExperimentPhase,
        long long experimentId,
        std::string_view reason)> logSkip;
    std::function<void(const FinalExperimentDispatchStats&)> printStats;
};

class FinalExperimentDispatchService final
{
public:
    FinalExperimentDispatchService(
        FinalExperimentDispatchConfiguration configuration,
        FinalExperimentDispatchOperations operations,
        std::ostream& errors);

    int runTrain(bool cancellationOnly);
    int runInference();
    int runAnalysis();

private:
    int runPhase(FinalExperimentPhase phase, bool cancellationOnly);
    int maximumCapacity(FinalExperimentPhase phase) const;

    FinalExperimentDispatchConfiguration configuration_;
    FinalExperimentDispatchOperations operations_;
    std::ostream& errors_;
};

} // namespace EA::SchedulerCore
