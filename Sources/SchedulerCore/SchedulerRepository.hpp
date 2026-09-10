#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace EA::SchedulerCore
{

// Persistence-shaped values at the SchedulerCore repository boundary.  The
// strings intentionally retain the database representation; domain parsing
// remains in the scheduler orchestration layer.
struct SchedulerExperimentRecord
{
    long long experimentId = -1;
    std::string symbol;
    int predictionHorizon = 0;
    double cNextThreshold = 0.0;
    std::optional<double> coreLrMult;
    std::optional<double> headLrMult;
    int targetEpochs = 0;
    int checkpointInterval = 20;
    std::string trainStart;
    std::string trainEnd;
    std::optional<std::string> inferStart;
    std::optional<std::string> inferEnd;
    std::optional<long long> lastModelId;
    std::optional<long long> resumeModelId;
    std::optional<std::string> trainLogPath;
    std::optional<std::string> inferLogPath;
    std::optional<std::string> analysisLogPath;
    std::string donchian20Mode;
    std::string featureWarmupScope;
    std::string donchianLookback;
    std::string featureAblationMask;
    bool resumeExpandInputWidth = false;
    std::string trainingObjectiveCanonical;
    std::string trainingObjectiveHash;
};

struct PendingSchedulerExperimentRecord
{
    SchedulerExperimentRecord experiment;
    std::string schedulerPriority = "normal";
    bool resumeRequested = false;
    std::string schedulerResumeOrigin = "none";
    std::optional<long long> activeWorkerAttemptId;
};

struct RunningSchedulerExperimentRecord
{
    SchedulerExperimentRecord experiment;
    std::string phase;
    std::optional<int> workerPid;
    double attemptStartedEpoch = 0.0;
};

struct SchedulerQueueSnapshot
{
    int pendingTrain = 0;
    int pendingInfer = 0;
    int pendingAnalyze = 0;
    int runningTrain = 0;
    int runningInfer = 0;
    int runningAnalyze = 0;
};

struct PreemptionVictimRecord
{
    long long experimentId = -1;
    std::string priority;
    long long workerAttemptId = -1;
};

struct PendingSelectionMetadata
{
    std::string priority = "normal";
    std::string resumeOrigin = "none";
    std::string updatedAt;
    long long experimentId = -1;
};

bool PendingSelectionPrecedes(
    const PendingSelectionMetadata& lhs,
    const PendingSelectionMetadata& rhs);

struct SpawnedWorkerAttemptUpdate
{
    long long workerAttemptId = -1;
    std::string schedulerInvocationId;
    long long schedulerFencingToken = 0;
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string phase;
    int workerPid = -1;
    std::string processStartIdentity;
    std::string canonicalExecutablePath;
    std::string commandLine;
};

bool HasCompleteSpawnedWorkerAttemptUpdate(
    const SpawnedWorkerAttemptUpdate& update) noexcept;

enum class SpawnPersistenceResult
{
    Updated,
    AttemptPreconditionRejected,
    LifecyclePreconditionRejected
};

struct WorkerAttemptLaunchFailureUpdate
{
    long long workerAttemptId = -1;
    std::string schedulerInvocationId;
    long long schedulerFencingToken = 0;
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string phase;
    int exitCode = 127;
    std::string diagnostic;
};

enum class LaunchFailurePersistenceResult
{
    Updated,
    AttemptPreconditionRejected,
    LifecyclePreconditionRejected
};

class SchedulerRepository
{
public:
    virtual ~SchedulerRepository() = default;

    virtual std::vector<PendingSchedulerExperimentRecord>
    loadPendingExperiments(std::string_view phase,
                           bool cancellationOnly) = 0;
    virtual std::vector<RunningSchedulerExperimentRecord>
    loadRunningExperiments() = 0;
    virtual SchedulerQueueSnapshot loadQueueSnapshot() = 0;
    virtual int countWorkersConsumingCapacity(
        std::string_view capacityClass) = 0;
    virtual std::optional<PreemptionVictimRecord>
    loadPreemptionVictim(
        std::string_view phase,
        int candidatePriorityRank) = 0;

    virtual SpawnPersistenceResult persistSpawnedWorkerAttempt(
        const SpawnedWorkerAttemptUpdate& update) = 0;
    virtual LaunchFailurePersistenceResult persistWorkerAttemptLaunchFailure(
        const WorkerAttemptLaunchFailureUpdate& update) = 0;
};

} // namespace EA::SchedulerCore
