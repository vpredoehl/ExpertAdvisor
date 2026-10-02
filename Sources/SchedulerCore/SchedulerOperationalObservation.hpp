#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::GlobalExperimentControl
{

struct ProcessObservation
{
    bool exists = false;
    bool inspectionSucceeded = false;
    bool permissionDenied = false;
    bool stopped = false;
    int pid = -1;
    int processGroupId = -1;
    std::string executable;
    std::string commandLine;
    std::string processStartIdentity;
};

struct NativeProcessStatus
{
    int pid = -1;
    int processGroupId = -1;
    std::string state;
    std::string commandLine;
};

class NativeProcessObservationBackend
{
public:
    virtual ~NativeProcessObservationBackend() = default;

    virtual bool ProcessExists(int pid, int& errorNumber) = 0;
    virtual std::optional<std::string> ReadStartIdentity(
        int pid,
        int& errorNumber) = 0;
    virtual std::optional<NativeProcessStatus> ReadStatus(
        int pid,
        int& errorNumber) = 0;
    virtual std::optional<std::string> ReadProcPidPath(
        int pid,
        int& errorNumber) = 0;
    // This is the kernel's exec-path record from KERN_PROCARGS2, not argv[0]
    // or a token parsed from `ps` output.
    virtual std::optional<std::string> ReadKernelExecutablePath(
        int pid,
        int& errorNumber) = 0;
};

class ProcessObserver
{
public:
    virtual ~ProcessObserver() = default;
    virtual ProcessObservation Observe(int pid) = 0;
    virtual int CallerPid() const = 0;
    virtual int CallerProcessGroupId() const = 0;
};

struct ManagedWorker
{
    std::optional<long long> workerAttemptId;
    std::string workerKind;
    std::string capacityClass;
    std::string ownershipOrigin;
    std::string attemptLifecycleState;
    std::string launchAttemptIdentity;
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string phase;
    std::string lifecycleStatus;
    int pid = -1;
    std::optional<int> processGroupId;
    std::optional<std::string> executable;
    std::optional<std::string> commandLine;
    std::optional<std::string> processStartIdentity;
    // Status readers that reconstruct a worker from the lifecycle row and its
    // exact active durable attempt set this false if those two persisted
    // identities disagree. Runtime control paths already enforce the same
    // invariant while locking the exact attempt.
    bool authoritativeBindingMatches = true;
};

enum class IdentityResult
{
    Validated,
    ProcessMissing,
    StalePid,
    IdentityValidationFailed,
    UnsafeProcessGroup,
    PermissionDenied,
    InspectionFailed
};

enum class ProcessExecutionState
{
    Unknown,
    Running,
    Stopped,
    Missing
};

struct ValidatedWorker
{
    ManagedWorker worker;
    ProcessObservation observation;
    IdentityResult identity = IdentityResult::InspectionFailed;
    std::string detail;
};

struct SchedulerWorkerCandidate
{
    int pid = -1;
    std::string kind;
    std::string commandLine;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
    bool stopped = false;
};

struct SchedulerWorkerClassification
{
    int pid = -1;
    std::string kind;
    bool managed = false;
    bool authoritative = false;
    bool detected = true;
    IdentityResult identity = IdentityResult::IdentityValidationFailed;
    ProcessExecutionState executionState = ProcessExecutionState::Unknown;
    std::string reason;
    std::string lifecycleStatus;
    std::string attemptLifecycleState;
    std::optional<long long> experimentId;
    std::optional<long long> checkpointEvalId;
    std::optional<std::string> expectedExecutable;
    std::optional<std::string> observedExecutable;
    std::optional<bool> executableIdentityMatch;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
};

struct SchedulerWorkerAggregate
{
    int workers = 0;
    double cpuPercent = 0.0;
    double memPercent = 0.0;
    double rssMb = 0.0;
};

struct SchedulerWorkerClassificationSummary
{
    SchedulerWorkerAggregate managedTrain;
    SchedulerWorkerAggregate managedInfer;
    SchedulerWorkerAggregate managedAnalyze;
    SchedulerWorkerAggregate managedRunningTrain;
    SchedulerWorkerAggregate managedRunningInfer;
    SchedulerWorkerAggregate managedRunningAnalyze;
    SchedulerWorkerAggregate managedPausedTrain;
    SchedulerWorkerAggregate managedPausedInfer;
    SchedulerWorkerAggregate managedPausedAnalyze;
    SchedulerWorkerAggregate unmanagedTrain;
    SchedulerWorkerAggregate unmanagedInfer;
    SchedulerWorkerAggregate unmanagedAnalyze;
    SchedulerWorkerAggregate identityMismatchTrain;
    SchedulerWorkerAggregate identityMismatchInfer;
    SchedulerWorkerAggregate identityMismatchAnalyze;
    SchedulerWorkerAggregate expectedMissingTrain;
    SchedulerWorkerAggregate expectedMissingInfer;
    SchedulerWorkerAggregate expectedMissingAnalyze;
};

std::optional<std::string> ReadProcessStartIdentity(int pid);
std::unique_ptr<ProcessObserver> CreateNativeProcessObserver();
std::unique_ptr<ProcessObserver> CreateNativeProcessObserverForTesting(
    std::unique_ptr<NativeProcessObservationBackend> backend);

ValidatedWorker ValidateManagedWorker(const ManagedWorker& worker,
                                      ProcessObserver& processes);
ValidatedWorker ValidateStoppedWorkerForSchedulerAdmission(
    const ManagedWorker& worker,
    ProcessObserver& processes);
ValidatedWorker ValidatePausedManagedWorker(
    const ManagedWorker& worker,
    ProcessObserver& processes);

std::vector<SchedulerWorkerClassification> ClassifySchedulerWorkers(
    const std::vector<SchedulerWorkerCandidate>& candidates,
    const std::vector<ManagedWorker>& authoritativeWorkers,
    ProcessObserver& processes);
SchedulerWorkerClassificationSummary SummarizeSchedulerWorkers(
    const std::vector<SchedulerWorkerClassification>& classifications);

struct ControlSnapshot
{
    std::string desiredState = "running";
    std::optional<long long> activeRequestId;
    std::optional<long long> currentPauseRequestId;
    std::optional<std::string> activeAction;
    std::optional<std::string> cancellationMode;
    bool inferBeforeCancel = false;
};

ControlSnapshot LoadControlSnapshot(pqxx::transaction_base& transaction);

const char* ToString(IdentityResult result);
const char* ToString(ProcessExecutionState state);

} // namespace EA::GlobalExperimentControl
