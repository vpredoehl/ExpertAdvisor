#pragma once

#include <chrono>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::GlobalExperimentControl
{

inline constexpr const char* kCoordinationLockName =
    "expertadvisor.global_experiment_execution_control.v1";

enum class Action
{
    PauseAll,
    ResumeAll,
    CancelAll
};

enum class CancellationMode
{
    Immediate,
    AfterNextCheckpoint
};

struct Command
{
    Action action = Action::PauseAll;
    std::optional<CancellationMode> cancellationMode;
    bool inferBeforeCancel = false;
    bool dryRun = false;
    bool confirmed = false;
    std::string invocationIdentity;
    std::optional<std::string> requesterIdentity;
    std::chrono::milliseconds terminationGrace{3000};
};

std::optional<std::string> ValidateCommand(const Command& command);

// If currentEpoch is itself backed by a durable periodic checkpoint it is the
// next safe boundary; otherwise this returns the first strictly later periodic
// checkpoint. A target at or after targetEpochs is not a future stop point.
std::optional<int> NextCancellationCheckpoint(
    int currentEpoch,
    int checkpointInterval,
    int targetEpochs,
    std::optional<int> latestDurableCheckpointEpoch);

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

struct ManagedWorker
{
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string phase;
    std::string lifecycleStatus;
    int pid = -1;
    std::optional<int> processGroupId;
    std::optional<std::string> executable;
    std::optional<std::string> commandLine;
    std::optional<std::string> processStartIdentity;
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

struct ValidatedWorker
{
    ManagedWorker worker;
    ProcessObservation observation;
    IdentityResult identity = IdentityResult::InspectionFailed;
    std::string detail;
};

class ProcessOperations
{
public:
    virtual ~ProcessOperations() = default;
    virtual ProcessObservation Observe(int pid) = 0;
    virtual bool SignalProcessGroup(int processGroupId,
                                    int signalNumber,
                                    int& errorNumber) = 0;
    virtual bool WaitForProcessGroupExit(
        int processGroupId,
        std::chrono::milliseconds timeout) = 0;
    virtual int CallerPid() const = 0;
    virtual int CallerProcessGroupId() const = 0;
};

std::optional<std::string> ReadProcessStartIdentity(int pid);
std::unique_ptr<ProcessOperations> CreateNativeProcessOperations();

ValidatedWorker ValidateManagedWorker(const ManagedWorker& worker,
                                      ProcessOperations& processes);

struct SignalOutcome
{
    IdentityResult identity = IdentityResult::InspectionFailed;
    std::string result;
    std::vector<int> signals;
    bool success = false;
    std::string detail;
};

SignalOutcome PauseWorker(const ManagedWorker& worker,
                          ProcessOperations& processes);
SignalOutcome ResumeWorker(const ManagedWorker& worker,
                           ProcessOperations& processes);
SignalOutcome CancelWorker(const ManagedWorker& worker,
                           bool resumeFirst,
                           std::chrono::milliseconds grace,
                           ProcessOperations& processes);

struct ControlSnapshot
{
    std::string desiredState = "running";
    std::optional<long long> activeRequestId;
    std::optional<long long> currentPauseRequestId;
    std::optional<std::string> activeAction;
    std::optional<std::string> cancellationMode;
    bool inferBeforeCancel = false;
};

void AcquireCoordinationLock(pqxx::transaction_base& transaction);
ControlSnapshot LoadControlSnapshot(pqxx::transaction_base& transaction);
bool NormalSchedulingAllowed(const ControlSnapshot& snapshot);
bool CancellationInferenceAllowed(const ControlSnapshot& snapshot);
bool CancellationCheckpointTrainAllowed(const ControlSnapshot& snapshot);

// Shared CLI contract for the final authoritative persisted request status.
int RequestExitCodeForPersistedStatus(const std::string& status);

// Claims (when permitted) and reconciles the exact active cancellation request.
// Every mutation is fenced by the persisted application owner and live lease.
// Returns false without mutation while a different owner holds a live lease.
bool ReconcileActiveCancellation(pqxx::work& transaction,
                                 const std::string& applicationOwner,
                                 bool claimExpiredLease);

int RunCommand(const std::string& connectionString,
               const Command& command,
               std::ostream& output,
               std::ostream& error);

// Test-only injection seam for deterministic replay/crash-window coverage.
// Production callers use RunCommand, which supplies native process operations.
int RunCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const Command& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes);

struct ExperimentResumeCommand
{
    long long experimentId = -1;
    bool dryRun = false;
    bool confirmed = false;
    std::string invocationIdentity;
    std::optional<std::string> requesterIdentity;
};

// Preserves lifecycle-paused resume behavior and adds an audited selective
// release path for a running worker suspended by the current global pause.
int RunExperimentResumeCommand(const std::string& connectionString,
                               const ExperimentResumeCommand& command,
                               std::ostream& output,
                               std::ostream& error);

int RunExperimentResumeCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const ExperimentResumeCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes);

const char* ToString(Action action);
const char* ToString(CancellationMode mode);
const char* ToString(IdentityResult result);

} // namespace EA::GlobalExperimentControl
