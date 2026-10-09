#pragma once

#include "SchedulerCore/SchedulerAuthorityService.hpp"

#include "SchedulerCore/SchedulerOperationalObservation.hpp"

#include <chrono>
#include <functional>
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

class ProcessOperations : public ProcessObserver
{
public:
    virtual ~ProcessOperations() = default;
    virtual bool SignalProcessGroup(int processGroupId,
                                    int signalNumber,
                                    int& errorNumber) = 0;
    virtual bool WaitForProcessGroupExit(
        int processGroupId,
        std::chrono::milliseconds timeout) = 0;
};

std::unique_ptr<ProcessOperations> CreateNativeProcessOperations();
std::unique_ptr<ProcessOperations> CreateNativeProcessOperationsForTesting(
    std::unique_ptr<NativeProcessObservationBackend> backend);

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
// Requires current scheduler authority and exact stopped-attempt row locks.
// Reasserts an operator pause after an external SIGCONT; never admits work.
SignalOutcome PauseExternallyResumedStoppedWorker(
    const ManagedWorker& worker, ProcessOperations& processes);
SignalOutcome ResumeWorker(const ManagedWorker& worker,
                           ProcessOperations& processes);
// The caller must hold scheduler authority and an exact durable stopped-attempt
// verification for this worker. Ordinary resume callers must use ResumeWorker.
SignalOutcome ResumeStoppedWorkerForSchedulerAdmission(
    const ManagedWorker& worker,
    ProcessOperations& processes);
SignalOutcome CancelWorker(const ManagedWorker& worker,
                           bool resumeFirst,
                           std::chrono::milliseconds grace,
                           ProcessOperations& processes);

void AcquireCoordinationLock(pqxx::transaction_base& transaction);
bool NormalSchedulingAllowed(const ControlSnapshot& snapshot);
bool CancellationInferenceAllowed(const ControlSnapshot& snapshot);
bool CancellationCheckpointTrainAllowed(const ControlSnapshot& snapshot);

struct CheckpointStopRecordResult
{
    bool recorded = false;
    bool cancellationRequested = false;
    bool inferenceRequested = false;
    std::optional<long long> cancellationRequestId;
    std::optional<long long> workerAttemptId;
    std::string detail;
};

// Authoritative checkpoint-stop persistence used by the training worker.
// The caller owns the transaction; this function preserves non-null audit
// identity/ownership and classifies conflicts instead of overwriting them.
CheckpointStopRecordResult RecordCheckpointStopReached(
    pqxx::work& transaction,
    const std::optional<long long>& experimentId,
    long long workerAttemptId,
    int epoch,
    long long modelId);

// Shared CLI contract for the final authoritative persisted request status.
int RequestExitCodeForPersistedStatus(const std::string& status);

// Claims (when permitted), repairs stranded checkpoint state, and reconciles
// the exact active cancellation request. Every mutation is fenced by the
// persisted application owner and live lease.
// Returns false without mutation while a different owner holds a live lease.
bool ReconcileActiveCancellation(pqxx::work& transaction,
                                 const std::string& applicationOwner,
                                 bool claimExpiredLease);

// Recover only a committed global pause plan. The caller owns the transaction;
// recovery independently fences the current scheduler invocation and lease.
// A foreign live administrative owner is never displaced.
bool ReconcileActivePause(
    pqxx::work& transaction,
    const EA::SchedulerCore::SchedulerAuthorityContext& authority);

enum class GlobalControlBoundary
{
    BeforeIntentCommit,
    AfterIntentCommit,
    BeforeStop,
    AfterStop,
    BeforeWorkerPersistence,
    BeforePauseCommit,
    AfterPauseCommit,
    BeforeResumeCommit,
    AfterResumeCommit,
    DuringReconciliation
};
using GlobalControlFaultInjector =
    std::function<void(GlobalControlBoundary)>;

bool ReconcileActivePauseWithProcessOperationsForTesting(
    pqxx::work& transaction,
    const EA::SchedulerCore::SchedulerAuthorityContext& authority,
    ProcessOperations& processes,
    const GlobalControlFaultInjector& fault = {});

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
    ProcessOperations& processes,
    const GlobalControlFaultInjector& fault = {});

struct ExperimentPauseCommand
{
    long long experimentId = -1;
    bool dryRun = false;
    bool confirmed = false;
    std::string invocationIdentity;
    std::optional<std::string> requesterIdentity;
};

struct ExperimentResumeCommand
{
    long long experimentId = -1;
    bool dryRun = false;
    bool confirmed = false;
    std::string invocationIdentity;
    std::optional<std::string> requesterIdentity;
};

struct CampaignMaterializationControlCommand
{
    long long materializationId = -1;
    bool dryRun = false;
    bool confirmed = false;
    std::string invocationIdentity;
    std::optional<std::string> requesterIdentity;
};

// A deliberately narrow, non-scheduler administrative bridge for an exact
// externally executing experiment-worker attempt whose prior observation was
// inconclusive. It never signals a process; a positively absent worker may be
// terminalized together with its still-bound lifecycle row.
struct WorkerAttemptReconciliationCommand
{
    long long workerAttemptId = -1;
    bool dryRun = false;
    bool confirmed = false;
};

// Repairs only the historical failed-final-inference corruption proven by a
// unique durable attempt/result relationship. It never launches work or
// creates replacement persistence rows.
struct HistoricalFailedInferenceRecoveryCommand
{
    long long experimentId = -1;
    bool dryRun = false;
    bool confirmed = false;
};

int RunHistoricalFailedInferenceRecoveryCommand(
    const std::string& connectionString,
    const HistoricalFailedInferenceRecoveryCommand& command,
    std::ostream& output,
    std::ostream& error);

bool IsHistoricalFailedInferenceRecoveryCli(int argc, const char* argv[]);

int RunHistoricalFailedInferenceRecoveryCli(
    int argc,
    const char* argv[],
    const std::string& connectionString,
    std::ostream& output,
    std::ostream& error);

int RunWorkerAttemptReconciliationCommand(
    const std::string& connectionString,
    const WorkerAttemptReconciliationCommand& command,
    std::ostream& output,
    std::ostream& error);

bool IsWorkerAttemptReconciliationCli(int argc, const char* argv[]);

int RunWorkerAttemptReconciliationCli(
    int argc,
    const char* argv[],
    const std::string& connectionString,
    std::ostream& output,
    std::ostream& error);

int RunWorkerAttemptReconciliationCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const WorkerAttemptReconciliationCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes);

// Individual pause retains exact worker identity and in-memory state while
// releasing scheduler capacity. Resume only queues priority admission; it
// never sends SIGCONT directly.
int RunExperimentPauseCommand(const std::string& connectionString,
                              const ExperimentPauseCommand& command,
                              std::ostream& output,
                              std::ostream& error);

int RunExperimentPauseCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const ExperimentPauseCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes);

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

int RunCampaignMaterializationPauseCommand(
    const std::string& connectionString,
    const CampaignMaterializationControlCommand& command,
    std::ostream& output,
    std::ostream& error);

int RunCampaignMaterializationPauseCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const CampaignMaterializationControlCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes);

int RunCampaignMaterializationResumeCommand(
    const std::string& connectionString,
    const CampaignMaterializationControlCommand& command,
    std::ostream& output,
    std::ostream& error);

int RunCampaignMaterializationResumeCommandWithProcessOperationsForTesting(
    const std::string& connectionString,
    const CampaignMaterializationControlCommand& command,
    std::ostream& output,
    std::ostream& error,
    ProcessOperations& processes);

const char* ToString(Action action);
const char* ToString(CancellationMode mode);

} // namespace EA::GlobalExperimentControl
