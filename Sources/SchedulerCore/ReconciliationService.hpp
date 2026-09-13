#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace EA::SchedulerCore
{

enum class AttemptObservationAction
{
    Defer,
    RetainLive,
    ReconcileMissing
};

struct AttemptObservation
{
    bool inspectionSucceeded = false;
    bool processExists = false;
    bool identityMatches = false;
    bool persistedStopped = false;
    bool processStopped = false;
};

struct AttemptObservationPlan
{
    AttemptObservationAction action = AttemptObservationAction::Defer;
    std::string_view lifecycleState;
    std::string_view reconciliationResult;
    std::string_view diagnostic;
    bool capacityConsumed = true;
    bool restoreRunningLifecycle = false;
};

AttemptObservationPlan PlanAttemptObservation(
    const AttemptObservation& observation) noexcept;

enum class MissingStoppedWorkerDisposition
{
    RestartEligible,
    RestartPreemptedTrainFromCheckpoint,
    FailPreemptedTrainWithoutCheckpoint
};

struct MissingStoppedWorkerPlan
{
    MissingStoppedWorkerDisposition disposition =
        MissingStoppedWorkerDisposition::RestartEligible;
    std::string_view eventResult;
};

MissingStoppedWorkerPlan PlanMissingStoppedWorker(
    std::string_view phase,
    std::string_view lifecycleStatus,
    bool resumeRequested,
    std::string_view resumeOrigin,
    bool checkpointAvailable) noexcept;

struct MissingProcessTerminalPlan
{
    std::string_view attemptLifecycleState;
    std::string_view reconciliationResult;
    std::string_view diagnostic;
};

MissingProcessTerminalPlan PlanMissingProcessTerminalization(
    bool completedEvidence) noexcept;

struct OrphanedRunningAttempt
{
    long long workerAttemptId = -1;
    long long experimentId = -1;
    std::optional<long long> checkpointEvalId;
    std::string lifecycleState;
    std::string phase;
};

struct OrphanedRunningExperimentReconciliationOperations
{
    std::function<std::vector<OrphanedRunningAttempt>()> loadCandidates;
    std::function<AttemptObservation(const OrphanedRunningAttempt&)>
        observeProcess;
    std::function<void(
        const OrphanedRunningAttempt&,
        const AttemptObservationPlan&)>
        persistProcessObservation;
    std::function<bool(const OrphanedRunningAttempt&)>
        reconcileMissingProcess;
};

// Owns PID-bearing orphan process-observation sequencing. The supplied
// operations retain transaction, row-lock, exact-attempt, process-control,
// authority/fencing, and phase-specific persistence mechanics.
class ReconciliationService final
{
public:
    explicit ReconciliationService(
        OrphanedRunningExperimentReconciliationOperations operations);

    int recoverOrphanedRunningExperiments();

private:
    OrphanedRunningExperimentReconciliationOperations operations_;
};

} // namespace EA::SchedulerCore
