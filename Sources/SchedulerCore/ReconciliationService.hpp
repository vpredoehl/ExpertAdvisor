#pragma once

#include <string_view>

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

} // namespace EA::SchedulerCore
