#include "ReconciliationService.hpp"

namespace EA::SchedulerCore
{

AttemptObservationPlan PlanAttemptObservation(
    const AttemptObservation& observation) noexcept
{
    if (!observation.inspectionSucceeded)
    {
        return {
            AttemptObservationAction::Defer,
            "identity_ambiguous",
            "",
            "process_identity_inspection_failed",
            true,
            false};
    }
    if (!observation.processExists)
    {
        return {
            AttemptObservationAction::ReconcileMissing,
            "",
            "",
            "",
            false,
            false};
    }

    const bool remainsStopped =
        observation.identityMatches &&
        observation.persistedStopped &&
        observation.processStopped;
    const bool unexpectedlyExecuting =
        observation.identityMatches &&
        observation.persistedStopped &&
        !observation.processStopped;
    return {
        AttemptObservationAction::RetainLive,
        observation.identityMatches
            ? (remainsStopped ? "stopped" : "observed")
            : "identity_ambiguous",
        observation.identityMatches
            ? "valid_process_observed"
            : "identity_mismatch",
        observation.identityMatches
            ? "live_worker_observed_without_relaunch"
            : "pid_reuse_or_worker_identity_mismatch",
        !remainsStopped,
        unexpectedlyExecuting};
}

MissingStoppedWorkerPlan PlanMissingStoppedWorker(
    std::string_view phase,
    std::string_view lifecycleStatus,
    bool resumeRequested,
    std::string_view resumeOrigin,
    bool checkpointAvailable) noexcept
{
    const bool preemptedTrainRestart =
        phase == "train" &&
        lifecycleStatus == "pending" &&
        resumeRequested &&
        resumeOrigin == "preemption";
    if (!preemptedTrainRestart)
    {
        return {
            MissingStoppedWorkerDisposition::RestartEligible,
            "process_missing_restart_eligible"};
    }
    if (checkpointAvailable)
    {
        return {
            MissingStoppedWorkerDisposition::
                RestartPreemptedTrainFromCheckpoint,
            "phase_a_checkpoint_restart"};
    }
    return {
        MissingStoppedWorkerDisposition::
            FailPreemptedTrainWithoutCheckpoint,
        "failed_no_valid_checkpoint"};
}

MissingProcessTerminalPlan PlanMissingProcessTerminalization(
    bool completedEvidence) noexcept
{
    return completedEvidence
        ? MissingProcessTerminalPlan{
              "completed",
              "process_missing_result_recovered",
              "exact_process_identity_absent"}
        : MissingProcessTerminalPlan{
              "failed",
              "process_missing_no_result",
              "exact_process_identity_absent"};
}

} // namespace EA::SchedulerCore
