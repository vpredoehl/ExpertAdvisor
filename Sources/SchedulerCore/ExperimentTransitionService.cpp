#include "ExperimentTransitionService.hpp"

#include <ostream>
#include <stdexcept>

namespace EA::SchedulerCore
{
namespace
{

std::optional<long long> ControlModelId(
    const ExperimentTransitionRecord& experiment)
{
    if (experiment.experiment.lastModelId)
        return experiment.experiment.lastModelId;
    return experiment.experiment.resumeModelId;
}

std::string RetryPhaseForExperiment(
    const ExperimentTransitionRecord& experiment)
{
    if (experiment.phase == "train" || experiment.phase == "infer" ||
        experiment.phase == "analyze")
    {
        return experiment.phase;
    }
    if (experiment.phase == "done" && ControlModelId(experiment))
        return "analyze";
    return "train";
}

void PrintRejected(
    std::ostream& output,
    ExperimentTransitionAction action,
    long long experimentId,
    const std::string& reason)
{
    output << "SCHEDULER_CONTROL_REJECTED"
           << ",action=" << ExperimentTransitionActionName(action)
           << ",experiment_id=" << experimentId
           << ",reason=" << reason
           << std::endl;
}

void PrintRetryCheckpointSelection(
    std::ostream& output,
    long long experimentId,
    const RetryTrainingCheckpointSelection& selection)
{
    output << "SCHEDULER_RETRY_CHECKPOINT_SELECTION"
           << ",experiment_id=" << experimentId
           << ",previous_resume_model_id="
           << (selection.previousResumeModelId
                   ? std::to_string(*selection.previousResumeModelId)
                   : "NULL")
           << ",selected_resume_model_id="
           << (selection.selectedResumeModelId
                   ? std::to_string(*selection.selectedResumeModelId)
                   : "NULL")
           << ",selected_completed_epoch="
           << (selection.selectedCompletedEpoch
                   ? std::to_string(*selection.selectedCompletedEpoch)
                   : "NULL")
           << ",promotion=" << (selection.promoted ? "1" : "0")
           << ",reason=" << selection.reason
           << std::endl;
}

} // namespace

const char* ExperimentTransitionActionName(
    ExperimentTransitionAction action) noexcept
{
    switch (action)
    {
        case ExperimentTransitionAction::Cancel:
            return "cancel";
        case ExperimentTransitionAction::RetryFailed:
            return "retry_failed";
        case ExperimentTransitionAction::RequeueTraining:
            return "requeue_training";
        case ExperimentTransitionAction::RequeueAnalysis:
            return "requeue_analysis";
        case ExperimentTransitionAction::RequeueInference:
            return "requeue_inference";
    }
    return "unknown";
}

ExperimentTransitionPlan PlanExperimentTransition(
    ExperimentTransitionAction action,
    const ExperimentTransitionRecord& experiment)
{
    ExperimentTransitionPlan plan{
        false, experiment.status, experiment.phase, {}};
    switch (action)
    {
        case ExperimentTransitionAction::Cancel:
            if (experiment.status == "running")
                plan.rejectionReason =
                    "running_experiment_cannot_be_cancelled";
            else if (experiment.status != "pending" &&
                     experiment.status != "paused")
                plan.rejectionReason =
                    "cancel_requires_pending_or_paused_status";
            else
            {
                plan.accepted = true;
                plan.newStatus = "cancelled";
            }
            break;

        case ExperimentTransitionAction::RetryFailed:
            if (experiment.status != "failed")
                plan.rejectionReason = "retry_requires_failed_status";
            else
            {
                plan.accepted = true;
                plan.newStatus = "pending";
                plan.newPhase = RetryPhaseForExperiment(experiment);
            }
            break;

        case ExperimentTransitionAction::RequeueTraining:
            if (experiment.status == "running")
                plan.rejectionReason =
                    "running_experiment_cannot_be_requeued";
            else if (experiment.status == "completed" ||
                     experiment.status == "cancelled")
                plan.rejectionReason =
                    "terminal_experiment_cannot_be_requeued";
            else if (experiment.status != "pending" &&
                     experiment.status != "failed")
                plan.rejectionReason =
                    "requeue_training_requires_pending_or_failed_status";
            else if (experiment.phase != "train" &&
                     experiment.phase != "infer" &&
                     experiment.phase != "analyze")
                plan.rejectionReason =
                    "requeue_training_requires_supported_phase";
            else if (experiment.currentEpoch &&
                     *experiment.currentEpoch >=
                         experiment.experiment.targetEpochs)
                plan.rejectionReason =
                    "requeue_training_has_no_remaining_epochs";
            else if (experiment.activeWorkerAttemptId ||
                     experiment.hasAttachedWorkerAttempt)
                plan.rejectionReason =
                    "requeue_training_worker_attempt_attached";
            else if (experiment.workerPid ||
                     experiment.workerProcessGroupId ||
                     experiment.workerProcessStartIdentity ||
                     experiment.workerExecutable ||
                     experiment.workerCommandLine)
                plan.rejectionReason =
                    "requeue_training_worker_identity_present";
            else
            {
                plan.accepted = true;
                plan.newStatus = "pending";
                plan.newPhase = "train";
            }
            break;

        case ExperimentTransitionAction::RequeueAnalysis:
            if (experiment.status == "running")
                plan.rejectionReason =
                    "running_experiment_cannot_be_requeued";
            else if (!ControlModelId(experiment))
                plan.rejectionReason =
                    "requeue_analysis_requires_model_id";
            else
            {
                plan.accepted = true;
                plan.newStatus = "pending";
                plan.newPhase = "analyze";
            }
            break;

        case ExperimentTransitionAction::RequeueInference:
            if (experiment.status == "running")
                plan.rejectionReason =
                    "running_experiment_cannot_be_requeued";
            else if (!ControlModelId(experiment))
                plan.rejectionReason =
                    "requeue_inference_requires_model_id";
            else
            {
                plan.accepted = true;
                plan.newStatus = "pending";
                plan.newPhase = "infer";
            }
            break;
    }
    return plan;
}

ExperimentTransitionService::ExperimentTransitionService(
    SchedulerRepository& repository,
    ExperimentTransitionCheckpointSelector& checkpointSelector,
    std::ostream& output)
    : repository_{repository},
      checkpointSelector_{checkpointSelector},
      output_{output}
{
}

int ExperimentTransitionService::run(
    const ExperimentTransitionRequest& request)
{
    if (request.experimentId <= 0)
        throw std::invalid_argument("valid experiment id required");

    const bool willApply = request.confirmed && !request.dryRun;
    const std::optional<ExperimentTransitionRecord> experiment =
        repository_.loadExperimentTransition(
            request.experimentId, willApply);
    if (!experiment)
    {
        PrintRejected(
            output_, request.action, request.experimentId,
            "experiment_not_found");
        return 1;
    }

    const char* action = ExperimentTransitionActionName(request.action);
    output_ << "SCHEDULER_CONTROL_ATTEMPT"
            << ",action=" << action
            << ",experiment_id=" << experiment->experiment.experimentId
            << ",current_status=" << experiment->status
            << ",current_phase=" << experiment->phase
            << std::endl;

    const ExperimentTransitionPlan plan =
        PlanExperimentTransition(request.action, *experiment);
    if (!plan.accepted)
    {
        PrintRejected(
            output_, request.action, request.experimentId,
            plan.rejectionReason);
        return 1;
    }

    output_ << "Experiment " << experiment->experiment.experimentId << '\n'
            << "Current: status=" << experiment->status
            << " phase=" << experiment->phase << '\n'
            << "Requested: status=" << plan.newStatus
            << " phase=" << plan.newPhase << '\n';

    std::optional<long long> selectedResumeModelId;
    if (request.action == ExperimentTransitionAction::RetryFailed &&
        experiment->phase == "train")
    {
        const RetryTrainingCheckpointSelection selection =
            checkpointSelector_.selectRetryTrainingCheckpoint(*experiment);
        PrintRetryCheckpointSelection(
            output_, request.experimentId, selection);
        if (selection.promoted)
            selectedResumeModelId = selection.selectedResumeModelId;
    }
    else if (request.action ==
             ExperimentTransitionAction::RequeueTraining)
    {
        const RequeueTrainingCheckpointSelection selection =
            checkpointSelector_.selectRequeueTrainingCheckpoint(*experiment);
        if (!selection.selectedResumeModelId)
        {
            PrintRejected(
                output_, request.action, request.experimentId,
                selection.reason);
            return 1;
        }
        selectedResumeModelId = selection.selectedResumeModelId;
        output_ << "SCHEDULER_REQUEUE_TRAINING_SELECTION"
                << ",experiment_id=" << request.experimentId
                << ",resume_model_id=" << *selectedResumeModelId
                << ",completed_epoch="
                << (selection.selectedCompletedEpoch
                        ? std::to_string(*selection.selectedCompletedEpoch)
                        : "NULL")
                << ",scheduler_priority="
                << experiment->schedulerPriority
                << std::endl;
    }

    if (request.dryRun)
    {
        output_ << "SCHEDULER_CONTROL_DRY_RUN"
                << ",action=" << action
                << ",experiment_id=" << request.experimentId
                << ",new_status=" << plan.newStatus
                << ",new_phase=" << plan.newPhase
                << std::endl;
        return 0;
    }

    if (!request.confirmed)
    {
        output_ << "Use --yes to apply." << std::endl;
        return 0;
    }

    const ExperimentTransitionPersistenceResult applied =
        repository_.applyExperimentTransition({
            request.action,
            request.experimentId,
            experiment->status,
            experiment->phase,
            experiment->schedulerPriority,
            experiment->schedulerResumeOrigin,
            plan.newStatus,
            plan.newPhase,
            selectedResumeModelId});
    if (applied != ExperimentTransitionPersistenceResult::Updated)
        throw std::runtime_error(
            "scheduler_control_atomic_predicate_changed");

    output_ << "SCHEDULER_CONTROL_APPLIED"
            << ",action=" << action
            << ",experiment_id=" << request.experimentId
            << ",new_status=" << plan.newStatus
            << ",new_phase=" << plan.newPhase;
    if (request.action == ExperimentTransitionAction::RequeueInference)
    {
        output_ <<
            ",operator_forced_final_inference_rerun_requested=1";
    }
    output_ << std::endl;
    return 0;
}

} // namespace EA::SchedulerCore
