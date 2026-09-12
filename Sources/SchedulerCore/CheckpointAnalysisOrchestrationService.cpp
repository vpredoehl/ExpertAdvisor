#include "CheckpointAnalysisOrchestrationService.hpp"

#include <ostream>
#include <stdexcept>
#include <utility>

namespace EA::SchedulerCore
{

CheckpointAnalysisFinalizationPlan PlanCheckpointAnalysisFinalization(
    const CheckpointAnalysisWorkResult& work,
    const std::optional<long long>& currentInferenceResultId)
{
    const bool complete =
        work.success && currentInferenceResultId.has_value() &&
        *currentInferenceResultId == work.inferenceResultId;
    const std::string failure =
        work.error.empty()
            ? "checkpoint_analysis_source_changed_before_finalize"
            : work.error;
    return {
        complete,
        complete ? "completed" : "failed",
        complete ? "checkpoint_analysis_completed"
                 : "checkpoint_analysis_failed",
        complete ? "result_persisted_exactly_once" : failure};
}

CheckpointAnalysisOrchestrationService::
CheckpointAnalysisOrchestrationService(
    CheckpointAnalysisOperations operations,
    std::ostream& output,
    std::ostream& errors,
    long long processId)
    : operations_{std::move(operations)},
      output_{output},
      errors_{errors},
      processId_{processId}
{
    if (processId_ <= 0)
        throw std::invalid_argument("valid scheduler process id required");
    if (!operations_.claim || !operations_.afterClaim ||
        !operations_.execute || !operations_.afterWork ||
        !operations_.finalize || !operations_.generateReports)
    {
        throw std::invalid_argument(
            "complete checkpoint analysis operations required");
    }
}

int CheckpointAnalysisOrchestrationService::runOne()
{
    const std::optional<CheckpointAnalysisClaim> claim = operations_.claim();
    if (!claim)
        return 0;

    output_ << "CHECKPOINT_ANALYSIS_CLAIMED"
            << ",checkpoint_eval_id=" << claim->checkpointEvalId
            << ",worker_attempt_id=" << claim->workerAttemptId
            << ",capacity_class=analyze"
            << std::endl;
    output_ << "CHECKPOINT_ANALYSIS_WORKER_STARTED"
            << ",experiment_id=" << claim->experimentId
            << ",pid=" << processId_
            << ",operation=checkpoint_analyze"
            << ",model_id=" << claim->checkpointModelId
            << ",checkpoint_eval_id=" << claim->checkpointEvalId
            << std::endl;
    output_.flush();
    errors_.flush();

    operations_.afterClaim(*claim);
    const CheckpointAnalysisWorkResult work = operations_.execute(*claim);
    operations_.afterWork(*claim, work);
    const bool completed = operations_.finalize(*claim, work);
    if (completed)
        operations_.generateReports();
    return completed ? 0 : 1;
}

} // namespace EA::SchedulerCore
