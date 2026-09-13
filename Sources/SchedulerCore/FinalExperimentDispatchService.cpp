#include "FinalExperimentDispatchService.hpp"

#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace EA::SchedulerCore
{
namespace
{

std::string_view PhaseName(FinalExperimentPhase phase)
{
    switch (phase)
    {
    case FinalExperimentPhase::Train:
        return "train";
    case FinalExperimentPhase::Infer:
        return "infer";
    case FinalExperimentPhase::Analyze:
        return "analyze";
    }
    throw std::logic_error("unknown final experiment phase");
}

bool CapacityHasSlot(int maximum, int used) noexcept
{
    return used < maximum;
}

} // namespace

FinalExperimentDispatchService::FinalExperimentDispatchService(
    FinalExperimentDispatchConfiguration configuration,
    FinalExperimentDispatchOperations operations,
    std::ostream& errors)
    : configuration_{configuration},
      operations_{std::move(operations)},
      errors_{errors}
{
    if (!operations_.load || !operations_.ensureLogDirectory ||
        !operations_.semanticPreflight ||
        !operations_.preemptOneLowerPriorityWorker ||
        !operations_.admitStoppedWorker ||
        !operations_.evaluateEligibility ||
        !operations_.emitDryRunCommand ||
        !operations_.reserveWorkerAttempt ||
        !operations_.capacityUsed ||
        !operations_.prepareReservedLaunch ||
        !operations_.launchPreparedWorker ||
        !operations_.logSkip || !operations_.printStats)
    {
        throw std::invalid_argument(
            "complete final experiment dispatch operations required");
    }
}

int FinalExperimentDispatchService::runTrain(bool cancellationOnly)
{
    return runPhase(FinalExperimentPhase::Train, cancellationOnly);
}

int FinalExperimentDispatchService::runInference()
{
    return runPhase(FinalExperimentPhase::Infer, false);
}

int FinalExperimentDispatchService::runAnalysis()
{
    return runPhase(FinalExperimentPhase::Analyze, false);
}

int FinalExperimentDispatchService::maximumCapacity(
    FinalExperimentPhase phase) const
{
    switch (phase)
    {
    case FinalExperimentPhase::Train:
        return configuration_.maxTrainProcesses;
    case FinalExperimentPhase::Infer:
        return configuration_.maxInferProcesses;
    case FinalExperimentPhase::Analyze:
        return configuration_.maxAnalyzeProcesses;
    }
    throw std::logic_error("unknown final experiment phase");
}

int FinalExperimentDispatchService::runPhase(
    FinalExperimentPhase phase,
    bool cancellationOnly)
{
    const FinalExperimentDispatchBatch batch =
        operations_.load(phase, cancellationOnly);
    if (!batch.launchAllowed)
        return 0;

    FinalExperimentDispatchStats stats;
    stats.phase = phase;
    stats.examined = static_cast<int>(batch.candidates.size());
    stats.freeSlots = batch.freeSlots;
    operations_.ensureLogDirectory();

    const int maximum = maximumCapacity(phase);
    int result = 0;
    for (const FinalExperimentDispatchCandidate& candidate :
         batch.candidates)
    {
        if (phase != FinalExperimentPhase::Analyze &&
            !operations_.semanticPreflight(candidate, phase))
        {
            ++stats.skipped;
            continue;
        }

        if (!configuration_.dryRun &&
            (phase == FinalExperimentPhase::Infer ||
             (phase == FinalExperimentPhase::Train && !cancellationOnly)))
        {
            operations_.preemptOneLowerPriorityWorker(
                candidate, phase, maximum);
        }

        if (!configuration_.dryRun &&
            phase != FinalExperimentPhase::Analyze &&
            candidate.hasActiveWorkerAttempt)
        {
            const FinalExperimentStoppedAdmission admission =
                operations_.admitStoppedWorker(candidate, phase, maximum);
            if (admission == FinalExperimentStoppedAdmission::Admitted)
            {
                ++stats.launched;
                continue;
            }
            if (admission ==
                FinalExperimentStoppedAdmission::DeferredNoCapacity)
            {
                ++stats.skipped;
                operations_.logSkip(
                    phase,
                    candidate.experimentId,
                    phase == FinalExperimentPhase::Train
                        ? "global_train_slots_full_stopped_worker_waiting"
                        : "global_infer_slots_full_stopped_worker_waiting");
                break;
            }
            if (admission == FinalExperimentStoppedAdmission::DeferredUnsafe ||
                admission == FinalExperimentStoppedAdmission::NotApplicable)
            {
                ++stats.skipped;
                operations_.logSkip(
                    phase,
                    candidate.experimentId,
                    "stopped_worker_admission_deferred");
                continue;
            }
        }

        if (phase == FinalExperimentPhase::Infer)
        {
            const FinalExperimentEligibility eligibility =
                operations_.evaluateEligibility(candidate, phase);
            if (eligibility != FinalExperimentEligibility::Eligible)
            {
                ++stats.skipped;
                if (eligibility == FinalExperimentEligibility::Failed)
                    result = 1;
                continue;
            }
        }
        else if (phase == FinalExperimentPhase::Analyze &&
                 !candidate.hasLastModel)
        {
            ++stats.skipped;
            continue;
        }

        if (configuration_.dryRun)
        {
            if (stats.launched >= stats.freeSlots)
            {
                ++stats.skipped;
                continue;
            }
            operations_.emitDryRunCommand(candidate, phase);
            ++stats.launched;
            continue;
        }

        if (phase == FinalExperimentPhase::Train)
        {
            const FinalExperimentEligibility eligibility =
                operations_.evaluateEligibility(candidate, phase);
            if (eligibility != FinalExperimentEligibility::Eligible)
            {
                ++stats.skipped;
                if (eligibility == FinalExperimentEligibility::Failed)
                    result = 1;
                continue;
            }
        }

        const std::optional<long long> attemptId =
            operations_.reserveWorkerAttempt(
                candidate, phase, maximum, cancellationOnly);
        if (!attemptId)
        {
            ++stats.skipped;
            const int used = operations_.capacityUsed(phase);
            const bool hasSlot = CapacityHasSlot(maximum, used);
            if (phase != FinalExperimentPhase::Analyze)
            {
                operations_.logSkip(
                    phase,
                    candidate.experimentId,
                    hasSlot
                        ? "claim_changed"
                        : (phase == FinalExperimentPhase::Train
                               ? "global_train_slots_full"
                               : "global_infer_slots_full"));
            }
            if (!hasSlot)
                break;
            continue;
        }

        if (phase == FinalExperimentPhase::Train)
            operations_.prepareReservedLaunch(candidate, phase, *attemptId);

        try
        {
            if (phase != FinalExperimentPhase::Train)
                operations_.prepareReservedLaunch(candidate, phase, *attemptId);
            operations_.launchPreparedWorker(candidate, phase, *attemptId);
            ++stats.launched;
        }
        catch (const std::exception& error)
        {
            errors_ << "EXPERIMENT_FAILED"
                    << ",experiment_id=" << candidate.experimentId
                    << ",phase=" << PhaseName(phase)
                    << ",worker_attempt_id=" << *attemptId
                    << ",error=" << error.what()
                    << std::endl;
            result = 1;
        }
    }

    operations_.printStats(stats);
    return result;
}

} // namespace EA::SchedulerCore
