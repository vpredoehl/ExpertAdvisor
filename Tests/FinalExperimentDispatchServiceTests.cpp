#include "SchedulerCore/FinalExperimentDispatchService.hpp"

#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using namespace EA::SchedulerCore;

std::string PhaseName(FinalExperimentPhase phase)
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
    return "unknown";
}

class RecordingDispatch
{
public:
    FinalExperimentDispatchBatch batch{true, 1, {}};
    std::vector<std::string> calls;
    FinalExperimentStoppedAdmission stopped =
        FinalExperimentStoppedAdmission::MissingProcessFallbackReady;
    FinalExperimentEligibility eligibility =
        FinalExperimentEligibility::Eligible;
    std::optional<long long> reservation = 41;
    int used = 0;
    bool semanticAccepted = true;
    bool throwDuringLaunch = false;
    FinalExperimentDispatchStats stats;

    FinalExperimentDispatchOperations operations()
    {
        return {
            [this](FinalExperimentPhase phase, bool cancellationOnly) {
                calls.push_back(
                    "load:" + PhaseName(phase) +
                    (cancellationOnly ? ":cancellation" : ":normal"));
                return batch;
            },
            [this] { calls.push_back("ensure_log"); },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase) {
                calls.push_back(
                    "semantic:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId));
                return semanticAccepted;
            },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase,
                   int maximum) {
                calls.push_back(
                    "preempt:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId) + ":" +
                    std::to_string(maximum));
            },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase,
                   int maximum) {
                calls.push_back(
                    "admit:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId) + ":" +
                    std::to_string(maximum));
                return stopped;
            },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase) {
                calls.push_back(
                    "eligible:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId));
                return eligibility;
            },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase) {
                calls.push_back(
                    "dry_run:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId));
            },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase,
                   int maximum,
                   bool cancellationOnly) {
                calls.push_back(
                    "reserve:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId) + ":" +
                    std::to_string(maximum) +
                    (cancellationOnly ? ":cancellation" : ":normal"));
                return reservation;
            },
            [this](FinalExperimentPhase phase) {
                calls.push_back("capacity:" + PhaseName(phase));
                return used;
            },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase,
                   long long attemptId) {
                calls.push_back(
                    "prepare:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId) + ":" +
                    std::to_string(attemptId));
            },
            [this](const FinalExperimentDispatchCandidate& candidate,
                   FinalExperimentPhase phase,
                   long long attemptId) {
                calls.push_back(
                    "launch:" + PhaseName(phase) + ":" +
                    std::to_string(candidate.experimentId) + ":" +
                    std::to_string(attemptId));
                if (throwDuringLaunch)
                    throw std::runtime_error("launch_failed");
            },
            [this](FinalExperimentPhase phase,
                   long long experimentId,
                   std::string_view reason) {
                calls.push_back(
                    "skip:" + PhaseName(phase) + ":" +
                    std::to_string(experimentId) + ":" +
                    std::string{reason});
            },
            [this](const FinalExperimentDispatchStats& value) {
                calls.push_back("stats:" + PhaseName(value.phase));
                stats = value;
            }};
    }
};

FinalExperimentDispatchConfiguration Configuration(bool dryRun = false)
{
    return {dryRun, 2, 3, 4};
}

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    {
        RecordingDispatch dispatch;
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runTrain(false) == 0);
        assert((dispatch.calls == std::vector<std::string>{
            "load:train:normal", "ensure_log", "stats:train"}));
        assert(dispatch.stats.examined == 0);
        assert(dispatch.stats.launched == 0);
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.candidates = {{0, 7, false, true}};
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runTrain(false) == 0);
        assert((dispatch.calls == std::vector<std::string>{
            "load:train:normal",
            "ensure_log",
            "semantic:train:7",
            "preempt:train:7:2",
            "eligible:train:7",
            "reserve:train:7:2:normal",
            "prepare:train:7:41",
            "launch:train:7:41",
            "stats:train"}));
        assert(dispatch.stats.examined == 1);
        assert(dispatch.stats.skipped == 0);
        assert(dispatch.stats.launched == 1);
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.candidates = {{0, 8, false, true}};
        dispatch.reservation = std::nullopt;
        dispatch.used = 3;
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runInference() == 0);
        assert((dispatch.calls == std::vector<std::string>{
            "load:infer:normal",
            "ensure_log",
            "semantic:infer:8",
            "preempt:infer:8:3",
            "eligible:infer:8",
            "reserve:infer:8:3:normal",
            "capacity:infer",
            "skip:infer:8:global_infer_slots_full",
            "stats:infer"}));
        assert(dispatch.stats.skipped == 1);
        assert(dispatch.stats.launched == 0);
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.candidates = {
            {0, 9, true, true}, {1, 10, false, true}};
        dispatch.stopped =
            FinalExperimentStoppedAdmission::DeferredNoCapacity;
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runTrain(true) == 0);
        assert((dispatch.calls == std::vector<std::string>{
            "load:train:cancellation",
            "ensure_log",
            "semantic:train:9",
            "admit:train:9:2",
            "skip:train:9:global_train_slots_full_stopped_worker_waiting",
            "stats:train"}));
        assert(dispatch.stats.examined == 2);
        assert(dispatch.stats.skipped == 1);
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.candidates = {
            {0, 11, false, false}, {1, 12, false, true}};
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runAnalysis() == 0);
        assert((dispatch.calls == std::vector<std::string>{
            "load:analyze:normal",
            "ensure_log",
            "reserve:analyze:12:4:normal",
            "prepare:analyze:12:41",
            "launch:analyze:12:41",
            "stats:analyze"}));
        assert(dispatch.stats.skipped == 1);
        assert(dispatch.stats.launched == 1);
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.candidates = {{0, 13, false, true}};
        dispatch.eligibility = FinalExperimentEligibility::Failed;
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runTrain(false) == 1);
        assert(dispatch.stats.skipped == 1);
        assert(dispatch.stats.launched == 0);
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.candidates = {{0, 14, false, true}};
        dispatch.throwDuringLaunch = true;
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runInference() == 1);
        assert(dispatch.stats.launched == 0);
        assert(errors.str() ==
               "EXPERIMENT_FAILED,experiment_id=14,phase=infer,"
               "worker_attempt_id=41,error=launch_failed\n");
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.freeSlots = 1;
        dispatch.batch.candidates = {
            {0, 15, false, true}, {1, 16, false, true}};
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(true), dispatch.operations(), errors};
        assert(service.runInference() == 0);
        assert((dispatch.calls == std::vector<std::string>{
            "load:infer:normal",
            "ensure_log",
            "semantic:infer:15",
            "eligible:infer:15",
            "dry_run:infer:15",
            "semantic:infer:16",
            "eligible:infer:16",
            "stats:infer"}));
        assert(dispatch.stats.skipped == 1);
        assert(dispatch.stats.launched == 1);
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.candidates = {{0, 17, false, true}};
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(true), dispatch.operations(), errors};
        assert(service.runTrain(false) == 0);
        assert((dispatch.calls == std::vector<std::string>{
            "load:train:normal",
            "ensure_log",
            "semantic:train:17",
            "dry_run:train:17",
            "stats:train"}));
    }

    {
        RecordingDispatch dispatch;
        dispatch.batch.launchAllowed = false;
        std::ostringstream errors;
        FinalExperimentDispatchService service{
            Configuration(), dispatch.operations(), errors};
        assert(service.runAnalysis() == 0);
        assert((dispatch.calls ==
                std::vector<std::string>{"load:analyze:normal"}));
    }

    {
        RecordingDispatch dispatch;
        FinalExperimentDispatchOperations incomplete =
            dispatch.operations();
        incomplete.reserveWorkerAttempt = {};
        std::ostringstream errors;
        bool rejected = false;
        try
        {
            FinalExperimentDispatchService service{
                Configuration(), std::move(incomplete), errors};
        }
        catch (const std::invalid_argument& error)
        {
            rejected = std::string{error.what()} ==
                "complete final experiment dispatch operations required";
        }
        assert(rejected);
    }

    return 0;
}
