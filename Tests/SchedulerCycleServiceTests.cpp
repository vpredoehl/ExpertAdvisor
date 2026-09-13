#include "SchedulerCore/SchedulerCycleService.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using EA::SchedulerCore::SchedulerCycleOperations;
using EA::SchedulerCore::SchedulerCyclePreparation;
using EA::SchedulerCore::SchedulerCycleService;

class RecordingCycle
{
public:
    SchedulerCyclePreparation preparation{
        true, true, false, false, 0};
    std::vector<std::string> calls;
    int trainResult = 0;
    int finalInferenceResult = 0;
    int finalAnalysisResult = 0;
    int checkpointInferenceResult = 0;
    int checkpointAnalysisResult = 0;
    bool throwAuthorityLoss = false;

    SchedulerCycleOperations operations()
    {
        return {
            [this] { calls.push_back("begin_poll"); },
            [this] {
                calls.push_back("prepare");
                if (throwAuthorityLoss)
                    throw std::runtime_error("authority_lost");
                return preparation;
            },
            [this](bool cancellationOnly) {
                calls.push_back(
                    cancellationOnly ? "train_cancellation" : "train");
                return trainResult;
            },
            [this] {
                calls.push_back("final_inference");
                return finalInferenceResult;
            },
            [this] {
                calls.push_back("final_analysis");
                return finalAnalysisResult;
            },
            [this] {
                calls.push_back("checkpoint_inference");
                return checkpointInferenceResult;
            },
            [this] {
                calls.push_back("checkpoint_analysis");
                return checkpointAnalysisResult;
            },
            [this] { calls.push_back("finish_poll"); }};
    }
};

} // namespace

int main()
{
    {
        RecordingCycle cycle;
        cycle.preparation.normalSchedulingAllowed = false;
        SchedulerCycleService service{cycle.operations()};
        assert(service.runOnce() == 0);
        assert((cycle.calls == std::vector<std::string>{
            "begin_poll", "prepare", "finish_poll"}));
    }

    {
        RecordingCycle cycle;
        cycle.preparation.result = 1;
        cycle.trainResult = 2;
        cycle.finalInferenceResult = 4;
        cycle.finalAnalysisResult = 8;
        cycle.checkpointInferenceResult = 16;
        cycle.checkpointAnalysisResult = 32;
        SchedulerCycleService service{cycle.operations()};
        assert(service.runOnce() == 63);
        assert((cycle.calls == std::vector<std::string>{
            "begin_poll",
            "prepare",
            "train",
            "final_inference",
            "final_analysis",
            "checkpoint_inference",
            "checkpoint_analysis",
            "finish_poll"}));
    }

    {
        RecordingCycle cycle;
        cycle.preparation.normalSchedulingAllowed = false;
        cycle.preparation.cancellationCheckpointTrainAllowed = true;
        cycle.preparation.cancellationInferenceAllowed = true;
        cycle.trainResult = 1;
        cycle.checkpointInferenceResult = 2;
        SchedulerCycleService service{cycle.operations()};
        assert(service.runOnce() == 3);
        assert((cycle.calls == std::vector<std::string>{
            "begin_poll",
            "prepare",
            "train_cancellation",
            "checkpoint_inference",
            "finish_poll"}));
    }

    {
        RecordingCycle cycle;
        cycle.throwAuthorityLoss = true;
        SchedulerCycleService service{cycle.operations()};
        bool rejected = false;
        try
        {
            (void)service.runOnce();
        }
        catch (const std::runtime_error& error)
        {
            rejected = std::string{error.what()} == "authority_lost";
        }
        assert(rejected);
        assert((cycle.calls == std::vector<std::string>{
            "begin_poll", "prepare"}));
    }

    {
        RecordingCycle cycle;
        cycle.preparation.ready = false;
        cycle.preparation.result = 1;
        SchedulerCycleService service{cycle.operations()};
        assert(service.runOnce() == 1);
        assert((cycle.calls == std::vector<std::string>{
            "begin_poll", "prepare"}));
    }

    {
        RecordingCycle cycle;
        SchedulerCycleOperations incomplete = cycle.operations();
        incomplete.runTrain = {};
        bool rejected = false;
        try
        {
            SchedulerCycleService service{std::move(incomplete)};
        }
        catch (const std::invalid_argument& error)
        {
            rejected = std::string{error.what()} ==
                "complete scheduler cycle operations required";
        }
        assert(rejected);
    }

    return 0;
}
