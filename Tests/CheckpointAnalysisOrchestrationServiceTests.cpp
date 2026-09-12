#include "SchedulerCore/CheckpointAnalysisOrchestrationService.hpp"

#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace EA::SchedulerCore;

class MemoryPort
{
public:
    std::optional<CheckpointAnalysisClaim> nextClaim;
    CheckpointAnalysisWorkResult work;
    bool finalizeResult = false;
    bool throwAfterClaim = false;
    int reports = 0;
    std::vector<std::string> calls;
    std::optional<CheckpointAnalysisClaim> finalizedClaim;
    std::optional<CheckpointAnalysisWorkResult> finalizedWork;

    CheckpointAnalysisOperations operations()
    {
        return {
            [this] { return claim(); },
            [this](const CheckpointAnalysisClaim& value) {
                afterClaim(value);
            },
            [this](const CheckpointAnalysisClaim& value) {
                return execute(value);
            },
            [this](const CheckpointAnalysisClaim& value,
                   const CheckpointAnalysisWorkResult& result) {
                afterWork(value, result);
            },
            [this](const CheckpointAnalysisClaim& value,
                   const CheckpointAnalysisWorkResult& result) {
                return finalize(value, result);
            },
            [this] { generateReports(); }};
    }

    std::optional<CheckpointAnalysisClaim> claim()
    {
        calls.push_back("claim");
        return nextClaim;
    }

    void afterClaim(const CheckpointAnalysisClaim&)
    {
        calls.push_back("after_claim");
        if (throwAfterClaim)
            throw std::runtime_error("authority_lost_after_claim");
    }

    CheckpointAnalysisWorkResult execute(
        const CheckpointAnalysisClaim&)
    {
        calls.push_back("execute");
        return work;
    }

    void afterWork(
        const CheckpointAnalysisClaim&,
        const CheckpointAnalysisWorkResult&)
    {
        calls.push_back("after_work");
    }

    bool finalize(
        const CheckpointAnalysisClaim& claim,
        const CheckpointAnalysisWorkResult& result)
    {
        calls.push_back("finalize");
        finalizedClaim = claim;
        finalizedWork = result;
        return finalizeResult;
    }

    void generateReports()
    {
        calls.push_back("reports");
        ++reports;
    }
};

CheckpointAnalysisClaim Claim()
{
    return {88, 41, 700, 20, 901};
}

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    {
        MemoryPort port;
        std::ostringstream output;
        std::ostringstream errors;
        CheckpointAnalysisOrchestrationService service{
            port.operations(), output, errors, 43210};
        assert(service.runOne() == 0);
        assert((port.calls == std::vector<std::string>{"claim"}));
        assert(output.str().empty());
    }

    {
        MemoryPort port;
        port.nextClaim = Claim();
        port.work = {true, {}, 777};
        port.finalizeResult = true;
        std::ostringstream output;
        std::ostringstream errors;
        CheckpointAnalysisOrchestrationService service{
            port.operations(), output, errors, 43210};
        assert(service.runOne() == 0);
        assert((port.calls == std::vector<std::string>{
            "claim", "after_claim", "execute", "after_work", "finalize",
            "reports"}));
        assert(port.reports == 1);
        assert(port.finalizedClaim->workerAttemptId == 901);
        assert(port.finalizedWork->inferenceResultId == 777);
        assert(output.str() ==
               "CHECKPOINT_ANALYSIS_CLAIMED,checkpoint_eval_id=88,"
               "worker_attempt_id=901,capacity_class=analyze\n"
               "CHECKPOINT_ANALYSIS_WORKER_STARTED,experiment_id=41,"
               "pid=43210,operation=checkpoint_analyze,model_id=700,"
               "checkpoint_eval_id=88\n");
        assert(errors.str().empty());
    }

    {
        MemoryPort port;
        port.nextClaim = Claim();
        port.work = {false, "missing_completed_checkpoint_inference", -1};
        std::ostringstream output;
        std::ostringstream errors;
        CheckpointAnalysisOrchestrationService service{
            port.operations(), output, errors, 43210};
        assert(service.runOne() == 1);
        assert(port.reports == 0);
        assert(port.finalizedWork->error ==
               "missing_completed_checkpoint_inference");
    }

    {
        const auto completed = PlanCheckpointAnalysisFinalization(
            {true, {}, 777}, 777);
        assert(completed.complete);
        assert(completed.attemptLifecycleState == "completed");
        assert(completed.reconciliationResult ==
               "checkpoint_analysis_completed");
        assert(completed.diagnostic == "result_persisted_exactly_once");

        const auto changed = PlanCheckpointAnalysisFinalization(
            {true, {}, 777}, 778);
        assert(!changed.complete);
        assert(changed.attemptLifecycleState == "failed");
        assert(changed.reconciliationResult == "checkpoint_analysis_failed");
        assert(changed.diagnostic ==
               "checkpoint_analysis_source_changed_before_finalize");

        const auto failed = PlanCheckpointAnalysisFinalization(
            {false, "checkpoint_inference_result_disappeared", 777}, 777);
        assert(!failed.complete);
        assert(failed.diagnostic ==
               "checkpoint_inference_result_disappeared");
    }

    {
        MemoryPort port;
        port.nextClaim = Claim();
        port.throwAfterClaim = true;
        std::ostringstream output;
        std::ostringstream errors;
        CheckpointAnalysisOrchestrationService service{
            port.operations(), output, errors, 43210};
        bool threw = false;
        try
        {
            (void)service.runOne();
        }
        catch (const std::runtime_error& error)
        {
            threw = std::string{error.what()} ==
                    "authority_lost_after_claim";
        }
        assert(threw);
        assert((port.calls ==
                std::vector<std::string>{"claim", "after_claim"}));
    }

    {
        MemoryPort port;
        std::ostringstream output;
        std::ostringstream errors;
        bool rejected = false;
        try
        {
            CheckpointAnalysisOrchestrationService invalid{
                port.operations(), output, errors, 0};
        }
        catch (const std::invalid_argument& error)
        {
            rejected = std::string{error.what()} ==
                       "valid scheduler process id required";
        }
        assert(rejected);
    }

    return 0;
}
