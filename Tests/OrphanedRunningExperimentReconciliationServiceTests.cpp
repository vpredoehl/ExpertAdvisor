#include "SchedulerCore/ReconciliationService.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace EA::SchedulerCore;

class RecordingReconciliation
{
public:
    std::vector<OrphanedRunningAttempt> candidates;
    AttemptObservation observation{true, true, true, false, false};
    bool missingProcessApplied = true;
    bool failDuringRecovery = false;
    std::vector<std::string> calls;

    OrphanedRunningExperimentReconciliationOperations operations()
    {
        return {
            [this] {
                calls.push_back("load");
                return candidates;
            },
            [this](const OrphanedRunningAttempt& attempt) {
                calls.push_back("observe:" + attempt.phase);
                return observation;
            },
            [this](const OrphanedRunningAttempt& attempt,
                   const AttemptObservationPlan& plan) {
                calls.push_back(
                    "persist:" + attempt.phase + ":" +
                    std::string{plan.diagnostic});
            },
            [this](const OrphanedRunningAttempt& attempt) {
                calls.push_back("missing:" + attempt.phase);
                if (failDuringRecovery)
                    throw std::runtime_error("recovery_failed");
                return missingProcessApplied;
            }};
    }
};

OrphanedRunningAttempt Attempt(
    std::string phase,
    std::string state = "running")
{
    OrphanedRunningAttempt attempt;
    attempt.workerAttemptId = 11;
    attempt.experimentId = 22;
    attempt.lifecycleState = std::move(state);
    attempt.phase = std::move(phase);
    return attempt;
}

void ExpectMissingProcessPhase(const std::string& phase)
{
    RecordingReconciliation recording;
    recording.candidates = {Attempt(phase)};
    recording.observation = {true, false, false, false, false};
    ReconciliationService service{recording.operations()};
    assert(service.recoverOrphanedRunningExperiments() == 1);
    assert((recording.calls == std::vector<std::string>{
        "load", "observe:" + phase, "missing:" + phase}));
}

} // namespace

int main()
{
    {
        RecordingReconciliation recording;
        ReconciliationService service{recording.operations()};
        assert(service.recoverOrphanedRunningExperiments() == 0);
        assert((recording.calls == std::vector<std::string>{"load"}));
    }
    {
        RecordingReconciliation recording;
        recording.candidates = {Attempt("infer")};
        ReconciliationService service{recording.operations()};
        assert(service.recoverOrphanedRunningExperiments() == 0);
        assert((recording.calls == std::vector<std::string>{
            "load", "observe:infer",
            "persist:infer:live_worker_observed_without_relaunch"}));
    }
    {
        RecordingReconciliation recording;
        recording.candidates = {Attempt("train")};
        recording.observation = {false, false, false, false, false};
        ReconciliationService service{recording.operations()};
        assert(service.recoverOrphanedRunningExperiments() == 0);
        assert(recording.calls.back() ==
               "persist:train:process_identity_inspection_failed");
    }
    {
        RecordingReconciliation recording;
        recording.candidates = {Attempt("analyze")};
        recording.observation = {true, true, false, false, false};
        ReconciliationService service{recording.operations()};
        assert(service.recoverOrphanedRunningExperiments() == 0);
        assert(recording.calls.back() ==
               "persist:analyze:pid_reuse_or_worker_identity_mismatch");
    }

    ExpectMissingProcessPhase("train");
    ExpectMissingProcessPhase("infer");
    ExpectMissingProcessPhase("analyze");

    {
        RecordingReconciliation recording;
        auto checkpoint = Attempt("infer");
        checkpoint.checkpointEvalId = 33;
        recording.candidates = {checkpoint};
        recording.observation = {true, false, false, false, false};
        ReconciliationService service{recording.operations()};
        assert(service.recoverOrphanedRunningExperiments() == 1);
        assert(recording.calls.back() == "missing:infer");
    }
    {
        RecordingReconciliation recording;
        recording.candidates = {Attempt("infer")};
        recording.observation = {true, false, false, false, false};
        recording.missingProcessApplied = false;
        ReconciliationService service{recording.operations()};
        assert(service.recoverOrphanedRunningExperiments() == 0);
        assert((recording.calls == std::vector<std::string>{
            "load", "observe:infer", "missing:infer"}));
    }
    {
        RecordingReconciliation recording;
        recording.candidates = {Attempt("infer", "stopped")};
        recording.observation = {true, false, false, true, false};
        ReconciliationService service{recording.operations()};
        assert(service.recoverOrphanedRunningExperiments() == 1);
        assert((recording.calls == std::vector<std::string>{
            "load", "observe:infer", "missing:infer"}));
    }
    {
        RecordingReconciliation recording;
        recording.candidates = {Attempt("train")};
        recording.observation = {true, false, false, false, false};
        recording.failDuringRecovery = true;
        ReconciliationService service{recording.operations()};
        bool threw = false;
        try
        {
            (void)service.recoverOrphanedRunningExperiments();
        }
        catch (const std::runtime_error& error)
        {
            threw = std::string{error.what()} == "recovery_failed";
        }
        assert(threw);
        assert(recording.calls.back() == "missing:train");
    }
    {
        const auto completed = PlanMissingProcessTerminalization(true);
        assert(completed.attemptLifecycleState == "completed");
        assert(completed.reconciliationResult ==
               "process_missing_result_recovered");
        assert(completed.diagnostic == "exact_process_identity_absent");
        const auto failed = PlanMissingProcessTerminalization(false);
        assert(failed.attemptLifecycleState == "failed");
        assert(failed.reconciliationResult == "process_missing_no_result");
        assert(failed.diagnostic == "exact_process_identity_absent");
    }
    {
        bool rejected = false;
        try
        {
            ReconciliationService service{
                OrphanedRunningExperimentReconciliationOperations{}};
            (void)service;
        }
        catch (const std::invalid_argument& error)
        {
            rejected = std::string{error.what()} ==
                "complete orphan reconciliation operations required";
        }
        assert(rejected);
    }
    return 0;
}
