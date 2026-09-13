#include "SchedulerCore/SchedulerChildCompletionService.hpp"

#include <cassert>
#include <cerrno>
#include <csignal>
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
using EA::ExperimentScheduler::ChildStatusKind;
using EA::ExperimentScheduler::ObservedChildStatus;

SchedulerOwnedChild Child(
    pid_t pid,
    std::optional<long long> attemptId = 101,
    std::optional<long long> checkpointEvalId = std::nullopt)
{
    SchedulerOwnedChild child;
    child.pid = pid;
    child.workerAttemptId = attemptId;
    child.experimentId = 71;
    child.phase = checkpointEvalId ? "checkpoint_infer" : "train";
    child.operation = checkpointEvalId ? "infer" : "train";
    child.checkpointEvalId = checkpointEvalId;
    return child;
}

ObservedChildStatus Running()
{
    return {.kind = ChildStatusKind::Running};
}

ObservedChildStatus Exited(int exitCode)
{
    return {
        .kind = ChildStatusKind::Exited,
        .exitCode = exitCode};
}

ObservedChildStatus Signaled(int signalNumber, bool coreDumped = false)
{
    return {
        .kind = ChildStatusKind::Signaled,
        .exitCode = -signalNumber,
        .signalNumber = signalNumber,
        .coreDumped = coreDumped};
}

class RecordingCompletion
{
public:
    ObservedChildStatus observation = Exited(0);
    bool checkpointStop = false;
    bool exactAttempt = true;
    bool failpoint = false;
    bool throwOnExperimentPersistence = false;
    int observeCount = 0;
    int stopRequests = 0;
    std::vector<std::string> calls;
    std::optional<SchedulerChildCompletionEvidence> evidence;
    int finalizedExitCode = -999;
    std::optional<int> finalizedSignal;
    std::string finalizedDiagnostic;

    SchedulerChildCompletionOperations operations()
    {
        SchedulerChildCompletionOperations result;
        result.observeChild = [this](pid_t pid) {
            ++observeCount;
            calls.push_back("observe:" + std::to_string(pid));
            return observation;
        };
        result.observeTerminalCheckpointStop =
            [this](const SchedulerOwnedChild&,
                   const ObservedChildStatus&) {
                calls.push_back("checkpoint_stop");
                return checkpointStop;
            };
        result.injectStaleReaperReplacementForTest =
            [this](const SchedulerOwnedChild&) {
                calls.push_back("inject_stale_replacement");
            };
        result.verifyExactActiveAttempt =
            [this](const SchedulerOwnedChild&) {
                calls.push_back("verify_exact_attempt");
                return exactAttempt;
            };
        result.staleReaperFailpointEnabled = [this] {
            calls.push_back("failpoint_enabled");
            return failpoint;
        };
        result.requestSchedulerStop = [this] {
            calls.push_back("request_stop");
            ++stopRequests;
        };
        result.persistUnexpectedStatus =
            [this](const SchedulerOwnedChild&, int rawStatus) {
                calls.push_back(
                    "persist_unexpected:" + std::to_string(rawStatus));
            };
        result.persistCheckpointCompletion =
            [this](const SchedulerOwnedChild&,
                   const SchedulerChildCompletionEvidence& value) {
                calls.push_back("persist_checkpoint");
                evidence = value;
            };
        result.persistExperimentCompletion =
            [this](const SchedulerOwnedChild&,
                   const SchedulerChildCompletionEvidence& value) {
                calls.push_back("persist_experiment");
                evidence = value;
                if (throwOnExperimentPersistence)
                    throw std::runtime_error("persistence_failed");
            };
        result.finalizeWorkerAttempt =
            [this](const SchedulerOwnedChild&,
                   int exitCode,
                   const std::optional<int>& signalNumber,
                   std::string_view diagnostic) {
                calls.push_back("finalize");
                finalizedExitCode = exitCode;
                finalizedSignal = signalNumber;
                finalizedDiagnostic = diagnostic;
            };
        return result;
    }
};

} // namespace

int main()
{
    using namespace EA::SchedulerCore;

    {
        bool rejected = false;
        try
        {
            std::ostringstream diagnostics;
            SchedulerChildCompletionService service{{}, diagnostics};
        }
        catch (const std::invalid_argument& error)
        {
            rejected = std::string{error.what()} ==
                "complete scheduler child completion operations required";
        }
        assert(rejected);
    }

    {
        RecordingCompletion completion;
        completion.observation = Running();
        SchedulerOwnedChildren children{{31, Child(31)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.size() == 1);
        assert(!children.at(31).observedStatus);
        assert(!children.at(31).terminationObserved);
        assert((completion.calls == std::vector<std::string>{"observe:31"}));
        assert(diagnostics.str().empty());
    }

    {
        RecordingCompletion completion;
        completion.observation = Exited(0);
        SchedulerOwnedChildren children{{32, Child(32)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert((completion.calls == std::vector<std::string>{
            "observe:32",
            "checkpoint_stop",
            "inject_stale_replacement",
            "verify_exact_attempt",
            "persist_experiment",
            "finalize"}));
        assert(completion.evidence->exitCode == 0);
        assert(!completion.evidence->signalNumber);
        assert(completion.evidence->error ==
               "child_exit_code_0;phase=train;exit_code=0");
        assert(completion.finalizedExitCode == 0);
        assert(completion.finalizedDiagnostic == "child_exited");
        assert(diagnostics.str().find(
            "SCHEDULER_CHILD_EXITED,experiment_id=71,worker_pid=32,pid=32,phase=train,operation=train,exit_code=0\n") !=
            std::string::npos);
    }

    {
        RecordingCompletion completion;
        completion.observation = Exited(9);
        SchedulerOwnedChildren children{{33, Child(33)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert(completion.evidence->exitCode == 9);
        assert(completion.evidence->error ==
               "child_exit_code_9;phase=train;exit_code=9");
        assert(completion.finalizedExitCode == 9);
    }

    {
        RecordingCompletion completion;
        completion.observation = Signaled(SIGTERM, true);
        SchedulerOwnedChildren children{{34, Child(34)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert(completion.evidence->exitCode == -SIGTERM);
        assert(completion.evidence->signalNumber == SIGTERM);
        assert(completion.evidence->coreDumped);
        assert(completion.evidence->error ==
               "child_signal_15;phase=train;signal=15;signal_name=SIGTERM;core_dumped=1");
        assert(completion.finalizedSignal == SIGTERM);
        assert(completion.finalizedDiagnostic == "child_signaled");
        assert(diagnostics.str().find(
            "SCHEDULER_CHILD_SIGNALED,experiment_id=71,worker_pid=34,pid=34,phase=train,operation=train,signal_number=15,signal=15,signal_name=SIGTERM,core_dumped=1\n") !=
            std::string::npos);
    }

    {
        RecordingCompletion completion;
        completion.observation = Exited(0);
        SchedulerOwnedChildren children{{35, Child(35, 101, 501)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert(completion.calls[4] == "persist_checkpoint");
    }

    {
        RecordingCompletion completion;
        completion.observation = {
            .kind = ChildStatusKind::WaitError,
            .errorNumber = ECHILD};
        SchedulerOwnedChildren children{{36, Child(36)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert((completion.calls == std::vector<std::string>{"observe:36"}));
        assert(diagnostics.str() ==
               "SCHEDULER_CHILD_WAIT_ERROR,experiment_id=71,phase=train,operation=train,worker_pid=36,pid=36,errno=" +
                   std::to_string(ECHILD) +
                   ",error=waitpid_failed,ownership=no_longer_owned\n");
    }

    {
        RecordingCompletion completion;
        SchedulerOwnedChildren children{{37, Child(37, std::nullopt)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert((completion.calls == std::vector<std::string>{"observe:37"}));
        assert(diagnostics.str().find(
            "reason=worker_attempt_id_missing") != std::string::npos);
    }

    {
        RecordingCompletion completion;
        completion.checkpointStop = true;
        SchedulerOwnedChildren children{{38, Child(38)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert((completion.calls == std::vector<std::string>{
            "observe:38", "checkpoint_stop"}));
    }

    {
        RecordingCompletion completion;
        completion.exactAttempt = false;
        completion.failpoint = true;
        SchedulerOwnedChildren children{{39, Child(39)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert((completion.calls == std::vector<std::string>{
            "observe:39",
            "checkpoint_stop",
            "inject_stale_replacement",
            "verify_exact_attempt",
            "failpoint_enabled",
            "request_stop"}));
        assert(completion.stopRequests == 1);
        assert(diagnostics.str().find(
            "reason=exact_active_attempt_changed") != std::string::npos);
    }

    {
        RecordingCompletion completion;
        completion.observation = {
            .kind = ChildStatusKind::Unexpected,
            .rawStatus = 4991};
        SchedulerOwnedChildren children{{40, Child(40)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        service.reap(children);
        assert(children.empty());
        assert((completion.calls == std::vector<std::string>{
            "observe:40",
            "checkpoint_stop",
            "inject_stale_replacement",
            "verify_exact_attempt",
            "persist_unexpected:4991",
            "finalize"}));
        assert(completion.finalizedExitCode == -1);
        assert(!completion.finalizedSignal);
        assert(completion.finalizedDiagnostic == "unexpected_wait_status");
        assert(diagnostics.str().find(
            "errno=0,error=unexpected_wait_status,raw_status=4991") !=
            std::string::npos);
    }

    {
        RecordingCompletion completion;
        completion.throwOnExperimentPersistence = true;
        SchedulerOwnedChildren children{{41, Child(41)}};
        std::ostringstream diagnostics;
        SchedulerChildCompletionService service{
            completion.operations(), diagnostics};
        bool threw = false;
        try
        {
            service.reap(children);
        }
        catch (const std::runtime_error& error)
        {
            threw = std::string{error.what()} == "persistence_failed";
        }
        assert(threw);
        assert(children.size() == 1);
        assert(children.at(41).observedStatus);
        assert(children.at(41).terminationObserved);
        assert(children.at(41).exitDiagnosticEmitted);
        assert(completion.observeCount == 1);

        completion.throwOnExperimentPersistence = false;
        completion.calls.clear();
        service.reap(children);
        assert(children.empty());
        assert(completion.observeCount == 1);
        assert((completion.calls == std::vector<std::string>{
            "checkpoint_stop",
            "inject_stale_replacement",
            "verify_exact_attempt",
            "persist_experiment",
            "finalize"}));
        const std::string output = diagnostics.str();
        const std::string marker = "SCHEDULER_CHILD_EXITED";
        assert(output.find(marker) != std::string::npos);
        assert(output.find(marker, output.find(marker) + marker.size()) ==
               std::string::npos);
    }

    return 0;
}
