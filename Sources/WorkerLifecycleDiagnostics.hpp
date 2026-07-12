#pragma once

#include <cerrno>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <unistd.h>

namespace EA::ExperimentScheduler
{

struct ChildLifecycleDiagnosticState
{
    bool terminationObserved = false;
    bool exitDiagnosticEmitted = false;
};

inline bool MarkChildTerminationObserved(ChildLifecycleDiagnosticState& state)
{
    const bool firstObservation = !state.terminationObserved;
    state.terminationObserved = true;
    return firstObservation;
}

inline bool MarkChildExitDiagnosticEmitted(ChildLifecycleDiagnosticState& state)
{
    if (state.exitDiagnosticEmitted)
        return false;
    state.exitDiagnosticEmitted = true;
    return true;
}

inline bool ShouldReconcileSchedulerOrphan(bool processExists,
                                           bool terminationObserved,
                                           bool successfulCompletionRecorded)
{
    return !processExists && !terminationObserved && !successfulCompletionRecorded;
}

inline bool IsCurrentWorkerAttemptEvidence(double evidenceEpoch,
                                           double attemptStartedEpoch)
{
    return evidenceEpoch >= attemptStartedEpoch;
}

inline std::string SchedulerLaunchFailureError(int errorNumber)
{
    return "launch_failed_errno_" + std::to_string(errorNumber);
}

inline std::string SchedulerChildLaunchFailureDiagnostic(
    long long experimentId,
    const std::string& phase,
    int errorNumber,
    const std::string& commandLine)
{
    return "SCHEDULER_CHILD_LAUNCH_FAILED,experiment_id=" +
           std::to_string(experimentId) +
           ",phase=" + phase +
           ",operation=" + phase +
           ",errno=" + std::to_string(errorNumber) +
           ",error=" + std::strerror(errorNumber) +
           ",command_line=" + commandLine;
}

inline void FlushWorkerLifecycleLogs()
{
    std::cout.flush();
    std::cerr.flush();
}

inline void LogWorkerStarted(
    const char* marker,
    const std::string& operation,
    const std::optional<long long>& experimentId = std::nullopt,
    const std::optional<long long>& modelId = std::nullopt,
    const std::optional<long long>& checkpointEvalId = std::nullopt)
{
    std::cout << marker
              << ",experiment_id="
              << (experimentId.has_value() ? std::to_string(*experimentId) : "NULL")
              << ",pid=" << static_cast<long long>(::getpid())
              << ",operation=" << operation
              << ",model_id="
              << (modelId.has_value() ? std::to_string(*modelId) : "NULL")
              << ",checkpoint_eval_id="
              << (checkpointEvalId.has_value()
                      ? std::to_string(*checkpointEvalId)
                      : "NULL")
              << std::endl;
    FlushWorkerLifecycleLogs();
}

} // namespace EA::ExperimentScheduler
