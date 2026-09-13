#include "SchedulerChildCompletionService.hpp"

#include <csignal>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace EA::SchedulerCore
{
namespace
{

using EA::ExperimentScheduler::ChildStatusKind;
using EA::ExperimentScheduler::ObservedChildStatus;

const char* SignalName(int signalNumber) noexcept
{
    switch (signalNumber)
    {
        case SIGHUP: return "SIGHUP";
        case SIGINT: return "SIGINT";
        case SIGQUIT: return "SIGQUIT";
        case SIGILL: return "SIGILL";
        case SIGABRT: return "SIGABRT";
        case SIGFPE: return "SIGFPE";
        case SIGKILL: return "SIGKILL";
        case SIGSEGV: return "SIGSEGV";
        case SIGPIPE: return "SIGPIPE";
        case SIGALRM: return "SIGALRM";
        case SIGTERM: return "SIGTERM";
#ifdef SIGBUS
        case SIGBUS: return "SIGBUS";
#endif
        default: return "UNKNOWN";
    }
}

SchedulerChildCompletionEvidence CompletionEvidence(
    const SchedulerOwnedChild& child,
    const ObservedChildStatus& observed)
{
    SchedulerChildCompletionEvidence evidence;
    evidence.exitCode = observed.exitCode;
    evidence.coreDumped = observed.coreDumped;
    if (observed.kind == ChildStatusKind::Signaled)
        evidence.signalNumber = observed.signalNumber;

    if (evidence.signalNumber)
    {
        evidence.error =
            "child_signal_" + std::to_string(*evidence.signalNumber) +
            ";phase=" + child.phase +
            ";signal=" + std::to_string(*evidence.signalNumber) +
            ";signal_name=" + SignalName(*evidence.signalNumber) +
            ";core_dumped=" + (evidence.coreDumped ? "1" : "0");
    }
    else
    {
        evidence.error =
            "child_exit_code_" + std::to_string(evidence.exitCode) +
            ";phase=" + child.phase +
            ";exit_code=" + std::to_string(evidence.exitCode);
    }
    return evidence;
}

} // namespace

SchedulerChildCompletionService::SchedulerChildCompletionService(
    SchedulerChildCompletionOperations operations,
    std::ostream& diagnostics)
    : operations_{std::move(operations)}, diagnostics_{diagnostics}
{
    if (!operations_.observeChild ||
        !operations_.observeTerminalCheckpointStop ||
        !operations_.injectStaleReaperReplacementForTest ||
        !operations_.verifyExactActiveAttempt ||
        !operations_.staleReaperFailpointEnabled ||
        !operations_.requestSchedulerStop ||
        !operations_.persistUnexpectedStatus ||
        !operations_.persistCheckpointCompletion ||
        !operations_.persistExperimentCompletion ||
        !operations_.finalizeWorkerAttempt)
    {
        throw std::invalid_argument(
            "complete scheduler child completion operations required");
    }
}

void SchedulerChildCompletionService::reap(
    SchedulerOwnedChildren& children)
{
    for (auto childIt = children.begin(); childIt != children.end();)
    {
        SchedulerOwnedChild& child = childIt->second;
        if (!child.observedStatus)
        {
            const ObservedChildStatus observed =
                operations_.observeChild(child.pid);
            if (observed.kind == ChildStatusKind::Running)
            {
                ++childIt;
                continue;
            }
            if (observed.kind == ChildStatusKind::WaitError)
            {
                diagnostics_ << "SCHEDULER_CHILD_WAIT_ERROR"
                             << ",experiment_id=" << child.experimentId
                             << ",phase=" << child.phase
                             << ",operation=" << child.operation
                             << ",worker_pid=" << child.pid
                             << ",pid=" << child.pid
                             << ",errno=" << observed.errorNumber
                             << ",error=waitpid_failed"
                             << ",ownership=no_longer_owned"
                             << std::endl;
                childIt = children.erase(childIt);
                continue;
            }
            child.observedStatus = observed;
            child.terminationObserved = true;
        }

        const ObservedChildStatus& observed = *child.observedStatus;
        if (!child.workerAttemptId)
        {
            diagnostics_ << "SCHEDULER_STALE_CHILD_REAP_REJECTED"
                         << ",experiment_id=" << child.experimentId
                         << ",pid=" << child.pid
                         << ",reason=worker_attempt_id_missing"
                         << std::endl;
            childIt = children.erase(childIt);
            continue;
        }
        if (operations_.observeTerminalCheckpointStop(child, observed))
        {
            childIt = children.erase(childIt);
            continue;
        }

        operations_.injectStaleReaperReplacementForTest(child);
        if (!operations_.verifyExactActiveAttempt(child))
        {
            diagnostics_ << "SCHEDULER_STALE_CHILD_REAP_REJECTED"
                         << ",experiment_id=" << child.experimentId
                         << ",worker_attempt_id=" << *child.workerAttemptId
                         << ",pid=" << child.pid
                         << ",reason=exact_active_attempt_changed"
                         << std::endl;
            if (operations_.staleReaperFailpointEnabled())
                operations_.requestSchedulerStop();
            childIt = children.erase(childIt);
            continue;
        }

        if (observed.kind != ChildStatusKind::Exited &&
            observed.kind != ChildStatusKind::Signaled)
        {
            diagnostics_ << "SCHEDULER_CHILD_WAIT_ERROR"
                         << ",experiment_id=" << child.experimentId
                         << ",phase=" << child.phase
                         << ",operation=" << child.operation
                         << ",worker_pid=" << child.pid
                         << ",pid=" << child.pid
                         << ",errno=0,error=unexpected_wait_status"
                         << ",raw_status=" << observed.rawStatus
                         << std::endl;
            operations_.persistUnexpectedStatus(child, observed.rawStatus);
            operations_.finalizeWorkerAttempt(
                child, -1, std::nullopt, "unexpected_wait_status");
            childIt = children.erase(childIt);
            continue;
        }

        const SchedulerChildCompletionEvidence evidence =
            CompletionEvidence(child, observed);
        if (!child.exitDiagnosticEmitted)
        {
            child.exitDiagnosticEmitted = true;
            if (observed.kind == ChildStatusKind::Exited)
            {
                diagnostics_ << "SCHEDULER_CHILD_EXITED"
                             << ",experiment_id=" << child.experimentId
                             << ",worker_pid=" << child.pid
                             << ",pid=" << child.pid
                             << ",phase=" << child.phase
                             << ",operation=" << child.operation
                             << ",exit_code=" << evidence.exitCode
                             << std::endl;
            }
            else
            {
                diagnostics_ << "SCHEDULER_CHILD_SIGNALED"
                             << ",experiment_id=" << child.experimentId
                             << ",worker_pid=" << child.pid
                             << ",pid=" << child.pid
                             << ",phase=" << child.phase
                             << ",operation=" << child.operation
                             << ",signal_number=" << *evidence.signalNumber
                             << ",signal=" << *evidence.signalNumber
                             << ",signal_name="
                             << SignalName(*evidence.signalNumber)
                             << ",core_dumped="
                             << (evidence.coreDumped ? 1 : 0)
                             << std::endl;
            }
        }

        if (child.checkpointEvalId)
            operations_.persistCheckpointCompletion(child, evidence);
        else
            operations_.persistExperimentCompletion(child, evidence);
        operations_.finalizeWorkerAttempt(
            child,
            evidence.exitCode,
            evidence.signalNumber,
            observed.kind == ChildStatusKind::Exited
                ? "child_exited"
                : "child_signaled");
        childIt = children.erase(childIt);
    }
}

} // namespace EA::SchedulerCore
