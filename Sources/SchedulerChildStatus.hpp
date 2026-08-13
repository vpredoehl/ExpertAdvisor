#pragma once

#include <cerrno>
#include <cstddef>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace EA::ExperimentScheduler
{

enum class ChildStatusKind
{
    Running,
    Exited,
    Signaled,
    WaitError,
    Unexpected
};

struct ObservedChildStatus
{
    ChildStatusKind kind = ChildStatusKind::Unexpected;
    int rawStatus = 0;
    int exitCode = -1;
    int signalNumber = 0;
    int errorNumber = 0;
    bool coreDumped = false;
};

inline void WriteSchedulerChildExecFailureDiagnostic(int fd,
                                                     long long experimentId,
                                                     const char* phase,
                                                     size_t phaseLength,
                                                     int errorNumber)
{
    char buffer[512];
    size_t used = 0;
    const auto append = [&](const char* text, size_t length) {
        for (size_t i = 0; i < length && used < sizeof(buffer); ++i)
            buffer[used++] = text[i];
    };
    const auto appendLiteral = [&](const char* text) {
        size_t length = 0;
        while (text[length] != '\0')
            ++length;
        append(text, length);
    };
    const auto appendNumber = [&](long long value) {
        char digits[32];
        size_t count = 0;
        unsigned long long magnitude = value < 0
            ? static_cast<unsigned long long>(-(value + 1)) + 1ULL
            : static_cast<unsigned long long>(value);
        if (value < 0 && used < sizeof(buffer))
            buffer[used++] = '-';
        do
        {
            digits[count++] = static_cast<char>('0' + magnitude % 10ULL);
            magnitude /= 10ULL;
        } while (magnitude != 0 && count < sizeof(digits));
        while (count > 0 && used < sizeof(buffer))
            buffer[used++] = digits[--count];
    };

    appendLiteral("SCHEDULER_CHILD_EXEC_FAILED,experiment_id=");
    appendNumber(experimentId);
    appendLiteral(",phase=");
    append(phase, phaseLength);
    appendLiteral(",errno=");
    appendNumber(errorNumber);
    appendLiteral(",error=");
    switch (errorNumber)
    {
        case ENOENT: appendLiteral("no_such_file_or_directory"); break;
        case EACCES: appendLiteral("permission_denied"); break;
        case ENOEXEC: appendLiteral("exec_format_error"); break;
        default: appendLiteral("exec_failed"); break;
    }
    appendLiteral("\n");

    size_t written = 0;
    while (written < used)
    {
        const ssize_t rc = ::write(fd, buffer + written, used - written);
        if (rc > 0)
            written += static_cast<size_t>(rc);
        else if (rc < 0 && errno == EINTR)
            continue;
        else
            break;
    }
}

inline ObservedChildStatus ObserveChildStatusNonBlocking(pid_t pid)
{
    ObservedChildStatus observed;
    pid_t waitResult;
    do
    {
        waitResult = ::waitpid(pid, &observed.rawStatus, WNOHANG);
    } while (waitResult < 0 && errno == EINTR);

    if (waitResult == 0)
    {
        observed.kind = ChildStatusKind::Running;
        return observed;
    }
    if (waitResult < 0)
    {
        observed.kind = ChildStatusKind::WaitError;
        observed.errorNumber = errno;
        return observed;
    }
    if (WIFEXITED(observed.rawStatus))
    {
        observed.kind = ChildStatusKind::Exited;
        observed.exitCode = WEXITSTATUS(observed.rawStatus);
        return observed;
    }
    if (WIFSIGNALED(observed.rawStatus))
    {
        observed.kind = ChildStatusKind::Signaled;
        observed.signalNumber = WTERMSIG(observed.rawStatus);
        observed.exitCode = -observed.signalNumber;
#ifdef WCOREDUMP
        observed.coreDumped = WCOREDUMP(observed.rawStatus);
#endif
        return observed;
    }
    observed.kind = ChildStatusKind::Unexpected;
    return observed;
}

} // namespace EA::ExperimentScheduler
