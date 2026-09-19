#include "LstmRuntimeLogging.hpp"

#include <atomic>

namespace
{

std::atomic<EA::RuntimeLogLevel> gRuntimeLogLevel{
    EA::RuntimeLogLevel::Summary};

} // namespace

namespace EA
{

RuntimeLogLevel RuntimeLogLevelValue() noexcept
{
    return gRuntimeLogLevel.load(std::memory_order_relaxed);
}

void SetRuntimeLogLevel(RuntimeLogLevel level) noexcept
{
    gRuntimeLogLevel.store(level, std::memory_order_relaxed);
}

bool RuntimeSummaryLoggingEnabled() noexcept
{
    return static_cast<int>(RuntimeLogLevelValue()) >=
        static_cast<int>(RuntimeLogLevel::Summary);
}

bool RuntimeDiagnosticLoggingEnabled() noexcept
{
    return RuntimeLogLevelValue() == RuntimeLogLevel::Diagnostic;
}

} // namespace EA

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled()
{
    return EA::RuntimeDiagnosticLoggingEnabled();
}
