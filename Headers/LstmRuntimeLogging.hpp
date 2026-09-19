#pragma once

namespace EA
{

enum class RuntimeLogLevel
{
    Quiet = 0,
    Summary = 1,
    Diagnostic = 2
};

RuntimeLogLevel RuntimeLogLevelValue() noexcept;
void SetRuntimeLogLevel(RuntimeLogLevel level) noexcept;
bool RuntimeSummaryLoggingEnabled() noexcept;
bool RuntimeDiagnosticLoggingEnabled() noexcept;

} // namespace EA

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled();
