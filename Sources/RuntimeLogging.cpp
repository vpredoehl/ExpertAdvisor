#include "RuntimeLogging.hpp"
#include <iostream>
#include "LstmRuntimeLogging.hpp"
#include "ModelRuntimeValidation.hpp"
namespace EA::RuntimeLogging
{
bool LogSummary() { return EA::RuntimeSummaryLoggingEnabled(); }
bool LogDiagnostic() { return EA::RuntimeDiagnosticLoggingEnabled(); }
std::ostream& DiagnosticOut()
{
    static ScopedDiagnosticCoutSilencer::NullLogBuffer nullBuffer;
    static std::ostream nullStream(&nullBuffer);
    return LogDiagnostic() ? std::cout : nullStream;
}
ScopedDiagnosticCoutSilencer::ScopedDiagnosticCoutSilencer()
{
    if (!LogDiagnostic()) previousBuffer = std::cout.rdbuf(nullStream.rdbuf());
}
ScopedDiagnosticCoutSilencer::~ScopedDiagnosticCoutSilencer()
{
    if (previousBuffer != nullptr) std::cout.rdbuf(previousBuffer);
}
void InstallModelRuntimeValidationDiagnostics()
{
    EA::ModelRuntimeValidation::SetDiagnostics({ LogSummary, DiagnosticOut });
}
} // namespace EA::RuntimeLogging
