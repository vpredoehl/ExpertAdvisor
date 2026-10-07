#pragma once

#include <ostream>
#include <streambuf>

namespace EA::RuntimeLogging
{
bool LogSummary();
bool LogDiagnostic();
std::ostream& DiagnosticOut();
class ScopedDiagnosticCoutSilencer
{
public:
    class NullLogBuffer : public std::streambuf
    {
    public:
        int overflow(int c) override { return c; }
    };
    ScopedDiagnosticCoutSilencer();
    ~ScopedDiagnosticCoutSilencer();
    ScopedDiagnosticCoutSilencer(const ScopedDiagnosticCoutSilencer&) = delete;
    ScopedDiagnosticCoutSilencer& operator=(const ScopedDiagnosticCoutSilencer&) = delete;
private:
    NullLogBuffer nullBuffer;
    std::ostream nullStream { &nullBuffer };
    std::streambuf* previousBuffer = nullptr;
};
void InstallModelRuntimeValidationDiagnostics();
} // namespace EA::RuntimeLogging
