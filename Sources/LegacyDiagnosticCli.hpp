#pragma once

#include <optional>

namespace EA::LegacyDiagnosticCli
{
std::optional<int> TryRun(int argc, const char* argv[]);
}
