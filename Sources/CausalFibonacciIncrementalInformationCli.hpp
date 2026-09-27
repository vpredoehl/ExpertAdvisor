#pragma once

// Deliberately narrow operational boundary for the frozen research harness.
// It never extracts prices, opens PostgreSQL, starts a worker, or computes a
// historical result.  Extraction and confirmation remain separately
// authorized, read-only operations to be added only after their artifacts are
// independently qualified.

#include "CausalFibonacciIncrementalInformation.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace EA::CausalFibonacciIncrementalInformation::Cli {
inline std::optional<int> TryRun(int argc, const char* const argv[])
{
    if (argc < 2 || std::string_view(argv[1]) != "--fibonacci-incremental-information-verify-artifact")
        return std::nullopt;
    try {
        if (argc != 3) throw std::invalid_argument("usage: --fibonacci-incremental-information-verify-artifact ARTIFACT_DIRECTORY");
        VerifyFrozenProtocolDocument("docs/phases/target-generation/FibonacciExtensions/FIBONACCI_LAYOUT9_INCREMENTAL_INFORMATION_PROTOCOL.md");
        VerifyArtifactDirectory(std::filesystem::path(argv[2]));
        std::cout << "FIBONACCI_INCREMENTAL_ARTIFACT_VERIFIED protocol=" << kProtocolId << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FIBONACCI_INCREMENTAL_ARTIFACT_VERIFICATION_ERROR " << error.what() << '\n';
        return 1;
    }
}
} // namespace EA::CausalFibonacciIncrementalInformation::Cli
