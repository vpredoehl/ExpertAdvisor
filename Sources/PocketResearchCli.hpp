#pragma once

#include "PocketProspectiveEvaluatorCli.hpp"

#include <iostream>
#include <string_view>

namespace EA::Pocket::Research
{
// Dedicated executable boundary. All evaluator behavior remains in the shared
// prospective CLI; this target owns no scientific, source, or artifact logic.
inline int RunCli(int argc, const char* const argv[])
{
    const auto result = Prospective::Cli::TryRun(argc, argv);
    if (result.has_value()) return *result;
    std::cerr << "POCKET_RESEARCH_ERROR expected --pocket-prospective-evaluator "
              << "or --verify-pocket-prospective-artifact\n";
    return 2;
}
} // namespace EA::Pocket::Research
