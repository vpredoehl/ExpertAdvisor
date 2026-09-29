#pragma once

#include "PocketConfirmationFreezeCli.hpp"
#include "PocketConfirmationExecutionCli.hpp"
#include "PocketConfirmationDerivedCli.hpp"
#include "PocketProspectiveDerivedAnalyzerCli.hpp"
#include "PocketProspectiveEvaluatorCli.hpp"

#include <iostream>
#include <string_view>

namespace EA::Pocket::Research
{
// Dedicated executable boundary. All evaluator behavior remains in the shared
// prospective CLI; this target owns no scientific, source, or artifact logic.
inline int RunCli(int argc, const char* const argv[])
{
    const auto execution = Prospective::Confirmation::Execution::Cli::TryRun(argc, argv);
    if (execution.has_value()) return *execution;
    const auto confirmationDerived = Prospective::Confirmation::DerivedReport::Cli::TryRun(argc, argv);
    if (confirmationDerived.has_value()) return *confirmationDerived;
    const auto confirmation = Prospective::Confirmation::Cli::TryRun(argc, argv);
    if (confirmation.has_value()) return *confirmation;
    const auto derived = Prospective::Derived::Cli::TryRun(argc, argv);
    if (derived.has_value()) return *derived;
    const auto result = Prospective::Cli::TryRun(argc, argv);
    if (result.has_value()) return *result;
    std::cerr << "POCKET_RESEARCH_ERROR expected --pocket-prospective-evaluator, "
              << "--pocket-confirmation-evaluator, --verify-pocket-confirmation-artifact, "
              << "--derive-pocket-confirmation-report, or --verify-pocket-confirmation-derived-report\n";
    return 2;
}
} // namespace EA::Pocket::Research
