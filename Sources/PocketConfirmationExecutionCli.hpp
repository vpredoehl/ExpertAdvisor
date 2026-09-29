#pragma once

#include "PocketConfirmationFreezeCli.hpp"
#include "PocketProspectiveEvaluatorCli.hpp"

namespace EA::Pocket::Prospective::Confirmation::Execution::Cli
{
inline std::optional<int> TryRun(int argc, const char* const argv[])
{
    if (argc < 2 || std::string_view(argv[1]) != "--pocket-confirmation-evaluator") return std::nullopt;
    try {
        const ::EA::Pocket::Prospective::Cli::Options options=::EA::Pocket::Prospective::Cli::Parse(argc,argv);
        if (!options.execute || options.validateOnly)
            throw std::invalid_argument("POCKET_CONFIRMATION_EXECUTE_REQUIRED");
        (void)LoadAndValidateConfiguration(options.configuration);
        return ::EA::Pocket::Prospective::Cli::ExecuteFrozenStudy(EvaluatorConfiguration(),
            CanonicalConfigurationText(), kPrimaryArtifactSchema, options,
            "POCKET_CONFIRMATION");
    } catch (const std::exception& error) {
        std::cerr << "POCKET_CONFIRMATION_ERROR " << error.what() << '\n'; return 1;
    }
}
} // namespace EA::Pocket::Prospective::Confirmation::Execution::Cli
