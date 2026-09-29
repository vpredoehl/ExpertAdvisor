#pragma once

#include "PocketConfirmationFreeze.hpp"
#include "PocketProspectiveDerivedAnalyzer.hpp"

namespace EA::Pocket::Prospective::Confirmation::DerivedReport
{
inline void Analyze(const std::filesystem::path& source, const std::filesystem::path& target,
    std::string_view git, std::string_view executable)
{
    ::EA::Pocket::Prospective::Derived::AnalyzeContract(source,target,git,executable,
        EvaluatorConfiguration(),CanonicalConfigurationText(),kPrimaryArtifactSchema,kDerivedReportSchema);
}
inline void Verify(const std::filesystem::path& directory)
{
    ::EA::Pocket::Prospective::Derived::Writer::VerifyContract(directory,kDerivedReportSchema,
        FrozenConfiguration().configurationSha256);
}
} // namespace EA::Pocket::Prospective::Confirmation::DerivedReport
