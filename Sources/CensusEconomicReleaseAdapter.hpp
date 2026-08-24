#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{

// Normalizes a Census Bureau newsroom release number such as CB24-40 into
// the shared agency-prefixed identity contract.
std::string NormalizeCensusReleaseId(
    const std::string& releaseId);

AuthoritativeEconomicEventCandidate ParseCensusEconomicReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl);

// Loads the shared deterministic local manifest, verifies every declared
// SHA-256 digest, and emits one publication-level raw event per entry.
std::vector<AuthoritativeEconomicEventCandidate>
LoadCensusEconomicReleaseManifest(
    const std::filesystem::path& manifestPath);

} // namespace EA::EconomicCalendar
