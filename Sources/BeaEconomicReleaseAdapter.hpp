#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{

std::string NormalizeBeaReleaseId(
    const std::string& releaseId);

AuthoritativeEconomicEventCandidate ParseBeaEconomicReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl);

// Loads the Phase 2-compatible deterministic local manifest, verifies every
// declared SHA-256 digest, and emits one publication-level raw event per entry.
std::vector<AuthoritativeEconomicEventCandidate>
LoadBeaEconomicReleaseManifest(
    const std::filesystem::path& manifestPath);

} // namespace EA::EconomicCalendar
