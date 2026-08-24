#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Sources/BeaEconomicReleaseAdapter.hpp"
#include "../Sources/CensusEconomicReleaseAdapter.hpp"
#include "../Sources/DolEtaWeeklyClaimsAdapter.hpp"
#include "../Sources/EconomicEventImportValidation.hpp"
#include "../Sources/FederalReserveEconomicReleaseAdapter.hpp"

using EA::EconomicCalendar::AuthoritativeEconomicEventCandidate;

namespace
{

std::string Value(const std::optional<std::string>& value)
{
    return value.value_or("");
}

std::vector<AuthoritativeEconomicEventCandidate> Load(
    const std::string& agency,
    const std::filesystem::path& manifest)
{
    using namespace EA::EconomicCalendar;
    if (agency == "dol-eta")
        return LoadDolEtaWeeklyClaimsManifest(manifest);
    if (agency == "bea")
        return LoadBeaEconomicReleaseManifest(manifest);
    if (agency == "census")
        return LoadCensusEconomicReleaseManifest(manifest);
    if (agency == "federal-reserve")
        return LoadFederalReserveEconomicReleaseManifest(manifest);
    throw std::invalid_argument("unsupported audit agency: " + agency);
}

} // namespace


int main(int argc, const char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "usage: EconomicEventManifestAudit AGENCY MANIFEST\n";
        return 64;
    }

    std::vector<AuthoritativeEconomicEventCandidate> candidates;
    try
    {
        candidates = Load(argv[1], argv[2]);
    }
    catch (const std::exception& error)
    {
        std::cerr << "ECONOMIC_EVENT_MANIFEST_LOAD_FAILED"
                  << ",agency=" << argv[1]
                  << ",error=" << error.what() << '\n';
        return 2;
    }

    std::cout << "source_agency\traw_event_family\tsource_event_id\t"
                 "source_url\tevent_timestamp_unix_micros\t"
                 "timestamp_confidence\tsource_release_date\t"
                 "source_release_time\tsource_timezone\treference_period\n";
    for (const auto& candidate : candidates)
    {
        std::cout << candidate.sourceAgency << '\t'
                  << candidate.eventFamily << '\t'
                  << candidate.sourceEventId << '\t'
                  << candidate.sourceUrl << '\t'
                  << candidate.eventTimestampUnixMicros << '\t'
                  << candidate.historicalTimeConfidence << '\t'
                  << Value(candidate.sourceReleaseDate) << '\t'
                  << Value(candidate.sourceReleaseTime) << '\t'
                  << Value(candidate.sourceTimezone) << '\t'
                  << Value(candidate.referencePeriod) << '\n';
    }

    try
    {
        const auto validated =
            EA::EconomicCalendar::ValidateAndOrderEconomicEventCandidates(
                candidates);
        std::cerr << "ECONOMIC_EVENT_MANIFEST_AUDIT_COMPLETE"
                  << ",agency=" << argv[1]
                  << ",candidates=" << validated.size() << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ECONOMIC_EVENT_MANIFEST_VALIDATION_FAILED"
                  << ",agency=" << argv[1]
                  << ",error=" << error.what() << '\n';
        return 3;
    }
}
