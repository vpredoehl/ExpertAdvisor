#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../Sources/BeaEconomicReleaseAdapter.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string Read(const std::filesystem::path& path)
{
    std::ifstream input{path};
    assert(input);
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

std::int64_t UtcMicros(
    int year, int month, int day, int hour, int minute)
{
    std::tm fields{};
    fields.tm_year = year - 1900;
    fields.tm_mon = month - 1;
    fields.tm_mday = day;
    fields.tm_hour = hour;
    fields.tm_min = minute;
    return static_cast<std::int64_t>(timegm(&fields)) * 1000000;
}

void ExpectFailure(
    const std::function<void()>& operation,
    const std::string& expected)
{
    try
    {
        operation();
        assert(false && "expected failure");
    }
    catch (const std::exception& error)
    {
        assert(std::string{error.what()}.find(expected) != std::string::npos);
    }
}

} // namespace


int main(int argc, const char* argv[])
{
    assert(argc == 2);
    const std::filesystem::path fixtures{argv[1]};

    assert(NormalizeBeaReleaseId("BEA 10-37") == "bea:10-37");
    assert(NormalizeBeaReleaseId(" BEA 2024–061 ") == "bea:24-61");
    ExpectFailure(
        [] { (void)NormalizeBeaReleaseId("BEA July 30"); },
        "release_id_malformed");

    const std::string advance = Read(fixtures / "2010-gdp-advance.txt");
    const auto advanceCandidate = ParseBeaEconomicReleaseArtifact(
        advance,
        "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate");
    assert(advanceCandidate.currency == "USD");
    assert(advanceCandidate.sourceAgency == "BEA");
    assert(advanceCandidate.eventFamily == "GDP_ADVANCE");
    assert(advanceCandidate.sourceEventId == "bea:10-37");
    assert(advanceCandidate.sourceUrl ==
           "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate");
    assert(advanceCandidate.referencePeriod ==
           std::optional<std::string>{"Q2 2010"});
    assert(advanceCandidate.sourceReleaseDate ==
           std::optional<std::string>{"2010-07-30"});
    assert(advanceCandidate.sourceReleaseTime ==
           std::optional<std::string>{"08:30:00"});
    assert(advanceCandidate.sourceTimezone ==
           std::optional<std::string>{"America/New_York"});
    assert(advanceCandidate.historicalTimeConfidence == "exact");
    assert(advanceCandidate.eventImportance == 3);
    assert(advanceCandidate.eventTimestampUnixMicros ==
           UtcMicros(2010, 7, 30, 12, 30));

    const auto secondCandidate = ParseBeaEconomicReleaseArtifact(
        Read(fixtures / "2010-gdp-second.txt"),
        "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-second-estimate-corporate-profits-2nd-quarter");
    assert(secondCandidate.eventFamily == "GDP_SECOND");
    assert(secondCandidate.sourceEventId == "bea:10-41");
    assert(secondCandidate.referencePeriod ==
           std::optional<std::string>{"Q2 2010"});

    const auto thirdCandidate = ParseBeaEconomicReleaseArtifact(
        Read(fixtures / "2010-gdp-third.txt"),
        "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-third-estimate-corporate-profits-2nd-quarter-2010");
    assert(thirdCandidate.eventFamily == "GDP_THIRD");
    assert(thirdCandidate.sourceEventId == "bea:10-47");

    const auto personalCandidate = ParseBeaEconomicReleaseArtifact(
        Read(fixtures / "2010-personal-income-outlays.txt"),
        "https://www.bea.gov/news/2010/personal-income-and-outlays-june-2010");
    assert(personalCandidate.eventFamily == "PERSONAL_INCOME_OUTLAYS");
    assert(personalCandidate.sourceEventId == "bea:10-38");
    assert(personalCandidate.referencePeriod ==
           std::optional<std::string>{"2010-06"});
    assert(personalCandidate.eventTimestampUnixMicros ==
           UtcMicros(2010, 8, 3, 12, 30));

    const auto modernGdp = ParseBeaEconomicReleaseArtifact(
        Read(fixtures / "2024-gdp-third.txt"),
        "https://www.bea.gov/news/2024/gross-domestic-product-third-estimate-corporate-profits-revised-estimate-and-gdp-1");
    assert(modernGdp.eventFamily == "GDP_THIRD");
    assert(modernGdp.sourceEventId == "bea:24-61");
    assert(modernGdp.referencePeriod ==
           std::optional<std::string>{"Q3 2024"});
    assert(modernGdp.eventTimestampUnixMicros ==
           UtcMicros(2024, 12, 19, 13, 30));

    const auto modernPersonal = ParseBeaEconomicReleaseArtifact(
        Read(fixtures / "2025-personal-income-outlays.txt"),
        "https://www.bea.gov/news/2025/personal-income-and-outlays-july-2025");
    assert(modernPersonal.eventFamily == "PERSONAL_INCOME_OUTLAYS");
    assert(modernPersonal.referencePeriod ==
           std::optional<std::string>{"2025-07"});

    auto missingTime = std::regex_replace(
        advance,
        std::regex{R"(EMBARGOED UNTIL RELEASE AT 8:30 A\.M\. EDT, )"},
        "");
    ExpectFailure(
        [&] {
            (void)ParseBeaEconomicReleaseArtifact(
                missingTime,
                "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate");
        },
        "release_time_missing");

    auto unsupportedZone = advance;
    unsupportedZone.replace(unsupportedZone.find("EDT"), 3, "CDT");
    ExpectFailure(
        [&] {
            (void)ParseBeaEconomicReleaseArtifact(
                unsupportedZone,
                "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate");
        },
        "timezone_unsupported");

    auto contradictoryZone = advance;
    contradictoryZone.replace(contradictoryZone.find("EDT"), 3, "EST");
    ExpectFailure(
        [&] {
            (void)ParseBeaEconomicReleaseArtifact(
                contradictoryZone,
                "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate");
        },
        "timezone_contradiction");

    auto malformedIdentity = advance;
    malformedIdentity.replace(malformedIdentity.find("BEA 10-37"), 9, "BEA BROKE");
    ExpectFailure(
        [&] {
            (void)ParseBeaEconomicReleaseArtifact(
                malformedIdentity,
                "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate");
        },
        "release_id_malformed");

    auto ambiguousVintage = advance;
    const auto vintagePosition = ambiguousVintage.find("(advance estimate)");
    ambiguousVintage.replace(
        vintagePosition,
        std::string{"(advance estimate)"}.size(),
        "(advance estimate; second estimate)");
    ExpectFailure(
        [&] {
            (void)ParseBeaEconomicReleaseArtifact(
                ambiguousVintage,
                "https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate");
        },
        "vintage_ambiguous");

    ExpectFailure(
        [&] {
            (void)ParseBeaEconomicReleaseArtifact(
                advance,
                "https://example.com/news/2010/gdp");
        },
        "source_url_not_canonical_first_party");

    const auto manifestCandidates =
        LoadBeaEconomicReleaseManifest(fixtures / "manifest.tsv");
    assert(manifestCandidates.size() == 6);
    assert(std::count_if(
        manifestCandidates.begin(), manifestCandidates.end(),
        [](const auto& candidate)
        {
            return candidate.eventFamily == "PERSONAL_INCOME_OUTLAYS";
        }) == 2);
    // Each PIO publication remains one row even though its artifact names
    // income, outlays, PCE, disposable income, and price-related statistics.
    ExpectFailure(
        [&] {
            (void)LoadBeaEconomicReleaseManifest(
                fixtures / "manifest_bad_hash.tsv");
        },
        "sha256_mismatch");
    ExpectFailure(
        [&] {
            (void)LoadBeaEconomicReleaseManifest(
                fixtures / "manifest_malformed.tsv");
        },
        "parser_version_invalid");
    ExpectFailure(
        [&] {
            (void)LoadBeaEconomicReleaseManifest(
                fixtures / "manifest_unsupported_type.tsv");
        },
        "artifact_type_unsupported");
    return 0;
}
