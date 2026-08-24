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

#include "../Sources/EconomicEventImportValidation.hpp"
#include "../Sources/FederalReserveEconomicReleaseAdapter.hpp"

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
        if (std::string{error.what()}.find(expected) == std::string::npos)
        {
            throw std::runtime_error(
                "expected diagnostic containing '" + expected +
                "', got '" + error.what() + "'");
        }
    }
}

} // namespace


int main(int argc, const char* argv[])
{
    assert(argc == 2);
    const std::filesystem::path fixtures{argv[1]};

    const std::string historicalStatement =
        Read(fixtures / "2010-fomc-statement.txt");
    const auto historicalStatementCandidate =
        ParseFederalReserveEconomicReleaseArtifact(
            historicalStatement,
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100810a.htm",
            "fomc_statement");
    assert(historicalStatementCandidate.currency == "USD");
    assert(historicalStatementCandidate.sourceAgency == "FEDERAL_RESERVE");
    assert(historicalStatementCandidate.eventFamily == "FOMC_STATEMENT");
    assert(historicalStatementCandidate.sourceEventId ==
           "federal_reserve:monetary20100810a");
    assert(historicalStatementCandidate.referencePeriod ==
           std::optional<std::string>{"meeting ending 2010-08-10"});
    assert(historicalStatementCandidate.sourceReleaseDate ==
           std::optional<std::string>{"2010-08-10"});
    assert(!historicalStatementCandidate.sourceReleaseTime);
    assert(historicalStatementCandidate.sourceTimezone ==
           std::optional<std::string>{"America/New_York"});
    assert(historicalStatementCandidate.historicalTimeConfidence == "date_only");
    assert(historicalStatementCandidate.eventImportance == 3);
    // Conservative shared date-only boundary: start of the following New York day.
    assert(historicalStatementCandidate.eventTimestampUnixMicros ==
           UtcMicros(2010, 8, 11, 4, 0));

    const std::string modernStatement =
        Read(fixtures / "2024-fomc-statement.txt");
    const auto modernStatementCandidate =
        ParseFederalReserveEconomicReleaseArtifact(
            modernStatement,
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
            "fomc_statement");
    assert(modernStatementCandidate.eventFamily == "FOMC_STATEMENT");
    assert(modernStatementCandidate.sourceEventId ==
           "federal_reserve:monetary20240612a");
    assert(modernStatementCandidate.sourceReleaseTime ==
           std::optional<std::string>{"14:00:00"});
    assert(modernStatementCandidate.historicalTimeConfidence == "exact");
    assert(modernStatementCandidate.eventTimestampUnixMicros ==
           UtcMicros(2024, 6, 12, 18, 0));

    const auto historicalMinutesCandidate =
        ParseFederalReserveEconomicReleaseArtifact(
            Read(fixtures / "2010-fomc-minutes.txt"),
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20100831a.htm",
            "fomc_minutes");
    assert(historicalMinutesCandidate.eventFamily == "FOMC_MINUTES");
    assert(historicalMinutesCandidate.sourceEventId ==
           "federal_reserve:monetary20100831a");
    assert(historicalMinutesCandidate.referencePeriod ==
           std::optional<std::string>{"meeting 2010-08-10"});
    assert(historicalMinutesCandidate.sourceReleaseTime ==
           std::optional<std::string>{"14:00:00"});
    assert(historicalMinutesCandidate.historicalTimeConfidence == "exact");
    assert(historicalMinutesCandidate.eventTimestampUnixMicros ==
           UtcMicros(2010, 8, 31, 18, 0));

    const auto modernMinutesCandidate =
        ParseFederalReserveEconomicReleaseArtifact(
            Read(fixtures / "2024-fomc-minutes.txt"),
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240703a.htm",
            "fomc_minutes");
    assert(modernMinutesCandidate.referencePeriod ==
           std::optional<std::string>{"meeting 2024-06-11/2024-06-12"});
    assert(modernMinutesCandidate.eventTimestampUnixMicros ==
           UtcMicros(2024, 7, 3, 18, 0));

    const auto historicalBeigeBookCandidate =
        ParseFederalReserveEconomicReleaseArtifact(
            Read(fixtures / "2010-beige-book.txt"),
            "https://www.federalreserve.gov/fomc/beigebook/2010/20100113/fullreport20100113.pdf",
            "beige_book");
    assert(historicalBeigeBookCandidate.eventFamily == "BEIGE_BOOK");
    assert(historicalBeigeBookCandidate.sourceEventId ==
           "federal_reserve:beigebook-20100113");
    assert(historicalBeigeBookCandidate.referencePeriod ==
           std::optional<std::string>{"publication 2010-01-13"});
    assert(historicalBeigeBookCandidate.eventImportance == 2);
    assert(historicalBeigeBookCandidate.historicalTimeConfidence == "exact");
    assert(historicalBeigeBookCandidate.eventTimestampUnixMicros ==
           UtcMicros(2010, 1, 13, 19, 0));

    const auto modernBeigeBookCandidate =
        ParseFederalReserveEconomicReleaseArtifact(
            Read(fixtures / "2024-beige-book.txt"),
            "https://www.federalreserve.gov/monetarypolicy/files/BeigeBook_20240417.pdf",
            "beige_book");
    assert(modernBeigeBookCandidate.sourceEventId ==
           "federal_reserve:beigebook-20240417");
    assert(modernBeigeBookCandidate.eventTimestampUnixMicros ==
           UtcMicros(2024, 4, 17, 18, 0));

    const auto manifestCandidates =
        LoadFederalReserveEconomicReleaseManifest(fixtures / "manifest.tsv");
    assert(manifestCandidates.size() == 6);
    assert(ValidateAndOrderEconomicEventCandidates(manifestCandidates).size() == 6);
    for (const std::string family :
         {"FOMC_STATEMENT", "FOMC_MINUTES", "BEIGE_BOOK"})
    {
        assert(std::count_if(
            manifestCandidates.begin(), manifestCandidates.end(),
            [&](const auto& candidate)
            {
                return candidate.eventFamily == family;
            }) == 2);
    }
    assert(std::none_of(
        manifestCandidates.begin(), manifestCandidates.end(),
        [](const auto& candidate)
        {
            return candidate.historicalTimeConfidence == "reconstructed";
        }));

    auto missingTime = std::regex_replace(
        modernStatement,
        std::regex{R"(For release at 2:00 p\.m\. EDT)"},
        "");
    ExpectFailure(
        [&] {
            (void)ParseFederalReserveEconomicReleaseArtifact(
                missingTime,
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
                "fomc_statement");
        },
        "release_time_missing");

    auto unsupportedZone = modernStatement;
    unsupportedZone.replace(unsupportedZone.find("EDT"), 3, "CDT");
    ExpectFailure(
        [&] {
            (void)ParseFederalReserveEconomicReleaseArtifact(
                unsupportedZone,
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
                "fomc_statement");
        },
        "timezone_unsupported");

    auto contradictoryZone = modernStatement;
    contradictoryZone.replace(contradictoryZone.find("EDT"), 3, "EST");
    ExpectFailure(
        [&] {
            (void)ParseFederalReserveEconomicReleaseArtifact(
                contradictoryZone,
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
                "fomc_statement");
        },
        "timezone_contradiction");

    ExpectFailure(
        [&] {
            (void)ParseFederalReserveEconomicReleaseArtifact(
                modernStatement,
                "https://example.com/newsevents/pressreleases/monetary20240612a.htm",
                "fomc_statement");
        },
        "source_url_not_canonical_first_party");
    ExpectFailure(
        [&] {
            (void)ParseFederalReserveEconomicReleaseArtifact(
                modernStatement,
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240613a.htm",
                "fomc_statement");
        },
        "source_url_release_date_mismatch");
    ExpectFailure(
        [&] {
            (void)ParseFederalReserveEconomicReleaseArtifact(
                modernStatement,
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
                "fomc_minutes");
        },
        "classification_mismatch");

    const std::string ambiguous = modernStatement +
        "\nBeige Book - June 12, 2024\nFor use at 2:00 p.m. EDT\n";
    ExpectFailure(
        [&] {
            (void)ParseFederalReserveEconomicReleaseArtifact(
                ambiguous,
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
                "fomc_statement");
        },
        "classification_ambiguous");

    ExpectFailure(
        [&] {
            (void)LoadFederalReserveEconomicReleaseManifest(
                fixtures / "manifest_bad_hash.tsv");
        },
        "sha256_mismatch");
    ExpectFailure(
        [&] {
            (void)LoadFederalReserveEconomicReleaseManifest(
                fixtures / "manifest_missing_artifact.tsv");
        },
        "artifact_missing");
    ExpectFailure(
        [&] {
            (void)LoadFederalReserveEconomicReleaseManifest(
                fixtures / "manifest_malformed.tsv");
        },
        "parser_version_invalid");
    ExpectFailure(
        [&] {
            (void)LoadFederalReserveEconomicReleaseManifest(
                fixtures / "manifest_unsupported_type.tsv");
        },
        "artifact_type_unsupported");
    ExpectFailure(
        [&] {
            (void)LoadFederalReserveEconomicReleaseManifest(
                fixtures / "missing-manifest.tsv");
        },
        "manifest_missing");
    return 0;
}
