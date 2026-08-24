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
#include <vector>

#include "../Sources/CensusEconomicReleaseAdapter.hpp"

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

struct Case
{
    const char* fixture;
    const char* url;
    const char* family;
    const char* identity;
    const char* reference;
    const char* date;
    const char* time;
    std::int64_t utcMicros;
};

} // namespace


int main(int argc, const char* argv[])
{
    assert(argc == 2);
    const std::filesystem::path fixtures{argv[1]};

    assert(NormalizeCensusReleaseId("CB10-84") == "census:cb10-84");
    assert(NormalizeCensusReleaseId(" CB 24–040 ") == "census:cb24-40");
    ExpectFailure(
        [] { (void)NormalizeCensusReleaseId("Census 24-40"); },
        "release_id_malformed");

    const std::vector<Case> cases{
        {"2010-retail-sales-advance.txt",
         "https://www2.census.gov/retail/releases/historical/marts/adv1005.pdf",
         "RETAIL_SALES_ADVANCE", "census:cb10-84", "2010-05",
         "2010-06-11", "08:30:00", UtcMicros(2010, 6, 11, 12, 30)},
        {"2024-retail-sales-advance.txt",
         "https://www2.census.gov/retail/releases/historical/marts/adv2402.pdf",
         "RETAIL_SALES_ADVANCE", "census:cb24-40", "2024-02",
         "2024-03-14", "08:30:00", UtcMicros(2024, 3, 14, 12, 30)},
        {"2010-new-residential-construction.txt",
         "https://www.census.gov/construction/nrc/pdf/newresconst_201005.pdf",
         "NEW_RESIDENTIAL_CONSTRUCTION", "census:cb10-89", "2010-05",
         "2010-06-16", "08:30:00", UtcMicros(2010, 6, 16, 12, 30)},
        {"2024-new-residential-construction.txt",
         "https://www.census.gov/construction/nrc/pdf/newresconst_202404.pdf",
         "NEW_RESIDENTIAL_CONSTRUCTION", "census:cb24-78", "2024-04",
         "2024-05-16", "08:30:00", UtcMicros(2024, 5, 16, 12, 30)},
        {"2010-new-residential-sales.txt",
         "https://www.census.gov/construction/nrs/pdf/newressales_201005.pdf",
         "NEW_RESIDENTIAL_SALES", "census:cb10-92", "2010-05",
         "2010-06-23", "10:00:00", UtcMicros(2010, 6, 23, 14, 0)},
        {"2024-new-residential-sales.txt",
         "https://www.census.gov/construction/nrs/pdf/newressales_202412.pdf",
         "NEW_RESIDENTIAL_SALES", "census:cb25-13", "2024-12",
         "2025-01-27", "10:00:00", UtcMicros(2025, 1, 27, 15, 0)},
        {"2010-manufacturers-orders.txt",
         "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2010/may10prel.pdf",
         "MANUFACTURERS_ORDERS", "census:cb10-100", "2010-05",
         "2010-07-02", "10:00:00", UtcMicros(2010, 7, 2, 14, 0)},
        {"2024-manufacturers-orders.txt",
         "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2024/feb24prel.pdf",
         "MANUFACTURERS_ORDERS", "census:cb24-51", "2024-02",
         "2024-04-02", "10:00:00", UtcMicros(2024, 4, 2, 14, 0)},
        {"2010-durable-goods-advance.txt",
         "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/adv/2010/may10adv.pdf",
         "DURABLE_GOODS_ADVANCE", "census:cb10-94", "2010-05",
         "2010-06-24", "08:30:00", UtcMicros(2010, 6, 24, 12, 30)},
        {"2024-durable-goods-advance.txt",
         "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/adv/2024/feb24adv.pdf",
         "DURABLE_GOODS_ADVANCE", "census:cb24-50", "2024-02",
         "2024-03-26", "08:30:00", UtcMicros(2024, 3, 26, 12, 30)},
        {"2010-construction-spending.txt",
         "https://www.census.gov/construction/c30/pdf/pr201005.pdf",
         "CONSTRUCTION_SPENDING", "census:cb10-99", "2010-05",
         "2010-07-01", "10:00:00", UtcMicros(2010, 7, 1, 14, 0)},
        {"2024-construction-spending.txt",
         "https://www.census.gov/construction/c30/pdf/pr202402.pdf",
         "CONSTRUCTION_SPENDING", "census:cb24-55", "2024-02",
         "2024-04-01", "10:00:00", UtcMicros(2024, 4, 1, 14, 0)}};

    for (const auto& test : cases)
    {
        const std::string artifact = Read(fixtures / test.fixture);
        const auto first = ParseCensusEconomicReleaseArtifact(artifact, test.url);
        const auto second = ParseCensusEconomicReleaseArtifact(artifact, test.url);
        assert(first.currency == "USD");
        assert(first.sourceAgency == "CENSUS");
        assert(first.eventFamily == test.family);
        assert(first.sourceEventId == test.identity);
        assert(first.sourceUrl == test.url);
        assert(first.referencePeriod == std::optional<std::string>{test.reference});
        assert(first.sourceReleaseDate == std::optional<std::string>{test.date});
        assert(first.sourceReleaseTime == std::optional<std::string>{test.time});
        assert(first.sourceTimezone == std::optional<std::string>{"America/New_York"});
        assert(first.historicalTimeConfidence == "exact");
        assert(first.eventImportance == 3);
        assert(first.eventTimestampUnixMicros == test.utcMicros);
        assert(second.sourceEventId == first.sourceEventId);
        assert(second.eventFamily == first.eventFamily);
        assert(second.eventTimestampUnixMicros == first.eventTimestampUnixMicros);
    }

    const std::string retail = Read(fixtures / "2010-retail-sales-advance.txt");
    const std::string retailUrl =
        "https://www2.census.gov/retail/releases/historical/marts/adv1005.pdf";

    auto missingTime = std::regex_replace(
        retail,
        std::regex{R"(FOR IMMEDIATE RELEASE\s+FRIDAY, JUNE 11, 2010, AT 8:30 A\.M\. EDT\s*)"},
        "");
    ExpectFailure(
        [&] { (void)ParseCensusEconomicReleaseArtifact(missingTime, retailUrl); },
        "release_time_missing");

    auto contradictoryZone = retail;
    contradictoryZone.replace(contradictoryZone.find("EDT"), 3, "EST");
    ExpectFailure(
        [&] { (void)ParseCensusEconomicReleaseArtifact(contradictoryZone, retailUrl); },
        "timezone_contradiction");

    auto missingIdentity = retail;
    missingIdentity.replace(missingIdentity.find("CB10-84"), 7, "");
    ExpectFailure(
        [&] { (void)ParseCensusEconomicReleaseArtifact(missingIdentity, retailUrl); },
        "identity_missing");

    auto malformedIdentity = retail;
    malformedIdentity.replace(malformedIdentity.find("CB10-84"), 7, "CB BROKEN");
    ExpectFailure(
        [&] { (void)ParseCensusEconomicReleaseArtifact(malformedIdentity, retailUrl); },
        "release_id_malformed");

    const std::string ambiguous = retail +
        "\nNEW RESIDENTIAL SALES IN MAY 2010\n";
    ExpectFailure(
        [&] { (void)ParseCensusEconomicReleaseArtifact(ambiguous, retailUrl); },
        "family_ambiguous");

    ExpectFailure(
        [&] {
            (void)ParseCensusEconomicReleaseArtifact(
                retail,
                "https://example.com/retail/releases/historical/marts/adv1005.pdf");
        },
        "source_url_not_canonical_first_party");

    ExpectFailure(
        [&] {
            (void)ParseCensusEconomicReleaseArtifact(
                retail,
                "https://www2.census.gov/retail/releases/historical/marts/adv1006.pdf");
        },
        "reference_period_mismatch");

    ExpectFailure(
        [&] {
            (void)ParseCensusEconomicReleaseArtifact(
                retail,
                "https://www.census.gov/foreign-trade/Press-Release/current_press_release/ft900.pdf");
        },
        "source_url_not_canonical_first_party");

    const auto manifestCandidates =
        LoadCensusEconomicReleaseManifest(fixtures / "manifest.tsv");
    assert(manifestCandidates.size() == cases.size());
    for (const auto& test : cases)
    {
        assert(std::count_if(
            manifestCandidates.begin(), manifestCandidates.end(),
            [&](const auto& candidate)
            {
                return candidate.sourceEventId == test.identity;
            }) == 1);
    }

    ExpectFailure(
        [&] {
            (void)LoadCensusEconomicReleaseManifest(
                fixtures / "manifest_bad_hash.tsv");
        },
        "sha256_mismatch");
    ExpectFailure(
        [&] {
            (void)LoadCensusEconomicReleaseManifest(
                fixtures / "manifest_malformed.tsv");
        },
        "parser_version_invalid");
    ExpectFailure(
        [&] {
            (void)LoadCensusEconomicReleaseManifest(
                fixtures / "manifest_unsupported_type.tsv");
        },
        "artifact_type_unsupported");
    return 0;
}
