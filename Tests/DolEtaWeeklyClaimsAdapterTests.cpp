#include <cassert>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../Sources/DolEtaWeeklyClaimsAdapter.hpp"

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

    assert(NormalizeDolEtaReleaseId(" USDL 10-990-NAT ") ==
           "dol_eta:usdl-10-990-nat");
    assert(NormalizeDolEtaReleaseId("usdl.10 / 990 _ NaT") ==
           "dol_eta:usdl-10-990-nat");
    assert(NormalizeDolEtaReleaseId("USDL10-990-NAT") ==
           "dol_eta:usdl-10-990-nat");
    assert(NormalizeDolEtaReleaseId("USDL 24-24-2399-NAT") ==
           "dol_eta:usdl-24-24-2399-nat");
    ExpectFailure(
        [] { (void)NormalizeDolEtaReleaseId("USDL 10 NAT"); },
        "release_id_malformed");

    const std::string legacy = Read(fixtures / "2010-07-22.html");
    const auto candidate = ParseDolEtaWeeklyClaimsArtifact(
        legacy,
        "https://oui.doleta.gov/press/2010/072210.asp");
    assert(candidate.currency == "USD");
    assert(candidate.sourceAgency == "DOL_ETA");
    assert(candidate.eventFamily == "WEEKLY_CLAIMS");
    assert(candidate.sourceEventId == "dol_eta:usdl-10-990-nat");
    assert(candidate.sourceUrl ==
           "https://oui.doleta.gov/press/2010/072210.asp");
    assert(candidate.referencePeriod ==
           std::optional<std::string>{"week ending 2010-07-17"});
    assert(candidate.sourceReleaseDate ==
           std::optional<std::string>{"2010-07-22"});
    assert(candidate.sourceReleaseTime ==
           std::optional<std::string>{"08:30:00"});
    assert(candidate.sourceTimezone ==
           std::optional<std::string>{"America/New_York"});
    assert(candidate.historicalTimeConfidence == "exact");
    assert(candidate.eventImportance == 3);
    assert(candidate.eventTimestampUnixMicros ==
           UtcMicros(2010, 7, 22, 12, 30));

    auto separatedEmbargoFields = legacy;
    const std::string compactEmbargo =
        "8:30 A.M. (EDT), THURSDAY July 22, 2010";
    const auto compactEmbargoPosition = separatedEmbargoFields.find(compactEmbargo);
    assert(compactEmbargoPosition != std::string::npos);
    separatedEmbargoFields.replace(
        compactEmbargoPosition,
        compactEmbargo.size(),
        "8:30 A.M. (EDT), THURSDAY Media Contact: July 22, 2010");
    assert(ParseDolEtaWeeklyClaimsArtifact(
               separatedEmbargoFields,
               "https://oui.doleta.gov/press/2010/072210.asp")
               .sourceEventId == "dol_eta:usdl-10-990-nat");

    const auto reusedReleaseNumber = ParseDolEtaWeeklyClaimsArtifact(
        "USDL 16-567-NAT RELEASE IS EMBARGOED UNTIL 8:30 A.M. (EDT), "
        "THURSDAY March 17, 2016 UNEMPLOYMENT INSURANCE WEEKLY CLAIMS "
        "In the week ending March 12, initial claims were reported.",
        "https://oui.doleta.gov/press/2016/031716.pdf");
    assert(reusedReleaseNumber.sourceEventId ==
           "dol_eta:artifact-2016-031716-pdf");

    const std::string modern = Read(fixtures / "2025-01-02.txt");
    const auto modernCandidate = ParseDolEtaWeeklyClaimsArtifact(
        modern,
        "https://oui.doleta.gov/press/2025/010225.pdf");
    assert(modernCandidate.sourceEventId ==
           "dol_eta:artifact-2025-010225-pdf");
    assert(modernCandidate.referencePeriod ==
           std::optional<std::string>{"week ending 2024-12-28"});
    assert(modernCandidate.eventTimestampUnixMicros ==
           UtcMicros(2025, 1, 2, 13, 30));

    auto missingTime = legacy;
    missingTime = std::regex_replace(
        missingTime,
        std::regex{"8:30 A.M. \\(EDT\\),"},
        "");
    ExpectFailure(
        [&] {
            (void)ParseDolEtaWeeklyClaimsArtifact(
                missingTime,
                "https://oui.doleta.gov/press/2010/072210.asp");
        },
        "release_time_missing");

    auto contradictoryZone = legacy;
    const auto zonePosition = contradictoryZone.find("(EDT)");
    assert(zonePosition != std::string::npos);
    contradictoryZone.replace(zonePosition, 5, "(EST)");
    ExpectFailure(
        [&] {
            (void)ParseDolEtaWeeklyClaimsArtifact(
                contradictoryZone,
                "https://oui.doleta.gov/press/2010/072210.asp");
        },
        "timezone_contradiction");

    const auto wrongWinterEdt2012 = ParseDolEtaWeeklyClaimsArtifact(
        "USDL 12-001-NAT RELEASE IS EMBARGOED UNTIL 8:30 A.M. (EDT), "
        "THURSDAY January 5, 2012 UNEMPLOYMENT INSURANCE WEEKLY CLAIMS "
        "In the week ending December 31, 2011, initial claims were reported.",
        "https://oui.doleta.gov/press/2012/010512.asp");
    assert(wrongWinterEdt2012.historicalTimeConfidence == "reconstructed");
    assert(wrongWinterEdt2012.sourceReleaseTime ==
           std::optional<std::string>{"08:30:00"});
    assert(wrongWinterEdt2012.eventTimestampUnixMicros ==
           UtcMicros(2012, 1, 5, 13, 30));

    const auto wrongSummerEst2012 = ParseDolEtaWeeklyClaimsArtifact(
        "USDL 12-002-NAT RELEASE IS EMBARGOED UNTIL 8:30 A.M. (EST), "
        "THURSDAY June 21, 2012 UNEMPLOYMENT INSURANCE WEEKLY CLAIMS "
        "In the week ending June 16, 2012, initial claims were reported.",
        "https://oui.doleta.gov/press/2012/062112.asp");
    assert(wrongSummerEst2012.historicalTimeConfidence == "reconstructed");
    assert(wrongSummerEst2012.eventTimestampUnixMicros ==
           UtcMicros(2012, 6, 21, 12, 30));

    const auto wrongWinterEdt2011 = ParseDolEtaWeeklyClaimsArtifact(
        "USDL 11-001-NAT RELEASE IS EMBARGOED UNTIL 8:30 A.M. (EDT), "
        "THURSDAY December 29, 2011 UNEMPLOYMENT INSURANCE WEEKLY CLAIMS "
        "In the week ending December 24, 2011, initial claims were reported.",
        "https://oui.doleta.gov/press/2011/122911.asp");
    assert(wrongWinterEdt2011.historicalTimeConfidence == "reconstructed");
    assert(wrongWinterEdt2011.eventTimestampUnixMicros ==
           UtcMicros(2011, 12, 29, 13, 30));

    const auto neutralEastern2012 = ParseDolEtaWeeklyClaimsArtifact(
        "USDL 12-003-NAT RELEASE IS EMBARGOED UNTIL 8:30 A.M. (Eastern), "
        "THURSDAY September 27, 2012 UNEMPLOYMENT INSURANCE WEEKLY CLAIMS "
        "In the week ending September 22, 2012, initial claims were reported.",
        "https://oui.doleta.gov/press/2012/092712.asp");
    assert(neutralEastern2012.historicalTimeConfidence == "exact");
    assert(neutralEastern2012.eventTimestampUnixMicros ==
           UtcMicros(2012, 9, 27, 12, 30));

    auto unsupportedZone = legacy;
    const auto secondZonePosition = unsupportedZone.find("(EDT)");
    unsupportedZone.replace(secondZonePosition, 5, "(CDT)");
    ExpectFailure(
        [&] {
            (void)ParseDolEtaWeeklyClaimsArtifact(
                unsupportedZone,
                "https://oui.doleta.gov/press/2010/072210.asp");
        },
        "timezone_unsupported");

    auto malformedId = legacy;
    const auto idPosition = malformedId.find("USDL 10-990-NAT");
    malformedId.replace(idPosition, 17, "USDL BROKEN");
    ExpectFailure(
        [&] {
            (void)ParseDolEtaWeeklyClaimsArtifact(
                malformedId,
                "https://oui.doleta.gov/press/2010/072210.asp");
        },
        "release_id_malformed");

    const auto manifestCandidates =
        LoadDolEtaWeeklyClaimsManifest(fixtures / "manifest.tsv");
    assert(manifestCandidates.size() == 2);
    assert(manifestCandidates[0].eventFamily == "WEEKLY_CLAIMS");
    assert(manifestCandidates[1].eventFamily == "WEEKLY_CLAIMS");
    return 0;
}
