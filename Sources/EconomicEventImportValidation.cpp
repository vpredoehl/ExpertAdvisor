#include "EconomicEventImportValidation.hpp"

#include "../Headers/HistoricalFxTimestamp.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace EA::EconomicCalendar
{
namespace
{

bool IsBlank(const std::string& value)
{
    return std::all_of(
        value.begin(),
        value.end(),
        [](unsigned char character)
        {
            return std::isspace(character) != 0;
        });
}

std::string IdentityPrefix(const std::string& agency)
{
    std::string result;
    result.reserve(agency.size());

    for (unsigned char character : agency)
    {
        if (std::isalnum(character) != 0)
            result.push_back(static_cast<char>(std::tolower(character)));
        else if (character == '_')
            result.push_back('_');
        else
            throw std::invalid_argument("economic_event_source_agency_invalid");
    }

    return result;
}

struct CivilDate
{
    int year = 0;
    unsigned month = 0;
    unsigned day = 0;
};

CivilDate ParseDate(const std::string& text)
{
    static const std::regex pattern{
        R"(^([0-9]{4})-([0-9]{2})-([0-9]{2})$)"};
    std::smatch match;

    if (!std::regex_match(text, match, pattern))
        throw std::invalid_argument("economic_event_source_release_date_invalid");

    CivilDate result{
        std::stoi(match[1].str()),
        static_cast<unsigned>(std::stoi(match[2].str())),
        static_cast<unsigned>(std::stoi(match[3].str()))};

    const std::chrono::year_month_day date{
        std::chrono::year{result.year},
        std::chrono::month{result.month},
        std::chrono::day{result.day}};

    if (!date.ok())
        throw std::invalid_argument("economic_event_source_release_date_invalid");

    return result;
}

std::string NextDate(const std::string& text)
{
    const CivilDate parsed = ParseDate(text);
    const std::chrono::year_month_day next{
        std::chrono::sys_days{
            std::chrono::year{parsed.year} /
            std::chrono::month{parsed.month} /
            std::chrono::day{parsed.day}} +
        std::chrono::days{1}};

    const int year = static_cast<int>(next.year());
    const unsigned month = static_cast<unsigned>(next.month());
    const unsigned day = static_cast<unsigned>(next.day());

    auto twoDigits = [](unsigned value)
    {
        std::string result = std::to_string(value);
        if (result.size() == 1)
            result.insert(result.begin(), '0');
        return result;
    };

    return std::to_string(year) + "-" +
        twoDigits(month) + "-" + twoDigits(day);
}

std::int64_t NewYorkUnixMicros(const std::string& civil)
{
    PriceTP timestamp;
    if (!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(
            civil,
            timestamp))
    {
        throw std::invalid_argument(
            "economic_event_source_civil_timestamp_invalid");
    }

    return std::chrono::duration_cast<std::chrono::microseconds>(
        timestamp.time_since_epoch()).count();
}

void ValidateCandidate(
    const AuthoritativeEconomicEventCandidate& candidate)
{
    static const std::regex currencyPattern{R"(^[A-Z]{3}$)"};
    static const std::regex identityPattern{
        R"(^[a-z0-9_]+:[a-z0-9][a-z0-9._-]*$)"};
    static const std::regex timePattern{
        R"(^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$)"};

    if (!std::regex_match(candidate.currency, currencyPattern))
        throw std::invalid_argument("economic_event_currency_invalid");
    if (candidate.eventFamily.empty() || IsBlank(candidate.eventFamily))
        throw std::invalid_argument("economic_event_family_missing");
    if (candidate.sourceAgency.empty() || IsBlank(candidate.sourceAgency))
        throw std::invalid_argument("economic_event_source_agency_missing");
    if (candidate.sourceUrl.empty() || IsBlank(candidate.sourceUrl) ||
        candidate.sourceUrl.rfind("https://", 0) != 0)
    {
        throw std::invalid_argument("economic_event_source_url_invalid");
    }
    if (!std::regex_match(candidate.sourceEventId, identityPattern) ||
        candidate.sourceEventId.rfind(
            IdentityPrefix(candidate.sourceAgency) + ":",
            0) != 0)
    {
        throw std::invalid_argument("economic_event_source_event_id_invalid");
    }
    if (candidate.eventImportance < 1 || candidate.eventImportance > 3)
        throw std::invalid_argument("economic_event_importance_invalid");

    const bool exact = candidate.historicalTimeConfidence == "exact";
    const bool reconstructed =
        candidate.historicalTimeConfidence == "reconstructed";
    const bool dateOnly =
        candidate.historicalTimeConfidence == "date_only";

    if (!exact && !reconstructed && !dateOnly)
        throw std::invalid_argument("economic_event_confidence_invalid");
    if (!candidate.sourceReleaseDate || !candidate.sourceTimezone)
        throw std::invalid_argument("economic_event_release_provenance_missing");
    if (*candidate.sourceTimezone != "America/New_York")
        throw std::invalid_argument("economic_event_source_timezone_unsupported");

    ParseDate(*candidate.sourceReleaseDate);

    if (dateOnly)
    {
        if (candidate.sourceReleaseTime)
            throw std::invalid_argument("economic_event_date_only_has_release_time");

        const std::int64_t expected = NewYorkUnixMicros(
            NextDate(*candidate.sourceReleaseDate) + " 00:00:00");
        if (candidate.eventTimestampUnixMicros != expected)
            throw std::invalid_argument("economic_event_local_utc_mismatch");
        return;
    }

    if (!candidate.sourceReleaseTime ||
        !std::regex_match(*candidate.sourceReleaseTime, timePattern))
    {
        throw std::invalid_argument("economic_event_source_release_time_invalid");
    }

    const std::int64_t expected = NewYorkUnixMicros(
        *candidate.sourceReleaseDate + " " +
        *candidate.sourceReleaseTime);
    if (candidate.eventTimestampUnixMicros != expected)
        throw std::invalid_argument("economic_event_local_utc_mismatch");
}

} // namespace


std::vector<AuthoritativeEconomicEventCandidate>
ValidateAndOrderEconomicEventCandidates(
    std::vector<AuthoritativeEconomicEventCandidate> candidates)
{
    for (const auto& candidate : candidates)
        ValidateCandidate(candidate);

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const auto& left, const auto& right)
        {
            return std::tie(
                left.eventTimestampUnixMicros,
                left.sourceAgency,
                left.eventFamily,
                left.sourceEventId) <
                std::tie(
                    right.eventTimestampUnixMicros,
                    right.sourceAgency,
                    right.eventFamily,
                    right.sourceEventId);
        });

    std::set<std::pair<std::string, std::string>> identities;
    std::set<std::tuple<std::string, std::string, std::int64_t>> timestamps;

    for (const auto& candidate : candidates)
    {
        if (!identities.emplace(
                candidate.sourceAgency,
                candidate.sourceEventId).second)
        {
            throw std::invalid_argument(
                "economic_event_duplicate_source_event_id_in_batch");
        }

        if (!timestamps.emplace(
                candidate.sourceAgency,
                candidate.eventFamily,
                candidate.eventTimestampUnixMicros).second)
        {
            throw std::invalid_argument(
                "economic_event_duplicate_agency_family_timestamp_in_batch");
        }
    }

    return candidates;
}

} // namespace EA::EconomicCalendar
