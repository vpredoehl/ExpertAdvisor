#include "BeaEconomicReleaseAdapter.hpp"

#include "../Headers/HistoricalFxTimestamp.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{
namespace
{

constexpr int kBeaImportance = 3;

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        throw std::invalid_argument("bea_manifest_artifact_missing:" + path.string());

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof())
        throw std::runtime_error("bea_manifest_artifact_read_failed:" + path.string());
    return contents.str();
}

std::string Sha256(const std::string& content)
{
    if (content.size() > std::numeric_limits<CC_LONG>::max())
        throw std::invalid_argument("bea_manifest_artifact_too_large");

    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(
        content.data(),
        static_cast<CC_LONG>(content.size()),
        digest.data());

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char byte : digest)
        output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

std::vector<std::string> SplitTabs(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true)
    {
        const std::size_t separator = line.find('\t', start);
        if (separator == std::string::npos)
        {
            fields.push_back(line.substr(start));
            return fields;
        }
        fields.push_back(line.substr(start, separator - start));
        start = separator + 1;
    }
}

bool IsInside(
    const std::filesystem::path& root,
    const std::filesystem::path& path)
{
    const auto relative = path.lexically_relative(root);
    return !relative.empty() && !relative.is_absolute() &&
        *relative.begin() != "..";
}

std::filesystem::path ResolveManifestPath(
    const std::filesystem::path& root,
    const std::string& relativeText)
{
    const std::filesystem::path relative{relativeText};
    if (relative.empty() || relative.is_absolute())
        throw std::invalid_argument("bea_manifest_artifact_path_invalid");

    const auto resolved = std::filesystem::weakly_canonical(root / relative);
    if (!IsInside(root, resolved))
        throw std::invalid_argument("bea_manifest_artifact_path_escapes_root");
    return resolved;
}

void VerifyHash(
    const std::filesystem::path& path,
    const std::string& expected,
    std::string* content = nullptr)
{
    static const std::regex digestPattern{R"(^[0-9a-f]{64}$)"};
    if (!std::regex_match(expected, digestPattern))
        throw std::invalid_argument("bea_manifest_sha256_invalid");

    const std::string bytes = ReadFile(path);
    if (Sha256(bytes) != expected)
        throw std::invalid_argument("bea_manifest_sha256_mismatch:" + path.string());
    if (content)
        *content = bytes;
}

void ReplaceAll(
    std::string& value,
    const std::string& from,
    const std::string& to)
{
    std::size_t offset = 0;
    while ((offset = value.find(from, offset)) != std::string::npos)
    {
        value.replace(offset, from.size(), to);
        offset += to.size();
    }
}

std::string VisibleText(const std::string& artifact)
{
    std::string text = artifact;
    ReplaceAll(text, "&nbsp;", " ");
    ReplaceAll(text, "&#160;", " ");
    ReplaceAll(text, "&amp;", "&");
    ReplaceAll(text, "&quot;", "\"");
    ReplaceAll(text, "&ndash;", "–");
    ReplaceAll(text, "&mdash;", "—");
    ReplaceAll(text, "&#8211;", "–");
    ReplaceAll(text, "&#8212;", "—");
    // Several historical BEA pages contain a literal UTF-8 non-breaking
    // space instead of the equivalent HTML entity. Normalize that exact
    // sequence before byte-wise whitespace collapsing.
    ReplaceAll(text, "\xC2\xA0", " ");
    text = std::regex_replace(text, std::regex{R"(<[^>]*>)"}, " ");

    std::string collapsed;
    collapsed.reserve(text.size());
    bool previousSpace = true;
    for (unsigned char character : text)
    {
        if (std::isspace(character) != 0)
        {
            if (!previousSpace)
                collapsed.push_back(' ');
            previousSpace = true;
        }
        else
        {
            collapsed.push_back(static_cast<char>(character));
            previousSpace = false;
        }
    }
    if (!collapsed.empty() && collapsed.back() == ' ')
        collapsed.pop_back();
    return collapsed;
}

std::string Lower(std::string value)
{
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

unsigned MonthNumber(const std::string& month)
{
    static const std::array<const char*, 12> months{
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"};
    const std::string lower = Lower(month);
    const auto found = std::find(months.begin(), months.end(), lower);
    if (found == months.end())
        throw std::invalid_argument("bea_release_month_invalid");
    return static_cast<unsigned>(std::distance(months.begin(), found) + 1);
}

std::string DateText(int year, unsigned month, unsigned day)
{
    const std::chrono::year_month_day date{
        std::chrono::year{year},
        std::chrono::month{month},
        std::chrono::day{day}};
    if (!date.ok())
        throw std::invalid_argument("bea_release_date_invalid");

    std::ostringstream output;
    output << std::setfill('0') << std::setw(4) << year << '-'
           << std::setw(2) << month << '-' << std::setw(2) << day;
    return output.str();
}

struct ParsedDate
{
    int year = 0;
    unsigned month = 0;
    unsigned day = 0;
};

ParsedDate ParseLongDate(const std::string& text)
{
    static const std::regex pattern{
        R"(^([A-Za-z]+) ([0-9]{1,2}), ([0-9]{4})$)"};
    std::smatch match;
    if (!std::regex_match(text, match, pattern))
        throw std::invalid_argument("bea_release_date_invalid");

    ParsedDate result{
        std::stoi(match[3].str()),
        MonthNumber(match[1].str()),
        static_cast<unsigned>(std::stoi(match[2].str()))};
    (void)DateText(result.year, result.month, result.day);
    return result;
}

std::int64_t UnixMicros(const std::string& civil)
{
    PriceTP instant;
    if (!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(civil, instant))
        throw std::invalid_argument("bea_release_civil_timestamp_invalid");
    return std::chrono::duration_cast<std::chrono::microseconds>(
        instant.time_since_epoch()).count();
}

unsigned QuarterNumber(const std::string& value)
{
    const std::string lower = Lower(value);
    if (lower == "1" || lower == "first")
        return 1;
    if (lower == "2" || lower == "second")
        return 2;
    if (lower == "3" || lower == "third")
        return 3;
    if (lower == "4" || lower == "fourth")
        return 4;
    throw std::invalid_argument("bea_gdp_reference_quarter_invalid");
}

struct ParsedPublication
{
    std::string family;
    std::string referencePeriod;
};

ParsedPublication ParsePublicationHeading(const std::string& afterIdentity)
{
    const std::string lower = Lower(afterIdentity);
    const std::size_t gdpBody = lower.find("real gross domestic product");
    const bool beginsGdp = lower.rfind("gross domestic product", 0) == 0 ||
        lower.rfind("gdp (", 0) == 0;

    if (beginsGdp)
    {
        if (gdpBody == std::string::npos || gdpBody == 0 || gdpBody > 1000)
            throw std::invalid_argument("bea_gdp_release_heading_invalid");
        const std::string heading = afterIdentity.substr(0, gdpBody);

        static const std::regex vintagePattern{
            R"(\b(advance|second|third|initial|updated)\s+estimate\b)",
            std::regex::icase};
        std::set<std::string> vintages;
        for (std::sregex_iterator iterator{
                 heading.begin(), heading.end(), vintagePattern}, end;
             iterator != end; ++iterator)
        {
            vintages.insert(Lower((*iterator)[1].str()));
        }
        if (vintages.size() != 1)
            throw std::invalid_argument("bea_gdp_vintage_ambiguous");

        static const std::regex numericQuarter{
            R"(\b([1-4])(?:st|nd|rd|th)\s+quarter(?:\s+and\s+(?:annual|year))?\s+([0-9]{4})\b)",
            std::regex::icase};
        static const std::regex wordQuarter{
            R"(\b(first|second|third|fourth)\s+quarter(?:\s+and\s+(?:annual|year))?[ ,]+([0-9]{4})\b)",
            std::regex::icase};
        std::smatch quarter;
        if (!std::regex_search(heading, quarter, numericQuarter) &&
            !std::regex_search(heading, quarter, wordQuarter))
        {
            throw std::invalid_argument("bea_gdp_reference_period_missing");
        }

        const unsigned quarterNumber = QuarterNumber(quarter[1].str());
        const int year = std::stoi(quarter[2].str());
        if (year < 1900 || year > 2200)
            throw std::invalid_argument("bea_gdp_reference_period_invalid");

        std::string family;
        if (*vintages.begin() == "advance" || *vintages.begin() == "initial")
            family = "GDP_ADVANCE";
        else if (*vintages.begin() == "second")
            family = "GDP_SECOND";
        else
            family = "GDP_THIRD";
        return {family, "Q" + std::to_string(quarterNumber) + " " +
            std::to_string(year)};
    }

    // During the 2018-2019 shutdown BEA published February personal income
    // together with the delayed January personal-outlays occurrence.  The
    // calendar family represents the complete Income and Outlays occurrence,
    // so its causal reference month is the explicitly named outlays month.
    static const std::regex splitPersonalHeading{
        R"(^Personal Income(?:,|:)\s+([A-Za-z]+)\s+([0-9]{4});\s+Personal Outlays(?:,|:)\s+([A-Za-z]+)\s+([0-9]{4})\b)",
        std::regex::icase};
    std::smatch splitPersonal;
    if (std::regex_search(afterIdentity, splitPersonal, splitPersonalHeading))
    {
        (void)MonthNumber(splitPersonal[1].str());
        const int incomeYear = std::stoi(splitPersonal[2].str());
        const unsigned outlaysMonth = MonthNumber(splitPersonal[3].str());
        const int outlaysYear = std::stoi(splitPersonal[4].str());
        if (incomeYear != outlaysYear)
            throw std::invalid_argument("bea_personal_income_reference_period_invalid");

        std::ostringstream reference;
        reference << std::setfill('0') << std::setw(4) << outlaysYear << '-'
                  << std::setw(2) << outlaysMonth;
        return {"PERSONAL_INCOME_OUTLAYS", reference.str()};
    }

    static const std::regex personalHeading{
        R"(^Personal Income and Outlays(?:,|:|\s+for)\s+([A-Za-z]+)\s+([0-9]{4})\b)",
        std::regex::icase};
    std::smatch personal;
    if (std::regex_search(afterIdentity, personal, personalHeading))
    {
        const unsigned month = MonthNumber(personal[1].str());
        const int year = std::stoi(personal[2].str());
        std::ostringstream reference;
        reference << std::setfill('0') << std::setw(4) << year << '-'
                  << std::setw(2) << month;
        return {"PERSONAL_INCOME_OUTLAYS", reference.str()};
    }

    static const std::regex combinedPersonalHeading{
        R"(^Personal Income and Outlays(?:,|:)\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)\s+([0-9]{4})\b)",
        std::regex::icase};
    std::smatch combinedPersonal;
    if (std::regex_search(afterIdentity, combinedPersonal, combinedPersonalHeading))
    {
        const unsigned firstMonth = MonthNumber(combinedPersonal[1].str());
        const unsigned secondMonth = MonthNumber(combinedPersonal[2].str());
        const int year = std::stoi(combinedPersonal[3].str());
        if (secondMonth != firstMonth + 1)
            throw std::invalid_argument("bea_personal_income_reference_period_invalid");

        std::ostringstream reference;
        reference << std::setfill('0') << std::setw(4) << year << '-'
                  << std::setw(2) << firstMonth << '/' << std::setw(4) << year
                  << '-' << std::setw(2) << secondMonth;
        return {"PERSONAL_INCOME_OUTLAYS", reference.str()};
    }

    if (lower.rfind("personal income and outlays", 0) == 0)
        throw std::invalid_argument("bea_personal_income_reference_period_missing");
    throw std::invalid_argument("bea_release_family_unsupported");
}

} // namespace


std::string NormalizeBeaReleaseId(const std::string& releaseId)
{
    static const std::regex pattern{
        R"(^\s*BEA\s+([0-9]{2}|[0-9]{4})\s*(?:-|–|—)\s*([0-9]{1,3})\s*$)",
        std::regex::icase};
    std::smatch match;
    if (!std::regex_match(releaseId, match, pattern))
        throw std::invalid_argument("bea_release_id_malformed");

    const int year = std::stoi(match[1].str());
    const int sequence = std::stoi(match[2].str());
    if (sequence <= 0)
        throw std::invalid_argument("bea_release_id_malformed");
    const int shortYear = match[1].str().size() == 4 ? year % 100 : year;
    std::ostringstream output;
    output << "bea:" << std::setfill('0') << std::setw(2) << shortYear
           << '-' << sequence;
    return output.str();
}


AuthoritativeEconomicEventCandidate ParseBeaEconomicReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl)
{
    static const std::regex sourceUrlPattern{
        R"(^https://www\.bea\.gov/news/([0-9]{4})/[a-z0-9][a-z0-9-]*$)"};
    std::smatch urlMatch;
    if (!std::regex_match(canonicalSourceUrl, urlMatch, sourceUrlPattern))
        throw std::invalid_argument("bea_source_url_not_canonical_first_party");

    const std::string text = VisibleText(artifact);
    static const std::regex embargoPattern{
        R"(EMBARGOED\s+(?:UNTIL RELEASE AT|FOR RELEASE:)\s+([0-9]{1,2}):([0-9]{2})\s*([AP])\.?M\.?\s*,?\s*([A-Za-z]+),\s+(?:[A-Za-z]+,?\s+)?([A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4})\.?)",
        std::regex::icase};
    std::smatch embargo;
    if (!std::regex_search(text, embargo, embargoPattern))
        throw std::invalid_argument("bea_authoritative_release_time_missing");

    int hour = std::stoi(embargo[1].str());
    const int minute = std::stoi(embargo[2].str());
    const std::string meridiem = Lower(embargo[3].str());
    const std::string zone = Lower(embargo[4].str());
    if (hour < 1 || hour > 12 || minute < 0 || minute > 59)
        throw std::invalid_argument("bea_authoritative_release_time_invalid");
    if (meridiem == "a")
        hour = hour == 12 ? 0 : hour;
    else
        hour = hour == 12 ? 12 : hour + 12;

    if (zone != "edt" && zone != "est" && zone != "eastern")
        throw std::invalid_argument("bea_release_timezone_unsupported");

    const ParsedDate release = ParseLongDate(embargo[5].str());
    const std::string releaseDate =
        DateText(release.year, release.month, release.day);
    if (std::stoi(urlMatch[1].str()) != release.year)
        throw std::invalid_argument("bea_source_url_release_year_mismatch");

    std::ostringstream timeOutput;
    timeOutput << std::setfill('0') << std::setw(2) << hour << ':'
               << std::setw(2) << minute << ":00";
    const std::string releaseTime = timeOutput.str();
    const std::int64_t instant = UnixMicros(releaseDate + " " + releaseTime);

    if (zone == "edt" || zone == "est")
    {
        std::tm civil{};
        civil.tm_year = release.year - 1900;
        civil.tm_mon = static_cast<int>(release.month) - 1;
        civil.tm_mday = static_cast<int>(release.day);
        civil.tm_hour = hour;
        civil.tm_min = minute;
        const std::time_t naive = timegm(&civil);
        const std::int64_t offsetSeconds =
            instant / 1000000 - static_cast<std::int64_t>(naive);
        if ((zone == "edt" && offsetSeconds != 4 * 60 * 60) ||
            (zone == "est" && offsetSeconds != 5 * 60 * 60))
        {
            throw std::invalid_argument("bea_release_timezone_contradiction");
        }
    }

    static const std::regex releaseIdPattern{
        R"(^\s*((?:BEA\s+)?[0-9]{2,4}\s*(?:-|–|—)\s*[0-9]{1,3})\b)",
        std::regex::icase};
    static const std::regex beaMarker{R"(\bBEA\b)", std::regex::icase};
    const std::size_t identitySearchStart = static_cast<std::size_t>(
        embargo.position() + embargo.length());
    const std::string identitySuffix = text.substr(identitySearchStart);
    std::smatch releaseId;
    if (!std::regex_search(identitySuffix, releaseId, releaseIdPattern))
    {
        if (std::regex_search(text, beaMarker))
            throw std::invalid_argument("bea_release_id_malformed");
        throw std::invalid_argument("bea_authoritative_identity_missing");
    }
    std::string releaseIdText = releaseId[1].str();
    if (Lower(releaseIdText).rfind("bea", 0) != 0)
        releaseIdText.insert(0, "BEA ");
    const std::string sourceEventId = NormalizeBeaReleaseId(releaseIdText);
    const int identityYear = std::stoi(sourceEventId.substr(4, 2));
    if (identityYear != release.year % 100)
        throw std::invalid_argument("bea_release_id_year_mismatch");

    std::string afterIdentity = identitySuffix.substr(
        static_cast<std::size_t>(releaseId.position() + releaseId.length()));
    while (!afterIdentity.empty() && afterIdentity.front() == ' ')
        afterIdentity.erase(afterIdentity.begin());
    const ParsedPublication publication = ParsePublicationHeading(afterIdentity);

    AuthoritativeEconomicEventCandidate candidate;
    candidate.currency = "USD";
    candidate.eventFamily = publication.family;
    candidate.eventTimestampUnixMicros = instant;
    candidate.sourceAgency = "BEA";
    candidate.sourceEventId = sourceEventId;
    candidate.sourceUrl = canonicalSourceUrl;
    candidate.referencePeriod = publication.referencePeriod;
    candidate.eventImportance = kBeaImportance;
    candidate.historicalTimeConfidence = "exact";
    candidate.sourceReleaseDate = releaseDate;
    candidate.sourceReleaseTime = releaseTime;
    candidate.sourceTimezone = "America/New_York";
    return candidate;
}


std::vector<AuthoritativeEconomicEventCandidate>
LoadBeaEconomicReleaseManifest(
    const std::filesystem::path& manifestPath)
{
    const auto canonicalManifest = std::filesystem::weakly_canonical(manifestPath);
    const auto root = canonicalManifest.parent_path();
    std::ifstream input{canonicalManifest};
    if (!input)
        throw std::invalid_argument("bea_manifest_missing");

    std::string line;
    if (!std::getline(input, line) || line != "manifest_version\t1")
        throw std::invalid_argument("bea_manifest_version_invalid");
    if (!std::getline(input, line) ||
        line != "parser_version\tbea_economic_release_v2")
    {
        throw std::invalid_argument("bea_manifest_parser_version_invalid");
    }
    if (!std::getline(input, line) ||
        line != "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
                "source_artifact_path\tsource_artifact_sha256\textractor")
    {
        throw std::invalid_argument("bea_manifest_columns_invalid");
    }

    std::vector<AuthoritativeEconomicEventCandidate> candidates;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const auto fields = SplitTabs(line);
        if (fields.size() != 7)
            throw std::invalid_argument("bea_manifest_entry_invalid");

        const auto artifactPath = ResolveManifestPath(root, fields[0]);
        std::string artifact;
        VerifyHash(artifactPath, fields[1], &artifact);

        const std::string& type = fields[2];
        if (type == "html")
        {
            if (fields[4] != fields[0] || fields[5] != fields[1] ||
                fields[6] != "none")
            {
                throw std::invalid_argument("bea_manifest_html_provenance_invalid");
            }
        }
        else if (type == "text_fixture")
        {
            if (fields[4] != fields[0] || fields[5] != fields[1] ||
                fields[6] != "first_party_excerpt_v1")
            {
                throw std::invalid_argument("bea_manifest_fixture_provenance_invalid");
            }
        }
        else
        {
            throw std::invalid_argument("bea_manifest_artifact_type_unsupported");
        }

        candidates.push_back(
            ParseBeaEconomicReleaseArtifact(artifact, fields[3]));
    }

    if (!input.eof())
        throw std::runtime_error("bea_manifest_read_failed");
    if (candidates.empty())
        throw std::invalid_argument("bea_manifest_empty");
    return candidates;
}

} // namespace EA::EconomicCalendar
