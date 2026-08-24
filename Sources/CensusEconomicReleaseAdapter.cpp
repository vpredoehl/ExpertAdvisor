#include "CensusEconomicReleaseAdapter.hpp"

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

constexpr int kCensusImportance = 3;

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        throw std::invalid_argument("census_manifest_artifact_missing:" + path.string());

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof())
        throw std::runtime_error("census_manifest_artifact_read_failed:" + path.string());
    return contents.str();
}

std::string Sha256(const std::string& content)
{
    if (content.size() > std::numeric_limits<CC_LONG>::max())
        throw std::invalid_argument("census_manifest_artifact_too_large");

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
        throw std::invalid_argument("census_manifest_artifact_path_invalid");

    const auto resolved = std::filesystem::weakly_canonical(root / relative);
    if (!IsInside(root, resolved))
        throw std::invalid_argument("census_manifest_artifact_path_escapes_root");
    return resolved;
}

void VerifyHash(
    const std::filesystem::path& path,
    const std::string& expected,
    std::string* content = nullptr)
{
    static const std::regex digestPattern{R"(^[0-9a-f]{64}$)"};
    if (!std::regex_match(expected, digestPattern))
        throw std::invalid_argument("census_manifest_sha256_invalid");

    const std::string bytes = ReadFile(path);
    if (Sha256(bytes) != expected)
        throw std::invalid_argument("census_manifest_sha256_mismatch:" + path.string());
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
    // Narrow pdftotext repairs observed in the 2015 M3 archive. These restore
    // words visibly split by the authoritative PDF's embedded text layout.
    ReplaceAll(text, "RELEAS E", "RELEASE");
    ReplaceAll(text, "SEPTEMB ER", "SEPTEMBER");
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
        throw std::invalid_argument("census_release_month_invalid");
    return static_cast<unsigned>(std::distance(months.begin(), found) + 1);
}

unsigned ShortMonthNumber(const std::string& month)
{
    static const std::array<const char*, 12> months{
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"};
    const std::string lower = Lower(month);
    const auto found = std::find(months.begin(), months.end(), lower);
    if (found == months.end())
        throw std::invalid_argument("census_source_url_month_invalid");
    return static_cast<unsigned>(std::distance(months.begin(), found) + 1);
}

std::string DateText(int year, unsigned month, unsigned day)
{
    const std::chrono::year_month_day date{
        std::chrono::year{year},
        std::chrono::month{month},
        std::chrono::day{day}};
    if (!date.ok())
        throw std::invalid_argument("census_release_date_invalid");

    std::ostringstream output;
    output << std::setfill('0') << std::setw(4) << year << '-'
           << std::setw(2) << month << '-' << std::setw(2) << day;
    return output.str();
}

std::string ReferenceText(int year, unsigned month)
{
    if (year < 1900 || year > 2099 || month < 1 || month > 12)
        throw std::invalid_argument("census_reference_period_invalid");
    std::ostringstream output;
    output << std::setfill('0') << std::setw(4) << year << '-'
           << std::setw(2) << month;
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
        throw std::invalid_argument("census_release_date_invalid");

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
        throw std::invalid_argument("census_release_civil_timestamp_invalid");
    return std::chrono::duration_cast<std::chrono::microseconds>(
        instant.time_since_epoch()).count();
}

struct SourcePublication
{
    std::string family;
    int referenceYear = 0;
    unsigned referenceMonth = 0;
};

SourcePublication ParseSourcePublication(const std::string& url)
{
    std::smatch match;
    static const std::regex retail{
        R"(^https://www2\.census\.gov/retail/releases/historical/marts/adv([0-9]{2})([0-9]{2})\.pdf$)"};
    if (std::regex_match(url, match, retail))
    {
        return {
            "RETAIL_SALES_ADVANCE",
            2000 + std::stoi(match[1].str()),
            static_cast<unsigned>(std::stoi(match[2].str()))};
    }

    static const std::regex residentialConstruction{
        R"(^https://www\.census\.gov/construction/nrc/pdf/newresconst_([0-9]{4})([0-9]{2})\.pdf$)"};
    if (std::regex_match(url, match, residentialConstruction))
    {
        return {
            "NEW_RESIDENTIAL_CONSTRUCTION",
            std::stoi(match[1].str()),
            static_cast<unsigned>(std::stoi(match[2].str()))};
    }

    static const std::regex residentialSales{
        R"(^https://www\.census\.gov/construction/nrs/pdf/newressales_([0-9]{4})([0-9]{2})\.pdf$)"};
    if (std::regex_match(url, match, residentialSales))
    {
        return {
            "NEW_RESIDENTIAL_SALES",
            std::stoi(match[1].str()),
            static_cast<unsigned>(std::stoi(match[2].str()))};
    }

    static const std::regex m3{
        R"(^https://www\.census\.gov/manufacturing/m3/historical_data/pressreleases/(adv|prel)/([0-9]{4})/([a-z]{3})([0-9]{2})(adv|prel)\.pdf$)"};
    if (std::regex_match(url, match, m3))
    {
        if (match[1].str() != match[5].str() ||
            std::stoi(match[4].str()) != std::stoi(match[2].str()) % 100)
        {
            throw std::invalid_argument("census_source_url_identity_contradiction");
        }
        return {
            match[1].str() == "adv"
                ? "DURABLE_GOODS_ADVANCE"
                : "MANUFACTURERS_ORDERS",
            std::stoi(match[2].str()),
            ShortMonthNumber(match[3].str())};
    }

    static const std::regex constructionSpending{
        R"(^https://www\.census\.gov/construction/c30/pdf/pr([0-9]{4})([0-9]{2})\.pdf$)"};
    if (std::regex_match(url, match, constructionSpending))
    {
        return {
            "CONSTRUCTION_SPENDING",
            std::stoi(match[1].str()),
            static_cast<unsigned>(std::stoi(match[2].str()))};
    }

    throw std::invalid_argument("census_source_url_not_canonical_first_party");
}

struct PublicationHeading
{
    std::string family;
    int referenceYear = 0;
    unsigned referenceMonth = 0;
};

PublicationHeading ParsePublicationHeading(const std::string& text)
{
    const std::string header = text.substr(0, std::min<std::size_t>(text.size(), 1600));
    struct FamilyPattern
    {
        const char* family;
        std::regex pattern;
    };
    const std::vector<FamilyPattern> patterns{
        {"RETAIL_SALES_ADVANCE", std::regex{
            R"(ADVANCE MONTHLY SALES FOR RETAIL AND FOOD SERVICES,?\s+([A-Za-z]+)\s+([0-9]{4})\b)",
            std::regex::icase}},
        {"NEW_RESIDENTIAL_CONSTRUCTION", std::regex{
            R"((?:MONTHLY\s+)?NEW RESIDENTIAL CONSTRUCTION(?:\s+IN|,)?\s+([A-Za-z]+)\s+([0-9]{4})\b)",
            std::regex::icase}},
        {"NEW_RESIDENTIAL_SALES", std::regex{
            R"((?:MONTHLY\s+)?NEW RESIDENTIAL SALES(?:\s+IN|,)?\s+([A-Za-z]+)\s+([0-9]{4})\b)",
            std::regex::icase}},
        {"MANUFACTURERS_ORDERS", std::regex{
            R"((?:MONTHLY\s+)?FULL REPORT ON MANUFACTURERS(?:'|’)[ ]*SHIPMENTS,\s*INVENTORIES,?\s+AND\s+ORDERS\s+([A-Za-z]+)\s+([0-9]{4})\b)",
            std::regex::icase}},
        {"MANUFACTURERS_ORDERS", std::regex{
            R"((?:MONTHLY\s+)?FULL REPORT ON MANUFACTURERS(?:'|’)[ ]*SHIPMENTS,\s*INVENTORIES,?\s+AND\s+ORDERS\s+[A-Za-z]+/([A-Za-z]+)\s+([0-9]{4})\b)",
            std::regex::icase}},
        {"DURABLE_GOODS_ADVANCE", std::regex{
            R"((?:MONTHLY\s+)?ADVANCE REPORT ON (?:DURABLE GOODS )?MANUFACTURERS(?:'|’)[ ]*SHIPMENTS,\s*INVENTORIES,?\s+AND\s+ORDERS\s+([A-Za-z]+)\s+([0-9]{4})\b)",
            std::regex::icase}},
        {"CONSTRUCTION_SPENDING", std::regex{
            R"(MONTHLY CONSTRUCTION SPENDING,\s+([A-Za-z]+)\s+([0-9]{4})\b)",
            std::regex::icase}},
        {"CONSTRUCTION_SPENDING", std::regex{
            R"(\b([A-Za-z]+)\s+([0-9]{4})\s+CONSTRUCTION AT\b)",
            std::regex::icase}}};

    std::vector<PublicationHeading> matches;
    for (const auto& entry : patterns)
    {
        std::smatch match;
        if (std::regex_search(header, match, entry.pattern))
        {
            matches.push_back({
                entry.family,
                std::stoi(match[2].str()),
                MonthNumber(match[1].str())});
        }
    }

    if (matches.empty())
        throw std::invalid_argument("census_release_family_unsupported");
    if (matches.size() != 1)
        throw std::invalid_argument("census_release_family_ambiguous");
    return matches.front();
}

struct TimestampEvidence
{
    int hour = 0;
    int minute = 0;
    std::string meridiem;
    std::string zone;
    ParsedDate date;
    bool dateOnly = false;
};

TimestampEvidence ParseTimestampEvidence(const std::string& text)
{
    static const std::regex releaseAt{
        R"(FOR RELEASE AT\s+([0-9]{1,2}):([0-9]{2})\s*([AP])\.?M\.?\s+(EDT|EST|EASTERN),\s+(?:[A-Z]+,\s+)?([A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4}))",
        std::regex::icase};
    static const std::regex immediate{
        R"(FOR IMMEDIATE RELEASE\s+(?:[A-Z]+,?\s+)?([A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4}),?\s+AT\s+([0-9]{1,2}):([0-9]{2})\s*([AP])\.?M\.?\s+(EDT|EST|EASTERN))",
        std::regex::icase};

    std::smatch match;
    TimestampEvidence result;
    if (std::regex_search(text, match, releaseAt))
    {
        result.hour = std::stoi(match[1].str());
        result.minute = std::stoi(match[2].str());
        result.meridiem = Lower(match[3].str());
        result.zone = Lower(match[4].str());
        result.date = ParseLongDate(match[5].str());
    }
    else if (std::regex_search(text, match, immediate))
    {
        result.date = ParseLongDate(match[1].str());
        result.hour = std::stoi(match[2].str());
        result.minute = std::stoi(match[3].str());
        result.meridiem = Lower(match[4].str());
        result.zone = Lower(match[5].str());
    }
    else
    {
        static const std::regex immediateDateOnly{
            R"(FOR IMMEDIATE RELEASE\s+(?:[A-Z]+,?\s+)?([A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4})\b(?!,?\s+AT\b))",
            std::regex::icase};
        if (!std::regex_search(text, match, immediateDateOnly))
            throw std::invalid_argument("census_authoritative_release_time_missing");
        result.date = ParseLongDate(match[1].str());
        result.dateOnly = true;
        return result;
    }

    if (result.hour < 1 || result.hour > 12 ||
        result.minute < 0 || result.minute > 59)
    {
        throw std::invalid_argument("census_authoritative_release_time_invalid");
    }
    if (result.meridiem == "a")
        result.hour = result.hour == 12 ? 0 : result.hour;
    else
        result.hour = result.hour == 12 ? 12 : result.hour + 12;
    return result;
}

} // namespace


std::string NormalizeCensusReleaseId(const std::string& releaseId)
{
    static const std::regex pattern{
        R"(^\s*CB\s*([0-9]{2})\s*(?:-|‐|–|—)\s*([0-9]{1,3})\s*$)",
        std::regex::icase};
    std::smatch match;
    if (!std::regex_match(releaseId, match, pattern))
        throw std::invalid_argument("census_release_id_malformed");

    const int sequence = std::stoi(match[2].str());
    if (sequence <= 0)
        throw std::invalid_argument("census_release_id_malformed");
    return "census:cb" + match[1].str() + '-' + std::to_string(sequence);
}


AuthoritativeEconomicEventCandidate ParseCensusEconomicReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl)
{
    const SourcePublication source = ParseSourcePublication(canonicalSourceUrl);
    const std::string text = VisibleText(artifact);
    const PublicationHeading publication = ParsePublicationHeading(text);
    if (publication.family != source.family)
        throw std::invalid_argument("census_publication_url_title_mismatch");
    const bool shutdownCombinedPeriod =
        (source.family == "NEW_RESIDENTIAL_CONSTRUCTION" ||
         source.family == "NEW_RESIDENTIAL_SALES") &&
        source.referenceYear == 2013 && source.referenceMonth == 9 &&
        publication.referenceYear == 2013 && publication.referenceMonth == 10;
    if (!shutdownCombinedPeriod &&
        (publication.referenceYear != source.referenceYear ||
         publication.referenceMonth != source.referenceMonth))
    {
        throw std::invalid_argument("census_source_url_reference_period_mismatch");
    }

    const TimestampEvidence timestamp = ParseTimestampEvidence(text);
    const std::string releaseDate = DateText(
        timestamp.date.year,
        timestamp.date.month,
        timestamp.date.day);
    std::optional<std::string> releaseTime;
    std::int64_t instant = 0;
    if (timestamp.dateOnly)
    {
        const std::chrono::year_month_day next{
            std::chrono::sys_days{
                std::chrono::year{timestamp.date.year} /
                std::chrono::month{timestamp.date.month} /
                std::chrono::day{timestamp.date.day}} + std::chrono::days{1}};
        instant = UnixMicros(
            DateText(
                static_cast<int>(next.year()),
                static_cast<unsigned>(next.month()),
                static_cast<unsigned>(next.day())) + " 00:00:00");
    }
    else
    {
        std::ostringstream timeOutput;
        timeOutput << std::setfill('0') << std::setw(2) << timestamp.hour << ':'
                   << std::setw(2) << timestamp.minute << ":00";
        releaseTime = timeOutput.str();
        instant = UnixMicros(releaseDate + " " + *releaseTime);
    }

    if (!timestamp.dateOnly &&
        (timestamp.zone == "edt" || timestamp.zone == "est"))
    {
        std::tm civil{};
        civil.tm_year = timestamp.date.year - 1900;
        civil.tm_mon = static_cast<int>(timestamp.date.month) - 1;
        civil.tm_mday = static_cast<int>(timestamp.date.day);
        civil.tm_hour = timestamp.hour;
        civil.tm_min = timestamp.minute;
        const std::time_t naive = timegm(&civil);
        const std::int64_t offsetSeconds =
            instant / 1000000 - static_cast<std::int64_t>(naive);
        if ((timestamp.zone == "edt" && offsetSeconds != 4 * 60 * 60) ||
            (timestamp.zone == "est" && offsetSeconds != 5 * 60 * 60))
        {
            throw std::invalid_argument("census_release_timezone_contradiction");
        }
    }

    static const std::regex releaseIdPattern{
        R"(\bCB\s*[0-9]{2}\s*(?:-|‐|–|—)\s*[0-9]{1,3}\b)",
        std::regex::icase};
    std::set<std::string> identities;
    for (std::sregex_iterator iterator{
             text.begin(), text.end(), releaseIdPattern}, end;
         iterator != end; ++iterator)
    {
        identities.insert(NormalizeCensusReleaseId(iterator->str()));
    }
    if (identities.empty())
    {
        static const std::regex censusIdMarker{R"(\bCB)", std::regex::icase};
        if (std::regex_search(text, censusIdMarker))
            throw std::invalid_argument("census_release_id_malformed");
        throw std::invalid_argument("census_authoritative_identity_missing");
    }
    if (identities.size() != 1)
        throw std::invalid_argument("census_authoritative_identity_ambiguous");

    const std::string sourceEventId = *identities.begin();
    const int identityYear = std::stoi(sourceEventId.substr(9, 2));
    const bool delayedShutdownIdentity =
        publication.family == "NEW_RESIDENTIAL_SALES" &&
        source.referenceYear == 2018 && source.referenceMonth == 11 &&
        timestamp.date.year == 2019 && timestamp.date.month == 1 &&
        identityYear == 18;
    if (!delayedShutdownIdentity && identityYear != timestamp.date.year % 100)
        throw std::invalid_argument("census_release_id_year_mismatch");

    AuthoritativeEconomicEventCandidate candidate;
    candidate.currency = "USD";
    candidate.eventFamily = publication.family;
    candidate.eventTimestampUnixMicros = instant;
    candidate.sourceAgency = "CENSUS";
    candidate.sourceEventId = sourceEventId;
    candidate.sourceUrl = canonicalSourceUrl;
    candidate.referencePeriod = ReferenceText(
        publication.referenceYear,
        publication.referenceMonth);
    candidate.eventImportance = kCensusImportance;
    candidate.historicalTimeConfidence =
        timestamp.dateOnly ? "date_only" : "exact";
    candidate.sourceReleaseDate = releaseDate;
    candidate.sourceReleaseTime = releaseTime;
    candidate.sourceTimezone = "America/New_York";
    return candidate;
}


std::vector<AuthoritativeEconomicEventCandidate>
LoadCensusEconomicReleaseManifest(
    const std::filesystem::path& manifestPath)
{
    const auto canonicalManifest = std::filesystem::weakly_canonical(manifestPath);
    const auto root = canonicalManifest.parent_path();
    std::ifstream input{canonicalManifest};
    if (!input)
        throw std::invalid_argument("census_manifest_missing");

    std::string line;
    if (!std::getline(input, line) || line != "manifest_version\t1")
        throw std::invalid_argument("census_manifest_version_invalid");
    if (!std::getline(input, line) ||
        line != "parser_version\tcensus_economic_release_v1")
    {
        throw std::invalid_argument("census_manifest_parser_version_invalid");
    }
    if (!std::getline(input, line) ||
        line != "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
                "source_artifact_path\tsource_artifact_sha256\textractor")
    {
        throw std::invalid_argument("census_manifest_columns_invalid");
    }

    std::vector<AuthoritativeEconomicEventCandidate> candidates;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const auto fields = SplitTabs(line);
        if (fields.size() != 7)
            throw std::invalid_argument("census_manifest_entry_invalid");

        const auto artifactPath = ResolveManifestPath(root, fields[0]);
        std::string artifact;
        VerifyHash(artifactPath, fields[1], &artifact);

        const std::string& type = fields[2];
        if (type == "pdf_text")
        {
            if (fields[4] == "-" || fields[4] == fields[0] ||
                fields[6].rfind("pdftotext-", 0) != 0)
            {
                throw std::invalid_argument("census_manifest_pdf_provenance_invalid");
            }
            VerifyHash(ResolveManifestPath(root, fields[4]), fields[5]);
        }
        else if (type == "text_fixture")
        {
            if (fields[4] != fields[0] || fields[5] != fields[1] ||
                fields[6] != "first_party_excerpt_v1")
            {
                throw std::invalid_argument("census_manifest_fixture_provenance_invalid");
            }
        }
        else
        {
            throw std::invalid_argument("census_manifest_artifact_type_unsupported");
        }

        candidates.push_back(
            ParseCensusEconomicReleaseArtifact(artifact, fields[3]));
    }

    if (!input.eof())
        throw std::runtime_error("census_manifest_read_failed");
    if (candidates.empty())
        throw std::invalid_argument("census_manifest_empty");
    return candidates;
}

} // namespace EA::EconomicCalendar
