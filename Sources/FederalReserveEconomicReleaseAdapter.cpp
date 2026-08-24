#include "FederalReserveEconomicReleaseAdapter.hpp"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace EA::EconomicCalendar
{
namespace
{

constexpr int kFomcImportance = 3;
constexpr int kBeigeBookImportance = 2;

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        throw std::invalid_argument(
            "federal_reserve_manifest_artifact_missing:" + path.string());

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof())
        throw std::runtime_error(
            "federal_reserve_manifest_artifact_read_failed:" + path.string());
    return contents.str();
}

std::string Sha256(const std::string& content)
{
    if (content.size() > std::numeric_limits<CC_LONG>::max())
        throw std::invalid_argument("federal_reserve_manifest_artifact_too_large");

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
        throw std::invalid_argument(
            "federal_reserve_manifest_artifact_path_invalid");

    const auto resolved = std::filesystem::weakly_canonical(root / relative);
    if (!IsInside(root, resolved))
        throw std::invalid_argument(
            "federal_reserve_manifest_artifact_path_escapes_root");
    return resolved;
}

void VerifyHash(
    const std::filesystem::path& path,
    const std::string& expected,
    std::string* content = nullptr)
{
    static const std::regex digestPattern{R"(^[0-9a-f]{64}$)"};
    if (!std::regex_match(expected, digestPattern))
        throw std::invalid_argument("federal_reserve_manifest_sha256_invalid");

    const std::string bytes = ReadFile(path);
    if (Sha256(bytes) != expected)
        throw std::invalid_argument(
            "federal_reserve_manifest_sha256_mismatch:" + path.string());
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
    ReplaceAll(text, "&ndash;", "-");
    ReplaceAll(text, "&mdash;", "-");
    ReplaceAll(text, "&#8211;", "-");
    ReplaceAll(text, "&#8212;", "-");
    ReplaceAll(text, "–", "-");
    ReplaceAll(text, "—", "-");
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
        throw std::invalid_argument("federal_reserve_release_month_invalid");
    return static_cast<unsigned>(std::distance(months.begin(), found) + 1);
}

std::string DateText(int year, unsigned month, unsigned day)
{
    const std::chrono::year_month_day date{
        std::chrono::year{year},
        std::chrono::month{month},
        std::chrono::day{day}};
    if (!date.ok())
        throw std::invalid_argument("federal_reserve_release_date_invalid");

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
        throw std::invalid_argument("federal_reserve_release_date_invalid");

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
        throw std::invalid_argument(
            "federal_reserve_release_civil_timestamp_invalid");
    return std::chrono::duration_cast<std::chrono::microseconds>(
        instant.time_since_epoch()).count();
}

std::string NextDate(const std::string& dateText)
{
    const ParsedDate parsed = ParseLongDate(
        [&]
        {
            std::tm fields{};
            if (strptime(dateText.c_str(), "%Y-%m-%d", &fields) == nullptr)
                throw std::invalid_argument("federal_reserve_release_date_invalid");
            static const std::array<const char*, 12> names{
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"};
            return std::string{names[static_cast<std::size_t>(fields.tm_mon)]} +
                " " + std::to_string(fields.tm_mday) + ", " +
                std::to_string(fields.tm_year + 1900);
        }());
    const std::chrono::year_month_day next{
        std::chrono::sys_days{
            std::chrono::year{parsed.year} /
            std::chrono::month{parsed.month} /
            std::chrono::day{parsed.day}} + std::chrono::days{1}};
    return DateText(
        static_cast<int>(next.year()),
        static_cast<unsigned>(next.month()),
        static_cast<unsigned>(next.day()));
}

struct SourceIdentity
{
    std::string id;
    std::optional<std::string> urlDate;
    bool beigeBook = false;
};

SourceIdentity ParseSourceIdentity(const std::string& url)
{
    std::smatch match;
    static const std::regex pressRelease{
        R"(^https://www\.federalreserve\.gov/newsevents/pressreleases/monetary([0-9]{4})([0-9]{2})([0-9]{2})([a-z])\.htm$)"};
    if (std::regex_match(url, match, pressRelease))
    {
        const std::string date = DateText(
            std::stoi(match[1].str()),
            static_cast<unsigned>(std::stoi(match[2].str())),
            static_cast<unsigned>(std::stoi(match[3].str())));
        return {
            "federal_reserve:monetary" + match[1].str() + match[2].str() +
                match[3].str() + match[4].str(),
            date,
            false};
    }

    static const std::vector<std::regex> beigePatterns{
        std::regex{R"(^https://www\.federalreserve\.gov/fomc/beigebook/([0-9]{4})/([0-9]{8})/fullreport\2\.pdf$)"},
        std::regex{R"(^https://www\.federalreserve\.gov/monetarypolicy/beigebook/files/(?:fullreport|Beige[Bb]ook_)([0-9]{8})\.pdf$)"},
        std::regex{R"(^https://www\.federalreserve\.gov/monetarypolicy/files/Beige[Bb]ook_([0-9]{8})\.pdf$)"},
        std::regex{R"(^https://www\.federalreserve\.gov/monetarypolicy/beigebook([0-9]{6})\.htm$)"},
        std::regex{R"(^https://www\.federalreserve\.gov/fomc/beigebook/([0-9]{4})/([0-9]{8})/default\.htm$)"},
        std::regex{R"(^https://www\.federalreserve\.gov/publications/files/BeigeBook_([0-9]{8})\.pdf$)"}};
    for (std::size_t index = 0; index < beigePatterns.size(); ++index)
    {
        if (!std::regex_match(url, match, beigePatterns[index]))
            continue;
        std::string compact;
        if (index == 0 || index == 4)
        {
            compact = match[2].str();
            if (compact.substr(0, 4) != match[1].str())
                throw std::invalid_argument(
                    "federal_reserve_beige_book_url_identity_contradiction");
        }
        else
        {
            compact = match[1].str();
        }
        const std::string date = compact.size() == 8
            ? DateText(
                std::stoi(compact.substr(0, 4)),
                static_cast<unsigned>(std::stoi(compact.substr(4, 2))),
                static_cast<unsigned>(std::stoi(compact.substr(6, 2))))
            : std::string{};
        return {
            "federal_reserve:beigebook-" + compact,
            date.empty() ? std::nullopt : std::optional<std::string>{date},
            true};
    }
    throw std::invalid_argument(
        "federal_reserve_source_url_not_canonical_first_party");
}

struct PublicationTime
{
    std::optional<std::string> releaseTime;
    std::string confidence;
    std::int64_t instant = 0;
};

PublicationTime ParsePublicationTime(
    const std::string& text,
    const std::string& releaseDate,
    bool beigeBook)
{
    static const std::regex explicitTime{
        R"((?:For release|For use) at\s+([0-9]{1,2})(?::([0-9]{2}))?\s*([ap])\.?m\.?,?\s*(E\.?[DS]\.?T\.?|Eastern))",
        std::regex::icase};
    std::smatch match;
    if (std::regex_search(text, match, explicitTime))
    {
        int hour = std::stoi(match[1].str());
        const int minute = match[2].matched ? std::stoi(match[2].str()) : 0;
        const std::string meridiem = Lower(match[3].str());
        std::string zone = Lower(match[4].str());
        zone.erase(std::remove(zone.begin(), zone.end(), '.'), zone.end());
        if (hour < 1 || hour > 12 || minute < 0 || minute > 59)
            throw std::invalid_argument(
                "federal_reserve_authoritative_release_time_invalid");
        if (meridiem == "a")
            hour = hour == 12 ? 0 : hour;
        else
            hour = hour == 12 ? 12 : hour + 12;

        std::ostringstream output;
        output << std::setfill('0') << std::setw(2) << hour << ':'
               << std::setw(2) << minute << ":00";
        const std::string releaseTime = output.str();
        const std::int64_t instant = UnixMicros(releaseDate + " " + releaseTime);

        if (zone == "edt" || zone == "est")
        {
            const auto parsed = [&]
            {
                std::tm result{};
                if (strptime(releaseDate.c_str(), "%Y-%m-%d", &result) == nullptr)
                    throw std::invalid_argument("federal_reserve_release_date_invalid");
                return result;
            }();
            std::tm civil = parsed;
            civil.tm_hour = hour;
            civil.tm_min = minute;
            const std::time_t naive = timegm(&civil);
            const std::int64_t offsetSeconds =
                instant / 1000000 - static_cast<std::int64_t>(naive);
            if ((zone == "edt" && offsetSeconds != 4 * 60 * 60) ||
                (zone == "est" && offsetSeconds != 5 * 60 * 60))
            {
                throw std::invalid_argument(
                    "federal_reserve_release_timezone_contradiction");
            }
        }
        return {releaseTime, "exact", instant};
    }

    static const std::regex unsupportedZone{
        R"((?:For release|For use) at.{0,60}\b(?:C[DS]T|M[DS]T|P[DS]T|UTC|GMT)\b)",
        std::regex::icase};
    if (std::regex_search(text, unsupportedZone))
        throw std::invalid_argument(
            "federal_reserve_release_timezone_unsupported");

    static const std::regex explicitMarker{
        R"((?:For release|For use) at\b)", std::regex::icase};
    static const std::regex immediateMarker{
        R"(For immediate release\b)", std::regex::icase};
    if (std::regex_search(text, explicitMarker))
        throw std::invalid_argument(
            "federal_reserve_authoritative_release_time_invalid");
    if (!beigeBook && !std::regex_search(text, immediateMarker))
        throw std::invalid_argument(
            "federal_reserve_authoritative_release_time_missing");

    // Shared Phase-2 semantics use the next New York civil midnight as the
    // conservative causal boundary for a publication known only by date.
    return {
        std::nullopt,
        "date_only",
        UnixMicros(NextDate(releaseDate) + " 00:00:00")};
}

std::string ExtractReleaseDate(
    const std::string& text,
    const std::string& parserType)
{
    std::regex pattern;
    if (parserType == "fomc_statement")
    {
        pattern = std::regex{
            R"(([A-Za-z]+ [0-9]{1,2}, [0-9]{4})\s+(?:Federal Reserve issues\s+)?FOMC statement\b)",
            std::regex::icase};
    }
    else if (parserType == "fomc_minutes")
    {
        pattern = std::regex{
            R"(([A-Za-z]+ [0-9]{1,2}, [0-9]{4})\s+Minutes of (?:the )?Federal Open Market Committee[, ])",
            std::regex::icase};
    }
    else
    {
        const std::regex titleDate{
            R"(Beige Book\s*(?:-|:)\s*([A-Za-z]+ [0-9]{1,2}, [0-9]{4}))",
            std::regex::icase};
        std::smatch titleMatch;
        if (std::regex_search(text, titleMatch, titleDate))
        {
            const ParsedDate parsed = ParseLongDate(titleMatch[1].str());
            return DateText(parsed.year, parsed.month, parsed.day);
        }
        pattern = std::regex{
            R"(For use at[^\r\n]{0,80}\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s*((?:January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{1,2}, [0-9]{4}))",
            std::regex::icase};
    }

    std::smatch match;
    if (!std::regex_search(text, match, pattern))
        throw std::invalid_argument("federal_reserve_publication_date_missing");
    const ParsedDate parsed = ParseLongDate(match[1].str());
    return DateText(parsed.year, parsed.month, parsed.day);
}

std::string MeetingReferenceFromMinutes(const std::string& text)
{
    static const std::regex title{
        R"(Minutes of (?:the )?Federal Open Market Committee,\s*([A-Za-z]+)\s+([0-9]{1,2})(?:\s*-\s*([0-9]{1,2}))?,\s*([0-9]{4}))",
        std::regex::icase};
    std::smatch match;
    if (!std::regex_search(text, match, title))
    {
        static const std::regex complexTitle{
            R"((?:Minutes of (?:the )?Federal Open Market Committee,|Committee meeting held on)\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-9]{1,2}(?:\s*(?:-|–|—)\s*(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+)?[0-9]{1,2})?(?:\s+and\s+(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+)?[0-9]{1,2}(?:\s*(?:-|–|—)\s*[0-9]{1,2})?)?,\s*[0-9]{4}))",
            std::regex::icase};
        if (!std::regex_search(text, match, complexTitle))
            throw std::invalid_argument(
                "federal_reserve_minutes_meeting_association_missing");
        std::string association = match[1].str();
        association = std::regex_replace(
            association, std::regex{R"(\s+)"}, " ");
        association = std::regex_replace(
            association, std::regex{R"(\s*(?:–|—)\s*)"}, "-");
        return "meetings " + association;
    }
    const unsigned month = MonthNumber(match[1].str());
    const int year = std::stoi(match[4].str());
    const std::string start = DateText(
        year, month, static_cast<unsigned>(std::stoi(match[2].str())));
    if (!match[3].matched)
        return "meeting " + start;
    const std::string end = DateText(
        year, month, static_cast<unsigned>(std::stoi(match[3].str())));
    return "meeting " + start + "/" + end;
}

void ValidateClassification(
    const std::string& text,
    const std::string& parserType,
    bool beigeUrl)
{
    if (parserType != "fomc_statement" && parserType != "fomc_minutes" &&
        parserType != "beige_book")
    {
        throw std::invalid_argument(
            "federal_reserve_parser_type_unsupported");
    }
    const bool statement = std::regex_search(
        text,
        std::regex{
            R"([A-Za-z]+ [0-9]{1,2}, [0-9]{4}\s+(?:Federal Reserve issues\s+)?FOMC statement\b)",
                   std::regex::icase});
    const bool minutes = std::regex_search(
        text,
        std::regex{
            R"([A-Za-z]+ [0-9]{1,2}, [0-9]{4}\s+Minutes of (?:the )?Federal Open Market Committee[, ])",
                   std::regex::icase});
    const bool beigeTitle = std::regex_search(
        text,
        std::regex{R"(\b(?:The\s+)?Beige Book\b)", std::regex::icase});
    const bool beige = beigeTitle && (beigeUrl || std::regex_search(
        text, std::regex{R"(For use at\b)", std::regex::icase}));
    const int count = static_cast<int>(statement) + static_cast<int>(minutes) +
        static_cast<int>(beige);
    if (count != 1)
        throw std::invalid_argument(
            "federal_reserve_artifact_classification_ambiguous");
    if ((parserType == "fomc_statement" && (!statement || beigeUrl)) ||
        (parserType == "fomc_minutes" && (!minutes || beigeUrl)) ||
        (parserType == "beige_book" && (!beige || !beigeUrl)))
    {
        throw std::invalid_argument(
            "federal_reserve_artifact_classification_mismatch");
    }
}

} // namespace


AuthoritativeEconomicEventCandidate
ParseFederalReserveEconomicReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl,
    const std::string& parserType)
{
    const SourceIdentity identity = ParseSourceIdentity(canonicalSourceUrl);
    const std::string text = VisibleText(artifact);
    ValidateClassification(text, parserType, identity.beigeBook);
    std::string releaseDate;
    try
    {
        releaseDate = ExtractReleaseDate(text, parserType);
    }
    catch (const std::invalid_argument& error)
    {
        if (parserType != "beige_book" || !identity.urlDate ||
            std::string{error.what()} != "federal_reserve_publication_date_missing")
        {
            throw;
        }
        // Modern Beige Book PDFs establish the publication only to a month on
        // their cover.  The immutable occurrence URL and official archive
        // index establish the day; absent a clock time, date_only remains the
        // conservative causal representation.
        releaseDate = *identity.urlDate;
    }
    if (identity.urlDate && *identity.urlDate != releaseDate)
        throw std::invalid_argument(
            "federal_reserve_source_url_release_date_mismatch");

    const PublicationTime publication = ParsePublicationTime(
        text, releaseDate, parserType == "beige_book");

    AuthoritativeEconomicEventCandidate candidate;
    candidate.currency = "USD";
    candidate.eventFamily = parserType == "fomc_statement"
        ? "FOMC_STATEMENT"
        : parserType == "fomc_minutes" ? "FOMC_MINUTES" : "BEIGE_BOOK";
    candidate.eventTimestampUnixMicros = publication.instant;
    candidate.sourceAgency = "FEDERAL_RESERVE";
    candidate.sourceEventId = identity.id;
    candidate.sourceUrl = canonicalSourceUrl;
    if (parserType == "fomc_statement")
        candidate.referencePeriod = "meeting ending " + releaseDate;
    else if (parserType == "fomc_minutes")
        candidate.referencePeriod = MeetingReferenceFromMinutes(text);
    else
        candidate.referencePeriod = "publication " + releaseDate;
    candidate.eventImportance = parserType == "beige_book"
        ? kBeigeBookImportance
        : kFomcImportance;
    candidate.historicalTimeConfidence = publication.confidence;
    candidate.sourceReleaseDate = releaseDate;
    candidate.sourceReleaseTime = publication.releaseTime;
    candidate.sourceTimezone = "America/New_York";
    return candidate;
}


std::vector<AuthoritativeEconomicEventCandidate>
LoadFederalReserveEconomicReleaseManifest(
    const std::filesystem::path& manifestPath)
{
    const auto canonicalManifest = std::filesystem::weakly_canonical(manifestPath);
    const auto root = canonicalManifest.parent_path();
    std::ifstream input{canonicalManifest};
    if (!input)
        throw std::invalid_argument("federal_reserve_manifest_missing");

    std::string line;
    if (!std::getline(input, line) || line != "manifest_version\t1")
        throw std::invalid_argument("federal_reserve_manifest_version_invalid");
    if (!std::getline(input, line) ||
        line != "parser_version\tfederal_reserve_economic_release_v2")
    {
        throw std::invalid_argument(
            "federal_reserve_manifest_parser_version_invalid");
    }
    if (!std::getline(input, line) ||
        line != "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
                "source_artifact_path\tsource_artifact_sha256\textractor")
    {
        throw std::invalid_argument("federal_reserve_manifest_columns_invalid");
    }

    std::vector<AuthoritativeEconomicEventCandidate> candidates;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const auto fields = SplitTabs(line);
        if (fields.size() != 7)
            throw std::invalid_argument("federal_reserve_manifest_entry_invalid");

        const auto artifactPath = ResolveManifestPath(root, fields[0]);
        std::string artifact;
        VerifyHash(artifactPath, fields[1], &artifact);

        std::string parserType;
        const std::string& type = fields[2];
        if (type == "fomc_statement_html")
            parserType = "fomc_statement";
        else if (type == "fomc_minutes_html")
            parserType = "fomc_minutes";
        else if (type == "beige_book_pdf_text")
            parserType = "beige_book";
        else if (type == "fomc_statement_text_fixture" ||
                 type == "fomc_minutes_text_fixture" ||
                 type == "beige_book_text_fixture")
        {
            parserType = type.substr(0, type.find("_text_fixture"));
        }
        else
        {
            throw std::invalid_argument(
                "federal_reserve_manifest_artifact_type_unsupported");
        }

        if (type.ends_with("_html"))
        {
            if (fields[4] != fields[0] || fields[5] != fields[1] ||
                fields[6] != "none")
            {
                throw std::invalid_argument(
                    "federal_reserve_manifest_html_provenance_invalid");
            }
        }
        else if (type == "beige_book_pdf_text")
        {
            if (fields[4] == "-" || fields[6].rfind("pdftotext-", 0) != 0)
                throw std::invalid_argument(
                    "federal_reserve_manifest_pdf_provenance_invalid");
            VerifyHash(ResolveManifestPath(root, fields[4]), fields[5]);
        }
        else if (fields[4] != fields[0] || fields[5] != fields[1] ||
                 fields[6] != "first_party_excerpt_v1")
        {
            throw std::invalid_argument(
                "federal_reserve_manifest_fixture_provenance_invalid");
        }

        candidates.push_back(ParseFederalReserveEconomicReleaseArtifact(
            artifact, fields[3], parserType));
    }

    if (!input.eof())
        throw std::runtime_error("federal_reserve_manifest_read_failed");
    if (candidates.empty())
        throw std::invalid_argument("federal_reserve_manifest_empty");
    return candidates;
}

} // namespace EA::EconomicCalendar
