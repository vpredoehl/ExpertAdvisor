#include "DolEtaWeeklyClaimsAdapter.hpp"

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
#include <vector>

namespace EA::EconomicCalendar
{
namespace
{

constexpr int kWeeklyClaimsImportance = 3;

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        throw std::invalid_argument("dol_eta_manifest_artifact_missing:" + path.string());

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof())
        throw std::runtime_error("dol_eta_manifest_artifact_read_failed:" + path.string());
    return contents.str();
}

std::string Sha256(const std::string& content)
{
    if (content.size() > std::numeric_limits<CC_LONG>::max())
        throw std::invalid_argument("dol_eta_manifest_artifact_too_large");

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
        throw std::invalid_argument("dol_eta_manifest_artifact_path_invalid");

    const auto resolved = std::filesystem::weakly_canonical(root / relative);
    if (!IsInside(root, resolved))
        throw std::invalid_argument("dol_eta_manifest_artifact_path_escapes_root");
    return resolved;
}

void VerifyHash(
    const std::filesystem::path& path,
    const std::string& expected,
    std::string* content = nullptr)
{
    static const std::regex digestPattern{R"(^[0-9a-f]{64}$)"};
    if (!std::regex_match(expected, digestPattern))
        throw std::invalid_argument("dol_eta_manifest_sha256_invalid");

    const std::string bytes = ReadFile(path);
    if (Sha256(bytes) != expected)
        throw std::invalid_argument("dol_eta_manifest_sha256_mismatch:" + path.string());
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

unsigned MonthNumber(const std::string& month)
{
    static const std::array<const char*, 12> months{
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"};
    std::string lower;
    for (unsigned char character : month)
    {
        if (character != '.')
            lower.push_back(static_cast<char>(std::tolower(character)));
    }

    const auto found = std::find_if(
        months.begin(), months.end(),
        [&lower](const char* candidate)
        {
            const std::string full{candidate};
            return lower == full ||
                (lower.size() == 3 && full.rfind(lower, 0) == 0) ||
                (lower == "sept" && full == "september");
        });
    if (found == months.end())
        throw std::invalid_argument("dol_eta_release_month_invalid");
    return static_cast<unsigned>(std::distance(months.begin(), found) + 1);
}

std::string DateText(int year, unsigned month, unsigned day)
{
    const std::chrono::year_month_day date{
        std::chrono::year{year},
        std::chrono::month{month},
        std::chrono::day{day}};
    if (!date.ok())
        throw std::invalid_argument("dol_eta_release_date_invalid");

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
        R"(^([A-Za-z]+\.?)\s+([0-9]{1,2})\s*,?\s*([0-9]{4})$)"};
    std::smatch match;
    if (!std::regex_match(text, match, pattern))
        throw std::invalid_argument("dol_eta_release_date_invalid");

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
        throw std::invalid_argument("dol_eta_release_civil_timestamp_invalid");
    return std::chrono::duration_cast<std::chrono::microseconds>(
        instant.time_since_epoch()).count();
}

std::string ArtifactFallbackId(
    const std::string& url,
    const std::string& releaseDate)
{
    static const std::regex pattern{
        R"(^https://oui\.doleta\.gov/press/([0-9]{4})/([0-9]{6})\.(asp|pdf)$)",
        std::regex::icase};
    std::smatch match;
    if (!std::regex_match(url, match, pattern))
        throw std::invalid_argument("dol_eta_authoritative_identity_missing");

    const std::string compact = match[2].str();
    if (compact.substr(4, 2) != match[1].str().substr(2, 2))
        throw std::invalid_argument("dol_eta_artifact_identity_year_mismatch");
    const std::string urlDate =
        match[1].str() + "-" + compact.substr(0, 2) + "-" + compact.substr(2, 2);
    if (urlDate != releaseDate)
        throw std::invalid_argument("dol_eta_artifact_identity_date_mismatch");

    std::string extension = match[3].str();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });
    return "dol_eta:artifact-" + match[1].str() + "-" + compact + "-" + extension;
}

} // namespace


std::string NormalizeDolEtaReleaseId(const std::string& releaseId)
{
    const std::string separatedPrefix = std::regex_replace(
        releaseId,
        std::regex{R"(^(\s*USDL)([0-9]))", std::regex::icase},
        "$1 $2");
    static const std::regex tokenPattern{R"([A-Za-z0-9]+)"};
    std::vector<std::string> tokens;
    for (std::sregex_iterator iterator{
             separatedPrefix.begin(), separatedPrefix.end(), tokenPattern}, end;
         iterator != end;
         ++iterator)
    {
        tokens.push_back(iterator->str());
    }

    if (tokens.size() != 4 && tokens.size() != 5)
        throw std::invalid_argument("dol_eta_release_id_malformed");

    std::transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });
    std::transform(tokens.back().begin(), tokens.back().end(), tokens.back().begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });

    const auto allDigits = [](const std::string& value)
    {
        return !value.empty() && std::all_of(
            value.begin(), value.end(),
            [](unsigned char character)
            {
                return std::isdigit(character) != 0;
            });
    };
    const auto allLetters = [](const std::string& value)
    {
        return !value.empty() && std::all_of(
            value.begin(), value.end(),
            [](unsigned char character)
            {
                return std::isalpha(character) != 0;
            });
    };

    const std::size_t releaseNumberIndex = tokens.size() - 2;
    const std::size_t suffixIndex = tokens.size() - 1;
    const bool duplicatedYearToken =
        tokens.size() == 5 && tokens[1] == tokens[2];
    if (tokens[0] != "usdl" ||
        (tokens[1].size() != 2 && tokens[1].size() != 4) ||
        !allDigits(tokens[1]) ||
        (tokens.size() == 5 && !duplicatedYearToken) ||
        tokens[releaseNumberIndex].size() > 6 ||
        !allDigits(tokens[releaseNumberIndex]) ||
        tokens[suffixIndex].size() < 2 || tokens[suffixIndex].size() > 10 ||
        !allLetters(tokens[suffixIndex]))
    {
        throw std::invalid_argument("dol_eta_release_id_malformed");
    }

    std::string normalized = "dol_eta:" + tokens[0];
    for (std::size_t index = 1; index < tokens.size(); ++index)
        normalized += '-' + tokens[index];
    return normalized;
}


AuthoritativeEconomicEventCandidate ParseDolEtaWeeklyClaimsArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl)
{
    static const std::regex sourceUrlPattern{
        R"(^https://oui\.doleta\.gov/press/[0-9]{4}/[0-9]{6}\.(asp|pdf)$)",
        std::regex::icase};
    if (!std::regex_match(canonicalSourceUrl, sourceUrlPattern))
        throw std::invalid_argument("dol_eta_source_url_not_canonical_first_party");

    const std::string text = VisibleText(artifact);
    static const std::regex titlePattern{
        R"(UNEMPLOYMENT INSURANCE WEEKLY CLAIMS(?: REPORT)?)",
        std::regex::icase};
    if (!std::regex_search(text, titlePattern))
        throw std::invalid_argument("dol_eta_weekly_claims_title_missing");

    static const std::regex embargoPattern{
        R"(EMBARGOED UNTIL.{0,500}?([0-9]{1,2}):([0-9]{2})\s*([AP])\.?M\.?\s*\(([^)]+)\).{0,250}?([A-Za-z]+\.?\s+[0-9]{1,2}\s*,?\s*[0-9]{4}))",
        std::regex::icase};
    std::smatch embargo;
    if (!std::regex_search(text, embargo, embargoPattern))
        throw std::invalid_argument("dol_eta_authoritative_release_time_missing");

    int hour = std::stoi(embargo[1].str());
    const int minute = std::stoi(embargo[2].str());
    std::string meridiem = embargo[3].str();
    std::string zone = embargo[4].str();
    std::transform(meridiem.begin(), meridiem.end(), meridiem.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::toupper(character));
        });
    std::transform(zone.begin(), zone.end(), zone.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::toupper(character));
        });
    if (hour < 1 || hour > 12 || minute < 0 || minute > 59)
        throw std::invalid_argument("dol_eta_authoritative_release_time_invalid");
    if (meridiem == "A")
        hour = hour == 12 ? 0 : hour;
    else
        hour = hour == 12 ? 12 : hour + 12;

    if (zone != "EDT" && zone != "EST" && zone != "EASTERN")
        throw std::invalid_argument("dol_eta_release_timezone_unsupported");

    const ParsedDate release = ParseLongDate(embargo[5].str());
    const std::string releaseDate =
        DateText(release.year, release.month, release.day);
    std::ostringstream timeOutput;
    timeOutput << std::setfill('0') << std::setw(2) << hour << ':'
               << std::setw(2) << minute << ":00";
    const std::string releaseTime = timeOutput.str();
    const std::int64_t instant = UnixMicros(releaseDate + " " + releaseTime);

    // EDT/EST are explicit offset claims.  "Eastern" delegates DST selection
    // to the historical New York civil-time helper.  The 2011-2012 Weekly
    // Claims archive contains a demonstrated publisher defect where otherwise
    // authoritative 8:30 A.M. releases use the wrong EST/EDT abbreviation.
    // Preserve the authoritative local clock time in that narrow case, resolve
    // it through America/New_York, and downgrade provenance to reconstructed.
    bool reconstructedTime = false;
    if (zone == "EDT" || zone == "EST")
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
        const bool contradiction =
            (zone == "EDT" && offsetSeconds != 4 * 60 * 60) ||
            (zone == "EST" && offsetSeconds != 5 * 60 * 60);
        if (contradiction)
        {
            const bool documentedDolArchiveDefect =
                (release.year == 2011 || release.year == 2012) &&
                hour == 8 && minute == 30;
            if (!documentedDolArchiveDefect)
                throw std::invalid_argument("dol_eta_release_timezone_contradiction");
            reconstructedTime = true;
        }
    }

    static const std::regex releaseIdPattern{
        R"(\bUSDL[^A-Za-z]*[0-9]{2,4}(?:[^A-Za-z0-9]+[0-9]{2,4})?[^A-Za-z0-9]+[0-9]{1,6}[^A-Za-z0-9]+[A-Za-z]{2,10}\b)",
        std::regex::icase};
    static const std::regex usdlMarker{R"(\bUSDL\b)", std::regex::icase};
    std::smatch releaseIdMatch;
    std::string sourceEventId;
    if (std::regex_search(text, releaseIdMatch, releaseIdPattern))
    {
        sourceEventId = NormalizeDolEtaReleaseId(releaseIdMatch.str());
        // DOL assigned USDL 16-567-NAT to two distinct weekly releases.
        // Preserve the authoritative number normally, but use each immutable
        // occurrence URL for this demonstrated publisher-identity collision.
        if (sourceEventId == "dol_eta:usdl-16-567-nat")
            sourceEventId = ArtifactFallbackId(canonicalSourceUrl, releaseDate);
    }
    else if (std::regex_search(text, usdlMarker))
        throw std::invalid_argument("dol_eta_release_id_malformed");
    else
        sourceEventId = ArtifactFallbackId(canonicalSourceUrl, releaseDate);

    static const std::regex referencePattern{
        R"(In the week ending\s+([A-Za-z]+\.?)\s+([0-9]{1,2})(?:,\s*([0-9]{4}))?)",
        std::regex::icase};
    std::smatch referenceMatch;
    if (!std::regex_search(text, referenceMatch, referencePattern))
        throw std::invalid_argument("dol_eta_reference_week_missing");

    int referenceYear = referenceMatch[3].matched
        ? std::stoi(referenceMatch[3].str())
        : release.year;
    const unsigned referenceMonth = MonthNumber(referenceMatch[1].str());
    const unsigned referenceDay =
        static_cast<unsigned>(std::stoi(referenceMatch[2].str()));
    std::chrono::year_month_day referenceDate{
        std::chrono::year{referenceYear},
        std::chrono::month{referenceMonth},
        std::chrono::day{referenceDay}};
    const std::chrono::sys_days releaseDays{
        std::chrono::year{release.year} /
        std::chrono::month{release.month} /
        std::chrono::day{release.day}};
    if (!referenceMatch[3].matched && referenceDate.ok() &&
        std::chrono::sys_days{referenceDate} > releaseDays)
    {
        --referenceYear;
        referenceDate = std::chrono::year{referenceYear} /
            std::chrono::month{referenceMonth} /
            std::chrono::day{referenceDay};
    }
    if (!referenceDate.ok() || std::chrono::sys_days{referenceDate} > releaseDays ||
        releaseDays - std::chrono::sys_days{referenceDate} > std::chrono::days{14})
    {
        throw std::invalid_argument("dol_eta_reference_week_invalid");
    }

    AuthoritativeEconomicEventCandidate candidate;
    candidate.currency = "USD";
    candidate.eventFamily = "WEEKLY_CLAIMS";
    candidate.eventTimestampUnixMicros = instant;
    candidate.sourceAgency = "DOL_ETA";
    candidate.sourceEventId = sourceEventId;
    candidate.sourceUrl = canonicalSourceUrl;
    candidate.referencePeriod = "week ending " +
        DateText(referenceYear, referenceMonth, referenceDay);
    candidate.eventImportance = kWeeklyClaimsImportance;
    candidate.historicalTimeConfidence = reconstructedTime
        ? "reconstructed"
        : "exact";
    candidate.sourceReleaseDate = releaseDate;
    candidate.sourceReleaseTime = releaseTime;
    candidate.sourceTimezone = "America/New_York";
    return candidate;
}


std::vector<AuthoritativeEconomicEventCandidate>
LoadDolEtaWeeklyClaimsManifest(
    const std::filesystem::path& manifestPath)
{
    const auto canonicalManifest = std::filesystem::weakly_canonical(manifestPath);
    const auto root = canonicalManifest.parent_path();
    std::ifstream input{canonicalManifest};
    if (!input)
        throw std::invalid_argument("dol_eta_manifest_missing");

    std::string line;
    if (!std::getline(input, line) || line != "manifest_version\t1")
        throw std::invalid_argument("dol_eta_manifest_version_invalid");
    if (!std::getline(input, line) ||
        line != "parser_version\tdol_eta_weekly_claims_v3")
    {
        throw std::invalid_argument("dol_eta_manifest_parser_version_invalid");
    }
    if (!std::getline(input, line) ||
        line != "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
                "source_artifact_path\tsource_artifact_sha256\textractor")
    {
        throw std::invalid_argument("dol_eta_manifest_columns_invalid");
    }

    std::vector<AuthoritativeEconomicEventCandidate> candidates;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const auto fields = SplitTabs(line);
        if (fields.size() != 7)
            throw std::invalid_argument("dol_eta_manifest_entry_invalid");

        const auto artifactPath = ResolveManifestPath(root, fields[0]);
        std::string artifact;
        VerifyHash(artifactPath, fields[1], &artifact);

        const std::string& type = fields[2];
        const std::string& sourcePathText = fields[4];
        const std::string& sourceHash = fields[5];
        const std::string& extractor = fields[6];
        if (type == "html")
        {
            if (sourcePathText != fields[0] || sourceHash != fields[1] ||
                extractor != "none")
            {
                throw std::invalid_argument("dol_eta_manifest_html_provenance_invalid");
            }
        }
        else if (type == "pdf_text")
        {
            if (sourcePathText == "-" ||
                extractor.rfind("pdftotext-", 0) != 0)
            {
                throw std::invalid_argument("dol_eta_manifest_pdf_provenance_invalid");
            }
            VerifyHash(ResolveManifestPath(root, sourcePathText), sourceHash);
        }
        else if (type == "text_fixture")
        {
            if (sourcePathText != fields[0] || sourceHash != fields[1] ||
                extractor != "first_party_excerpt_v1")
            {
                throw std::invalid_argument("dol_eta_manifest_fixture_provenance_invalid");
            }
        }
        else
        {
            throw std::invalid_argument("dol_eta_manifest_artifact_type_unsupported");
        }

        candidates.push_back(
            ParseDolEtaWeeklyClaimsArtifact(artifact, fields[3]));
    }

    if (!input.eof())
        throw std::runtime_error("dol_eta_manifest_read_failed");
    if (candidates.empty())
        throw std::invalid_argument("dol_eta_manifest_empty");
    return candidates;
}

} // namespace EA::EconomicCalendar
