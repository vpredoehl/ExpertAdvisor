#include "BlsScheduleReleaseAdapter.hpp"

#include "../Headers/HistoricalFxTimestamp.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace EA::EconomicCalendar
{
namespace
{

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream input{path, std::ios::binary};
    if (!input)
        throw std::invalid_argument("bls_manifest_artifact_missing:" + path.string());
    std::ostringstream output;
    output << input.rdbuf();
    if (!input.good() && !input.eof())
        throw std::runtime_error("bls_manifest_artifact_read_failed:" + path.string());
    return output.str();
}

std::string Sha256(const std::string& content)
{
    if (content.size() > std::numeric_limits<CC_LONG>::max())
        throw std::invalid_argument("bls_manifest_artifact_too_large");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(content.data(), static_cast<CC_LONG>(content.size()), digest.data());
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const unsigned char byte : digest)
        output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

std::vector<std::string> SplitTabs(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true)
    {
        const auto separator = line.find('\t', start);
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

std::filesystem::path Resolve(
    const std::filesystem::path& root,
    const std::string& text)
{
    const std::filesystem::path relative{text};
    if (relative.empty() || relative.is_absolute())
        throw std::invalid_argument("bls_manifest_artifact_path_invalid");
    const auto resolved = std::filesystem::weakly_canonical(root / relative);
    if (!IsInside(root, resolved))
        throw std::invalid_argument("bls_manifest_artifact_path_escapes_root");
    return resolved;
}

void VerifyHash(
    const std::filesystem::path& path,
    const std::string& expected,
    std::string* content = nullptr)
{
    static const std::regex digestPattern{R"(^[0-9a-f]{64}$)"};
    if (!std::regex_match(expected, digestPattern))
        throw std::invalid_argument("bls_manifest_sha256_invalid");
    const std::string bytes = ReadFile(path);
    if (Sha256(bytes) != expected)
        throw std::invalid_argument("bls_manifest_sha256_mismatch:" + path.string());
    if (content)
        *content = bytes;
}

std::string Slug(std::string value)
{
    std::string result;
    bool separator = false;
    for (const unsigned char character : value)
    {
        if (std::isalnum(character) != 0)
        {
            if (separator && !result.empty())
                result.push_back('-');
            result.push_back(static_cast<char>(std::tolower(character)));
            separator = false;
        }
        else
        {
            separator = true;
        }
    }
    if (result.empty())
        throw std::invalid_argument("bls_reference_period_invalid");
    return result;
}

std::int64_t NewYorkMicros(const std::string& civil)
{
    PriceTP timestamp;
    if (!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(civil, timestamp))
        throw std::invalid_argument("bls_release_timestamp_invalid");
    return std::chrono::duration_cast<std::chrono::microseconds>(
        timestamp.time_since_epoch()).count();
}

std::map<std::string, std::string> ParseFields(const std::string& artifact)
{
    std::istringstream input{artifact};
    std::string line;
    if (!std::getline(input, line) || line != "bls_schedule_row_version\t1")
        throw std::invalid_argument("bls_schedule_row_version_invalid");
    std::map<std::string, std::string> fields;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const auto values = SplitTabs(line);
        if (values.size() != 2 || values[0].empty() || values[1].empty() ||
            !fields.emplace(values[0], values[1]).second)
        {
            throw std::invalid_argument("bls_schedule_row_invalid");
        }
    }
    static const std::array<const char*, 7> required{
        "family", "release_date", "release_time", "title",
        "reference_period", "schedule_url", "source_event_id"};
    if (fields.size() != required.size() || std::any_of(
            required.begin(), required.end(),
            [&](const char* name) { return !fields.contains(name); }))
    {
        throw std::invalid_argument("bls_schedule_row_fields_invalid");
    }
    return fields;
}

std::string ExpectedTitle(const std::string& family)
{
    if (family == "CPI")
        return "Consumer Price Index";
    if (family == "EMPLOYMENT")
        return "Employment Situation";
    if (family == "EMPLOYMENT_ANNUAL")
        return "Employment Situation of Veterans";
    if (family == "JOLTS")
        return "Job Openings and Labor Turnover";
    if (family == "PPI")
        return "Producer Price Index";
    throw std::invalid_argument("bls_family_unsupported");
}

std::string FamilyIdentity(const std::string& family)
{
    std::string value = family;
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char character)
        {
            return character == '_'
                ? '-'
                : static_cast<char>(std::tolower(character));
        });
    return value;
}

} // namespace


AuthoritativeEconomicEventCandidate ParseBlsScheduleReleaseArtifact(
    const std::string& artifact,
    const std::string& canonicalSourceUrl)
{
    static const std::regex urlPattern{
        R"(^https://www\.bls\.gov/schedule/([0-9]{4})/home\.htm$)"};
    static const std::regex datePattern{R"(^([0-9]{4})-[0-9]{2}-[0-9]{2}$)"};
    static const std::regex timePattern{R"(^([0-9]{2}):([0-9]{2}):00$)"};
    std::smatch urlMatch;
    if (!std::regex_match(canonicalSourceUrl, urlMatch, urlPattern))
        throw std::invalid_argument("bls_source_url_not_canonical_first_party");

    const auto fields = ParseFields(artifact);
    if (fields.at("schedule_url") != canonicalSourceUrl)
        throw std::invalid_argument("bls_schedule_url_mismatch");
    std::smatch dateMatch;
    if (!std::regex_match(fields.at("release_date"), dateMatch, datePattern) ||
        dateMatch[1].str() != urlMatch[1].str() ||
        !std::regex_match(fields.at("release_time"), timePattern))
    {
        throw std::invalid_argument("bls_release_provenance_invalid");
    }
    const std::string expectedTitle = ExpectedTitle(fields.at("family"));
    const bool validJoltsVariant = fields.at("family") == "JOLTS" &&
        fields.at("title") == "Job Openings and Labor Turnover Survey";
    if (fields.at("title") != expectedTitle && !validJoltsVariant)
        throw std::invalid_argument("bls_title_family_mismatch");

    const std::string expectedIdentity =
        "bls:" + FamilyIdentity(fields.at("family")) + '-' +
        Slug(fields.at("reference_period"));
    if (fields.at("source_event_id") != expectedIdentity)
        throw std::invalid_argument("bls_source_event_id_mismatch");

    AuthoritativeEconomicEventCandidate candidate;
    candidate.currency = "USD";
    candidate.eventFamily = fields.at("family");
    candidate.eventTimestampUnixMicros = NewYorkMicros(
        fields.at("release_date") + ' ' + fields.at("release_time"));
    candidate.sourceAgency = "BLS";
    candidate.sourceEventId = expectedIdentity;
    candidate.sourceUrl = canonicalSourceUrl;
    candidate.referencePeriod = fields.at("reference_period");
    candidate.eventImportance = 3;
    candidate.historicalTimeConfidence = "exact";
    candidate.sourceReleaseDate = fields.at("release_date");
    candidate.sourceReleaseTime = fields.at("release_time");
    candidate.sourceTimezone = "America/New_York";
    return candidate;
}


std::vector<AuthoritativeEconomicEventCandidate>
LoadBlsScheduleReleaseManifest(const std::filesystem::path& manifestPath)
{
    const auto canonicalManifest = std::filesystem::weakly_canonical(manifestPath);
    const auto root = canonicalManifest.parent_path();
    std::ifstream input{canonicalManifest};
    if (!input)
        throw std::invalid_argument("bls_manifest_missing");
    std::string line;
    if (!std::getline(input, line) || line != "manifest_version\t1")
        throw std::invalid_argument("bls_manifest_version_invalid");
    if (!std::getline(input, line) ||
        line != "parser_version\tbls_schedule_release_v1")
    {
        throw std::invalid_argument("bls_manifest_parser_version_invalid");
    }
    if (!std::getline(input, line) ||
        line != "artifact_path\tartifact_sha256\tartifact_type\tsource_url\t"
                "source_artifact_path\tsource_artifact_sha256\textractor")
    {
        throw std::invalid_argument("bls_manifest_columns_invalid");
    }

    std::vector<AuthoritativeEconomicEventCandidate> candidates;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        const auto fields = SplitTabs(line);
        if (fields.size() != 7 || fields[2] != "bls_schedule_row" ||
            fields[4] == fields[0] || fields[6] != "bls_schedule_row_v1")
        {
            throw std::invalid_argument("bls_manifest_entry_invalid");
        }
        std::string artifact;
        VerifyHash(Resolve(root, fields[0]), fields[1], &artifact);
        VerifyHash(Resolve(root, fields[4]), fields[5]);
        candidates.push_back(ParseBlsScheduleReleaseArtifact(artifact, fields[3]));
    }
    if (!input.eof())
        throw std::runtime_error("bls_manifest_read_failed");
    if (candidates.empty())
        throw std::invalid_argument("bls_manifest_empty");
    return candidates;
}

} // namespace EA::EconomicCalendar
