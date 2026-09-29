#include "ControlledReplicationStudyArchive.hpp"

#include "ControlledReplicationStudySpecification.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace EA::ControlledReplicationStudyArchive
{
namespace
{

namespace Study = EA::ControlledReplicationStudy;

std::string ReadBytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("study_archive_artifact_missing");
    std::ostringstream bytes;
    bytes << input.rdbuf();
    if (!input.eof() && input.fail())
        throw std::runtime_error("study_archive_artifact_read_failed");
    return bytes.str();
}

std::string Escape(std::string_view value)
{
    std::ostringstream output;
    constexpr char hex[] = "0123456789ABCDEF";
    for (const unsigned char character : value)
    {
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '_' ||
            character == '-' || character == '.' || character == ':' ||
            character == '/')
            output << static_cast<char>(character);
        else
            output << '%' << hex[character >> 4] << hex[character & 0x0f];
    }
    return output.str();
}

int Hex(char character)
{
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

std::string Unescape(std::string_view value)
{
    std::string output;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        if (value[index] != '%')
        {
            output.push_back(value[index]);
            continue;
        }
        if (index + 2 >= value.size())
            throw std::runtime_error("study_archive_registry_bad_escape");
        const int high = Hex(value[index + 1]);
        const int low = Hex(value[index + 2]);
        if (high < 0 || low < 0)
            throw std::runtime_error("study_archive_registry_bad_escape");
        output.push_back(static_cast<char>((high << 4) | low));
        index += 2;
    }
    return output;
}

bool TaggedHash(const std::string& value)
{
    return value.size() == 24 && value.starts_with("fnv1a64:") &&
        std::all_of(value.begin() + 8, value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}

bool Sha256Hex(const std::string& value)
{
    return value.size() == 64 && std::all_of(
        value.begin(), value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}

long long Positive(std::string_view value, const char* reason)
{
    if (value.empty() || std::any_of(value.begin(), value.end(),
                                     [](unsigned char c) {
                                         return !std::isdigit(c);
                                     }))
        throw std::runtime_error(reason);
    std::size_t consumed = 0;
    const long long result = std::stoll(std::string(value), &consumed);
    if (consumed != value.size() || result <= 0)
        throw std::runtime_error(reason);
    return result;
}

std::vector<std::string> SplitTabs(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true)
    {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos) return fields;
        begin = end + 1;
    }
}

std::filesystem::path AbsoluteNormalized(
    const std::filesystem::path& path)
{
    return std::filesystem::absolute(path).lexically_normal();
}

std::filesystem::path ResolveWithinRoot(
    const std::filesystem::path& root,
    const std::filesystem::path& relative,
    const char* reason)
{
    if (relative.empty() || relative.is_absolute())
        throw std::runtime_error(reason);
    for (const auto& component : relative)
        if (component == "..") throw std::runtime_error(reason);
    const auto normalizedRoot = AbsoluteNormalized(root);
    const auto resolved = AbsoluteNormalized(normalizedRoot / relative);
    auto rootIt = normalizedRoot.begin();
    auto resolvedIt = resolved.begin();
    for (; rootIt != normalizedRoot.end(); ++rootIt, ++resolvedIt)
        if (resolvedIt == resolved.end() || *rootIt != *resolvedIt)
            throw std::runtime_error(reason);
    return resolved;
}

void RequireExistingPathWithinRoot(
    const std::filesystem::path& root,
    const std::filesystem::path& path,
    const char* reason)
{
    std::error_code error;
    const auto canonicalRoot = std::filesystem::weakly_canonical(root, error);
    if (error) throw std::runtime_error(reason);
    const auto canonicalPath = std::filesystem::weakly_canonical(path, error);
    if (error) throw std::runtime_error(reason);
    auto rootIt = canonicalRoot.begin();
    auto pathIt = canonicalPath.begin();
    for (; rootIt != canonicalRoot.end(); ++rootIt, ++pathIt)
        if (pathIt == canonicalPath.end() || *rootIt != *pathIt)
            throw std::runtime_error(reason);
}

std::string ArtifactRelativePath(const std::string& identityHash)
{
    if (!TaggedHash(identityHash))
        throw std::runtime_error("study_archive_identity_malformed");
    return "studies/study_" + identityHash.substr(8) + ".txt";
}

RegistryEntry ParseEntry(const std::string& line)
{
    const auto fields = SplitTabs(line);
    if (fields.size() != 8 || fields[0] != kRegistryRecordType)
        throw std::runtime_error("study_archive_registry_malformed");
    RegistryEntry entry;
    entry.studyIdentifier = Unescape(fields[1]);
    entry.studyIdentityHash = fields[2];
    entry.artifactPath = fields[3];
    entry.artifactSha256 = fields[4];
    entry.freezeTimestamp = Unescape(fields[5]);
    entry.studyType = Unescape(fields[6]);
    entry.studyVersion = static_cast<int>(
        Positive(fields[7], "study_archive_registry_version_malformed"));
    if (!TaggedHash(entry.studyIdentityHash) ||
        !Sha256Hex(entry.artifactSha256) || entry.studyIdentifier.empty() ||
        entry.studyType.empty() || entry.studyVersion <= 0 ||
        entry.freezeTimestamp.empty())
        throw std::runtime_error("study_archive_registry_malformed");
    return entry;
}

std::vector<RegistryEntry> LoadRegistry(
    const std::filesystem::path& registryPath,
    bool missingIsEmpty)
{
    std::ifstream input(registryPath, std::ios::binary);
    if (!input)
    {
        if (missingIsEmpty) return {};
        throw std::runtime_error("study_archive_registry_missing");
    }
    std::vector<RegistryEntry> entries;
    std::set<std::string> identities;
    std::set<std::string> identifiers;
    for (std::string line; std::getline(input, line);)
    {
        if (line.empty()) continue;
        const RegistryEntry entry = ParseEntry(line);
        if (!identities.insert(entry.studyIdentityHash).second ||
            !identifiers.insert(entry.studyIdentifier).second)
            throw std::runtime_error("study_archive_registry_duplicate");
        entries.push_back(entry);
    }
    if (!input.eof() && input.fail())
        throw std::runtime_error("study_archive_registry_read_failed");
    return entries;
}

RegistryEntry BuildEntry(const Study::Specification& specification,
                         const std::string& artifactSha256)
{
    RegistryEntry entry;
    entry.studyIdentifier = specification.studyIdentifier;
    entry.studyIdentityHash = specification.identityHash;
    entry.artifactPath = ArtifactRelativePath(specification.identityHash);
    entry.artifactSha256 = artifactSha256;
    entry.freezeTimestamp = specification.freezeTimestamp;
    entry.studyType = specification.studyType;
    entry.studyVersion = specification.version;
    return entry;
}

void VerifyArtifactBytes(const RegistryEntry& entry,
                         const std::filesystem::path& archiveRoot)
{
    const auto artifactPath = ResolveWithinRoot(
        archiveRoot, entry.artifactPath, "study_archive_artifact_path_invalid");
    RequireExistingPathWithinRoot(
        archiveRoot, artifactPath, "study_archive_artifact_path_invalid");
    const std::string content = ReadBytes(artifactPath);
    if (Sha256(content) != entry.artifactSha256)
        throw std::runtime_error("study_archive_artifact_sha256_mismatch");
    const auto specification = Study::Parse(content);
    const auto validation = Study::Validate(specification);
    if (!validation.valid)
        throw std::runtime_error("study_archive_embedded_identity_invalid");
    if (validation.identityHash != entry.studyIdentityHash ||
        specification.studyIdentifier != entry.studyIdentifier ||
        specification.freezeTimestamp != entry.freezeTimestamp ||
        specification.studyType != entry.studyType ||
        specification.version != entry.studyVersion)
        throw std::runtime_error("study_archive_registry_identity_mismatch");
    if (content != Study::Render(specification))
        throw std::runtime_error("study_archive_artifact_not_canonical");
}

void ValidateRegistryPaths(const std::vector<RegistryEntry>& entries,
                           const std::filesystem::path& archiveRoot)
{
    for (const auto& entry : entries)
    {
        const auto path = ResolveWithinRoot(
            archiveRoot, entry.artifactPath,
            "study_archive_artifact_path_invalid");
        if (!std::filesystem::exists(path))
            throw std::runtime_error("study_archive_artifact_missing");
        RequireExistingPathWithinRoot(
            archiveRoot, path, "study_archive_artifact_path_invalid");
    }
}

} // namespace

std::string Sha256(std::string_view bytes)
{
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const unsigned char byte : digest)
        output << std::setw(2) << static_cast<unsigned int>(byte);
    return output.str();
}

std::string RenderRegistryEntry(const RegistryEntry& entry)
{
    if (entry.registryVersion != kRegistryVersion ||
        !TaggedHash(entry.studyIdentityHash) ||
        !Sha256Hex(entry.artifactSha256) || entry.studyIdentifier.empty() ||
        entry.artifactPath.empty() || entry.freezeTimestamp.empty() ||
        entry.studyType.empty() || entry.studyVersion <= 0)
        throw std::invalid_argument("study_archive_registry_entry_invalid");
    std::ostringstream output;
    output << kRegistryRecordType << '\t'
           << Escape(entry.studyIdentifier) << '\t'
           << entry.studyIdentityHash << '\t'
           << entry.artifactPath << '\t'
           << entry.artifactSha256 << '\t'
           << Escape(entry.freezeTimestamp) << '\t'
           << Escape(entry.studyType) << '\t'
           << entry.studyVersion << '\n';
    return output.str();
}

RegistrationResult Register(const std::string& sourceArtifactPath,
                            const std::filesystem::path& archiveRoot,
                            const std::filesystem::path& registryPath)
{
    const auto normalizedRoot = AbsoluteNormalized(archiveRoot);
    const auto absoluteRegistry = AbsoluteNormalized(registryPath);
    const auto normalizedRegistry = ResolveWithinRoot(
        normalizedRoot, std::filesystem::relative(absoluteRegistry, normalizedRoot),
        "study_archive_registry_path_invalid");
    if (std::filesystem::exists(normalizedRegistry))
        RequireExistingPathWithinRoot(
            normalizedRoot, normalizedRegistry,
            "study_archive_registry_path_invalid");
    const std::string content = ReadBytes(sourceArtifactPath);
    const auto specification = Study::Parse(content);
    const auto validation = Study::Validate(specification);
    if (!validation.valid)
        throw std::runtime_error("study_archive_study_invalid:" + validation.reason);
    if (content != Study::Render(specification))
        throw std::runtime_error("study_archive_artifact_not_canonical");

    const std::string artifactSha256 = Sha256(content);
    const RegistryEntry expected = BuildEntry(specification, artifactSha256);
    const auto entries = LoadRegistry(normalizedRegistry, true);
    ValidateRegistryPaths(entries, normalizedRoot);
    for (const auto& existing : entries)
    {
        if (existing.studyIdentityHash == expected.studyIdentityHash ||
            existing.studyIdentifier == expected.studyIdentifier)
        {
            if (RenderRegistryEntry(existing) != RenderRegistryEntry(expected))
                throw std::runtime_error("study_archive_conflicting_duplicate");
            VerifyArtifactBytes(existing, normalizedRoot);
            return {existing, true, false, false};
        }
    }

    const auto artifactPath = ResolveWithinRoot(
        normalizedRoot, expected.artifactPath,
        "study_archive_artifact_path_invalid");
    RequireExistingPathWithinRoot(
        normalizedRoot, artifactPath, "study_archive_artifact_path_invalid");
    std::error_code error;
    std::filesystem::create_directories(artifactPath.parent_path(), error);
    if (error) throw std::runtime_error("study_archive_directory_create_failed");
    bool artifactWritten = false;
    if (std::filesystem::exists(artifactPath))
    {
        if (ReadBytes(artifactPath) != content)
            throw std::runtime_error("study_archive_artifact_conflict");
    }
    else
    {
        std::ofstream output(artifactPath, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("study_archive_artifact_write_failed");
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!output) throw std::runtime_error("study_archive_artifact_write_failed");
        artifactWritten = true;
    }

    std::filesystem::create_directories(normalizedRegistry.parent_path(), error);
    if (error) throw std::runtime_error("study_archive_directory_create_failed");
    RequireExistingPathWithinRoot(
        normalizedRoot, normalizedRegistry,
        "study_archive_registry_path_invalid");
    std::ofstream registry(normalizedRegistry, std::ios::binary | std::ios::app);
    if (!registry) throw std::runtime_error("study_archive_registry_write_failed");
    const std::string rendered = RenderRegistryEntry(expected);
    registry.write(rendered.data(), static_cast<std::streamsize>(rendered.size()));
    if (!registry) throw std::runtime_error("study_archive_registry_write_failed");
    return {expected, false, artifactWritten, true};
}

RegistryEntry Verify(const std::string& studyIdentityHash,
                     const std::filesystem::path& archiveRoot,
                     const std::filesystem::path& registryPath)
{
    if (!TaggedHash(studyIdentityHash))
        throw std::invalid_argument("study_archive_identity_malformed");
    const auto normalizedRoot = AbsoluteNormalized(archiveRoot);
    const auto absoluteRegistry = AbsoluteNormalized(registryPath);
    const auto normalizedRegistry = ResolveWithinRoot(
        normalizedRoot, std::filesystem::relative(absoluteRegistry, normalizedRoot),
        "study_archive_registry_path_invalid");
    if (std::filesystem::exists(normalizedRegistry))
        RequireExistingPathWithinRoot(
            normalizedRoot, normalizedRegistry,
            "study_archive_registry_path_invalid");
    const auto entries = LoadRegistry(normalizedRegistry, false);
    ValidateRegistryPaths(entries, normalizedRoot);
    const auto found = std::find_if(
        entries.begin(), entries.end(), [&](const RegistryEntry& entry) {
            return entry.studyIdentityHash == studyIdentityHash;
        });
    if (found == entries.end())
        throw std::runtime_error("study_archive_study_unregistered");
    VerifyArtifactBytes(*found, normalizedRoot);
    return *found;
}

namespace
{

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        if (character == ',' || character == '\n' || character == '\r' ||
            character == '\t') character = '_';
    }
    return value;
}

} // namespace

int RunFreezeCommand(const std::string& sourceArtifactPath,
                     const std::filesystem::path& archiveRoot,
                     const std::filesystem::path& registryPath,
                     std::ostream& output,
                     std::ostream& errors)
{
    try
    {
        const RegistrationResult result =
            Register(sourceArtifactPath, archiveRoot, registryPath);
        output << "CONTROLLED_REPLICATION_STUDY_FREEZE"
               << ",status=" << (result.alreadyRegistered ?
                                      "already_registered" : "registered")
               << ",study_identifier=" << MachineText(result.entry.studyIdentifier)
               << ",study_identity_hash=" << result.entry.studyIdentityHash
               << ",artifact_path=" << result.entry.artifactPath
               << ",artifact_sha256=" << result.entry.artifactSha256
               << ",freeze_timestamp=" << MachineText(result.entry.freezeTimestamp)
               << ",artifact_written=" << (result.artifactWritten ? "true" : "false")
               << ",registry_written=" << (result.registryWritten ? "true" : "false")
               << ",read_only=false\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "CONTROLLED_REPLICATION_STUDY_FREEZE_FAILED"
               << ",reason=" << MachineText(error.what())
               << ",read_only=false,exit_code=3\n";
        return 3;
    }
}

int RunVerifyCommand(const std::string& studyIdentityHash,
                     const std::filesystem::path& archiveRoot,
                     const std::filesystem::path& registryPath,
                     std::ostream& output,
                     std::ostream& errors)
{
    try
    {
        const RegistryEntry entry =
            Verify(studyIdentityHash, archiveRoot, registryPath);
        output << "CONTROLLED_REPLICATION_STUDY_ARCHIVE_VERIFICATION"
               << ",status=verified"
               << ",study_identifier=" << MachineText(entry.studyIdentifier)
               << ",study_identity_hash=" << entry.studyIdentityHash
               << ",artifact_path=" << entry.artifactPath
               << ",artifact_sha256=" << entry.artifactSha256
               << ",semantic_identity_verified=true"
               << ",artifact_bytes_verified=true"
               << ",read_only=true\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "CONTROLLED_REPLICATION_STUDY_ARCHIVE_VERIFICATION_FAILED"
               << ",reason=" << MachineText(error.what())
               << ",read_only=true,exit_code=3\n";
        return 3;
    }
}

} // namespace EA::ControlledReplicationStudyArchive
