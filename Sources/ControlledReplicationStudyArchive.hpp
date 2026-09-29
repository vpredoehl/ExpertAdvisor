#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>

namespace EA::ControlledReplicationStudyArchive
{

inline constexpr int kRegistryVersion = 1;
inline constexpr const char* kRegistryRecordType =
    "controlled_replication_study_registry_v1";
inline constexpr const char* kDefaultArchiveRoot =
    "docs/archive/controlled-replication";
inline constexpr const char* kDefaultRegistryPath =
    "docs/archive/controlled-replication/registry.tsv";

struct RegistryEntry
{
    int registryVersion = kRegistryVersion;
    std::string studyIdentifier;
    std::string studyIdentityHash;
    std::string artifactPath;
    std::string artifactSha256;
    std::string freezeTimestamp;
    std::string studyType;
    int studyVersion = 0;
};

struct RegistrationResult
{
    RegistryEntry entry;
    bool alreadyRegistered = false;
    bool artifactWritten = false;
    bool registryWritten = false;
};

std::string Sha256(std::string_view bytes);
std::string RenderRegistryEntry(const RegistryEntry& entry);

RegistrationResult Register(const std::string& sourceArtifactPath,
                            const std::filesystem::path& archiveRoot,
                            const std::filesystem::path& registryPath);

RegistryEntry Verify(const std::string& studyIdentityHash,
                     const std::filesystem::path& archiveRoot,
                     const std::filesystem::path& registryPath);

int RunFreezeCommand(const std::string& sourceArtifactPath,
                     const std::filesystem::path& archiveRoot,
                     const std::filesystem::path& registryPath,
                     std::ostream& output,
                     std::ostream& errors);

int RunVerifyCommand(const std::string& studyIdentityHash,
                     const std::filesystem::path& archiveRoot,
                     const std::filesystem::path& registryPath,
                     std::ostream& output,
                     std::ostream& errors);

} // namespace EA::ControlledReplicationStudyArchive
