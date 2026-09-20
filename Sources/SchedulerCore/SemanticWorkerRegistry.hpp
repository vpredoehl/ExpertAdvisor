#pragma once

#include "SchedulerSemanticAdmission.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace EA::Scheduler
{

inline constexpr int kSemanticWorkerRegistrySchemaVersion = 4;
inline constexpr int kRoleAwareSemanticWorkerRegistrySchemaVersion = 3;
inline constexpr int kLegacySemanticWorkerRegistrySchemaVersion = 2;
inline constexpr int kLegacySemanticWorkerArtifactManifestSchemaVersion = 1;
inline constexpr int kSemanticWorkerArtifactManifestSchemaVersion = 2;
inline constexpr int kSemanticWorkerRuntimeManifestSchemaVersion = 1;

enum class SemanticWorkerArtifactKind
{
    Current,
    Historical
};

enum class SemanticWorkerRole { Infer, Train };

struct SemanticWorkerArtifact
{
    int semanticLayoutVersion = 0;
    std::size_t modelInputWidth = 0;
    SemanticWorkerArtifactKind kind = SemanticWorkerArtifactKind::Historical;
    SemanticWorkerRole role = SemanticWorkerRole::Infer;
    int artifactManifestSchemaVersion = kSemanticWorkerArtifactManifestSchemaVersion;
    std::string sourceCommit;
    std::string sha256;
    std::string canonicalExecutablePath;
    std::string canonicalManifestPath;
    std::string runtimeIdentity;
    std::set<std::string> capabilities;
};

struct SemanticWorkerRuntimeResource
{
    std::string builtIdentity;
    std::string runtimeName;
    std::string sha256;
    std::string canonicalPath;
};

struct SemanticWorkerRuntimePackage
{
    std::string identity;
    std::string canonicalDirectoryPath;
    std::string canonicalManifestPath;
    std::map<std::string, SemanticWorkerRuntimeResource> resources;
};

struct SemanticWorkerRuntimeValidation
{
    bool ready = false;
    std::string diagnostic;
    std::string canonicalRuntimeDirectoryPath;
};

struct SemanticWorkerSelection
{
    bool selected = false;
    std::string diagnostic;
    std::string canonicalExecutablePath;
    int semanticLayoutVersion = 0;
    std::size_t maximumInputWidth = 0;
    std::string reason;
};

struct SemanticWorkerRegistryLoadRequest
{
    std::string registryPath;
    std::optional<std::string> legacyLayout6ExecutableAssertion;
    int expectedCurrentSemanticLayoutVersion =
        EA::kModelInputSemanticLayoutVersion;
    std::size_t expectedCurrentModelInputWidth =
        EA::kCurrentModelInputWidth;
};

class SemanticWorkerRegistry final
{
public:
    static SemanticWorkerRegistry Load(
        const SemanticWorkerRegistryLoadRequest& request);

    const std::string& canonicalRegistryPath() const noexcept;
    // Preserves the historical current-worker meaning: training/reference,
    // with an inference fallback only for legacy layout-only registries.
    const SemanticWorkerArtifact& currentWorker() const;
    const SemanticWorkerArtifact* find(int semanticLayoutVersion) const noexcept;
    const SemanticWorkerArtifact* find(
        int semanticLayoutVersion, SemanticWorkerRole role) const noexcept;
    SemanticWorkerRuntimeValidation validateRuntimeForExecutable(
        const std::string& canonicalExecutablePath) const;
    SemanticWorkerSelection selectInferenceWorker(
        const PersistedWorkerSemanticIdentity& persisted) const;
    SemanticWorkerSelection selectTrainingReferenceWorker(
        const PersistedWorkerSemanticIdentity& persisted) const;

private:
    std::string canonicalRegistryPath_;
    int currentLayoutVersion_ = 0;
    std::map<std::string, SemanticWorkerRuntimePackage> runtimes_;
    std::map<std::pair<int, SemanticWorkerRole>, SemanticWorkerArtifact> workers_;
};

std::string ValidateAndCanonicalizeWorkerExecutable(
    const std::string& configuredPath,
    const std::string& optionName);

} // namespace EA::Scheduler
