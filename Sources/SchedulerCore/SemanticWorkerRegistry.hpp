#pragma once

#include "SchedulerSemanticAdmission.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace EA::Scheduler
{

inline constexpr int kSemanticWorkerRegistrySchemaVersion = 1;

enum class SemanticWorkerArtifactKind
{
    Current,
    Historical
};

struct SemanticWorkerArtifact
{
    int semanticLayoutVersion = 0;
    std::size_t modelInputWidth = 0;
    SemanticWorkerArtifactKind kind = SemanticWorkerArtifactKind::Historical;
    std::string sourceCommit;
    std::string sha256;
    std::string canonicalExecutablePath;
    std::string canonicalManifestPath;
    std::set<std::string> capabilities;
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
    const SemanticWorkerArtifact& currentWorker() const;
    const SemanticWorkerArtifact* find(int semanticLayoutVersion) const noexcept;
    SemanticWorkerSelection selectInferenceWorker(
        const PersistedWorkerSemanticIdentity& persisted) const;

private:
    std::string canonicalRegistryPath_;
    int currentLayoutVersion_ = 0;
    std::map<int, SemanticWorkerArtifact> workers_;
};

std::string ValidateAndCanonicalizeWorkerExecutable(
    const std::string& configuredPath,
    const std::string& optionName);

} // namespace EA::Scheduler
