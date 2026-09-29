#include "ControlledReplicationStudyArchive.hpp"
#include "ControlledReplicationStudySpecification.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace Study = EA::ControlledReplicationStudy;
namespace Archive = EA::ControlledReplicationStudyArchive;

namespace
{

Study::Specification Fixture()
{
    Study::Specification specification;
    specification.studyIdentifier = "archive-fixture";
    specification.interventionField = "feature_ablation_mask";
    specification.controlSemantics = "empty_string";
    specification.controlValue = "";
    specification.treatmentValue = "tg4_inner_break_any";
    specification.replicationDimension = "fresh_initialization_seed";
    specification.allowedContextDimensions = {"symbol"};
    specification.requiredConfiguredIdentityFields = {
        "symbol", "prediction_horizon", "feature_ablation_mask",
        "fresh_initialization_seed"};
    specification.requiredExecutionProvenanceFields = {
        "training_execution_identity", "producer_worker_attempt_id"};
    specification.aggregationPolicy =
        "unweighted_mean_of_context_family_means";
    specification.freezeTimestamp = "2026-09-28T10:00:00Z";
    specification.contexts = {
        {"1", {{"symbol", "CADCHF"}, {"prediction_horizon", "4"}},
         {{1001, 1002, 44}, {1003, 1004, 45}}},
        {"2", {{"symbol", "AUDCAD"}, {"prediction_horizon", "4"}},
         {{1005, 1006, 44}, {1007, 1008, 45}}}};
    specification.identityHash = Study::IdentityHash(specification);
    return specification;
}

std::string Read(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    std::ostringstream bytes;
    bytes << input.rdbuf();
    return bytes.str();
}

void Write(const std::filesystem::path& path, const std::string& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << bytes;
}

template <typename Function>
void ExpectThrow(Function&& function, const std::string& reason)
{
    try
    {
        function();
    }
    catch (const std::exception& error)
    {
        assert(std::string(error.what()).find(reason) != std::string::npos);
        return;
    }
    assert(false);
}

} // namespace

int main()
{
    const auto root = std::filesystem::temp_directory_path() /
        "controlled_replication_study_archive_fixture";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const auto source = root / "source.txt";
    const auto archiveRoot = root / "archive";
    const auto registry = archiveRoot / "registry.tsv";
    std::filesystem::create_directories(root);

    const auto specification = Fixture();
    const std::string artifact = Study::Render(specification);
    Write(source, artifact);

    const auto registered = Archive::Register(source.string(), archiveRoot,
                                               registry);
    assert(!registered.alreadyRegistered);
    assert(registered.artifactWritten);
    assert(registered.registryWritten);
    assert(registered.entry.studyIdentityHash == specification.identityHash);
    assert(registered.entry.artifactSha256 == Archive::Sha256(artifact));
    assert(Read(registry) == Archive::RenderRegistryEntry(registered.entry));

    const auto verified = Archive::Verify(specification.identityHash,
                                          archiveRoot, registry);
    assert(verified.studyIdentityHash == specification.identityHash);
    assert(verified.artifactSha256 == registered.entry.artifactSha256);

    const std::string registryBeforeDuplicate = Read(registry);
    const auto duplicate = Archive::Register(source.string(), archiveRoot,
                                              registry);
    assert(duplicate.alreadyRegistered);
    assert(!duplicate.artifactWritten && !duplicate.registryWritten);
    assert(Read(registry) == registryBeforeDuplicate);

    auto conflicting = specification;
    conflicting.freezeTimestamp = "2026-09-28T11:00:00Z";
    conflicting.identityHash = Study::IdentityHash(conflicting);
    const auto conflictingSource = root / "conflicting.txt";
    Write(conflictingSource, Study::Render(conflicting));
    ExpectThrow(
        [&] { (void)Archive::Register(conflictingSource.string(), archiveRoot,
                                      registry); },
        "study_archive_conflicting_duplicate");

    const auto archived = archiveRoot / registered.entry.artifactPath;
    const std::string archivedBytes = Read(archived);
    Write(archived, archivedBytes + "\n");
    ExpectThrow(
        [&] { (void)Archive::Verify(specification.identityHash, archiveRoot,
                                    registry); },
        "study_archive_artifact_sha256_mismatch");
    Write(archived, archivedBytes);

    const std::string originalRegistry = Read(registry);
    std::string semanticMutation = archivedBytes;
    const std::string treatment = "treatment_value=tg4_inner_break_any";
    const std::size_t treatmentOffset = semanticMutation.find(treatment);
    assert(treatmentOffset != std::string::npos);
    semanticMutation.replace(treatmentOffset, treatment.size(),
                             "treatment_value=other_mask");
    Write(archived, semanticMutation);
    std::string mutatedRegistry = originalRegistry;
    const std::string oldSha = registered.entry.artifactSha256;
    const std::string newSha = Archive::Sha256(semanticMutation);
    const std::size_t shaOffset = mutatedRegistry.find(oldSha);
    assert(shaOffset != std::string::npos);
    mutatedRegistry.replace(shaOffset, oldSha.size(), newSha);
    Write(registry, mutatedRegistry);
    ExpectThrow(
        [&] { (void)Archive::Verify(specification.identityHash, archiveRoot,
                                    registry); },
        "study_archive_embedded_identity_invalid");
    Write(archived, archivedBytes);
    Write(registry, originalRegistry);

    std::filesystem::remove(archived);
    ExpectThrow(
        [&] { (void)Archive::Verify(specification.identityHash, archiveRoot,
                                    registry); },
        "study_archive_artifact_missing");
    Write(archived, archivedBytes);

    Write(registry, "malformed\n");
    ExpectThrow(
        [&] { (void)Archive::Verify(specification.identityHash, archiveRoot,
                                    registry); },
        "study_archive_registry_malformed");

    Archive::RegistryEntry traversal = registered.entry;
    traversal.artifactPath = "../outside.txt";
    Write(registry, Archive::RenderRegistryEntry(traversal));
    ExpectThrow(
        [&] { (void)Archive::Verify(specification.identityHash, archiveRoot,
                                    registry); },
        "study_archive_artifact_path_invalid");

    Write(registry, originalRegistry);
    const auto outside = root / "outside.txt";
    Write(outside, archivedBytes);
    std::filesystem::remove(archived, error);
    std::filesystem::create_symlink(outside, archived, error);
    if (!error)
    {
        ExpectThrow(
            [&] { (void)Archive::Verify(specification.identityHash, archiveRoot,
                                        registry); },
            "study_archive_artifact_path_invalid");
        std::filesystem::remove(archived, error);
    }

    std::filesystem::remove_all(root, error);
    return 0;
}
