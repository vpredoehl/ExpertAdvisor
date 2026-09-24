#include "ScientificExecutionProvenanceBackfillService.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <string>
#include <vector>

namespace
{
EA::ScientificExecutionProvenanceBackfill::Options Parse(
    std::initializer_list<const char*> values)
{
    std::vector<const char*> argv{"backfill-test"};
    argv.insert(argv.end(), values.begin(), values.end());
    return EA::ScientificExecutionProvenanceBackfill::Parse(
        static_cast<int>(argv.size()), argv.data());
}
}

int main()
{
    using namespace EA::ScientificExecutionProvenanceBackfill;
    assert(LowerHex(std::string(64, 'a'), 64));
    assert(!LowerHex(std::string(64, 'A'), 64));
    assert(!LowerHex("not-a-sha", 64));
    assert(JsonString("{\"storage\":\"immutable\"}", "storage") == "immutable");
    assert(JsonInt("{\"semantic_layout\":8}", "semantic_layout") == 8);

    const auto dry = Parse({"--backfill-scientific-execution-provenance", "--dry-run"});
    assert(!dry.apply && !dry.confirmed);
    const auto apply = Parse({"--backfill-scientific-execution-provenance", "--apply", "--yes", "--experiment-id=652"});
    assert(apply.apply && apply.confirmed && apply.experimentId == 652);
    bool rejected = false;
    try { (void)Parse({"--backfill-scientific-execution-provenance", "--apply"}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    // Each publication link must identify the same immutable runtime package;
    // two individually valid packages are not worker-local provenance.
    const auto temporary = std::filesystem::temp_directory_path() /
        ("ea_runtime_identity_test_" + std::to_string(::getpid()));
    const auto root = temporary / "Builds" / "SemanticWorkers";
    const auto worker = root / "layout8" / "train" / "commit" / "sha";
    const auto makePackage = [&](const std::string& defaultBytes,
                                 const std::string& metaBytes) {
        const auto staging = temporary / ("staging_" + defaultBytes);
        std::filesystem::create_directories(staging);
        { std::ofstream out{staging / "default.metallib"}; out << defaultBytes; }
        { std::ofstream out{staging / "MetaNN.metallib"}; out << metaBytes; }
        const std::string defaultHash = FileSha256(staging / "default.metallib");
        const std::string metaHash = FileSha256(staging / "MetaNN.metallib");
        { std::ofstream out{staging / "manifest.json"};
          out << "{\"schema_version\":1,\"storage\":\"immutable\","
              << "\"resources\":[{\"built_identity\":\"default.metallib\",\"runtime_name\":\"default.metallib\",\"sha256\":\"" << defaultHash
              << "\"},{\"built_identity\":\"MetaNN_metal.metallib\",\"runtime_name\":\"MetaNN.metallib\",\"sha256\":\"" << metaHash << "\"}]}"; }
        const std::string identity = FileSha256(staging / "manifest.json");
        const auto package = root / "runtime" / identity;
        std::filesystem::create_directories(package.parent_path());
        std::filesystem::rename(staging, package);
        return package;
    };
    const auto packageA = makePackage("default-a", "meta-a");
    const auto packageB = makePackage("default-b", "meta-b");
    std::filesystem::create_directories(worker);
    const auto link = [&](const char* name, const std::filesystem::path& target) {
        std::filesystem::create_symlink(std::filesystem::relative(target, worker), worker / name);
    };
    link("default.metallib", packageA / "default.metallib");
    link("MetaNN.metallib", packageB / "MetaNN.metallib");
    assert(!RecoverPublishedRuntimeIdentity(root, worker));
    std::filesystem::remove_all(temporary);
    return 0;
}
