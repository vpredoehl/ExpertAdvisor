#include "../Sources/SchedulerCore/SemanticWorkerRegistry.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

namespace
{

namespace fs = std::filesystem;

constexpr const char* kCommit6 =
    "6666666666666666666666666666666666666666";
constexpr const char* kCommit7 =
    "7777777777777777777777777777777777777777";
constexpr const char* kHash6 =
    "4f330de39e18484e875518d51f1d60ebbc278db85b99111b70a68a3f1d538764";
constexpr const char* kHash7 =
    "a6496dd5198a58c7ecf91ce16a7edfb4b586e007429063c6ba47da6b66c68d25";

struct Fixture
{
    fs::path root;
    fs::path executable6;
    fs::path executable7;

    Fixture()
    {
        char pattern[] = "/tmp/ea-semantic-registry.XXXXXX";
        const char* created = ::mkdtemp(pattern);
        assert(created != nullptr);
        root = created;
        executable6 = artifact(6, kCommit6, kHash6) / "LSTM_Release";
        executable7 = artifact(7, kCommit7, kHash7) / "LSTM_Release";
        writeExecutable(executable6, "worker-six\n");
        writeExecutable(executable7, "worker-seven\n");
        writeManifest(6, kCommit6, kHash6, "historical", {"infer"});
        writeManifest(
            7, kCommit7, kHash7, "current", {"train", "infer", "analyze"});
        writeRegistry();
    }

    ~Fixture()
    {
        std::error_code error;
        fs::remove_all(root, error);
    }

    fs::path artifact(int layout, const char* commit, const char* hash) const
    {
        return root / ("layout" + std::to_string(layout)) / commit / hash;
    }

    static void write(const fs::path& path, const std::string& contents)
    {
        fs::create_directories(path.parent_path());
        std::ofstream output{path, std::ios::binary | std::ios::trunc};
        assert(output.good());
        output << contents;
        output.close();
        assert(output.good());
    }

    static void writeExecutable(const fs::path& path, const std::string& contents)
    {
        write(path, contents);
        assert(::chmod(path.c_str(), 0700) == 0);
    }

    void writeManifest(
        int layout, const char* commit, const char* hash,
        const char* rule, std::initializer_list<const char*> capabilities)
    {
        std::string capabilityJson;
        for (const char* capability : capabilities)
        {
            if (!capabilityJson.empty()) capabilityJson += ',';
            capabilityJson += "\"" + std::string{capability} + "\"";
        }
        write(artifact(layout, commit, hash) / "manifest.json",
            "{\"schema_version\":1,\"semantic_layout\":" +
            std::to_string(layout) +
            ",\"storage\":\"immutable\",\"model_input_width\":77,"
            "\"source_commit\":\"" + commit + "\",\"sha256\":\"" + hash +
            "\",\"executable_identity\":\"LSTM_Release\","
            "\"capabilities\":[" + capabilityJson + "]}");
        (void)rule;
    }

    void writeRegistry(const std::string& suffix = {})
    {
        write(root / "registry.json",
            "{\"schema_version\":1,\"current_layout\":7,\"workers\":["
            "{\"semantic_layout\":6,\"worker_rule\":\"historical\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kCommit6} + "\",\"sha256\":\"" + kHash6 +
            "\",\"executable\":\"layout6/" + kCommit6 + "/" + kHash6 +
            "/LSTM_Release\",\"manifest\":\"layout6/" + kCommit6 + "/" +
            kHash6 + "/manifest.json\",\"capabilities\":[\"infer\"]},"
            "{\"semantic_layout\":7,\"worker_rule\":\"current\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kCommit7} + "\",\"sha256\":\"" + kHash7 +
            "\",\"executable\":\"layout7/" + kCommit7 + "/" + kHash7 +
            "/LSTM_Release\",\"manifest\":\"layout7/" + kCommit7 + "/" +
            kHash7 +
            "/manifest.json\",\"capabilities\":[\"train\",\"infer\",\"analyze\"]}" +
            suffix + "]}");
    }

    EA::Scheduler::SemanticWorkerRegistry load(
        std::optional<std::string> assertion = std::nullopt) const
    {
        return EA::Scheduler::SemanticWorkerRegistry::Load({
            (root / "registry.json").string(), std::move(assertion), 7, 77});
    }
};

std::string Failure(const std::function<void()>& operation)
{
    try
    {
        operation();
    }
    catch (const std::invalid_argument& error)
    {
        return error.what();
    }
    assert(false && "operation should have failed closed");
    return {};
}

bool Contains(const std::string& value, const std::string& expected)
{
    return value.find(expected) != std::string::npos;
}

} // namespace

int main()
{
    using EA::Scheduler::PersistedWorkerSemanticIdentity;

    Fixture valid;
    const auto registry = valid.load();
    assert(registry.currentWorker().semanticLayoutVersion == 7);
    assert(registry.currentWorker().canonicalExecutablePath ==
           fs::canonical(valid.executable7));

    const auto selected6 = registry.selectInferenceWorker({{77}, {6}, true});
    assert(selected6.selected);
    assert(selected6.canonicalExecutablePath == fs::canonical(valid.executable6));
    assert(selected6.reason == "immutable_historical_semantic_worker");
    const auto selected7 = registry.selectInferenceWorker({{77}, {7}, true});
    assert(selected7.selected);
    assert(selected7.canonicalExecutablePath == fs::canonical(valid.executable7));
    assert(selected7.reason == "current_published_semantic_worker");
    const auto unknown = registry.selectInferenceWorker({{77}, {8}, true});
    assert(!unknown.selected);
    assert(Contains(unknown.diagnostic, "semantic_worker_layout_unsupported"));
    assert(!registry.selectInferenceWorker({{76}, {7}, true}).selected);
    assert(!registry.selectInferenceWorker(PersistedWorkerSemanticIdentity{}).selected);

    assert(valid.load(valid.executable6.string()).find(6) != nullptr);
    assert(Contains(Failure([&] {
        (void)valid.load(valid.executable7.string());
    }), "legacy_layout6_worker_registry_conflict"));

    Fixture malformed;
    Fixture::write(malformed.root / "registry.json", "{not-json");
    assert(Contains(Failure([&] { (void)malformed.load(); }),
                    "semantic_worker_registry_malformed"));

    Fixture duplicate;
    duplicate.writeRegistry(
        ",{\"semantic_layout\":6,\"worker_rule\":\"historical\","
        "\"model_input_width\":77,\"source_commit\":\"" +
        std::string{kCommit6} + "\",\"sha256\":\"" + kHash6 +
        "\",\"executable\":\"layout6/" + kCommit6 + "/" + kHash6 +
        "/LSTM_Release\",\"manifest\":\"layout6/" + kCommit6 + "/" +
        kHash6 + "/manifest.json\",\"capabilities\":[\"infer\"]}");
    assert(Contains(Failure([&] { (void)duplicate.load(); }),
                    "semantic_worker_registry_duplicate_layout"));

    Fixture missing;
    fs::remove(missing.executable6);
    assert(Contains(Failure([&] { (void)missing.load(); }),
                    "semantic_worker_artifact_missing"));

    Fixture hashMismatch;
    Fixture::writeExecutable(hashMismatch.executable6, "tampered\n");
    assert(Contains(Failure([&] { (void)hashMismatch.load(); }),
                    "semantic_worker_hash_mismatch"));

    Fixture manifestMismatch;
    manifestMismatch.writeManifest(
        6, kCommit6, kHash6, "historical", {"infer", "analyze"});
    assert(Contains(Failure([&] { (void)manifestMismatch.load(); }),
                    "semantic_worker_manifest_mismatch"));

    Fixture aliasedArtifact;
    const fs::path aliasedTarget =
        aliasedArtifact.executable6.parent_path() / "worker-alias-target";
    fs::rename(aliasedArtifact.executable6, aliasedTarget);
    fs::create_symlink(aliasedTarget.filename(), aliasedArtifact.executable6);
    assert(Contains(Failure([&] { (void)aliasedArtifact.load(); }),
                    "artifact_path_not_canonical"));
    return 0;
}
