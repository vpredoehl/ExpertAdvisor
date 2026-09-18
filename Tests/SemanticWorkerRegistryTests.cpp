#include "../Sources/SchedulerCore/SemanticWorkerRegistry.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

#include <sys/stat.h>
#include <sys/wait.h>
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
    "fe0fc41298f0aab9a9e9665db1869a1c5f078774db34c40b5f60816656e4e6cc";
constexpr const char* kRuntimeIdentity =
    "769b8f08c5f9d83cd68cb6bbe049176cbdbf67afd0b68ea052d4dd5549bd4550";
constexpr const char* kDefaultHash =
    "1cccc7d9e2aad77b0b6bb6eb7705e1478fcd454ca02cc981a0f3d371139514e8";
constexpr const char* kMetaNNHash =
    "4ddd01395b253fb9bd1ad27a87ac2f299512b74e2fa2dcea4d445c4c22913f65";

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
        writeExecutable(executable7,
            "#!/bin/sh\n"
            "worker_dir=${0%/*}\n"
            "test -r \"$worker_dir/default.metallib\" || exit 40\n"
            "test -r \"$worker_dir/MetaNN.metallib\" || exit 41\n"
            "exit 0\n");
        writeManifest(6, kCommit6, kHash6, "historical", {"infer"});
        writeManifest(
            7, kCommit7, kHash7, "current", {"train", "infer", "analyze"});
        writeRuntime();
        linkRuntime(executable6.parent_path());
        linkRuntime(executable7.parent_path());
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

    void writeRuntime()
    {
        const fs::path directory = root / "runtime" / kRuntimeIdentity;
        write(directory / "MetaNN.metallib", "metann-library\n");
        write(directory / "default.metallib", "default-library\n");
        write(directory / "manifest.json",
            "{\"resources\":[{\"built_identity\":\"MetaNN_metal.metallib\","
            "\"runtime_name\":\"MetaNN.metallib\",\"sha256\":\"" +
            std::string{kMetaNNHash} +
            "\"},{\"built_identity\":\"default.metallib\",\"runtime_name\":"
            "\"default.metallib\",\"sha256\":\"" + kDefaultHash +
            "\"}],\"schema_version\":1,\"storage\":\"immutable\"}");
    }

    void linkRuntime(const fs::path& directory)
    {
        const fs::path runtime = root / "runtime" / kRuntimeIdentity;
        fs::create_symlink(fs::relative(runtime / "default.metallib", directory),
                           directory / "default.metallib");
        fs::create_symlink(fs::relative(runtime / "MetaNN.metallib", directory),
                           directory / "MetaNN.metallib");
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
            "{\"schema_version\":2,\"current_layout\":7,\"runtimes\":["
            "{\"identity\":\"" + std::string{kRuntimeIdentity} +
            "\",\"directory\":\"runtime/" + kRuntimeIdentity +
            "\",\"manifest\":\"runtime/" + kRuntimeIdentity +
            "/manifest.json\"}],\"workers\":["
            "{\"semantic_layout\":6,\"worker_rule\":\"historical\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kCommit6} + "\",\"sha256\":\"" + kHash6 +
            "\",\"executable\":\"layout6/" + kCommit6 + "/" + kHash6 +
            "/LSTM_Release\",\"manifest\":\"layout6/" + kCommit6 + "/" +
            kHash6 + "/manifest.json\",\"runtime_identity\":\"" +
            kRuntimeIdentity + "\",\"capabilities\":[\"infer\"]},"
            "{\"semantic_layout\":7,\"worker_rule\":\"current\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kCommit7} + "\",\"sha256\":\"" + kHash7 +
            "\",\"executable\":\"layout7/" + kCommit7 + "/" + kHash7 +
            "/LSTM_Release\",\"manifest\":\"layout7/" + kCommit7 + "/" +
            kHash7 +
            "/manifest.json\",\"runtime_identity\":\"" + kRuntimeIdentity +
            "\",\"capabilities\":[\"train\",\"infer\",\"analyze\"]}" +
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
    const auto runtime7 = registry.validateRuntimeForExecutable(
        selected7.canonicalExecutablePath);
    assert(runtime7.ready);
    assert(runtime7.diagnostic == "semantic_worker_runtime_ready");
    assert(runtime7.canonicalRuntimeDirectoryPath ==
           fs::canonical(valid.executable7).parent_path());
    const pid_t launched = ::fork();
    assert(launched >= 0);
    if (launched == 0)
    {
        assert(::chdir("/") == 0);
        ::execl(selected7.canonicalExecutablePath.c_str(),
                selected7.canonicalExecutablePath.c_str(),
                static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int launchStatus = 0;
    assert(::waitpid(launched, &launchStatus, 0) == launched);
    assert(WIFEXITED(launchStatus));
    assert(WEXITSTATUS(launchStatus) == 0);
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
        kHash6 + "/manifest.json\",\"runtime_identity\":\"" +
        kRuntimeIdentity + "\",\"capabilities\":[\"infer\"]}");
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

    Fixture missingRuntime;
    fs::remove(missingRuntime.executable7.parent_path() / "MetaNN.metallib");
    assert(Contains(Failure([&] { (void)missingRuntime.load(); }),
                    "semantic_worker_runtime_dependency_missing:resource=MetaNN.metallib"));

    Fixture runtimeRemovedAfterLoad;
    const auto loadedBeforeRemoval = runtimeRemovedAfterLoad.load();
    fs::remove(runtimeRemovedAfterLoad.executable7.parent_path() /
               "default.metallib");
    const auto missingAtAdmission =
        loadedBeforeRemoval.validateRuntimeForExecutable(
            fs::canonical(runtimeRemovedAfterLoad.executable7).string());
    assert(!missingAtAdmission.ready);
    assert(Contains(missingAtAdmission.diagnostic,
                    "semantic_worker_runtime_dependency_missing:resource=default.metallib"));
    return 0;
}
