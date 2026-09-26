#include "../Sources/SchedulerCore/SemanticWorkerRegistry.hpp"
#include "../Sources/SchedulerCore/TrainingWorkerCommand.hpp"
#include "../Sources/SchedulerCore/TrainingWorkerSelection.hpp"
#include "FeatureAblation.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
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
constexpr const char* kInferCommit7 =
    "8888888888888888888888888888888888888888";
constexpr const char* kInferHash7 =
    "4f63f256ba59195f21f95070086d23c9e5293d824e69d89fc7164902a8a09319";
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
    fs::path inferenceExecutable7;

    Fixture()
    {
        char pattern[] = "/tmp/ea-semantic-registry.XXXXXX";
        const char* created = ::mkdtemp(pattern);
        assert(created != nullptr);
        root = created;
        executable6 = artifact(6, kCommit6, kHash6) / "LSTM_Release";
        executable7 = artifact(7, kCommit7, kHash7) / "LSTM_Release";
        inferenceExecutable7 = roleArtifact(7, "infer", kInferCommit7,
                                            kInferHash7) / "lstm-infer-worker";
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

    fs::path roleArtifact(int layout, const char* role, const char* commit,
                          const char* hash) const
    {
        return root / ("layout" + std::to_string(layout)) / role / commit / hash;
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

    static void replaceText(const fs::path& path,
                            const std::string& before,
                            const std::string& after)
    {
        std::ifstream input{path, std::ios::binary};
        assert(input.good());
        std::string contents{
            std::istreambuf_iterator<char>{input},
            std::istreambuf_iterator<char>{}};
        input.close();
        const std::size_t offset = contents.find(before);
        assert(offset != std::string::npos);
        contents.replace(offset, before.size(), after);
        write(path, contents);
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

    void writeRoleAwareRegistry()
    {
        writeExecutable(inferenceExecutable7, "infer-worker\n");
        write(inferenceExecutable7.parent_path() / "manifest.json",
            "{\"schema_version\":2,\"semantic_layout\":7,\"storage\":\"immutable\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kInferCommit7} + "\",\"sha256\":\"" + kInferHash7 +
            "\",\"executable_identity\":\"lstm-infer-worker\","
            "\"worker_role\":\"infer\",\"capabilities\":[\"infer\"]}");
        linkRuntime(inferenceExecutable7.parent_path());
        write(root / "registry.json",
            "{\"schema_version\":4,\"current_layout\":7,\"runtimes\":["
            "{\"identity\":\"" + std::string{kRuntimeIdentity} +
            "\",\"directory\":\"runtime/" + kRuntimeIdentity +
            "\",\"manifest\":\"runtime/" + kRuntimeIdentity +
            "/manifest.json\"}],\"workers\":["
            "{\"semantic_layout\":6,\"worker_role\":\"infer\","
            "\"artifact_manifest_schema_version\":1,\"worker_rule\":\"historical\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kCommit6} + "\",\"sha256\":\"" + kHash6 +
            "\",\"executable\":\"layout6/" + kCommit6 + "/" + kHash6 +
            "/LSTM_Release\",\"manifest\":\"layout6/" + kCommit6 + "/" +
            kHash6 + "/manifest.json\",\"runtime_identity\":\"" +
            kRuntimeIdentity + "\",\"capabilities\":[\"infer\"]},"
            "{\"semantic_layout\":7,\"worker_role\":\"train\","
            "\"artifact_manifest_schema_version\":1,\"worker_rule\":\"current\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kCommit7} + "\",\"sha256\":\"" + kHash7 +
            "\",\"executable\":\"layout7/" + kCommit7 + "/" + kHash7 +
            "/LSTM_Release\",\"manifest\":\"layout7/" + kCommit7 + "/" +
            kHash7 + "/manifest.json\",\"runtime_identity\":\"" +
            kRuntimeIdentity + "\",\"capabilities\":[\"train\",\"infer\",\"analyze\"]},"
            "{\"semantic_layout\":7,\"worker_role\":\"infer\","
            "\"artifact_manifest_schema_version\":2,\"worker_rule\":\"current\","
            "\"model_input_width\":77,\"source_commit\":\"" +
            std::string{kInferCommit7} + "\",\"sha256\":\"" + kInferHash7 +
            "\",\"executable\":\"layout7/infer/" + kInferCommit7 + "/" +
            kInferHash7 + "/lstm-infer-worker\",\"manifest\":\"layout7/infer/" +
            kInferCommit7 + "/" + kInferHash7 +
            "/manifest.json\",\"runtime_identity\":\"" + kRuntimeIdentity +
            "\",\"capabilities\":[\"infer\"]}]}");
    }

    void writeHistoricalTrainingRegistry()
    {
        const fs::path historicalExecutable =
            artifact(8, kCommit6, kHash6) / "LSTM_Release";
        const fs::path currentExecutable =
            artifact(9, kCommit7, kHash7) / "LSTM_Release";
        const fs::path currentInferenceExecutable =
            roleArtifact(9, "infer", kInferCommit7, kInferHash7) /
            "lstm-infer-worker";
        writeExecutable(historicalExecutable, "worker-six\n");
        writeExecutable(currentExecutable,
            "#!/bin/sh\n"
            "worker_dir=${0%/*}\n"
            "test -r \"$worker_dir/default.metallib\" || exit 40\n"
            "test -r \"$worker_dir/MetaNN.metallib\" || exit 41\n"
            "exit 0\n");
        writeExecutable(currentInferenceExecutable, "infer-worker\n");
        write(historicalExecutable.parent_path() / "manifest.json",
            "{\"schema_version\":1,\"semantic_layout\":8,"
            "\"storage\":\"immutable\",\"model_input_width\":80,"
            "\"source_commit\":\"" + std::string{kCommit6} +
            "\",\"sha256\":\"" + kHash6 +
            "\",\"executable_identity\":\"LSTM_Release\","
            "\"capabilities\":[\"train\"]}");
        write(currentExecutable.parent_path() / "manifest.json",
            "{\"schema_version\":1,\"semantic_layout\":9,"
            "\"storage\":\"immutable\",\"model_input_width\":103,"
            "\"source_commit\":\"" + std::string{kCommit7} +
            "\",\"sha256\":\"" + kHash7 +
            "\",\"executable_identity\":\"LSTM_Release\","
            "\"capabilities\":[\"train\",\"infer\",\"analyze\"]}");
        write(currentInferenceExecutable.parent_path() / "manifest.json",
            "{\"schema_version\":2,\"semantic_layout\":9,"
            "\"storage\":\"immutable\",\"model_input_width\":103,"
            "\"source_commit\":\"" + std::string{kInferCommit7} +
            "\",\"sha256\":\"" + kInferHash7 +
            "\",\"executable_identity\":\"lstm-infer-worker\","
            "\"worker_role\":\"infer\",\"capabilities\":[\"infer\"]}");
        linkRuntime(historicalExecutable.parent_path());
        linkRuntime(currentExecutable.parent_path());
        linkRuntime(currentInferenceExecutable.parent_path());
        write(root / "registry.json",
            "{\"schema_version\":4,\"current_layout\":9,\"runtimes\":["
            "{\"identity\":\"" + std::string{kRuntimeIdentity} +
            "\",\"directory\":\"runtime/" + kRuntimeIdentity +
            "\",\"manifest\":\"runtime/" + kRuntimeIdentity +
            "/manifest.json\"}],\"workers\":["
            "{\"semantic_layout\":8,\"worker_role\":\"train\","
            "\"artifact_manifest_schema_version\":1,"
            "\"worker_rule\":\"historical\",\"model_input_width\":80,"
            "\"source_commit\":\"" + std::string{kCommit6} +
            "\",\"sha256\":\"" + kHash6 +
            "\",\"executable\":\"layout8/" + kCommit6 + "/" + kHash6 +
            "/LSTM_Release\",\"manifest\":\"layout8/" + kCommit6 + "/" +
            kHash6 + "/manifest.json\",\"runtime_identity\":\"" +
            kRuntimeIdentity + "\",\"capabilities\":[\"train\"]},"
            "{\"semantic_layout\":9,\"worker_role\":\"train\","
            "\"artifact_manifest_schema_version\":1,"
            "\"worker_rule\":\"current\",\"model_input_width\":103,"
            "\"source_commit\":\"" + std::string{kCommit7} +
            "\",\"sha256\":\"" + kHash7 +
            "\",\"executable\":\"layout9/" + kCommit7 + "/" + kHash7 +
            "/LSTM_Release\",\"manifest\":\"layout9/" + kCommit7 + "/" +
            kHash7 + "/manifest.json\",\"runtime_identity\":\"" +
            kRuntimeIdentity +
            "\",\"capabilities\":[\"train\",\"infer\",\"analyze\"]},"
            "{\"semantic_layout\":9,\"worker_role\":\"infer\","
            "\"artifact_manifest_schema_version\":2,"
            "\"worker_rule\":\"current\",\"model_input_width\":103,"
            "\"source_commit\":\"" + std::string{kInferCommit7} +
            "\",\"sha256\":\"" + kInferHash7 +
            "\",\"executable\":\"layout9/infer/" + kInferCommit7 + "/" +
            kInferHash7 +
            "/lstm-infer-worker\",\"manifest\":\"layout9/infer/" +
            kInferCommit7 + "/" + kInferHash7 +
            "/manifest.json\",\"runtime_identity\":\"" +
            kRuntimeIdentity + "\",\"capabilities\":[\"infer\"]}]}");
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

void AssertTrainingCommandAblationIdentity(
    const EA::Scheduler::TrainingWorkerSelection& historical,
    const EA::Scheduler::TrainingWorkerSelection& current)
{
    const auto freshControlCommand =
        EA::Scheduler::BeginTrainingWorkerCommand(
            historical.canonicalExecutablePath, "");
    assert(freshControlCommand.size() == 2);
    assert(freshControlCommand.front() ==
           historical.canonicalExecutablePath);
    assert(freshControlCommand[1] == "--train");

    const std::string noncanonicalMask =
        "return_direction_imbalance, return_sign_persistence";
    const std::string canonicalMask =
        EA::FeatureAblationMask::Parse(noncanonicalMask).CanonicalText();
    assert(canonicalMask ==
           "return_sign_persistence,return_direction_imbalance");
    const std::string expectedAblationArgument =
        "--ablate-features=" + canonicalMask;

    const auto freshAblationCommand =
        EA::Scheduler::BeginTrainingWorkerCommand(
            historical.canonicalExecutablePath, canonicalMask);
    assert(freshAblationCommand.size() == 3);
    assert(freshAblationCommand.front() ==
           historical.canonicalExecutablePath);
    assert(freshAblationCommand[1] == "--train");
    assert(freshAblationCommand[2] == expectedAblationArgument);
    assert(std::find(freshAblationCommand.begin(),
                     freshAblationCommand.end(),
                     "--ablate-features=" + noncanonicalMask) ==
           freshAblationCommand.end());

    // Resumed and fresh training share this command prefix. These cases
    // protect both sides of BuildTrainCommand's later resume branch.
    const auto resumedControlCommand =
        EA::Scheduler::BeginTrainingWorkerCommand(
            historical.canonicalExecutablePath, "");
    assert(std::none_of(
        resumedControlCommand.begin(), resumedControlCommand.end(),
        [](const std::string& argument) {
            return argument.starts_with("--ablate-features");
        }));
    const auto resumedAblationCommand =
        EA::Scheduler::BeginTrainingWorkerCommand(
            historical.canonicalExecutablePath, canonicalMask);
    assert(resumedAblationCommand.front() ==
           historical.canonicalExecutablePath);
    assert(std::count(resumedAblationCommand.begin(),
                      resumedAblationCommand.end(),
                      expectedAblationArgument) == 1);
    assert(std::find(resumedAblationCommand.begin(),
                     resumedAblationCommand.end(),
                     current.canonicalExecutablePath) ==
           resumedAblationCommand.end());
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
    const auto legacyRegistryFreshTraining =
        registry.selectTrainingReferenceWorker({});
    assert(legacyRegistryFreshTraining.selected);
    assert(legacyRegistryFreshTraining.canonicalExecutablePath ==
           fs::canonical(valid.executable7));
    const auto legacyRegistryExplicitTraining =
        registry.selectTrainingReferenceWorker({{77}, {7}, true});
    assert(legacyRegistryExplicitTraining.selected);
    assert(legacyRegistryExplicitTraining.canonicalExecutablePath ==
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

    Fixture roleAware;
    roleAware.writeRoleAwareRegistry();
    const auto roleAwareRegistry = roleAware.load();
    const auto roleAwareInference =
        roleAwareRegistry.selectInferenceWorker({{77}, {7}, true});
    const auto roleAwareTraining =
        roleAwareRegistry.selectTrainingReferenceWorker({{77}, {7}, true});
    assert(roleAwareInference.selected);
    assert(roleAwareTraining.selected);
    assert(roleAwareInference.canonicalExecutablePath ==
           fs::canonical(roleAware.inferenceExecutable7));
    assert(roleAwareTraining.canonicalExecutablePath ==
           fs::canonical(roleAware.executable7));
    assert(roleAwareRegistry.currentWorker().canonicalExecutablePath ==
           fs::canonical(roleAware.executable7));
    assert(roleAwareRegistry.find(7, EA::Scheduler::SemanticWorkerRole::Infer) !=
           roleAwareRegistry.find(7, EA::Scheduler::SemanticWorkerRole::Train));
    const auto freshLegacyTraining =
        EA::Scheduler::SelectTrainingWorker({}, roleAwareRegistry);
    assert(freshLegacyTraining.selected);
    assert(freshLegacyTraining.canonicalExecutablePath ==
           fs::canonical(roleAware.executable7));
    assert(freshLegacyTraining.reason == "current_published_semantic_worker");
    const auto modelIdentityMissing = EA::Scheduler::SelectTrainingWorker(
        {{}, {}, true}, roleAwareRegistry);
    assert(!modelIdentityMissing.selected);
    assert(modelIdentityMissing.diagnostic ==
           "semantic_worker_identity_unavailable");
    const auto widthOnly = EA::Scheduler::SelectTrainingWorker(
        {{77}, {}, false}, roleAwareRegistry);
    const auto layoutOnly = EA::Scheduler::SelectTrainingWorker(
        {{}, {7}, false}, roleAwareRegistry);
    assert(!widthOnly.selected && !layoutOnly.selected);
    assert(widthOnly.diagnostic == "semantic_worker_identity_incomplete");
    assert(layoutOnly.diagnostic == "semantic_worker_identity_incomplete");
    const auto unsupportedHistorical = EA::Scheduler::SelectTrainingWorker(
        {{77}, {6}, true}, roleAwareRegistry);
    assert(!unsupportedHistorical.selected);
    assert(Contains(unsupportedHistorical.diagnostic,
                    "semantic_worker_layout_unsupported:layout=6"));
    const auto widthMismatch = EA::Scheduler::SelectTrainingWorker(
        {{76}, {7}, true}, roleAwareRegistry);
    assert(!widthMismatch.selected);
    assert(widthMismatch.diagnostic == "semantic_worker_incompatible");

    if (const char* externalRegistry = std::getenv("EA_SEMANTIC_REGISTRY_UNDER_TEST"))
    {
        const auto published = EA::Scheduler::SemanticWorkerRegistry::Load({
            externalRegistry, std::nullopt, 7, 77});
        const auto publishedInference =
            published.selectInferenceWorker({{77}, {7}, true});
        const auto publishedTraining =
            published.selectTrainingReferenceWorker({{77}, {7}, true});
        assert(publishedInference.selected);
        assert(publishedTraining.selected);
        assert(publishedInference.canonicalExecutablePath !=
               publishedTraining.canonicalExecutablePath);
        assert(published.currentWorker().canonicalExecutablePath ==
               publishedTraining.canonicalExecutablePath);
    }

    if (const char* rolloverRegistry =
            std::getenv("EA_SEMANTIC_REGISTRY_ROLLOVER_UNDER_TEST"))
    {
        const auto published = EA::Scheduler::SemanticWorkerRegistry::Load({
            rolloverRegistry, std::nullopt, 8, 80});
        const auto infer6 = published.selectInferenceWorker({{77}, {6}, true});
        const auto train7 =
            published.selectTrainingReferenceWorker({{77}, {7}, true});
        const auto infer7 = published.selectInferenceWorker({{77}, {7}, true});
        const auto train8 =
            published.selectTrainingReferenceWorker({{80}, {8}, true});
        const auto infer8 = published.selectInferenceWorker({{80}, {8}, true});
        assert(infer6.selected && infer6.reason ==
               "immutable_historical_semantic_worker");
        assert(train7.selected && train7.reason ==
               "immutable_historical_semantic_worker");
        assert(infer7.selected && infer7.reason ==
               "immutable_historical_semantic_worker");
        assert(train8.selected && train8.reason ==
               "current_published_semantic_worker");
        assert(infer8.selected && infer8.reason ==
               "current_published_semantic_worker");
        assert(train7.canonicalExecutablePath != train8.canonicalExecutablePath);
        assert(infer7.canonicalExecutablePath != infer8.canonicalExecutablePath);
        assert(!published.selectInferenceWorker({{80}, {7}, true}).selected);
        assert(!published.selectInferenceWorker({{77}, {8}, true}).selected);
        assert(!published.selectTrainingReferenceWorker({{80}, {7}, true}).selected);
        assert(!published.selectTrainingReferenceWorker({{77}, {8}, true}).selected);
    }

    Fixture combinedHistoricalTraining;
    combinedHistoricalTraining.writeHistoricalTrainingRegistry();
    const auto combinedRegistry = EA::Scheduler::SemanticWorkerRegistry::Load({
        (combinedHistoricalTraining.root / "registry.json").string(),
        std::nullopt,
        9,
        103});
    const auto combinedHistorical = EA::Scheduler::SelectTrainingWorker(
        {{80}, {8}, true}, combinedRegistry);
    const auto combinedCurrent = EA::Scheduler::SelectTrainingWorker(
        {{103}, {9}, true}, combinedRegistry);
    assert(combinedHistorical.selected);
    assert(combinedHistorical.semanticLayoutVersion == 8);
    assert(combinedHistorical.maximumInputWidth == 80);
    assert(combinedHistorical.reason ==
           "immutable_historical_semantic_worker");
    assert(combinedCurrent.selected);
    assert(combinedCurrent.semanticLayoutVersion == 9);
    assert(combinedCurrent.maximumInputWidth == 103);
    assert(combinedCurrent.reason == "current_published_semantic_worker");
    assert(combinedHistorical.canonicalExecutablePath !=
           combinedCurrent.canonicalExecutablePath);
    AssertTrainingCommandAblationIdentity(
        combinedHistorical, combinedCurrent);

    if (const char* trainingSelectionRegistry =
            std::getenv("EA_TRAINING_SELECTION_REGISTRY_UNDER_TEST"))
    {
        const auto published = EA::Scheduler::SemanticWorkerRegistry::Load({
            trainingSelectionRegistry,
            std::nullopt,
            EA::kModelInputSemanticLayoutVersion,
            EA::kCurrentModelInputWidth});
        const auto historical = EA::Scheduler::SelectTrainingWorker(
            {{80}, {8}, true}, published);
        const auto current = EA::Scheduler::SelectTrainingWorker(
            {{103}, {9}, true}, published);
        assert(historical.selected);
        assert(historical.semanticLayoutVersion == 8);
        assert(historical.maximumInputWidth == 80);
        assert(historical.reason == "immutable_historical_semantic_worker");
        assert(current.selected);
        assert(current.semanticLayoutVersion == 9);
        assert(current.maximumInputWidth == 103);
        assert(current.reason == "current_published_semantic_worker");
        assert(historical.canonicalExecutablePath !=
               current.canonicalExecutablePath);
        const auto* historicalArtifact =
            published.findByCanonicalExecutable(
                historical.canonicalExecutablePath);
        assert(historicalArtifact != nullptr);
        assert(historicalArtifact->role ==
               EA::Scheduler::SemanticWorkerRole::Train);
        assert(historicalArtifact->semanticLayoutVersion == 8);
        assert(historicalArtifact->modelInputWidth == 80);
        assert(!historicalArtifact->sourceCommit.empty());
        assert(!historicalArtifact->sha256.empty());
        assert(!historicalArtifact->runtimeIdentity.empty());
        assert(!historicalArtifact->canonicalManifestPath.empty());
        assert(published.validateRuntimeForExecutable(
                   historical.canonicalExecutablePath).ready);
        assert(EA::Scheduler::SelectTrainingWorker({}, published)
                   .canonicalExecutablePath == current.canonicalExecutablePath);
        assert(!EA::Scheduler::SelectTrainingWorker(
                    {{}, {}, true}, published).selected);
        assert(!EA::Scheduler::SelectTrainingWorker(
                    {{80}, {}, false}, published).selected);
        assert(!EA::Scheduler::SelectTrainingWorker(
                    {{}, {8}, false}, published).selected);
        assert(!EA::Scheduler::SelectTrainingWorker(
                    {{80}, {7}, true}, published).selected);
        assert(!EA::Scheduler::SelectTrainingWorker(
                    {{103}, {8}, true}, published).selected);
        assert(!EA::Scheduler::SelectTrainingWorker(
                    {{80}, {9}, true}, published).selected);

        // Existing role-aware inference selection remains independent.
        const auto inference8 = published.selectInferenceWorker(
            {{80}, {8}, true});
        const auto inference9 = published.selectInferenceWorker(
            {{103}, {9}, true});
        assert(inference8.selected && inference8.semanticLayoutVersion == 8);
        assert(inference9.selected && inference9.semanticLayoutVersion == 9);
        assert(inference8.canonicalExecutablePath !=
               historical.canonicalExecutablePath);
    }

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
    runtimeRemovedAfterLoad.writeRoleAwareRegistry();
    const auto loadedBeforeRemoval = runtimeRemovedAfterLoad.load();
    const auto selectedTrainingBeforeRemoval =
        loadedBeforeRemoval.selectTrainingReferenceWorker(
            {{77}, {7}, true});
    assert(selectedTrainingBeforeRemoval.selected);
    fs::remove(runtimeRemovedAfterLoad.executable7.parent_path() /
               "default.metallib");
    const auto missingAtAdmission =
        loadedBeforeRemoval.validateRuntimeForExecutable(
            selectedTrainingBeforeRemoval.canonicalExecutablePath);
    assert(!missingAtAdmission.ready);
    assert(Contains(missingAtAdmission.diagnostic,
                    "semantic_worker_runtime_dependency_missing:resource=default.metallib"));

    Fixture trainingCapabilityMissing;
    trainingCapabilityMissing.writeRoleAwareRegistry();
    const std::string allCapabilities =
        "\"capabilities\":[\"train\",\"infer\",\"analyze\"]";
    const std::string noTrainCapability =
        "\"capabilities\":[\"infer\",\"analyze\"]";
    Fixture::replaceText(
        trainingCapabilityMissing.executable7.parent_path() / "manifest.json",
        allCapabilities,
        noTrainCapability);
    Fixture::replaceText(
        trainingCapabilityMissing.root / "registry.json",
        allCapabilities,
        noTrainCapability);
    assert(Contains(Failure([&] {
        (void)trainingCapabilityMissing.load();
    }), "semantic_worker_capability_mismatch:layout=7:role_required=train"));
    return 0;
}
