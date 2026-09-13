#include "../Sources/SchedulerCore/InferenceWorkerSelection.hpp"

#include <array>
#include <cassert>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace
{

template <typename Operation>
bool ThrowsInvalidArgument(Operation operation)
{
    try
    {
        operation();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    namespace Scheduler = EA::Scheduler;
    const Scheduler::WorkerSemanticCapability current;
    const auto layout6 = Scheduler::EvaluateSemanticWorkerAdmission(
        "infer", {{77}, {6}}, current);
    assert(!layout6.admissible);
    assert(layout6.diagnostic == "semantic_worker_incompatible");
    assert(Scheduler::EvaluateSemanticWorkerAdmission(
               "infer", {{77}, {7}}, current).admissible);
    assert(Scheduler::EvaluateSemanticWorkerAdmission(
               "train", {{75}, {5}}, current).admissible);
    assert(Scheduler::EvaluateSemanticWorkerAdmission(
               "analyze", {{77}, {6}}, current).admissible);
    assert(!Scheduler::EvaluateSemanticWorkerAdmission(
                "infer", {{77}, std::nullopt}, current).admissible);
    assert(!Scheduler::EvaluateSemanticWorkerAdmission(
                "infer", {}, current).admissible);
    Scheduler::PersistedWorkerSemanticIdentity newTraining;
    assert(Scheduler::EvaluateSemanticWorkerAdmission(
               "train", newTraining, current).admissible);
    Scheduler::PersistedWorkerSemanticIdentity resumedWithoutIdentity;
    resumedWithoutIdentity.modelIdentityExpected = true;
    assert(!Scheduler::EvaluateSemanticWorkerAdmission(
                "train", resumedWithoutIdentity, current).admissible);

    assert(Scheduler::EvaluateLegacyMarkerlessModelAdmission(
               EA::kReturnAutocorrelationModelInputWidth, current).admissible);
    assert(!Scheduler::EvaluateLegacyMarkerlessModelAdmission(
                current.maximumInputWidth + 1, current).admissible);

    constexpr std::array<EA::ModelInputSemanticLayoutRegistryEntry, 6>
        historicalRegistry{{
            {1, EA::kHistoricalLevelProximityModelInputWidth, 0},
            {2, EA::kReturnAutocorrelationModelInputWidth, 1},
            {3, EA::kEconomicEventModelInputWidth, 2},
            {4, EA::kEconomicEventConsensusModelInputWidth, 3},
            {5, EA::kEconomicEventReleaseActualModelInputWidth, 4},
            {6, EA::kCausalEconomicEventSurpriseModelInputWidth, 5}}};
    Scheduler::WorkerSemanticCapability historical;
    historical.layoutVersion = 6;
    historical.maximumInputWidth = 77;
    historical.registry = historicalRegistry;
    assert(Scheduler::EvaluateSemanticWorkerAdmission(
               "infer", {{77}, {6}}, historical).admissible);
    assert(!Scheduler::EvaluateSemanticWorkerAdmission(
                "infer", {{77}, {7}}, historical).admissible);

    const Scheduler::InferenceWorkerRoutingConfiguration routing{
        "/current/LSTM_Release", "/legacy/LSTM_Release"};
    const auto selected7 = Scheduler::SelectInferenceWorker(
        {{77}, {7}}, routing);
    assert(selected7.selected);
    assert(selected7.canonicalExecutablePath == "/current/LSTM_Release");
    assert(selected7.semanticLayoutVersion == 7);
    assert(selected7.reason == "current_semantic_layout");

    const auto selected6 = Scheduler::SelectInferenceWorker(
        {{77}, {6}}, routing);
    assert(selected6.selected);
    assert(selected6.canonicalExecutablePath == "/legacy/LSTM_Release");
    assert(selected6.semanticLayoutVersion == 6);
    assert(selected6.reason == "legacy_semantic_layout");

    const auto unavailable6 = Scheduler::SelectInferenceWorker(
        {{77}, {6}}, {"/current/LSTM_Release", std::nullopt});
    assert(!unavailable6.selected);
    assert(unavailable6.diagnostic ==
           "semantic_worker_unavailable_for_layout=6");
    assert(!Scheduler::SelectInferenceWorker({{77}, {5}}, routing).selected);
    assert(!Scheduler::SelectInferenceWorker({{75}, {6}}, routing).selected);
    assert(!Scheduler::SelectInferenceWorker({{76}, {7}}, routing).selected);
    assert(!Scheduler::SelectInferenceWorker(
                {{77}, std::nullopt}, routing).selected);
    assert(!Scheduler::SelectInferenceWorker({}, routing).selected);

    char directoryTemplate[] = "/tmp/ea_layout6_worker.XXXXXX";
    const char* directory = ::mkdtemp(directoryTemplate);
    assert(directory != nullptr);
    assert(ThrowsInvalidArgument([&] {
        (void)Scheduler::ValidateAndCanonicalizeWorkerExecutable(
            directory, "--legacy-layout6-infer-worker");
    }));
    const std::string executable = std::string{directory} + "/worker";
    const std::string executableAlias =
        std::string{directory} + "/worker-alias";
    const int descriptor = ::open(
        executable.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0700);
    assert(descriptor >= 0);
    assert(::close(descriptor) == 0);
    char* expectedCanonical = ::realpath(executable.c_str(), nullptr);
    assert(expectedCanonical != nullptr);
    assert(Scheduler::ValidateAndCanonicalizeWorkerExecutable(
               executable, "--legacy-layout6-infer-worker") ==
           expectedCanonical);
    assert(::symlink(executable.c_str(), executableAlias.c_str()) == 0);
    assert(Scheduler::ValidateAndCanonicalizeWorkerExecutable(
               executableAlias, "--legacy-layout6-infer-worker") ==
           expectedCanonical);
    std::free(expectedCanonical);
    assert(ThrowsInvalidArgument([&] {
        (void)Scheduler::ValidateAndCanonicalizeWorkerExecutable(
            "relative/worker", "--legacy-layout6-infer-worker");
    }));
    assert(ThrowsInvalidArgument([&] {
        (void)Scheduler::ValidateAndCanonicalizeWorkerExecutable(
            std::string{directory} + "/missing",
            "--legacy-layout6-infer-worker");
    }));
    assert(::chmod(executable.c_str(), 0600) == 0);
    assert(ThrowsInvalidArgument([&] {
        (void)Scheduler::ValidateAndCanonicalizeWorkerExecutable(
            executable, "--legacy-layout6-infer-worker");
    }));
    assert(::unlink(executableAlias.c_str()) == 0);
    assert(::unlink(executable.c_str()) == 0);
    assert(::rmdir(directory) == 0);
    return 0;
}
