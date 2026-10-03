#include "../Sources/SchedulerCore/InferenceWorkerSelection.hpp"

#include <array>
#include <cassert>

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
               "infer", {{80}, {8}}, current).admissible);
    assert(Scheduler::EvaluateSemanticWorkerAdmission(
               "infer", {{127}, {12}}, current).admissible);
    assert(!Scheduler::EvaluateSemanticWorkerAdmission(
                "infer", {{127}, {11}}, current).admissible);
    assert(!Scheduler::EvaluateSemanticWorkerAdmission(
                "infer", {{80}, {7}}, current).admissible);
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

    // Scheduler generation and semantic-worker generation are separate.  An
    // older scheduler capability still rejects Layout 10 when it is asked to
    // act as the worker, but registry-routed identity loading must leave the
    // exact Layout-10/114 compatibility decision to the worker registry.
    Scheduler::WorkerSemanticCapability layout9Scheduler;
    layout9Scheduler.layoutVersion = 9;
    layout9Scheduler.maximumInputWidth = 103;
    constexpr std::array<EA::ModelInputSemanticLayoutRegistryEntry, 1>
        layout9OnlyRegistry{{{9, 103, 8}}};
    constexpr std::array<std::size_t, 1> layout9OnlyWidths{{103}};
    layout9Scheduler.registry = layout9OnlyRegistry;
    layout9Scheduler.registeredInputWidths = layout9OnlyWidths;
    const Scheduler::PersistedWorkerSemanticIdentity layout10{{114}, {10}, false};
    assert(!Scheduler::EvaluateSemanticWorkerAdmission(
                "train", layout10, layout9Scheduler).admissible);
    assert(Scheduler::EvaluateRegistryRoutedSemanticIdentity(
               "train", layout10).admissible);
    assert(Scheduler::EvaluateRegistryRoutedSemanticIdentity(
               "infer", layout10).admissible);

    const Scheduler::PersistedWorkerSemanticIdentity incompleteLayout10{
        {114}, {}, false};
    assert(!Scheduler::EvaluateRegistryRoutedSemanticIdentity(
                "train", incompleteLayout10).admissible);

    return 0;
}
