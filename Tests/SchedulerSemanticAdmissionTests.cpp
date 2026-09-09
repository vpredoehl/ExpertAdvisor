#include "../Sources/SchedulerCore/SchedulerSemanticAdmission.hpp"

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
    return 0;
}
