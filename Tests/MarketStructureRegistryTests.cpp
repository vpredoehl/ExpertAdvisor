#include "FeatureAblation.hpp"
#include "MarketStructureRegistry.hpp"
#include "ModelInputContract.hpp"

#include <cassert>
#include <chrono>
#include <string>
#include <vector>

namespace
{
using namespace std::chrono_literals;

class RecordingConfluenceEngine final : public EA::MarketStructure::ConfluenceEngine
{
public:
    std::vector<EA::MarketStructure::ConfluenceObservation> Describe(
        const std::vector<EA::MarketStructure::Observation>& observations,
        std::chrono::sys_seconds decisionTime) const override
    {
        return {{"descriptive-test-v1", decisionTime, observations}};
    }
};

template <typename Fn>
bool ThrowsInvalidArgument(Fn&& fn)
{
    try { fn(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

void TestRegistryAndHierarchicalResolution()
{
    using EA::FeatureAblationMask;
    assert(EA::MarketStructure::FindFamily("fibonacci") != nullptr);
    assert(EA::MarketStructure::FindFamily("tg_structure") != nullptr);
    // Pockets has an existing causal detector but no Tensor channel yet.
    assert(EA::MarketStructure::FindFamily("pockets") != nullptr);
    assert(EA::MarketStructure::FindFamily("elliott_wave") == nullptr);
    assert(EA::MarketStructure::FindFamily("confluence") == nullptr);

    const auto fib = FeatureAblationMask::Resolve("fibonacci.*");
    assert(fib.resolvedMask.CanonicalText() ==
           EA::kCausalFibonacciStructuralAblationMaskText);
    assert(fib.resolvedMask.tensorColumns().size() == 23);
    assert(fib.requestedCanonicalText == "fibonacci.*");

    const auto up = FeatureAblationMask::Resolve("fibonacci.up.recent.*");
    assert(up.resolvedMask.tensorColumns().size() == 11);
    assert(up.resolvedMask.CanonicalText().find("fib_down_") ==
           std::string::npos);

    const auto exact = FeatureAblationMask::Resolve(
        "fibonacci.up.recent.median_pullback_0618_signed_atr");
    assert(exact.resolvedMask.CanonicalText() ==
           "fib_up_recent_median_pullback_0618_signed_atr");

    const auto tg = FeatureAblationMask::Resolve("tg_structure.*");
    assert(tg.resolvedMask.CanonicalText() == EA::kTG4AblationMaskText);
    assert(tg.resolvedMask.tensorColumns().size() == 3);
}

void TestRegistryValidationFailsClosed()
{
    using EA::MarketStructure::Channel;
    using EA::MarketStructure::Family;
    using EA::MarketStructure::ValidateCatalog;

    const std::array<Family, 1> families{{
        {"family", 1, "detector-v1", "available-at"},
    }};
    const std::array<Channel, 2> validChannels{{
        {"family.one", "legacy_one", "family", 10, 1},
        {"family.two", "legacy_two", "family", 11, 1},
    }};
    ValidateCatalog(families, validChannels);

    auto duplicateColumn = validChannels;
    duplicateColumn[1].tensorColumn = duplicateColumn[0].tensorColumn;
    assert(ThrowsInvalidArgument([&] {
        ValidateCatalog(families, duplicateColumn);
    }));

    auto ambiguousIdentity = validChannels;
    ambiguousIdentity[1].persistedFeatureId = ambiguousIdentity[0].featureId;
    assert(ThrowsInvalidArgument([&] {
        ValidateCatalog(families, ambiguousIdentity);
    }));

    auto unknownFamily = validChannels;
    unknownFamily[1].familyId = "missing";
    assert(ThrowsInvalidArgument([&] {
        ValidateCatalog(families, unknownFamily);
    }));
}

void TestCanonicalAndHistoricalResolutionStability()
{
    using EA::FeatureAblationMask;
    const auto left = FeatureAblationMask::Resolve(
        "tg_structure.*,fibonacci.up.recent.*,fibonacci.down.recent.*");
    const auto right = FeatureAblationMask::Resolve(
        "fibonacci.down.recent.*,fibonacci.up.recent.*,tg_structure.*,"
        "fibonacci.up.recent.*");
    assert(left.resolvedMask.CanonicalText() == right.resolvedMask.CanonicalText());
    assert(right.resolvedMask.tensorColumns() == left.resolvedMask.tensorColumns());
    assert(right.requestedCanonicalText ==
           "fibonacci.down.recent.*,fibonacci.up.recent.*,tg_structure.*");

    // The persisted value is concrete, ordered, and contains no wildcard. A
    // future registry addition cannot alter this historical interpretation.
    const std::string persisted = left.resolvedMask.CanonicalText();
    assert(persisted.find('*') == std::string::npos);
    assert(FeatureAblationMask::Parse(persisted).CanonicalText() == persisted);

    // experiment_unique_identity_uidx already includes this persisted field;
    // these canonical values are therefore the distinct/equivalent duplicate
    // identities supplied to its existing duplicate-detection path.
    const std::string control = FeatureAblationMask::Parse("").CanonicalText();
    const std::string fibonacci = FeatureAblationMask::Resolve(
        "fibonacci.*").resolvedMask.CanonicalText();
    const std::string tgStructure = FeatureAblationMask::Resolve(
        "tg_structure.*").resolvedMask.CanonicalText();
    assert(control != fibonacci && control != tgStructure &&
           fibonacci != tgStructure);

    const auto duplicate = FeatureAblationMask::Resolve(
        "fibonacci.*,fibonacci.*,fib_up_recent_h1_count_log");
    assert(duplicate.resolvedMask.CanonicalText() ==
           EA::kCausalFibonacciStructuralAblationMaskText);

    assert(ThrowsInvalidArgument([] {
        (void)FeatureAblationMask::Resolve("pockets.*");
    }));
    assert(ThrowsInvalidArgument([] {
        (void)FeatureAblationMask::Resolve("fibonacci.retracement.*");
    }));
    assert(ThrowsInvalidArgument([] {
        (void)FeatureAblationMask::Resolve("fibonacci.*.bad");
    }));
    assert(ThrowsInvalidArgument([] {
        (void)FeatureAblationMask::Resolve("fibonacci.unknown.channel");
    }));
}

void TestLayoutAndFixedWidthParity()
{
    using EA::FeatureAblationMask;
    const auto fib = FeatureAblationMask::ParseForSemanticLayout("fibonacci.*", 9);
    assert(ThrowsInvalidArgument([] {
        (void)FeatureAblationMask::ParseForSemanticLayout("fibonacci.*", 8);
    }));
    const auto tg = FeatureAblationMask::ParseForSemanticLayout("tg_structure.*", 8);
    assert(tg.CanonicalText() == EA::kTG4AblationMaskText);

    const auto contract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    std::vector<float> tensor(feature_size);
    for (std::size_t i = 0; i < tensor.size(); ++i)
        tensor[i] = static_cast<float>(i + 1);
    std::vector<float> training(EA::kCurrentModelInputWidth, -1.0f);
    std::vector<float> inference(EA::kCurrentModelInputWidth, -1.0f);
    EA::CopyTensorFeaturesForModelInput(
        training.data(), tensor.data(), contract, fib);
    EA::CopyTensorFeaturesForModelInput(
        inference.data(), tensor.data(), contract, fib);
    assert(training == inference);
    assert(training.size() == EA::kCurrentModelInputWidth);
    for (const std::size_t column : fib.tensorColumns())
        assert(training[column] == 0.0f);
    for (std::size_t column = feature_size;
         column < EA::kCurrentModelInputWidth; ++column)
        assert(training[column] == -1.0f);
}

void TestWildcardRequestIsDeferredUntilSemanticLayoutSelection()
{
    using EA::FeatureAblationMask;
    // This is the parser-facing queue representation. It deliberately retains
    // the wildcard instead of applying FeatureAblationMask::Parse's legacy
    // default-layout behavior.
    const std::string requested =
        FeatureAblationMask::CanonicalizeRequestedExpression(
            " fibonacci.* , fibonacci.* ");
    assert(requested == "fibonacci.*");
    assert(requested.find("fib_") == std::string::npos);

    // Layout 8 has no Fibonacci feature channels. Resolving the retained
    // request against its actual layout therefore rejects it. The old queue
    // path resolved this at parser time with implicit layout 9 and would have
    // produced a 23-channel Fibonacci mask instead.
    assert(ThrowsInvalidArgument([&] {
        (void)FeatureAblationMask::Resolve(requested, 8);
    }));
    const auto layout9 = FeatureAblationMask::Resolve(requested, 9);
    assert(layout9.resolvedMask.CanonicalText() ==
           EA::kCausalFibonacciStructuralAblationMaskText);
}

void TestCausalObservationAndConfluenceIndependence()
{
    using EA::MarketStructure::CausallyAvailableObservations;
    using EA::MarketStructure::Observation;
    const auto t0 = std::chrono::sys_seconds{100s};
    const auto t1 = std::chrono::sys_seconds{101s};
    const auto t2 = std::chrono::sys_seconds{102s};
    std::vector<Observation> detectorOutput{{
        "fibonacci", "causal-fibonacci-structural-v1", t0, t2,
        "test-source-v1"}};

    assert(CausallyAvailableObservations(detectorOutput, t1).empty());
    const auto available = CausallyAvailableObservations(detectorOutput, t2);
    assert(available.size() == 1);
    assert(detectorOutput[0].availableAt == t2);

    RecordingConfluenceEngine confluence;
    const auto descriptions = confluence.Describe(available, t2);
    assert(descriptions.size() == 1);
    assert(descriptions[0].components == available);
    // Confluence consumes values; it cannot mutate the detector's output.
    assert(detectorOutput[0].availableAt == t2);

    assert(ThrowsInvalidArgument([&] {
        (void)CausallyAvailableObservations(
            {{"fibonacci", "causal-fibonacci-structural-v1", t2, t1,
              "test-source-v1"}}, t2);
    }));
}
} // namespace

int main()
{
    TestRegistryAndHierarchicalResolution();
    TestRegistryValidationFailsClosed();
    TestCanonicalAndHistoricalResolutionStability();
    TestLayoutAndFixedWidthParity();
    TestWildcardRequestIsDeferredUntilSemanticLayoutSelection();
    TestCausalObservationAndConfluenceIndependence();
}
