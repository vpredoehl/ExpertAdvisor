#include "CausalFibonacciStructuralFeatures.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

namespace {
using EA::CausalFibonacciFeatures::StructureState;
using EA::CausalFibonacciStructural::AdvanceH1H2EventState;
using EA::CausalFibonacciStructural::H1H2EventState;

StructureState State(bool up, std::optional<std::size_t> h1,
                     std::optional<std::size_t> h2, double level = 12.0) {
    StructureState value;
    value.upAB = up;
    value.levels.fill(level);
    value.events.h1BeyondBar = h1;
    value.events.h2RejectionBar = h2;
    return value;
}

void TestCausalEventsAndHorizon() {
    H1H2EventState events;
    AdvanceH1H2EventState(events, 10, true, true, true);
    assert(events.touchBar == 10 && events.h1BeyondBar == 10);
    assert(!events.h2RejectionBar); // H2 requires a prior completed touch.
    AdvanceH1H2EventState(events, 11, false, false, true);
    assert(events.h2RejectionBar == 11);
    auto age0 = EA::CausalFibonacciFeatures::Aggregate(
        {State(true, 10, std::nullopt)}, 10, 10.0, 2.0, .1);
    assert(age0[EA::CausalFibonacciFeatures::UpRecentH1YoungestAge20] == 0.0f);
    auto age20 = EA::CausalFibonacciFeatures::Aggregate(
        {State(true, 10, std::nullopt)}, 30, 10.0, 2.0, .1);
    assert(age20[EA::CausalFibonacciFeatures::UpRecentUnionCountLog] > 0.0f);
    auto age21 = EA::CausalFibonacciFeatures::Aggregate(
        {State(true, 10, std::nullopt)}, 31, 10.0, 2.0, .1);
    assert(age21[EA::CausalFibonacciFeatures::UpRecentUnionCountLog] == 0.0f);
}

void TestUnionOverlapCountsAgesAndDirections() {
    const auto values = EA::CausalFibonacciFeatures::Aggregate(
        {State(true, 20, 19, 14.0), State(true, 20, std::nullopt, 10.0),
         State(false, std::nullopt, 20, 6.0)},
        20, 10.0, 2.0, .1);
    using namespace EA::CausalFibonacciFeatures;
    assert(std::fabs(values[UpRecentUnionCountLog] - std::log1p(2.0)) < 1e-6);
    assert(std::fabs(values[UpRecentH1CountLog] - std::log1p(2.0)) < 1e-6);
    assert(std::fabs(values[UpRecentH2CountLog] - std::log1p(1.0)) < 1e-6);
    assert(std::fabs(values[UpRecentBothCountLog] - std::log1p(1.0)) < 1e-6);
    assert(values[UpRecentH1YoungestAge20] == 0.0f);
    assert(std::fabs(values[UpRecentH2YoungestAge20] - .05f) < 1e-6f);
    assert(std::fabs(values[DownRecentUnionCountLog] - std::log1p(1.0)) < 1e-6);
    assert(values[DownRecentH1CountLog] == 0.0f);
    assert(std::fabs(values[DownRecentH2CountLog] - std::log1p(1.0)) < 1e-6);
}

void TestExactMedianPermutationAndScale() {
    using namespace EA::CausalFibonacciFeatures;
    assert(ExactNearestRankSignedMedian({}) == 0.0);
    assert(ExactNearestRankSignedMedian({4.0}) == 4.0);
    assert(ExactNearestRankSignedMedian({4.0, 1.0}) == 1.0); // nearest rank 50th
    assert(ExactNearestRankSignedMedian({3.0, 1.0, 2.0}) == 2.0);
    assert(ExactNearestRankSignedMedian({4.0, 1.0, 3.0, 2.0}) == 2.0);
    assert(ExactNearestRankSignedMedian({-4.0, 1.0, -1.0, 3.0, -1.0}) == -1.0);
    const auto left = Aggregate({State(true, 1, {}, 14), State(true, 1, {}, 10)},
                                1, 10, 2.0, .1);
    const auto right = Aggregate({State(true, 1, {}, 10), State(true, 1, {}, 14)},
                                 1, 10, 2.0, .1);
    assert(left == right && left[UpRecentMedian1272] == 0.0f);
    const auto mirror = Aggregate({State(false, 1, {}, 6)}, 1, 10, 2.0, .1);
    assert(std::fabs(mirror[DownRecentMedian1272] - 2.0f) < 1e-6f);
    const auto pip = Aggregate({State(true, 1, {}, 11)}, 1, 10, .05, .1);
    assert(std::fabs(pip[UpRecentMedian1272] - 10.0f) < 1e-6f);
    const auto invalid = Aggregate({State(true, 1, {}, 11)}, 1,
        std::numeric_limits<double>::quiet_NaN(), 2.0, .1);
    assert(invalid[RecentPriceScaleValid] == 0.0f && invalid[UpRecentMedian1272] == 0.0f);
    auto invalidLevel = State(true, 1, {}, 11);
    invalidLevel.levels[0] = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { (void)Aggregate({invalidLevel}, 1, 10, 2.0, .1); }
    catch (const std::runtime_error& error) {
        rejected = std::string{error.what()} ==
            "CAUSAL_FIBONACCI_REQUIRED_LEVEL_INVALID";
    }
    assert(rejected); // Invalid geometry cannot masquerade as a zero median.
}

void TestUnclippedCountAndEmptyState() {
    std::vector<StructureState> states(98, State(true, 50, {}));
    const auto values = EA::CausalFibonacciFeatures::Aggregate(states, 50, 10, 2.0, .1);
    assert(std::fabs(values[EA::CausalFibonacciFeatures::UpRecentUnionCountLog] -
                     std::log1p(98.0)) < 1e-6);
    assert(values[EA::CausalFibonacciFeatures::UpRecentUnionCountLog] > std::log1p(97.0));
    const auto empty = EA::CausalFibonacciFeatures::Aggregate({}, 50, 10, 2.0, .1);
    for (std::size_t i = 1; i < empty.size(); ++i) assert(empty[i] == 0.0f);
}

void TestProducerHasNoPreconfirmationStateAndResets() {
    EA::CausalFibonacciFeatures::Producer first{"eurusdrmp"};
    EA::CausalFibonacciFeatures::Producer second{"eurusdrmp"};
    for (std::size_t i = 0; i < 4; ++i) {
        const EA::TG1A::Candle bar{static_cast<std::int64_t>(i), 1.0, 1.1, .9, 1.0};
        const auto a = first.AddCompletedBar(bar);
        const auto b = second.AddCompletedBar(bar);
        assert(a == b); // independent streams cannot leak state.
        for (std::size_t col = 1; col < a.size(); ++col) assert(a[col] == 0.0f);
    }
}

EA::TG1A::Candle Candle(std::size_t bar, double high, double low, double close) {
    return {static_cast<std::int64_t>(bar), close, high, low, close};
}

// This is the audited structural-diagnostic event ordering expressed directly
// in the test from TG1A/TG3 and the shared outcome-blind primitives.  It never
// calls CausalFibonacciFeatures::Producer: its purpose is to catch a producer
// synchronization or event-ordering regression rather than re-test Aggregate.
class DiagnosticReferenceReplay final {
public:
    struct Result {
        std::array<float, EA::CausalFibonacciFeatures::kFeatureCount> features;
        std::size_t structures = 0;
        std::size_t upStructures = 0;
        std::size_t downStructures = 0;
        std::size_t h1Events = 0;
        std::size_t h2Events = 0;
    };

    DiagnosticReferenceReplay()
        : geometry_(configuration_.geometry(), {"eurusdrmp", "15m"}),
          tracker_(configuration_.FibonacciConfigurationForSymbol("eurusdrmp"),
                   {"eurusdrmp", "15m"}) {}

    Result AddCompletedBar(const EA::TG1A::Candle& candle) {
        const auto update = geometry_.AddCompletedBar(candle);
        tracker_.Advance(update.bar, candle.timestamp);
        tracker_.ObserveConfirmedFractals(update.newlyConfirmedFractals);

        std::map<std::string, bool> live;
        for (const auto& ab : tracker_.ABStructures()) {
            const auto key = Key(ab.identity);
            live[key] = true;
            if (!states_.contains(key)) states_.emplace(key, MakeState(ab));
        }
        for (auto it = states_.begin(); it != states_.end(); )
            if (!live.contains(it->first)) it = states_.erase(it); else ++it;

        // Keep this exact order aligned with the validated diagnostic path:
        // geometry, tracker advance, confirmed fractals, live-set sync, then
        // event observation on this completed bar.
        for (auto& [key, state] : states_) {
            const auto& far = state.levels[0];
            EA::CausalFibonacciStructural::AdvanceH1H2EventState(
                state.events, update.bar,
                state.upAB ? candle.high >= far.zoneLowerPrice
                           : candle.low <= far.zoneUpperPrice,
                state.upAB ? candle.close > far.zoneUpperPrice
                           : candle.close < far.zoneLowerPrice,
                state.upAB ? candle.close < far.zoneLowerPrice
                           : candle.close > far.zoneUpperPrice);
        }

        std::vector<StructureState> structures;
        structures.reserve(states_.size());
        Result result;
        for (const auto& [key, state] : states_) {
            structures.push_back({state.upAB,
                {state.levels[0].price, state.levels[1].price,
                 state.levels[2].price, state.levels[3].price,
                 state.levels[4].price}, state.events});
            ++result.structures;
            state.upAB ? ++result.upStructures : ++result.downStructures;
            if (state.events.h1BeyondBar) ++result.h1Events;
            if (state.events.h2RejectionBar) ++result.h2Events;
        }
        result.features = EA::CausalFibonacciFeatures::Aggregate(
            structures, update.bar, candle.close, geometry_.CurrentAtr(),
            configuration_.CanonicalPipSize("eurusdrmp"));
        return result;
    }

private:
    struct RuntimeState {
        bool upAB = true;
        std::array<EA::FibonacciResearch::FibonacciLevel, 5> levels;
        EA::CausalFibonacciStructural::H1H2EventState events;
    };

    static std::string Key(const EA::TG3::ABIdentity& identity) {
        return std::to_string(identity.availabilityBar) + ':' +
            std::to_string(identity.bBar) + ':' +
            std::to_string(static_cast<int>(identity.direction)) + ':' +
            std::to_string(identity.aTimestamp) + ':' +
            std::to_string(identity.bTimestamp);
    }

    RuntimeState MakeState(const EA::TG3::ABStructure& ab) const {
        const double tolerance = configuration_.FibonacciConfigurationForSymbol(
            "eurusdrmp").absolutePriceTolerance;
        return {ab.identity.direction == EA::TG3::ABDirection::UpAB,
            {EA::FibonacciResearch::CalculateExtensionLevel(ab, 1.272, tolerance),
             EA::FibonacciResearch::CalculateExtensionLevel(ab, 1.618, tolerance),
             EA::FibonacciResearch::CalculatePullbackLevel(ab, .382, tolerance),
             EA::FibonacciResearch::CalculatePullbackLevel(ab, .500, tolerance),
             EA::FibonacciResearch::CalculatePullbackLevel(ab, .618, tolerance)}, {}};
    }

    EA::CausalFibonacciStructuralFeatureConfiguration::Configuration configuration_;
    EA::TG1A::CausalFractalTrendLineGeometry geometry_;
    EA::TG3::FibonacciConfluenceTracker tracker_;
    std::map<std::string, RuntimeState> states_;
};

void TestProducerDifferentialAgainstDiagnosticReference() {
    using namespace EA::CausalFibonacciFeatures;
    const auto Verify = [](const std::vector<EA::TG1A::Candle>& bars,
                           bool expectUp, bool expectDown) {
        Producer production{"eurusdrmp"};
        DiagnosticReferenceReplay reference;
        bool sawUp = false, sawDown = false, sawH1 = false, sawH2 = false;
        std::size_t priorH1Events = 0, priorH2Events = 0;
        for (std::size_t bar = 0; bar < bars.size(); ++bar) {
            const auto actual = production.AddCompletedBar(bars[bar]);
            const auto expected = reference.AddCompletedBar(bars[bar]);
            assert(actual == expected.features); // all 23 model-input values.
            if (bar < 7) assert(expected.structures == 0); // no A/B before B confirmation.
            sawUp = sawUp || expected.upStructures != 0;
            sawDown = sawDown || expected.downStructures != 0;
            if (expected.h1Events > priorH1Events) {
                sawH1 = true;
                assert(actual[expectUp ? UpRecentH1CountLog : DownRecentH1CountLog] >
                       0.0f);
            }
            if (expected.h2Events > priorH2Events) {
                sawH2 = true;
                assert(actual[expectUp ? UpRecentH2CountLog : DownRecentH2CountLog] >
                       0.0f);
            }
            priorH1Events = expected.h1Events;
            priorH2Events = expected.h2Events;
        }
        assert(sawUp == expectUp && sawDown == expectDown);
        assert(sawH1);
        return sawH2;
    };

    std::vector<EA::TG1A::Candle> up{
        Candle(0, 1.2, .9, 1.0), Candle(1, 1.1, .8, .9),
        Candle(2, 1.2, .5, .8), Candle(3, 1.3, .8, 1.1),
        Candle(4, 1.4, .9, 1.2), Candle(5, 2.0, 1.0, 1.8),
        Candle(6, 1.5, 1.1, 1.3), Candle(7, 1.4, 1.0, 1.2),
        Candle(8, 2.6, 2.2, 2.5), Candle(9, 2.4, 2.1, 2.3)};
    for (std::size_t bar = 10; bar <= 30; ++bar)
        up.push_back(Candle(bar, 2.4, 2.1, 2.3));
    assert(Verify(up, true, false)); // H1/H2 at age 0; H1 age 20, then 21 excluded.

    const std::vector<EA::TG1A::Candle> down{
        Candle(0, 2.5, 2.2, 2.3), Candle(1, 2.6, 2.1, 2.3),
        Candle(2, 3.0, 2.2, 2.7), Candle(3, 2.7, 2.1, 2.4),
        Candle(4, 2.6, 2.0, 2.3), Candle(5, 2.5, 1.5, 1.8),
        Candle(6, 2.4, 2.0, 2.2), Candle(7, 2.3, 2.1, 2.2),
        Candle(8, 1.3, .8, 1.0)};
    assert(!Verify(down, false, true));
}

void TestTrueProducerStreamingPrefix() {
    using namespace EA::CausalFibonacciFeatures;
    const std::vector<EA::TG1A::Candle> prefix{
        Candle(0, 1.2, .9, 1.0), Candle(1, 1.1, .8, .9),
        Candle(2, 1.2, .5, .8), Candle(3, 1.3, .8, 1.1),
        Candle(4, 1.4, .9, 1.2), Candle(5, 2.0, 1.0, 1.8),
        Candle(6, 1.5, 1.1, 1.3), Candle(7, 1.4, 1.0, 1.2),
        Candle(8, 2.6, 2.2, 2.5), Candle(9, 2.4, 2.1, 2.3),
        Candle(10, 2.4, 2.1, 2.3), Candle(11, 2.4, 2.1, 2.3)};
    Producer prefixOnly{"eurusdrmp"};
    Producer extended{"eurusdrmp"};
    std::vector<std::array<float, kFeatureCount>> prefixOnlyRows, extendedRows;
    for (const auto& bar : prefix) {
        prefixOnlyRows.push_back(prefixOnly.AddCompletedBar(bar));
        extendedRows.push_back(extended.AddCompletedBar(bar));
        assert(prefixOnlyRows.back() == extendedRows.back());
    }
    assert(prefixOnlyRows[8][UpRecentH1CountLog] > 0.0f);
    assert(prefixOnlyRows[9][UpRecentBothCountLog] > 0.0f);
    const auto retainedPrefix = extendedRows;
    for (std::size_t bar = 12; bar != 28; ++bar)
        (void)extended.AddCompletedBar(Candle(bar, 2.4, 2.1, 2.3));
    assert(extendedRows == retainedPrefix);
    assert(prefixOnlyRows == retainedPrefix);
}

void TestActualProducerCausalUpABH1H2AndAges() {
    using namespace EA::CausalFibonacciFeatures;
    Producer producer{"eurusdrmp"};
    const std::array<EA::TG1A::Candle, 8> setup{{
        Candle(0, 1.2, .9, 1.0), Candle(1, 1.1, .8, .9),
        Candle(2, 1.2, .5, .8), Candle(3, 1.3, .8, 1.1),
        Candle(4, 1.4, .9, 1.2), Candle(5, 2.0, 1.0, 1.8),
        Candle(6, 1.5, 1.1, 1.3), Candle(7, 1.4, 1.0, 1.2)}};
    for (std::size_t bar = 0; bar < setup.size(); ++bar) {
        const auto values = producer.AddCompletedBar(setup[bar]);
        if (bar < 7)
            assert(values[UpRecentUnionCountLog] == 0.0f); // B unconfirmed.
    }
    // Extension 1.272 of the confirmed .5 -> 2.0 UpAB is 2.408.
    const auto h1 = producer.AddCompletedBar(Candle(8, 2.6, 2.2, 2.5));
    assert(h1[UpRecentUnionCountLog] > 0.0f &&
           h1[UpRecentH1CountLog] > 0.0f &&
           h1[UpRecentH2CountLog] == 0.0f &&
           h1[UpRecentH1YoungestAge20] == 0.0f);
    const auto h2 = producer.AddCompletedBar(Candle(9, 2.4, 2.1, 2.3));
    assert(h2[UpRecentUnionCountLog] > 0.0f &&
           h2[UpRecentH1CountLog] > 0.0f &&
           h2[UpRecentH2CountLog] > 0.0f &&
           h2[UpRecentBothCountLog] > 0.0f &&
           h2[UpRecentH1YoungestAge20] > 0.0f &&
           h2[UpRecentH2YoungestAge20] == 0.0f);
    for (std::size_t bar = 10; bar <= 28; ++bar)
        (void)producer.AddCompletedBar(Candle(bar, 2.4, 2.1, 2.3));
    const auto age20 = producer.AddCompletedBar(Candle(29, 2.4, 2.1, 2.3));
    assert(age20[UpRecentUnionCountLog] > 0.0f);
    const auto age21 = producer.AddCompletedBar(Candle(30, 2.4, 2.1, 2.3));
    assert(age21[UpRecentUnionCountLog] == 0.0f);
}

void TestActualProducerDownABAndFrozenConfiguration() {
    using namespace EA::CausalFibonacciFeatures;
    namespace Config = EA::CausalFibonacciStructuralFeatureConfiguration;
    static_assert(Config::kMaxABAgeBars == 2048);
    static_assert(Config::kMaxActiveABStructures == 512);
    static_assert(Config::kRetracementRatio == .6180339887498949);
    const Config::Configuration configuration;
    assert(Config::kName == "causal-fibonacci-layout9-symmetric-structural-v1");
    const auto fibonacci = configuration.FibonacciConfigurationForSymbol("eurusdrmp");
    assert(configuration.timeframe() == "15m");
    assert(fibonacci.directionalStudyPolicy ==
           EA::TG3::DirectionalStudyPolicy::SymmetricDirectionalDiagnostic);
    assert(fibonacci.maxABAgeBars == 2048 && fibonacci.maxActiveABStructures == 512);
    assert(fibonacci.absolutePriceTolerance == .0001);

    Producer producer{"eurusdrmp"};
    const std::array<EA::TG1A::Candle, 8> setup{{
        Candle(0, 2.5, 2.2, 2.3), Candle(1, 2.6, 2.1, 2.3),
        Candle(2, 3.0, 2.2, 2.7), Candle(3, 2.7, 2.1, 2.4),
        Candle(4, 2.6, 2.0, 2.3), Candle(5, 2.5, 1.5, 1.8),
        Candle(6, 2.4, 2.0, 2.2), Candle(7, 2.3, 2.1, 2.2)}};
    for (const auto& bar : setup) (void)producer.AddCompletedBar(bar);
    const auto downH1 = producer.AddCompletedBar(Candle(8, 1.3, .8, 1.0));
    assert(downH1[DownRecentUnionCountLog] > 0.0f &&
           downH1[DownRecentH1CountLog] > 0.0f &&
           downH1[UpRecentUnionCountLog] == 0.0f);
}
} // namespace

int main() {
    TestCausalEventsAndHorizon();
    TestUnionOverlapCountsAgesAndDirections();
    TestExactMedianPermutationAndScale();
    TestUnclippedCountAndEmptyState();
    TestProducerHasNoPreconfirmationStateAndResets();
    TestProducerDifferentialAgainstDiagnosticReference();
    TestTrueProducerStreamingPrefix();
    TestActualProducerCausalUpABH1H2AndAges();
    TestActualProducerDownABAndFrozenConfiguration();
}
