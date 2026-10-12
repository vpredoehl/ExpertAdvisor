#include "CausalFibonacciTrendlineConfluenceDiagnostic.hpp"
#include "TG4ProductionStreamingPulseAdapter.hpp"

#include <cassert>
#include <iostream>
#include <random>

namespace
{
namespace Diagnostic = EA::TG3::TrendlineDiagnostic;
using EA::TG3::ABStructure;
using EA::TG3::ABDirection;
using EA::TG3::DirectionalStudyPolicy;
using EA::TG1A::TrendLineCandidate;
using EA::TG1A::TrendLineDirection;
constexpr std::size_t kDecisionBar = 10;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

ABStructure AB(bool down = false)
{
    ABStructure ab;
    ab.identity = {down ? ABDirection::DownAB : ABDirection::UpAB,
        2, 200, 5, 500, 7, 700};
    ab.aPrice = down ? 120.0 : 100.0;
    ab.bPrice = down ? 100.0 : 120.0;
    ab.aConfirmationBar = 4;
    ab.bConfirmationBar = 7;
    return ab;
}

TrendLineCandidate Line(double projected = 110.0, bool down = false)
{
    TrendLineCandidate line;
    line.direction = down ? TrendLineDirection::DTL : TrendLineDirection::UTL;
    line.anchor1Bar = 1;
    line.anchor1Timestamp = 100;
    line.anchor1ConfirmationBar = 3;
    line.anchor2Bar = 4;
    line.anchor2Timestamp = 400;
    line.anchor2ConfirmationBar = 6;
    line.creationBar = 6;
    line.creationTimestamp = 600;
    line.currentBar = kDecisionBar;
    line.anchor1Price = projected;
    line.anchor2Price = projected;
    return line;
}

Diagnostic::Configuration Config(double threshold = 0.25)
{
    Diagnostic::Configuration config(threshold);
    config.retracementRatios = {0.5};
    return config;
}

Diagnostic::Result Evaluate(const ABStructure& ab, const TrendLineCandidate& line,
                            const Diagnostic::Configuration& config,
                            double close = 110.0, std::optional<double> atr = 2.0)
{
    return Diagnostic::Evaluate(kDecisionBar, close, atr,
        std::span{&ab, 1}, std::span{&line, 1}, config);
}

template<typename Function>
void MustReject(Function function)
{
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

void TestRetracementAndDirectionPolicy()
{
    auto config = Config();
    config.retracementRatios = {1.0, 0.25, 0.0, 0.5};
    const auto up = Evaluate(AB(), Line(), config);
    const std::vector<double> expectedUp{120.0, 115.0, 110.0, 100.0};
    for (std::size_t i = 0; i < expectedUp.size(); ++i)
        assert(up.measurements[i].fibPrice == expectedUp[i]);
    assert(Evaluate(AB(true), Line(110.0, true), config).measurements.empty());
    config.directionalStudyPolicy = DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
    const auto down = Evaluate(AB(true), Line(110.0, true), config);
    const std::vector<double> expectedDown{100.0, 105.0, 110.0, 120.0};
    for (std::size_t i = 0; i < expectedDown.size(); ++i)
    {
        assert(down.measurements[i].fibPrice == expectedDown[i]);
        assert(down.measurements[i].identity.ab.direction == ABDirection::DownAB);
        assert(down.measurements[i].identity.line.direction == TrendLineDirection::DTL);
    }
    assert(Evaluate(AB(), Line(110.0, true), config).measurements.empty());
    assert(Evaluate(AB(true), Line(), config).measurements.empty());
    Diagnostic::Configuration defaults(0.0);
    assert((defaults.retracementRatios == std::vector<double>{0.382, 0.5, 0.6180339887498949}));
    auto slope = Line(101.0);
    slope.rawPriceSlopePerBar = 1.0;
    assert(Evaluate(AB(), slope, Config()).measurements.front().linePrice == 110.0);
    slope = Line(110.5); slope.currentAtr = 99.0;
    assert(Evaluate(AB(), slope, Config()).measurements.front().normalizedStructureDistance == 0.25);
}

void TestThresholdsAndSeparateMarketDistance()
{
    const auto exact = Evaluate(AB(), Line(110.5), Config()).measurements.front();
    assert(exact.normalizedStructureDistance == 0.25);
    assert(exact.normalizedMarketDistance == 0.125);
    assert(exact.intersectionThresholdMatch && exact.configuredThresholdMatch);
    assert(!exact.marketProximityThresholdMatch.has_value());
    const auto outside = Evaluate(AB(), Line(std::nextafter(110.5, kInf)), Config());
    assert(!outside.measurements.front().intersectionThresholdMatch);
    auto config = Config(0.0);
    config.marketProximityAtrThreshold = 0.25;
    const auto distant = Evaluate(AB(), Line(), config, 130.0).measurements.front();
    assert(distant.normalizedStructureDistance == 0.0);
    assert(distant.normalizedMarketDistance == 10.0);
    assert(distant.intersectionThresholdMatch);
    assert(distant.marketProximityThresholdMatch == false);
    assert(!distant.configuredThresholdMatch);
    assert(Evaluate(AB(), Line(), config, 110.5).measurements.front().configuredThresholdMatch);
    assert(!Evaluate(AB(), Line(), config, std::nextafter(110.5, kInf))
        .measurements.front().configuredThresholdMatch);
}

void TestUnavailableAtrAndFiniteArithmetic()
{
    auto config = Config();
    config.marketProximityAtrThreshold = 100.0;
    for (const auto atr : {std::optional<double>{}, std::optional<double>{0.0},
                          std::optional<double>{-1.0}, std::optional<double>{kNaN},
                          std::optional<double>{kInf}, std::optional<double>{-kInf}})
    {
        const auto value = Evaluate(AB(), Line(), config, 110.0, atr).measurements.front();
        assert(!value.normalizedStructureDistance && !value.normalizedMarketDistance);
        assert(!value.intersectionThresholdMatch && !value.configuredThresholdMatch);
        assert(value.marketProximityThresholdMatch == false);
        assert(std::isfinite(value.fibPrice) && std::isfinite(value.linePrice));
    }
    for (double invalid : {kNaN, kInf, -kInf})
    {
        auto ab = AB(); ab.aPrice = invalid;
        assert(Evaluate(ab, Line(), config).measurements.empty());
        ab = AB(); ab.bPrice = invalid;
        assert(Evaluate(ab, Line(), config).measurements.empty());
        auto line = Line(); line.anchor1Price = invalid;
        assert(Evaluate(AB(), line, config).measurements.empty());
        line = Line(); line.anchor2Price = invalid;
        assert(Evaluate(AB(), line, config).measurements.empty());
        line = Line(); line.rawPriceSlopePerBar = invalid;
        assert(Evaluate(AB(), line, config).measurements.empty());
        MustReject([&] { Evaluate(AB(), Line(), config, invalid); });
    }
    auto line = Line(std::numeric_limits<double>::max());
    line.rawPriceSlopePerBar = std::numeric_limits<double>::max();
    assert(Evaluate(AB(), line, config).measurements.empty());
    auto ab = AB();
    ab.aPrice = -std::numeric_limits<double>::max();
    ab.bPrice = std::numeric_limits<double>::max();
    const auto midpoint = Evaluate(ab, Line(0.0), config, 0.0).measurements.front();
    assert(midpoint.fibPrice == 0.0 && midpoint.normalizedStructureDistance == 0.0);
    const auto overflow = Evaluate(AB(), Line(), config, std::numeric_limits<double>::max(),
        std::numeric_limits<double>::denorm_min()).measurements.front();
    assert(overflow.normalizedStructureDistance == 0.0);
    assert(!overflow.normalizedMarketDistance && !overflow.configuredThresholdMatch);
    const auto structureOverflow = Evaluate(AB(), Line(111.0), config, 110.0,
        std::numeric_limits<double>::denorm_min()).measurements.front();
    assert(!structureOverflow.normalizedStructureDistance && !structureOverflow.intersectionThresholdMatch);
    ab.aPrice = -std::numeric_limits<double>::max(); ab.bPrice = 100.0;
    config.retracementRatios = {1.0};
    const auto scaled = Evaluate(ab, Line(std::numeric_limits<double>::max()), config,
        0.0, std::numeric_limits<double>::max()).measurements.front();
    assert(scaled.normalizedStructureDistance == 2.0 && scaled.normalizedMarketDistance == 0.0);
}

void TestCausalEligibility()
{
    for (auto field : {&ABStructure::aConfirmationBar, &ABStructure::bConfirmationBar})
    {
        auto ab = AB(); ab.*field = kDecisionBar + 1;
        assert(Evaluate(ab, Line(), Config()).measurements.empty());
    }
    auto ab = AB(); ab.identity.availabilityBar = kDecisionBar + 1;
    assert(Evaluate(ab, Line(), Config()).measurements.empty());
    for (auto field : {&TrendLineCandidate::creationBar,
                       &TrendLineCandidate::anchor1ConfirmationBar,
                       &TrendLineCandidate::anchor2ConfirmationBar,
                       &TrendLineCandidate::currentBar})
    {
        auto line = Line(); line.*field = kDecisionBar + 1;
        assert(Evaluate(AB(), line, Config()).measurements.empty());
    }
    auto line = Line(); line.currentBar = kDecisionBar - 1;
    assert(Evaluate(AB(), line, Config()).measurements.empty());
    ab = AB(); ab.identity.aBar = ab.identity.bBar;
    assert(Evaluate(ab, Line(), Config()).measurements.empty());
    line = Line(); line.anchor1Bar = line.anchor2Bar;
    assert(Evaluate(AB(), line, Config()).measurements.empty());
    ab = AB(); ab.identity.availabilityBar = kDecisionBar;
    ab.aConfirmationBar = ab.bConfirmationBar = kDecisionBar;
    line = Line(); line.creationBar = kDecisionBar;
    line.anchor1ConfirmationBar = line.anchor2ConfirmationBar = kDecisionBar;
    assert(Evaluate(ab, line, Config()).measurements.size() == 1);
    ab.aConfirmationBar = 1;
    assert(Evaluate(ab, line, Config()).measurements.empty());
    line.anchor2ConfirmationBar = 3;
    assert(Evaluate(AB(), line, Config()).measurements.empty());
}

void TestIdentityOrderingUniquenessAndBounds()
{
    std::vector<ABStructure> abs{AB(), AB()};
    abs[1].identity.availabilityTimestamp += 1;
    std::vector<TrendLineCandidate> lines{Line(), Line(111.0)};
    lines[1].creationTimestamp += 1;
    auto config = Config(); config.retracementRatios = {0.5, 0.25, 0.5};
    const auto evaluate = [&] { return Diagnostic::Evaluate(kDecisionBar, 110.0, 2.0, abs, lines, config); };
    const auto expected = evaluate();
    assert(expected.measurements.size() == 8 && expected.pairsExamined == 4);
    assert(expected.measurements[0].identity.ratio == 0.25);
    assert(expected.measurements[1].identity.ratio == 0.5);
    assert(expected.measurements[2].identity.line.creationTimestamp == 601);
    assert(expected.measurements[4].identity.ab.availabilityTimestamp == 701);
    for (std::size_t i = 0; i < expected.measurements.size(); ++i)
        for (std::size_t j = i + 1; j < expected.measurements.size(); ++j)
            assert(expected.measurements[i].identity != expected.measurements[j].identity);
    abs.push_back(abs.front()); lines.push_back(lines.front());
    std::reverse(abs.begin(), abs.end()); std::reverse(lines.begin(), lines.end());
    assert(evaluate() == expected);
    config.maxMeasurements = 3;
    const auto capped = evaluate();
    assert(capped.measurementCapacityReached && !capped.pairCapacityReached);
    assert(capped.measurements.size() == 3);
    assert(std::equal(capped.measurements.begin(), capped.measurements.end(), expected.measurements.begin()));
    config.maxMeasurements = 100; config.maxPairEvaluations = 1;
    const auto pairCapped = evaluate();
    assert(pairCapped.pairCapacityReached && pairCapped.pairsExamined == 1);
    assert(pairCapped.measurements.size() == 2);
    config.maxPairEvaluations = 4; config.maxMeasurements = 8;
    assert(evaluate() == expected); // Exactly filling bounds is not truncation.
    config.maxABStructures = 2; MustReject(evaluate);
    config.maxABStructures = 3; config.maxTrendlines = 2; MustReject(evaluate);
    config.maxTrendlines = 3;
    abs.back().aPrice += 1.0; MustReject(evaluate);
    abs.back() = abs.front(); lines.back().rawPriceSlopePerBar = 0.1; MustReject(evaluate);
    auto zeroConfig = Config(); zeroConfig.retracementRatios = {-0.0, 0.0};
    const auto zero = Evaluate(AB(), Line(), zeroConfig);
    assert(zero.measurements.size() == 1 && !std::signbit(zero.measurements.front().identity.ratio));
    // Unsupported pairs still consume the work budget; a caller cannot bypass
    // the bound through a large cross-direction Cartesian product.
    abs = {AB(), AB()}; abs[1].identity.availabilityTimestamp += 1;
    lines = {Line(110.0, true), Line(111.0, true)}; lines[1].creationTimestamp += 1;
    config = Config(); config.maxPairEvaluations = 1;
    const auto unsupported = evaluate();
    assert(unsupported.measurements.empty() && unsupported.pairsExamined == 1 && unsupported.pairCapacityReached);
}

void TestConfigurationValidation()
{
    for (double invalid : {-0.1, 1.1, kNaN, kInf, -kInf})
    {
        auto config = Config(); config.retracementRatios = {invalid};
        MustReject([&] { Evaluate(AB(), Line(), config); });
    }
    for (double invalid : {-0.1, kNaN, kInf, -kInf})
    {
        auto config = Config(invalid);
        MustReject([&] { Evaluate(AB(), Line(), config); });
        config = Config(); config.marketProximityAtrThreshold = invalid;
        MustReject([&] { Evaluate(AB(), Line(), config); });
    }
    for (auto bound : {&Diagnostic::Configuration::maxABStructures,
                      &Diagnostic::Configuration::maxTrendlines,
                      &Diagnostic::Configuration::maxRatios,
                      &Diagnostic::Configuration::maxPairEvaluations,
                      &Diagnostic::Configuration::maxMeasurements})
    {
        auto config = Config(); config.*bound = 0;
        MustReject([&] { Evaluate(AB(), Line(), config); });
    }
    auto config = Config(); config.retracementRatios.clear();
    MustReject([&] { Evaluate(AB(), Line(), config); });
    config.retracementRatios = {0.5, 0.5}; config.maxRatios = 1;
    MustReject([&] { Evaluate(AB(), Line(), config); });
    config = Config(); config.directionalStudyPolicy = static_cast<DirectionalStudyPolicy>(99);
    MustReject([&] { Evaluate(AB(), Line(), config); });
    assert(Diagnostic::Evaluate(kDecisionBar, 110.0, 2.0, {}, {}, Config()).measurements.empty());
}

auto StateSnapshot(const EA::TG3::CausalFibonacciConfluenceIntegration& integration)
{
    const auto& geometry = integration.Behavior().Classification().Geometry();
    std::vector<std::tuple<EA::TG3::ABIdentity, double, double, std::size_t, std::size_t>> abs;
    for (const auto& ab : integration.ABStructures())
        abs.emplace_back(ab.identity, ab.aPrice, ab.bPrice, ab.aConfirmationBar, ab.bConfirmationBar);
    std::vector<std::string> lines;
    for (const auto& line : geometry.Candidates()) lines.push_back(geometry.FormatDiagnostic(line));
    std::vector<std::string> observations;
    for (const auto& observation : integration.Observations())
        observations.push_back(integration.FormatDiagnostic(observation));
    return std::tuple{geometry.CompletedBarCount(), geometry.CurrentAtr(), abs, lines,
        observations, integration.FormatSummaryDiagnostic()};
}

Diagnostic::Result EvaluateIntegration(const EA::TG3::CausalFibonacciConfluenceIntegration& integration,
                                      std::size_t bar, double close,
                                      const Diagnostic::Configuration& config)
{
    const auto before = StateSnapshot(integration);
    const auto& geometry = integration.Behavior().Classification().Geometry();
    const auto result = Diagnostic::Evaluate(bar, close, geometry.CurrentAtr(),
        integration.ABStructures(), geometry.Candidates(), config);
    assert(before == StateSnapshot(integration));
    return result;
}

std::vector<Feature> Candles()
{
    // Same proven production-pulse fixture seed and geometry as the existing
    // adapter regression, including all four observable pulse states.
    std::mt19937_64 generator(0x54473450554c5345ULL);
    std::uniform_int_distribution<int> delta(-6, 6);
    std::uniform_int_distribution<int> wick(1, 8);
    std::vector<Feature> bars;
    double close = 1.2;
    for (std::size_t i = 0; i < 5000; ++i)
    {
        const double open = close;
        close += static_cast<double>(delta(generator)) * 0.0001;
        const double high = std::max(open, close) + static_cast<double>(wick(generator)) * 0.0001;
        const double low = std::min(open, close) - static_cast<double>(wick(generator)) * 0.0001;
        bars.push_back({static_cast<float>(open), static_cast<float>(close),
            static_cast<float>(high), static_cast<float>(low),
            PriceTP{std::chrono::seconds{1'700'000'100 + static_cast<std::int64_t>(i) * 900}}, 0.0f});
    }
    return bars;
}

// Example diagnostic replay: one authoritative TG3 integration, exactly one
// AddCompletedBar per candle, then Evaluate using its synchronized state.
void TestBatchStreamPrefixEquivalenceAndUnchangedProductionPulses()
{
    const auto bars = Candles();
    const auto production = EA::ProductionTG1TG3Pulse::Configuration::TG4ADerivedSourceUTLUpABOnlyV1();
    const auto fibConfig = production.FibonacciConfigurationForSymbol("eurusdrmp");
    assert((fibConfig.retracementRatios == std::vector<double>{0.6180339887498949}));
    assert(fibConfig.directionalStudyPolicy == DirectionalStudyPolicy::SourceUTLUpABOnly);
    EA::TG3::CausalFibonacciConfluenceIntegration streaming(
        EA::TG1B::CalibrationConfiguration(production.values().referenceBarScale),
        fibConfig, production.values().geometry, production.values().behavior);
    EA::TG4Pulse::ProductionStreamingAdapter baseline("eurusdrmp");
    Diagnostic::Configuration diagnostic(0.25);
    diagnostic.directionalStudyPolicy = DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
    // A small research output budget keeps replay cheap and exercises the
    // deterministic capped prefix throughout batch/stream comparisons.
    diagnostic.maxMeasurements = 32;
    diagnostic.maxPairEvaluations = 128;
    std::vector<EA::TG1A::Candle> prefix;
    std::size_t measurementCount = 0;
    std::size_t breakCount = 0;
    std::array<std::size_t, 4> pulseStates{};
    for (std::size_t i = 0; i < bars.size(); ++i)
    {
        const auto& bar = bars[i];
        prefix.push_back({bar.time.time_since_epoch().count(), bar.open, bar.high, bar.low, bar.close});
        const auto update = streaming.AddCompletedBar(prefix.back());
        const bool verifyPrefix = i < 128 || (i + 1) % 1000 == 0;
        const auto& geometry = streaming.Behavior().Classification().Geometry();
        const auto result = verifyPrefix ? EvaluateIntegration(streaming, update.bar, bar.close, diagnostic) :
            Diagnostic::Evaluate(update.bar, bar.close, geometry.CurrentAtr(),
                streaming.ABStructures(), geometry.Candidates(), diagnostic);
        measurementCount += result.measurements.size();
        if (verifyPrefix)
        {
            auto shuffled = prefix;
            std::reverse(shuffled.begin(), shuffled.end());
            const auto historical = EA::TG3::CausalFibonacciConfluenceIntegration::FromHistorical(
                shuffled, EA::TG1B::CalibrationConfiguration(production.values().referenceBarScale),
                fibConfig, production.values().geometry, production.values().behavior);
            assert(result == EvaluateIntegration(historical, i, bar.close, diagnostic));
        }
        // The same TG3 update still projects exactly the production pulse after
        // diagnostic evaluation; comparison continues across all future bars.
        std::vector<EA::TG3::ConfluenceState> states;
        for (const auto& observation : update.newlyCreatedConfluenceObservations)
            states.push_back(observation.confluenceState);
        const auto expected = baseline.AddCompletedCanonicalBar(bar);
        breakCount += expected.bits[0];
        ++pulseStates[expected.bits[0] + expected.bits[1] + expected.bits[2]];
        assert(update.bar == i && expected.barStart == bar.time);
        assert(expected.bits == EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar(states));
    }
    assert(measurementCount > 0 && breakCount > 0);
    for (auto count : pulseStates) assert(count > 0);
    std::cout << "DIAGNOSTIC_REPLAY bars=" << bars.size() << ",measurements="
              << measurementCount << ",pulse_bars=" << breakCount << '\n';
}
} // namespace

int main()
{
    TestRetracementAndDirectionPolicy();
    TestThresholdsAndSeparateMarketDistance();
    TestUnavailableAtrAndFiniteArithmetic();
    TestCausalEligibility();
    TestIdentityOrderingUniquenessAndBounds();
    TestConfigurationValidation();
    TestBatchStreamPrefixEquivalenceAndUnchangedProductionPulses();
    std::cout << "CausalFibonacciTrendlineConfluenceDiagnosticTests passed\n";
}
