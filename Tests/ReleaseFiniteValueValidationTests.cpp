#include "CausalFibonacciConfluenceIntegration.hpp"
#include "CausalPriceLevelV2Engine.hpp"
#include "ModelInputExpansion.hpp"
#include "ProductionTG1TG3PulseConfiguration.hpp"
#include "InferenceProfitability.hpp"
#include "SchedulerCore/CheckpointEvaluationService.hpp"
#include "StrategyEvaluationCore/StrategyEvaluation.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
int failures = 0;
int checks = 0;

void Check(bool accepted, const std::string& name)
{
    ++checks;
    if (!accepted)
    {
        ++failures;
        std::cerr << "FAIL " << name << '\n';
    }
}

void Reject(const std::string& name, const auto& operation)
{
    bool rejected = false;
    try { operation(); }
    catch (const std::invalid_argument&) { rejected = true; }
    Check(rejected, name);
}

void Accept(const std::string& name, const auto& operation)
{
    bool accepted = true;
    try { operation(); }
    catch (const std::exception&) { accepted = false; }
    Check(accepted, name);
}

EA::TG3::Configuration FibonacciConfiguration()
{
    EA::TG3::Configuration configuration;
    configuration.retracementRatios = {0.25, 0.5, 0.75};
    return configuration;
}

void CheckNonfinite(double value, float floatValue, const std::string& kind)
{
    using namespace EA;
    for (auto field : {&TG1A::Configuration::interveningPriceTolerance,
                       &TG1A::Configuration::touchPriceTolerance})
    {
        TG1A::Configuration configuration;
        configuration.*field = value;
        Reject("TG1A tolerance " + kind, [&] {
            (void)TG1A::CausalFractalTrendLineGeometry(configuration);
        });
    }
    Reject("TG1B reference scale " + kind, [&] {
        (void)TG1B::CalibrationConfiguration(value);
    });
    for (auto field : {&TG2::Configuration::breakPriceTolerance,
                       &TG2::Configuration::retestPriceTolerance,
                       &TG2::Configuration::outerTargetPriceTolerance})
    {
        TG2::Configuration configuration;
        configuration.*field = value;
        Reject("TG2 tolerance " + kind, [&] {
            (void)TG2::TrendLineBehaviorTracker(configuration);
        });
    }
    Reject("TG3 tolerance " + kind, [&] {
        auto configuration = FibonacciConfiguration();
        configuration.absolutePriceTolerance = value;
        (void)TG3::FibonacciConfluenceTracker(configuration);
    });
    Reject("TG3 ratio " + kind, [&] {
        auto configuration = FibonacciConfiguration();
        configuration.retracementRatios = {value};
        (void)TG3::FibonacciConfluenceTracker(configuration);
    });
    Reject("production canonical double " + kind, [&] {
        (void)ProductionTG1TG3Pulse::StableDouble(value);
    });
    Reject("production Fibonacci tolerance " + kind, [&] {
        auto input = ProductionTG1TG3Pulse::TG4ADerivedSourceUTLUpABOnlyV1Input();
        input.fibonacciTolerancePips = value;
        ProductionTG1TG3Pulse::Validate(input);
    });
    Reject("production pip size " + kind, [&] {
        auto input = ProductionTG1TG3Pulse::TG4ADerivedSourceUTLUpABOnlyV1Input();
        input.canonicalFxPipSizes.begin()->second = value;
        ProductionTG1TG3Pulse::Validate(input);
    });
    Reject("price level V2 scale " + kind, [&] {
        auto configuration = PriceLevel::V2::ProductionConfiguration();
        configuration.scaleMultiplier = value;
        PriceLevel::V2::ValidateConfiguration(configuration);
    });
    for (auto field : {&TG1A::Candle::open, &TG1A::Candle::high,
                       &TG1A::Candle::low, &TG1A::Candle::close})
    {
        TG1A::Candle candle{1, 1.0, 1.0, 1.0, 1.0};
        candle.*field = value;
        Reject("TG1A candle " + kind, [&] {
            TG1A::CausalFractalTrendLineGeometry geometry;
            (void)geometry.AddCompletedBar(candle);
        });
        Reject("TG2 candle " + kind, [&] {
            TG2::TrendLineBehaviorTracker tracker;
            (void)tracker.ObserveCompletedBar(0, candle, {});
        });
    }
    Reject("TG3 confirmed fractal " + kind, [&] {
        TG3::FibonacciConfluenceTracker tracker(FibonacciConfiguration());
        tracker.Advance(2, 3);
        (void)tracker.ObserveConfirmedFractals(
            {{TG1A::FractalKind::Low, 0, 1, value, 2, 3}});
    });
    Reject("profitability canonical statistic " + kind, [&] {
        InferenceProfitability::Statistics statistics;
        statistics.aggregateTerminalHorizonLogReturnSum = value;
        (void)InferenceProfitability::StatisticsCanonicalText(statistics);
    });
    // Observe intentionally skips invalid price pairs rather than throwing.
    for (bool invalidDecision : {false, true})
    {
        InferenceProfitability::Accumulator accumulator;
        accumulator.Observe(InferenceProfitability::kUpClass,
                            invalidDecision ? floatValue : 100.0f,
                            invalidDecision ? 110.0f : floatValue);
        Check(accumulator.statistics().actionableCount == 0,
              "profitability skips invalid price " + kind);
    }
    Reject("strategy entry price " + kind, [&] {
        StrategyEvaluation::FixedStopLossStrategy strategy({1, 0.001});
        (void)strategy.InitialStopPrice(InferenceProfitability::kUpClass,
                                       floatValue);
    });
    for (bool leaderScore : {false, true})
    {
        ExperimentScheduler::CheckpointPolicyConfig configuration;
        if (leaderScore) configuration.minLeaderScore = value;
        else configuration.minInferAccuracy = value;
        Check(SchedulerCore::CheckpointPolicyConfigurationError(
                  configuration, false).has_value(),
              "checkpoint threshold " + kind);
    }
}
}

int main(int argc, char** argv)
{
    // Supply IEEE bit patterns at runtime. Neither constant folding nor NDEBUG
    // may remove the validation oracle or assume the test inputs are finite.
    if (argc != 7) return 2;
    static_assert(EA::kModelInputSemanticLayoutVersion == 13);
    static_assert(EA::kCurrentModelInputWidth == 171);
    constexpr std::array<const char*, 3> kinds{
        "NaN", "+infinity", "-infinity"};
    for (int index = 0; index != 3; ++index)
    {
        const auto bits = static_cast<std::uint64_t>(
            std::strtoull(argv[1 + index], nullptr, 16));
        const auto floatBits = static_cast<std::uint32_t>(
            std::strtoul(argv[4 + index], nullptr, 16));
        CheckNonfinite(std::bit_cast<double>(bits),
                       std::bit_cast<float>(floatBits),
                       kinds[index]);
    }
    Accept("valid finite TG1A", [] {
        (void)EA::TG1A::CausalFractalTrendLineGeometry{};
    });
    Accept("valid finite TG1B", [] {
        (void)EA::TG1B::CalibrationConfiguration{1.0};
    });
    Accept("valid finite TG2", [] {
        (void)EA::TG2::TrendLineBehaviorTracker{};
    });
    Accept("valid finite TG3", [] {
        (void)EA::TG3::FibonacciConfluenceTracker{FibonacciConfiguration()};
    });
    Accept("valid finite production configuration", [] {
        (void)EA::ProductionTG1TG3Pulse::Configuration::TG4ADerivedSourceUTLUpABOnlyV1();
    });
    Accept("valid finite price level", [] {
        EA::PriceLevel::V2::ValidateConfiguration(
            EA::PriceLevel::V2::ProductionConfiguration());
    });
    Accept("valid finite statistic", [] {
        (void)EA::InferenceProfitability::StatisticsCanonicalText({});
    });
    Accept("valid finite strategy", [] {
        EA::StrategyEvaluation::FixedStopLossStrategy strategy({1, 0.001});
        Check(strategy.InitialStopPrice(EA::InferenceProfitability::kUpClass,
                                        100.0f) > 0.0, "finite stop price");
    });
    EA::ExperimentScheduler::CheckpointPolicyConfig configuration;
    configuration.minLeaderScore = 0.5;
    configuration.minInferAccuracy = 0.5;
    Check(!EA::SchedulerCore::CheckpointPolicyConfigurationError(
              configuration, false), "valid finite checkpoint thresholds");
    std::cout << "ReleaseFiniteValueValidationTests checks=" << checks
              << " failures=" << failures << " layout=13 width=171\n";
    return failures == 0 ? 0 : 1;
}
