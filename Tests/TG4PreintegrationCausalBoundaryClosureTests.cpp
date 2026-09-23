#include "CanonicalMarketDataRange.hpp"
#include "ProductionTG1TG3PulseConfiguration.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace
{
using EA::CanonicalMarketData::AbsoluteHalfOpenRange;
using EA::CanonicalMarketData::FormatAbsoluteUtc;
using EA::ProductionTG1TG3Pulse::Configuration;
using EA::ProductionTG1TG3Pulse::Input;

PriceTP At(std::int64_t seconds)
{
    return PriceTP{std::chrono::seconds{seconds}};
}

struct CanonicalBar
{
    PriceTP timestamp;
    double open;
    double high;
    double low;
    double close;
    std::int64_t intervalSeconds = 900;

    bool operator==(const CanonicalBar&) const = default;
};

std::vector<CanonicalBar> TensorConsumerRows(
    const std::vector<CanonicalBar>& source, const AbsoluteHalfOpenRange& range)
{
    std::vector<CanonicalBar> result;
    for (const CanonicalBar& bar : source)
        if (range.Contains(bar.timestamp)) result.push_back(bar);
    return result;
}

std::vector<CanonicalBar> FutureTGConsumerRows(
    const std::vector<CanonicalBar>& source, const AbsoluteHalfOpenRange& range)
{
    // This is intentionally a separate consumer loop.  Both paths use only
    // the canonical absolute contract, mirroring future Tensor/TG readers.
    std::vector<CanonicalBar> result;
    for (const CanonicalBar& bar : source)
    {
        if (bar.timestamp < range.start || !(bar.timestamp < range.end)) continue;
        result.push_back(bar);
    }
    return result;
}

void AssertParity(const std::vector<CanonicalBar>& source,
                  const AbsoluteHalfOpenRange& range,
                  const std::vector<CanonicalBar>& expected)
{
    const auto tensor = TensorConsumerRows(source, range);
    const auto tg = FutureTGConsumerRows(source, range);
    assert(tensor == expected);
    assert(tg == expected);
    assert(tensor == tg);
    for (const CanonicalBar& bar : tensor)
        assert(bar.intervalSeconds == EA::CanonicalMarketData::kCanonicalIntervalSeconds);
    assert(TensorConsumerRows(source, range) == tensor);
    assert(FutureTGConsumerRows(source, range) == tg);
}

struct FakeTransaction
{
    std::string quote(const std::string& value) const { return "'" + value + "'"; }
    std::string quote(int value) const { return "'" + std::to_string(value) + "'"; }
};

void TestCanonicalAbsoluteHalfOpenRange()
{
    const std::vector<CanonicalBar> ordinary{
        {At(0), 1, 3, 0, 2}, {At(900), 2, 4, 1, 3},
        {At(1800), 3, 5, 2, 4}, {At(2700), 4, 6, 3, 5}};
    AssertParity(ordinary, {At(0), At(2700)},
                 {ordinary[0], ordinary[1], ordinary[2]});
    AssertParity(ordinary, {At(900), At(1800)}, {ordinary[1]});
    AssertParity(ordinary, {At(2700), At(3600)}, {ordinary[3]});

    const std::vector<CanonicalBar> noBars;
    AssertParity(noBars, {At(0), At(900)}, {});
    AssertParity(ordinary, {At(900), At(1800)}, {ordinary[1]});

    FakeTransaction transaction;
    const std::string sql =
        EA::CanonicalMarketData::CanonicalHalfOpenCandlestickQuery(
            transaction, "eurusdrmp", {At(0), At(900)});
    assert(sql.find("< '1970-01-01 00:15:00+00'::timestamptz") !=
           std::string::npos);
    assert(sql.find("ORDER BY dt") != std::string::npos);
}

void TestDstAndGapParity()
{
    // 2025-03-09 01:45 EST -> 03:00 EDT: an absolute range contains only
    // available completed bars; the missing civil hour is never synthesized.
    const std::vector<CanonicalBar> spring{
        {At(1741502700), 1, 2, 0, 1.5}, // 2025-03-09 06:45:00+00
        {At(1741503600), 2, 3, 1, 2.5}, // 2025-03-09 07:00:00+00
        {At(1741504500), 3, 4, 2, 3.5}};
    assert(FormatAbsoluteUtc(spring[0].timestamp) == "2025-03-09 06:45:00+00");
    assert(FormatAbsoluteUtc(spring[1].timestamp) == "2025-03-09 07:00:00+00");
    AssertParity(spring, {At(1741502700), At(1741504500)},
                 {spring[0], spring[1]});

    // Fall-back range crosses the duplicated civil hour using absolute UTC
    // boundaries; no civil timestamp is accepted from a caller.
    const std::vector<CanonicalBar> fall{
        {At(1762058700), 1, 2, 0, 1.5}, // 2025-11-02 04:45:00+00
        {At(1762066800), 2, 3, 1, 2.5}, // 2025-11-02 07:00:00+00
        {At(1762067700), 3, 4, 2, 3.5}};
    assert(FormatAbsoluteUtc(fall[0].timestamp) == "2025-11-02 04:45:00+00");
    assert(FormatAbsoluteUtc(fall[1].timestamp) == "2025-11-02 07:00:00+00");
    AssertParity(fall, {At(1762058700), At(1762067700)},
                 {fall[0], fall[1]});

    // Friday to Sunday opening gap: available source bars retain their order
    // and OHLC identity; the range does not fabricate weekend intervals.
    const std::vector<CanonicalBar> weekend{
        {At(1741995900), 1, 2, 0, 1.5}, // Fri 2025-03-14 23:45 UTC
        {At(1742158800), 2, 3, 1, 2.5}, // Sun 2025-03-16 21:00 UTC
        {At(1742159700), 3, 4, 2, 3.5}};
    AssertParity(weekend, {At(1741995000), At(1742159700)},
                 {weekend[0], weekend[1]});
}

std::string HashFor(Input value)
{
    return Configuration{std::move(value)}.identity().hash;
}

void AssertSemanticFieldChangesIdentity()
{
    const Input baseline =
        EA::ProductionTG1TG3Pulse::TG4ADerivedSourceUTLUpABOnlyV1Input();
    const std::string original = HashFor(baseline);
    auto changed = [&](auto mutate)
    {
        Input copy = baseline;
        mutate(copy);
        assert(HashFor(std::move(copy)) != original);
    };
    changed([](Input& v) { v.barIdentity += "_changed"; });
    changed([](Input& v) { v.marketRangeContractVersion += "_changed"; });
    changed([](Input& v) { v.ordering += "_changed"; });
    changed([](Input& v) { v.gapPolicy += "_changed"; });
    changed([](Input& v) { v.fractalConfirmationSemantics += "_changed"; });
    changed([](Input& v) { v.geometry.interveningPriceTolerance = 0.00001; });
    changed([](Input& v) { v.geometry.touchPriceTolerance = 0.00001; });
    changed([](Input& v) { ++v.geometry.maxFractalAnchorLookbackBars; });
    changed([](Input& v) { ++v.geometry.maxConfirmedFractalsPerKind; });
    changed([](Input& v) { ++v.geometry.maxCandidateAgeBars; });
    changed([](Input& v) { ++v.geometry.maxCandidates; });
    changed([](Input& v) { ++v.geometry.atrPeriod; });
    changed([](Input& v) { v.referenceBarScale = 15.0; });
    changed([](Input& v) { v.angleBands += "_changed"; });
    changed([](Input& v) { v.classificationSnapshotSemantics += "_changed"; });
    changed([](Input& v) {
        v.behavior.breakPolicy = EA::TG2::BreakPolicy::CompletedWickBeyondLine;
    });
    changed([](Input& v) { v.behavior.breakPriceTolerance = 0.00001; });
    changed([](Input& v) { v.fibonacci.retracementRatios[0] = 0.5; });
    changed([](Input& v) {
        v.fibonacci.directionalStudyPolicy =
            EA::TG3::DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
    });
    changed([](Input& v) { v.fibonacciTolerancePips = 2.0; });
    changed([](Input& v) { v.canonicalFxPipSizes["eurusdrmp"] = 0.001; });
    changed([](Input& v) { v.canonicalFxPipSizes.erase("cadchfrmp"); });
    changed([](Input& v) { ++v.fibonacci.maxConfirmedFractalsPerKind; });
    changed([](Input& v) { ++v.fibonacci.maxABAgeBars; });
    changed([](Input& v) { ++v.fibonacci.maxActiveABStructures; });

    const std::string payload =
        EA::ProductionTG1TG3Pulse::CanonicalPayload(baseline);
    for (const std::string& key : {
             "tg1_max_confirmed_fractals_per_kind=",
             "tg1_max_candidates=", "tg2_break_policy=",
             "tg2_outer_pairing_policy=", "tg3_anchor_selection=",
             "tg3_confluence_policy=", "tg3_max_active_ab_structures="})
        assert(payload.find(key) != std::string::npos);

    Input operational = baseline;
    ++operational.behavior.maxRetainedBreakObservations;
    assert(HashFor(operational) == original);
    assert(EA::ProductionTG1TG3Pulse::OperationalRetentionPayload(operational) !=
           EA::ProductionTG1TG3Pulse::OperationalRetentionPayload(baseline));
}

void TestProductionConfiguration()
{
    const Configuration first =
        Configuration::TG4ADerivedSourceUTLUpABOnlyV1();
    const Configuration second =
        Configuration::TG4ADerivedSourceUTLUpABOnlyV1();
    assert(first.identity().canonicalPayload == second.identity().canonicalPayload);
    assert(first.identity().hash == second.identity().hash);
    assert(first.values().configurationSchema ==
           "tg1-tg3-causal-pulse-configuration-v1");
    assert(first.values().name == "tg4a-derived-source-utl-up-ab-only-v1");
    assert(first.values().timeframe == "15m");
    assert(first.values().candlePeriod == 15);
    assert(first.values().candleUnit == "minute");
    assert(first.values().expectedIntervalSeconds == 900);
    assert(first.values().geometry.interveningPriceTolerance == 0.0);
    assert(first.values().geometry.touchPriceTolerance == 0.0);
    assert(first.values().geometry.maxFractalAnchorLookbackBars == 512);
    assert(first.values().geometry.maxConfirmedFractalsPerKind == 64);
    assert(first.values().geometry.maxCandidateAgeBars == 512);
    assert(first.values().geometry.maxCandidates == 4096);
    assert(first.values().geometry.atrPeriod == 14);
    assert(first.values().referenceBarScale == 14.0);
    assert(first.values().behavior.breakPolicy ==
           EA::TG2::BreakPolicy::CompletedCloseBeyondLine);
    assert(first.values().behavior.breakPriceTolerance == 0.0);
    assert(first.values().behavior.retestPriceTolerance == 0.0);
    assert(first.values().behavior.outerTargetPriceTolerance == 0.0);
    assert(first.values().behavior.retestHorizonBars == 20);
    assert(first.values().behavior.outerTargetHorizonBars == 20);
    assert(first.values().behavior.maxActiveBreakObservations == 4096);
    assert(first.values().behavior.maxRetainedBreakObservations == 4096);
    assert(first.values().fibonacci.retracementRatios ==
           std::vector<double>{0.6180339887498949});
    assert(first.values().fibonacci.directionalStudyPolicy ==
           EA::TG3::DirectionalStudyPolicy::SourceUTLUpABOnly);
    assert(first.values().fibonacci.maxConfirmedFractalsPerKind == 128);
    assert(first.values().fibonacci.maxABAgeBars == 2048);
    assert(first.values().fibonacci.maxActiveABStructures == 512);
    assert(first.values().fibonacci.maxActiveConfluenceObservations == 4096);
    assert(first.values().fibonacci.maxRetainedConfluenceObservations == 4096);
    assert(first.values().fibonacciTolerancePips == 1.0);
    const std::map<std::string, double> expectedPipSizes{
        {"audcadrmp", 0.0001}, {"audchfrmp", 0.0001},
        {"audjpyrmp", 0.01},   {"audnzdrmp", 0.0001},
        {"audusdrmp", 0.0001}, {"cadchfrmp", 0.0001},
        {"cadjpyrmp", 0.01},   {"chfjpyrmp", 0.01},
        {"euraudrmp", 0.0001}, {"eurcadrmp", 0.0001},
        {"eurchfrmp", 0.0001}, {"eurgbprmp", 0.0001},
        {"eurjpyrmp", 0.01},   {"eurnzdrmp", 0.0001},
        {"eurusdrmp", 0.0001}, {"gbpaudrmp", 0.0001},
        {"gbpcadrmp", 0.0001}, {"gbpnzdrmp", 0.0001},
        {"gbpusdrmp", 0.0001}, {"nzdchfrmp", 0.0001},
        {"nzdcadrmp", 0.0001}, {"nzdjpyrmp", 0.01},
        {"usdcadrmp", 0.0001}, {"usdjpyrmp", 0.01}};
    assert(first.values().canonicalFxPipSizes == expectedPipSizes);
    for (const auto& [symbol, pipSize] : first.values().canonicalFxPipSizes)
    {
        assert(!symbol.empty());
        assert(pipSize > 0.0);
    }
    for (const auto& [symbol, pipSize] : std::map<std::string, double>{
             {"audcadrmp", 0.0001}, {"audusdrmp", 0.0001},
             {"eurusdrmp", 0.0001}, {"gbpusdrmp", 0.0001},
             {"usdcadrmp", 0.0001}, {"usdjpyrmp", 0.01}})
        assert(first.values().canonicalFxPipSizes.at(symbol) == pipSize);
    assert(first.FibonacciConfigurationForSymbol("eurusdrmp").absolutePriceTolerance ==
           0.0001);
    assert(first.FibonacciConfigurationForSymbol("usdjpyrmp").absolutePriceTolerance ==
           0.01);
    assert(first.FibonacciConfigurationForSymbol("cadchfrmp").absolutePriceTolerance ==
           0.0001);
    assert(first.FibonacciConfigurationForSymbol("audchfrmp").absolutePriceTolerance ==
           0.0001);
    assert(first.FibonacciConfigurationForSymbol("cadjpyrmp").absolutePriceTolerance ==
           0.01);
    bool unknownRejected = false;
    try
    {
        (void)first.FibonacciConfigurationForSymbol("unknownrmp");
    }
    catch (const std::invalid_argument&)
    {
        unknownRejected = true;
    }
    assert(unknownRejected);
    AssertSemanticFieldChangesIdentity();
}
} // namespace

int main()
{
    TestCanonicalAbsoluteHalfOpenRange();
    TestDstAndGapParity();
    TestProductionConfiguration();
    std::cout << "TG4PreintegrationCausalBoundaryClosureTests passed\n";
}
