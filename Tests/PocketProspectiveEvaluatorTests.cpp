#include "PocketProspectiveEvaluator.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <limits>

namespace
{
using namespace EA::Pocket;
using namespace EA::Pocket::Prospective;

CompletedBar Bar(std::int64_t t, double open, double high, double low, double close)
{ return {t, open, high, low, close}; }

std::vector<CompletedBar> ReferenceBars(std::size_t count, std::int64_t start = 1262304000)
{
    std::vector<CompletedBar> bars;
    for (std::size_t i = 0; i < count; ++i) bars.push_back(Bar(start + static_cast<std::int64_t>(i) * 900, 5, 10, 0, 5));
    return bars;
}

PocketObservation BullishObservation(std::size_t confirmation = 21, std::int64_t timestamp = 1262322900)
{
    return {PocketDirection::Bullish, {10, 11}, confirmation - 1, timestamp - 900,
        confirmation, timestamp, confirmation, timestamp, "15m_completed"};
}

PocketObservation BearishObservation(std::size_t confirmation = 21, std::int64_t timestamp = 1262322900)
{
    return {PocketDirection::Bearish, {9, 10}, confirmation - 1, timestamp - 900,
        confirmation, timestamp, confirmation, timestamp, "15m_completed"};
}

void TestConfigurationAndFirewall()
{
    const RunConfiguration config = LoadAndValidateConfiguration("Scripts/pocket_prospective_preconfirmation_v1.conf");
    VerifyFrozenProtocolDocument();
    assert(config.configurationSha256 == Sha256(CanonicalPayload()));
    assert(config.lookbacks == kLookbacks && config.horizons == kHorizons);
    assert(SourceFor(config, "USDJPY").table == "usdjpyrmp");
    bool unknown = false; try { (void)SourceFor(config, "XAUUSD"); } catch (const std::invalid_argument&) { unknown = true; } assert(unknown);
    auto bars = ReferenceBars(22);
    bars.back().timestamp = kPreconfirmationEnd;
    bool firewall = false;
    try { (void)PreflightCompletedBars(config, "AUDCAD", "audcadrmp", "forex:postgresql:1", bars); }
    catch (const std::invalid_argument&) { firewall = true; }
    assert(firewall);
    bars = ReferenceBars(22);
    (void)PreflightCompletedBars(config, "AUDCAD", "audcadrmp", "forex:postgresql:1", bars);
    bars[3].timestamp = bars[2].timestamp;
    bool duplicate = false;
    try { (void)PreflightCompletedBars(config, "AUDCAD", "audcadrmp", "forex:postgresql:1", bars); }
    catch (const std::invalid_argument&) { duplicate = true; }
    assert(duplicate);
}

void TestDetectorParityAndSensitivity()
{
    auto bars = ReferenceBars(15);
    const auto start = bars.back().timestamp + 900;
    bars.push_back(Bar(start, 10, 12, 9, 11));
    bars.push_back(Bar(start + 900, 12, 13, 11.5, 12));
    CausalPocketDetector defaultDetector("15m_completed");
    CausalPocketDetector explicit15("15m_completed", 15);
    std::optional<PocketObservation> left, right;
    for (const auto& bar : bars) { left = defaultDetector.AddCompletedBar(bar); right = explicit15.AddCompletedBar(bar); }
    assert(left && right && left->eventBar == right->eventBar && left->range.upper == right->range.upper);
    for (const std::size_t lookback : {std::size_t{10}, std::size_t{20}}) {
        CausalPocketDetector detector("15m_completed", lookback);
        std::optional<PocketObservation> observed;
        for (const auto& bar : bars) observed = detector.AddCompletedBar(bar);
        assert((lookback == 10 && observed.has_value()) || (lookback == 20 && !observed.has_value()));
    }
    bool zeroRejected = false; try { CausalPocketDetector invalid("15m_completed", 0); (void)invalid; }
    catch (const std::invalid_argument&) { zeroRejected = true; } assert(zeroRejected);
}

void TestBoundedOutcomes()
{
    auto bars = ReferenceBars(22); // confirmation is index 21
    bars[21] = Bar(bars[21].timestamp, 10.5, 11, 10, 10.5);
    const auto bullish = BullishObservation(21, bars[21].timestamp);
    bars.push_back(Bar(bars[21].timestamp + 900, 11, 12, 11, 11)); // touch equality
    bars.push_back(Bar(bars[21].timestamp + 1800, 11, 11, 9, 10)); // close equality
    bars.push_back(Bar(bars[21].timestamp + 2700, 10, 10, 9, 9.5));
    bars.push_back(Bar(bars[21].timestamp + 3600, 9.5, 12, 9, 11));
    // This malformed post-horizon bar proves the evaluator never reads beyond h=4.
    bars.push_back(Bar(bars[21].timestamp + 4500, std::numeric_limits<double>::quiet_NaN(), 1, 0, 0));
    const OutcomeLabel outcome = EvaluateBoundedOutcome(bullish, bars, 4, kPreconfirmationEnd);
    assert(outcome.complete && outcome.touchAt == 1 && outcome.closeAt == 2);
    assert(outcome.mfe == 1.5 && outcome.mae == 1.5 && outcome.directionalCloseReturn == 0.5);

    auto bearishBars = ReferenceBars(22);
    bearishBars[21] = Bar(bearishBars[21].timestamp, 9.5, 10, 9, 9.5);
    bearishBars.push_back(Bar(bearishBars[21].timestamp + 900, 9, 9, 8, 9)); // touch equality low boundary
    bearishBars.push_back(Bar(bearishBars[21].timestamp + 1800, 9, 11, 8, 10)); // close equality
    bearishBars.push_back(Bar(bearishBars[21].timestamp + 2700, 10, 11, 8, 9));
    bearishBars.push_back(Bar(bearishBars[21].timestamp + 3600, 9, 10, 7, 8));
    const OutcomeLabel bearish = EvaluateBoundedOutcome(BearishObservation(21, bearishBars[21].timestamp), bearishBars, 4, kPreconfirmationEnd);
    assert(bearish.complete && bearish.touchAt == 1 && bearish.closeAt == 2);

    auto gap = bars; gap[23].timestamp += 1;
    const OutcomeLabel censored = EvaluateBoundedOutcome(bullish, gap, 4, kPreconfirmationEnd);
    assert(!censored.complete && censored.censor == CensorReason::Gap && censored.validFutureBars == 1);
    const OutcomeLabel tail = EvaluateBoundedOutcome(bullish, std::vector<CompletedBar>(bars.begin(), bars.begin() + 23), 4, kPreconfirmationEnd);
    assert(!tail.complete && tail.censor == CensorReason::Tail);
}

EvaluatedObservation Record(std::string symbol, std::size_t confirmation, std::int64_t timestamp, bool touch)
{
    EvaluatedObservation record;
    record.identity = symbol + std::to_string(confirmation); record.symbol = std::move(symbol);
    record.partition = "exploratory"; record.lookback = 15; record.observation = BullishObservation(confirmation, timestamp);
    for (std::size_t i = 0; i < 3; ++i) { record.outcomes[i].horizon = kHorizons[i]; record.outcomes[i].complete = true; }
    if (touch) record.outcomes[0].touchAt = 1;
    return record;
}

void TestMetricsBootstrapAndThinning()
{
    std::vector<EvaluatedObservation> records;
    records.push_back(Record("AUDCAD", 21, 1262322900, true));
    records.push_back(Record("AUDCAD", 85, 1262380500, false));
    records.push_back(Record("AUDCAD", 149, 1262438100, true));
    std::vector<const EvaluatedObservation*> pointers; for (const auto& record : records) pointers.push_back(&record);
    const auto first = AggregateMetric(pointers, 0, DerivedBootstrapSeed(FrozenConfiguration().configurationSha256), 2000);
    const auto second = AggregateMetric(pointers, 0, DerivedBootstrapSeed(FrozenConfiguration().configurationSha256), 2000);
    assert(first.eligible == 3 && first.complete == 3 && first.touches == 2 && first.touchRate == second.touchRate);
    assert(first.touchLower95 == second.touchLower95 && first.touchUpper95 == second.touchUpper95);
    const auto thinned = GreedyTemporalThin(records);
    assert(thinned.size() == 3); // equality at last + 64 must be retained
    records[1].observation.confirmationBar = 84;
    assert(GreedyTemporalThin(records).size() == 2);
}

void TestAtomicArtifacts()
{
    const auto target = std::filesystem::temp_directory_path() / ("pocket_phase4_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(target);
    std::vector<EvaluatedObservation> records{Record("AUDCAD", 21, 1262322900, true)};
    ImmutableArtifactWriter(target).Publish(FrozenConfiguration(), "synthetic=true", records);
    ImmutableArtifactWriter::VerifyDirectory(target);
    bool existing = false; try { ImmutableArtifactWriter again(target); (void)again; }
    catch (const std::invalid_argument&) { existing = true; } assert(existing);
    std::ofstream tamper(target / "observations.csv", std::ios::app); tamper << "tamper\n"; tamper.close();
    bool tampered = false; try { ImmutableArtifactWriter::VerifyDirectory(target); }
    catch (const std::invalid_argument&) { tampered = true; } assert(tampered);
    std::filesystem::remove_all(target);
}

void TestScientificArtifactEquivalenceAcrossExecutableProvenance()
{
    const auto root = std::filesystem::temp_directory_path() / ("pocket_phase4_equivalence_" + std::to_string(::getpid()));
    const auto monolith = root / "lstm";
    const auto dedicated = root / "pocket_research";
    std::filesystem::remove_all(root);
    const std::vector<EvaluatedObservation> records{Record("AUDCAD", 21, 1262322900, true)};
    ImmutableArtifactWriter(monolith).Publish(FrozenConfiguration(),
        "git=test;executable_sha256=lstm", records);
    ImmutableArtifactWriter(dedicated).Publish(FrozenConfiguration(),
        "git=test;executable_sha256=pocket_research", records);
    for (const char* name : {"configuration.conf", "observations.csv", "aggregates.csv"})
        assert(ReadTextFile(monolith / name) == ReadTextFile(dedicated / name));
    const auto withoutProvenance = [](std::string manifest) {
        const std::size_t start = manifest.find("provenance=");
        assert(start != std::string::npos);
        const std::size_t end = manifest.find('\n', start);
        manifest.erase(start, end - start + 1);
        return manifest;
    };
    assert(withoutProvenance(ReadTextFile(monolith / "manifest.txt")) ==
           withoutProvenance(ReadTextFile(dedicated / "manifest.txt")));
    std::filesystem::remove_all(root);
}
} // namespace

int main()
{
    TestConfigurationAndFirewall();
    TestDetectorParityAndSensitivity();
    TestBoundedOutcomes();
    TestMetricsBootstrapAndThinning();
    TestAtomicArtifacts();
    TestScientificArtifactEquivalenceAcrossExecutableProvenance();
    std::cout << "PocketProspectiveEvaluatorTests passed\n";
}
