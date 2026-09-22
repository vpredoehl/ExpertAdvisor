#include "TG4HistoricalEmpiricalEvaluation.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using EA::TG1A::Candle;
using EA::TG2::ResolutionState;
using EA::TG3::ConfluenceState;
using EA::TG4::AggregateAccumulator;
using EA::TG4::EvaluationConfiguration;
using EA::TG4::HistoricalEvaluator;
using EA::TG4::ObservationRecord;
using EA::TG4::TemporalRange;

constexpr std::int64_t kStart = 1'700'000'000;
constexpr std::int64_t kHour = 3'600;

std::int64_t Timestamp(std::size_t bar)
{
    return kStart + static_cast<std::int64_t>(bar) * kHour;
}

Candle Bar(std::size_t index, double high, double low, double close)
{
    return {Timestamp(index), close, high, low, close};
}

std::vector<Candle> IntegratedCandles()
{
    const std::vector<double> lows{
        10.0, 9.0, 5.0, 8.0, 10.0, 9.0,
        10.0, 7.0, 10.0, 10.0, 7.0, 7.1};
    const std::vector<double> highs{
        20.0, 20.5, 21.0, 23.0, 25.0, 22.0,
        21.0, 20.5, 21.0, 20.0, 8.1, 8.0};
    std::vector<Candle> candles;
    for (std::size_t index = 0; index < lows.size(); ++index)
        candles.push_back(Bar(index, highs[index], lows[index],
                              0.5 * (highs[index] + lows[index])));
    return candles;
}

EvaluationConfiguration Config()
{
    EvaluationConfiguration result;
    result.configurationSchema = "tg4-analysis-configuration-v1";
    result.name = "tg4-test-experimental";
    result.provenance =
        "experimental unit-test convention; not selected from market results";
    result.timeframe = "1h";
    result.candlePeriod = 1;
    result.candleUnit = "hour";
    result.expectedIntervalSeconds = kHour;
    result.materialGapMultiple = 1.5;
    result.minimumHumanReportResolvedN = 2;
    result.maxPendingTG4Records = 64;
    result.geometry.atrPeriod = 3;
    result.geometry.maxCandidates = 64;
    result.referenceBarScale = 50.0;
    result.behavior.retestHorizonBars = 2;
    result.behavior.outerTargetHorizonBars = 3;
    result.behavior.maxActiveBreakObservations = 64;
    result.behavior.maxRetainedBreakObservations = 64;
    result.fibonacci.retracementRatios = {0.25, 0.5};
    result.fibonacci.absolutePriceTolerance = 0.0;
    result.fibonacci.maxConfirmedFractalsPerKind = 32;
    result.fibonacci.maxABAgeBars = 100;
    result.fibonacci.maxActiveABStructures = 32;
    result.fibonacci.maxActiveConfluenceObservations = 64;
    result.fibonacci.maxRetainedConfluenceObservations = 64;
    return result;
}

TemporalRange Range(std::int64_t scoreStart = Timestamp(0))
{
    return {Timestamp(0), scoreStart, Timestamp(12), Timestamp(12)};
}

std::vector<ObservationRecord> Evaluate(
    const std::vector<Candle>& candles,
    TemporalRange range = Range())
{
    std::vector<ObservationRecord> records;
    HistoricalEvaluator evaluator(
        "eurusdrmp", Config(), range,
        [&records](const ObservationRecord& record)
        {
            records.push_back(record);
        });
    for (const Candle& candle : candles) evaluator.AddCompletedBar(candle);
    evaluator.Finalize();
    return records;
}

void TestWilsonKnownCasesAndZeroDenominator()
{
    const auto zero = EA::TG4::Wilson95(0, 0);
    assert(zero.denominator == 0);
    assert(!zero.rate.has_value());
    assert(!zero.lower95.has_value());
    assert(!zero.upper95.has_value());

    const auto half = EA::TG4::Wilson95(5, 5);
    assert(half.denominator == 10);
    assert(std::fabs(*half.rate - 0.5) < 1e-12);
    assert(std::fabs(*half.lower95 - 0.2365930905) < 1e-9);
    assert(std::fabs(*half.upper95 - 0.7634069095) < 1e-9);

    const auto none = EA::TG4::Wilson95(0, 10);
    assert(std::fabs(*none.lower95) < 1e-12);
    assert(std::fabs(*none.upper95 - 0.2775327999) < 1e-9);
}

void TestPartitionAssignment()
{
    assert(EA::TG4::TemporalPartitionName(EA::TG4::PartitionForTimestamp(
        EA::TG4::ParseUtcDateOrTimestamp("2019-12-31T23:59:59Z"))) ==
        "exploratory_2010_2019");
    assert(EA::TG4::TemporalPartitionName(EA::TG4::PartitionForTimestamp(
        EA::TG4::ParseUtcDateOrTimestamp("2020-01-01"))) ==
        "calibration_2020_2022");
    assert(EA::TG4::TemporalPartitionName(EA::TG4::PartitionForTimestamp(
        EA::TG4::ParseUtcDateOrTimestamp("2023-01-01"))) ==
        "validation_2023_2024");
    assert(EA::TG4::TemporalPartitionName(EA::TG4::PartitionForTimestamp(
        EA::TG4::ParseUtcDateOrTimestamp("2025-01-01"))) ==
        "confirmation_2025");
    assert(EA::TG4::TemporalPartitionName(EA::TG4::PartitionForTimestamp(
        EA::TG4::ParseUtcDateOrTimestamp("2026-01-01"))) ==
        "outside_named_study");
}

void TestPreconfirmationNamedRangeCannotLoad2025()
{
    const TemporalRange range = EA::TG4::PreconfirmationStudyRange();
    const std::int64_t confirmationStart =
        EA::TG4::ParseUtcDateOrTimestamp("2025-01-01");
    assert(range.warmupStart ==
           EA::TG4::ParseUtcDateOrTimestamp("2010-01-01"));
    assert(range.scoreStart == range.warmupStart);
    assert(range.scoreEnd == confirmationStart);
    assert(range.outcomeEnd == confirmationStart);
    EA::TG4::ValidateTemporalRange(range);
}

void TestSymbolAwarePipToleranceAndConfigurationFingerprint()
{
    EvaluationConfiguration configuration = Config();
    configuration.fibonacciPriceTolerancePips = 1.0;
    assert(std::fabs(EA::TG4::CanonicalFxPipSize("eurusdrmp") - 0.0001) <
           1e-15);
    assert(std::fabs(EA::TG4::CanonicalFxPipSize("usdjpyrmp") - 0.01) <
           1e-15);
    assert(std::fabs(EA::TG4::EffectiveFibonacciAbsolutePriceTolerance(
                         configuration, "audcadrmp") - 0.0001) < 1e-15);
    assert(std::fabs(EA::TG4::EffectiveFibonacciAbsolutePriceTolerance(
                         configuration, "usdjpyrmp") - 0.01) < 1e-15);

    bool unknownRejected = false;
    try { (void)EA::TG4::CanonicalFxPipSize("unknown"); }
    catch (const std::invalid_argument&) { unknownRejected = true; }
    assert(unknownRejected);

    const std::string frozen = EA::TG4::ConfigurationFingerprint(configuration);
    EvaluationConfiguration changed = configuration;
    changed.referenceBarScale = 14.0;
    assert(EA::TG4::ConfigurationFingerprint(changed) != frozen);
    changed = configuration;
    changed.fibonacci.retracementRatios = {0.6180339887498949};
    assert(EA::TG4::ConfigurationFingerprint(changed) != frozen);
    changed = configuration;
    changed.fibonacciPriceTolerancePips = 0.5;
    assert(EA::TG4::ConfigurationFingerprint(changed) != frozen);

    const std::string metadata = EA::TG4::EffectiveConfigurationJson(
        configuration, Range(), "baseline");
    assert(metadata.find("\"configuration_fingerprint\": \"") !=
           std::string::npos);
    assert(metadata.find("\"price_tolerance_convention\":"
                         "\"canonical_fx_pips\"") != std::string::npos);
    assert(metadata.find("\"usdjpyrmp\":0.01") != std::string::npos);
    assert(metadata.find("\"eurusdrmp\":0.0001") != std::string::npos);
}

void TestFrozenConfigurationLoads(const std::filesystem::path& path)
{
    const EvaluationConfiguration frozen = EA::TG4::LoadConfigurationFile(path);
    assert(frozen.name == "tg4a-first-study-preconfirmation-frozen-v1");
    assert(frozen.referenceBarScale == 14.0);
    assert(frozen.geometry.atrPeriod == 14);
    assert(frozen.fibonacci.retracementRatios.size() == 1);
    assert(frozen.fibonacci.retracementRatios.front() ==
           0.6180339887498949);
    assert(frozen.fibonacciPriceTolerancePips == 1.0);
    assert(EA::TG4::EffectiveFibonacciAbsolutePriceTolerance(
               frozen, "eurusdrmp") == 0.0001);
    assert(EA::TG4::EffectiveFibonacciAbsolutePriceTolerance(
               frozen, "usdjpyrmp") == 0.01);
    const std::string fingerprint =
        EA::TG4::ConfigurationFingerprint(frozen);
    assert(fingerprint == "fnv1a64:bf809ce38a4a444a");
    std::cout << "TG4_FROZEN_CONFIGURATION name=" << frozen.name
              << ",fingerprint=" << fingerprint << '\n';
}

void TestHistoricalStreamingTG4PrefixParityAndFrozenCohort()
{
    const auto candles = IntegratedCandles();
    EvaluationConfiguration configuration = Config();
    std::vector<std::string> streamingEmissions;
    HistoricalEvaluator streaming(
        "eurusdrmp", configuration, Range(),
        [&streamingEmissions](const ObservationRecord& record)
        {
            streamingEmissions.push_back(EA::TG4::FrozenCausalSnapshot(record));
        });

    bool sawPending = false;
    std::string frozenAtObservation;
    for (std::size_t index = 0; index < candles.size(); ++index)
    {
        streaming.AddCompletedBar(candles[index]);
        std::vector<std::string> ignored;
        HistoricalEvaluator historical(
            "eurusdrmp", configuration, Range(),
            [&ignored](const ObservationRecord& record)
            {
                ignored.push_back(EA::TG4::FrozenCausalSnapshot(record));
            });
        for (std::size_t prefix = 0; prefix <= index; ++prefix)
            historical.AddCompletedBar(candles[prefix]);
        assert(streaming.PendingRecords().size() ==
               historical.PendingRecords().size());
        for (std::size_t record = 0;
             record < streaming.PendingRecords().size(); ++record)
            assert(EA::TG4::FrozenCausalSnapshot(
                       streaming.PendingRecords()[record]) ==
                   EA::TG4::FrozenCausalSnapshot(
                       historical.PendingRecords()[record]));
        if (!streaming.PendingRecords().empty() && !sawPending)
        {
            sawPending = true;
            frozenAtObservation = EA::TG4::FrozenCausalSnapshot(
                streaming.PendingRecords().front());
        }
    }
    streaming.Finalize();
    assert(sawPending);
    assert(!streamingEmissions.empty());
    assert(streamingEmissions.front() == frozenAtObservation);

    const auto batch = Evaluate(candles);
    assert(batch.size() == streamingEmissions.size());
    for (std::size_t index = 0; index < batch.size(); ++index)
        assert(EA::TG4::FrozenCausalSnapshot(batch[index]) ==
               streamingEmissions[index]);
}

void TestWarmupWithoutPrePartitionScoring()
{
    const auto candles = IntegratedCandles();
    const auto all = Evaluate(candles);
    assert(!all.empty());
    const std::int64_t breakTimestamp = all.front().behavior.breakEvent.timestamp;
    const auto scored = Evaluate(candles, Range(breakTimestamp));
    assert(scored.size() == all.size());
    const auto excluded = Evaluate(candles, Range(breakTimestamp + 1));
    assert(excluded.empty());
}

ObservationRecord Synthetic(std::uint64_t sequence,
                            bool paired,
                            ResolutionState retest,
                            ResolutionState outer,
                            ConfluenceState confluence)
{
    ObservationRecord record;
    record.eventIdentity = "synthetic-" + std::to_string(sequence);
    record.symbol = sequence % 2 == 0 ? "audcadrmp" : "eurusdrmp";
    record.timeframe = "1h";
    record.partition = EA::TG4::TemporalPartition::Validation2023To2024;
    record.behavior.breakEvent.eventSequence = sequence;
    record.behavior.breakEvent.candidate.direction =
        sequence % 2 == 0 ? EA::TG1A::TrendLineDirection::DTL
                          : EA::TG1A::TrendLineDirection::UTL;
    record.behavior.breakEvent.frozenClassification =
        EA::TG1B::SteepnessClassification::Inner;
    record.behavior.retest.state = retest;
    record.behavior.outerTarget.state = paired
        ? outer : ResolutionState::StructurallyIneligible;
    record.behavior.outerTargetAfterRetest.state =
        paired && retest == ResolutionState::Succeeded
            ? outer : ResolutionState::StructurallyIneligible;
    if (paired)
    {
        EA::TG2::PairedOuterLine line;
        line.candidate.direction = record.behavior.breakEvent.candidate.direction;
        record.behavior.pairedOuter = line;
        EA::TG1B::ClassifiedTrendLineCandidate outerCandidate;
        outerCandidate.classification = EA::TG1B::SteepnessClassification::Outer;
        outerCandidate.calibratedAngleMagnitudeDegrees = 30.0;
        record.frozenOuter = outerCandidate;
    }
    EA::TG1B::ClassifiedTrendLineCandidate innerCandidate;
    innerCandidate.classification = EA::TG1B::SteepnessClassification::Inner;
    innerCandidate.calibratedAngleMagnitudeDegrees = 60.0;
    record.frozenInner = innerCandidate;
    record.confluence.confluenceState = confluence;
    return record;
}

void TestAccountingAndCohortSeparations()
{
    const std::vector<ObservationRecord> records{
        Synthetic(1, true, ResolutionState::Succeeded,
                  ResolutionState::Succeeded, ConfluenceState::Confluence),
        Synthetic(2, true, ResolutionState::Succeeded,
                  ResolutionState::Failed, ConfluenceState::NoConfluence),
        Synthetic(3, true, ResolutionState::Failed,
                  ResolutionState::Succeeded, ConfluenceState::NoConfluence),
        Synthetic(4, false, ResolutionState::Censored,
                  ResolutionState::Failed,
                  ConfluenceState::StructurallyIneligible),
        Synthetic(5, true, ResolutionState::Pending,
                  ResolutionState::Censored, ConfluenceState::Confluence)};
    AggregateAccumulator first;
    AggregateAccumulator second;
    for (const auto& record : records) first.Add(record);
    for (auto it = records.rbegin(); it != records.rend(); ++it) second.Add(*it);

    const EA::TG4::CohortKey overall{
        "__event_weighted__", "all", "all", "all"};
    const auto& one = first.Cohorts().at(overall);
    const auto& two = second.Cohorts().at(overall);
    assert(one.totalObservations == 5);
    assert(one.paired == 4 && one.unpaired == 1);
    assert(one.confluent == 2 && one.nonConfluent == 2 &&
           one.confluenceIneligible == 1);
    assert(one.retest.successes == 2 && one.retest.failures == 1 &&
           one.retest.censored == 1 && one.retest.pending == 1);
    assert(one.outerTarget.successes == 2 && one.outerTarget.failures == 1 &&
           one.outerTarget.censored == 1 &&
           one.outerTarget.structurallyIneligible == 1);
    assert(one.totalObservations == two.totalObservations);
    assert(one.outerTarget.successes == two.outerTarget.successes);
    assert(first.Cohorts().contains(
        {"__event_weighted__", "all", "all", "pairing=paired"}));
    assert(first.Cohorts().contains(
        {"__event_weighted__", "all", "all", "pairing=unpaired"}));
    assert(first.Cohorts().contains(
        {"__event_weighted__", "all", "all", "retest=succeeded"}));
    assert(first.Cohorts().contains(
        {"__event_weighted__", "all", "all", "confluence=confluent"}));
    assert(first.Comparisons().contains(
        {"__event_weighted__", "all", "all", "retest_vs_no_retest"}));
    assert(first.Comparisons().contains(
        {"__event_weighted__", "all", "all", "confluence_vs_no_confluence"}));
}

void TestDeterministicObservationSchemaAndOrdering()
{
    const auto records = Evaluate(IntegratedCandles());
    assert(!records.empty());
    const std::string header = EA::TG4::ObservationCsvHeader();
    const std::string first = EA::TG4::ObservationCsvRow(records.front(), Config());
    const std::string second = EA::TG4::ObservationCsvRow(records.front(), Config());
    assert(first == second);
    assert(header.find("schema_version,event_identity,symbol,timeframe") == 0);
    assert(first.find("tg4-inner-break-observation-v1") == 0);
    assert(std::count(header.begin(), header.end(), ',') ==
           std::count(first.begin(), first.end(), ','));
    for (std::size_t index = 1; index < records.size(); ++index)
        assert(records[index - 1].behavior.breakEvent.eventSequence <
               records[index].behavior.breakEvent.eventSequence);
}

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    assert(input);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

void TestDeterministicArtifacts()
{
    const auto records = Evaluate(IntegratedCandles());
    assert(!records.empty());
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path first = std::filesystem::temp_directory_path() /
        ("ea_tg4_artifacts_first_" + std::to_string(nonce));
    const std::filesystem::path second = std::filesystem::temp_directory_path() /
        ("ea_tg4_artifacts_second_" + std::to_string(nonce));
    {
        EA::TG4::StudyArtifactWriter writer(first, Config(), Range(), "baseline");
        for (const auto& record : records) writer.Write(record);
        EA::TG4::DataQualityAudit audit;
        audit.symbol = "eurusdrmp";
        audit.usableRows = IntegratedCandles().size();
        writer.AddDataQuality(audit);
        writer.Complete();
    }
    {
        EA::TG4::StudyArtifactWriter writer(second, Config(), Range(), "baseline");
        for (const auto& record : records) writer.Write(record);
        EA::TG4::DataQualityAudit audit;
        audit.symbol = "eurusdrmp";
        audit.usableRows = IntegratedCandles().size();
        writer.AddDataQuality(audit);
        writer.Complete();
    }
    const std::vector<std::string> artifacts{
        "observations.csv", "cohorts.csv", "comparisons.csv",
        "angle_distributions.csv", "equal_symbol_rates.csv",
        "data_quality.csv", "data_gaps.csv", "metadata.json", "report.md"};
    for (const std::string& artifact : artifacts)
        assert(ReadFile(first / artifact) == ReadFile(second / artifact));
    std::filesystem::remove_all(first);
    std::filesystem::remove_all(second);
}

void TestGapAndDuplicateHandling()
{
    EvaluationConfiguration configuration = Config();
    std::vector<ObservationRecord> records;
    HistoricalEvaluator gaps(
        "eurusdrmp", configuration,
        {Timestamp(0), Timestamp(0), Timestamp(4), Timestamp(4)},
        [&records](const ObservationRecord& record) { records.push_back(record); });
    gaps.AddCompletedBar(Bar(0, 11.0, 9.0, 10.0));
    gaps.AddCompletedBar(Bar(1, 11.0, 9.0, 10.0));
    Candle delayed = Bar(3, 11.0, 9.0, 10.0);
    gaps.AddCompletedBar(delayed);
    assert(gaps.DataQuality().materialGaps.size() == 1);

    bool duplicateRejected = false;
    try
    {
        gaps.AddCompletedBar(delayed);
    }
    catch (const std::invalid_argument&) { duplicateRejected = true; }
    assert(duplicateRejected);
    assert(gaps.DataQuality().duplicateTimestamps == 1);
}

void TestBoundedRepresentativeLargeStream()
{
    EvaluationConfiguration configuration = Config();
    configuration.timeframe = "1h";
    configuration.geometry.maxFractalAnchorLookbackBars = 32;
    configuration.geometry.maxConfirmedFractalsPerKind = 8;
    configuration.geometry.maxCandidateAgeBars = 32;
    configuration.geometry.maxCandidates = 32;
    configuration.behavior.retestHorizonBars = 3;
    configuration.behavior.outerTargetHorizonBars = 3;
    configuration.behavior.maxActiveBreakObservations = 32;
    configuration.behavior.maxRetainedBreakObservations = 32;
    configuration.fibonacci.maxConfirmedFractalsPerKind = 8;
    configuration.fibonacci.maxABAgeBars = 32;
    configuration.fibonacci.maxActiveABStructures = 16;
    configuration.fibonacci.maxActiveConfluenceObservations = 32;
    configuration.fibonacci.maxRetainedConfluenceObservations = 32;
    configuration.maxPendingTG4Records = 32;

    constexpr std::size_t barCount = 50'000;
    constexpr std::int64_t start = 1'262'304'000;
    const TemporalRange range{
        start, start, start + static_cast<std::int64_t>(barCount) * kHour,
        start + static_cast<std::int64_t>(barCount) * kHour};
    std::size_t emitted = 0;
    HistoricalEvaluator evaluator(
        "eurusdrmp", configuration, range,
        [&emitted](const ObservationRecord&) { ++emitted; });
    const auto began = std::chrono::steady_clock::now();
    const std::vector<Candle> pattern = IntegratedCandles();
    for (std::size_t index = 0; index < barCount; ++index)
    {
        Candle candle = pattern[index % pattern.size()];
        candle.timestamp = start + static_cast<std::int64_t>(index) * kHour;
        evaluator.AddCompletedBar(candle);
    }
    evaluator.Finalize();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - began);
    assert(evaluator.PendingRecords().empty());
    assert(evaluator.PeakPendingRecordCount() <=
           configuration.maxPendingTG4Records);
    assert(evaluator.EmittedRecordCount() == emitted);
    assert(emitted > 1'000);
    assert(elapsed.count() < 20'000);
    std::cout << "TG4_BOUNDED_STREAM bars=" << barCount
              << ",observations=" << emitted
              << ",peak_pending=" << evaluator.PeakPendingRecordCount()
              << ",milliseconds=" << elapsed.count() << '\n';
}

} // namespace

int main(int argc, char* argv[])
{
    assert(argc == 2);
    TestWilsonKnownCasesAndZeroDenominator();
    TestPartitionAssignment();
    TestPreconfirmationNamedRangeCannotLoad2025();
    TestSymbolAwarePipToleranceAndConfigurationFingerprint();
    TestFrozenConfigurationLoads(argv[1]);
    TestHistoricalStreamingTG4PrefixParityAndFrozenCohort();
    TestWarmupWithoutPrePartitionScoring();
    TestAccountingAndCohortSeparations();
    TestDeterministicObservationSchemaAndOrdering();
    TestDeterministicArtifacts();
    TestGapAndDuplicateHandling();
    TestBoundedRepresentativeLargeStream();
    std::cout << "TG4HistoricalEmpiricalEvaluationTests passed\n";
    return 0;
}
