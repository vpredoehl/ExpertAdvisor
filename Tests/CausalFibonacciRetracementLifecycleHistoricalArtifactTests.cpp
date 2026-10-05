#include "CausalFibonacciRetracementLifecycleHistoricalArtifact.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace Fib = EA::FibonacciResearch::RetracementLifecycle;
namespace TG3 = EA::TG3;

namespace
{
TG3::ABStructure UpAB(std::size_t availabilityBar = 6)
{
    TG3::ABStructure ab;
    ab.identity.direction = TG3::ABDirection::UpAB;
    ab.identity.aBar = 1;
    ab.identity.aTimestamp = 100;
    ab.identity.bBar = 4;
    ab.identity.bTimestamp = 500;
    ab.identity.availabilityBar = availabilityBar;
    ab.identity.availabilityTimestamp = static_cast<std::int64_t>(
        (availabilityBar + 1) * 100);
    ab.aPrice = 1.0000;
    ab.aConfirmationBar = 3;
    ab.aConfirmationTimestamp = 400;
    ab.bPrice = 1.1000;
    ab.bConfirmationBar = 6;
    ab.bConfirmationTimestamp = 700;
    ab.priceRange = .1000;
    return ab;
}

EA::TG1A::Candle C(std::int64_t timestamp, double open, double high,
                   double low, double close)
{
    return {timestamp, open, high, low, close};
}

Fib::HistoricalRecord Record(Fib::DTargetHypothesis hypothesis,
                             std::size_t availabilityBar)
{
    const TG3::ABStructure ab = UpAB(availabilityBar);
    const double d = hypothesis == Fib::DTargetHypothesis::Extension1272
        ? 1.1272 : 1.1618;
    Fib::Tracker tracker(ab, d);
    const std::size_t bar = availabilityBar;
    const std::int64_t timestamp = ab.identity.availabilityTimestamp;
    tracker.AddCompletedBar(bar, C(timestamp, 1.0700, 1.0750, 1.0600, 1.0650));
    tracker.AddCompletedBar(bar + 1,
        C(timestamp + 100, 1.0650, 1.1300, 1.0640, 1.1280));
    tracker.Finalize();
    return {hypothesis, tracker.GetRecord()};
}

std::string Read(const std::filesystem::path& path)
{
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), {}};
}

void TestRowsPreserveRawLifecycleAndHypotheses()
{
    const Fib::HistoricalRecord first = Record(Fib::DTargetHypothesis::Extension1272, 6);
    const Fib::HistoricalRecord second = Record(Fib::DTargetHypothesis::Extension1618, 6);
    assert(!first.lifecycle.rightCensored);
    assert(second.lifecycle.rightCensored);
    const std::string header = Fib::LifecycleObservationCsvHeader();
    const std::string row = Fib::LifecycleObservationCsvRow(first, "audcadrmp", "15m");
    assert(header.find("retracement_0382_price") != std::string::npos);
    assert(header.find("retracement_0618_price") != std::string::npos);
    assert(header.find("a_penetration_bar") != std::string::npos);
    assert(header.find("a_close_beyond_bar") != std::string::npos);
    assert(header.find("retracement_0382_directional_close_mfe_ab_ranges") !=
           std::string::npos);
    assert(row.find("\"1.0618000000000001\"") != std::string::npos ||
           row.find("\"1.0618\"") != std::string::npos);
    assert(row.find("\"1.272\"") != std::string::npos);
    assert(Fib::LifecycleObservationCsvRow(second, "audcadrmp", "15m")
               .find("\"1.618\"") != std::string::npos);
    assert(row.find("\"false\"") != std::string::npos);
    // Optional event cells are explicit empty CSV strings, not invented values.
    assert(row.find("\"\",\"\",\"\"") != std::string::npos);
}

void TestWriterSortsRecordsDeterministically()
{
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "ea_retracement_artifact_test";
    std::filesystem::remove_all(output);
    Fib::HistoricalArtifactWriter writer(output,
        {"audcadrmp", "15m", "config.conf", "fingerprint", "commit",
         "[start,end)", "reproduce"});
    writer.AddRecord(Record(Fib::DTargetHypothesis::Extension1618, 7));
    writer.AddRecord(Record(Fib::DTargetHypothesis::Extension1618, 6));
    writer.AddRecord(Record(Fib::DTargetHypothesis::Extension1272, 6));
    writer.SetDataQuality({3, 3, 0, 700, 900});
    writer.Complete();
    const std::string observations = Read(output / "observations.csv");
    const std::size_t first1272 = observations.find("\"1.272\"");
    const std::size_t first1618 = observations.find("\"1.618\"");
    assert(first1272 != std::string::npos && first1618 != std::string::npos);
    assert(first1272 < first1618);
    const std::string manifest = Read(output / "manifest.json");
    assert(manifest.find("\"database_access\": \"read_only_repeatable_read\"") !=
           std::string::npos);
    assert(manifest.find("\"observation_count\": 3") != std::string::npos);
    std::filesystem::remove_all(output);
}
} // namespace

int main()
{
    TestRowsPreserveRawLifecycleAndHypotheses();
    TestWriterSortsRecordsDeterministically();
    std::cout << "CausalFibonacciRetracementLifecycleHistoricalArtifactTests passed\n";
}
