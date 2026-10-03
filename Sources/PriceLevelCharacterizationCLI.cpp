#include "CanonicalMarketDataRange.hpp"
#include "CausalPriceLevelEngine.hpp"
#include "HistoricalFxTimestamp.hpp"
#include "PriceLevelCharacterization.hpp"

#include <pqxx/pqxx>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace PL = EA::PriceLevel;
namespace PLC = EA::PriceLevel::Characterization;

constexpr std::array<const char*, 28> kDefaultSymbols{
    "audcadrmp", "audchfrmp", "audjpyrmp", "audnzdrmp", "audusdrmp",
    "cadchfrmp", "cadjpyrmp", "chfjpyrmp", "euraudrmp", "eurcadrmp",
    "eurchfrmp", "eurgbprmp", "eurjpyrmp", "eurnzdrmp", "eurusdrmp",
    "gbpaudrmp", "gbpcadrmp", "gbpchfrmp", "gbpjpyrmp", "gbpnzdrmp",
    "gbpusdrmp", "nzdcadrmp", "nzdchfrmp", "nzdjpyrmp", "nzdusdrmp",
    "usdcadrmp", "usdchfrmp", "usdjpyrmp"};

std::vector<std::string> DefaultSymbols()
{
    std::vector<std::string> result;
    result.reserve(kDefaultSymbols.size());
    for (const char* symbol : kDefaultSymbols)
    {
        std::string value(symbol);
        result.push_back(std::move(value));
    }
    return result;
}

struct Options final
{
    PriceTP start;
    PriceTP end;
    std::filesystem::path output;
    std::optional<std::string> connection;
    std::vector<std::string> symbols = DefaultSymbols();
    std::vector<std::size_t> radii{1, 2, 3, 4, 6, 8};
    std::vector<std::size_t> scaleLookbacks{32, 96, 384};
    std::optional<PL::Configuration> v1Candidate;
};

std::vector<std::string> Split(std::string_view text, char delimiter)
{
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t end = text.find(delimiter, begin);
        const std::string value(text.substr(begin, end - begin));
        if (value.empty()) throw std::invalid_argument("PRICE_LEVEL_CLI_LIST_EMPTY_ITEM");
        result.push_back(value);
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return result;
}

std::vector<std::size_t> ParseSizes(std::string_view value)
{
    std::vector<std::size_t> result;
    for (const std::string& item : Split(value, ','))
    {
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(item, &consumed);
        if (consumed != item.size() || parsed == 0 ||
            parsed > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("PRICE_LEVEL_CLI_SIZE_INVALID");
        result.push_back(static_cast<std::size_t>(parsed));
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
        throw std::invalid_argument("PRICE_LEVEL_CLI_SIZE_DUPLICATE");
    return result;
}

PriceTP ParseUtcDate(std::string_view source)
{
    if (source.size() != 10 || source[4] != '-' || source[7] != '-')
        throw std::invalid_argument("PRICE_LEVEL_CLI_DATE_MUST_BE_YYYY_MM_DD");
    const auto number = [&source](std::size_t begin, std::size_t length) {
        int value = 0;
        for (std::size_t index = begin; index < begin + length; ++index)
        {
            if (source[index] < '0' || source[index] > '9')
                throw std::invalid_argument("PRICE_LEVEL_CLI_DATE_INVALID");
            value = value * 10 + (source[index] - '0');
        }
        return value;
    };
    const std::chrono::year_month_day date{
        std::chrono::year{number(0, 4)}, std::chrono::month{static_cast<unsigned>(number(5, 2))},
        std::chrono::day{static_cast<unsigned>(number(8, 2))}};
    if (!date.ok()) throw std::invalid_argument("PRICE_LEVEL_CLI_DATE_INVALID");
    return PriceTP{std::chrono::sys_days{date}};
}

PL::Configuration ParseV1Candidate(std::string_view source)
{
    const std::vector<std::string> values = Split(source, ',');
    if (values.size() != 5)
        throw std::invalid_argument("PRICE_LEVEL_CLI_V1_CANDIDATE_REQUIRES_5_FIELDS");
    const auto sizeAt = [&values](std::size_t index) {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(values[index], &consumed);
        if (consumed != values[index].size() || value == 0 ||
            value > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("PRICE_LEVEL_CLI_V1_CANDIDATE_SIZE_INVALID");
        return static_cast<std::size_t>(value);
    };
    std::size_t widthConsumed = 0;
    const double width = std::stod(values[1], &widthConsumed);
    if (widthConsumed != values[1].size())
        throw std::invalid_argument("PRICE_LEVEL_CLI_V1_CANDIDATE_WIDTH_INVALID");
    PL::Configuration result{sizeAt(0), width, sizeAt(2), sizeAt(3), sizeAt(4),
                             std::chrono::seconds{900}};
    PL::ValidateConfiguration(result);
    return result;
}

Options ParseOptions(int argc, const char* const argv[])
{
    Options result;
    bool startSet = false;
    bool endSet = false;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view option(argv[index]);
        const auto require = [&]() -> std::string_view {
            if (++index >= argc)
                throw std::invalid_argument("PRICE_LEVEL_CLI_OPTION_VALUE_MISSING:" +
                                            std::string(option));
            return argv[index];
        };
        if (option == "--start") { result.start = ParseUtcDate(require()); startSet = true; }
        else if (option == "--end") { result.end = ParseUtcDate(require()); endSet = true; }
        else if (option == "--output-dir") result.output = std::string(require());
        else if (option == "--connection") result.connection = std::string(require());
        else if (option == "--symbols") result.symbols = Split(require(), ',');
        else if (option == "--pivot-radii") result.radii = ParseSizes(require());
        else if (option == "--scale-lookbacks") result.scaleLookbacks = ParseSizes(require());
        else if (option == "--v1-candidate") result.v1Candidate = ParseV1Candidate(require());
        else if (option == "--help")
        {
            std::cout << "usage: price_level_characterization --start YYYY-MM-DD --end YYYY-MM-DD "
                      << "--output-dir DIRECTORY [--symbols a,b] [--pivot-radii 1,2,3,4,6,8] "
                      << "[--scale-lookbacks 32,96,384] "
                      << "[--v1-candidate radius,width,max_active,max_age,max_evidence] "
                      << "[--connection CONNECTION]\n";
            std::exit(0);
        }
        else throw std::invalid_argument("PRICE_LEVEL_CLI_UNKNOWN_OPTION:" + std::string(option));
    }
    if (!startSet || !endSet || result.output.empty())
        throw std::invalid_argument("PRICE_LEVEL_CLI_START_END_AND_OUTPUT_REQUIRED");
    if (!(result.start < result.end))
        throw std::invalid_argument("PRICE_LEVEL_CLI_START_MUST_PRECEDE_END");
    if (result.symbols.empty()) throw std::invalid_argument("PRICE_LEVEL_CLI_SYMBOLS_EMPTY");
    return result;
}

std::string DefaultConnection()
{
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* user = std::getenv("FOREX_DB_USER");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" + std::string(host && *host ? host : "127.0.0.1") +
        " gssencmode=disable user=" + std::string(user && *user ? user : "pqxx") +
        " dbname=" + std::string(database && *database ? database : "forex") +
        " application_name=price_level_phase5a_characterization_read_only";
}

int UtcYear(PriceTP time)
{
    return static_cast<int>(std::chrono::year_month_day{
        std::chrono::floor<std::chrono::days>(time)}.year());
}

PL::CompletedBar ParseBar(const std::string& timestamp, double open, double high,
                          double low, double close)
{
    PriceTP instant;
    if (!EA::HistoricalFxTimestamp::ParseNewYorkCivilTimestamp(timestamp, instant))
        throw std::runtime_error("PRICE_LEVEL_SOURCE_TIMESTAMP_UNPARSEABLE:" + timestamp);
    return {std::chrono::sys_seconds{instant.time_since_epoch()}, open, high, low, close};
}

struct ScaleAccumulator final
{
    PLC::RunningDistribution range;
    PLC::RunningDistribution mean;
    PLC::RunningDistribution median;

    void AddRange(double value) { range.Add(value); }
    void AddScale(const PLC::PrecedingRangeScale::Sample& value)
    {
        if (value.count == 0) return;
        mean.Add(value.mean);
        median.Add(value.median);
    }
};

struct DetectorAccumulator final
{
    std::array<std::uint64_t, 9> interactions{};
    PLC::RunningDistribution activeCounts;
    PLC::RunningDistribution lifetimes;
    PLC::RunningDistribution pivotsPerEndedLevel;
    std::uint64_t endedLevels = 0;
    std::uint64_t saturatedEndedLevels = 0;
    std::map<std::string, PL::Level> knownLevels;
    std::size_t evidenceCap = 0;

    explicit DetectorAccumulator(std::size_t cap = 0) : evidenceCap(cap) {}

    static std::size_t Index(PL::InteractionKind kind)
    {
        return static_cast<std::size_t>(kind);
    }

    void Add(std::size_t currentBar, const PL::Update& update)
    {
        activeCounts.Add(static_cast<double>(update.activeLevels.size()));
        for (const PL::Observation& observation : update.observations)
        {
            ++interactions[Index(observation.kind)];
            if (observation.kind != PL::InteractionKind::level_expired &&
                observation.kind != PL::InteractionKind::level_evicted)
                continue;
            const auto found = knownLevels.find(observation.levelIdentity);
            if (found == knownLevels.end()) continue;
            ++endedLevels;
            lifetimes.Add(static_cast<double>(currentBar - found->second.availableBar));
            pivotsPerEndedLevel.Add(static_cast<double>(found->second.pivotObservationCount));
            if (found->second.retainedPivotEvidence.size() >= evidenceCap)
                ++saturatedEndedLevels;
            knownLevels.erase(found);
        }
        for (const PL::Level& level : update.activeLevels)
            knownLevels[level.identity] = level;
    }
};

void WriteDistribution(std::ostream& out, const PLC::RunningDistribution& value)
{
    out << value.count << ',' << value.retainedSampleSize() << ',' << value.mean() << ','
        << (value.count ? value.minimum : 0.0) << ','
        << (value.count ? value.Quantile(0.5) : 0.0) << ','
        << (value.count ? value.Quantile(0.9) : 0.0) << ','
        << (value.count ? value.maximum : 0.0);
}

void WriteDetector(std::ostream& out, std::string_view scope, std::string_view symbol,
                   int year, const DetectorAccumulator& value)
{
    out << scope << ',' << symbol << ',' << year;
    for (std::uint64_t count : value.interactions) out << ',' << count;
    out << ',' << value.endedLevels << ',' << value.saturatedEndedLevels << ',';
    WriteDistribution(out, value.activeCounts);
    out << ',';
    WriteDistribution(out, value.lifetimes);
    out << ',';
    WriteDistribution(out, value.pivotsPerEndedLevel);
    out << '\n';
}

void WriteScale(std::ostream& out, std::string_view scope, std::string_view symbol,
                int year, std::size_t radius, std::size_t lookback,
                std::string_view timing, const ScaleAccumulator& value)
{
    const auto write = [&out](std::string_view statistic,
                              const PLC::RunningDistribution& distribution) {
        out << ',' << statistic << ',';
        WriteDistribution(out, distribution);
        out << '\n';
    };
    out << scope << ',' << symbol << ',' << year << ',' << radius << ',' << lookback
        << ',' << timing;
    write("range", value.range);
    if (lookback == 0) return;
    out << scope << ',' << symbol << ',' << year << ',' << radius << ',' << lookback
        << ',' << timing;
    write("preceding_range_mean", value.mean);
    out << scope << ',' << symbol << ',' << year << ',' << radius << ',' << lookback
        << ',' << timing;
    write("preceding_range_median", value.median);
}

int Run(const Options& options)
{
    const std::filesystem::path incompleteOutput =
        options.output.string() + ".incomplete";
    if (std::filesystem::exists(options.output) ||
        std::filesystem::exists(incompleteOutput))
        throw std::invalid_argument("PRICE_LEVEL_OUTPUT_DIRECTORY_ALREADY_EXISTS");
    std::filesystem::create_directories(incompleteOutput);

    std::ofstream pivotFile(incompleteOutput / "pivot_radius.csv");
    std::ofstream scaleFile(incompleteOutput / "scale.csv");
    std::ofstream detectorFile(incompleteOutput / "detector_metrics.csv");
    if (!pivotFile || !scaleFile || !detectorFile)
        throw std::runtime_error("PRICE_LEVEL_OUTPUT_OPEN_FAILED");
    pivotFile << "scope,symbol,radius,bars,strict_pivot_highs,strict_pivot_lows,total_pivots,pivots_per_100_bars\n";
    scaleFile << "scope,symbol,year,pivot_radius,lookback,timing,statistic,count,retained_sample_size,mean,min,p50,p90,max\n";
    detectorFile << "scope,symbol,year,level_established,level_reinforced,touch,cross_up,cross_down,retest,role_reversal,level_expired,level_evicted,ended_levels,saturated_ended_levels,active_count_samples,active_count_sample_size,active_count_mean,active_count_min,active_count_p50,active_count_p90,active_count_max,lifetime_samples,lifetime_sample_size,lifetime_mean,lifetime_min,lifetime_p50,lifetime_p90,lifetime_max,pivot_observation_samples,pivot_observation_sample_size,pivot_observation_mean,pivot_observation_min,pivot_observation_p50,pivot_observation_p90,pivot_observation_max\n";
    pivotFile << std::setprecision(17);
    scaleFile << std::setprecision(17);
    detectorFile << std::setprecision(17);

    const auto started = std::chrono::steady_clock::now();
    std::vector<PLC::PivotCounts> aggregatePivot(options.radii.size());
    std::uint64_t aggregateBars = 0;
    std::map<int, ScaleAccumulator> aggregateRangesByYear;
    std::map<std::tuple<int, std::size_t, std::size_t, std::string>, ScaleAccumulator>
        aggregatePivotScales;
    DetectorAccumulator aggregateDetector(options.v1Candidate ?
        options.v1Candidate->maxRetainedPivotEvidence : 0);
    std::map<int, DetectorAccumulator> aggregateDetectorsByYear;

    pqxx::connection connection(options.connection.value_or(DefaultConnection()));
    pqxx::read_transaction transaction(connection);
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    const EA::CanonicalMarketData::AbsoluteHalfOpenRange range{options.start, options.end};

    for (const std::string& symbol : options.symbols)
    {
        PLC::StrictPivotCharacterizer pivots(options.radii);
        std::vector<PLC::PrecedingRangeScale> scales;
        scales.reserve(options.scaleLookbacks.size());
        for (std::size_t lookback : options.scaleLookbacks) scales.emplace_back(lookback);
        std::map<int, ScaleAccumulator> rangesByYear;
        std::map<std::tuple<int, std::size_t, std::size_t, std::string>, ScaleAccumulator>
            pivotScales;
        std::optional<PL::CausalPriceLevelEngine> detector;
        std::map<int, DetectorAccumulator> detectorsByYear;
        DetectorAccumulator detectorAll(options.v1Candidate ?
            options.v1Candidate->maxRetainedPivotEvidence : 0);
        if (options.v1Candidate)
            detector.emplace(symbol, *options.v1Candidate);

        const std::string query = EA::CanonicalMarketData::CanonicalHalfOpenCandlestickCte(
            transaction, symbol, range) +
            "SELECT to_char(dt,'YYYY-MM-DD HH24:MI:SS'),open::double precision,"
            "high::double precision,low::double precision,close::double precision "
            "FROM bounded ORDER BY dt";
        auto stream = transaction.stream<std::string, double, double, double, double>(query);
        std::size_t barIndex = 0;
        for (const auto& [timestamp, open, high, low, close] : stream)
        {
            const PL::CompletedBar bar = ParseBar(timestamp, open, high, low, close);
            const int year = UtcYear(PriceTP{bar.barStart.time_since_epoch()});
            std::vector<PLC::PrecedingRangeScale::Sample> before;
            before.reserve(scales.size());
            for (const PLC::PrecedingRangeScale& scale : scales)
                before.push_back(scale.BeforeCurrentBar());
            pivots.AddCompletedBar(bar, before, options.scaleLookbacks,
                [&](const PLC::PivotScaleSample& sample) {
                    const int pivotYear = UtcYear(PriceTP{sample.pivotBarStart.time_since_epoch()});
                    const auto pivotKey = std::make_tuple(pivotYear, sample.pivotRadius,
                        sample.scaleLookback, std::string{"pivot_time"});
                    pivotScales[pivotKey].AddScale(sample.pivotTimeScale);
                    aggregatePivotScales[pivotKey].AddScale(sample.pivotTimeScale);
                    const auto confirmationKey = std::make_tuple(pivotYear, sample.pivotRadius,
                        sample.scaleLookback, std::string{"confirmation_time"});
                    pivotScales[confirmationKey].AddScale(sample.confirmationTimeScale);
                    aggregatePivotScales[confirmationKey].AddScale(sample.confirmationTimeScale);
                });
            const double barRange = bar.high - bar.low;
            rangesByYear[year].AddRange(barRange);
            aggregateRangesByYear[year].AddRange(barRange);
            for (PLC::PrecedingRangeScale& scale : scales) scale.AddCompletedRange(barRange);
            if (detector)
            {
                const PL::Update update = detector->AddCompletedBar(bar);
                detectorAll.Add(barIndex, update);
                aggregateDetector.Add(barIndex, update);
                auto [iterator, inserted] = detectorsByYear.try_emplace(
                    year, options.v1Candidate->maxRetainedPivotEvidence);
                (void)inserted;
                iterator->second.Add(barIndex, update);
                auto [aggregateIterator, aggregateInserted] =
                    aggregateDetectorsByYear.try_emplace(
                        year, options.v1Candidate->maxRetainedPivotEvidence);
                (void)aggregateInserted;
                aggregateIterator->second.Add(barIndex, update);
            }
            ++barIndex;
        }
        if (barIndex == 0)
            throw std::runtime_error("PRICE_LEVEL_SOURCE_NO_BARS:" + symbol);
        aggregateBars += barIndex;
        for (std::size_t index = 0; index < options.radii.size(); ++index)
        {
            const PLC::PivotCounts count = pivots.counts()[index];
            aggregatePivot[index].highs += count.highs;
            aggregatePivot[index].lows += count.lows;
            pivotFile << "symbol," << symbol << ',' << options.radii[index] << ',' << barIndex
                      << ',' << count.highs << ',' << count.lows << ',' << count.total() << ','
                      << (100.0 * static_cast<double>(count.total()) / static_cast<double>(barIndex))
                      << '\n';
        }
        for (const auto& [year, values] : rangesByYear)
            WriteScale(scaleFile, "symbol", symbol, year, 0, 0, "all_completed_bars", values);
        for (const auto& [key, values] : pivotScales)
        {
            const auto& [year, radius, lookback, timing] = key;
            WriteScale(scaleFile, "symbol", symbol, year, radius, lookback, timing, values);
        }
        if (detector)
        {
            WriteDetector(detectorFile, "symbol", symbol, 0, detectorAll);
            for (const auto& [year, values] : detectorsByYear)
                WriteDetector(detectorFile, "symbol", symbol, year, values);
        }
        std::cout << "PRICE_LEVEL_PHASE5A_SYMBOL symbol=" << symbol << ",bars=" << barIndex << '\n';
    }
    transaction.commit();

    for (std::size_t index = 0; index < options.radii.size(); ++index)
    {
        const PLC::PivotCounts count = aggregatePivot[index];
        pivotFile << "aggregate,all," << options.radii[index] << ',' << aggregateBars << ','
                  << count.highs << ',' << count.lows << ',' << count.total() << ','
                  << (100.0 * static_cast<double>(count.total()) /
                      static_cast<double>(aggregateBars)) << '\n';
    }
    for (const auto& [year, values] : aggregateRangesByYear)
        WriteScale(scaleFile, "aggregate", "all", year, 0, 0, "all_completed_bars", values);
    for (const auto& [key, values] : aggregatePivotScales)
    {
        const auto& [year, radius, lookback, timing] = key;
        WriteScale(scaleFile, "aggregate", "all", year, radius, lookback, timing, values);
    }
    if (options.v1Candidate)
    {
        WriteDetector(detectorFile, "aggregate", "all", 0, aggregateDetector);
        for (const auto& [year, values] : aggregateDetectorsByYear)
            WriteDetector(detectorFile, "aggregate", "all", year, values);
    }

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::ofstream manifest(incompleteOutput / "manifest.txt");
    manifest << "study_contract=" << PLC::kStudyContract << '\n'
             << "source_contract=" << EA::CanonicalMarketData::kAbsoluteHalfOpenContractVersion << '\n'
             << "read_only=true\n"
             << "target_column_used=false\n"
             << "start_utc=" << EA::CanonicalMarketData::FormatAbsoluteUtc(options.start) << '\n'
             << "end_utc_exclusive=" << EA::CanonicalMarketData::FormatAbsoluteUtc(options.end) << '\n'
             << "bars_processed=" << aggregateBars << '\n'
             << "wall_clock_seconds=" << elapsed << '\n'
             << "v1_candidate=";
    if (options.v1Candidate) manifest << PL::CanonicalConfigurationIdentity(*options.v1Candidate);
    else manifest << "not_requested";
    manifest << '\n'
             << "adaptive_width_detector=not_implemented_research_scale_only\n"
             << "scale_semantics=preceding_completed_high_low_ranges;startup_uses_available_predecessor_prefix\n"
             << "pivot_scale_time=before_originating_pivot_bar\n"
             << "confirmation_scale_time=before_confirmation_bar\n"
             << "quantiles=exact_when_count_at_most_65536;otherwise_deterministic_hash_ranked_sample\n";
    manifest.close();
    pivotFile.close();
    scaleFile.close();
    detectorFile.close();
    std::filesystem::rename(incompleteOutput, options.output);
    std::cout << "PRICE_LEVEL_PHASE5A_COMPLETE bars=" << aggregateBars
              << ",wall_clock_seconds=" << elapsed << ",output=" << options.output.string() << '\n';
    return 0;
}
} // namespace

int main(int argc, const char* argv[])
{
    try { return Run(ParseOptions(argc, argv)); }
    catch (const std::exception& error)
    {
        std::cerr << "PRICE_LEVEL_PHASE5A_ERROR " << error.what() << '\n';
        return 1;
    }
}
