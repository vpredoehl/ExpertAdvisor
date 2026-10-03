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
#include <memory>
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
    bool adaptiveStudy = false;
    bool ageTimingStudy = false;
    std::vector<std::size_t> adaptiveLookbacks{32, 64, 128};
    std::vector<double> adaptiveMultipliers{0.5, 1.0, 2.0};
    std::vector<double> fixedControlWidths{0.0005, 0.05};
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

std::vector<double> ParseDoubles(std::string_view value)
{
    std::vector<double> result;
    for (const std::string& item : Split(value, ','))
    {
        std::size_t consumed = 0;
        const double parsed = std::stod(item, &consumed);
        if (consumed != item.size() || !std::isfinite(parsed) || parsed < 0.0)
            throw std::invalid_argument("PRICE_LEVEL_CLI_DOUBLE_INVALID");
        result.push_back(parsed);
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
        throw std::invalid_argument("PRICE_LEVEL_CLI_DOUBLE_DUPLICATE");
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
        else if (option == "--adaptive-study") result.adaptiveStudy = true;
        else if (option == "--age-timing-study") result.ageTimingStudy = true;
        else if (option == "--adaptive-lookbacks") result.adaptiveLookbacks = ParseSizes(require());
        else if (option == "--adaptive-multipliers") result.adaptiveMultipliers = ParseDoubles(require());
        else if (option == "--fixed-control-widths") result.fixedControlWidths = ParseDoubles(require());
        else if (option == "--help")
        {
            std::cout << "usage: price_level_characterization --start YYYY-MM-DD --end YYYY-MM-DD "
                      << "--output-dir DIRECTORY [--symbols a,b] [--pivot-radii 1,2,3,4,6,8] "
                      << "[--scale-lookbacks 32,96,384] "
                      << "[--v1-candidate radius,width,max_active,max_age,max_evidence] "
                      << "[--adaptive-study|--age-timing-study] [--adaptive-lookbacks 32,64,128] "
                      << "[--adaptive-multipliers 0.5,1,2] "
                      << "[--fixed-control-widths 0.0005,0.05] "
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
    if (result.adaptiveStudy && result.ageTimingStudy)
        throw std::invalid_argument("PRICE_LEVEL_CLI_STUDY_MODES_MUTUALLY_EXCLUSIVE");
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

// Metrics for the isolated candidate study.  This is intentionally separate
// from the legacy optional v1 replay CSV so the research contract cannot be
// mistaken for a production detector configuration.
struct StudyAccumulator final
{
    std::array<std::uint64_t, 9> interactions{};
    std::uint64_t bars = 0;
    std::uint64_t candidatePivots = 0;
    std::uint64_t evidenceSaturatedReinforcements = 0;
    std::uint64_t endedLevels = 0;
    std::uint64_t saturatedEndedLevels = 0;
    PLC::RunningDistribution activeCounts;
    PLC::RunningDistribution lifetimes;
    PLC::RunningDistribution retestLatencies;
    PLC::RunningDistribution pivotsPerEndedLevel;
    std::map<std::string, PL::Level> knownV1Levels;
    std::map<std::string, std::size_t> v1CrossBars;
    std::size_t evidenceCap = 0;

    explicit StudyAccumulator(std::size_t cap) : evidenceCap(cap) {}

    void AddAdaptive(std::size_t currentBar, const PLC::AdaptiveUpdate& update)
    {
        ++bars;
        activeCounts.Add(static_cast<double>(update.activeLevelCount));
        for (const PLC::AdaptiveObservation& observation : update.observations)
        {
            ++interactions[static_cast<std::size_t>(observation.kind)];
            if (observation.kind == PL::InteractionKind::level_established ||
                observation.kind == PL::InteractionKind::level_reinforced)
                ++candidatePivots;
            if (observation.retainedEvidenceAlreadySaturated)
                ++evidenceSaturatedReinforcements;
            if (observation.retestLatencyBars)
                retestLatencies.Add(static_cast<double>(*observation.retestLatencyBars));
            if (observation.kind != PL::InteractionKind::level_expired &&
                observation.kind != PL::InteractionKind::level_evicted)
                continue;
            ++endedLevels;
            lifetimes.Add(static_cast<double>(currentBar - observation.level.availableBar));
            pivotsPerEndedLevel.Add(
                static_cast<double>(observation.level.pivotObservationCount));
            if (observation.level.retainedPivotEvidenceCount >= evidenceCap)
                ++saturatedEndedLevels;
        }
    }

    void AddV1(std::size_t currentBar, const PL::Update& update)
    {
        ++bars;
        activeCounts.Add(static_cast<double>(update.activeLevels.size()));
        for (const PL::Observation& observation : update.observations)
        {
            ++interactions[static_cast<std::size_t>(observation.kind)];
            if (observation.kind == PL::InteractionKind::level_established ||
                observation.kind == PL::InteractionKind::level_reinforced)
                ++candidatePivots;
            if (observation.kind == PL::InteractionKind::cross_up ||
                observation.kind == PL::InteractionKind::cross_down)
                v1CrossBars[observation.levelIdentity] = currentBar;
            if (observation.kind == PL::InteractionKind::retest)
            {
                const auto cross = v1CrossBars.find(observation.levelIdentity);
                if (cross != v1CrossBars.end() && currentBar > cross->second)
                    retestLatencies.Add(static_cast<double>(currentBar - cross->second));
            }
            if (observation.kind != PL::InteractionKind::level_expired &&
                observation.kind != PL::InteractionKind::level_evicted)
                continue;
            const auto level = knownV1Levels.find(observation.levelIdentity);
            if (level == knownV1Levels.end()) continue;
            ++endedLevels;
            lifetimes.Add(static_cast<double>(currentBar - level->second.availableBar));
            pivotsPerEndedLevel.Add(
                static_cast<double>(level->second.pivotObservationCount));
            if (level->second.retainedPivotEvidence.size() >= evidenceCap)
                ++saturatedEndedLevels;
            knownV1Levels.erase(level);
            v1CrossBars.erase(observation.levelIdentity);
        }
        for (const PL::Level& level : update.activeLevels)
            knownV1Levels[level.identity] = level;
    }
};

// Lifecycle cohorts are keyed by the calendar year in which the level became
// available (the confirmation bar).  This avoids treating a year-end active
// population as though it had naturally ended.  A row with year zero is the
// complete requested-window cohort.
struct LifecycleSlice final
{
    std::uint64_t established = 0;
    std::uint64_t reinforced = 0;
    std::uint64_t ageExpired = 0;
    std::uint64_t capacityEvicted = 0;
    std::uint64_t otherTerminated = 0;
    std::uint64_t rightCensored = 0;
    PLC::RunningDistribution endedLifetimes;
    PLC::RunningDistribution censoredLifetimes;
    std::array<std::uint64_t, 6> observedPastAge{};
};

struct LifecycleRecord final
{
    int establishmentYear = 0;
    std::size_t availableBar = 0;
};

class LifecycleAccumulator final
{
public:
    void Add(std::size_t currentBar, int currentYear, const PLC::AdaptiveUpdate& update)
    {
        for (const PLC::AdaptiveObservation& observation : update.observations)
        {
            if (observation.kind == PL::InteractionKind::level_established)
            {
                const auto [iterator, inserted] = active_.emplace(observation.level.identity,
                    LifecycleRecord{currentYear, observation.level.availableBar});
                if (!inserted)
                    throw std::logic_error("PRICE_LEVEL_LIFECYCLE_DUPLICATE_ESTABLISHMENT");
                ++slices_[currentYear].established;
                ++all_.established;
            }
            else if (observation.kind == PL::InteractionKind::level_reinforced)
            {
                const auto found = active_.find(observation.level.identity);
                if (found == active_.end())
                    throw std::logic_error("PRICE_LEVEL_LIFECYCLE_UNKNOWN_REINFORCEMENT");
                ++slices_[found->second.establishmentYear].reinforced;
                ++all_.reinforced;
            }
            else if (observation.kind == PL::InteractionKind::level_expired ||
                     observation.kind == PL::InteractionKind::level_evicted)
            {
                const auto found = active_.find(observation.level.identity);
                if (found == active_.end())
                    throw std::logic_error("PRICE_LEVEL_LIFECYCLE_UNKNOWN_TERMINATION");
                Finish(slices_[found->second.establishmentYear], currentBar,
                       found->second.availableBar, observation.kind);
                Finish(all_, currentBar, found->second.availableBar, observation.kind);
                active_.erase(found);
            }
        }
    }

    void RightCensorEndOfWindow(const std::vector<PLC::AdaptiveLevel>& activeLevels,
                                std::size_t finalBar)
    {
        for (const PLC::AdaptiveLevel& level : activeLevels)
        {
            const auto found = active_.find(level.identity);
            if (found == active_.end())
                throw std::logic_error("PRICE_LEVEL_LIFECYCLE_UNKNOWN_CENSORED_LEVEL");
            Censor(slices_[found->second.establishmentYear], finalBar,
                   found->second.availableBar);
            Censor(all_, finalBar, found->second.availableBar);
            active_.erase(found);
        }
        if (!active_.empty())
            throw std::logic_error("PRICE_LEVEL_LIFECYCLE_UNFINALIZED_LEVELS");
    }

    const LifecycleSlice& all() const noexcept { return all_; }
    const std::map<int, LifecycleSlice>& byYear() const noexcept { return slices_; }

private:
    static constexpr std::array<std::size_t, 6> kSurvivalAges{128, 256, 512,
                                                                1024, 2048, 4096};

    static void AddObservedAge(LifecycleSlice& slice, std::size_t lifetime)
    {
        for (std::size_t index = 0; index < kSurvivalAges.size(); ++index)
            if (lifetime >= kSurvivalAges[index]) ++slice.observedPastAge[index];
    }

    static void Finish(LifecycleSlice& slice, std::size_t currentBar,
                       std::size_t availableBar, PL::InteractionKind reason)
    {
        const std::size_t lifetime = currentBar - availableBar;
        slice.endedLifetimes.Add(static_cast<double>(lifetime));
        AddObservedAge(slice, lifetime);
        if (reason == PL::InteractionKind::level_expired) ++slice.ageExpired;
        else if (reason == PL::InteractionKind::level_evicted) ++slice.capacityEvicted;
        else ++slice.otherTerminated;
    }

    static void Censor(LifecycleSlice& slice, std::size_t finalBar,
                       std::size_t availableBar)
    {
        const std::size_t lifetime = finalBar - availableBar;
        ++slice.rightCensored;
        slice.censoredLifetimes.Add(static_cast<double>(lifetime));
        AddObservedAge(slice, lifetime);
    }

    std::map<std::string, LifecycleRecord> active_;
    LifecycleSlice all_;
    std::map<int, LifecycleSlice> slices_;
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

void WriteStudyDistribution(std::ostream& out, const PLC::RunningDistribution& value)
{
    WriteDistribution(out, value);
}

struct CandidateDescriptor final
{
    std::string family;
    std::string identity;
    std::size_t radius = 0;
    std::size_t lookback = 0;
    double multiplier = 0.0;
    std::string timing;
    double fixedWidth = 0.0;
    std::size_t maxActive = 0;
    std::size_t maxAge = 0;
    std::size_t maxEvidence = 0;
};

void WriteStudy(std::ostream& out, std::string_view scope, std::string_view symbol,
                int year, const CandidateDescriptor& candidate,
                const StudyAccumulator& value)
{
    out << scope << ',' << symbol << ',' << year << ',' << candidate.family << ','
        << candidate.identity << ',' << candidate.radius << ',' << candidate.lookback << ','
        << candidate.multiplier << ',' << candidate.timing << ',' << candidate.fixedWidth << ','
        << candidate.maxActive << ',' << candidate.maxAge << ',' << candidate.maxEvidence << ','
        << value.bars << ',' << value.candidatePivots;
    for (std::uint64_t count : value.interactions) out << ',' << count;
    out << ',' << value.evidenceSaturatedReinforcements << ',' << value.endedLevels << ','
        << value.saturatedEndedLevels << ',';
    WriteStudyDistribution(out, value.activeCounts);
    out << ',';
    WriteStudyDistribution(out, value.lifetimes);
    out << ',';
    WriteStudyDistribution(out, value.retestLatencies);
    out << ',';
    WriteStudyDistribution(out, value.pivotsPerEndedLevel);
    out << '\n';
}

void WriteLifecycleDistribution(std::ostream& out, const PLC::RunningDistribution& value)
{
    WriteDistribution(out, value);
}

double Fraction(std::uint64_t numerator, std::uint64_t denominator)
{
    return denominator == 0 ? 0.0 :
        static_cast<double>(numerator) / static_cast<double>(denominator);
}

void WriteLifecycle(std::ostream& out, std::string_view scope, std::string_view symbol,
                    int establishmentYear, const CandidateDescriptor& candidate,
                    const LifecycleSlice& value)
{
    out << scope << ',' << symbol << ',' << establishmentYear << ',' << candidate.identity << ','
        << candidate.radius << ',' << candidate.lookback << ',' << candidate.multiplier << ','
        << candidate.timing << ',' << candidate.maxActive << ',' << candidate.maxAge << ','
        << candidate.maxEvidence << ',' << value.established << ',' << value.reinforced << ','
        << value.ageExpired << ',' << value.capacityEvicted << ',' << value.otherTerminated << ','
        << value.rightCensored << ',' << Fraction(value.ageExpired, value.established);
    for (const std::uint64_t survived : value.observedPastAge)
        out << ',' << survived << ',' << Fraction(survived, value.established);
    out << ',';
    WriteLifecycleDistribution(out, value.endedLifetimes);
    out << ',';
    WriteLifecycleDistribution(out, value.censoredLifetimes);
    out << '\n';
}

struct AgeCurveRow final
{
    CandidateDescriptor candidate;
    const StudyAccumulator* metrics = nullptr;
    const LifecycleSlice* lifecycle = nullptr;
};

double RatePerThousand(std::uint64_t count, std::uint64_t bars)
{
    return bars == 0 ? 0.0 : 1000.0 * static_cast<double>(count) /
        static_cast<double>(bars);
}

double AdjacentPercentChange(double previous, double current)
{
    return previous == 0.0 ? 0.0 : 100.0 * (current - previous) / previous;
}

void WriteAgeCurveSummary(std::ostream& out, std::vector<AgeCurveRow> rows)
{
    std::sort(rows.begin(), rows.end(), [](const AgeCurveRow& left, const AgeCurveRow& right) {
        return std::tie(left.candidate.radius, left.candidate.timing, left.candidate.maxAge) <
               std::tie(right.candidate.radius, right.candidate.timing, right.candidate.maxAge);
    });
    out << "pivot_radius,scale_timing,max_age_bars,bars,levels_established,levels_reinforced,"
           "age_expired,capacity_evicted,right_censored,age_expiration_fraction,"
           "established_per_1000_bars,reinforcement_fraction,touch_per_1000_bars,"
           "cross_up_per_1000_bars,cross_down_per_1000_bars,retest_per_1000_bars,"
           "role_reversal_per_1000_bars,active_p50,active_p90,active_max,"
           "ended_lifetime_p50,ended_lifetime_p90,censored_lifetime_p50,censored_lifetime_p90,"
           "previous_max_age_bars,established_rate_adjacent_pct_change,"
           "reinforcement_fraction_adjacent_pct_change,retest_rate_adjacent_pct_change,"
           "active_p50_adjacent_pct_change\n";
    std::map<std::pair<std::size_t, std::string>, AgeCurveRow> previous;
    for (const AgeCurveRow& row : rows)
    {
        const StudyAccumulator& metrics = *row.metrics;
        const LifecycleSlice& lifecycle = *row.lifecycle;
        const auto interaction = [&metrics](PL::InteractionKind kind) {
            return metrics.interactions[static_cast<std::size_t>(kind)];
        };
        const double establishedRate = RatePerThousand(interaction(PL::InteractionKind::level_established),
                                                       metrics.bars);
        const double reinforcedFraction = Fraction(interaction(PL::InteractionKind::level_reinforced),
                                                   metrics.candidatePivots);
        const double retestRate = RatePerThousand(interaction(PL::InteractionKind::retest), metrics.bars);
        const double activeP50 = metrics.activeCounts.count ? metrics.activeCounts.Quantile(0.5) : 0.0;
        const auto key = std::make_pair(row.candidate.radius, row.candidate.timing);
        const auto found = previous.find(key);
        const std::uint64_t priorAge = found == previous.end() ? 0 : found->second.candidate.maxAge;
        const StudyAccumulator* prior = found == previous.end() ? nullptr : found->second.metrics;
        out << row.candidate.radius << ',' << row.candidate.timing << ',' << row.candidate.maxAge << ','
            << metrics.bars << ',' << lifecycle.established << ',' << lifecycle.reinforced << ','
            << lifecycle.ageExpired << ',' << lifecycle.capacityEvicted << ','
            << lifecycle.rightCensored << ',' << Fraction(lifecycle.ageExpired, lifecycle.established) << ','
            << establishedRate << ',' << reinforcedFraction << ','
            << RatePerThousand(interaction(PL::InteractionKind::touch), metrics.bars) << ','
            << RatePerThousand(interaction(PL::InteractionKind::cross_up), metrics.bars) << ','
            << RatePerThousand(interaction(PL::InteractionKind::cross_down), metrics.bars) << ','
            << retestRate << ','
            << RatePerThousand(interaction(PL::InteractionKind::role_reversal), metrics.bars) << ','
            << activeP50 << ','
            << (metrics.activeCounts.count ? metrics.activeCounts.Quantile(0.9) : 0.0) << ','
            << (metrics.activeCounts.count ? metrics.activeCounts.maximum : 0.0) << ','
            << (lifecycle.endedLifetimes.count ? lifecycle.endedLifetimes.Quantile(0.5) : 0.0) << ','
            << (lifecycle.endedLifetimes.count ? lifecycle.endedLifetimes.Quantile(0.9) : 0.0) << ','
            << (lifecycle.censoredLifetimes.count ? lifecycle.censoredLifetimes.Quantile(0.5) : 0.0) << ','
            << (lifecycle.censoredLifetimes.count ? lifecycle.censoredLifetimes.Quantile(0.9) : 0.0) << ','
            << priorAge << ','
            << (prior ? AdjacentPercentChange(RatePerThousand(
                    prior->interactions[static_cast<std::size_t>(PL::InteractionKind::level_established)],
                    prior->bars), establishedRate) : 0.0) << ','
            << (prior ? AdjacentPercentChange(Fraction(
                    prior->interactions[static_cast<std::size_t>(PL::InteractionKind::level_reinforced)],
                    prior->candidatePivots), reinforcedFraction) : 0.0) << ','
            << (prior ? AdjacentPercentChange(RatePerThousand(
                    prior->interactions[static_cast<std::size_t>(PL::InteractionKind::retest)], prior->bars),
                    retestRate) : 0.0) << ','
            << (prior ? AdjacentPercentChange(prior->activeCounts.count ?
                    prior->activeCounts.Quantile(0.5) : 0.0, activeP50) : 0.0) << '\n';
        previous[key] = row;
    }
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
    std::ofstream studyFile(incompleteOutput / "candidate_detector_metrics.csv");
    std::ofstream lifecycleFile(incompleteOutput / "candidate_lifecycle_metrics.csv");
    std::ofstream ageCurveFile(incompleteOutput / "age_curve_summary.csv");
    if (!pivotFile || !scaleFile || !detectorFile || !studyFile || !lifecycleFile || !ageCurveFile)
        throw std::runtime_error("PRICE_LEVEL_OUTPUT_OPEN_FAILED");
    pivotFile << "scope,symbol,radius,bars,strict_pivot_highs,strict_pivot_lows,total_pivots,pivots_per_100_bars\n";
    scaleFile << "scope,symbol,year,pivot_radius,lookback,timing,statistic,count,retained_sample_size,mean,min,p50,p90,max\n";
    detectorFile << "scope,symbol,year,level_established,level_reinforced,touch,cross_up,cross_down,retest,role_reversal,level_expired,level_evicted,ended_levels,saturated_ended_levels,active_count_samples,active_count_sample_size,active_count_mean,active_count_min,active_count_p50,active_count_p90,active_count_max,lifetime_samples,lifetime_sample_size,lifetime_mean,lifetime_min,lifetime_p50,lifetime_p90,lifetime_max,pivot_observation_samples,pivot_observation_sample_size,pivot_observation_mean,pivot_observation_min,pivot_observation_p50,pivot_observation_p90,pivot_observation_max\n";
    studyFile << "scope,symbol,year,family,configuration_identity,pivot_radius,scale_lookback,scale_multiplier,scale_timing,fixed_zone_half_width,max_active_levels,max_age_bars,max_retained_pivot_evidence,bars,candidate_pivots,level_established,level_reinforced,touch,cross_up,cross_down,retest,role_reversal,level_expired,level_evicted,evidence_saturated_reinforcements,ended_levels,saturated_ended_levels,active_samples,active_retained_sample_size,active_mean,active_min,active_p50,active_p90,active_max,lifetime_samples,lifetime_retained_sample_size,lifetime_mean,lifetime_min,lifetime_p50,lifetime_p90,lifetime_max,retest_latency_samples,retest_latency_retained_sample_size,retest_latency_mean,retest_latency_min,retest_latency_p50,retest_latency_p90,retest_latency_max,pivots_per_ended_level_samples,pivots_per_ended_level_retained_sample_size,pivots_per_ended_level_mean,pivots_per_ended_level_min,pivots_per_ended_level_p50,pivots_per_ended_level_p90,pivots_per_ended_level_max\n";
    lifecycleFile << "scope,symbol,establishment_year,configuration_identity,pivot_radius,scale_lookback,scale_multiplier,scale_timing,max_active_levels,max_age_bars,max_retained_pivot_evidence,levels_established,levels_reinforced,levels_age_expired,levels_capacity_evicted,levels_other_terminated,levels_right_censored_end_of_requested_window,age_expiration_fraction_of_established,observed_survived_128_bars,observed_survived_128_fraction,observed_survived_256_bars,observed_survived_256_fraction,observed_survived_512_bars,observed_survived_512_fraction,observed_survived_1024_bars,observed_survived_1024_fraction,observed_survived_2048_bars,observed_survived_2048_fraction,observed_survived_4096_bars,observed_survived_4096_fraction,ended_lifetime_samples,ended_lifetime_retained_sample_size,ended_lifetime_mean,ended_lifetime_min,ended_lifetime_p50,ended_lifetime_p90,ended_lifetime_max,censored_lifetime_samples,censored_lifetime_retained_sample_size,censored_lifetime_mean,censored_lifetime_min,censored_lifetime_p50,censored_lifetime_p90,censored_lifetime_max\n";
    pivotFile << std::setprecision(17);
    scaleFile << std::setprecision(17);
    detectorFile << std::setprecision(17);
    studyFile << std::setprecision(17);
    lifecycleFile << std::setprecision(17);
    ageCurveFile << std::setprecision(17);

    constexpr std::array<std::size_t, 3> kPhase5AStudyRadii{2, 3, 4};
    constexpr std::array<std::size_t, 2> kAgeTimingStudyRadii{3, 4};
    constexpr std::array<std::size_t, 6> kAgeTimingStudyAges{128, 256, 512, 1024, 2048, 4096};
    constexpr std::size_t kStudyMaxActiveLevels = 64;
    constexpr std::size_t kStudyMaxAgeBars = 512;
    constexpr std::size_t kStudyMaxRetainedEvidence = 32;
    std::vector<CandidateDescriptor> studyCandidates;
    if (options.adaptiveStudy)
    {
        for (const std::size_t radius : kPhase5AStudyRadii)
        {
            for (const double width : options.fixedControlWidths)
            {
                const PL::Configuration fixed{radius, width, kStudyMaxActiveLevels,
                    kStudyMaxAgeBars, kStudyMaxRetainedEvidence, std::chrono::seconds{900}};
                studyCandidates.push_back({"fixed_width_v1_control",
                    PL::CanonicalConfigurationIdentity(fixed), radius, 0, 0.0, "not_applicable",
                    width, kStudyMaxActiveLevels, kStudyMaxAgeBars,
                    kStudyMaxRetainedEvidence});
            }
            for (const std::size_t lookback : options.adaptiveLookbacks)
            {
                for (const double multiplier : options.adaptiveMultipliers)
                {
                    for (const PLC::ScaleTiming timing : {PLC::ScaleTiming::pivot_time,
                                                           PLC::ScaleTiming::confirmation_time})
                    {
                        const PLC::AdaptiveConfiguration adaptive{radius, lookback, multiplier,
                            timing, kStudyMaxActiveLevels, kStudyMaxAgeBars,
                            kStudyMaxRetainedEvidence, std::chrono::seconds{900}};
                        studyCandidates.push_back({"adaptive_width_research",
                            PLC::CanonicalAdaptiveConfigurationIdentity(adaptive), radius,
                            lookback, multiplier,
                            std::string{PLC::CanonicalScaleTiming(timing)}, 0.0,
                            kStudyMaxActiveLevels, kStudyMaxAgeBars,
                            kStudyMaxRetainedEvidence});
                    }
                }
            }
        }
    }
    else if (options.ageTimingStudy)
    {
        for (const std::size_t radius : kAgeTimingStudyRadii)
        {
            for (const PLC::ScaleTiming timing : {PLC::ScaleTiming::pivot_time,
                                                   PLC::ScaleTiming::confirmation_time})
            {
                for (const std::size_t age : kAgeTimingStudyAges)
                {
                    const PLC::AdaptiveConfiguration adaptive{radius, 64, 1.0, timing,
                        kStudyMaxActiveLevels, age, kStudyMaxRetainedEvidence,
                        std::chrono::seconds{900}};
                    studyCandidates.push_back({"adaptive_width_research_age_timing",
                        PLC::CanonicalAdaptiveConfigurationIdentity(adaptive), radius, 64, 1.0,
                        std::string{PLC::CanonicalScaleTiming(timing)}, 0.0,
                        kStudyMaxActiveLevels, age, kStudyMaxRetainedEvidence});
                }
            }
        }
    }
    std::vector<StudyAccumulator> aggregateStudy;
    aggregateStudy.reserve(studyCandidates.size());
    for (const CandidateDescriptor& candidate : studyCandidates)
        aggregateStudy.emplace_back(candidate.maxEvidence);
    std::vector<LifecycleAccumulator> aggregateLifecycles(studyCandidates.size());

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
        std::vector<std::size_t> allScaleLookbacks = options.scaleLookbacks;
        if (options.adaptiveStudy || options.ageTimingStudy)
            allScaleLookbacks.insert(allScaleLookbacks.end(), options.adaptiveLookbacks.begin(),
                                     options.adaptiveLookbacks.end());
        std::sort(allScaleLookbacks.begin(), allScaleLookbacks.end());
        allScaleLookbacks.erase(std::unique(allScaleLookbacks.begin(), allScaleLookbacks.end()),
                                allScaleLookbacks.end());
        PLC::StrictPivotCharacterizer pivots(options.radii);
        std::vector<PLC::PrecedingRangeScale> scales;
        scales.reserve(allScaleLookbacks.size());
        for (std::size_t lookback : allScaleLookbacks) scales.emplace_back(lookback);
        std::map<std::size_t, std::size_t> scaleIndex;
        for (std::size_t index = 0; index < allScaleLookbacks.size(); ++index)
            scaleIndex.emplace(allScaleLookbacks[index], index);
        std::map<int, ScaleAccumulator> rangesByYear;
        std::map<std::tuple<int, std::size_t, std::size_t, std::string>, ScaleAccumulator>
            pivotScales;
        std::optional<PL::CausalPriceLevelEngine> detector;
        std::map<int, DetectorAccumulator> detectorsByYear;
        DetectorAccumulator detectorAll(options.v1Candidate ?
            options.v1Candidate->maxRetainedPivotEvidence : 0);
        if (options.v1Candidate)
            detector.emplace(symbol, *options.v1Candidate);
        struct CandidateRunner final
        {
            CandidateDescriptor descriptor;
            std::optional<PLC::AdaptiveWidthResearchDetector> adaptive;
            std::optional<PL::CausalPriceLevelEngine> fixed;
            StudyAccumulator all;
            std::map<int, StudyAccumulator> byYear;
            LifecycleAccumulator lifecycle;
            std::vector<PLC::AdaptiveLevel> finalActiveLevels;

            explicit CandidateRunner(CandidateDescriptor value)
                : descriptor(std::move(value)), all(descriptor.maxEvidence) {}
        };
        std::vector<std::unique_ptr<CandidateRunner>> candidateRunners;
        candidateRunners.reserve(studyCandidates.size());
        for (const CandidateDescriptor& candidate : studyCandidates)
        {
            auto runner = std::make_unique<CandidateRunner>(candidate);
            if (candidate.family == "adaptive_width_research" ||
                candidate.family == "adaptive_width_research_age_timing")
            {
                const PLC::AdaptiveConfiguration adaptive{candidate.radius, candidate.lookback,
                    candidate.multiplier,
                    candidate.timing == "pivot_time" ? PLC::ScaleTiming::pivot_time :
                                                        PLC::ScaleTiming::confirmation_time,
                    candidate.maxActive, candidate.maxAge, candidate.maxEvidence,
                    std::chrono::seconds{900}};
                runner->adaptive.emplace(symbol, adaptive);
            }
            else
            {
                const PL::Configuration fixed{candidate.radius, candidate.fixedWidth,
                    candidate.maxActive, candidate.maxAge, candidate.maxEvidence,
                    std::chrono::seconds{900}};
                runner->fixed.emplace(symbol, fixed);
            }
            candidateRunners.push_back(std::move(runner));
        }

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
            pivots.AddCompletedBar(bar, before, allScaleLookbacks,
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
            for (std::size_t candidateIndex = 0; candidateIndex < candidateRunners.size();
                 ++candidateIndex)
            {
                CandidateRunner& candidate = *candidateRunners[candidateIndex];
                auto [yearMetrics, ignored] = candidate.byYear.try_emplace(
                    year, candidate.descriptor.maxEvidence);
                (void)ignored;
                if (candidate.adaptive)
                {
                    const auto index = scaleIndex.find(candidate.descriptor.lookback);
                    if (index == scaleIndex.end())
                        throw std::logic_error("PRICE_LEVEL_ADAPTIVE_LOOKBACK_NOT_CONFIGURED");
                    const PLC::AdaptiveUpdate update = candidate.adaptive->AddCompletedBar(
                        bar, before[index->second]);
                    candidate.all.AddAdaptive(barIndex, update);
                    yearMetrics->second.AddAdaptive(barIndex, update);
                    aggregateStudy[candidateIndex].AddAdaptive(barIndex, update);
                    if (options.ageTimingStudy)
                    {
                        candidate.lifecycle.Add(barIndex, year, update);
                        aggregateLifecycles[candidateIndex].Add(barIndex, year, update);
                        candidate.finalActiveLevels = update.activeLevels;
                    }
                }
                else
                {
                    const PL::Update update = candidate.fixed->AddCompletedBar(bar);
                    candidate.all.AddV1(barIndex, update);
                    yearMetrics->second.AddV1(barIndex, update);
                    aggregateStudy[candidateIndex].AddV1(barIndex, update);
                }
            }
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
        for (const std::unique_ptr<CandidateRunner>& candidate : candidateRunners)
        {
            WriteStudy(studyFile, "symbol", symbol, 0, candidate->descriptor, candidate->all);
            for (const auto& [year, metrics] : candidate->byYear)
                WriteStudy(studyFile, "symbol", symbol, year, candidate->descriptor, metrics);
            if (options.ageTimingStudy)
            {
                candidate->lifecycle.RightCensorEndOfWindow(candidate->finalActiveLevels,
                                                            barIndex - 1);
                WriteLifecycle(lifecycleFile, "symbol", symbol, 0, candidate->descriptor,
                               candidate->lifecycle.all());
                for (const auto& [year, lifecycle] : candidate->lifecycle.byYear())
                    WriteLifecycle(lifecycleFile, "symbol", symbol, year, candidate->descriptor,
                                   lifecycle);
            }
        }
        if (options.ageTimingStudy)
        {
            for (std::size_t candidateIndex = 0; candidateIndex < candidateRunners.size();
                 ++candidateIndex)
            {
                aggregateLifecycles[candidateIndex].RightCensorEndOfWindow(
                    candidateRunners[candidateIndex]->finalActiveLevels, barIndex - 1);
            }
        }
        std::cout << "PRICE_LEVEL_CHARACTERIZATION_SYMBOL symbol=" << symbol << ",bars=" << barIndex << '\n';
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
    for (std::size_t index = 0; index < studyCandidates.size(); ++index)
        WriteStudy(studyFile, "aggregate", "all", 0, studyCandidates[index],
                   aggregateStudy[index]);
    if (options.ageTimingStudy)
    {
        std::vector<AgeCurveRow> ageCurveRows;
        ageCurveRows.reserve(studyCandidates.size());
        for (std::size_t index = 0; index < studyCandidates.size(); ++index)
        {
            WriteLifecycle(lifecycleFile, "aggregate", "all", 0, studyCandidates[index],
                           aggregateLifecycles[index].all());
            for (const auto& [year, lifecycle] : aggregateLifecycles[index].byYear())
                WriteLifecycle(lifecycleFile, "aggregate", "all", year, studyCandidates[index],
                               lifecycle);
            ageCurveRows.push_back({studyCandidates[index], &aggregateStudy[index],
                                    &aggregateLifecycles[index].all()});
        }
        WriteAgeCurveSummary(ageCurveFile, std::move(ageCurveRows));
    }

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::ofstream manifest(incompleteOutput / "manifest.txt");
    const char* sourceIdentity = std::getenv("EA_PRICE_LEVEL_SOURCE_ID");
    manifest << "study_contract=" << (options.ageTimingStudy ? PLC::kAgeTimingStudyContract :
                                        PLC::kStudyContract) << '\n'
             << "source_contract=" << EA::CanonicalMarketData::kAbsoluteHalfOpenContractVersion << '\n'
             << "source_git_identity=" << (sourceIdentity && *sourceIdentity ? sourceIdentity : "not_supplied") << '\n'
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
             << "adaptive_width_detector=" << ((options.adaptiveStudy || options.ageTimingStudy)
                 ? "isolated_research_only" : "not_requested") << '\n'
             << "scale_semantics=preceding_completed_high_low_ranges;startup_uses_available_predecessor_prefix\n"
             << "pivot_scale_time=before_originating_pivot_bar\n"
             << "confirmation_scale_time=before_confirmation_bar\n"
             << "quantiles=exact_when_count_at_most_65536;otherwise_deterministic_hash_ranked_sample\n"
             << "candidate_configurations=" << studyCandidates.size() << '\n'
             << "candidate_pivot_radii=" << (options.ageTimingStudy ? "3,4" :
                 (options.adaptiveStudy ? "2,3,4" : "not_requested"))
             << '\n'
             << "adaptive_merge_rule=candidate_pivot_merges_only_when_inside_existing_frozen_zone;nearest_anchor_then_identity;candidate_width_used_only_for_new_level\n"
             << "adaptive_width_freezing=established_level_width_never_changes\n"
             << "lifetime_boundary=availability_confirmation_bar;ended_lifetime=current_bar_minus_available_bar;right_censored_lifetime=last_requested_bar_minus_available_bar\n"
             << "age_expiration_boundary=level_active_through_available_bar_plus_max_age_bars;forced_expiration_before_interactions_at_available_bar_plus_max_age_bars_plus_one\n"
             << "age_timing_primary_grid=" << (options.ageTimingStudy
                 ? "radius=3,4;lookback=64;multiplier=1;timing=pivot_time,confirmation_time;max_age=128,256,512,1024,2048,4096;max_active=64;max_evidence=32;bar_duration_seconds=900"
                 : "not_requested") << '\n';
    manifest.close();
    pivotFile.close();
    scaleFile.close();
    detectorFile.close();
    studyFile.close();
    lifecycleFile.close();
    ageCurveFile.close();
    std::filesystem::rename(incompleteOutput, options.output);
    std::cout << "PRICE_LEVEL_CHARACTERIZATION_COMPLETE bars=" << aggregateBars
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
